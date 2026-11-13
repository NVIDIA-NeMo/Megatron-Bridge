# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests of the ERNIE 5.5 context-parallel training utilities.

* :func:`ernie55_per_row_lm_loss`: the reference implementation's per-row loss, whose row is the
  micro-batch (an SBHD micro-batch's samples are pooled): the micro-batch's token mean with its
  supervised tokens counted over the CP group, ``num_tokens = 1``, the ``CP`` factor that DDP's mean
  over DP x CP cancels, and reporting that sums to the mean row loss. On CP = 2 and 4 gloo ranks,
  the ranks' losses and gradients (averaged like DDP) equal the CP = 1 loss of the whole
  micro-batch, also when a rank holds no supervised token.
* :func:`build_ernie55_thd_cp_batch`: padding to a multiple of ``2 * CP`` with label ``-100`` and
  continuing positions, one ``cu_seqlens`` tensor in all four fields, prebuilt partition routes and
  host ``cu_seqlens``, and this rank's contiguous or zigzag rows; a contiguous pack shorter than
  ``CP * min_local_rows`` gets more pads on its last sample (:func:`ernie55_cp_min_local_rows`, the rows the
  H layers' CP boundary exchange needs on every rank).
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import pytest
import torch
from megatron.core import parallel_state

from megatron.bridge.models.ernie.ernie55_context_parallel import (
    IGNORE_LABEL,
    build_ernie55_thd_cp_batch,
    ernie55_cp_min_local_rows,
    ernie55_per_row_lm_loss,
)
from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel, ops
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases


pytestmark = pytest.mark.unit


def _per_row_loss(losses, mask, cp_group=None):
    return ernie55_per_row_lm_loss(mask, losses, cp_group=cp_group, check_for_nan_in_loss=False)


# ---------------------------------------------------------------------------------------------
# Per-row LM loss without context parallelism
# ---------------------------------------------------------------------------------------------


class TestPerRowLoss:
    def test_one_row_is_its_token_mean(self):
        losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        mask = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
        loss, num_tokens, metrics = _per_row_loss(losses, mask)
        assert loss.item() == pytest.approx((1.0 + 3.0 + 4.0) / 3)
        assert num_tokens.item() == 1 and num_tokens.dtype == torch.int
        assert metrics["lm loss"].tolist() == pytest.approx([(1.0 + 3.0 + 4.0) / 3, 1.0])

    def test_the_samples_of_a_micro_batch_are_one_row(self):
        """The reference's row is the micro-batch: an SBHD micro-batch's samples are pooled."""
        losses = torch.tensor([[1.0, 1.0, 1.0, 1.0], [5.0, 7.0, 0.0, 0.0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])
        loss, _, metrics = _per_row_loss(losses, mask)
        assert loss.item() == pytest.approx(16.0 / 6)  # not the mean of the samples' means, (1 + 6) / 2
        assert metrics["lm loss"].tolist() == pytest.approx([16.0 / 6, 1.0])

    def test_a_micro_batch_without_supervised_tokens_is_zero(self):
        losses = torch.tensor([[3.0, 3.0], [2.0, 4.0]])
        loss, _, metrics = _per_row_loss(losses, torch.zeros(2, 2))
        assert torch.isfinite(loss) and loss.item() == 0.0
        assert metrics["lm loss"].tolist() == pytest.approx([0.0, 1.0])

    def test_gradient_is_the_token_mean_of_the_micro_batch(self):
        losses = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
        mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        loss, _, _ = _per_row_loss(losses, mask)
        loss.backward()
        torch.testing.assert_close(losses.grad, mask / 3)

    def test_packed_row_and_flat_mask(self):
        losses = torch.tensor([2.0, 4.0, 6.0])
        loss, _, _ = _per_row_loss(losses, torch.tensor([1.0, 1.0, 0.0]))
        assert loss.item() == pytest.approx(3.0)

    def test_shape_checks(self):
        with pytest.raises(ValueError, match="losses"):
            _per_row_loss(torch.zeros(5), torch.ones(1, 4))
        with pytest.raises(ValueError, match="loss_mask"):
            _per_row_loss(torch.zeros(8), torch.ones(2, 2, 2))

    def test_schedule_contract(self):
        """Megatron-Core divides a three-tuple loss by num_tokens and the micro-batch count only."""
        losses = torch.tensor([[2.0, 4.0]])
        loss, num_tokens, _ = _per_row_loss(losses, torch.ones(1, 2))
        num_microbatches = 4
        scheduled = loss / torch.clamp(num_tokens, min=1) / num_microbatches
        assert scheduled.item() == pytest.approx(3.0 / num_microbatches)


# ---------------------------------------------------------------------------------------------
# Packed THD batches without context parallelism
# ---------------------------------------------------------------------------------------------


def _samples(lengths: list[int], seed: int = 0) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for length in lengths:
        tokens = torch.randint(1, 1000, (length,), generator=generator)
        labels = tokens.clone()
        labels[: length // 3] = IGNORE_LABEL  # a masked prompt
        samples.append((tokens, labels))
    return samples


def _padded_reference(samples, multiple: int, pad_token_id: int):
    """The whole padded pack, written out independently: tokens, labels, positions, padded lengths."""
    tokens, labels, positions, lengths = [], [], [], []
    for sample_tokens, sample_labels in samples:
        length = sample_tokens.numel()
        padded = -(-length // multiple) * multiple
        tokens += sample_tokens.tolist() + [pad_token_id] * (padded - length)
        labels += sample_labels.tolist() + [IGNORE_LABEL] * (padded - length)
        positions += list(range(padded))
        lengths.append(padded)
    return torch.tensor(tokens), torch.tensor(labels), torch.tensor(positions), lengths


class TestThdBatch:
    def test_without_context_parallelism_samples_are_concatenated(self):
        samples = _samples([5, 3, 7])
        batch = build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=None)
        tokens, labels, positions, lengths = _padded_reference(samples, 1, 0)
        assert torch.equal(batch.tokens[0], tokens) and torch.equal(batch.labels[0], labels)
        assert torch.equal(batch.position_ids[0], positions)
        assert torch.equal(batch.loss_mask[0], (labels != IGNORE_LABEL).float())
        params = batch.packed_seq_params
        assert params.qkv_format == "thd" and params.cu_seqlens_q.tolist() == [0, 5, 8, 15]
        assert params.cu_seqlens_q.dtype == torch.int32
        for name in ("cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
            assert getattr(params, name) is params.cu_seqlens_q, name
        assert (params.max_seqlen_q, params.max_seqlen_kv) == (7, 7)
        # The pack's last boundary is noted on the host, so the D layers never read it from the device.
        assert ops.pack_end(params) == 15

    def test_max_seqlen_can_be_pinned(self):
        batch = build_ernie55_thd_cp_batch(_samples([5, 3]), pad_token_id=0, cp_group=None, max_seqlen=64)
        assert batch.packed_seq_params.max_seqlen_q == 64
        with pytest.raises(ValueError, match="max_seqlen"):
            build_ernie55_thd_cp_batch(_samples([5, 3]), pad_token_id=0, cp_group=None, max_seqlen=4)

    @pytest.mark.parametrize(
        "samples,match",
        [
            ([], "at least one sample"),
            ([(torch.ones(4, dtype=torch.long), torch.ones(3, dtype=torch.long))], "one length"),
            ([(torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.long))], "1-D"),
            ([(torch.ones(0, dtype=torch.long), torch.ones(0, dtype=torch.long))], "non-empty"),
        ],
        ids=["empty", "length-mismatch", "2-d", "empty-sample"],
    )
    def test_malformed_samples(self, samples, match):
        with pytest.raises(ValueError, match=match):
            build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=None)


# ---------------------------------------------------------------------------------------------
# Rank side (CP gloo ranks via ernie55_cp_harness)
# ---------------------------------------------------------------------------------------------


@dataclass
class _Rank:
    rank: int
    world_size: int
    cp_group: torch.distributed.ProcessGroup


def setup_rank(world_size: int) -> _Rank:
    """CP = ``world_size`` process groups (TP = PP = 1)."""
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=world_size
    )
    return _Rank(torch.distributed.get_rank(), world_size, parallel_state.get_context_parallel_group())


def _all_reduce_sum(tensor: torch.Tensor, group) -> torch.Tensor:
    tensor = tensor.detach().clone()
    torch.distributed.all_reduce(tensor, group=group)
    return tensor


def case_per_row_loss_matches_cp1(rank: _Rank) -> None:
    """Ranks' losses and DDP-averaged gradients equal the CP = 1 per-row loss of the whole micro-batch.

    Per-token losses are ``w * c`` for one scalar parameter ``w`` shared by the ranks, so ``dL/dw``
    is a parameter gradient that DDP would average over the CP group. The loss is computed in fp32
    (like Bridge's masked next-token loss), so the CP split changes only the fp32 summation order.
    Two micro-batches: a packed row (``b = 1``) and a ``[3, s]`` batch, whose samples form one row;
    in both, CP rank 0 holds no supervised token.
    """
    cp_size, group = rank.world_size, rank.cp_group
    local = 6
    generator = torch.Generator().manual_seed(7)
    columns = slice(rank.rank * local, (rank.rank + 1) * local)
    for samples in (1, 3):
        coefficients = torch.rand(samples, cp_size * local, generator=generator, dtype=torch.float64) + 0.5
        mask = (torch.rand(samples, cp_size * local, generator=generator) < 0.6).double()
        mask[:, :local] = 0.0  # CP rank 0 holds no supervised token
        mask[-1, -1] = 1.0

        weight = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
        expected, _, _ = _per_row_loss(weight * coefficients, mask)
        (expected_grad,) = torch.autograd.grad(expected, weight)
        # The reference implementation's per-row loss: the micro-batch's token mean.
        assert expected.item() == pytest.approx((1.5 * coefficients * mask).sum().item() / mask.sum().item())

        weight_local = weight.detach().clone().requires_grad_(True)
        loss, num_tokens, metrics = _per_row_loss(weight_local * coefficients[:, columns], mask[:, columns], group)
        loss.backward()
        assert num_tokens.item() == 1
        # A rank without a supervised token contributes 0, never NaN.
        assert torch.isfinite(loss) and (rank.rank != 0 or loss.item() == 0.0)
        assert _all_reduce_sum(loss, group).item() / cp_size == pytest.approx(expected.item(), rel=1e-6)
        assert _all_reduce_sum(weight_local.grad, group).item() / cp_size == pytest.approx(
            expected_grad.item(), rel=1e-6
        )
        reporting = _all_reduce_sum(metrics["lm loss"], group)
        assert reporting[1].item() == pytest.approx(1.0)  # one micro-batch
        assert reporting[0].item() / reporting[1].item() == pytest.approx(expected.item(), rel=1e-6)


def case_batch_partitions(rank: _Rank) -> None:
    """Samples are padded to 2 * CP rows and each rank keeps its contiguous or zigzag rows."""
    cp_size = rank.world_size
    samples = _samples([5, 2 * cp_size, 3 * cp_size + 1, 9], seed=3)
    tokens, labels, positions, lengths = _padded_reference(samples, 2 * cp_size, pad_token_id=0)
    assert all(length % (2 * cp_size) == 0 for length in lengths)
    starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    local = sum(lengths) // cp_size
    expected_rows = {
        "contiguous": torch.arange(rank.rank * local, (rank.rank + 1) * local),
        "zigzag": torch.tensor(
            [
                row
                for start, length in zip(starts[:-1], lengths)
                for chunk in (rank.rank, 2 * cp_size - 1 - rank.rank)
                for row in range(
                    start + chunk * (length // (2 * cp_size)), start + (chunk + 1) * (length // (2 * cp_size))
                )
            ]
        ),
    }
    for mode, rows in expected_rows.items():
        batch = build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=rank.cp_group, cp_partition_mode=mode)
        assert torch.equal(batch.tokens[0], tokens[rows]), mode
        assert torch.equal(batch.labels[0], labels[rows]), mode
        assert torch.equal(batch.position_ids[0], positions[rows]), mode
        assert torch.equal(batch.loss_mask[0], (labels[rows] != IGNORE_LABEL).float()), mode
        params = batch.packed_seq_params
        assert params.cp_partition_mode == mode
        assert params.cu_seqlens_q.tolist() == starts
        assert params.cu_seqlens_kv_padded is params.cu_seqlens_q_padded is params.cu_seqlens_q
        assert params.cp_partition_route is not None
        assert list(params.thd_cp_host_cu_seqlens_q) == starts == list(params.thd_cp_host_cu_seqlens_kv)
        # The D layer accepts the metadata without a device-to-host synchronization.
        context_parallel.check_thd_cp_metadata(
            params, cp_size=cp_size, local_rows=local, expected_mode=mode, need_route=True, what="test"
        )
    with pytest.raises(ValueError, match="cp_partition_mode"):
        build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=rank.cp_group, cp_partition_mode="striped")


def case_short_pack_is_padded_for_the_boundary_exchange(rank: _Rank) -> None:
    """A contiguous pack below CP * min_local_rows rows grows its last sample's padding; zigzag does not."""
    cp_size = rank.world_size
    multiple = 2 * cp_size
    samples = _samples([5, 9], seed=11)
    tokens, labels, positions, lengths = _padded_reference(samples, multiple, pad_token_id=0)
    min_local_rows = 3 * multiple + 1  # rounds the pack up to the next multiple of 2 * CP
    total = -(-cp_size * min_local_rows // multiple) * multiple
    extra = total - sum(lengths)
    assert extra > 0
    tokens = torch.cat([tokens, tokens.new_zeros(extra)])
    labels = torch.cat([labels, labels.new_full((extra,), IGNORE_LABEL)])
    positions = torch.cat([positions, torch.arange(lengths[-1], lengths[-1] + extra)])
    lengths = [*lengths[:-1], lengths[-1] + extra]
    starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    local = total // cp_size
    assert local >= min_local_rows
    rows = torch.arange(rank.rank * local, (rank.rank + 1) * local)
    batch = build_ernie55_thd_cp_batch(
        samples, pad_token_id=0, cp_group=rank.cp_group, cp_partition_mode="contiguous", min_local_rows=min_local_rows
    )
    assert torch.equal(batch.tokens[0], tokens[rows])
    assert torch.equal(batch.labels[0], labels[rows])
    assert torch.equal(batch.position_ids[0], positions[rows])
    params = batch.packed_seq_params
    assert params.cu_seqlens_q.tolist() == starts
    assert params.max_seqlen_q == max(lengths)
    assert list(params.thd_cp_host_cu_seqlens_q) == starts
    context_parallel.check_thd_cp_metadata(
        params, cp_size=cp_size, local_rows=local, expected_mode="contiguous", need_route=True, what="test"
    )
    # A pack that already has the rows is unchanged; the zigzag partition (no boundary exchange) ignores the minimum.
    for mode, minimum in (("contiguous", 1), ("zigzag", min_local_rows)):
        batch = build_ernie55_thd_cp_batch(
            samples, pad_token_id=0, cp_group=rank.cp_group, cp_partition_mode=mode, min_local_rows=minimum
        )
        assert batch.packed_seq_params.cu_seqlens_q.tolist()[-1] == sum(_padded_reference(samples, multiple, 0)[3])
    with pytest.raises(ValueError, match="min_local_rows"):
        build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=rank.cp_group, min_local_rows=0)


CASES = [
    "case_per_row_loss_matches_cp1",
    "case_batch_partitions",
    "case_short_pack_is_padded_for_the_boundary_exchange",
]


@pytest.fixture(scope="module", params=[2, 4], ids=["cp2", "cp4"])
def cp_results(request, tmp_path_factory):
    """Run every case once on ``CP`` spawned gloo ranks."""
    if context_parallel.load_cp_layout_api() is None:
        pytest.skip("Megatron-Core main has no CP partition converter; ERNIE 5.5 rejects CP > 1 there")
    return run_cases(
        world_size=request.param,
        module_name=__name__,
        setup_name="setup_rank",
        case_names=CASES,
        workdir=tmp_path_factory.mktemp(f"ernie55_cp_loss{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_context_parallel(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())


@pytest.mark.parametrize(
    "ratios,window,expected",
    [
        ([0, 128, 0, 0], 128, 128),  # an H layer: max(csa_window_size, 128)
        ([0, 128], 256, 256),
        ([128], 64, 128),
        ([4, 0], 0, 8),  # Megatron-Core's d_comp is 8 for ratio 4
        ([0, 0], 128, 1),  # no DSv4 layer, no boundary exchange
        (None, 128, 1),
    ],
)
def test_cp_min_local_rows(ratios, window, expected):
    config = types.SimpleNamespace(csa_compress_ratios=ratios, csa_window_size=window)
    assert ernie55_cp_min_local_rows(config) == expected


def test_batch_needs_the_converter_at_cp(monkeypatch):
    """Megatron-Core main: building a CP > 1 batch raises (no routes, no zigzag D layer)."""
    monkeypatch.setattr(context_parallel, "load_cp_layout_api", lambda: None)
    fake_group = types.SimpleNamespace(size=lambda: 2, rank=lambda: 0)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="CP partition converter"):
        build_ernie55_thd_cp_batch(_samples([4]), pad_token_id=0, cp_group=fake_group)
