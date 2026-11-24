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

"""CPU test of the whole ERNIE 5.5 stack with its H layers in the zigzag CP partition: CP = 2 and 4 against CP = 1.

Megatron-Core's opt-in ``dsv4_cp_zigzag`` (with ``cp_partition_mode="zigzag"``) runs the H layers on the per-sequence
zigzag partition: rank ``r`` holds chunks ``r`` and ``2 * CP - 1 - r`` of the micro-batch's one document, which
balances the H layers' causal work over the ranks. The D layers and the MTP depth then get their input in the
partition they compute in and convert nothing. On ``CP`` gloo ranks, the model of ``test_ernie55_mhc_mtp.py`` (H, D,
dense and MoE layers, hyper-connections, the multimax head and one SD-MTP depth, ``"H-DE/DE"``) runs one document
through the training step's batch builder (:func:`~megatron.bridge.models.ernie.ernie55_step.get_batch`, which pads it
to ``2 * CP`` chunks of whole compression blocks) in the zigzag partition, padded alike in the contiguous partition,
and once on every rank alone (CP = 1), with the training emulation of ``test_ernie55_cp_mtp.py`` (per-row LM and MTP losses, indexer
loss scale ``CP``, gradients averaged over the CP group):

* the per-token LM losses and the final-norm rows of every rank equal the CP = 1 rows at the rank's global positions;
* every parameter gradient equals the CP = 1 gradient (relative to the largest gradient of the same parameter);
* the logged MTP loss and the ranks' summed indexer losses equal CP = 1;
* in the zigzag partition no layer converts its rows (Megatron-Core's CP partition converter never runs), while the
  contiguous stack's D layers do;
* the same holds with full recompute of every layer. (The CP64 recipe's selective recompute of layer norms and mHC
  shares storages through a C++ extension that Megatron-Core compiles at run time; the GPU tests cover it.)

Megatron-Core's DSv4 CP path builds its row maps and indices with CuTe DSL kernels, which need a GPU. Here they run
through Megatron-Core's host oracles of the same maps (``csa_utils.packed_layout``) and a PyTorch gather for the
projection compaction (:func:`setup_rank`); ``test_ernie55_cp_gpu.py`` runs the kernels. Parameters are fp32, so CP
changes only the order of sums.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import re
import types

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, ernie55_per_row_lm_loss
from megatron.bridge.models.ernie.modeling_ernie55 import PER_ROW_MTP_LOSS_SUPPORTED, context_parallel
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases
from tests.unit_tests.models.ernie.test_ernie55_cp_model import (
    PAD_TOKEN,
    _all_reduce_sum,
    _error,
    _model_parallel,
    _Rank,
)
from tests.unit_tests.models.ernie.test_ernie55_cp_model import setup_rank as _setup_cp_model_rank
from tests.unit_tests.models.ernie.test_ernie55_mhc_mtp import VOCAB, _build_model


logger = logging.getLogger(__name__)

_HAS_H_ZIGZAG = "dsv4_cp_zigzag" in {field.name for field in dataclasses.fields(TransformerConfig)}

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not _HAS_H_ZIGZAG, reason="needs Megatron-Core's dsv4_cp_zigzag (H-layer zigzag CP)"),
    pytest.mark.skipif(not PER_ROW_MTP_LOSS_SUPPORTED, reason="needs Megatron-Core dev's grouped MTP roll"),
]

# H, dense, D, MoE and one SD-MTP depth (D, MoE).
PATTERN = "H-DE/DE"
# One document padded by the batch builder to 1024 rows: 2 * CP chunks of 256 rows (two compression blocks) at CP = 2
# and of 128 rows (one block, the H layers' 128-row window) at CP = 4.
DOCUMENT_LEN = 987
PADDED_LEN = 1024
PROMPT_LEN = 50
# The D layers select every candidate of the document: selection, not ranking, so CP cannot change it.
TOPK = PADDED_LEN
ROW_TOL = 2e-5
GRAD_TOL = 1e-4


def _cpu_build_attention_indices(
    cu_seqlens,
    global_start,
    l_local,
    d_window,
    window_size,
    ratio,
    compressed_width,
    compressed_topk=None,
    *,
    compressed_base=None,
    compressed_is_sequence_major=False,
    **kwargs,
):
    """``thd_layout_kernels.build_attention_indices`` through Megatron-Core's host oracle (default layouts only)."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import packed_layout

    if compressed_base is not None or compressed_is_sequence_major:
        raise NotImplementedError("the CPU stand-in supports the default compressed layout only")
    return packed_layout._build_attention_indices(
        cu_seqlens, global_start, l_local, d_window, window_size, ratio, compressed_width, compressed_topk, **kwargs
    )


def _cpu_projection_compact(local_kv, local_score, boundary_kv, boundary_score, layout):
    """``CompressorProjectionCompact``: compact row ``i`` is source row ``compact_to_source[i]`` of
    ``[boundary | local]``, zeros where it is -1; autograd scatters the gradients back."""
    rows = layout.compact_to_source.long() + 1

    def compact(local, boundary):
        source = torch.cat((local.new_zeros((1,) + tuple(local.shape[1:])), boundary, local))
        return source.index_select(0, rows)

    return compact(local_kv, boundary_kv), compact(local_score, boundary_score)


def setup_rank(world_size: int) -> _Rank:
    """The CPU stand-ins of ``test_ernie55_cp_model.py`` plus host versions of DSv4's CP layout kernels."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import packed_layout, thd_layout_kernels

    rank = _setup_cp_model_rank(world_size)
    thd_layout_kernels._require_cute = lambda *args, **kwargs: None
    thd_layout_kernels._compressor_row_maps = packed_layout.compressor_row_maps_reference
    thd_layout_kernels._compressor_rank_rows = packed_layout.compressor_rank_rows_reference
    thd_layout_kernels.build_attention_indices = _cpu_build_attention_indices
    thd_layout_kernels.CompressorProjectionCompact = types.SimpleNamespace(apply=_cpu_projection_compact)
    return rank


def _document(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(input_ids, labels)`` of one document whose prompt carries no loss and whose last token has no target."""
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, VOCAB, (DOCUMENT_LEN,), generator=generator)
    labels = tokens.roll(-1)
    labels[-1] = IGNORE_LABEL
    labels[:PROMPT_LEN] = IGNORE_LABEL
    return tokens, labels


def _micro_batch(tokens: torch.Tensor, labels: torch.Tensor) -> dict:
    """The training step's batch contract: one row, one document, with the pad token."""
    return dict(
        tokens=tokens.unsqueeze(0),
        labels=labels.unsqueeze(0),
        cu_seqlens=torch.tensor([0, tokens.numel()]),
        pad_token_id=PAD_TOKEN,
    )


def _zigzag_rows(cp_size: int, cp_rank: int) -> torch.Tensor:
    chunk = PADDED_LEN // (2 * cp_size)
    second = 2 * cp_size - 1 - cp_rank
    return torch.cat(
        (torch.arange(cp_rank * chunk, (cp_rank + 1) * chunk), torch.arange(second * chunk, (second + 1) * chunk))
    )


def _contiguous_rows(cp_size: int, cp_rank: int) -> torch.Tensor:
    local = PADDED_LEN // cp_size
    return torch.arange(cp_rank * local, (cp_rank + 1) * local)


@dataclasses.dataclass
class _Run:
    losses: torch.Tensor
    final_norm: torch.Tensor
    grads: dict[str, torch.Tensor]
    mtp_logged: torch.Tensor
    indexer_loss: torch.Tensor
    conversions: int


@contextlib.contextmanager
def _count_conversions():
    """Count the rows Megatron-Core's CP partition converter moves (every call of its ``convert``)."""
    api = context_parallel.require_cp_layout_api("test")
    module = __import__(api.module, fromlist=["CpPartitionModeConverter"])
    converter = getattr(module, "CpPartitionModeConverter", None)
    if converter is None:
        from megatron.core.context_parallel_layout.conversion import CpPartitionModeConverter as converter
    counter = {"calls": 0}
    original = converter.convert

    def counting(self, *args, **kwargs):
        counter["calls"] += 1
        return original(self, *args, **kwargs)

    converter.convert = counting
    try:
        yield counter
    finally:
        converter.convert = original


def _run(model, batch, *, cp_group, cp_size: int) -> _Run:
    """One per-row training micro-batch: forward, per-row LM loss, backward."""
    MTPLossLoggingHelper.tracker.clear()
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    # The schedule's scales for one micro-batch with calculate_per_token_loss=False.
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
    dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(float(cp_size)))
    model.zero_grad(set_to_none=True)
    final_norm = []
    handle = model.decoder.final_norm.register_forward_hook(
        lambda module, args, output: final_norm.append(output.detach().clone())
    )
    try:
        with _count_conversions() as counter:
            losses = model(
                input_ids=batch.tokens,
                position_ids=batch.position_ids,
                attention_mask=None,
                labels=batch.labels.clamp(min=0),
                loss_mask=batch.loss_mask,
                packed_seq_params=batch.packed_seq_params,
            )
            loss, num_tokens, _ = ernie55_per_row_lm_loss(batch.loss_mask, losses, cp_group=cp_group)
            assert int(num_tokens) == 1
            loss.backward()
    finally:
        handle.remove()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    return _Run(
        losses=losses.detach(),
        final_norm=final_norm[0],
        grads=grads,
        mtp_logged=MTPLossLoggingHelper.tracker["loss_sums"].detach().clone(),
        indexer_loss=dsa.DSAIndexerLossLoggingHelper.tracker["values"].detach().clone(),
        conversions=counter["calls"],
    )


def _get_batch(config, cp_group, document):
    from megatron.bridge.models.ernie.ernie55_step import get_batch

    return get_batch(iter([_micro_batch(*document)]), config, cp_group=cp_group, device="cpu")


def _compare_with_cp1(rank: _Rank, *, seed: int, **overrides) -> None:
    """Run the stack in the zigzag and the contiguous partition and on one rank; compare both with CP = 1."""
    cp_size = rank.world_size
    document = _document(seed)
    values = dict(
        params_dtype=torch.float32,
        dsa_indexer_topk=TOPK,
        cp_comm_type="all_gather",
        ernie_per_row_loss=True,
        calculate_per_token_loss=False,
        **overrides,
    )

    # CP = 1 on every rank, on the document padded as the CP > 1 batches pad it (the indexer loss counts pad rows).
    reference = _build_model(_model_parallel(1), pattern=PATTERN, seed=seed, context_parallel_size=1, **values)
    padded = (
        torch.cat([document[0], document[0].new_full((PADDED_LEN - DOCUMENT_LEN,), PAD_TOKEN)]),
        torch.cat([document[1], document[1].new_full((PADDED_LEN - DOCUMENT_LEN,), IGNORE_LABEL)]),
    )
    reference_batch = _get_batch(reference.config, None, padded)
    expected = _run(reference, reference_batch, cp_group=None, cp_size=1)
    parallel_state.destroy_model_parallel()

    errors, grad_errors, conversions = {}, {}, {}
    for mode in ("zigzag", "contiguous"):
        pg_collection = _model_parallel(cp_size)
        cp_group = pg_collection.cp
        try:
            model = _build_model(
                pg_collection,
                pattern=PATTERN,
                seed=seed,
                context_parallel_size=cp_size,
                cp_partition_mode=mode,
                dsv4_cp_zigzag=mode == "zigzag",
                **values,
            )
            # The zigzag batch pads the document itself; the contiguous partition would pad it only to 2 * CP rows,
            # so it gets the document padded alike.
            batch = _get_batch(model.config, cp_group, document if mode == "zigzag" else padded)
            params = batch.packed_seq_params
            assert params.cp_partition_mode == mode
            assert [int(n) for n in params.thd_cp_host_cu_seqlens_q] == [0, PADDED_LEN]
            rows = (_zigzag_rows if mode == "zigzag" else _contiguous_rows)(cp_size, rank.rank)
            for field in ("tokens", "labels", "position_ids"):
                assert torch.equal(getattr(batch, field)[0], getattr(reference_batch, field)[0, rows]), (mode, field)
            actual = _run(model, batch, cp_group=cp_group, cp_size=cp_size)
            assert params.cp_partition_mode == mode, f"{mode}: cp_partition_mode left at {params.cp_partition_mode!r}"
            conversions[mode] = actual.conversions

            mask = batch.loss_mask[0].bool()
            errors[f"{mode} loss rows"] = _error(actual.losses[0][mask], expected.losses[0, rows][mask])
            errors[f"{mode} final-norm rows"] = _error(actual.final_norm, expected.final_norm[rows])
            errors[f"{mode} MTP loss"] = _error(
                _all_reduce_sum(actual.mtp_logged, cp_group) / cp_size, expected.mtp_logged
            )
            errors[f"{mode} indexer loss summed over CP"] = _error(
                _all_reduce_sum(actual.indexer_loss, cp_group), expected.indexer_loss
            )
            assert actual.grads.keys() == expected.grads.keys(), sorted(actual.grads.keys() ^ expected.grads.keys())
            kind_scale = {}
            for name, grad in expected.grads.items():
                kind = re.sub(r"\.layers\.\d+\.", ".layers.*.", name)
                kind_scale[kind] = max(kind_scale.get(kind, 0.0), grad.double().abs().max().item())
            for name, grad in actual.grads.items():
                grad_errors[f"{mode} {name}"] = _error(
                    _all_reduce_sum(grad, cp_group) / cp_size,
                    expected.grads[name],
                    kind_scale[re.sub(r"\.layers\.\d+\.", ".layers.*.", name)],
                )
        finally:
            parallel_state.destroy_model_parallel()

    worst_grad = max(grad_errors, key=grad_errors.get)
    logger.info(
        "[ernie55-cp-h-zigzag rank %d CP%d %s] %s; worst gradient %s %.2e over %d; conversions %s",
        rank.rank,
        cp_size,
        overrides,
        {key: f"{value:.2e}" for key, value in errors.items()},
        worst_grad,
        grad_errors[worst_grad],
        len(grad_errors),
        conversions,
    )
    failures = [f"{what}: {error:.3e} > {ROW_TOL:.0e}" for what, error in errors.items() if error > ROW_TOL]
    failures += [
        f"gradient of {name}: {error:.3e} > {GRAD_TOL:.0e}" for name, error in grad_errors.items() if error > GRAD_TOL
    ]
    # The zigzag stack converts nothing; the contiguous stack's D layers convert to zigzag and back.
    if conversions["zigzag"] != 0:
        failures.append(f"the zigzag stack ran the CP partition converter {conversions['zigzag']} times")
    if conversions["contiguous"] == 0:
        failures.append("the contiguous stack never converted: the conversion counter sees nothing")
    assert not failures, "\n".join(failures)


def case_h_zigzag_stack(rank: _Rank) -> None:
    """The ERNIE 5.5 stack with H layers in the zigzag partition equals CP = 1 (and the contiguous stack)."""
    _compare_with_cp1(rank, seed=21)


def case_h_zigzag_stack_full_recompute(rank: _Rank) -> None:
    """The same with full recompute of every layer: the no-grad forward and the replay rebuild the same rows."""
    _compare_with_cp1(rank, seed=23, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)


CASES = ["case_h_zigzag_stack", "case_h_zigzag_stack_full_recompute"]


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
        workdir=tmp_path_factory.mktemp(f"ernie55_cp_h_zigzag{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_h_layer_zigzag_stack_under_context_parallelism(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())
