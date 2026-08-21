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

"""Fail-first wiring and exactness tests for expert LoRA at expert-TP > 1.

Each defect-demonstrating test was introduced one commit earlier carrying
``@pytest.mark.xfail(strict=True)`` and failing against the pre-fix adapter code — the
fail-first record lives in that commit. The markers were deleted together with the fix, so
every demonstration now asserts for real. Behavior pins are plain tests throughout.

The harness runs the real shipped adapters through the real MCore suppression logic on CPU: a
one-rank gloo init plus a mock process group reporting ``size() == 2`` makes MCore's
``get_pg_size`` (megatron/core/utils.py) return 2, so ``ColumnParallelLinear`` /
``RowParallelLinear`` construct expert-TP-sharded weights with ``explicit_expert_comm=True``
and the defective forward/backward paths execute unpatched. Only the collectives are replaced,
by mathematically faithful fakes that assert their own inputs (the closed-form ground truth for
these fakes is tests/unit_tests/peft/test_expert_row_parallel_adapter_algebra.py and
test_expert_column_parallel_adapter_backward_algebra.py). Real collectives are covered by
tests/unit_tests/peft/test_expert_lora_etp_distributed.py on 2 GPUs.

The fix's completion ops live in ``megatron.bridge.peft.utils`` only after the fix commit;
``_patch_fix_ops_if_present`` patches them when they exist so the same test bodies hold in both
worlds (pre-fix: the names are absent and the defective path runs; post-fix: the fakes model
the new collectives faithfully, including their backward semantics).
"""

import contextlib
import datetime
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from megatron.core.model_parallel_config import ModelParallelConfig

import megatron.bridge.peft.utils as peft_utils
from megatron.bridge.peft.utils import GroupedExpertLinearAdapter, ParallelLinearAdapter
from tests.unit_tests.peft.test_utils import MockModelParallelConfig, make_mock_pg_collection


ETP = 2
TOKENS, IN, DIM, OUT = 6, 8, 4, 8
IN_SHARD, DIM_SHARD, OUT_SHARD = IN // ETP, DIM // ETP, OUT // ETP
TOL = 1e-10

# Patch site for the collectives MCore's linear layers call (name-imports in that module).
_MCORE_LAYERS = "megatron.core.tensor_parallel.layers"


@pytest.fixture(scope="module", autouse=True)
def _gloo_single_rank():
    """One-rank gloo backend so MCore reads mock process-group sizes instead of 1."""

    created = not dist.is_initialized()
    if created:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        dist.init_process_group(backend="gloo", world_size=1, rank=0, timeout=datetime.timedelta(minutes=5))
    yield
    if created and dist.is_initialized():
        dist.destroy_process_group()


def _etp2_config() -> ModelParallelConfig:
    """Real config: fp64 for closed-form tolerances, CPU init, simulated ETP=2."""

    return ModelParallelConfig(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=ETP,
        sequence_parallel=False,
        params_dtype=torch.float64,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
    )


def _tp2_config() -> ModelParallelConfig:
    """Real config for the dense-TP=2 shared-expert-overlap flavor."""

    return ModelParallelConfig(
        tensor_model_parallel_size=ETP,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.float64,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
    )


def _seeded(shape, seed):
    return torch.randn(*shape, dtype=torch.float64, generator=torch.Generator().manual_seed(seed))


def _fake_gather(rank, slots):
    """Faithful model of MCore's forward all-gather (backward = own-rank slice, no sum).

    ``torch.cat`` over constants with the live tensor at this rank's slot reproduces both
    directions of ``_GatherFromModelParallelRegion``: forward concatenation, and a backward
    that routes only the own-rank slice of the incoming gradient — no cross-rank sum.
    """

    def fake(t, group=None):
        assert torch.allclose(t.detach(), slots[rank], atol=TOL), "gather fake fed wrong shard"
        return torch.cat([slots[s] if s != rank else t for s in range(ETP)], dim=-1)

    return fake


def _fake_reduce(other_partial):
    """Faithful model of a forward all-reduce (sum); backward of reduce_from is identity."""

    def fake(t, group=None):
        return t + other_partial

    return fake


class _BackwardAddConstant(torch.autograd.Function):
    """Forward identity; backward adds a precomputed other-rank gradient term.

    Models ``copy_to_tensor_model_parallel_region``'s backward all-reduce in a
    single-process simulation, where the other simulated rank's contribution is a
    closed-form constant rather than a live collective.
    """

    @staticmethod
    def forward(ctx, t, other_grad):
        ctx.save_for_backward(other_grad)
        return t

    @staticmethod
    def backward(ctx, grad):
        (other_grad,) = ctx.saved_tensors
        return grad + other_grad, None


def _fake_copy_to(other_grad):
    def fake(t, group=None):
        return _BackwardAddConstant.apply(t, other_grad)

    return fake


class _FakeAllGatherLastDim(torch.autograd.Function):
    """Forward AG (cat with the live own-rank slot); backward reduce-scatter-sum.

    Models ``all_gather_last_dim_from_tensor_parallel_region``: the backward sums
    gradient slices across ranks, then hands this rank its own slice. The other
    rank's slice-sum contribution enters as a precomputed constant.
    """

    @staticmethod
    def forward(ctx, t, rank, slots, other_grad_slice):
        ctx.rank = rank
        ctx.width = t.shape[-1]
        ctx.save_for_backward(other_grad_slice)
        parts = [slots[s] if s != rank else t for s in range(ETP)]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def backward(ctx, grad):
        (other_grad_slice,) = ctx.saved_tensors
        own = grad[..., ctx.rank * ctx.width : (ctx.rank + 1) * ctx.width]
        return own + other_grad_slice, None, None, None


def _fake_ag_last_dim(rank, slots, other_grad_slice):
    def fake(t, group=None):
        assert torch.allclose(t.detach(), slots[rank], atol=TOL), "AG fake fed wrong shard"
        return _FakeAllGatherLastDim.apply(t, rank, slots, other_grad_slice)

    return fake


def _patch_fix_ops_if_present(stack, *, copy_to=None, reduce_from=None, ag_last_dim=None):
    """Patch the fix's completion ops at their bridge import site — only once they exist.

    Pre-fix (fail-first form) the names are absent from ``megatron.bridge.peft.utils`` and the
    defective path runs unpatched; post-fix these fakes model the new collectives.
    """

    for name, fake in (
        ("copy_to_tensor_model_parallel_region", copy_to),
        ("reduce_from_tensor_model_parallel_region", reduce_from),
        ("all_gather_last_dim_from_tensor_parallel_region", ag_last_dim),
    ):
        if fake is not None and hasattr(peft_utils, name):
            stack.enter_context(patch.object(peft_utils, name, fake))


# ---------------------------------------------------------------------------
# 2a/2d flag: linear_out gather_output truth table (construction-level)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("is_expert", "input_is_parallel", "base_linear_is_parallel", "dtc", "expected_gather"),
    [
        pytest.param(
            True,
            True,
            True,
            True,
            False,
            id="expert-fc2",
        ),
        pytest.param(
            False,
            True,
            True,
            True,
            False,
            id="shared-overlap-fc2",
        ),
        pytest.param(False, True, True, False, True, id="dense-fc2-keeps-gather"),
        pytest.param(True, False, True, True, False, id="expert-fc1"),
        pytest.param(False, False, True, True, False, id="shared-overlap-fc1"),
        pytest.param(True, True, False, True, True, id="replicated-base-keeps-gather"),
    ],
)
def test_lin_out_gather_output_truth_table(
    is_expert, input_is_parallel, base_linear_is_parallel, dtc, expected_gather
):
    """Pin the constructed gather_output flag per flavor (post-fix contract)."""

    config = MockModelParallelConfig()
    config.expert_tensor_parallel_size = ETP
    config.tensor_model_parallel_size = ETP
    with (
        patch.object(peft_utils, "ColumnParallelLinear") as mock_col,
        patch.object(peft_utils, "RowParallelLinear"),
    ):
        ParallelLinearAdapter(
            IN,
            OUT,
            DIM,
            base_linear_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc2",
            activation="identity",
            input_is_parallel=input_is_parallel,
            is_expert=is_expert,
            disable_tensor_parallel_comm=dtc,
            base_linear_is_parallel=base_linear_is_parallel,
            model_parallel_config=config,
            pg_collection=make_mock_pg_collection(etp_size=ETP, tp_size=ETP),
        )
    # linear_out is always the last ColumnParallelLinear constructed.
    assert mock_col.call_args.kwargs["gather_output"] is expected_gather


# ---------------------------------------------------------------------------
# 2a: default-path expert fc2 — missing reduce + full delta where a partial term is due
# ---------------------------------------------------------------------------


def _build_expert_adapter(rank, *, input_is_parallel):
    """Real ParallelLinearAdapter(is_expert=True) as simulated ETP rank ``rank``."""

    return ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="decoder.layers.0.mlp.experts.local_experts.0.linear_fc2"
        if input_is_parallel
        else "decoder.layers.0.mlp.experts.local_experts.0.linear_fc1",
        activation="identity",
        input_is_parallel=input_is_parallel,
        is_expert=True,
        alpha=DIM,  # alpha/dim == 1 keeps the closed forms scale-free
        disable_tensor_parallel_comm=True,
        model_parallel_config=_etp2_config(),
        pg_collection=make_mock_pg_collection(etp_size=ETP, etp_rank=rank),
    )


def _expert_fc2_fixtures():
    x = _seeded((TOKENS, IN), 0)
    a = _seeded((DIM, IN), 1)
    b = _seeded((OUT, DIM), 2)
    z_partials = [
        x[:, s * IN_SHARD : (s + 1) * IN_SHARD] @ a[:, s * IN_SHARD : (s + 1) * IN_SHARD].t() for s in range(ETP)
    ]
    # What each rank's broken pipeline emits into the (pre-fix) output gather.
    out_slots = [z_partials[s] @ b[s * OUT_SHARD : (s + 1) * OUT_SHARD, :].t() for s in range(ETP)]
    return x, a, b, z_partials, out_slots


def test_expert_fc2_adapter_is_exact_under_the_dispatcher_etp_sum():
    """Forward exactness of the default expert fc2 adapter under the dispatcher's ETP sum."""

    x, a, b, z_partials, out_slots = _expert_fc2_fixtures()
    outputs = []
    for rank in range(ETP):
        adapter = _build_expert_adapter(rank, input_is_parallel=True)
        adapter.linear_in.weight.data.copy_(a[:, rank * IN_SHARD : (rank + 1) * IN_SHARD])
        adapter.linear_out.weight.data.copy_(b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])
        x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                patch(f"{_MCORE_LAYERS}.gather_from_tensor_model_parallel_region", _fake_gather(rank, out_slots))
            )
            _patch_fix_ops_if_present(
                stack,
                copy_to=_fake_copy_to(torch.zeros(TOKENS, DIM, dtype=torch.float64)),
                reduce_from=_fake_reduce(z_partials[1 - rank]),
            )
            outputs.append(adapter(x_r))
    combined = outputs[0] + outputs[1]  # the dispatcher's expert-TP sum, per token
    assert torch.allclose(combined, x @ (b @ a).t(), atol=TOL), (combined / (x @ (b @ a).t())).flatten()[:4]


# ---------------------------------------------------------------------------
# 2b: default-path expert fc1 — green forward, broken dL/dA (V4)
# ---------------------------------------------------------------------------


def _expert_fc1_fixtures():
    x = _seeded((TOKENS, IN), 0)
    a = _seeded((DIM, IN), 1)
    b = _seeded((OUT, DIM), 2)
    g = [_seeded((TOKENS, OUT_SHARD), 10 + r) for r in range(ETP)]
    z_slots = [x @ a[s * DIM_SHARD : (s + 1) * DIM_SHARD, :].t() for s in range(ETP)]
    return x, a, b, g, z_slots


def _run_expert_fc1(rank, x, a, b, g, z_slots):
    adapter = _build_expert_adapter(rank, input_is_parallel=False)
    adapter.linear_in.weight.data.copy_(a[rank * DIM_SHARD : (rank + 1) * DIM_SHARD, :])
    adapter.linear_out.weight.data.copy_(b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])
    # The compensating backward all-reduce sums dL/dz across ranks; the other rank's
    # term is the closed-form constant g_other @ B_other.
    other_dz = g[1 - rank] @ b[(1 - rank) * OUT_SHARD : (2 - rank) * OUT_SHARD, :]
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(f"{_MCORE_LAYERS}.gather_from_tensor_model_parallel_region", _fake_gather(rank, z_slots))
        )
        _patch_fix_ops_if_present(stack, copy_to=_fake_copy_to(other_dz))
        out = adapter(x)
        (out * g[rank]).sum().backward()
    return adapter, out


def test_expert_fc1_forward_and_dL_dB_are_exact():
    """Pins: the fc1 forward is exact and dL/dB is unaffected — the green-forward trap."""

    x, a, b, g, z_slots = _expert_fc1_fixtures()
    z = x @ a.t()
    for rank in range(ETP):
        adapter, out = _run_expert_fc1(rank, x, a, b, g, z_slots)
        b_r = b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :]
        assert torch.allclose(out, z @ b_r.t(), atol=TOL)
        assert torch.allclose(adapter.linear_out.weight.grad, g[rank].t() @ z, atol=TOL)


def test_expert_fc1_dL_dA_recovers_cross_rank_terms():
    x, a, b, g, z_slots = _expert_fc1_fixtures()
    dz_sum = sum(g[s] @ b[s * OUT_SHARD : (s + 1) * OUT_SHARD, :] for s in range(ETP))
    for rank in range(ETP):
        adapter, _ = _run_expert_fc1(rank, x, a, b, g, z_slots)
        want = dz_sum[:, rank * DIM_SHARD : (rank + 1) * DIM_SHARD].t() @ x
        got = adapter.linear_in.weight.grad
        assert torch.allclose(got, want, atol=TOL), (got - want).abs().max() / want.abs().max()


# ---------------------------------------------------------------------------
# 2c: GroupedExpertLinearAdapter — severed autograd (V2)
# ---------------------------------------------------------------------------

N_EXPERTS = 2
# Uneven but ETP-divisible: the production per-expert path pads each expert's tokens to a
# multiple of ETP, and the closed-form slot constants must match the tensors the collectives
# actually see. The pad path itself is exercised by the distributed file's empty-split case
# and the algebra files' non-divisible token count.
SPLITS = [2, 4]


def _build_grouped_adapter(rank, *, input_is_parallel):
    return GroupedExpertLinearAdapter(
        in_features=IN,
        out_features=OUT,
        dim=DIM,
        num_local_experts=N_EXPERTS,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc2"
        if input_is_parallel
        else "decoder.layers.0.mlp.experts.linear_fc1",
        activation="identity",
        input_is_parallel=input_is_parallel,
        alpha=DIM,
        model_parallel_config=_etp2_config(),
        params_device=torch.device("cpu"),
        params_dtype=torch.float64,
        pg_collection=make_mock_pg_collection(etp_size=ETP, etp_rank=rank),
    )


def _grouped_fixtures():
    x = _seeded((sum(SPLITS), IN), 0)
    a = [_seeded((DIM, IN), 1 + e) for e in range(N_EXPERTS)]
    b = [_seeded((OUT, DIM), 11 + e) for e in range(N_EXPERTS)]
    return x, a, b


def _expert_inputs(x):
    outs, start = [], 0
    for split in SPLITS:
        outs.append(x.narrow(0, start, split))
        start += split
    return outs


def _sequential_slot_fake(expected_slot_lists, rank):
    """Bare-collective fake: fills the production empty_like buffers, call-order sequenced.

    The severance under test is the production ``empty_like`` + ``cat`` — the fake only
    supplies exact values (with a faithfulness assert on the live input).
    """

    calls = iter(expected_slot_lists)

    def fake(gathered, tensor, group=None):
        slots = next(calls)
        assert torch.allclose(tensor.detach(), slots[rank], atol=TOL), "all_gather fake fed wrong shard"
        for s in range(ETP):
            gathered[s].copy_(slots[s])

    return fake


def test_grouped_fc1_gather_severs_dL_dA():
    x, a, b = _grouped_fixtures()
    rank = 0
    adapter = _build_grouped_adapter(rank, input_is_parallel=False)
    for e in range(N_EXPERTS):
        adapter.linear_in.weight.data[e].copy_(a[e][rank * DIM_SHARD : (rank + 1) * DIM_SHARD, :])
        adapter.linear_out.weight.data[e].copy_(b[e][rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])
    x = x.clone().requires_grad_(True)
    hidden_slots = [
        [x_e.detach() @ a[e][s * DIM_SHARD : (s + 1) * DIM_SHARD, :].t() for s in range(ETP)]
        for e, x_e in enumerate(_expert_inputs(x))
    ]
    g = _seeded((sum(SPLITS), OUT_SHARD), 20)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.all_gather", _sequential_slot_fake(hidden_slots, rank)))
        ag_fakes = iter(
            [
                _fake_ag_last_dim(rank, slots, torch.zeros(SPLITS[e], DIM_SHARD, dtype=torch.float64))
                for e, slots in enumerate(hidden_slots)
            ]
        )
        _patch_fix_ops_if_present(stack, ag_last_dim=lambda t, group=None: next(ag_fakes)(t, group=group))
        out = adapter(x, SPLITS)
        (out * g).sum().backward()
    # Pin: the fc1 forward is exact even today (concatenated distinct slices).
    for e, (x_e, start) in enumerate(zip(_expert_inputs(x), [0, SPLITS[0]])):
        want = (x_e.detach() @ a[e].t()) @ b[e][rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :].t()
        assert torch.allclose(out[start : start + SPLITS[e]].detach(), want, atol=TOL)
    assert adapter.linear_in.weight.grad is not None
    assert x.grad is not None


def test_grouped_fc2_output_requires_grad():
    x, a, b = _grouped_fixtures()
    rank = 0
    adapter = _build_grouped_adapter(rank, input_is_parallel=True)
    for e in range(N_EXPERTS):
        adapter.linear_in.weight.data[e].copy_(a[e][:, rank * IN_SHARD : (rank + 1) * IN_SHARD])
        adapter.linear_out.weight.data[e].copy_(b[e][rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])
    x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
    out_slots = [
        [
            (x_e[:, s * IN_SHARD : (s + 1) * IN_SHARD] @ a[e][:, s * IN_SHARD : (s + 1) * IN_SHARD].t())
            @ b[e][s * OUT_SHARD : (s + 1) * OUT_SHARD, :].t()
            for s in range(ETP)
        ]
        for e, x_e in enumerate(_expert_inputs(x))
    ]
    z_other = [
        x_e[:, (1 - rank) * IN_SHARD : (2 - rank) * IN_SHARD]
        @ a[e][:, (1 - rank) * IN_SHARD : (2 - rank) * IN_SHARD].t()
        for e, x_e in enumerate(_expert_inputs(x))
    ]
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("torch.distributed.all_gather", _sequential_slot_fake(out_slots, rank)))
        reduce_fakes = iter([_fake_reduce(z_other[e]) for e in range(N_EXPERTS)])
        _patch_fix_ops_if_present(
            stack,
            copy_to=_fake_copy_to(torch.zeros(1, dtype=torch.float64)),
            reduce_from=lambda t, group=None: next(reduce_fakes)(t, group=group),
        )
        out = adapter(x_r, SPLITS)
    assert out.requires_grad


def test_grouped_fc2_is_exact_under_the_dispatcher_etp_sum():
    x, a, b = _grouped_fixtures()
    outputs = []
    for rank in range(ETP):
        adapter = _build_grouped_adapter(rank, input_is_parallel=True)
        for e in range(N_EXPERTS):
            adapter.linear_in.weight.data[e].copy_(a[e][:, rank * IN_SHARD : (rank + 1) * IN_SHARD])
            adapter.linear_out.weight.data[e].copy_(b[e][rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])
        x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
        out_slots = [
            [
                (x_e[:, s * IN_SHARD : (s + 1) * IN_SHARD] @ a[e][:, s * IN_SHARD : (s + 1) * IN_SHARD].t())
                @ b[e][s * OUT_SHARD : (s + 1) * OUT_SHARD, :].t()
                for s in range(ETP)
            ]
            for e, x_e in enumerate(_expert_inputs(x))
        ]
        z_other = [
            x_e[:, (1 - rank) * IN_SHARD : (2 - rank) * IN_SHARD]
            @ a[e][:, (1 - rank) * IN_SHARD : (2 - rank) * IN_SHARD].t()
            for e, x_e in enumerate(_expert_inputs(x))
        ]
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("torch.distributed.all_gather", _sequential_slot_fake(out_slots, rank)))
            reduce_fakes = iter([_fake_reduce(z_other[e]) for e in range(N_EXPERTS)])
            _patch_fix_ops_if_present(
                stack,
                copy_to=_fake_copy_to(torch.zeros(1, dtype=torch.float64)),
                reduce_from=lambda t, group=None: next(reduce_fakes)(t, group=group),
            )
            outputs.append(adapter(x_r, SPLITS))
    combined = outputs[0] + outputs[1]
    want = torch.cat(
        [x_e @ (b[e] @ a[e]).t() for e, x_e in enumerate(_expert_inputs(x))],
        dim=0,
    )
    assert torch.allclose(combined.detach(), want, atol=TOL), (combined.detach() / want).flatten()[:4]


# ---------------------------------------------------------------------------
# 2d: shared-expert-overlap fc2 — delta counted TP times (V5)
# ---------------------------------------------------------------------------


def _build_shared_overlap_adapter(rank):
    return ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="decoder.layers.0.mlp.shared_experts.linear_fc2",
        activation="identity",
        input_is_parallel=True,
        is_expert=False,
        alpha=DIM,
        disable_tensor_parallel_comm=True,
        model_parallel_config=_tp2_config(),
        pg_collection=make_mock_pg_collection(tp_size=ETP, tp_rank=rank),
    )


def test_shared_expert_fc2_delta_is_counted_once_by_post_forward_comm():
    x = _seeded((TOKENS, IN), 0)
    a = _seeded((DIM, IN), 1)
    b = _seeded((OUT, DIM), 2)
    z_partials = [
        x[:, s * IN_SHARD : (s + 1) * IN_SHARD] @ a[:, s * IN_SHARD : (s + 1) * IN_SHARD].t() for s in range(ETP)
    ]
    z_full = sum(z_partials)
    out_slots = [z_full @ b[s * OUT_SHARD : (s + 1) * OUT_SHARD, :].t() for s in range(ETP)]
    outputs = []
    for rank in range(ETP):
        adapter = _build_shared_overlap_adapter(rank)
        adapter.linear_in.weight.data.copy_(a[:, rank * IN_SHARD : (rank + 1) * IN_SHARD])
        adapter.linear_out.weight.data.copy_(b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])
        x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
        with contextlib.ExitStack() as stack:
            # The adapter's own RowParallelLinear is NOT expert-suppressed here: its forward
            # calls the mcore reduce (z completes correctly — only the output side is broken).
            stack.enter_context(
                patch(f"{_MCORE_LAYERS}.reduce_from_tensor_model_parallel_region", _fake_reduce(z_partials[1 - rank]))
            )
            stack.enter_context(
                patch(f"{_MCORE_LAYERS}.gather_from_tensor_model_parallel_region", _fake_gather(rank, out_slots))
            )
            outputs.append(adapter(x_r))
    combined = outputs[0] + outputs[1]  # SharedExpertMLP.post_forward_comm's dense-TP sum
    assert torch.allclose(combined, x @ (b @ a).t(), atol=TOL), (combined / (x @ (b @ a).t())).flatten()[:4]
