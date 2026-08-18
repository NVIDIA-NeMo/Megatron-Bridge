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

"""Two-rank real-collective tests for expert LoRA exactness at expert-TP = 2.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/peft/test_expert_lora_etp_distributed.py

This file discharges the one modeling assumption the CPU harness
(tests/unit_tests/peft/test_expert_lora_etp.py) leans on: that its faked collectives are
faithful to MCore + c10d. Here real NCCL groups and real ``parallel_state`` topology drive the
same defect sites. One topology serves every defect: TP=2 gives the dense group for the
shared-expert-overlap case (V5), ETP=2 the expert group for V1/V2/V4, world size 2, DP=1.

Each defect-demonstrating assertion carries ``@pytest.mark.xfail(strict=True)`` — the
fail-first record; the fix commit deletes the markers. All collectives run unconditionally
before any assert so a failing assert cannot desync the two ranks.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import megatron.core.parallel_state as parallel_state
import pytest
import torch
import torch.distributed as dist
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.peft.utils import GroupedExpertLinearAdapter, ParallelLinearAdapter


_WORLD_SIZE = 2
_ETP_SIZE = 2
TOKENS, IN, DIM, OUT = 8, 8, 4, 8
IN_SHARD, DIM_SHARD, OUT_SHARD = IN // _ETP_SIZE, DIM // _ETP_SIZE, OUT // _ETP_SIZE
N_EXPERTS = 2


@contextmanager
def _distributed_tp_etp() -> Iterator[ProcessGroupCollection]:
    """Initialize the two-rank TP=2 x ETP=2 topology used by every test here."""
    owns_process_group = not dist.is_initialized()
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # TF32 (the H100 default for fp32 matmuls) adds ~1e-3 relative noise, swamping the
    # 1e-4 tolerances these closed-form comparisons use. The defect signals here are
    # 2x-class, but exact tolerances keep the pins meaningful.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if owns_process_group:
        dist.init_process_group(backend="nccl")
    if owns_model_parallel:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=_ETP_SIZE,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=_ETP_SIZE,
        )
    model_parallel_cuda_manual_seed(2026, force_reset_rng=True)

    try:
        yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_process_group and dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def pg_collection() -> Iterator[ProcessGroupCollection]:
    """Provide one shared two-rank TP=2 x ETP=2 topology for this module."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    with _distributed_tp_etp() as pgs:
        yield pgs


def _config(*, tp_size: int, etp_size: int, regather: bool = False) -> ModelParallelConfig:
    return ModelParallelConfig(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=etp_size,
        sequence_parallel=False,
        params_dtype=torch.float32,
        gradient_accumulation_fusion=False,
    )


def _full_weights(device):
    """Full (unsharded) fp32 A/B and inputs, identical on every rank by fixed seed."""
    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(TOKENS, IN, generator=gen).to(device)
    a = torch.randn(DIM, IN, generator=gen).to(device)
    b = torch.randn(OUT, DIM, generator=gen).to(device)
    g = torch.randn(TOKENS, OUT, generator=gen).to(device)
    return x, a, b, g


def _copy_expert_shards(adapter, a, b, rank, *, input_is_parallel):
    with torch.no_grad():
        if input_is_parallel:
            adapter.linear_in.weight.copy_(a[:, rank * IN_SHARD : (rank + 1) * IN_SHARD])
        else:
            adapter.linear_in.weight.copy_(a[rank * DIM_SHARD : (rank + 1) * DIM_SHARD, :])
        adapter.linear_out.weight.copy_(b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])


# ---------------------------------------------------------------------------
# V1: default ParallelLinearAdapter, fc2 flavor — forward exactness
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="expert-fc2: A@h stays a per-rank partial and the gathered delta is summed once per rank "
    "by the dispatcher's expert-TP reduce — real-collective form of the CPU demonstration",
)
def test_default_fc2_adapter_delta_matches_merged_reference(pg_collection) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rank = parallel_state.get_expert_tensor_parallel_rank()
    x, a, b, g = _full_weights(device)

    adapter = ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc2",
        activation="identity",
        input_is_parallel=True,
        is_expert=True,
        alpha=DIM,
        disable_tensor_parallel_comm=True,
        model_parallel_config=_config(tp_size=_ETP_SIZE, etp_size=_ETP_SIZE),
        pg_collection=pg_collection,
    ).to(device)
    _copy_expert_shards(adapter, a, b, rank, input_is_parallel=True)

    x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
    delta = adapter(x_r)
    # The dispatcher's expert-TP sum (combine_preprocess reduce), run before any assert.
    combined = delta.detach().clone()
    dist.all_reduce(combined, group=pg_collection.expt_tp)

    reference = x @ (b @ a).t()
    torch.testing.assert_close(combined, reference, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# V4: default ParallelLinearAdapter, fc1 flavor — green forward, broken dL/dA
# ---------------------------------------------------------------------------


def _run_fc1(pg_collection, *, regather: bool = False):
    device = torch.device("cuda", torch.cuda.current_device())
    rank = parallel_state.get_expert_tensor_parallel_rank()
    x, a, b, g = _full_weights(device)

    adapter = ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="decoder.layers.0.mlp.experts.linear_fc1",
        activation="identity",
        input_is_parallel=False,
        is_expert=True,
        alpha=DIM,
        disable_tensor_parallel_comm=True,
        model_parallel_config=_config(tp_size=_ETP_SIZE, etp_size=_ETP_SIZE),
        pg_collection=pg_collection,
        sequence_parallel_input_regather=regather,
    ).to(device)
    _copy_expert_shards(adapter, a, b, rank, input_is_parallel=False)

    g_r = g[:, rank * OUT_SHARD : (rank + 1) * OUT_SHARD]
    out = adapter(x)
    (out * g_r).sum().backward()

    # All collectives before any assert: gather both ranks' dL/dA shards.
    da_local = adapter.linear_in.weight.grad.detach().clone()
    da_shards = [torch.empty_like(da_local) for _ in range(_ETP_SIZE)]
    dist.all_gather(da_shards, da_local, group=pg_collection.expt_tp)

    z = x @ a.t()
    dz_full = sum(
        g[:, s * OUT_SHARD : (s + 1) * OUT_SHARD] @ b[s * OUT_SHARD : (s + 1) * OUT_SHARD, :] for s in range(_ETP_SIZE)
    )
    return adapter, out, g_r, z, dz_full, da_shards, b, x, rank


@pytest.mark.gpu
def test_default_fc1_forward_and_dL_dB_are_exact(pg_collection) -> None:
    """Pins: the fc1 forward is exact and dL/dB unaffected — the green-forward trap."""
    adapter, out, g_r, z, _, _, b, _, rank = _run_fc1(pg_collection)
    b_r = b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :]
    torch.testing.assert_close(out, z @ b_r.t(), rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(adapter.linear_out.weight.grad, g_r.t() @ z, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="expert-fc1-dgrad: allreduce_dgrad is forced False under explicit_expert_comm — each rank's "
    "dL/dA keeps only its own-rank term (~72% relative error measured on 2xH100)",
)
def test_default_fc1_dL_dA_matches_merged_reference(pg_collection) -> None:
    _, _, _, _, dz_full, da_shards, _, x, _ = _run_fc1(pg_collection)
    for s in range(_ETP_SIZE):
        want = dz_full[:, s * DIM_SHARD : (s + 1) * DIM_SHARD].t() @ x
        torch.testing.assert_close(da_shards[s], want, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_default_fc1_regather_toggle_is_inert_for_experts(pg_collection) -> None:
    """Pin (§6.6): sequence_parallel_input_regather must not change expert-adapter results."""
    _, out_plain, _, _, _, _, _, _, _ = _run_fc1(pg_collection, regather=False)
    _, out_regather, _, _, _, _, _, _, _ = _run_fc1(pg_collection, regather=True)
    torch.testing.assert_close(out_plain, out_regather, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# V2: GroupedExpertLinearAdapter — severed autograd, real bare all_gather
# ---------------------------------------------------------------------------


def _build_grouped(pg_collection, *, input_is_parallel):
    device = torch.device("cuda", torch.cuda.current_device())
    adapter = GroupedExpertLinearAdapter(
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
        model_parallel_config=_config(tp_size=_ETP_SIZE, etp_size=_ETP_SIZE),
        params_device=device,
        params_dtype=torch.float32,
        pg_collection=pg_collection,
    )
    return adapter, device


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="grouped-autograd: the real bare torch.distributed.all_gather severs autograd — grouped fc2 "
    "adapter output carries no graph (dL/dA = dL/dB = 0, silently)",
)
def test_grouped_fc2_output_requires_grad(pg_collection) -> None:
    adapter, device = _build_grouped(pg_collection, input_is_parallel=True)
    x = torch.randn(TOKENS, IN_SHARD, device=device)
    out = adapter(x, [TOKENS // 2, TOKENS - TOKENS // 2])
    dist.barrier()
    assert out.requires_grad


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="grouped-autograd: grouped fc1 forward is exact but the gather severs the graph — dL/dA is None",
)
def test_grouped_fc1_dL_dA_flows(pg_collection) -> None:
    adapter, device = _build_grouped(pg_collection, input_is_parallel=False)
    x = torch.randn(TOKENS, IN, device=device)
    out = adapter(x, [TOKENS // 2, TOKENS - TOKENS // 2])
    out.sum().backward()
    dist.barrier()
    assert adapter.linear_in.weight.grad is not None


def _per_expert_weights(device):
    """Deterministic full A/B per expert, identical on every rank by fixed seeds."""
    a, b = [], []
    for e in range(N_EXPERTS):
        gen = torch.Generator().manual_seed(3000 + e)
        a.append(torch.randn(DIM, IN, generator=gen).to(device))
        b.append(torch.randn(OUT, DIM, generator=gen).to(device))
    return a, b


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="grouped-fc2: the adapter emits an unsummed-partial delta gathered identically on "
    "every rank — the real dispatcher-style expert-TP sum yields 2 * cat_s((x_s A_s^T) B_s^T), "
    "not the merged reference",
)
def test_grouped_fc2_delta_matches_merged_reference(pg_collection) -> None:
    """Real-collective form of the CPU grouped-fc2 wrong-value demonstration."""
    adapter, device = _build_grouped(pg_collection, input_is_parallel=True)
    rank = parallel_state.get_expert_tensor_parallel_rank()
    a, b = _per_expert_weights(device)
    with torch.no_grad():
        for e in range(N_EXPERTS):
            adapter.linear_in.weight[e].copy_(a[e][:, rank * IN_SHARD : (rank + 1) * IN_SHARD])
            adapter.linear_out.weight[e].copy_(b[e][rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])

    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(TOKENS, IN, generator=gen).to(device)
    x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
    splits = [TOKENS // 2, TOKENS - TOKENS // 2]

    delta = adapter(x_r, splits)
    # The dispatcher's expert-TP sum, run before any assert.
    combined = delta.detach().clone()
    dist.all_reduce(combined, group=pg_collection.expt_tp)

    want = torch.cat(
        [x[e * splits[0] : e * splits[0] + splits[e]] @ (b[e] @ a[e]).t() for e in range(N_EXPERTS)],
        dim=0,
    )
    torch.testing.assert_close(combined, want, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# MultiLoRA V5-analog: dense wrapper on a suppressed-comm row-parallel base
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="multi-lora-overlap: the dense wrapper gathers its full-width delta on every "
    "rank while the suppressed-comm base's partials are summed downstream across dense TP "
    "— the delta is counted exactly TP times",
)
def test_multi_lora_shared_expert_fc2_delta_counted_once(pg_collection) -> None:
    """Real-collective MultiLoRALinear on a base whose own TP comms are suppressed.

    Mirrors what ``SharedExpertMLP`` does under ``moe_shared_expert_overlap`` to legacy
    linears: sets ``explicit_expert_comm = True`` on the base so it emits full-width
    partials that ``post_forward_comm`` later sums across the dense TP group.
    """
    from megatron.core.tensor_parallel import RowParallelLinear

    from megatron.bridge.peft.multi_lora_layers import MultiLoRALinear

    device = torch.device("cuda", torch.cuda.current_device())
    rank = parallel_state.get_tensor_model_parallel_rank()
    config = _config(tp_size=_ETP_SIZE, etp_size=1)
    config.params_dtype = torch.bfloat16  # torch._grouped_mm operates in bf16

    # torch._grouped_mm requires 16-byte-aligned strides: every bf16 GEMM operand's
    # last dim must be a multiple of 8 (same constraint the repo documents for
    # GroupedExpertLinearAdapter's grouped_mm fast path). Local dims, sized so that
    # in/2, dim, and out/2 are all >= 8.
    m_in, m_dim, m_out = 16, 8, 16
    m_in_shard, m_out_shard = m_in // _ETP_SIZE, m_out // _ETP_SIZE

    base = RowParallelLinear(
        m_in,
        m_out,
        config=config,
        init_method=torch.nn.init.xavier_normal_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        tp_group=pg_collection.tp,
    ).to(device)
    # What shared_experts.py does to legacy linears under moe_shared_expert_overlap.
    base.explicit_expert_comm = True

    wrapper = MultiLoRALinear(
        to_wrap=base,
        n_adapters=2,
        dim=m_dim,
        alpha=m_dim,  # alpha/rank == 1
        full_name="decoder.layers.0.mlp.shared_experts.linear_fc2",
    )
    a, b = [], []
    for slot in range(2):
        gen = torch.Generator().manual_seed(4000 + slot)
        a.append(torch.randn(m_dim, m_in, generator=gen).to(device))
        b.append(torch.randn(m_out, m_dim, generator=gen).to(device))
    with torch.no_grad():
        for s in range(2):
            wrapper.adapters[s].linear_in.weight.copy_(
                a[s][:, rank * m_in_shard : (rank + 1) * m_in_shard].to(torch.bfloat16)
            )
            wrapper.adapters[s].linear_out.weight.copy_(
                b[s][rank * m_out_shard : (rank + 1) * m_out_shard, :].to(torch.bfloat16)
            )
        wrapper.alpha_values.fill_(1.0)
        wrapper.rank_values.fill_(1.0)

    gen = torch.Generator().manual_seed(2026)
    x = torch.randn(TOKENS, m_in, generator=gen).to(device=device, dtype=torch.bfloat16)
    x_r = x[:, rank * m_in_shard : (rank + 1) * m_in_shard].contiguous()
    splits = torch.tensor([TOKENS // 2, TOKENS - TOKENS // 2], dtype=torch.int32, device=device)
    wrapper.tokens_per_adapter = splits
    wrapper.tokens_per_adapter_total = TOKENS

    out, _ = wrapper(x_r)
    base_out, _ = base(x_r)
    delta = (out - base_out).detach().clone()
    # SharedExpertMLP.post_forward_comm's dense-TP sum, run before any assert.
    combined_delta = delta.float()
    dist.all_reduce(combined_delta, group=pg_collection.tp)
    base_combined = base_out.detach().float().clone()
    dist.all_reduce(base_combined, group=pg_collection.tp)

    n0 = TOKENS // 2
    want = torch.cat(
        [
            (x[:n0].float() @ (b[0] @ a[0]).t().float()),
            (x[n0:].float() @ (b[1] @ a[1]).t().float()),
        ],
        dim=0,
    )
    # bf16 GEMMs: loose per-element tolerance (observed ~7e-2 relative on H100), but an
    # order of magnitude tighter than the 2x-overcount defect signal.
    torch.testing.assert_close(combined_delta, want, rtol=1.5e-1, atol=5e-1)


@pytest.mark.gpu
def test_grouped_adapter_handles_empty_expert_split(pg_collection) -> None:
    """Pin (§6.2): a zero-token expert must not hang or error — identical collective
    sequences on every ETP rank (the dispatcher replicates the token set across ETP)."""
    adapter, device = _build_grouped(pg_collection, input_is_parallel=False)
    x = torch.randn(TOKENS, IN, device=device)
    out = adapter(x, [0, TOKENS])
    dist.barrier()
    assert out.shape[0] == TOKENS


# ---------------------------------------------------------------------------
# V5: shared-expert fc2 under moe_shared_expert_overlap — dense-TP overcount
# ---------------------------------------------------------------------------


@pytest.mark.gpu
@pytest.mark.xfail(
    strict=True,
    reason="shared-overlap: the gathered full-width delta is summed once per dense-TP rank by "
    "post_forward_comm — combined comes out exactly 2x the merged reference",
)
def test_shared_expert_fc2_delta_counted_once_by_tp_sum(pg_collection) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    rank = parallel_state.get_tensor_model_parallel_rank()
    x, a, b, g = _full_weights(device)

    adapter = ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="decoder.layers.0.mlp.shared_experts.linear_fc2",
        activation="identity",
        input_is_parallel=True,
        is_expert=False,
        alpha=DIM,
        disable_tensor_parallel_comm=True,
        model_parallel_config=_config(tp_size=_ETP_SIZE, etp_size=1),
        pg_collection=pg_collection,
    ).to(device)
    _copy_expert_shards(adapter, a, b, rank, input_is_parallel=True)

    x_r = x[:, rank * IN_SHARD : (rank + 1) * IN_SHARD]
    delta = adapter(x_r)
    # SharedExpertMLP.post_forward_comm's dense-TP sum, run before any assert.
    combined = delta.detach().clone()
    dist.all_reduce(combined, group=pg_collection.tp)

    reference = x @ (b @ a).t()
    torch.testing.assert_close(combined, reference, rtol=1e-4, atol=1e-4)


@pytest.mark.gpu
def test_shared_expert_fc1_is_exact_under_overlap(pg_collection) -> None:
    """Pin: the fc1 flavor under overlap already emits the correct output shard today."""
    device = torch.device("cuda", torch.cuda.current_device())
    rank = parallel_state.get_tensor_model_parallel_rank()
    x, a, b, _ = _full_weights(device)

    adapter = ParallelLinearAdapter(
        IN,
        OUT,
        DIM,
        base_linear_name="decoder.layers.0.mlp.shared_experts.linear_fc1",
        activation="identity",
        input_is_parallel=False,
        is_expert=False,
        alpha=DIM,
        disable_tensor_parallel_comm=True,
        model_parallel_config=_config(tp_size=_ETP_SIZE, etp_size=1),
        pg_collection=pg_collection,
    ).to(device)
    with torch.no_grad():
        adapter.linear_in.weight.copy_(a[rank * DIM_SHARD : (rank + 1) * DIM_SHARD, :])
        adapter.linear_out.weight.copy_(b[rank * OUT_SHARD : (rank + 1) * OUT_SHARD, :])

    out = adapter(x)
    out_shards = [torch.empty_like(out) for _ in range(_ETP_SIZE)]
    dist.all_gather(out_shards, out.detach().contiguous(), group=pg_collection.tp)

    z = x @ a.t()
    for s in range(_ETP_SIZE):
        want = z @ b[s * OUT_SHARD : (s + 1) * OUT_SHARD, :].t()
        torch.testing.assert_close(out_shards[s], want, rtol=1e-4, atol=1e-4)
