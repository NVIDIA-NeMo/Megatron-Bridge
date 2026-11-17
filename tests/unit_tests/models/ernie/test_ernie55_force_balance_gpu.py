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

"""GPU test of ERNIE 5.5's forced-balance benchmark routing at EP4, behind the alltoall and the HybridEP dispatchers.

Needs one node with 4 GPUs and Transformer Engine (the HybridEP cases also DeepEP's HybridEP). Launch it on all four
ranks::

    uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_force_balance_gpu.py

Without CUDA, Transformer Engine or a 4-rank launch, the tests skip; so do the HybridEP cases without HybridEP, unless
``ERNIE55_GPU_STRICT=1``, which turns that skip into a failure. The alltoall cases need no HybridEP, so every such node
checks the recompute replay with both trackers.

Each rank routes its own tokens through the ``E`` layer's MoE module of ``test_ernie55_moe_gpu.py`` (TEGroupedMLP, 4 of
the 16 experts per rank) with ``ernie_router_force_balance=True``, and the expert bias of EP rank 2's experts raised by
5 (natural routing would send every token there). The RNG tracker is Transformer Engine's (graph-safe: what the perf
runs use, ``model.use_te_rng_tracker=true``, which CUDA graphs require; the recompute rewinds it through
Megatron-Core's ``"cloned"`` snapshot) or Megatron-Core's (the recipes' default; the ``"live"`` snapshot). Per case
(alltoall and HybridEP, each without and with Megatron-Core's MoE recompute; each with both trackers), every rank
checks, over 4 micro-batches of 1024 tokens:

* every token gets exactly ``k`` experts, and every micro-batch draws a new selection;
* with the MoE recompute, the forward runs without grad and its replay with grad, and the replay selects exactly the
  forward's experts (bitwise routing map), so the local experts receive the same tokens per expert in both;
* the tokens each local expert receives equal the routed counts of that expert summed over the EP group;
* the per-expert counts summed over ranks and micro-batches (``N = 16384`` tokens) are each within 5 sigma of
  ``N k / E = 4096`` (sigma = ``sqrt(N p (1 - p)) = 55.4``, ``p = k / E``; a correct router misses with probability
  below ``16 * 5.8e-7``), although the bias favors EP rank 2's experts;
* the four ranks draw different selections for the same tokens (the expert-parallel RNG state is seeded per EP rank);
* the output and the input gradient are finite, and Megatron-Core's expert-bias update (an NCCL all-reduce of the
  counts) moves every bias by the update rate or not at all.

Every rank finishes its checks and collectives before the outcome is gathered, so a failed check on one rank does not
leave the others waiting in a collective. :func:`run_force_balance_case` is device independent, so the checks can be
dry-run on CPU with one gloo rank.
"""

from __future__ import annotations

import logging
import math
import os

import pytest
import torch

from tests.unit_tests.models.ernie.test_ernie55_expert_load_gpu import _hybridep_available
from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import (  # noqa: F401 - pytest fixtures
    SEED,
    _fresh_moe_metrics,
    build_moe,
    ep4,
    make_config,
)


logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="the ERNIE 5.5 force-balance GPU test needs CUDA devices"
    ),
]

HYBRIDEP = dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep")
CASES = {
    "alltoall": {},
    # The replay check of both trackers, which a node without HybridEP does not skip.
    "alltoall_recompute": dict(recompute_granularity="selective", recompute_modules=["moe"]),
    "hybridep": HYBRIDEP,
    "hybridep_recompute": dict(HYBRIDEP, recompute_granularity="selective", recompute_modules=["moe"]),
}
HOT_RANK = 2
HOT_BIAS = 5.0
MICROBATCHES = 4
TOKENS = 1024
SIGMAS = 5.0


def _watch(moe) -> list[tuple[str, bool, torch.Tensor]]:
    """``(what, grad enabled, copy)`` of every router routing map and every experts' tokens-per-expert input."""
    seen: list[tuple[str, bool, torch.Tensor]] = []

    def router_output(module, args, output) -> None:
        seen.append(("routing map", torch.is_grad_enabled(), output[1].detach().clone()))

    def expert_input(module, args) -> None:
        seen.append(("tokens per expert", torch.is_grad_enabled(), torch.as_tensor(args[1]).detach().cpu().clone()))

    moe.router.register_forward_hook(router_output)
    moe.experts.register_forward_pre_hook(expert_input)
    return seen


def _forward_backward(moe, hidden: torch.Tensor, generator: torch.Generator) -> tuple[bool, bool]:
    """One micro-batch; returns whether the output and the input gradient are finite."""
    hidden = hidden.requires_grad_(True)
    output, _ = moe(hidden)
    cotangent = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    output.backward(cotangent.to(device=output.device, dtype=output.dtype))
    return bool(torch.isfinite(output).all()), bool(torch.isfinite(hidden.grad).all())


def run_force_balance_case(moe, pg_collection, *, microbatches: int, tokens: int, device) -> tuple[list[str], dict]:
    """Run ``microbatches`` micro-batches through ``moe`` (forced balance) and check them on this rank.

    Args:
        moe: The MoE layer, with ``ernie_router_force_balance=True``.
        pg_collection: Process groups; ``ep`` is the expert-parallel group and ``tp_dp_cp`` the bias-update group.
        microbatches: Micro-batches of this rank's own tokens.
        tokens: Tokens per micro-batch and rank.
        device: Device of the tokens.

    Returns:
        The failures on this rank, and a summary for the log. Every rank runs every collective.
    """
    import torch.distributed as dist
    from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias

    failures: list[str] = []
    ep_group, ep_rank, ep_size = pg_collection.ep, pg_collection.ep.rank(), pg_collection.ep.size()
    rank, local = dist.get_rank(), moe.num_local_experts
    config, router = moe.config, moe.router
    experts, topk = config.num_moe_experts, config.moe_router_topk
    replayed = bool(moe.moe_layer_recompute)
    seen = _watch(moe)

    routed_total = torch.zeros(experts, dtype=torch.int64, device=device)
    maps = []
    for microbatch in range(microbatches):
        first = len(seen)
        generator = torch.Generator().manual_seed(SEED + 1000 * microbatch + 17 * rank)
        hidden = torch.randn(tokens, 1, config.hidden_size, generator=generator, dtype=torch.float64)
        finite = _forward_backward(moe, hidden.to(device=device, dtype=config.params_dtype), generator)
        if not all(finite):
            failures.append(f"micro-batch {microbatch}: output/input gradient finite {finite}")
        events = seen[first:]
        route_events = [(grad, value) for what, grad, value in events if what == "routing map"]
        expert_events = [(grad, value) for what, grad, value in events if what == "tokens per expert"]
        # With the MoE recompute: the forward without grad, then its replay with grad.
        expected_grads = [False, True] if replayed else [True]
        route_grads, expert_grads = [grad for grad, _ in route_events], [grad for grad, _ in expert_events]
        if route_grads != expected_grads or expert_grads != expected_grads:
            failures.append(
                f"micro-batch {microbatch}: router/expert calls with grad {route_grads}/{expert_grads}, expected "
                f"{expected_grads}"
            )
            routed = torch.zeros(experts, dtype=torch.int64, device=device)
        else:
            forward_map = route_events[0][1]
            maps.append(forward_map)
            if not torch.equal(forward_map.sum(dim=-1), torch.full_like(forward_map.sum(dim=-1), topk)):
                failures.append(f"micro-batch {microbatch}: not {topk} experts per token")
            if replayed:
                if not torch.equal(route_events[1][1], forward_map):
                    failures.append(f"micro-batch {microbatch}: the recompute replay selected other experts")
                if not torch.equal(expert_events[1][1], expert_events[0][1]):
                    failures.append(f"micro-batch {microbatch}: the replay dispatched other tokens per expert")
            routed = forward_map.sum(dim=0).to(torch.int64)
        routed_total += routed
        # Every rank joins this collective, whatever its own checks found.
        routed_global = routed.clone()
        dist.all_reduce(routed_global, group=ep_group)
        received = expert_events[0][1].to(torch.int64).tolist() if expert_events else None
        expected = routed_global[ep_rank * local : (ep_rank + 1) * local].tolist()
        if received != expected:
            failures.append(f"micro-batch {microbatch}: local experts received {received}, routed {expected}")
    if len(maps) == microbatches and any(torch.equal(a, b) for a, b in zip(maps, maps[1:])):
        failures.append("two micro-batches drew the same selection")

    # Load: the per-expert counts over ranks and micro-batches are Binomial(N, k / E).
    dist.all_reduce(routed_total, group=ep_group)
    total_tokens = tokens * microbatches * ep_size
    p = topk / experts
    mean, sigma = total_tokens * p, math.sqrt(total_tokens * p * (1 - p))
    deviation = (routed_total.double() - mean).abs().max().item() / sigma
    if deviation > SIGMAS:
        failures.append(f"per-expert counts {routed_total.tolist()}: max deviation {deviation:.2f} sigma > {SIGMAS}")

    # The same tokens on every rank: the expert-parallel RNG state differs per EP rank, so the selections differ.
    generator = torch.Generator().manual_seed(SEED + 999)
    hidden = torch.randn(tokens, 1, config.hidden_size, generator=generator, dtype=torch.float64)
    first = len(seen)
    _forward_backward(moe, hidden.to(device=device, dtype=config.params_dtype), generator)
    same_tokens_map = next(value for what, _, value in seen[first:] if what == "routing map").to(torch.uint8)
    gathered = [torch.zeros_like(same_tokens_map) for _ in range(ep_size)]
    dist.all_gather(gathered, same_tokens_map, group=ep_group)
    if ep_size > 1 and any(torch.equal(gathered[ep_rank], other) for i, other in enumerate(gathered) if i != ep_rank):
        failures.append("another EP rank drew the same selection for the same tokens")

    # Megatron-Core's bias update on the forced counts, an all-reduce of the int64 counts and
    # b <- b + rate * sign(mean(c) - c) in fp32 (written out as get_updated_expert_bias computes it).
    counts = router.local_tokens_per_expert.clone()
    dist.all_reduce(counts, group=pg_collection.tp_dp_cp)
    direction = torch.sign(counts.sum() - counts * experts).to(router.expert_bias.dtype)
    expected_bias = router.expert_bias + direction * config.moe_router_bias_update_rate
    _update_router_expert_bias([router], config, tp_dp_cp_group=pg_collection.tp_dp_cp)
    if not (torch.isfinite(router.expert_bias).all() and torch.equal(router.expert_bias, expected_bias)):
        failures.append(f"bias update {router.expert_bias.tolist()}, expected {expected_bias.tolist()}")

    summary = {
        "recompute replay": "bitwise equal routing maps" if replayed else "no recompute",
        "per-expert counts": f"{int(routed_total.min())}..{int(routed_total.max())} (mean {mean:.0f})",
        "max deviation (sigma)": round(deviation, 2),
    }
    return failures, summary


@pytest.mark.parametrize("tracker", ["te", "megatron"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_forced_balance_routes_uniformly_and_replays_the_same_experts(ep4, case, tracker):
    import torch.distributed as dist
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if CASES[case].get("moe_flex_dispatcher_backend") == "hybridep" and not _hybridep_available():
        reason = "HybridEP (DeepEP's hybrid_ep) is not installed"
        if os.environ.get("ERNIE55_GPU_STRICT") == "1":
            pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
        pytest.skip(reason)
    device = torch.cuda.current_device()
    model_parallel_cuda_manual_seed(SEED, te_rng_tracker=tracker == "te", force_reset_rng=True)
    try:
        config = make_config(ernie_router_force_balance=True, use_te_rng_tracker=tracker == "te", **CASES[case])
        moe = build_moe(config, ep4, device=device)
        assert moe.router.force_balance and moe.moe_layer_recompute is ("recompute" in case)
        local = moe.num_local_experts
        with torch.no_grad():
            moe.router.expert_bias[HOT_RANK * local : (HOT_RANK + 1) * local] += HOT_BIAS
        failures, summary = run_force_balance_case(moe, ep4, microbatches=MICROBATCHES, tokens=TOKENS, device=device)
    finally:
        model_parallel_cuda_manual_seed(SEED, force_reset_rng=True)
    logger.info("force balance %s, %s tracker, rank %d: %s", case, tracker, dist.get_rank(), summary)
    failed_ranks = torch.tensor([int(bool(failures))], device=device)
    dist.all_reduce(failed_ranks)
    assert not failures, f"{case}, {tracker} tracker, rank {dist.get_rank()}: {failures}"
    assert failed_ranks.item() == 0, f"{case}, {tracker} tracker: {failed_ranks.item()} rank(s) failed; see their logs"
