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

"""CPU tests of ERNIE 5.5's forced-balance benchmark routing (``ernie_router_force_balance``).

* Off (the default): the router's outputs are bitwise those of the base routing code, and it draws nothing from the
  RNG tracker.
* On: every token goes to ``k`` distinct experts that Megatron-Core's ``apply_random_logits`` (``RandomSTE``) draws on
  the expert-parallel RNG tracker. Neither the scores nor the expert bias select. The gate values are the unbiased
  scores of the drawn experts, normalized and scaled (fp64 oracle), with the oracle's gradients, and stay finite with
  saturated scores. The bias update is the sign rule on the forced counts.
* Load: over ``N`` tokens each expert's count is Binomial(``N``, ``p = k / E``). With ``N = 65536``, ``E = 64``,
  ``k = 8``: mean 8192, sigma ``sqrt(N p (1 - p)) = 84.7``. Every count must be within 5 sigma of the mean (a correct
  router misses that with probability below ``64 * 5.8e-7 = 3.7e-5``), and ``T = (E - 1) / E * sum((c - mean)^2) /
  (N p (1 - p))``, which is chi-square with ``E - 1 = 63`` degrees of freedom, must stay at most 140 (the
  Wilson-Hilferty tail beyond it is about 1e-7). The same scores and bias with the flag off send every token to the
  same ``k`` experts.
* Recompute: Megatron-Core's MoE recompute (``"moe"`` in ``recompute_modules``) and its full recompute
  (``precision_aware_checkpoint`` around the layer) replay the forward's experts, because
  ``tensor_parallel.checkpoint`` restores the RNG tracker; output, input gradient, every parameter gradient and the
  token counts are bitwise those without recompute. Each with two trackers: Megatron-Core's (state tensors, which the
  checkpoint snapshots as they are: its ``"live"`` restore) and a graph-safe one (generators that advance in place,
  like Transformer Engine's under ``use_te_rng_tracker``, the benchmark's setting: the ``"cloned"`` snapshot written
  back into the live generators). The controls replay other experts: a checkpoint that does not restore the tracker,
  and a snapshot that keeps the live generators of the graph-safe tracker (Megatron-Core main's).
* Configuration: a bool, off by default; Megatron-Core's ``moe_router_force_load_balancing`` stays rejected and points
  at the flag; with a graph-safe RNG tracker the recompute is rejected in FP8/FP4 and on a Megatron-Core whose
  checkpoint cannot rewind that tracker. Every rank logs one warning.

The RNG trackers run on CPU, with the module's two CUDA state accessors pointed at the CPU generator. Megatron-Core's
own ``CudaRNGStatesTracker`` holds CPU generator states, so ``fork``, ``get_states``, ``set_states`` and the
checkpoint's snapshot and restore run their own code. The graph-safe tracker holds CPU generators: only its ``fork``,
which advances them in place as Transformer Engine's does, is written here; ``is_graph_safe_cuda_rng_tracker``, the
checkpoint's ``"cloned"`` snapshot (``clone_state``) and its in-place restore are Megatron-Core's. The MoE layers are
those of ``test_ernie55_moe.py`` (all-gather dispatcher, ``SequentialMLP`` experts).
"""

import contextlib
import logging
import math

import pytest
import torch
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias
from megatron.core.tensor_parallel import random as tp_random
from megatron.core.transformer.utils import precision_aware_checkpoint

from megatron.bridge.models.ernie.modeling_ernie55 import fused_dispatch, validate_ernie55_moe_config
from megatron.bridge.models.ernie.modeling_ernie55 import moe as ernie_moe
from tests.unit_tests.models.ernie.test_ernie55_moe import (  # noqa: F401 - pytest fixtures
    BATCH,
    EXPERTS,
    HIDDEN,
    PROB_TOL,
    RATE,
    SEQ,
    TOL,
    TOPK,
    _assert_close,
    _config,
    _fresh_moe_metrics,
    _moe,
    _oracle_scores,
    _planted_near_tie_scores,
    _router,
    _router_tensors,
    pg_collection,
)


pytestmark = pytest.mark.unit

SEED = 1234
# The states model_parallel_cuda_manual_seed(SEED) adds on rank 0 (TP, EP and ETP rank 0).
_TRACKER_SEEDS = {
    tp_random.get_data_parallel_rng_tracker_name(): SEED,
    tp_random._MODEL_PARALLEL_RNG_TRACKER_NAME: SEED + 2718,
    tp_random.get_expert_parallel_rng_tracker_name(): SEED + 1024,
}
_MOE_RECOMPUTE = dict(recompute_granularity="selective", recompute_modules=["moe"])
_FULL_RECOMPUTE = dict(recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1)


def _get_cpu_rng_state(device="cuda", clone: bool = False, graph_safe: bool = False):
    """``_get_cuda_rng_state`` on the CPU generator: its state tensor, or with ``graph_safe`` the generator itself (the
    live handle ``graphsafe_get_state`` returns) or, with ``clone``, a copy of it (``clone_state``)."""
    if not graph_safe:
        return torch.get_rng_state()
    return torch.default_generator.clone_state() if clone else torch.default_generator


def _set_cpu_rng_state(new_state, device=-1, graph_safe: bool = False) -> None:
    """``_set_cuda_rng_state`` on the CPU generator; a graph-safe state is a generator, whose contents are taken."""
    if graph_safe:
        torch.default_generator.set_state(new_state.get_state())
    else:
        torch.set_rng_state(new_state)


def _install_cpu_rng_tracker(monkeypatch, tracker, states: dict) -> None:
    """Install ``tracker`` with ``states`` as the process tracker and point the CUDA state accessors at the CPU
    generator, which ``RandomSTE`` draws from for CPU tensors."""
    tracker.set_states(states)
    monkeypatch.setattr(tp_random, "_CUDA_RNG_STATE_TRACKER", tracker)
    monkeypatch.setattr(tp_random, "_CUDA_RNG_STATE_TRACKER_INITIALIZED", True)
    monkeypatch.setattr(tp_random, "_get_cuda_rng_state", _get_cpu_rng_state)
    monkeypatch.setattr(tp_random, "_set_cuda_rng_state", _set_cpu_rng_state)
    # fork() warns when the CPU generator moves inside it; here the CPU generator is the one it forks.
    monkeypatch.setattr(logging.getLogger(tp_random.__name__), "disabled", True)


@pytest.fixture
def cpu_rng_tracker(monkeypatch):
    """Megatron-Core's ``CudaRNGStatesTracker`` on the CPU generator, installed as the process tracker.

    ``fork`` sets the generator to a tracked state and stores the advanced state back; the checkpoint's
    ``_get_all_rng_states``/``_set_all_rng_states`` snapshot and restore the tracker (its ``"live"`` snapshot of the
    state tensors): all Megatron-Core code. Only the two accessors of the CUDA generator are pointed at the CPU one.
    """
    tracker = tp_random.CudaRNGStatesTracker()
    states = {name: torch.Generator().manual_seed(seed).get_state() for name, seed in _TRACKER_SEEDS.items()}
    _install_cpu_rng_tracker(monkeypatch, tracker, states)
    return tracker


class _GraphSafeCpuRNGStatesTracker(tp_random.CudaRNGStatesTracker):
    """A graph-safe RNG tracker on the CPU generator, with the semantics of Transformer Engine's.

    Its states are ``torch.Generator`` objects, and ``get_states`` returns these live objects. ``fork`` draws from the
    named generator and leaves it advanced in place, as Transformer Engine's ``fork`` does by handing the generator to
    ``graphsafe_set_state`` (which CPU generators lack, so the state is copied in and back). ``use_cudagraphable_rng``
    is what ``is_graph_safe_cuda_rng_tracker`` checks, so the checkpoint snapshots the contents of the generators
    (``"cloned"``) and writes them back into these objects.
    """

    def __init__(self) -> None:
        super().__init__()  # use_cudagraphable_rng=True here would assert that torch has CUDA graphs
        self.use_cudagraphable_rng = True

    @contextlib.contextmanager
    def fork(self, name: str = tp_random._MODEL_PARALLEL_RNG_TRACKER_NAME):
        generator, outer = self.states_[name], torch.default_generator.get_state()
        torch.default_generator.set_state(generator.get_state())
        try:
            yield
        finally:
            generator.set_state(torch.default_generator.get_state())
            torch.default_generator.set_state(outer)


@pytest.fixture
def graph_safe_cpu_rng_tracker(monkeypatch):
    """:class:`_GraphSafeCpuRNGStatesTracker` on the CPU generator, installed as the process tracker.

    The benchmark's case (Transformer Engine's tracker): the checkpoint's ``_get_all_rng_states`` takes the
    ``"cloned"`` snapshot and ``_set_all_rng_states`` writes it back into the live generators in place, Megatron-Core's
    code since dev 2f2f8ebae (#5841). The same seeds as :func:`cpu_rng_tracker`.
    """
    tracker = _GraphSafeCpuRNGStatesTracker()
    _install_cpu_rng_tracker(
        monkeypatch, tracker, {name: torch.Generator().manual_seed(seed) for name, seed in _TRACKER_SEEDS.items()}
    )
    # The checkpoint asks this before a graph-safe snapshot or restore; a CPU build of torch raises instead of False.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert tp_random.is_graph_safe_cuda_rng_tracker(tracker)
    return tracker


# The recompute tests run with each tracker; Megatron-Core's checkpoint takes this kind of snapshot of it.
_RNG_TRACKERS = {"megatron": "cpu_rng_tracker", "graph_safe": "graph_safe_cpu_rng_tracker"}
_SNAPSHOT_KINDS = {"megatron": "live", "graph_safe": "cloned"}


def _copy_states(states: dict) -> dict:
    """A copy of tracker states (state tensors, or a graph-safe tracker's generators), to start several runs from the
    same draws (``set_states`` keeps the dict it gets)."""
    return {
        name: state.clone_state() if isinstance(state, torch.Generator) else state.clone()
        for name, state in states.items()
    }


def _record_snapshot_kinds(monkeypatch) -> list[str]:
    """Record the kind (``"live"`` or ``"cloned"``, the last item of ``_get_all_rng_states``) of every RNG snapshot
    that Megatron-Core's checkpoint takes."""
    kinds = []
    snapshot = tp_random._get_all_rng_states

    def recorded():
        states = snapshot()
        kinds.append(states[-1])
        return states

    monkeypatch.setattr(tp_random, "_get_all_rng_states", recorded)
    return kinds


def _record_routing(router) -> list[tuple[bool, torch.Tensor]]:
    """Record ``(grad enabled, routing map)`` of every ``routing`` call of ``router`` (forward and replay)."""
    calls = []
    routing = router.routing

    def recorded(logits, padding_mask=None, **kwargs):
        probs, routing_map = routing(logits, padding_mask=padding_mask, **kwargs)
        calls.append((torch.is_grad_enabled(), routing_map.detach().clone()))
        return probs, routing_map

    router.routing = recorded
    return calls


def _hidden(seed: int) -> torch.Tensor:
    return torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(seed)).requires_grad_(True)


def _base_routing(router, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The routing without the forced-balance switch, statement for statement, without the z-loss and counting."""
    scores = logits.reshape(-1, router.config.num_moe_experts).float()
    top_indices = torch.topk(scores + router.expert_bias, k=router.topk, dim=-1).indices
    probs = scores.gather(-1, top_indices)
    if router.config.ernie_norm_topk_prob and router.topk > 1:
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    probs = probs * router.routed_scaling_factor_param[top_indices]
    routing_probs = torch.zeros_like(scores).scatter(-1, top_indices, probs)
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, top_indices, True)
    return routing_probs, routing_map


def _oracle_gates(
    scores: torch.Tensor, routing_map: torch.Tensor, scale: torch.Tensor, *, normalize: bool
) -> torch.Tensor:
    """fp64 gate values ``[N, E]`` of a given selection: the selected unbiased scores, normalized, times the scale."""
    selected = routing_map.to(scores.dtype)
    gates = scores * selected
    if normalize and routing_map.sum(dim=-1).max() > 1:
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-20)
    return gates * scale * selected


# ---------------------------------------------------------------------------------------------
# Off: the base routing, no draw
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("normalize", [True, False])
def test_off_is_bitwise_the_base_routing_and_draws_nothing(pg_collection, cpu_rng_tracker, topk, normalize):
    experts = 16
    config = _config(num_moe_experts=experts, moe_router_topk=topk, ernie_norm_topk_prob=normalize)
    assert config.ernie_router_force_balance is False
    router = _router(config, pg_collection, seed=3)
    assert router.force_balance is False
    with torch.no_grad():
        router.expert_bias.copy_(torch.linspace(-0.08, 0.08, experts))
    near_ties = _planted_near_tie_scores(router.expert_bias, rows=64, topk=topk, margin=1e-6, seed=topk)
    random_scores = 2 * torch.rand(256, experts, generator=torch.Generator().manual_seed(5))
    before = cpu_rng_tracker.get_states()
    for scores in (near_ties, random_scores):
        probs, routing_map = router.routing(scores)
        expected_probs, expected_map = _base_routing(router, scores)
        assert torch.equal(probs, expected_probs) and torch.equal(routing_map, expected_map)
    after = cpu_rng_tracker.get_states()
    assert all(after[name] is before[name] for name in before), "the off path forked the RNG tracker"


# ---------------------------------------------------------------------------------------------
# On: selection, load, gates, gradients, bias
# ---------------------------------------------------------------------------------------------


def test_selection_is_uniform_and_ignores_the_scores_and_the_bias(pg_collection, cpu_rng_tracker):
    """Scores and bias that send every token to experts 0..k-1; forced, every expert gets N k / E within 5 sigma."""
    experts, topk, tokens = 64, 8, 65536
    scores = torch.full((tokens, experts), 0.1)
    scores[:, :topk] = 1.9
    bias = torch.zeros(experts)
    bias[:topk] = 1.0

    natural = _router(_config(num_moe_experts=experts, moe_router_topk=topk), pg_collection)
    with torch.no_grad():
        natural.expert_bias.copy_(bias)
    _, natural_map = natural.routing(scores)
    assert natural_map[:, :topk].all() and not natural_map[:, topk:].any()

    forced = _router(
        _config(num_moe_experts=experts, moe_router_topk=topk, ernie_router_force_balance=True), pg_collection
    )
    assert forced.force_balance is True
    with torch.no_grad():
        forced.expert_bias.copy_(bias)
    start = _copy_states(cpu_rng_tracker.get_states())
    probs, routing_map = forced.routing(scores)
    assert torch.equal(routing_map.sum(dim=-1), torch.full((tokens,), topk)), "k distinct experts per token"
    assert torch.equal(forced.local_tokens_per_expert, routing_map.sum(dim=0)), "the forced selection is counted"

    counts = routing_map.sum(dim=0).double()
    p = topk / experts
    mean, variance = tokens * p, tokens * p * (1 - p)
    deviation = (counts - mean).abs().max().item()
    assert deviation <= 5 * math.sqrt(variance), f"max |count - {mean}| = {deviation} > 5 sigma"
    statistic = (experts - 1) / experts * ((counts - mean) ** 2).sum().item() / variance
    assert statistic <= 140, f"chi-square statistic {statistic:.1f} (63 degrees of freedom) > 140"

    # The draw alone selects: other scores and another bias from the same RNG state select the same experts.
    cpu_rng_tracker.set_states(_copy_states(start))
    with torch.no_grad():
        forced.expert_bias.copy_(-bias)
    _, same_map = forced.routing(torch.rand(tokens, experts, generator=torch.Generator().manual_seed(7)))
    assert torch.equal(same_map, routing_map)
    # The tracker moves on: the next call draws other experts.
    _, next_map = forced.routing(scores)
    assert not torch.equal(next_map, routing_map)
    assert torch.isfinite(probs).all() and (probs >= 0).all()


@pytest.mark.parametrize("normalize", [True, False])
def test_gates_and_gradients_match_the_oracle_of_the_drawn_selection(pg_collection, cpu_rng_tracker, normalize):
    """Gates are the unbiased scores of the drawn experts, normalized and scaled; gradients reach every gate param."""
    router = _router(_config(ernie_norm_topk_prob=normalize, ernie_router_force_balance=True), pg_collection, seed=11)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    hidden_module = hidden.float().requires_grad_(True)
    hidden_oracle = hidden_module.detach().double().requires_grad_(True)
    state, _ = _router_tensors(router)

    probs, routing_map = router(hidden_module)
    scores = _oracle_scores(hidden_oracle.reshape(-1, HIDDEN), state["weight"], state["weight_1"])
    natural_map = torch.zeros_like(routing_map).scatter(
        -1, (scores + router.expert_bias.double()).topk(TOPK, dim=-1).indices, True
    )
    assert not torch.equal(routing_map, natural_map), "the draw happened to pick the natural selection"
    expected = _oracle_gates(scores, routing_map, state["routed_scaling_factor_param"], normalize=normalize)
    assert (probs.double() - expected.detach()).abs().max().item() <= PROB_TOL
    assert torch.equal(probs != 0, routing_map)

    cotangent = torch.randn(probs.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    (probs * cotangent.float()).sum().backward()
    (expected * cotangent).sum().backward()
    for name in ("weight", "weight_1", "routed_scaling_factor_param"):
        _assert_close(getattr(router, name).grad, state[name].grad, TOL, f"d {name}")
    _assert_close(hidden_module.grad, hidden_oracle.grad, TOL, "d input")
    assert router.expert_bias.grad is None


def test_saturated_scores_keep_the_gates_and_gradients_finite(pg_collection, cpu_rng_tracker):
    """Drawn experts whose scores are exactly 0 (saturated sigmoids) give 0 gates and finite gradients, never NaN."""
    router = _router(_config(ernie_router_force_balance=True), pg_collection, seed=13)
    scores = torch.zeros(512, EXPERTS)
    scores[::2, : EXPERTS // 2] = 2.0  # odd rows: all zero; even rows: half the experts saturated high
    scores.requires_grad_(True)
    probs, routing_map = router.routing(scores)
    assert torch.isfinite(probs).all() and (probs >= 0).all()
    assert not probs[1::2].any(), "rows of zero scores must give zero gates"
    cotangent = torch.rand(probs.shape, generator=torch.Generator().manual_seed(6)) - 0.5
    (probs * cotangent).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.isfinite(router.routed_scaling_factor_param.grad).all()


def test_bias_update_is_the_sign_rule_on_the_forced_counts_and_never_selects(pg_collection, cpu_rng_tracker):
    config = _config(ernie_router_force_balance=True)
    router = _router(config, pg_collection, seed=17)
    initial_bias = torch.linspace(-0.5, 0.5, EXPERTS)
    with torch.no_grad():
        router.expert_bias.copy_(initial_bias)
    scores = torch.rand(4096, EXPERTS, generator=torch.Generator().manual_seed(8))
    start = _copy_states(cpu_rng_tracker.get_states())
    _, routing_map = router.routing(scores)
    counts = router.local_tokens_per_expert.clone()
    assert torch.equal(counts, routing_map.sum(dim=0)) and int(counts.sum()) == 4096 * TOPK

    _update_router_expert_bias([router], config, tp_dp_cp_group=pg_collection.tp_dp_cp)
    direction = torch.sign(counts.double().mean() - counts.double())
    assert torch.isfinite(router.expert_bias).all()
    torch.testing.assert_close(
        router.expert_bias.double(), initial_bias.double() + RATE * direction, rtol=0, atol=2**-24
    )

    cpu_rng_tracker.set_states(_copy_states(start))
    _, same_map = router.routing(scores)
    assert torch.equal(same_map, routing_map), "the updated bias changed the forced selection"


# ---------------------------------------------------------------------------------------------
# Recompute: the replay selects the forward's experts
# ---------------------------------------------------------------------------------------------


def _forward_backward(moe, hidden: torch.Tensor, *, full_recompute: bool) -> torch.Tensor:
    if full_recompute:
        # What megatron.core.recompute.checkpointed_forward does around the layers (BF16: tensor_parallel.checkpoint).
        output = precision_aware_checkpoint(lambda x: moe(x)[0], moe.config, None, hidden)
    else:
        output, _ = moe(hidden)
    cotangent = torch.randn(output.shape, generator=torch.Generator().manual_seed(99))
    output.backward(cotangent)
    return output.detach()


@pytest.mark.parametrize("rng_tracker", sorted(_RNG_TRACKERS))
@pytest.mark.parametrize("recompute", ["moe", "full"])
def test_recompute_replays_the_same_experts(pg_collection, request, monkeypatch, recompute, rng_tracker):
    """Forward without grad, replay with grad: same experts, several micro-batches in a row, each with a new draw.

    With the graph-safe tracker this is the benchmark's path: the checkpoint's ``"cloned"`` snapshot, written back
    into the live generators before the replay.
    """
    request.getfixturevalue(_RNG_TRACKERS[rng_tracker])
    kinds = _record_snapshot_kinds(monkeypatch)
    config = _config(ernie_router_force_balance=True, **(_MOE_RECOMPUTE if recompute == "moe" else _FULL_RECOMPUTE))
    moe = _moe(config, pg_collection, seed=31)
    assert moe.moe_layer_recompute is (recompute == "moe")
    calls = _record_routing(moe.router)
    for microbatch in range(3):
        _forward_backward(moe, _hidden(40 + microbatch), full_recompute=recompute == "full")
    assert set(kinds) == {_SNAPSHOT_KINDS[rng_tracker]}, f"the checkpoint took {kinds} RNG snapshots"
    assert [grad for grad, _ in calls] == [False, True] * 3, "a no-grad forward and a grad replay per micro-batch"
    maps = [routing_map for _, routing_map in calls]
    for microbatch in range(3):
        forward, replay = maps[2 * microbatch], maps[2 * microbatch + 1]
        assert torch.equal(forward, replay), f"micro-batch {microbatch}: the replay selected other experts"
        assert torch.equal(forward.sum(dim=-1), torch.full((SEQ * BATCH,), TOPK))
    assert not torch.equal(maps[0], maps[2]) and not torch.equal(maps[2], maps[4]), "every micro-batch draws anew"


@pytest.mark.parametrize("rng_tracker", sorted(_RNG_TRACKERS))
@pytest.mark.parametrize("recompute", ["moe", "full"])
def test_recompute_is_bitwise_the_run_without_it(pg_collection, request, recompute, rng_tracker):
    """Same draws, with and without recompute: output, gradients and token counts are bitwise equal."""
    tracker = request.getfixturevalue(_RNG_TRACKERS[rng_tracker])
    start = _copy_states(tracker.get_states())
    results = {}
    for replayed in (False, True):
        tracker.set_states(_copy_states(start))
        overrides = {} if not replayed else (_MOE_RECOMPUTE if recompute == "moe" else _FULL_RECOMPUTE)
        moe = _moe(_config(ernie_router_force_balance=True, **overrides), pg_collection, seed=33)
        hidden = _hidden(50)
        output = _forward_backward(moe, hidden, full_recompute=replayed and recompute == "full")
        grads = {name: param.grad.clone() for name, param in moe.named_parameters()}
        results[replayed] = (output, hidden.grad.clone(), grads, moe.router.local_tokens_per_expert.clone())
    (out_a, dx_a, grads_a, counts_a), (out_b, dx_b, grads_b, counts_b) = results[False], results[True]
    assert torch.equal(out_a, out_b) and torch.equal(dx_a, dx_b) and torch.equal(counts_a, counts_b)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), name


def test_control_a_checkpoint_that_keeps_the_tracker_replays_other_experts(
    pg_collection, cpu_rng_tracker, monkeypatch
):
    """A checkpoint that restores every RNG state but the tracker's: the replay draws from the advanced state."""
    restore = tp_random._set_all_rng_states

    def restore_all_but_the_tracker(cpu_state, cuda_state, tracker_states, *args, **kwargs):
        restore(cpu_state, cuda_state, tp_random.get_cuda_rng_tracker().get_states(), *args, **kwargs)

    monkeypatch.setattr(tp_random, "_set_all_rng_states", restore_all_but_the_tracker)
    moe = _moe(_config(ernie_router_force_balance=True, **_MOE_RECOMPUTE), pg_collection, seed=35)
    calls = _record_routing(moe.router)
    _forward_backward(moe, _hidden(60), full_recompute=False)
    (forward_grad, forward), (replay_grad, replay) = calls
    assert (forward_grad, replay_grad) == (False, True)
    assert not torch.equal(forward, replay)


def test_control_a_snapshot_of_the_live_graph_safe_generators_replays_other_experts(
    pg_collection, graph_safe_cpu_rng_tracker, monkeypatch
):
    """Megatron-Core main's snapshot (before dev 2f2f8ebae) keeps the graph-safe tracker's live generators. The forward
    advances them before the replay restores them, so the replay draws anew. Only the snapshot differs from the
    graph-safe replay test above; the restore is Megatron-Core dev's for a ``"live"`` snapshot."""

    def live_snapshot():
        tracker = tp_random.get_cuda_rng_tracker()
        return torch.get_rng_state(), tp_random._get_cuda_rng_state(graph_safe=True), tracker.get_states(), "live"

    monkeypatch.setattr(tp_random, "_get_all_rng_states", live_snapshot)
    moe = _moe(_config(ernie_router_force_balance=True, **_MOE_RECOMPUTE), pg_collection, seed=35)
    calls = _record_routing(moe.router)
    _forward_backward(moe, _hidden(60), full_recompute=False)
    (forward_grad, forward), (replay_grad, replay) = calls
    assert (forward_grad, replay_grad) == (False, True)
    assert not torch.equal(forward, replay)


# ---------------------------------------------------------------------------------------------
# Configuration and warning
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["true", 1, None])
def test_flag_must_be_a_bool(value):
    with pytest.raises(ValueError, match="ernie_router_force_balance must be a bool"):
        _config(ernie_router_force_balance=value)
    # The router's own check, for configs that never ran Ernie55TransformerConfig.finalize.
    with pytest.raises(ValueError, match=r"^ERNIE 5\.5 MoE: ernie_router_force_balance must be a bool"):
        validate_ernie55_moe_config(_with_fields(ernie_router_force_balance=value))


def test_megatron_core_force_load_balancing_stays_rejected_and_points_at_the_flag():
    with pytest.raises(ValueError, match=r"moe_router_force_load_balancing .*ernie_router_force_balance=True"):
        _config(moe_router_force_load_balancing=True)


def _with_fields(**fields):
    """A finalized config with fields changed afterwards (Megatron-Core's own FP8 checks need Transformer Engine)."""
    config = _config()
    for name, value in fields.items():
        assert hasattr(config, name), name
        setattr(config, name, value)
    return config


_GRAPH_SAFE = {"use_te_rng_tracker": dict(use_te_rng_tracker=True), "cuda_graphs": dict(cuda_graph_impl="local")}


@pytest.mark.parametrize("recompute", [_MOE_RECOMPUTE, _FULL_RECOMPUTE], ids=["moe", "full"])
@pytest.mark.parametrize("tracker", sorted(_GRAPH_SAFE))
@pytest.mark.parametrize("precision", [dict(fp8="hybrid"), dict(fp4="nvfp4")], ids=["fp8", "fp4"])
def test_replay_through_transformer_engines_checkpoint_is_rejected(recompute, tracker, precision):
    config = _with_fields(ernie_router_force_balance=True, **recompute, **_GRAPH_SAFE[tracker], **precision)
    with pytest.raises(ValueError, match=r"^ERNIE 5\.5 MoE: .*needs BF16"):
        validate_ernie55_moe_config(config)


@pytest.mark.parametrize("tracker", sorted(_GRAPH_SAFE))
def test_replay_needs_a_megatron_core_that_rewinds_graph_safe_states(monkeypatch, tracker):
    config = _with_fields(ernie_router_force_balance=True, **_MOE_RECOMPUTE, **_GRAPH_SAFE[tracker])
    assert ernie_moe._checkpoint_rewinds_graph_safe_rng(), "this Megatron-Core (dev 2f2f8ebae or later) rewinds"
    validate_ernie55_moe_config(config)

    def main_set_all_rng_states(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker):
        """The signature of Megatron-Core main's, which snapshots generator handles."""

    monkeypatch.setattr(tp_random, "_set_all_rng_states", main_set_all_rng_states)
    with pytest.raises(ValueError, match=r"^ERNIE 5\.5 MoE: .*2f2f8ebae \(#5841\)"):
        validate_ernie55_moe_config(config)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param(dict(fp8="hybrid", use_te_rng_tracker=True), id="fp8-without-recompute"),
        pytest.param(dict(fp8="hybrid", **_MOE_RECOMPUTE), id="fp8-recompute-plain-tracker"),
        pytest.param(dict(use_te_rng_tracker=True, **_FULL_RECOMPUTE), id="bf16-full-recompute-te-tracker"),
        pytest.param(
            dict(
                recompute_granularity="selective", recompute_modules=["moe_act"], fp8="hybrid", use_te_rng_tracker=True
            ),
            id="moe-act-recompute-keeps-the-router",
        ),
    ],
)
def test_configurations_without_an_unsafe_replay_are_accepted(fields):
    validate_ernie55_moe_config(_with_fields(ernie_router_force_balance=True, **fields))


def test_every_rank_warns_once(pg_collection, monkeypatch, caplog):
    monkeypatch.setattr(fused_dispatch, "_STATE", fused_dispatch._DispatchState())
    with caplog.at_level(logging.WARNING, logger=fused_dispatch.__name__):
        _router(_config(), pg_collection)
        assert not [record for record in caplog.records if "ernie_router_force_balance" in record.getMessage()]
        for _ in range(2):
            _router(_config(ernie_router_force_balance=True), pg_collection)
    warnings = [record for record in caplog.records if "ernie_router_force_balance" in record.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "BENCHMARK ONLY" in warnings[0].getMessage() and "CHANGES THE TRAINING MATH" in warnings[0].getMessage()
