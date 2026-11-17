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

"""GPU test of the ERNIE 5.5 expert-load diagnostic at EP4, behind the alltoall and the HybridEP dispatchers.

Needs one node with 4 GPUs and Transformer Engine (the HybridEP cases also DeepEP's HybridEP). Launch it on all four
ranks, with cuBLAS's reproducible workspace setting (the comparison runs with deterministic algorithms)::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_expert_load_gpu.py

Without CUDA, Transformer Engine or a 4-rank launch, the tests skip; so do the HybridEP cases without HybridEP, unless
``ERNIE55_GPU_STRICT=1``, which turns that skip into a failure.

Each rank routes its own tokens through the ``E`` layer's MoE module of ``test_ernie55_moe_gpu.py`` (TEGroupedMLP,
4 of 16 experts per rank), with the expert bias raised for the experts of EP rank 2 so that the ranks receive
different loads, and an :class:`~megatron.bridge.models.ernie.modeling_ernie55.expert_load.ExpertLoadRecorder`
attached for 2 iterations of 2 micro-batches. Per case (alltoall; HybridEP; HybridEP with Megatron-Core's MoE
recompute), every rank checks:

* one record per micro-batch, in order, also when the recompute replays the forward in the backward;
* the routed counts stay on the GPU until the window ends (the probe copies nothing to the host inside it);
* the tokens each local expert receives equal the routed counts of that expert summed over the EP group, and EP rank
  2 receives the most;
* the file holds the records;
* the probe changes nothing. The same tokens run through the layer 4 times before the probe is attached and once with
  it, all with ``torch.use_deterministic_algorithms`` (warnings only), so that Megatron-Core's unfused unpermute adds
  with ``index_add_`` instead of the atomic ``scatter_add_``. Everything the probe could touch must be bitwise equal
  in every pass: the router's probabilities and routing map, and the experts' inputs (dispatched tokens, tokens per
  expert and probabilities), which are copies. The output and the input gradient must be bitwise equal when the 4
  plain passes are; otherwise (a dispatcher whose sums depend on arrival order) they must stay within twice the
  largest difference between two plain passes. Both are logged.

Every rank finishes its checks and collectives before the outcome is gathered, so a failed check on one rank does not
leave the others waiting in a collective.
"""

from __future__ import annotations

import itertools
import json
import logging
import os

import pytest
import torch

from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import (  # noqa: F401 - pytest fixtures
    EP,
    HIDDEN,
    SEED,
    TOKENS,
    _fresh_moe_metrics,
    build_moe,
    ep4,
    make_config,
)


logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the ERNIE 5.5 expert-load GPU test needs CUDA devices"),
]

HYBRIDEP = dict(moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="hybridep")
CASES = {
    "alltoall": {},
    "hybridep": HYBRIDEP,
    "hybridep_recompute": dict(HYBRIDEP, recompute_granularity="selective", recompute_modules=["moe"]),
}
HOT_RANK = 2
HOT_BIAS = 0.2
STEPS, MICROBATCHES = 2, 2
PLAIN_PASSES = 4


def _hybridep_available() -> bool:
    try:
        from megatron.core.transformer.moe import fused_a2a
    except Exception:  # noqa: BLE001 - any import failure means no HybridEP
        return False
    return bool(getattr(fused_a2a, "HAVE_HYBRIDEP", False))


@pytest.fixture
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms(True, warn_only=True)`` for one test, then the previous mode.

    Uninitialized memory is not filled: the passes compare the layer as training runs it.
    """
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    yield
    torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    torch.utils.deterministic.fill_uninitialized_memory = previous[2]


def _watch(moe) -> list[tuple[str, torch.Tensor | None]]:
    """Copies of what the probe could change, appended at every router and experts call of ``moe``.

    The router's outputs (probabilities and routing map) and the experts' inputs (dispatched tokens, tokens per expert
    and probabilities): the probe wraps the router's ``routing`` and the dispatcher's ``dispatch_postprocess``.
    """
    seen: list[tuple[str, torch.Tensor | None]] = []

    def keep(name: str, value) -> None:
        seen.append((name, None if value is None else torch.as_tensor(value).detach().clone()))

    def router_outputs(module, args, output) -> None:
        keep("router probs", output[0])
        keep("router routing map", output[1])

    def expert_inputs(module, args) -> None:
        for name, value in zip(("dispatched tokens", "tokens per expert", "dispatched probs"), args):
            keep(name, value)

    moe.router.register_forward_hook(router_outputs)
    moe.experts.register_forward_pre_hook(expert_inputs)
    return seen


def _forward_backward(moe, seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """One micro-batch of this rank's tokens; returns the output and the input gradient."""
    import torch.distributed as dist

    generator = torch.Generator().manual_seed(SEED + 1000 * seed + 17 * dist.get_rank())
    hidden = torch.randn(TOKENS, 1, HIDDEN, generator=generator, dtype=torch.float64)
    hidden = hidden.to(device=device, dtype=moe.config.params_dtype).requires_grad_(True)
    output, _ = moe(hidden)
    cotangent = torch.randn(output.shape, generator=generator, dtype=torch.float64)
    output.backward(cotangent.to(device=device, dtype=output.dtype))
    return output.detach().clone(), hidden.grad.detach().clone()


def _pass(moe, seen: list, seed: int, device) -> tuple[torch.Tensor, torch.Tensor, list]:
    """One micro-batch: the output, the input gradient and what :func:`_watch` saw during it."""
    first = len(seen)
    output, d_input = _forward_backward(moe, seed=seed, device=device)
    return output, d_input, seen[first:]


def _bitwise(a: torch.Tensor | None, b: torch.Tensor | None) -> bool:
    if a is None or b is None:
        return a is b
    return a.dtype == b.dtype and a.shape == b.shape and a.device == b.device and torch.equal(a, b)


def _max_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.double() - b.double()).abs().max().item()


def probe_changes_nothing(plain: list, probed: tuple) -> tuple[list[str], dict[str, str]]:
    """Compare passes of the same tokens without the probe (``plain``) and with it (``probed``).

    Args:
        plain: :func:`_pass` results before the probe was attached (at least 2).
        probed: The :func:`_pass` result of the same tokens with the probe attached.

    Returns:
        The failures, and per compared tensor how it compared (for the log).
    """
    failures, outcome = [], {}
    reference = plain[-1][2]
    names = [name for name, _ in reference]
    for label, watched in [*((f"plain pass {k}", p[2]) for k, p in enumerate(plain)), ("probed pass", probed[2])]:
        if [name for name, _ in watched] != names:
            failures.append(f"{label}: saw {[name for name, _ in watched]}, the last plain pass {names}")
            continue
        for (name, value), (_, expected) in zip(watched, reference):
            if not _bitwise(value, expected):
                failures.append(f"{label}: {name} is not bitwise that of the last plain pass")
    if not failures:
        outcome["router outputs and expert inputs"] = f"bitwise equal in all {len(plain) + 1} passes ({len(names)})"
    for index, name in enumerate(("output", "d input")):
        values = [result[index] for result in plain]
        spread = max(_max_difference(a, b) for a, b in itertools.combinations(values, 2))
        difference = _max_difference(values[-1], probed[index])
        if all(_bitwise(values[0], other) for other in values[1:]):
            same = _bitwise(values[-1], probed[index])
            outcome[name] = "plain passes bitwise repeatable; probed pass " + ("bitwise equal" if same else "differs")
            if not same:
                failures.append(f"{name}: differs by {difference} with the probe attached (plain passes bitwise)")
        else:
            outcome[name] = f"plain passes differ by up to {spread}; probed pass by {difference}"
            if difference > 2 * spread:
                failures.append(f"{name}: differs by {difference} with the probe attached (plain passes {spread})")
    return failures, outcome


@pytest.mark.parametrize("case", sorted(CASES))
def test_expert_load_records_match_the_dispatch(ep4, case, tmp_path, deterministic_algorithms):
    import torch.distributed as dist

    from megatron.bridge.models.ernie.modeling_ernie55.expert_load import ExpertLoadRecorder

    if CASES[case].get("moe_flex_dispatcher_backend") == "hybridep" and not _hybridep_available():
        reason = "HybridEP (DeepEP's hybrid_ep) is not installed"
        if os.environ.get("ERNIE55_GPU_STRICT") == "1":
            pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
        pytest.skip(reason)
    device = torch.cuda.current_device()
    moe = build_moe(make_config(**CASES[case]), ep4, device=device)
    assert moe.moe_layer_recompute is ("recompute" in case)
    ep_group, ep_rank, local = ep4.ep, ep4.ep.rank(), moe.num_local_experts
    with torch.no_grad():
        moe.router.expert_bias[HOT_RANK * local : (HOT_RANK + 1) * local] += HOT_BIAS
    failures: list[str] = []
    seen = _watch(moe)

    # The same tokens (seed 0) without the probe, then with it as the first probed micro-batch.
    plain = [_pass(moe, seen, seed=0, device=device) for _ in range(PLAIN_PASSES)]
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, STEPS))
    recorder.attach(moe)
    results = []
    for step in range(STEPS):
        for microbatch in range(MICROBATCHES):
            recorder.begin_microbatch(step)
            results.append(_pass(moe, seen, seed=step * MICROBATCHES + microbatch, device=device))
    labels = [(record[0], record[1]) for record in recorder._records]
    if labels != [(step, microbatch) for step in range(STEPS) for microbatch in range(MICROBATCHES)]:
        failures.append(f"records {labels}")
    if not all(isinstance(record[4], torch.Tensor) and record[4].is_cuda for record in recorder._records):
        failures.append("routed counts left the GPU inside the window")
    received_on = sorted({"host" if isinstance(record[3], list) else "device" for record in recorder._records})

    # Every rank takes part in every collective, whatever its own checks found.
    records = recorder.records()
    for record in records:
        routed = torch.tensor(record["routed"], dtype=torch.int64, device=device)
        dist.all_reduce(routed, group=ep_group)
        expected = routed[ep_rank * local : (ep_rank + 1) * local].tolist()
        if record["received"] != expected:
            failures.append(
                f"step {record['step']} mb {record['microbatch']}: received {record['received']}, {expected}"
            )
    received = torch.tensor([sum(sum(record["received"]) for record in records)], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(received) for _ in range(EP)]
    dist.all_gather(gathered, received, group=ep_group)
    loads = [int(load.item()) for load in gathered]
    if loads.index(max(loads)) != HOT_RANK:
        failures.append(
            f"received per EP rank {loads}: the raised expert bias did not make rank {HOT_RANK} the busiest"
        )

    recorder.begin_microbatch(STEPS)
    payload = json.loads(open(recorder.dumped_path).read())
    if payload["records"] != records or payload["ep_rank"] != ep_rank or payload["num_local_experts"] != local:
        failures.append("the file does not hold the records")

    unchanged, outcome = probe_changes_nothing(plain, results[0])
    failures += unchanged
    logger.info(
        "expert load %s, rank %d: received per EP rank %s (hot rank %d), received counts read on the %s; probe on vs "
        "off: %s (CUBLAS_WORKSPACE_CONFIG=%s)",
        case,
        dist.get_rank(),
        loads,
        HOT_RANK,
        "/".join(received_on),
        outcome,
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "unset"),
    )
    failed_ranks = torch.tensor([int(bool(failures))], device=device)
    dist.all_reduce(failed_ranks)
    assert not failures, f"{case}, rank {dist.get_rank()}: {failures}"
    assert failed_ranks.item() == 0, f"{case}: {failed_ranks.item()} rank(s) failed; see their logs"
