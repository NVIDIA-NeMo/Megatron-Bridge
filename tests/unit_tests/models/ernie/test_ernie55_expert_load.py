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

"""CPU tests of the opt-in ERNIE 5.5 expert-load diagnostic (``modeling_ernie55.expert_load``).

* Off (the default) installs nothing and never reads the training state.
* On, one record per MoE layer and training micro-batch inside the window, with or without recompute (Megatron-Core's
  own MoE recompute and reentrant checkpointing), and nothing outside the window or for evaluation forwards.
* The counts: ``routed`` is ``routing_map.sum(0)`` without padding rows, ``received`` the dispatcher's
  ``tokens_per_expert`` (equal to ``routed`` on one rank without padding).
* Outputs and gradients are bitwise those of a layer without the diagnostic.
* The file: format, window, layers and records, written when the window ends.

The MoE layers run on CPU with the all-gather dispatcher and ``SequentialMLP`` experts (``test_ernie55_moe.py``);
the GPU file checks HybridEP at EP 4.
"""

import atexit
import json
from types import SimpleNamespace

import pytest
import torch
import torch.utils.checkpoint
from megatron.core.tensor_parallel import random as tp_random

from megatron.bridge.models.ernie.modeling_ernie55 import expert_load
from megatron.bridge.models.ernie.modeling_ernie55.expert_load import (
    ENV_DIR,
    ENV_STEPS,
    ENV_SWITCH,
    FORMAT_VERSION,
    MAX_WINDOW_STEPS,
    ExpertLoadRecorder,
    expert_load_logging_enabled,
    expert_load_recorder,
    flush_expert_load,
    note_microbatch,
)
from tests.unit_tests.models.ernie.test_ernie55_moe import (  # noqa: F401 - pytest fixtures
    BATCH,
    EXPERTS,
    HIDDEN,
    SEQ,
    TOPK,
    _config,
    _fresh_moe_metrics,
    _moe,
    pg_collection,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _no_process_recorder(monkeypatch):
    """Start every test with the diagnostic off and no process recorder; drop the one a test created."""
    for name in (ENV_SWITCH, ENV_DIR, ENV_STEPS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(expert_load, "_RECORDER", None)
    yield
    recorder = expert_load._RECORDER
    if recorder is not None:
        atexit.unregister(recorder._dump_at_exit)


def _hidden(seed: int) -> torch.Tensor:
    return torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(seed)).requires_grad_(True)


def _routing_counts(moe, hidden: torch.Tensor, padding: torch.Tensor | None = None) -> list[int]:
    """Routed count per expert from the router alone (call outside the window: the probe records nothing)."""
    with torch.no_grad():
        _, routing_map = moe.router(hidden.detach())
    if padding is not None:
        routing_map = routing_map & ~padding.reshape(-1, 1)
    return routing_map.sum(dim=0).tolist()


def _train_step(moe, recorder: ExpertLoadRecorder, step: int, seeds: list[int], *, checkpoint: bool = False) -> None:
    """One iteration of ``len(seeds)`` micro-batches, each forward then backward (as with one pipeline stage)."""
    for seed in seeds:
        recorder.begin_microbatch(step)
        hidden = _hidden(seed)
        if checkpoint:
            output = torch.utils.checkpoint.checkpoint(lambda x: moe(x)[0], hidden, use_reentrant=True)
        else:
            output, _ = moe(hidden)
        output.square().sum().backward()


# ---------------------------------------------------------------------------------------------
# Switch and settings
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("value, on", [("1", True), ("true", True), ("ON", True), ("0", False), ("", False)])
def test_switch_values(monkeypatch, value, on):
    monkeypatch.setenv(ENV_SWITCH, value)
    assert expert_load_logging_enabled() is on


def test_switch_rejects_other_values(monkeypatch):
    monkeypatch.setenv(ENV_SWITCH, "2")
    with pytest.raises(ValueError, match=ENV_SWITCH):
        expert_load_logging_enabled()


def test_off_by_default_installs_nothing(pg_collection):
    """Without the switch the layer keeps its classes' methods, and the step hook never reads the state."""
    moe = _moe(_config(), pg_collection, seed=3)
    assert "routing" not in vars(moe.router)
    assert "dispatch_postprocess" not in vars(moe.token_dispatcher)
    assert expert_load_recorder() is None
    note_microbatch(SimpleNamespace(), training=True)  # no train_state: it must not be read
    assert flush_expert_load() is None


@pytest.mark.parametrize(
    "env, message",
    [
        ({}, ENV_DIR),
        ({ENV_STEPS: "5"}, "<start>:<end>"),
        ({ENV_STEPS: "4:4"}, "start < end"),
        ({ENV_STEPS: f"0:{MAX_WINDOW_STEPS + 1}"}, f"at most {MAX_WINDOW_STEPS}"),
    ],
)
def test_switch_needs_valid_settings(pg_collection, monkeypatch, tmp_path, env, message):
    monkeypatch.setenv(ENV_SWITCH, "1")
    if ENV_STEPS in env:
        monkeypatch.setenv(ENV_DIR, str(tmp_path))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        _moe(_config(), pg_collection, seed=3)


def test_window_comes_from_the_profiler_or_raises(tmp_path):
    recorder = ExpertLoadRecorder(str(tmp_path))
    with pytest.raises(ValueError, match=ENV_STEPS):
        recorder.begin_microbatch(0, profiling=SimpleNamespace(use_nsys_profiler=False, use_pytorch_profiler=False))
    profiling = SimpleNamespace(
        use_nsys_profiler=True, use_pytorch_profiler=False, profile_step_start=50, profile_step_end=52
    )
    recorder.begin_microbatch(0, profiling=profiling)
    assert recorder.window == (50, 52)


# ---------------------------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------------------------


def test_records_one_entry_per_layer_and_microbatch_inside_the_window(pg_collection, tmp_path):
    moe = _moe(_config(), pg_collection, seed=5)
    recorder = ExpertLoadRecorder(str(tmp_path), window=(1, 3))
    recorder.attach(moe)
    assert "routing" in vars(moe.router) and "dispatch_postprocess" in vars(moe.token_dispatcher)
    seeds = {step: [10 * step + 1, 10 * step + 2] for step in range(3)}
    for step in range(3):
        _train_step(moe, recorder, step, seeds[step])
    records = recorder.records()
    assert [(r["step"], r["microbatch"], r["slot"]) for r in records] == [(1, 0, 0), (1, 1, 0), (2, 0, 0), (2, 1, 0)]
    # Step 3 starts after the window: the file is written and nothing more is recorded.
    recorder.begin_microbatch(3)
    assert recorder.dumped_path is not None and not recorder.recording
    for record in records:
        expected = _routing_counts(moe, _hidden(seeds[record["step"]][record["microbatch"]]))
        assert record["routed"] == expected
        # One rank: the local experts receive exactly the routed tokens.
        assert record["received"] == expected
        assert sum(expected) == SEQ * BATCH * TOPK and len(expected) == EXPERTS


def test_padding_rows_are_not_routed_counts(pg_collection, tmp_path):
    moe = _moe(_config(), pg_collection, seed=6)
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, 1))
    recorder.attach(moe)
    padding = torch.zeros(SEQ, BATCH, dtype=torch.bool)
    padding[-4:, 0] = True
    hidden = _hidden(21)
    recorder.begin_microbatch(0)
    # The layer takes the padding mask batch-first, the router sees it sequence-first.
    output, _ = moe(hidden, padding_mask=padding.transpose(0, 1))
    output.sum().backward()
    (record,) = recorder.records()
    recorder.begin_microbatch(1)
    assert record["routed"] == _routing_counts(moe, hidden, padding)
    assert sum(record["routed"]) == (SEQ * BATCH - 4) * TOPK


@pytest.mark.parametrize("recompute", ["checkpoint", "megatron"])
def test_recompute_replay_is_not_recorded_again(pg_collection, tmp_path, monkeypatch, recompute):
    """A forward without autograd is recorded and its replay in the backward is not; outputs stay bitwise."""
    if recompute == "megatron":
        # Megatron-Core's MoE recompute (tensor_parallel.checkpoint) saves the CUDA RNG state; there is none on CPU.
        monkeypatch.setattr(tp_random, "_get_cuda_rng_state", lambda *args, **kwargs: torch.empty(0))
        monkeypatch.setattr(tp_random, "_set_cuda_rng_state", lambda *args, **kwargs: None)
        config = _config(recompute_granularity="selective", recompute_modules=["moe"])
    else:
        config = _config()
    moe = _moe(config, pg_collection, seed=7)
    assert moe.moe_layer_recompute is (recompute == "megatron")
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, 2))
    recorder.attach(moe)
    for step in range(2):
        _train_step(moe, recorder, step, [31 + step, 41 + step], checkpoint=recompute == "checkpoint")
        assert not recorder._probes[0].replays, "every recorded forward was replayed"
    records = recorder.records()
    recorder.begin_microbatch(2)
    assert [(r["step"], r["microbatch"]) for r in records] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert records[2]["routed"] == _routing_counts(moe, _hidden(32))
    assert records[2]["received"] == records[2]["routed"]


@pytest.mark.parametrize("no_grad_first", [False, True])
def test_layer_without_dispatcher_records_routed_counts(pg_collection, tmp_path, no_grad_first):
    """A megakernel layer (no token dispatcher) records at routing time, once per micro-batch, received null."""
    moe = _moe(_config(), pg_collection, seed=9)
    moe.token_dispatcher = None
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, 1))
    recorder.attach(moe)
    assert recorder._probes[0].direct
    hiddens = [_hidden(51), _hidden(52)]
    for hidden in hiddens:
        recorder.begin_microbatch(0)
        if no_grad_first:
            # MoE recompute: a forward without autograd, then its replay with autograd in the backward.
            with torch.no_grad():
                moe.router(hidden.detach())
        moe.router(hidden)
    records = recorder.records()
    recorder.begin_microbatch(1)
    assert [(r["step"], r["microbatch"], r["slot"]) for r in records] == [(0, 0, 0), (0, 1, 0)]
    for record, hidden in zip(records, hiddens):
        assert record["received"] is None
        assert record["routed"] == _routing_counts(moe, hidden)
        assert sum(record["routed"]) == SEQ * BATCH * TOPK
    assert not recorder._probes[0].replays
    assert "unrouted_dispatches" not in recorder.payload()


def test_evaluation_forwards_are_ignored(pg_collection, tmp_path):
    moe = _moe(_config(), pg_collection, seed=8)
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, 1))
    recorder.attach(moe)
    recorder.begin_microbatch(0)
    recorder.begin_microbatch(0, training=False)
    assert recorder.microbatch == 0, "an evaluation forward does not take a micro-batch number"
    moe.eval()
    with torch.no_grad():
        moe(_hidden(1))
    assert recorder.records() == [] and not recorder._probes[0].replays
    moe.train()
    output, _ = moe(_hidden(2))
    output.sum().backward()
    assert len(recorder.records()) == 1


def test_outputs_and_gradients_are_bitwise_unchanged(pg_collection, tmp_path):
    config = _config()
    plain, probed = _moe(config, pg_collection, seed=9), _moe(config, pg_collection, seed=9)
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, 1))
    recorder.attach(probed)
    recorder.begin_microbatch(0)
    results = []
    for moe in (plain, probed):
        hidden = _hidden(5)
        output, _ = moe(hidden)
        output.square().sum().backward()
        results.append((output.detach(), hidden.grad, {name: p.grad for name, p in moe.named_parameters()}))
    assert len(recorder.records()) == 1
    (out_a, dx_a, grads_a), (out_b, dx_b, grads_b) = results
    assert torch.equal(out_a, out_b) and torch.equal(dx_a, dx_b)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        assert torch.equal(grads_a[name], grads_b[name]), name


# ---------------------------------------------------------------------------------------------
# Output file and the process recorder
# ---------------------------------------------------------------------------------------------


def test_file_holds_the_layers_and_records(pg_collection, tmp_path, monkeypatch):
    """The environment path: two layers attached at construction, the step hook, and the file at the window end."""
    monkeypatch.setenv(ENV_SWITCH, "1")
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "expert_load"))
    monkeypatch.setenv(ENV_STEPS, "2:3")
    config = _config()
    first, second = _moe(config, pg_collection, seed=11), _moe(config, pg_collection, seed=12)
    second.set_layer_number(4)
    recorder = expert_load_recorder()
    assert recorder is not None and recorder.window == (2, 3)

    def state(step: int) -> SimpleNamespace:
        return SimpleNamespace(train_state=SimpleNamespace(step=step), cfg=SimpleNamespace(profiling=None))

    for step in (1, 2):
        for seed in (1, 2):
            note_microbatch(state(step), training=True)
            output, _ = second(first(_hidden(seed))[0])
            output.sum().backward()
    note_microbatch(state(3), training=True)
    path = tmp_path / "expert_load" / "expert_load_rank00000.json"
    assert recorder.dumped_path == str(path) and flush_expert_load() is None
    payload = json.loads(path.read_text())
    assert payload["format"] == FORMAT_VERSION and payload["window"] == [2, 3]
    assert payload["rank"] == 0 and payload["world_size"] == 1
    assert payload["num_experts"] == EXPERTS and payload["topk"] == TOPK
    assert payload["ep_rank"] == 0 and payload["ep_size"] == 1
    assert payload["num_local_experts"] == EXPERTS and payload["local_expert_indices"] == list(range(EXPERTS))
    assert payload["layers"] == [
        {"slot": 0, "layer_number": 1, "is_mtp_layer": False},
        {"slot": 1, "layer_number": 4, "is_mtp_layer": False},
    ]
    keys = [(r["step"], r["microbatch"], r["slot"]) for r in payload["records"]]
    assert keys == [(2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)]
    assert all(sum(r["routed"]) == SEQ * BATCH * TOPK for r in payload["records"])
    assert "unrouted_dispatches" not in payload
    assert not list((tmp_path / "expert_load").glob("*.tmp.*"))


def test_flush_writes_an_unfinished_window(pg_collection, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_SWITCH, "1")
    monkeypatch.setenv(ENV_DIR, str(tmp_path))
    monkeypatch.setenv(ENV_STEPS, "0:10")
    moe = _moe(_config(), pg_collection, seed=13)
    note_microbatch(SimpleNamespace(train_state=SimpleNamespace(step=0), cfg=None), training=True)
    output, _ = moe(_hidden(3))
    output.sum().backward()
    path = flush_expert_load()
    assert path is not None and len(json.loads(open(path).read())["records"]) == 1
    assert flush_expert_load() is None


def test_ranks_are_read_while_the_process_groups_exist(pg_collection, tmp_path, monkeypatch):
    """A file written at exit, after the process groups are gone, keeps the ranks read at the first micro-batch."""
    moe = _moe(_config(), pg_collection, seed=16)
    recorder = ExpertLoadRecorder(str(tmp_path), window=(0, 1))
    recorder.attach(moe)
    monkeypatch.setattr(expert_load, "_global_rank_and_size", lambda: (7, 8))
    recorder.begin_microbatch(0)
    output, _ = moe(_hidden(4))
    output.sum().backward()
    monkeypatch.setattr(expert_load, "_global_rank_and_size", lambda: (0, 1))
    recorder._dump_at_exit()
    payload = json.loads((tmp_path / "expert_load_rank00007.json").read_text())
    assert (payload["rank"], payload["world_size"], payload["ep_rank"], payload["ep_size"]) == (7, 8, 0, 1)
    assert len(payload["records"]) == 1


# ---------------------------------------------------------------------------------------------
# EP 4 on spawned gloo ranks: received = routed summed over the EP group
# ---------------------------------------------------------------------------------------------

EP_WORLD = 4
HOT_RANK = 2


def setup_ep_rank(world_size: int):
    """EP = ``world_size`` model-parallel state on a spawned gloo rank (CPU)."""
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    # Router buffers are allocated on torch.cuda.current_device().
    torch.cuda.current_device = lambda: "cpu"
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world_size
    )
    return ProcessGroupCollection.use_mpu_process_groups()


def case_received_is_the_routed_count_summed_over_ep(pg_collection) -> None:
    """Every rank routes its own tokens; each local expert receives the sum of every rank's routed count."""
    import tempfile

    config = _config(expert_model_parallel_size=EP_WORLD)
    moe = _moe(config, pg_collection, seed=15)
    ep_group, ep_rank, local = pg_collection.ep, pg_collection.ep.rank(), moe.num_local_experts
    assert local * EP_WORLD == EXPERTS
    with torch.no_grad():
        moe.router.expert_bias[HOT_RANK * local : (HOT_RANK + 1) * local] += 0.3
    with tempfile.TemporaryDirectory() as directory:
        recorder = ExpertLoadRecorder(directory, window=(0, 1))
        recorder.attach(moe)
        for microbatch in range(2):
            recorder.begin_microbatch(0)
            output, _ = moe(_hidden(100 * torch.distributed.get_rank() + microbatch))
            output.square().sum().backward()
        records = recorder.records()
        assert [(r["step"], r["microbatch"]) for r in records] == [(0, 0), (0, 1)]
        recorder.begin_microbatch(1)
        payload = json.loads(open(recorder.dumped_path).read())
    assert payload["ep_rank"] == ep_rank and payload["ep_size"] == EP_WORLD
    assert payload["local_expert_indices"] == list(range(ep_rank * local, (ep_rank + 1) * local))
    assert payload["records"] == records
    for record in records:
        routed = torch.tensor(record["routed"], dtype=torch.int64)
        assert int(routed.sum()) == SEQ * BATCH * TOPK
        torch.distributed.all_reduce(routed, group=ep_group)
        assert record["received"] == routed[ep_rank * local : (ep_rank + 1) * local].tolist()
    load = torch.tensor([sum(sum(r["received"]) for r in records)])
    loads = [torch.zeros_like(load) for _ in range(EP_WORLD)]
    torch.distributed.all_gather(loads, load, group=ep_group)
    loads = [int(value) for value in loads]
    assert loads.index(max(loads)) == HOT_RANK, loads


def test_ep4_received_counts_on_gloo_ranks(tmp_path):
    from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases

    case = "case_received_is_the_routed_count_summed_over_ep"
    results = run_cases(
        world_size=EP_WORLD, module_name=__name__, setup_name="setup_ep_rank", case_names=[case], workdir=tmp_path
    )
    failures = {rank: result["error"] for rank, result in results[case].items() if not result["ok"]}
    assert len(results[case]) == EP_WORLD
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())
