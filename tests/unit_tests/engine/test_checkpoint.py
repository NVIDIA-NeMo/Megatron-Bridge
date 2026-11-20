from types import SimpleNamespace

import pytest
import torch
import yaml

from megatron.bridge.engine import CheckpointHandle, TrainingEngine
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.checkpoint import (
    _CheckpointConfigView,
    checkpoint_root,
    read_user_state,
    save_engine_state,
    user_state_file,
    write_user_state,
)
from megatron.bridge.engine.engine import Topology
from megatron.bridge.engine.router_replay import RouterReplayState
from megatron.bridge.training.checkpointing import _get_run_config_tp_pp
from megatron.bridge.training.config import CheckpointConfig


def test_run_config_yaml_carries_topology_and_save_flags(tmp_path):
    model = SimpleNamespace(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=8,
        expert_model_parallel_size=16,
        expert_tensor_parallel_size=1,
    )
    checkpoint = CheckpointConfig(
        save=str(tmp_path), load=None, save_optim=True, save_rng=True, ckpt_format="torch_dist"
    )
    view = _CheckpointConfigView(model=model, optimizer=None, ddp=None, dist=None, rng=None, checkpoint=checkpoint)
    assert view.ft is None and view.dataset is None and view.scheduler.override_opt_param_scheduler is False
    path = tmp_path / "run_config.yaml"
    view.to_yaml(str(path))
    run_config = yaml.safe_load(path.read_text())
    assert _get_run_config_tp_pp(run_config["model"]) == (1, 1)
    assert run_config["model"]["context_parallel_size"] == 8
    assert run_config["checkpoint"]["save_optim"] is True and run_config["checkpoint"]["save_rng"] is True
    assert run_config["checkpoint"]["fully_parallel_save"] is checkpoint.fully_parallel_save


def test_user_state_sidecar_round_trip(tmp_path):
    root = str(tmp_path)
    write_user_state(root, 25, 0, {"cursor": 7, "weights": torch.arange(3)})
    write_user_state(root, 25, 1, None)
    assert user_state_file(root, 25, 0).endswith("user_state_0000025/rank_00000.pt")
    restored = read_user_state(root, 25, 0)
    assert restored["cursor"] == 7 and restored["weights"].tolist() == [0, 1, 2]
    assert (tmp_path / "user_state_0000025" / "rank_00001.pt").exists()
    assert read_user_state(root, 25, 1) is None
    assert read_user_state(root, 26, 0) is None
    write_user_state(root, 25, 0, None)
    assert read_user_state(root, 25, 0) is None
    assert checkpoint_root(root + "/iter_0000025") == root
    assert checkpoint_root(root + "/") == root


def _engine_with_window(open_window: bool) -> TrainingEngine:
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._trainable = True
    engine._closed = False
    engine._resident = True
    engine._router_replay = RouterReplayState([])
    engine._runtime = SimpleNamespace(
        topology=Topology(1, 1, 1, None, 1, 1, None, False), detach=lambda engine: None, pg_collection=None
    )
    engine._recipe = SimpleNamespace()
    engine._model = []
    engine._optimizer = None
    engine._step = 3
    engine._window = AccumulationWindow()
    engine._pending_save = None
    engine._async_checkpoint_state = None
    engine._peft = None
    if open_window:
        engine._window.add(num_loss_tokens=1, num_samples=1, loss_sum=0.0)
    return engine


def test_async_save_requires_the_worker_started_at_construction():
    with pytest.raises(ValueError, match="spawn_async_save_worker=False"):
        save_engine_state(
            None, [], None, step=0, root="/x", user_state=None, async_save=True, async_state=None, pg_collection=None
        )


def test_state_apis_require_a_completed_update():
    engine = _engine_with_window(open_window=True)
    with pytest.raises(RuntimeError, match="completed update"):
        engine.save_state("/nowhere")
    with pytest.raises(RuntimeError, match="completed update"):
        engine.load_state("/nowhere")


def test_save_and_load_state_delegate_and_track_pending_saves(monkeypatch):
    engine = _engine_with_window(open_window=False)
    waits = []
    saves = []

    def fake_save(recipe, model, optimizer, *, step, root, user_state, async_save, async_state, pg_collection, peft):
        assert peft is None
        saves.append((step, root, user_state, async_save))
        waiter = (lambda: waits.append(root)) if async_save else None
        return CheckpointHandle(path=f"{root}/iter_{step:07d}", _waiter=waiter)

    monkeypatch.setattr("megatron.bridge.engine.engine.save_engine_state", fake_save)
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.load_engine_state",
        lambda recipe, model, optimizer, *, path, peft: (9, {"from": path}),
    )
    first = engine.save_state("/a", user_state={"k": 1}, async_save=True)
    assert first.path == "/a/iter_0000003" and engine._pending_save is first
    engine.save_state("/b")
    assert waits == ["/a"] and engine._pending_save is None
    assert saves == [(3, "/a", {"k": 1}, True), (3, "/b", None, False)]
    assert engine.load_state("/c") == {"from": "/c"} and engine.step == 9
    engine.wait_for_save()
    assert waits == ["/a"]
    second = engine.save_state("/d", async_save=True)
    engine.wait_for_save()
    assert waits == ["/a", "/d"] and engine._pending_save is None
    third = engine.save_state("/e", async_save=True)
    engine.close()
    assert waits == ["/a", "/d", "/e"] and engine.closed
    second.wait()
    third.wait()
    assert waits == ["/a", "/d", "/e"]
