from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.engine import TrainingEngine
from megatron.bridge.engine.engine import Topology
from megatron.bridge.engine.export import ExportState, export_hf_weights, make_export_state


class _Bridge:
    def __init__(self):
        self.calls = []

    def get_conversion_tasks(self, model):
        self.calls.append("tasks")
        return ["task"]

    def export_hf_weights(self, model, **kwargs):
        self.calls.append(("stream", kwargs["conversion_tasks"]))
        yield SimpleNamespace(param_name="w", weight=torch.ones(2))

    def save_hf_pretrained(self, model, path, **kwargs):
        self.calls.append(("save", path))


def test_make_export_state_requires_hf_path_and_rejects_param_gather_overlap():
    with pytest.raises(ValueError, match="hf_path"):
        make_export_state(SimpleNamespace(hf_path=None, ddp=SimpleNamespace(overlap_param_gather=False)))
    with pytest.raises(NotImplementedError, match="overlap_param_gather"):
        make_export_state(SimpleNamespace(hf_path="x", ddp=SimpleNamespace(overlap_param_gather=True)))


def test_export_streams_with_cached_tasks_and_writes_directories():
    bridge = _Bridge()
    state = ExportState(bridge=bridge)
    first = list(export_hf_weights(state, [], path=None, streaming=True))
    second = list(export_hf_weights(state, [], path=None, streaming=True))
    assert [name for name, _ in first] == [name for name, _ in second] == ["w"]
    assert first[0][1].tolist() == second[0][1].tolist() == [1.0, 1.0]
    assert bridge.calls == ["tasks", ("stream", ["task"]), ("stream", ["task"])]
    assert export_hf_weights(state, [], path="/out", streaming=False) is None
    assert bridge.calls[-1] == ("save", "/out")
    with pytest.raises(ValueError, match="path"):
        export_hf_weights(state, [], path=None, streaming=False)


def test_engine_export_builds_state_once(monkeypatch):
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._closed = False
    engine._runtime = SimpleNamespace(topology=Topology(1, 1, 1, None, 1, 1, None, False))
    engine._recipe = SimpleNamespace(hf_path="x", ddp=SimpleNamespace(overlap_param_gather=False))
    engine._model = []
    engine._export_state = None
    built = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.make_export_state", lambda recipe: built.append(1) or ExportState(_Bridge())
    )
    assert engine.export(path="/a") is None
    assert list(engine.export(streaming=True))[0][0] == "w"
    assert built == [1]
