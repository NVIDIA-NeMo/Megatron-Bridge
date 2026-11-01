from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.engine import Datum, TrainingEngine, topology_from_recipe
from megatron.bridge.engine.engine import Topology, _SharedRuntime
from megatron.bridge.engine.recipes import check_supported, data_parallel_size


def _recipe_like() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=8,
            expert_model_parallel_size=16,
            expert_tensor_parallel_size=1,
            vocab_size=248320,
            seq_length=131072,
        ),
        ddp=SimpleNamespace(use_megatron_fsdp=False),
        dist=SimpleNamespace(
            lazy_mpu_init=False, use_megatron_fsdp=False, use_torch_fsdp2=False, use_decentralized_pg=False
        ),
        batch_layout="thd",
    )


def _unbuilt_engine(trainable: bool = True, closed: bool = False) -> TrainingEngine:
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._trainable = trainable
    engine._closed = closed
    engine._runtime = SimpleNamespace(topology=Topology(1, 1, 1, None, 1, 1, None, False))
    return engine


def _datums() -> list[Datum]:
    return [Datum(input_ids=torch.arange(3), target_ids=torch.arange(1, 4), loss_mask=torch.ones(3))]


def test_check_supported_accepts_baseline_shape():
    check_supported(_recipe_like())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: setattr(r.dist, "lazy_mpu_init", True), "lazy_mpu_init"),
        (lambda r: setattr(r.dist, "use_torch_fsdp2", True), "FSDP"),
        (lambda r: setattr(r.ddp, "use_megatron_fsdp", True), "FSDP"),
        (lambda r: setattr(r.model, "vocab_size", None), "vocab_size"),
        (lambda r: setattr(r.model, "seq_length", None), "seq_length"),
        (lambda r: setattr(r, "batch_layout", "sbhd"), "batch_layout"),
    ],
)
def test_check_supported_rejects_each_violation(mutate, message):
    recipe = _recipe_like()
    mutate(recipe)
    with pytest.raises(ValueError, match=message):
        check_supported(recipe)


def test_data_parallel_size_and_topology(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    recipe = _recipe_like()
    assert data_parallel_size(recipe) == 2
    assert topology_from_recipe(recipe) == Topology(16, 1, 1, None, 8, 16, 1, False)
    recipe.model.context_parallel_size = 4
    assert topology_from_recipe(recipe) != topology_from_recipe(_recipe_like())
    recipe.model.context_parallel_size = 3
    with pytest.raises(ValueError, match="not divisible"):
        data_parallel_size(recipe)


def test_constructor_by_name_is_not_implemented():
    with pytest.raises(NotImplementedError, match="from_recipe"):
        TrainingEngine("Qwen/Qwen3.5-35B-A3B")


def test_loss_dispatch_resolves_predefined_and_subclass_methods():
    class MyEngine(TrainingEngine):
        def my_loss(self, batch, output):
            return torch.zeros(())

    engine = MyEngine.__new__(MyEngine)
    assert engine.resolve_loss_method("cross_entropy").__func__ is TrainingEngine.cross_entropy
    assert engine.resolve_loss_method("ppo_clip").__func__ is TrainingEngine.ppo_clip
    assert engine.resolve_loss_method("my_loss")(None, None).item() == 0.0


@pytest.mark.parametrize(
    "name",
    ["optim_step", "close", "recipe", "preprocess", "postprocess", "_build", "", "does_not_exist", "normalization"],
)
def test_loss_dispatch_rejects_reserved_private_and_unknown_names(name):
    engine = _unbuilt_engine()
    with pytest.raises(ValueError):
        engine.resolve_loss_method(name)


def test_forward_backward_checks_loss_name_then_runs_pipeline_then_raises_placeholder():
    engine = _unbuilt_engine()
    with pytest.raises(ValueError, match="loss method"):
        engine.forward_backward(_datums(), loss_fn="nope")
    calls = []
    engine.preprocess = lambda data, training: calls.append(("preprocess", training)) or ["mb"]
    engine.check_microbatches = lambda microbatches: calls.append(("check", microbatches))
    with pytest.raises(NotImplementedError, match="phase B2"):
        engine.forward_backward(_datums())
    assert calls == [("preprocess", True), ("check", ["mb"])]


def test_forward_only_runs_preprocess_check_then_postprocess(monkeypatch):
    engine = _unbuilt_engine()
    calls = []
    engine.preprocess = lambda data, training: calls.append(("preprocess", training)) or ["mb"]
    engine.check_microbatches = lambda microbatches: calls.append(("check", microbatches))
    engine.postprocess = lambda microbatches, logprobs, gather_output: calls.append(("post", gather_output)) or "out"
    engine._runtime.pg_collection = None
    engine._model = []
    monkeypatch.setattr("megatron.bridge.engine.engine.run_forward_only", lambda model, mbs, pg_collection: ["lp"])
    assert engine.forward_only(_datums(), gather_output=False) == "out"
    assert calls == [("preprocess", False), ("check", ["mb"]), ("post", False)]


def test_frozen_engine_rejects_training_apis():
    engine = _unbuilt_engine(trainable=False)
    with pytest.raises(RuntimeError, match="trainable=False"):
        engine.forward_backward(_datums())
    with pytest.raises(RuntimeError, match="trainable=False"):
        engine.optim_step(None)


def test_closed_engine_rejects_calls():
    engine = _unbuilt_engine(closed=True)
    with pytest.raises(RuntimeError, match="closed"):
        engine.forward_only(_datums())


def test_shared_runtime_tracks_attached_engines():
    runtime = _SharedRuntime(Topology(1, 1, 1, None, 1, 1, None, False), pg_collection=None)
    a, b = _unbuilt_engine(), _unbuilt_engine()
    runtime.attach(a)
    runtime.attach(b)
    assert runtime.num_engines == 2
    runtime.detach(a)
    runtime.detach(a)
    assert runtime.num_engines == 1
