from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.engine import Datum, TrainingEngine, enforce_engine_conventions, topology_from_config
from megatron.bridge.engine.engine import Topology, _SharedRuntime


def _good_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            calculate_per_token_loss=True,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            virtual_pipeline_model_parallel_size=None,
            context_parallel_size=8,
            expert_model_parallel_size=16,
            expert_tensor_parallel_size=1,
        ),
        ddp=SimpleNamespace(average_in_collective=False),
        train=SimpleNamespace(micro_batch_size=1),
        checkpoint=SimpleNamespace(load=None),
        dist=SimpleNamespace(
            lazy_mpu_init=False, use_megatron_fsdp=False, use_torch_fsdp2=False, use_decentralized_pg=False
        ),
        peft=None,
    )


def _unbuilt_engine(trainable: bool = True, closed: bool = False) -> TrainingEngine:
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._trainable = trainable
    engine._closed = closed
    return engine


def test_conventions_accept_baseline_shape():
    enforce_engine_conventions(_good_cfg())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: setattr(c.model, "calculate_per_token_loss", False), "calculate_per_token_loss"),
        (lambda c: setattr(c.ddp, "average_in_collective", True), "average_in_collective"),
        (lambda c: setattr(c.train, "micro_batch_size", 2), "micro_batch_size"),
        (lambda c: setattr(c.checkpoint, "load", "/some/ckpt"), "checkpoint.load"),
        (lambda c: setattr(c.dist, "lazy_mpu_init", True), "lazy_mpu_init"),
        (lambda c: setattr(c.dist, "use_torch_fsdp2", True), "FSDP"),
        (lambda c: setattr(c, "peft", object()), "peft"),
    ],
)
def test_conventions_reject_each_violation(mutate, message):
    cfg = _good_cfg()
    mutate(cfg)
    with pytest.raises(ValueError, match=message):
        enforce_engine_conventions(cfg)


def test_conventions_report_all_violations_at_once():
    cfg = _good_cfg()
    cfg.model.calculate_per_token_loss = False
    cfg.train.micro_batch_size = 4
    with pytest.raises(ValueError) as excinfo:
        enforce_engine_conventions(cfg)
    assert "calculate_per_token_loss" in str(excinfo.value) and "micro_batch_size" in str(excinfo.value)


def test_topology_from_config_and_equality(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    a = topology_from_config(_good_cfg())
    b = topology_from_config(_good_cfg())
    assert a == b == Topology(16, 1, 1, None, 8, 16, 1, False)
    cfg = _good_cfg()
    cfg.model.context_parallel_size = 4
    assert topology_from_config(cfg) != a


def test_constructor_by_name_is_not_implemented():
    with pytest.raises(NotImplementedError, match="from_config"):
        TrainingEngine("Qwen/Qwen3.5-35B-A3B")


def test_loss_dispatch_resolves_predefined_and_subclass_methods():
    class MyEngine(TrainingEngine):
        def my_loss(self, datums, model_output):
            return torch.zeros(()), {}

    engine = MyEngine.__new__(MyEngine)
    assert engine.resolve_loss_method("cross_entropy").__func__ is TrainingEngine.cross_entropy
    assert engine.resolve_loss_method("ppo_clip").__func__ is TrainingEngine.ppo_clip
    assert engine.resolve_loss_method("my_loss")([], None)[0].item() == 0.0


@pytest.mark.parametrize("name", ["optim_step", "close", "config", "_build", "", "does_not_exist", "normalization"])
def test_loss_dispatch_rejects_reserved_private_and_unknown_names(name):
    engine = _unbuilt_engine()
    with pytest.raises(ValueError):
        engine.resolve_loss_method(name)


def test_forward_backward_validates_before_raising_not_implemented():
    engine = _unbuilt_engine()
    datums = [Datum(input_ids=torch.arange(3))]
    with pytest.raises(ValueError, match="loss method"):
        engine.forward_backward(datums, loss_fn="nope")
    with pytest.raises(NotImplementedError, match="phase B2"):
        engine.forward_backward(datums)


def test_frozen_engine_rejects_training_apis():
    engine = _unbuilt_engine(trainable=False)
    with pytest.raises(RuntimeError, match="trainable=False"):
        engine.forward_backward([Datum(input_ids=torch.arange(3))])
    with pytest.raises(RuntimeError, match="trainable=False"):
        engine.optim_step(None)


def test_closed_engine_rejects_calls():
    engine = _unbuilt_engine(closed=True)
    with pytest.raises(RuntimeError, match="closed"):
        engine.forward_only([Datum(input_ids=torch.arange(3))])


def test_shared_runtime_tracks_attached_engines():
    runtime = _SharedRuntime(Topology(1, 1, 1, None, 1, 1, None, False), pg_collection=None)
    a, b = _unbuilt_engine(), _unbuilt_engine()
    runtime.attach(a)
    runtime.attach(b)
    assert runtime.num_engines == 2
    runtime.detach(a)
    runtime.detach(a)
    assert runtime.num_engines == 1
