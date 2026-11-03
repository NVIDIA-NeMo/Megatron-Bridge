from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.engine import AdamParams, Datum, ForwardBackwardOutput, TrainingEngine, topology_from_recipe
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.datum import NormalizationPolicy, PreparedMicrobatch
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
        pad_to_max_length=False,
        mtp_target_scope="primary",
    )


def _unbuilt_engine(trainable: bool = True, closed: bool = False) -> TrainingEngine:
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._trainable = trainable
    engine._closed = closed
    engine._runtime = SimpleNamespace(topology=Topology(1, 1, 1, None, 1, 1, None, False))
    return engine


def _datums() -> list[Datum]:
    return [Datum(input_ids=torch.arange(3), target_ids=torch.arange(1, 4), loss_mask=torch.ones(3))]


def _prepared(mapping) -> PreparedMicrobatch:
    ids = torch.arange(4).unsqueeze(0)
    return PreparedMicrobatch(
        input_ids=ids,
        target_ids=ids,
        position_ids=ids,
        loss_mask=torch.ones(1, 4),
        padding_mask=torch.zeros(1, 4, dtype=torch.bool),
        packed_seq_params=None,
        mapping=mapping,
    )


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
        (lambda r: setattr(r, "mtp_target_scope", "response"), "mtp_target_scope"),
        (
            lambda r: (setattr(r.model, "mtp_detach_heads", True), setattr(r, "mtp_target_scope", "all_valid")),
            "mtp_detach_heads",
        ),
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


def test_detached_mtp_under_primary_scope_is_accepted():
    recipe = _recipe_like()
    recipe.model.mtp_detach_heads = True
    check_supported(recipe)


def test_loss_dispatch_rejects_requires_logits_placeholder():
    class MyEngine(TrainingEngine):
        def logits_loss(self, batch, output):
            return torch.zeros(())

        logits_loss.requires_logits = True

    engine = MyEngine.__new__(MyEngine)
    with pytest.raises(NotImplementedError, match="requires_logits"):
        engine.resolve_loss_method("logits_loss")


@pytest.mark.parametrize(
    "name",
    ["optim_step", "close", "recipe", "preprocess", "postprocess", "_build", "", "does_not_exist", "normalization"],
)
def test_loss_dispatch_rejects_reserved_private_and_unknown_names(name):
    engine = _unbuilt_engine()
    with pytest.raises(ValueError):
        engine.resolve_loss_method(name)


def test_forward_backward_runs_pipeline_and_opens_window(monkeypatch):
    engine = _unbuilt_engine()
    with pytest.raises(ValueError, match="loss method"):
        engine.forward_backward(_datums(), loss_fn="nope")
    calls = []
    engine._recipe = SimpleNamespace(ddp=SimpleNamespace(overlap_grad_reduce=False), mtp_target_scope="primary")
    engine._runtime.pg_collection = None
    engine._model = []
    engine._window = AccumulationWindow()
    engine.preprocess = lambda data, training: calls.append(("preprocess", training)) or ["mb"]
    engine.check_microbatches = lambda microbatches: calls.append(("check", microbatches))
    engine.summarize = lambda loss_sum, tokens: ForwardBackwardOutput(
        loss_sum=float(loss_sum), num_loss_tokens=int(tokens)
    )
    engine.count_samples = lambda microbatches: 2
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_backward",
        lambda model, mbs, loss_method, pg_collection, mtp_target_scope: (torch.tensor(6.0), torch.tensor(3)),
    )
    out = engine.forward_backward(_datums())
    assert out == ForwardBackwardOutput(loss_sum=6.0, num_loss_tokens=3)
    assert calls == [("preprocess", True), ("check", ["mb"])]
    assert engine._window.is_open and engine._window.num_loss_tokens == 3 and engine._window.num_samples == 2


def test_summarize_keeps_large_token_counts_exact(monkeypatch):
    engine = _unbuilt_engine()
    engine._runtime.pg_collection = SimpleNamespace(dp_cp=None)
    reduced = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None: reduced.append(tensor.dtype))
    count = 2**24 + 1
    out = engine.summarize(torch.tensor(1.5), torch.tensor(count, dtype=torch.int64))
    assert out == ForwardBackwardOutput(loss_sum=1.5, num_loss_tokens=count)
    assert reduced == [torch.float64]


def test_default_count_samples_is_unknown_for_foreign_mappings():
    engine = _unbuilt_engine()
    mb = _prepared(mapping=object())
    assert engine.count_samples([mb]) is None


def test_unknown_sample_count_is_rejected_under_sample_mean_before_compute(monkeypatch):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(ddp=SimpleNamespace(overlap_grad_reduce=False), mtp_target_scope="primary")
    engine._runtime.pg_collection = None
    engine._model = []
    engine._window = AccumulationWindow()
    engine.preprocess = lambda data, training: [_prepared(mapping=object())]
    engine.check_microbatches = lambda microbatches: None
    engine.summarize = lambda loss_sum, tokens: ForwardBackwardOutput(
        loss_sum=float(loss_sum), num_loss_tokens=int(tokens)
    )
    ran = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_backward",
        lambda model, mbs, loss_method, pg_collection, mtp_target_scope: (
            ran.append(1) or (torch.tensor(6.0), torch.tensor(3))
        ),
    )
    engine.normalization = NormalizationPolicy.SAMPLE_MEAN
    with pytest.raises(ValueError, match="count_samples returned None"):
        engine.forward_backward(_datums())
    assert ran == [] and not engine._window.is_open
    engine.normalization = NormalizationPolicy.TOKEN_MEAN
    engine.forward_backward(_datums())
    assert ran == [1] and engine._window.num_samples == 0


def test_overlap_grad_reduce_is_an_explicit_placeholder():
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(ddp=SimpleNamespace(overlap_grad_reduce=True))
    with pytest.raises(NotImplementedError, match="overlap_grad_reduce"):
        engine.forward_backward(_datums())


def test_optim_step_requires_open_window():
    engine = _unbuilt_engine()
    engine._window = AccumulationWindow()
    with pytest.raises(RuntimeError, match="without a preceding forward_backward"):
        engine.optim_step(AdamParams(1e-5))


def test_forward_only_runs_preprocess_check_then_postprocess(monkeypatch):
    engine = _unbuilt_engine()
    calls = []
    engine.preprocess = lambda data, training: calls.append(("preprocess", training)) or ["mb"]
    engine.check_microbatches = lambda microbatches: calls.append(("check", microbatches))
    engine.postprocess = lambda microbatches, logprobs, gather_output: calls.append(("post", gather_output)) or "out"
    engine._runtime.pg_collection = None
    engine._model = []
    engine._recipe = SimpleNamespace(mtp_target_scope="primary")
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_only", lambda model, mbs, pg_collection, mtp_target_scope: ["lp"]
    )
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
