from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker, get_moe_metrics_tracker
from megatron.core.transformer.moe.router_replay import RouterReplay

from megatron.bridge.engine import AdamParams, Datum, ForwardBackwardOutput, TrainingEngine, topology_from_recipe
from megatron.bridge.engine.accumulation import AccumulationWindow
from megatron.bridge.engine.batching import MicrobatchMapping, SequenceMapping, local_sample_count
from megatron.bridge.engine.datum import NormalizationPolicy, PreparedMicrobatch
from megatron.bridge.engine.engine import Topology, _build_frozen_model, _SharedRuntime
from megatron.bridge.engine.recipes import check_supported, data_parallel_size
from megatron.bridge.engine.router_replay import RouterReplayState


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
    engine._output_head = "lm"
    engine._resident = True
    engine._offloaded_bytes = 0
    engine._moe_metrics_tracker = MoEMetricsTracker()
    engine._router_replay = RouterReplayState([])
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


def test_router_replay_layers_use_the_recipes_layer_index():
    RouterReplay.clear_global_router_replay_instances()
    try:
        routers = [RouterReplay() for _ in range(3)]
        for router, number in zip(routers, (2, 4, 6)):
            router.layer_number = number
        engine = _unbuilt_engine()
        engine._router_replay = RouterReplayState(routers)
        engine._recipe = SimpleNamespace(router_replay=True, router_replay_layer_index=None)
        assert engine.router_replay_layers == [1, 3, 5]

        engine._recipe.router_replay_layer_index = lambda layer: layer // 2
        assert engine.router_replay_layers == [0, 1, 2]

        engine._recipe.router_replay_layer_index = lambda layer: 0
        with pytest.raises(ValueError, match="router_replay_layer_index"):
            engine.router_replay_layers

        engine._recipe.router_replay = False
        assert engine.router_replay_layers == []
    finally:
        RouterReplay.clear_global_router_replay_instances()


def test_check_supported_accepts_detached_mtp():
    recipe = _recipe_like()
    recipe.model.mtp_detach_heads = True
    recipe.model.mtp_num_layers = 1
    check_supported(recipe)


def test_check_supported_accepts_baseline_shape():
    check_supported(_recipe_like())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: setattr(r.dist, "lazy_mpu_init", True), "lazy_mpu_init"),
        (lambda r: setattr(r.dist, "use_torch_fsdp2", True), "FSDP"),
        (lambda r: setattr(r.ddp, "use_megatron_fsdp", True), "FSDP"),
        (lambda r: setattr(r.model, "gtp_weight_remat_size", 2), "GTP"),
        (lambda r: setattr(r.model, "expert_gtp_weight_remat_size", 2), "GTP"),
        (lambda r: setattr(r.model, "vocab_size", None), "vocab_size"),
        (lambda r: setattr(r.model, "seq_length", None), "seq_length"),
        (lambda r: setattr(r, "batch_layout", "sbhd"), "batch_layout"),
        (lambda r: setattr(r, "mtp_target_scope", "response"), "mtp_target_scope"),
    ],
)
def test_check_supported_rejects_each_violation(mutate, message):
    recipe = _recipe_like()
    mutate(recipe)
    with pytest.raises(ValueError, match=message):
        check_supported(recipe)


def _pipeline_recipe_like() -> SimpleNamespace:
    recipe = _recipe_like()
    recipe.model.pipeline_model_parallel_size = 2
    recipe.pad_to_max_length = True
    return recipe


def test_check_supported_accepts_pipeline_parallelism_with_padded_packs():
    check_supported(_pipeline_recipe_like())


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: setattr(r, "pad_to_max_length", False), "pad_to_max_length"),
        (lambda r: setattr(r.model, "virtual_pipeline_model_parallel_size", 2), "virtual pipeline"),
        (lambda r: setattr(r, "router_replay", True), "router_replay with pipeline"),
    ],
)
def test_check_supported_rejects_unsupported_pipeline_modes(mutate, message):
    recipe = _pipeline_recipe_like()
    recipe.model.num_moe_experts = 256
    mutate(recipe)
    with pytest.raises(ValueError, match=message):
        check_supported(recipe)


def _pipeline_engine(monkeypatch, *, last_stage: bool) -> TrainingEngine:
    engine = _unbuilt_engine()
    stage = SimpleNamespace(rank=lambda: 1 if last_stage else 0, size=lambda: 2)
    engine._runtime = SimpleNamespace(
        topology=Topology(16, 1, 2, None, 4, 8, None, False), pg_collection=SimpleNamespace(pp=stage)
    )
    engine._model = []
    engine._recipe = SimpleNamespace(model=SimpleNamespace(), mtp_target_scope="primary")
    engine.preprocess = lambda data, training: ["mb"]
    engine.check_microbatches = lambda microbatches: None
    engine.postprocess_forward_only = lambda mbs, scores, gather_output: "scores"
    monkeypatch.setattr("megatron.bridge.engine.engine.run_forward_only", lambda model, mbs, **kwargs: [])
    # Stand-in for the collective: the last stage's value arrives everywhere.
    monkeypatch.setattr("megatron.bridge.engine.engine.broadcast_from_last_stage", lambda value, group: "scores")
    return engine


def test_forward_only_with_pipeline_stages_delivers_the_last_stages_scores(monkeypatch):
    last = _pipeline_engine(monkeypatch, last_stage=True)
    first = _pipeline_engine(monkeypatch, last_stage=False)
    # Only the last stage has scores to postprocess; without the broadcast the others return None.
    assert last.forward_only(_datums(), broadcast_output=False) == "scores"
    assert first.forward_only(_datums(), broadcast_output=False) is None
    assert first.forward_only(_datums()) == "scores"


def test_pipeline_parallelism_placeholders(monkeypatch):
    engine = _pipeline_engine(monkeypatch, last_stage=True)
    engine._recipe.ddp = SimpleNamespace(overlap_grad_reduce=False)
    engine._window = AccumulationWindow()
    calls = (
        lambda: engine.forward_backward(_datums()),
        lambda: engine.save_state("/x"),
        lambda: engine.load_state("/x"),
        lambda: engine.export(path="/x"),
        lambda: engine.cpu(),
    )
    for call in calls:
        with pytest.raises(NotImplementedError, match="pipeline parallelism"):
            call()
    recipe = _pipeline_recipe_like()
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        TrainingEngine.from_recipe(recipe, output_head="value")


def test_data_parallel_size_and_topology(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "16")
    recipe = _recipe_like()
    assert data_parallel_size(recipe) == 2
    assert topology_from_recipe(recipe) == Topology(16, 1, 1, None, 8, 16, 1, False)
    recipe.model.context_parallel_size = 4
    assert topology_from_recipe(recipe) != topology_from_recipe(_recipe_like())
    recipe.model.context_parallel_size = 3
    with pytest.raises(ValueError, match="not a multiple of the recipe's minimum GPU count 48"):
        data_parallel_size(recipe)


def test_constructor_by_name_resolves_applies_overrides_then_initializes(monkeypatch):
    import megatron.bridge.engine.engine as engine_module

    resolved = SimpleNamespace(optimizer=SimpleNamespace(lr=2e-5), router_replay=False)
    seen = {}
    monkeypatch.setattr(
        engine_module,
        "resolve_recipe",
        lambda model, **kwargs: seen.update(model=model, **kwargs) or resolved,
    )
    monkeypatch.setattr(
        TrainingEngine, "_init_from_recipe", lambda self, recipe, **kwargs: seen.update(recipe=recipe, init=kwargs)
    )

    TrainingEngine(
        "/models/qwen35",
        seq_length=32768,
        checkpoint_path="/ckpt",
        hardware="GB300",
        overrides={"optimizer.lr": 1e-6, "router_replay": True},
        trainable=False,
    )

    assert seen["model"] == "/models/qwen35"
    assert (seen["seq_length"], seen["hardware"], seen["hf_revision"]) == (32768, "GB300", None)
    assert seen["recipe"] is resolved
    assert (resolved.optimizer.lr, resolved.router_replay) == (1e-6, True)
    assert seen["init"]["checkpoint_path"] == "/ckpt"
    assert seen["init"]["trainable"] is False
    assert seen["init"]["output_head"] == "lm"


def test_loss_dispatch_resolves_predefined_and_subclass_methods():
    class MyEngine(TrainingEngine):
        def my_loss(self, batch, output):
            return torch.zeros(())

    engine = MyEngine.__new__(MyEngine)
    assert engine.resolve_loss_method("cross_entropy").__func__ is TrainingEngine.cross_entropy
    assert engine.resolve_loss_method("ppo_clip").__func__ is TrainingEngine.ppo_clip
    assert engine.resolve_loss_method("my_loss")(None, None).item() == 0.0


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
    engine._recipe = SimpleNamespace(
        model=SimpleNamespace(), ddp=SimpleNamespace(overlap_grad_reduce=False), mtp_target_scope="primary"
    )
    engine._runtime.pg_collection = None
    engine._model = []
    engine._window = AccumulationWindow()
    engine.preprocess = lambda data, training: calls.append(("preprocess", training)) or ["mb"]
    engine.check_microbatches = lambda microbatches: calls.append(("check", microbatches))
    engine.postprocess_forward_backward = lambda mbs, loss_sum, tokens, metrics: ForwardBackwardOutput(
        loss_sum=float(loss_sum),
        num_loss_tokens=int(tokens),
        metrics={k: float(v) for k, v in metrics.items()},
        num_samples=2,
    )
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_backward",
        lambda model, mbs, loss_method, pg_collection, mrope, mtp_target_scope, output_head, router_replay: (
            calls.append(("run", mrope))
            or (
                torch.tensor(6.0),
                torch.tensor(3),
                {"clipped_tokens": torch.tensor(1.0)},
            )
        ),
    )
    out = engine.forward_backward(_datums())
    assert out == ForwardBackwardOutput(
        loss_sum=6.0, num_loss_tokens=3, metrics={"clipped_tokens": 1.0}, num_samples=2
    )
    assert calls == [("preprocess", True), ("check", ["mb"]), ("run", False)]
    assert engine._window.is_open and engine._window.num_loss_tokens == 3 and engine._window.num_samples == 2


def test_postprocess_forward_backward_keeps_large_token_counts_exact(monkeypatch):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(model=SimpleNamespace(), mtp_target_scope="primary")
    engine._runtime.pg_collection = SimpleNamespace(dp_cp=None, dp=None)
    reduced = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, group=None: reduced.append(tensor.dtype))
    foreign = [_prepared(mapping=object())]
    count = 2**24 + 1
    out = engine.postprocess_forward_backward(foreign, torch.tensor(1.5), torch.tensor(count, dtype=torch.int64), {})
    assert out == ForwardBackwardOutput(loss_sum=1.5, num_loss_tokens=count)  # counts unknown -> None
    assert reduced == [torch.float64]
    metrics = {"b": torch.tensor(2.0), "a": torch.tensor(1.0)}
    out = engine.postprocess_forward_backward(foreign, torch.tensor(1.5), torch.tensor(4), metrics)
    assert out.metrics == {"a": 1.0, "b": 2.0} and out.num_loss_tokens == 4
    assert out.num_samples is None
    assert reduced == [torch.float64, torch.float64]


def test_local_sample_count_reads_the_default_mapping_only():
    assert local_sample_count([_prepared(mapping=object())]) is None
    two = MicrobatchMapping(
        segments=[SequenceMapping(index=0, offset=0, length=2), SequenceMapping(index=1, offset=2, length=2)],
        total_tokens=4,
        cp_indices=torch.arange(4),
    )
    filler = MicrobatchMapping(segments=[], total_tokens=0, cp_indices=torch.tensor([]), filler=True)
    assert local_sample_count([_prepared(mapping=two), _prepared(mapping=filler)]) == 2


def test_unknown_sample_count_under_sample_mean_discards_the_window(monkeypatch):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(
        model=SimpleNamespace(), ddp=SimpleNamespace(overlap_grad_reduce=False), mtp_target_scope="primary"
    )
    engine._runtime.pg_collection = None
    engine._model = []
    engine._optimizer = None
    engine._window = AccumulationWindow()
    engine.preprocess = lambda data, training: [_prepared(mapping=object())]
    engine.check_microbatches = lambda microbatches: None
    engine.postprocess_forward_backward = lambda mbs, loss_sum, tokens, metrics: ForwardBackwardOutput(
        loss_sum=float(loss_sum), num_loss_tokens=int(tokens)
    )
    ran = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_backward",
        lambda model, mbs, loss_method, pg_collection, mrope, mtp_target_scope, output_head, router_replay: (
            ran.append(1) or (torch.tensor(6.0), torch.tensor(3), {})
        ),
    )
    engine.normalization = NormalizationPolicy.SAMPLE_MEAN
    with pytest.raises(ValueError, match="needs num_samples"):
        engine.forward_backward(_datums())
    assert ran == [1] and not engine._window.is_open  # the call ran; its window was discarded
    engine.normalization = NormalizationPolicy.TOKEN_MEAN
    engine.forward_backward(_datums())
    assert ran == [1, 1] and engine._window.num_samples == 0


def test_overlap_grad_reduce_is_an_explicit_placeholder():
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(ddp=SimpleNamespace(overlap_grad_reduce=True))
    with pytest.raises(NotImplementedError, match="overlap_grad_reduce"):
        engine.forward_backward(_datums())


def test_unnormalized_step_with_aux_losses_warns_once_and_proceeds(monkeypatch, caplog):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(model=SimpleNamespace(mtp_num_layers=1, mtp_detach_heads=False))
    engine._model = []
    engine._optimizer = None
    engine._window = AccumulationWindow()
    engine._warned_unscaled_aux = False
    engine._finalize_model_grads_func = lambda model, num_tokens: None
    engine._step = 0
    engine.normalization = NormalizationPolicy.NONE
    scales = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.finalize_and_step",
        lambda model, optimizer, *, finalize_model_grads_func, scale, adam_params: (
            scales.append(scale) or (True, 1.0, 0)
        ),
    )
    monkeypatch.setattr("megatron.bridge.engine.engine.zero_gradients", lambda model, optimizer: None)
    for _ in range(2):
        engine._window.add(num_loss_tokens=10, num_samples=1, loss_sum=1.0)
        with caplog.at_level("WARNING", logger="megatron.bridge.engine.engine"):
            engine.optim_step(AdamParams(1e-5))
    assert scales == [None, None]  # nothing normalized by the engine
    assert sum("mtp_grad_scale_func" in record.message for record in caplog.records) == 1


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
    engine.postprocess_forward_only = lambda mbs, logprobs, gather_output: (
        calls.append(("post", gather_output)) or "out"
    )
    engine._runtime.pg_collection = None
    engine._model = []
    # A builder-backed config carries position_embedding_type itself; the built model's config does not.
    engine._recipe = SimpleNamespace(
        model=SimpleNamespace(position_embedding_type="mrope"), mtp_target_scope="primary"
    )
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_only",
        lambda model, mbs, pg_collection, mrope, mtp_target_scope, output_head, router_replay: (
            calls.append(("run", mrope)) or [output_head]
        ),
    )
    assert engine.forward_only(_datums(), gather_output=False) == "out"
    assert calls == [("preprocess", False), ("check", ["mb"]), ("run", True), ("post", False)]
    # Each compute call installs the engine's own MoE metrics tracker (per-layer buffers differ per model).
    assert get_moe_metrics_tracker() is engine._moe_metrics_tracker


def test_check_microbatches_passes_the_replayable_shape(monkeypatch):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(
        batch_layout="thd", model=SimpleNamespace(vocab_size=100, moe_router_topk=8, num_moe_experts=256)
    )
    engine._runtime.pg_collection = SimpleNamespace(dp=None)
    seen = {}
    monkeypatch.setattr("megatron.bridge.engine.engine.check_microbatches", lambda mbs, **kwargs: seen.update(kwargs))
    # Recipe without router_replay: routing cannot be replayed, whatever routers exist.
    engine._recipe.router_replay = False
    engine.check_microbatches(["mb"])
    assert seen["routed_shape"] is None and seen["num_experts"] == 256
    # With the knob on, the shape is (replayable routers, topk).
    engine._recipe.router_replay = True
    engine._router_replay = SimpleNamespace(num_layers=40)
    engine.check_microbatches(["mb"])
    assert seen["routed_shape"] == (40, 8)
    with pytest.raises(ValueError, match="no microbatches"):
        engine.check_microbatches([])


def test_value_head_engine_defaults_to_value_loss_and_rejects_export(monkeypatch):
    engine = _unbuilt_engine()
    engine._output_head = "value"
    engine._recipe = SimpleNamespace(
        model=SimpleNamespace(), ddp=SimpleNamespace(overlap_grad_reduce=False), mtp_target_scope="primary"
    )
    engine._runtime.pg_collection = None
    engine._model = []
    engine._window = AccumulationWindow()
    engine.preprocess = lambda data, training: ["mb"]
    engine.check_microbatches = lambda microbatches: None
    engine.postprocess_forward_backward = lambda mbs, loss_sum, tokens, metrics: ForwardBackwardOutput(
        float(loss_sum), int(tokens), num_samples=1
    )
    seen = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_backward",
        lambda model, mbs, loss_method, pg_collection, mrope, mtp_target_scope, output_head, router_replay: (
            seen.append((loss_method.__func__, output_head)) or (torch.tensor(1.0), torch.tensor(2), {})
        ),
    )
    engine.forward_backward(_datums())
    assert seen == [(TrainingEngine.value_loss, "value")]
    with pytest.raises(NotImplementedError, match="value-head"):
        engine.export(path="/x")


def test_from_recipe_validates_output_head():
    with pytest.raises(ValueError, match="output_head"):
        TrainingEngine.from_recipe(_recipe_like(), output_head="reward")


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
    with pytest.raises(RuntimeError, match="closed"):
        engine.cpu()


def test_offloaded_engine_rejects_compute_and_state_calls(monkeypatch):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(ddp=SimpleNamespace(), optimizer=SimpleNamespace())
    engine._optimizer = None
    engine._model = []
    engine._window = AccumulationWindow()
    engine._pending_save = None
    engine._host_state = None
    moves = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.offload_to_cpu", lambda *a, **k: moves.append(("cpu", k)) or 1234
    )
    monkeypatch.setattr("megatron.bridge.engine.engine.reload_to_cuda", lambda *a, **k: moves.append(("cuda", k)) or 0)
    assert engine.device == "cuda" and engine.offloaded_bytes == 0
    assert engine.cpu() is engine and engine.device == "cpu"
    assert engine.offloaded_bytes == 1234
    engine.cpu()  # idempotent: no second move
    assert moves == [("cpu", {"keep_gradients": False})]
    for call in (
        lambda: engine.forward_only(_datums()),
        lambda: engine.forward_backward(_datums()),
        lambda: engine.optim_step(None),
        lambda: engine.zero_grad(),
        lambda: engine.save_state("x"),
        lambda: engine.load_state("x"),
        lambda: engine.export(),
    ):
        with pytest.raises(RuntimeError, match="cuda\\(\\) first"):
            call()
    assert engine.cuda() is engine and engine.device == "cuda" and engine.offloaded_bytes == 0
    engine.cuda()
    assert moves == [("cpu", {"keep_gradients": False}), ("cuda", {})]


def test_cpu_rejects_unsupported_modes_before_moving(monkeypatch):
    engine = _unbuilt_engine()
    engine._recipe = SimpleNamespace(ddp=SimpleNamespace(overlap_param_gather=True), optimizer=SimpleNamespace())
    engine._optimizer = None
    monkeypatch.setattr("megatron.bridge.engine.engine.offload_to_cpu", lambda *a, **k: pytest.fail("moved"))
    with pytest.raises(NotImplementedError, match="overlap_param_gather"):
        engine.cpu()
    assert engine.device == "cuda"


def test_shared_runtime_tracks_attached_engines():
    runtime = _SharedRuntime(Topology(1, 1, 1, None, 1, 1, None, False), pg_collection=None)
    a, b = _unbuilt_engine(), _unbuilt_engine()
    runtime.attach(a)
    runtime.attach(b)
    assert runtime.num_engines == 2
    runtime.detach(a)
    runtime.detach(a)
    assert runtime.num_engines == 1


def test_frozen_engine_builds_through_bridge_without_ddp(monkeypatch):
    calls = []

    def build(cfg, pg_collection, *, wrap_with_ddp):
        calls.append((cfg, pg_collection, wrap_with_ddp))
        return [torch.nn.Linear(2, 2)]

    monkeypatch.setattr("megatron.bridge.engine.engine._build_distributed_model", build)
    recipe = SimpleNamespace(model=object())
    model = _build_frozen_model(recipe, "pg")
    assert calls == [(recipe, "pg", False)]
    assert not any(parameter.requires_grad for parameter in model[0].parameters())
