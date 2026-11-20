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

import os
import re
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
import yaml
from megatron.core.transformer.moe.moe_logging import MoEMetricsTracker

from megatron.bridge.engine import LoraConfig, TrainingEngine
from megatron.bridge.engine.checkpoint import _checkpoint_state, load_engine_state, save_engine_state
from megatron.bridge.engine.engine import Topology
from megatron.bridge.engine.export import ExportState, export_hf_adapter
from megatron.bridge.engine.lora import build_lora
from megatron.bridge.engine.recipes.base import BRIDGE_LORA_TARGET_MODULES, TrainingRecipe
from megatron.bridge.engine.router_replay import RouterReplayState
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.utils import create_peft_hook
from megatron.bridge.training.state import TrainState


pytestmark = pytest.mark.unit

QWEN35_TARGETS = [*BRIDGE_LORA_TARGET_MODULES, "in_proj", "out_proj"]


def _recipe(targets=None, **model) -> SimpleNamespace:
    return SimpleNamespace(
        lora_target_modules=list(BRIDGE_LORA_TARGET_MODULES if targets is None else targets),
        model=SimpleNamespace(**model),
    )


# ---- lowering ---------------------------------------------------------------------------------


def test_build_lora_lowers_rank_alpha_dropout_and_takes_the_recipe_targets():
    peft = build_lora(LoraConfig(rank=32, alpha=64.0), _recipe(QWEN35_TARGETS))
    assert isinstance(peft, LoRA)
    assert (peft.dim, peft.alpha, peft.dropout) == (32, 64.0, 0.0)
    assert peft.target_modules == QWEN35_TARGETS
    # Bridge's default: one adapter shared by the local experts of a grouped expert linear, whose gradient
    # the expert-parallel sync installed with peft_enabled=True sums.
    assert peft.share_expert_adapters and not peft.experts_shared_outer_loras
    # On in the engine (off in Bridge's LoRA): routing stays the base model's, frozen parts get no adapters.
    assert peft.freeze_routing and peft.skip_frozen_modules


def test_build_lora_passes_the_routing_and_frozen_module_switches_to_bridge():
    peft = build_lora(LoraConfig(rank=8, alpha=16.0, freeze_routing=False, skip_frozen_modules=False), _recipe())
    assert not peft.freeze_routing and not peft.skip_frozen_modules


@pytest.mark.parametrize(
    "layout, share_expert_adapters, experts_shared_outer_loras",
    [("shared", True, False), ("per_expert", False, False), ("outer_shared", True, True)],
)
def test_build_lora_lowers_each_expert_adapter_layout_to_bridges_flags(
    layout, share_expert_adapters, experts_shared_outer_loras
):
    peft = build_lora(LoraConfig(rank=8, alpha=16.0, expert_adapters=layout), _recipe())
    assert (peft.share_expert_adapters, peft.experts_shared_outer_loras) == (
        share_expert_adapters,
        experts_shared_outer_loras,
    )


def test_build_lora_prefers_explicit_targets_and_copies_them():
    targets = ["linear_qkv"]
    peft = build_lora(LoraConfig(rank=8, alpha=16.0, target_modules=targets, dropout=0.1), _recipe(QWEN35_TARGETS))
    assert peft.target_modules == ["linear_qkv"] and peft.target_modules is not targets
    assert peft.dropout == 0.1


@pytest.mark.parametrize(
    "config, message",
    [
        (LoraConfig(rank=0, alpha=1.0), "rank must be a positive integer"),
        (LoraConfig(rank=8, alpha=0.0), "alpha must be positive"),
        (LoraConfig(rank=8, alpha=1.0, dropout=1.0), "dropout must be in"),
        (LoraConfig(rank=8, alpha=1.0, target_modules=[]), "target_modules is empty"),
        (LoraConfig(rank=8, alpha=1.0, expert_adapters="outer"), "expert_adapters must be one of"),
    ],
)
def test_build_lora_rejects_invalid_settings(config, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        build_lora(config, _recipe())


def test_training_recipe_defaults_to_bridge_lora_targets_per_instance():
    first = TrainingRecipe(model=None, optimizer=None, ddp=None, mixed_precision="bf16_mixed")
    second = TrainingRecipe(model=None, optimizer=None, ddp=None, mixed_precision="bf16_mixed")
    assert first.lora_target_modules == ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
    first.lora_target_modules.append("in_proj")
    assert second.lora_target_modules == list(BRIDGE_LORA_TARGET_MODULES)


# ---- the pre-wrap hook and Bridge's LoRA on real (CPU) modules -----------------------------------


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = torch.nn.Linear(4, 6, bias=False)
        self.linear_proj = torch.nn.Linear(6, 4, bias=False)
        self.router = torch.nn.Linear(4, 2, bias=False)

    def forward(self, x):
        return self.linear_proj(self.linear_qkv(x)), self.router(x)


def test_bridge_lora_freezes_the_base_and_disable_adapter_scores_it():
    torch.manual_seed(0)
    peft = build_lora(LoraConfig(rank=2, alpha=4.0), _recipe())
    model = [_Block()]
    x = torch.randn(3, 4)
    base_out, base_router = model[0](x)
    wrapped = create_peft_hook(peft)(model)
    trainable = sorted(name for name, p in wrapped[0].named_parameters() if p.requires_grad)
    assert peft.params_to_save == set(trainable)  # what the adapter-only checkpoint view keeps
    assert trainable == [
        "linear_proj.adapter.linear_in.weight",
        "linear_proj.adapter.linear_out.weight",
        "linear_qkv.adapter.linear_in.weight",
        "linear_qkv.adapter.linear_out.weight",
    ]
    # lora_B starts at zero, so the adapted model starts as the base model.
    assert torch.equal(wrapped[0](x)[0], base_out)
    for name, parameter in wrapped[0].named_parameters():
        if name.endswith("adapter.linear_out.weight"):
            parameter.data.fill_(0.1)
    adapted, router_out = wrapped[0](x)
    with peft.disable_adapter(wrapped):
        disabled, _ = wrapped[0](x)
    again, _ = wrapped[0](x)
    assert not torch.allclose(adapted, base_out)
    assert torch.equal(disabled, base_out) and torch.equal(again, adapted)
    assert torch.equal(router_out, base_router)  # not a target


# ---- vision-language targets match the way Bridge's LoRA matches them ----------------------------

# Where Bridge's Qwen3-VL model keeps its LoRA-able linears (bare model, as the pre-wrap hook sees it).
_LANGUAGE = (
    "language_model.decoder.layers.0.self_attention.linear_qkv",
    "language_model.decoder.layers.0.self_attention.linear_proj",
    "language_model.decoder.layers.0.mlp.linear_fc1",
    "language_model.decoder.layers.0.mlp.linear_fc2",
    "language_model.decoder.layers.1.self_attention.in_proj",
    "language_model.decoder.layers.1.self_attention.out_proj",
)
_VISION = (
    "vision_model.decoder.layers.0.self_attention.linear_qkv",
    "vision_model.decoder.layers.0.self_attention.linear_proj",
    "vision_model.decoder.layers.0.mlp.linear_fc1",
    "vision_model.decoder.layers.0.mlp.linear_fc2",
)
_PROJECTION = (
    "vision_model.merger.linear_fc1",
    "vision_model.merger.linear_fc2",
    "vision_model.decoder.deepstack_merger_list.0.linear_fc1",
    "vision_model.decoder.deepstack_merger_list.0.linear_fc2",
)


def _vision_language_model() -> torch.nn.Module:
    root = torch.nn.Module()
    for path in (*_LANGUAGE, *_VISION, *_PROJECTION):
        parent = root
        *parents, leaf = path.split(".")
        for part in parents:
            if part not in parent._modules:
                parent.add_module(part, torch.nn.Module())
            parent = parent._modules[part]
        parent.add_module(leaf, torch.nn.Linear(4, 4, bias=False))
    return root


# The parts Bridge's Qwen3-VL model freezes for each recipe flag (``Qwen3VLModel.freeze``), which the
# Qwen3.5-VL builder calls when it builds the model, before the pre-wrap hooks run.
_FROZEN_BY_FLAG = {
    "freeze_language_model": ("language_model.",),
    "freeze_vision_model": ("vision_model.decoder.layers.",),
    "freeze_vision_projection": ("vision_model.merger.", "vision_model.decoder.deepstack_merger_list."),
}


def _wrapped_with_lora(*, targets=None, skip_frozen_modules=True, **freeze) -> list[str]:
    """Freeze the toy vision-language model as the builder does, apply the engine's LoRA, and return the
    linears LoRA wrapped."""
    recipe = _recipe(QWEN35_TARGETS, vision_config=object(), **freeze)
    model = _vision_language_model()
    for flag, prefixes in _FROZEN_BY_FLAG.items():
        for name, parameter in model.named_parameters():
            if freeze.get(flag) and name.startswith(prefixes):
                parameter.requires_grad = False
    config = LoraConfig(rank=2, alpha=4.0, target_modules=targets, skip_frozen_modules=skip_frozen_modules)
    wrapped = create_peft_hook(build_lora(config, recipe))([model])
    return sorted(name for name, module in wrapped[0].named_modules() if type(module).__name__ == "LoRALinear")


def test_vision_language_lora_keeps_the_adapters_off_the_parts_the_recipe_freezes():
    frozen_vision = {"freeze_vision_model": True, "freeze_vision_projection": True}
    assert _wrapped_with_lora(**frozen_vision) == sorted(_LANGUAGE)
    assert _wrapped_with_lora(freeze_language_model=True) == sorted((*_VISION, *_PROJECTION))
    assert _wrapped_with_lora() == sorted((*_LANGUAGE, *_VISION, *_PROJECTION))  # nothing frozen
    with pytest.raises(ValueError, match="nothing would be trained"):
        _wrapped_with_lora(freeze_language_model=True, **frozen_vision)


def test_vision_language_lora_without_skipping_matches_plain_names_everywhere_and_patterns_as_given():
    frozen_vision = {"freeze_vision_model": True, "freeze_vision_projection": True}
    # As in Bridge's default LoRA, the freeze flags then do not steer the adapters: plain names reach every
    # part ...
    everywhere = _wrapped_with_lora(skip_frozen_modules=False, **frozen_vision)
    assert everywhere == sorted((*_LANGUAGE, *_VISION, *_PROJECTION))
    # ... and patterns keep them on the language model.
    language_only = [f"language_model.*.{name}" for name in QWEN35_TARGETS]
    assert _wrapped_with_lora(targets=language_only, skip_frozen_modules=False, **frozen_vision) == sorted(_LANGUAGE)


# ---- engine construction ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"output_head": "value", "checkpoint_path": "/base"}, NotImplementedError, "(a LoRA critic)"),
        ({"trainable": False, "checkpoint_path": "/base"}, ValueError, "lora needs trainable=True"),
        ({}, ValueError, "lora needs checkpoint_path"),
    ],
)
def test_lora_engine_validates_its_options_before_anything_is_built(monkeypatch, kwargs, error, message):
    monkeypatch.setattr("megatron.bridge.engine.engine.finalize_recipe", lambda recipe: pytest.fail("finalized"))
    with pytest.raises(error, match=re.escape(message)):
        TrainingEngine.from_recipe(
            _recipe(), lora=LoraConfig(rank=8, alpha=16.0), spawn_async_save_worker=False, **kwargs
        )


def test_lora_engine_appends_its_hooks_last_and_hands_the_lora_to_build(monkeypatch):
    built, loads = {}, []
    monkeypatch.setattr("megatron.bridge.engine.engine.finalize_recipe", lambda recipe: None)
    monkeypatch.setattr("megatron.bridge.engine.engine._acquire_runtime", lambda recipe: "runtime")
    monkeypatch.setattr("megatron.bridge.engine.engine.load_pretrained_weights", lambda *args: loads.append(args[1:]))
    monkeypatch.setattr(
        TrainingEngine, "_build", lambda self, recipe, runtime, **kwargs: built.update(kwargs, runtime=runtime)
    )

    def user_hook(model):
        return model

    TrainingEngine.from_recipe(
        _recipe(QWEN35_TARGETS),
        checkpoint_path="/base",
        lora=LoraConfig(rank=8, alpha=16.0),
        hooks=[user_hook],
        spawn_async_save_worker=False,
    )
    assert isinstance(built["peft"], LoRA) and built["peft"].dim == 8
    assert built["peft"].target_modules == QWEN35_TARGETS
    assert built["hooks"][0] is user_hook and len(built["hooks"]) == 3
    # The base loads into the bare model (no optimizer yet), then Bridge's PEFT hook applies LoRA.
    assert built["hooks"][1](["bare"]) == ["bare"] and loads == [(["bare"], None, "/base")]
    assert built["hooks"][2].__qualname__ == "create_peft_hook.<locals>.hook"  # Bridge's hook, nothing else
    assert built["checkpoint_path"] == "/base" and built["runtime"] == "runtime"


def _build_engine(monkeypatch, *, peft, checkpoint_path="/base"):
    """Run TrainingEngine._build with every Megatron-touching collaborator replaced; return (engine, calls)."""
    calls = {"registered": [], "loads": [], "config_funcs": []}
    prefix = "megatron.bridge.engine.engine."
    monkeypatch.setattr(prefix + "_register_pre_wrap_hook", lambda model_cfg, hook: calls["registered"].append(hook))
    monkeypatch.setattr(prefix + "_build_distributed_model", lambda recipe, pg_collection: ["chunk"])
    monkeypatch.setattr(prefix + "get_megatron_optimizer", lambda **kwargs: "optimizer")
    monkeypatch.setattr(prefix + "get_standard_config_overrides", lambda config: None)
    monkeypatch.setattr(prefix + "load_pretrained_weights", lambda *args: calls["loads"].append(args[1:]))
    monkeypatch.setattr(prefix + "get_model_config", lambda chunk: SimpleNamespace(finalize_model_grads_func="fin"))
    monkeypatch.setattr(
        prefix + "_update_model_config_funcs", lambda *args, **kwargs: calls["config_funcs"].append(kwargs)
    )
    monkeypatch.setattr(RouterReplayState, "mark", staticmethod(lambda: 0))
    monkeypatch.setattr(
        RouterReplayState, "claim", classmethod(lambda cls, mark, sequence_parallel=False: RouterReplayState([]))
    )
    recipe = SimpleNamespace(
        model=SimpleNamespace(sequence_parallel=False),
        optimizer=SimpleNamespace(),
        ddp=SimpleNamespace(),
        dist=SimpleNamespace(align_grad_reduce=True, use_gloo_process_groups=False, use_decentralized_pg=False),
    )
    runtime = SimpleNamespace(
        pg_collection="pgs", topology=Topology(16, 1, 1, None, 8, 16, 1, False), attach=lambda engine: None
    )
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._build(
        recipe,
        runtime,
        checkpoint_path=checkpoint_path,
        output_head="lm",
        trainable=True,
        peft=peft,
        hooks=["hook"],
        spawn_async_save_worker=False,
    )
    return engine, calls


def test_build_with_lora_leaves_the_load_to_the_hooks_and_installs_the_expert_adapter_sync(monkeypatch):
    peft = object()
    engine, calls = _build_engine(monkeypatch, peft=peft)
    assert calls["registered"] == ["hook"]
    assert calls["loads"] == []  # the LoRA hooks loaded the base into the bare model already
    assert [kwargs["peft_enabled"] for kwargs in calls["config_funcs"]] == [True]
    assert engine._peft is peft and engine._finalize_model_grads_func == "fin"


def test_build_without_lora_loads_after_the_build_as_before(monkeypatch):
    engine, calls = _build_engine(monkeypatch, peft=None)
    assert calls["loads"] == [(["chunk"], "optimizer", "/base")]
    assert [kwargs["peft_enabled"] for kwargs in calls["config_funcs"]] == [False]
    assert engine._peft is None


# ---- disable_adapters ---------------------------------------------------------------------------


class _FakePeft:
    def __init__(self):
        self.enabled = True

    @contextmanager
    def disable_adapter(self, model):
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True


def _scoring_engine(monkeypatch, peft) -> tuple[TrainingEngine, list[bool]]:
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._closed = False
    engine._resident = True
    engine._output_head = "lm"
    engine._peft = peft
    engine._model = []
    engine._moe_metrics_tracker = MoEMetricsTracker()
    engine._router_replay = RouterReplayState([])
    engine._recipe = SimpleNamespace(model=SimpleNamespace(), mtp_target_scope="primary")
    engine._runtime = SimpleNamespace(topology=Topology(1, 1, 1, None, 1, 1, None, False), pg_collection=None)
    engine.preprocess = lambda data, training: ["mb"]
    engine.check_microbatches = lambda microbatches: None
    engine.postprocess_forward_only = lambda mbs, scores, gather_output: scores
    adapter_states: list[bool] = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.run_forward_only",
        lambda model, mbs, **kwargs: adapter_states.append(peft.enabled) or ["scores"],
    )
    return engine, adapter_states


def test_disable_adapters_turns_the_adapters_off_for_that_call_only(monkeypatch):
    peft = _FakePeft()
    engine, adapter_states = _scoring_engine(monkeypatch, peft)
    assert engine.forward_only(["datum"], disable_adapters=True) == ["scores"]
    assert engine.forward_only(["datum"]) == ["scores"]
    assert adapter_states == [False, True] and peft.enabled


def test_disable_adapters_on_an_engine_without_lora_raises():
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._closed = False
    engine._resident = True
    engine._peft = None
    engine._moe_metrics_tracker = MoEMetricsTracker()
    with pytest.raises(ValueError, match="needs an engine built with lora"):
        engine.forward_only([], disable_adapters=True)


# ---- checkpoints --------------------------------------------------------------------------------


def _checkpoint_recipe() -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(), optimizer=None, ddp=None, dist=None, rng=None)


def test_checkpoint_view_carries_the_lora_only_when_given():
    assert _checkpoint_state(_checkpoint_recipe(), SimpleNamespace()).cfg.peft is None
    peft = object()
    assert _checkpoint_state(_checkpoint_recipe(), SimpleNamespace(), peft=peft).cfg.peft is peft


class _FakeCheckpointManager:
    """Stands in for Bridge's checkpoint manager: a save writes what the engine's load checks read back
    (the step's ``run_config.yaml`` through the config view, then the root's tracker), and a load returns the
    step the tracker names."""

    def __init__(self, seen: list):
        self.seen = seen
        self.checkpointing_context = {}

    def save(self, context, _):
        cfg, step = context.state.cfg, context.state.train_state.step
        step_dir = os.path.join(cfg.checkpoint.save, f"iter_{step:07d}")
        os.makedirs(step_dir, exist_ok=True)
        cfg.to_yaml(os.path.join(step_dir, "run_config.yaml"))
        torch.save(TrainState(step=step).state_dict(), os.path.join(cfg.checkpoint.save, "latest_train_state.pt"))
        self.seen.append(("save", cfg.peft, cfg.checkpoint.save))

    def load(self, context):
        checkpoint = context.state.cfg.checkpoint
        self.seen.append(("load", context.state.cfg.peft, checkpoint.load, checkpoint.pretrained_checkpoint))
        state = torch.load(os.path.join(checkpoint.load, "latest_train_state.pt"), weights_only=True)
        return int(state["step"]), None


@pytest.fixture
def checkpoint_calls(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(
        "megatron.bridge.engine.checkpoint.create_checkpoint_manager", lambda config: _FakeCheckpointManager(seen)
    )
    monkeypatch.setattr(
        "megatron.bridge.engine.checkpoint.sync_hybrid_device_optimizer_fp32_master_copies", lambda optimizer: None
    )
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    return seen


def _save(root: str, step: int, peft: LoRA | None, user_state=None) -> None:
    save_engine_state(
        _checkpoint_recipe(),
        [],
        None,
        step=step,
        root=root,
        user_state=user_state,
        async_save=False,
        async_state=None,
        pg_collection=None,
        peft=peft,
    )


def test_save_and_load_hand_the_lora_to_bridges_checkpointing(checkpoint_calls, tmp_path):
    peft = build_lora(LoraConfig(rank=8, alpha=16.0), _recipe())
    root = str(tmp_path)
    _save(root, 5, peft, user_state={"round": 2})
    step, user_state = load_engine_state(_checkpoint_recipe(), [], None, path=root, peft=peft)
    # Bridge's PEFT resume (adapter filter, non-strict load) applies when load is set, pretrained is not.
    assert checkpoint_calls == [("save", peft, root), ("load", peft, root, None)]
    assert step == 5 and user_state == {"round": 2}


def test_checkpoints_record_the_lora_settings(checkpoint_calls, tmp_path):
    peft = build_lora(LoraConfig(rank=8, alpha=16, target_modules=["linear_qkv", "*.in_proj"], dropout=0.1), _recipe())
    _save(str(tmp_path / "lora"), 3, peft)
    _save(str(tmp_path / "full"), 3, None)
    with open(tmp_path / "lora" / "iter_0000003" / "run_config.yaml") as handle:
        assert yaml.safe_load(handle)["lora"] == {
            "rank": 8,
            "alpha": 16.0,
            "dropout": 0.1,
            "target_modules": ["linear_qkv", "*.in_proj"],
            "expert_adapters": "shared",
            "freeze_routing": True,
            "skip_frozen_modules": True,
        }
    with open(tmp_path / "full" / "iter_0000003" / "run_config.yaml") as handle:
        assert yaml.safe_load(handle)["lora"] is None


@pytest.mark.parametrize(
    "saved, loading",
    [
        (LoraConfig(rank=8, alpha=16.0), LoraConfig(rank=8, alpha=32.0)),  # same tensors, other scaling
        (LoraConfig(rank=8, alpha=16.0, dropout=0.1), LoraConfig(rank=8, alpha=16.0)),
        # Fewer targets: Bridge's non-strict resume would drop the saved in_proj adapters without a word.
        (LoraConfig(rank=8, alpha=16.0, target_modules=["linear_qkv", "in_proj"]), LoraConfig(rank=8, alpha=16.0)),
        (LoraConfig(rank=8, alpha=16.0), LoraConfig(rank=8, alpha=16.0, expert_adapters="per_expert")),
        (LoraConfig(rank=8, alpha=16.0), LoraConfig(rank=8, alpha=16.0, freeze_routing=False)),
        (LoraConfig(rank=8, alpha=16.0, skip_frozen_modules=False), LoraConfig(rank=8, alpha=16.0)),
        (LoraConfig(rank=8, alpha=16.0), None),
        (None, LoraConfig(rank=8, alpha=16.0)),
    ],
)
def test_load_refuses_a_checkpoint_saved_with_other_lora_settings(checkpoint_calls, tmp_path, saved, loading):
    def lora(config):
        return None if config is None else build_lora(config, _recipe(["linear_qkv"]))

    root = str(tmp_path)
    _save(root, 4, lora(saved))
    for path in (root, os.path.join(root, "iter_0000004")):
        with pytest.raises(ValueError, match="iter_0000004 was saved with lora="):
            load_engine_state(_checkpoint_recipe(), [], None, path=path, peft=lora(loading))
    assert [call[0] for call in checkpoint_calls] == ["save"]  # refused before anything was read


def test_load_after_a_newer_save_to_the_same_root_takes_the_newer_step(checkpoint_calls, tmp_path):
    root = str(tmp_path)
    first = build_lora(LoraConfig(rank=8, alpha=16.0), _recipe())
    _save(root, 3, first)
    assert load_engine_state(_checkpoint_recipe(), [], None, path=root, peft=first)[0] == 3
    # Bridge reads the root's tracker through a per-process cache; the step-3 read must not be reused.
    second = build_lora(LoraConfig(rank=8, alpha=32.0), _recipe())
    _save(root, 6, second)
    assert load_engine_state(_checkpoint_recipe(), [], None, path=root, peft=second)[0] == 6


def test_engine_save_and_load_pass_their_lora(monkeypatch):
    from megatron.bridge.engine.accumulation import AccumulationWindow

    engine = TrainingEngine.__new__(TrainingEngine)
    engine._closed = False
    engine._resident = True
    engine._recipe = SimpleNamespace()
    engine._model = []
    engine._optimizer = None
    engine._step = 1
    engine._window = AccumulationWindow()
    engine._pending_save = None
    engine._async_checkpoint_state = None
    engine._runtime = SimpleNamespace(pg_collection=None)
    engine._peft = peft = object()
    seen = []
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.save_engine_state",
        lambda *args, peft, **kwargs: seen.append(("save", peft)) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        "megatron.bridge.engine.engine.load_engine_state",
        lambda *args, path, peft: seen.append(("load", peft)) or (2, None),
    )
    engine.save_state("/ckpt")
    engine.load_state("/ckpt")
    assert seen == [("save", peft), ("load", peft)] and engine.step == 2


# ---- export -------------------------------------------------------------------------------------


class _AdapterBridge:
    def __init__(self):
        self.calls = []

    def export_adapter_weights(self, model, **kwargs):
        self.calls.append(("stream", kwargs))
        yield SimpleNamespace(param_name="model.layers.0.self_attn.q_proj.lora_A.weight", weight=torch.ones(2, 4))

    def save_hf_adapter(self, model, path, **kwargs):
        self.calls.append(("save", path, kwargs))


def test_export_hf_adapter_streams_unmerged_adapters_and_writes_peft_directories():
    bridge = _AdapterBridge()
    state = ExportState(bridge=bridge)
    peft = object()
    items = list(export_hf_adapter(state, [], peft, path=None, streaming=True, base_model_name_or_path="hf/base"))
    assert [name for name, _ in items] == ["model.layers.0.self_attn.q_proj.lora_A.weight"]
    assert bridge.calls[0] == ("stream", {"cpu": False, "show_progress": False})
    assert export_hf_adapter(state, [], peft, path="/out", streaming=False, base_model_name_or_path="hf/base") is None
    assert bridge.calls[1] == (
        "save",
        "/out",
        {"peft_config": peft, "base_model_name_or_path": "hf/base", "show_progress": False},
    )
    with pytest.raises(ValueError, match="path"):
        export_hf_adapter(state, [], peft, path=None, streaming=False, base_model_name_or_path="hf/base")


def test_export_hf_adapter_rejects_outer_shared_expert_adapters_before_exporting():
    # Bridge names each expert's own outer-shared factor after that expert's HF tensor; with packed experts
    # (Qwen3.5) every expert has the same name, and the PEFT file would keep a single expert's factor.
    bridge = _AdapterBridge()
    peft = build_lora(LoraConfig(rank=2, alpha=4.0, expert_adapters="outer_shared"), _recipe())
    for kwargs in ({"path": None, "streaming": True}, {"path": "/out", "streaming": False}):
        with pytest.raises(NotImplementedError, match="expert_adapters='outer_shared'"):
            export_hf_adapter(ExportState(bridge=bridge), [], peft, base_model_name_or_path="hf/base", **kwargs)
    assert bridge.calls == []


def test_engine_export_adapter_only_needs_lora_and_an_explicit_base_model(monkeypatch):
    engine = TrainingEngine.__new__(TrainingEngine)
    engine._closed = False
    engine._resident = True
    engine._output_head = "lm"
    engine._recipe = SimpleNamespace(hf_path="hf/base")  # the layout; the adapters' base may differ
    engine._model = []
    engine._export_state = None
    engine._peft = None
    with pytest.raises(ValueError, match="adapter_only=True needs an engine built with lora"):
        engine.export(path="/out", adapter_only=True, base_model_name_or_path="hf/base")
    bridge = _AdapterBridge()
    monkeypatch.setattr("megatron.bridge.engine.engine.make_export_state", lambda recipe: ExportState(bridge))
    engine._peft = peft = object()
    with pytest.raises(ValueError, match="needs base_model_name_or_path"):
        engine.export(path="/out", adapter_only=True)
    assert engine.export(path="/out", adapter_only=True, base_model_name_or_path="org/sft-model") is None
    assert bridge.calls == [
        (
            "save",
            "/out",
            {
                "peft_config": peft,
                "base_model_name_or_path": "org/sft-model",
                "show_progress": False,
            },
        )
    ]
    # The adapter stream and the merged export record no base model, so naming one there is an error.
    for kwargs in ({"streaming": True, "adapter_only": True}, {"path": "/merged"}):
        with pytest.raises(ValueError, match="only applies to export"):
            engine.export(base_model_name_or_path="org/sft-model", **kwargs)
    assert len(bridge.calls) == 1
