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

"""Shared Core override composition and compatibility with Bridge providers."""

from dataclasses import dataclass
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig, ParamKey, _get_param_groups, get_standard_config_overrides
from megatron.core.optimizer.optimizer import get_param_group_identifier_tuple
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.transformer import TransformerConfig
from megatron.training.config.optimizer_config import LoRAPlusOptimizerOverrides, build_optimizer_config_overrides

from megatron.bridge.models import GPTModelProvider
from megatron.bridge.peft.lora import get_lora_plus_config_overrides
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.optim import _resolve_optimizer_config_overrides, setup_optimizer
from megatron.bridge.training.post_training.lsq import (
    LSQOptimizerConfigOverrideProvider,
    LSQOptimizerConfigOverrides,
)
from megatron.bridge.training.utils.omegaconf_utils import process_config_with_overrides


pytestmark = pytest.mark.unit


@dataclass
class WeightLRHook:
    multiplier: float = 4.0

    def __call__(self, *, optimizer_config, model):
        assert model[0].weight.requires_grad
        return {ParamKey(name=("weight", "*.weight")): {"max_lr": optimizer_config.lr * self.multiplier}}


def weight_lr_hook(*, optimizer_config, model):
    return WeightLRHook()(optimizer_config=optimizer_config, model=model)


@dataclass
class LoraProvider(OptimizerConfigOverrideProvider):
    ratio: float = 16.0

    def build_config_overrides(self, context):
        return get_lora_plus_config_overrides(context.optimizer_config, self.ratio)


def _model():
    model = torch.nn.Linear(4, 4)
    model.config = TransformerConfig(num_layers=1, hidden_size=4, num_attention_heads=1)
    model.adapter = torch.nn.Module()
    model.adapter.linear_in = torch.nn.Linear(4, 4, bias=False)
    model.adapter.linear_out = torch.nn.Linear(4, 4, bias=False)
    model.scale_amax_pre = torch.nn.Parameter(torch.ones(4, 4))
    return [model]


def _context(model=None):
    return OptimizerConfigOverrideProviderContext(
        SchedulerConfig(), OptimizerConfig(lr=0.001, min_lr=0.0001), _model() if model is None else model
    )


def _container():
    return ConfigContainer(
        train=TrainingConfig(),
        model=GPTModelProvider(num_layers=1, hidden_size=128, num_attention_heads=1, seq_length=8),
        optimizer=OptimizerConfig(lr=0.001, min_lr=0.0001),
        scheduler=SchedulerConfig(),
        dataset=GPTDatasetConfig(seq_length=8, random_seed=1234),
        logger=LoggerConfig(),
        tokenizer=TokenizerConfig(),
        checkpoint=CheckpointConfig(),
    )


@pytest.fixture
def single_rank_group(tmp_path):
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rdzv", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def _policy_by_parameter(context, overrides):
    groups = _get_param_groups(context.model, context.optimizer_config, overrides)
    return {
        id(param): {key: group[key] for key in ("max_lr", "min_lr", "wd_mult", "default_config")}
        for group in groups
        for param in group["params"]
    }


@pytest.mark.parametrize("mode", ["extend", "replace"])
@pytest.mark.parametrize("custom", [{}, WeightLRHook(), weight_lr_hook])
def test_mapping_and_callable_match_mlm_group_policy(single_rank_group, mode, custom):
    context = _context()
    bridge = _resolve_optimizer_config_overrides(context, OptimizerConfigOverrideProvider(), custom, mode)
    mlm = build_optimizer_config_overrides(context.optimizer_config, context.model, custom, mode=mode)
    actual = _policy_by_parameter(context, bridge)
    assert actual == _policy_by_parameter(context, mlm)
    assert actual[id(context.model[0].bias)]["wd_mult"] == (0.0 if mode == "extend" else 1.0)
    assert actual[id(context.model[0].weight)]["max_lr"] == (0.004 if custom else 0.001)


@pytest.mark.parametrize("override", [{"max_lr": 0.001}, {"wd_mult": 1.0}])
def test_noop_recipe_override_preserves_checkpoint_group_identity(single_rank_group, override):
    from megatron.core.optimizer.optimizer import get_param_group_identifier_tuple

    context = _context()
    context.model[0].register_parameter("second_weight", torch.nn.Parameter(torch.ones(4, 4)))
    resolved = _resolve_optimizer_config_overrides(context, None, {ParamKey(name="weight"): override}, "extend")
    groups = _get_param_groups(context.model, context.optimizer_config, resolved)
    identities = [get_param_group_identifier_tuple(group) for group in groups]
    assert len(groups) == len(set(identities)) == 2


@pytest.mark.parametrize("result", [None, {}])
def test_legacy_provider_empty_result_keeps_its_replacement_semantics(result):
    provider = Mock()
    provider.build_config_overrides.return_value = result
    context = _context()
    assert _resolve_optimizer_config_overrides(context, provider, None, "extend") == result
    provider.build_config_overrides.assert_called_once_with(context)


@pytest.mark.parametrize("kind", ["default", "lsq", "lora"])
def test_legacy_providers_adapt_as_complete_mappings(single_rank_group, kind):
    context = _context()
    provider = {
        "default": OptimizerConfigOverrideProvider(),
        "lsq": LSQOptimizerConfigOverrideProvider(scale_lr=0.008, scale_min_lr=0.0008, scale_wd_mult=0.0),
        "lora": LoraProvider(),
    }[kind]
    with patch("megatron.bridge.training.post_training.lsq._check_modelopt_lsq_gradients"):
        bridge = _resolve_optimizer_config_overrides(context, provider, None, "extend")
    mlm = build_optimizer_config_overrides(context.optimizer_config, context.model, bridge, mode="replace")
    actual = _policy_by_parameter(context, bridge)
    normalized = _policy_by_parameter(context, mlm)
    for param_id, policy in actual.items():
        assert {k: v for k, v in policy.items() if k != "default_config"} == {
            k: v for k, v in normalized[param_id].items() if k != "default_config"
        }
        # New mappings omit redundant base-LR overrides; legacy providers preserve
        # their existing group structure and logging classification for resume.
        if policy["default_config"] != normalized[param_id]["default_config"]:
            assert kind == "lora"
            assert policy["max_lr"] == context.optimizer_config.lr
            assert normalized[param_id]["default_config"] is True
    assert actual[id(context.model[0].bias)]["wd_mult"] == 0.0
    if kind == "lsq":
        assert actual[id(context.model[0].scale_amax_pre)]["max_lr"] == 0.008
    if kind == "lora":
        assert actual[id(context.model[0].adapter.linear_out.weight)]["max_lr"] == 0.016


@pytest.mark.parametrize("mode,custom", [("extend", {}), ("replace", None)])
def test_custom_provider_and_new_policy_are_not_silently_combined(mode, custom):
    with pytest.raises(ValueError, match="Choose either"):
        _resolve_optimizer_config_overrides(_context(), LoraProvider(), custom, mode)


@pytest.mark.parametrize("custom", [lambda **kwargs: None, {"weight": {"max_lr": 0.1}}])
def test_invalid_hook_fails_before_optimizer_creation(custom):
    with pytest.raises(TypeError, match="ParamKey"):
        _resolve_optimizer_config_overrides(_context(), None, custom, "extend")
    with pytest.raises(ValueError, match="Unknown optimizer override mode"):
        _resolve_optimizer_config_overrides(_context(), None, {}, "last_wins")


@pytest.mark.parametrize("kind", ["mapping", "callable", "function", "provider", "lsq", "lsq_provider"])
@pytest.mark.parametrize("yaml_round_trip", [False, True])
def test_policy_configuration_round_trip(tmp_path, kind, yaml_round_trip):
    cfg = _container()
    if kind == "provider":
        cfg.optimizer_config_override_provider = LoraProvider()
    elif kind == "lsq_provider":
        cfg.optimizer_config_override_provider = LSQOptimizerConfigOverrideProvider(0.008, 0.0008, 0.0)
    else:
        cfg.optimizer_config_overrides = {
            "mapping": {ParamKey(name=("weight", "*.weight")): {"max_lr": 0.004}},
            "callable": WeightLRHook(),
            "function": weight_lr_hook,
            "lsq": LSQOptimizerConfigOverrides(0.008, 0.0008, 0.0),
        }[kind]
    if yaml_round_trip:
        path = tmp_path / "recipe.yaml"
        cfg.to_yaml(str(path))
        restored = ConfigContainer.from_yaml(str(path))
    else:
        restored = ConfigContainer.from_dict(cfg.to_dict())
    context = _context()
    with patch("megatron.bridge.training.post_training.lsq._check_modelopt_lsq_gradients"):
        actual = _resolve_optimizer_config_overrides(
            context,
            restored.optimizer_config_override_provider,
            restored.optimizer_config_overrides,
            restored.optimizer_config_overrides_mode,
        )
        expected = _resolve_optimizer_config_overrides(
            context,
            cfg.optimizer_config_override_provider,
            cfg.optimizer_config_overrides,
            cfg.optimizer_config_overrides_mode,
        )
        assert actual == expected
    if kind.startswith("lsq"):
        assert actual[ParamKey(name=("*_amax_pre", "*_amax_post"))] == {
            "max_lr": 0.008,
            "min_lr": 0.0008,
            "wd_mult": 0.0,
        }


def test_setup_passes_actual_model_and_shared_mapping_to_core():
    context = _context()
    hook = Mock(return_value={ParamKey(name="weight"): {"max_lr": 0.004}})
    with (
        patch("megatron.training.optim.get_megatron_optimizer") as factory,
        patch("megatron.training.optim.build_optimizer_param_scheduler"),
    ):
        setup_optimizer(
            context.optimizer_config, context.scheduler_config, context.model, optimizer_config_overrides=hook
        )
    hook.assert_called_once_with(optimizer_config=context.optimizer_config, model=context.model)
    mapping = factory.call_args.kwargs["config_overrides"]
    assert mapping[ParamKey(name="weight")]["max_lr"] == 0.004
    assert any(value.get("wd_mult") == 0.0 for value in mapping.values())


def test_new_callable_receives_model_chunks_for_single_module_callers():
    model = _model()[0]
    context = _context(model)
    hook = Mock(return_value={})
    _resolve_optimizer_config_overrides(context, None, hook, "extend")
    hook.assert_called_once_with(optimizer_config=context.optimizer_config, model=[model])


def test_mup_does_not_silently_overwrite_identical_selector():
    context = _context()
    key = ParamKey(name="weight")
    context.model[0].config.use_mup = True
    context.model[0].config.mup_width_mult = 2.0
    with (
        patch(
            "megatron.training.config.optimizer_config.get_mup_config_overrides",
            return_value={key: {"max_lr": 0.0005}},
        ),
        patch("megatron.training.optim.get_megatron_optimizer") as factory,
        pytest.raises(ValueError, match="Conflicting optimizer overrides"),
    ):
        setup_optimizer(
            context.optimizer_config,
            context.scheduler_config,
            context.model,
            optimizer_config_overrides={key: {"max_lr": 0.004}},
        )
    factory.assert_not_called()


@pytest.mark.parametrize("mode", ["extend", "replace"])
def test_mup_overlap_conflicts_reach_core_parameter_validation(single_rank_group, mode):
    context = _context()
    context.model[0].config.use_mup = True
    context.model[0].config.mup_width_mult = 2.0
    with (
        patch(
            "megatron.training.optim.get_megatron_optimizer",
            side_effect=lambda **kwargs: _get_param_groups(
                kwargs["model_chunks"], kwargs["config"], kwargs["config_overrides"]
            ),
        ),
        pytest.raises(ValueError, match="Conflicting overrides for max_lr"),
    ):
        setup_optimizer(
            context.optimizer_config,
            context.scheduler_config,
            context.model,
            optimizer_config_overrides=WeightLRHook(),
            optimizer_config_overrides_mode=mode,
        )


def test_mimo_does_not_silently_discard_new_recipe_policy():
    from megatron.core.models.mimo.model.base import MimoModel

    model = MimoModel.__new__(MimoModel)
    torch.nn.Module.__init__(model)
    with pytest.raises(NotImplementedError, match="per module"):
        _resolve_optimizer_config_overrides(
            _context([model]), None, {ParamKey(name="weight"): {"max_lr": 0.1}}, "extend"
        )


@pytest.mark.parametrize("mode", ["extend", "replace"])
def test_new_mapping_preserves_lora_warmup_constraint(mode):
    context = _context()
    context.scheduler_config.lr_warmup_init = 0.0001
    with (
        patch("megatron.training.optim.get_megatron_optimizer") as factory,
        pytest.raises(ValueError, match=r"LoRA\+.*lr_warmup_init=0"),
    ):
        setup_optimizer(
            context.optimizer_config,
            context.scheduler_config,
            context.model,
            optimizer_config_overrides=get_lora_plus_config_overrides(context.optimizer_config, 16),
            optimizer_config_overrides_mode=mode,
        )
    factory.assert_not_called()


def test_lsq_callable_matches_legacy_and_mlm_groups(single_rank_group):
    context = _context()
    policy = LSQOptimizerConfigOverrides(0.008, 0.0008, 0.0)
    provider = LSQOptimizerConfigOverrideProvider(0.008, 0.0008, 0.0)
    with patch("megatron.bridge.training.post_training.lsq._check_modelopt_lsq_gradients"):
        legacy = _resolve_optimizer_config_overrides(context, provider, None, "extend")
        bridge = _resolve_optimizer_config_overrides(context, None, policy, "extend")
        mlm = build_optimizer_config_overrides(context.optimizer_config, context.model, policy)
    actual = _policy_by_parameter(context, bridge)
    assert actual == _policy_by_parameter(context, legacy) == _policy_by_parameter(context, mlm)
    scale_group = actual[id(context.model[0].scale_amax_pre)]
    assert (scale_group["max_lr"], scale_group["min_lr"], scale_group["wd_mult"]) == (0.008, 0.0008, 0.0)
    assert actual[id(context.model[0].bias)]["wd_mult"] == 0.0


@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_lsq_callable_settings_can_be_overridden(tmp_path, source):
    cfg = _container()
    cfg.optimizer_config_overrides = LSQOptimizerConfigOverrides(0.001, None, 1.0)
    if source == "cli":
        process_config_with_overrides(
            cfg,
            cli_overrides=[
                "optimizer_config_overrides.scale_lr=0.008",
                "optimizer_config_overrides.scale_min_lr=0.0008",
                "optimizer_config_overrides.scale_wd_mult=0.0",
            ],
        )
    else:
        path = tmp_path / "overrides.yaml"
        path.write_text(
            "optimizer_config_overrides:\n  scale_lr: 0.008\n  scale_min_lr: 0.0008\n  scale_wd_mult: 0.0\n"
        )
        process_config_with_overrides(cfg, config_filepath=str(path))
    context = _context()
    with patch("megatron.bridge.training.post_training.lsq._check_modelopt_lsq_gradients"):
        actual = _resolve_optimizer_config_overrides(context, None, cfg.optimizer_config_overrides, "extend")
    assert actual[ParamKey(name=("*_amax_pre", "*_amax_post"))] == {
        "max_lr": 0.008,
        "min_lr": 0.0008,
        "wd_mult": 0.0,
    }


@pytest.mark.parametrize("policy,single_model", [(None, True), ({}, False), ("custom", False)])
def test_legacy_setup_keeps_factory_provider_and_finalization_contract(single_rank_group, policy, single_model):
    """Exercise the downstream six-argument API through real optimizer updates."""
    from copy import deepcopy

    from megatron.core.optimizer.optimizer import FP32Optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection

    captured = {}

    class CustomConfig(OptimizerConfig):
        finalizations = 0

        def finalize(self):
            self.finalizations += 1
            self.lr *= 2

        def provide(self, *, model_chunks, config_overrides, use_gloo_process_groups, pg_collection):
            assert self.finalizations == 1
            captured.update(
                model=model_chunks, overrides=config_overrides, groups=pg_collection, gloo=use_gloo_process_groups
            )
            chunks = model_chunks if isinstance(model_chunks, list) else [model_chunks]
            param_groups = _get_param_groups(chunks, self, config_overrides)
            return FP32Optimizer(torch.optim.SGD(param_groups, lr=self.lr, momentum=0.9), self, None)

    class Provider(OptimizerConfigOverrideProvider):
        def build_config_overrides(self, context):
            assert context.optimizer_config is config.optimizer
            assert context.scheduler_config is config.scheduler
            assert context.model is model
            return overrides

    config = _container()
    config.train.train_iters = 20
    config.train.global_batch_size = 8
    config.train.micro_batch_size = 1
    config.scheduler.lr_warmup_iters = 2
    config.scheduler.start_weight_decay = config.scheduler.end_weight_decay = 0.01
    config.optimizer = CustomConfig(lr=0.001, min_lr=0.0001, clip_grad=0.0)
    config.dataset.reset_position_ids = False
    config.dataset.reset_attention_mask = False
    config.dataset.eod_mask_loss = False
    config.validate()
    model = _model()
    model = model[0] if single_model else model
    overrides = {ParamKey(name="weight"): {"max_lr": 0.008}} if policy == "custom" else policy
    groups = ProcessGroupCollection()
    scheduler_before = deepcopy(vars(config.scheduler))
    optimizer, scheduler = setup_optimizer(config.optimizer, config.scheduler, model, False, groups, Provider())
    assert captured["model"] is model
    assert captured["groups"] is groups and captured["gloo"] is False
    assert captured["overrides"] == overrides
    assert config.optimizer.finalizations == 1
    assert vars(config.scheduler) == scheduler_before
    assert scheduler.lr_decay_steps == 160 and scheduler.lr_warmup_steps == 16
    scheduler.step(increment=8)
    expected_lrs = {0.001, 0.004} if policy == "custom" else {0.001}
    assert {group["lr"] for group in optimizer.param_groups} == expected_lrs
    parameter = model.weight if single_model else model[0].weight
    before = parameter.detach().clone()
    for group in optimizer.param_groups:
        for param in group["params"]:
            param.main_grad = torch.ones_like(param)
    assert optimizer.step()[0]
    assert not torch.equal(parameter, before)
    assert optimizer.state_dict() and scheduler.state_dict()["num_steps"] == 8


def test_bridge_wrapper_preserves_legacy_step_rescaling_behavior():
    """The shared builder must not activate previously ignored Bridge settings."""

    def build_custom(**kwargs):
        return torch.optim.SGD(_model()[0].parameters(), lr=0.01)

    # A custom config owns construction; this mirrors existing config.provide users.
    class CustomConfig(OptimizerConfig):
        provide = staticmethod(build_custom)

    config = CustomConfig(lr=0.01, min_lr=0.001)
    schedule = SchedulerConfig(
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        step_lr_rescale_schedule="0:0.5,8:1.0",
        step_lr_rescale_warmup_samples=2,
    )
    schedule.lr_warmup_steps = 0
    schedule.lr_decay_steps = schedule.wd_incr_steps = 20
    optimizer, scheduler = setup_optimizer(config, schedule, _model())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    assert scheduler.step_lr_rescale_schedule == ()
    assert schedule.step_lr_rescale_schedule == "0:0.5,8:1.0"
    assert schedule.step_lr_rescale_warmup_samples == 2


@pytest.mark.parametrize(
    "ratio,b_peak,b_min", [(1.0, 0.001, 0.0001), (16.0, 0.016, 0.0016), (0.25, 0.00025, 0.000025)]
)
def test_lora_legacy_groups_resume_without_changing_parameter_order(single_rank_group, tmp_path, ratio, b_peak, b_min):
    """A pre-adoption checkpoint keeps its groups, Adam history and next updates."""
    config = OptimizerConfig(lr=0.001, min_lr=0.0001, weight_decay=0.1)
    # Capture the old helper's complete policy independently of the shared implementation.
    legacy = get_standard_config_overrides(config)
    legacy.update(
        {
            ParamKey(name="*.linear_in.weight"): {"max_lr": 0.001, "min_lr": 0.0001},
            ParamKey(name="*.linear_out.weight"): {"max_lr": b_peak, "min_lr": b_min},
        }
    )
    adopted = get_lora_plus_config_overrides(config, ratio)
    assert adopted == legacy

    def build(policy):
        model = _model()[0]
        groups = _get_param_groups([model], config, policy)
        names = {id(param): name for name, param in model.named_parameters()}
        signature = [
            (
                {key: value for key, value in group.items() if key != "params"},
                [names[id(param)] for param in group["params"]],
            )
            for group in groups
        ]
        optimizer = torch.optim.AdamW(groups, lr=config.lr)
        scheduler = OptimizerParamScheduler(optimizer, 0.0, 0.001, 0.0001, 2, 8, "linear", 0.1, 0.1, 8, "constant")
        return model, optimizer, scheduler, signature

    def step(model, optimizer, scheduler):
        scheduler.step(1)
        for i, param in enumerate(model.parameters()):
            param.grad = torch.full_like(param, 0.1 * (i + 1))
        optimizer.step()

    model, optimizer, scheduler, old_groups = build(legacy)
    for _ in range(3):
        step(model, optimizer, scheduler)
    path = tmp_path / "legacy-lora.pt"
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, path
    )
    restored, opt2, sched2, new_groups = build(adopted)
    assert old_groups == new_groups
    checkpoint = torch.load(path, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    opt2.load_state_dict(checkpoint["optimizer"])
    sched2.load_state_dict(checkpoint["scheduler"])
    for _ in range(3):
        step(model, optimizer, scheduler)
        step(restored, opt2, sched2)
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(optimizer.state[expected][key], opt2.state[actual][key], rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [1.0, 16.0, 0.25])
def test_shared_lora_callable_matches_mlm_and_legacy_schedule(single_rank_group, ratio):
    context = _context()
    policy = LoRAPlusOptimizerOverrides(ratio)
    bridge = _resolve_optimizer_config_overrides(context, None, policy, "extend")
    mlm = build_optimizer_config_overrides(context.optimizer_config, context.model, policy)
    groups = _get_param_groups(context.model, context.optimizer_config, bridge)
    identities = [get_param_group_identifier_tuple(group) for group in groups]
    assert len(identities) == len(set(identities))
    assert _policy_by_parameter(context, bridge) == _policy_by_parameter(context, mlm)
    old_groups = _get_param_groups(
        context.model, context.optimizer_config, get_lora_plus_config_overrides(context.optimizer_config, ratio)
    )
    schedules = []
    for param_groups in (groups, old_groups):
        optimizer = torch.optim.SGD(param_groups, lr=0.001)
        schedules.append(
            (
                optimizer,
                OptimizerParamScheduler(optimizer, 0.0, 0.001, 0.0001, 2, 8, "linear", 0.1, 0.1, 8, "constant"),
            )
        )
    for _ in range(9):
        effective = []
        for optimizer, scheduler in schedules:
            effective.append(
                {
                    id(param): (group["lr"], group["weight_decay"])
                    for group in optimizer.param_groups
                    for param in group["params"]
                }
            )
            scheduler.step(1)
        assert effective[0] == effective[1]


@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_shared_lora_callable_roundtrip_and_overrides(tmp_path, source):
    cfg = _container()
    cfg.optimizer_config_overrides = LoRAPlusOptimizerOverrides(16.0)
    if source == "cli":
        process_config_with_overrides(cfg, cli_overrides=["optimizer_config_overrides.ratio=8.0"])
    else:
        path = tmp_path / "lora-overrides.yaml"
        path.write_text("optimizer_config_overrides:\n  ratio: 8.0\n")
        process_config_with_overrides(cfg, config_filepath=str(path))
    restored = ConfigContainer.from_dict(cfg.to_dict())
    actual = restored.optimizer_config_overrides(optimizer_config=restored.optimizer, model=[])
    assert actual[ParamKey(name="*.linear_out.weight")] == {"max_lr": 0.008, "min_lr": 0.0008}


@pytest.mark.parametrize("initial_lr", [0.0, 0.00001])
def test_shared_lora_callable_reaches_bridge_setup(initial_lr):
    context = _context()
    context.scheduler_config.lr_warmup_init = initial_lr
    with (
        patch("megatron.training.optim.get_megatron_optimizer") as factory,
        patch("megatron.training.optim.build_optimizer_param_scheduler"),
    ):
        if initial_lr:
            with pytest.raises(ValueError, match="lr_warmup_init=0"):
                setup_optimizer(
                    context.optimizer_config,
                    context.scheduler_config,
                    context.model,
                    optimizer_config_overrides=LoRAPlusOptimizerOverrides(16.0),
                )
            factory.assert_not_called()
        else:
            setup_optimizer(
                context.optimizer_config,
                context.scheduler_config,
                context.model,
                optimizer_config_overrides=LoRAPlusOptimizerOverrides(16.0),
            )
            actual = factory.call_args.kwargs["config_overrides"]
            assert actual[ParamKey(name="*.linear_out.weight")] == {"max_lr": 0.016, "min_lr": 0.0016}
