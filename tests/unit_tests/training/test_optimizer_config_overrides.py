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
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig, ParamKey, _get_param_groups
from megatron.core.transformer import TransformerConfig
from megatron.training.config.optimizer_config import build_optimizer_config_overrides

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
    with pytest.raises(TypeError, match="ParamKey-to-override"):
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
        patch("megatron.bridge.training.optim.get_megatron_optimizer") as factory,
        patch("megatron.bridge.training.optim._get_scheduler"),
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
    with (
        patch(
            "megatron.bridge.training.optim.get_model_config",
            return_value=SimpleNamespace(use_mup=True, mup_width_mult=2.0),
        ),
        patch("megatron.bridge.training.optim.get_mup_config_overrides", return_value={key: {"max_lr": 0.0005}}),
        patch("megatron.bridge.training.optim.get_megatron_optimizer") as factory,
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
            "megatron.bridge.training.optim.get_megatron_optimizer",
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
        patch("megatron.bridge.training.optim.get_megatron_optimizer") as factory,
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
