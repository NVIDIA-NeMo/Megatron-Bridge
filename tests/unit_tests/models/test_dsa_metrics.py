# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.common.dsa_metrics import register_dsa_metric_hook
from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig


def make_config(use_provider):
    kwargs = dict(num_layers=1, hidden_size=16, num_attention_heads=1)
    if use_provider:
        return GPTModelProvider(**kwargs, vocab_size=64)
    return GPTModelConfig(transformer=TransformerConfig(**kwargs), vocab_size=64)


def hooks_for(config, use_provider):
    return getattr(config, "_post_wrap_hooks", []) if use_provider else config.post_wrap_hooks


@pytest.mark.unit
@pytest.mark.parametrize("use_provider", [False, True])
@pytest.mark.parametrize("coefficient", [None, 0.0, -1.0, 1.0])
@pytest.mark.parametrize("has_initializer", [False, True])
def test_dsa_post_wrap_hook_preserves_model_and_groups(use_provider, coefficient, has_initializer):
    config = make_config(use_provider)
    config.dsa_indexer_loss_coeff = coefficient
    groups = object()
    initializer = Mock()
    dsa_logging = SimpleNamespace()
    if has_initializer:
        dsa_logging.initialize_dsa_metric_tracker = initializer
    register_dsa_metric_hook(config, groups)
    model = []  # A PP rank without local metric writers must still participate.
    with patch.dict(
        "sys.modules", {"megatron.core.transformer.experimental_attention_variant.dsa_logging": dsa_logging}
    ):
        for hook in hooks_for(config, use_provider):
            assert hook(model) is model
    if coefficient is not None and coefficient > 0 and has_initializer:
        initializer.assert_called_once_with(model, groups)
    else:
        initializer.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("use_provider", [False, True])
def test_dsa_hook_rebinds_groups_and_preserves_user_hooks(use_provider):
    config = make_config(use_provider)
    config.dsa_indexer_loss_coeff = 1.0
    user_hook = Mock(side_effect=lambda model: model)
    if use_provider:
        config.register_post_wrap_hook(user_hook)
    else:
        config.post_wrap_hooks.append(user_hook)
    register_dsa_metric_hook(config, object())
    groups = object()
    register_dsa_metric_hook(config, groups)
    hooks = hooks_for(config, use_provider)
    assert len(hooks) == 2
    assert hooks[0] is user_hook
    model = [object(), object()]
    initializer = Mock()
    with patch.dict(
        "sys.modules",
        {
            "megatron.core.transformer.experimental_attention_variant.dsa_logging": SimpleNamespace(
                initialize_dsa_metric_tracker=initializer
            )
        },
    ):
        for hook in hooks:
            assert hook(model) is model
    initializer.assert_called_once_with(model, groups)
    user_hook.assert_called_once_with(model)
    config.dsa_indexer_loss_coeff = 0.0
    register_dsa_metric_hook(config, object())
    assert hooks_for(config, use_provider) == [user_hook]


def make_dsa_model_config(family, use_provider):
    from megatron.bridge.models.deepseek.deepseek_v4_config import DeepSeekV4ModelConfig
    from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider
    from megatron.bridge.models.glm_moe_dsa.glm5_config import GLM5ModelConfig
    from megatron.bridge.models.glm_moe_dsa.glm5_provider import GLM5ModelProvider

    kwargs = dict(num_layers=2, hidden_size=16, num_attention_heads=1)
    if use_provider:
        cls = DeepSeekV4HybridModelProvider if family == "deepseek_v4" else GLM5ModelProvider
        return cls(**kwargs, vocab_size=64)
    cls = DeepSeekV4ModelConfig if family == "deepseek_v4" else GLM5ModelConfig
    return cls(transformer=cls.transformer_config_class(**kwargs), vocab_size=64)


def build_dsa_model(config, use_provider, groups, model, initializer):
    def wrapped_model(*args, **kwargs):
        # Metric storage must be initialized after all local chunks are wrapped.
        initializer.assert_not_called()
        return model

    if use_provider:
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("megatron.bridge.models.model_provider.get_model", side_effect=wrapped_model) as build,
        ):
            result = config.provide_distributed_model(pg_collection=groups, wrap_with_ddp=False)
        assert build.call_args.kwargs["pg_collection"] is groups
    else:
        with patch(
            "megatron.training.models.hybrid.unimodal_build_distributed_models", side_effect=wrapped_model
        ) as build:
            result = config.get_builder_cls()(config).build_distributed_models(groups, wrap_with_ddp=False)
        assert build.call_args.args[2] is groups
    assert result is model


@pytest.mark.unit
@pytest.mark.parametrize("family", ["deepseek_v4", "glm5"])
@pytest.mark.parametrize("use_provider", [False, True])
@pytest.mark.parametrize("empty_stage", [False, True])
@pytest.mark.parametrize("has_initializer", [False, True])
def test_model_build_registers_dsa_hook(family, use_provider, empty_stage, has_initializer):
    config = make_dsa_model_config(family, use_provider)
    config.dsa_indexer_loss_coeff = 1.0
    groups = object()
    model = [] if empty_stage else [object(), object()]
    initializer = Mock()
    logging_module = SimpleNamespace()
    if has_initializer:
        logging_module.initialize_dsa_metric_tracker = initializer
    with patch.dict(
        "sys.modules", {"megatron.core.transformer.experimental_attention_variant.dsa_logging": logging_module}
    ):
        build_dsa_model(config, use_provider, groups, model, initializer)
    if has_initializer:
        initializer.assert_called_once_with(model, groups)
    else:
        initializer.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("family", ["deepseek_v4", "glm5"])
@pytest.mark.parametrize("use_provider", [False, True])
def test_model_rebuild_rebinds_dsa_groups_and_can_disable_tracking(family, use_provider):
    config = make_dsa_model_config(family, use_provider)
    user_hook = Mock(side_effect=lambda model: model)
    if use_provider:
        config.register_post_wrap_hook(user_hook)
        assert config.post_wrap_hook is not None  # Safe before process groups are bound.
    else:
        config.post_wrap_hooks.append(user_hook)
    initializer = Mock()
    model = [object(), object()]
    with patch.dict(
        "sys.modules",
        {
            "megatron.core.transformer.experimental_attention_variant.dsa_logging": SimpleNamespace(
                initialize_dsa_metric_tracker=initializer
            )
        },
    ):
        for coefficient in (1.0, 1.0, 0.0):
            config.dsa_indexer_loss_coeff = coefficient
            groups = object()
            initializer.reset_mock()
            user_hook.reset_mock()
            build_dsa_model(config, use_provider, groups, model, initializer)
            hooks = hooks_for(config, use_provider)
            assert hooks[0] is user_hook
            user_hook.assert_called_once_with(model)
            if coefficient > 0:
                initializer.assert_called_once_with(model, groups)
                assert len(hooks) == 2
            else:
                initializer.assert_not_called()
                assert hooks == [user_hook]


@pytest.mark.unit
@pytest.mark.parametrize(
    "saved_builder",
    [
        "megatron.bridge.models.hybrid.BridgeHybridModelBuilder",
        "megatron.bridge.models.hybrid.hybrid_builder.BridgeHybridModelBuilder",
    ],
)
def test_restored_deepseek_config_registers_dsa_hook(saved_builder):
    from megatron.bridge.models.common.base import ModelConfig

    config = make_dsa_model_config("deepseek_v4", False)
    config.dsa_indexer_loss_coeff = 1.0
    saved = config.as_dict()
    saved["_builder_"] = saved_builder
    restored = ModelConfig.from_dict(saved)
    initializer = Mock()
    groups = object()
    model = [object()]
    with patch.dict(
        "sys.modules",
        {
            "megatron.core.transformer.experimental_attention_variant.dsa_logging": SimpleNamespace(
                initialize_dsa_metric_tracker=initializer
            )
        },
    ):
        build_dsa_model(restored, False, groups, model, initializer)
    initializer.assert_called_once_with(model, groups)


@pytest.mark.unit
def test_deepseek_preserves_custom_builder():
    from megatron.bridge.models.hybrid import BridgeHybridModelBuilder

    class CustomBuilder(BridgeHybridModelBuilder):
        pass

    config = make_dsa_model_config("deepseek_v4", False)
    config.builder = "megatron.bridge.models.custom_dsa_test.CustomBuilder"
    with patch.dict(
        "sys.modules", {"megatron.bridge.models.custom_dsa_test": SimpleNamespace(CustomBuilder=CustomBuilder)}
    ):
        assert config.get_builder_cls() is CustomBuilder
