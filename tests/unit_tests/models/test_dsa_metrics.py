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
