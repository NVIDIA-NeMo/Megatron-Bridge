"""Checkpoint RoPE must map once, without mutating or retaining HF config."""

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType
from transformers import PretrainedConfig

from megatron.bridge.diffusion.conversion.nemotron_labs_diffusion.nemotron_labs_diffusion_bridge import (
    NemotronLabsDiffusionBridge,
)
from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import (
    Ministral3RotaryEmbedding,
    NemotronLabsDiffusionAttention,
)


pytestmark = [pytest.mark.unit]


def source_config(rope_type, factor=8.0):
    rope = {"rope_type": rope_type, "rope_theta": 1000000.0}
    if rope_type == "yarn":
        rope.update(
            factor=factor,
            original_max_position_embeddings=16384,
            beta_fast=32.0,
            beta_slow=1.0,
            mscale=1.0,
            mscale_all_dim=1.0,
            llama_4_scaling_beta=0.1,
        )
    return PretrainedConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=1024,
        max_position_embeddings=131072,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        rope_parameters=rope,
        block_size=16,
    )


def attention(config, layer, rope=None):
    pg = MagicMock()
    pg.tp.size.return_value = 1
    return NemotronLabsDiffusionAttention(
        config,
        layer,
        AttnMaskType.causal,
        "self",
        pg_collection=pg,
        rope_module=rope,
    )


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("rope_type,factor", [("default", 1.0), ("yarn", 8.0), ("yarn", 16.0)])
def test_checkpoint_rope_matches_hf_without_retaining_config(nested, rope_type, factor):
    source = source_config(rope_type, factor)
    original = copy.deepcopy(source.to_dict())
    wrapper = SimpleNamespace(text_config=source) if nested else source
    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=wrapper))
    assert provider.hf_config is None
    assert provider.rope_type == rope_type
    assert provider.num_query_groups == 2
    assert provider.block_size == 16
    # Training context changes must not alter checkpoint positional scaling.
    provider.seq_length = 49152
    expected = Ministral3RotaryEmbedding(copy.deepcopy(source))
    rope = Ministral3RotaryEmbedding.from_megatron_config(provider)
    assert not any(isinstance(value, PretrainedConfig) for value in vars(rope).values())
    for layer in (1, 2):
        module = attention(provider, layer, rope)
        assert module.rope_embedding_module is rope
        torch.testing.assert_close(module.rope_embedding_module.inv_freq, expected.inv_freq)
        assert module.rope_embedding_module.attention_scaling == expected.attention_scaling
        assert module.beta == (0.1 if rope_type == "yarn" else None)
    assert source.to_dict() == original


def test_explicit_native_override_applies_to_every_layer_without_source_mutation():
    source = source_config("yarn")
    original = copy.deepcopy(source.to_dict())
    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source))
    provider.yarn_rotary_scaling_factor = 4.0
    expected_source = copy.deepcopy(source)
    expected_source.rope_parameters["factor"] = 4.0
    expected = Ministral3RotaryEmbedding(expected_source)
    modules = [attention(provider, layer) for layer in (1, 2)]
    for module in modules:
        torch.testing.assert_close(module.rope_embedding_module.inv_freq, expected.inv_freq)
    assert source.to_dict() == original


def test_yarn_without_query_scaling_is_supported():
    source = source_config("yarn")
    del source.rope_parameters["llama_4_scaling_beta"]
    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source))
    assert attention(provider, 1).beta is None
    assert provider.yarn_rotary_scaling_factor == 8.0


def test_query_scaling_requires_explicit_parameters():
    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source_config("default")))
    provider.apply_llama4_style_query_key_layer_scaling = True
    with pytest.raises(ValueError, match="requires beta"):
        attention(provider, 1)


def test_unsupported_rope_fails_at_conversion():
    source = source_config("default")
    source.rope_parameters["rope_type"] = "dynamic"
    with pytest.raises(ValueError, match="Unsupported.*dynamic"):
        NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source))


def test_legacy_provider_cannot_silently_use_native_defaults():
    source = source_config("yarn")
    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source))
    provider.hf_config = source
    with pytest.raises(ValueError, match="Legacy.*hf_config"):
        Ministral3RotaryEmbedding.from_megatron_config(provider)


@pytest.mark.parametrize("rope_type", ["default", "yarn"])
def test_native_provider_checkpoint_config_roundtrip(rope_type):
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
    from megatron.bridge.utils.instantiate_utils import instantiate

    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source_config(rope_type)))
    serialized = _ConfigContainerBase._convert_value_to_dict(provider)
    assert serialized["hf_config"] is None
    restored = instantiate(serialized)
    assert restored.rope_type == provider.rope_type
    assert restored.llama4_scaling_beta == provider.llama4_scaling_beta
    expected = Ministral3RotaryEmbedding.from_megatron_config(provider)
    actual = Ministral3RotaryEmbedding.from_megatron_config(restored)
    torch.testing.assert_close(actual.inv_freq, expected.inv_freq)
    assert actual.attention_scaling == expected.attention_scaling


@pytest.mark.parametrize("use_factory", [False, True])
@pytest.mark.parametrize("build_fails", [False, True])
def test_provide_preserves_layer_spec_checkpoint_config(use_factory, build_fails):
    from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
    from megatron.core.transformer.dot_product_attention import DotProductAttention
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

    from megatron.bridge.models import Ministral3ModelProvider
    from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
    from megatron.bridge.utils.instantiate_utils import instantiate

    spec = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=SelfAttention,
                submodules=SelfAttentionSubmodules(
                    linear_qkv=torch.nn.Identity,
                    core_attention=DotProductAttention,
                    linear_proj=torch.nn.Identity,
                ),
            )
        ),
    )
    provider = NemotronLabsDiffusionBridge().provider_bridge(SimpleNamespace(config=source_config("yarn")))
    original_spec = (lambda config: spec) if use_factory else spec
    provider.transformer_layer_spec = original_spec
    serialized_spec = _ConfigContainerBase._convert_value_to_dict(spec)
    model = object()

    def build(config, pre_process, post_process, vp_stage):
        assert (pre_process, post_process, vp_stage) == (True, False, 0)
        runtime_spec = config.transformer_layer_spec
        assert runtime_spec is not spec
        core_attention = runtime_spec.submodules.self_attention.submodules.core_attention
        assert core_attention.func is NemotronLabsDiffusionAttention
        assert isinstance(core_attention.keywords["rope_module"], Ministral3RotaryEmbedding)
        if build_fails:
            raise RuntimeError("model construction failed")
        return model

    with patch.object(Ministral3ModelProvider, "provide_language_model", autospec=True, side_effect=build):
        if build_fails:
            with pytest.raises(RuntimeError, match="model construction failed"):
                provider.provide(pre_process=True, post_process=False, vp_stage=0)
        else:
            assert provider.provide(pre_process=True, post_process=False, vp_stage=0) is model

    assert provider.transformer_layer_spec is original_spec
    assert spec.submodules.self_attention.submodules.core_attention is DotProductAttention
    assert _ConfigContainerBase._convert_value_to_dict(spec) == serialized_spec
    if not use_factory:
        restored = instantiate(_ConfigContainerBase._convert_value_to_dict(provider))
        assert (
            restored.transformer_layer_spec.submodules.self_attention.submodules.core_attention is DotProductAttention
        )
