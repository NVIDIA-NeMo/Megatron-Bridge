# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import pytest
from megatron.core import parallel_state
from megatron.core.transformer.enums import AttnBackend, AttnMaskType

from megatron.bridge.models.gpt.dca import (
    DualChunkGPTModelConfig,
    DualChunkGPTModelProvider,
    get_dca_gpt_layer_spec,
    validate_dual_chunk_gpt_config,
)
from megatron.bridge.models.transformer.dca import (
    DualChunkAttention,
    DualChunkSelfAttention,
    DualChunkTransformerConfig,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _pipeline_parallel_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0)


def _make_transformer_config(**overrides: object) -> DualChunkTransformerConfig:
    defaults = {
        "num_layers": 2,
        "hidden_size": 32,
        "num_attention_heads": 4,
        "num_query_groups": 4,
        "apply_rope_fusion": False,
        "use_cpu_initialization": True,
        "transformer_impl": "local",
        "dca_chunk_size": 16,
        "dca_local_size": 4,
    }
    defaults.update(overrides)
    return DualChunkTransformerConfig(**defaults)


def _make_gpt_config(**overrides: object) -> DualChunkGPTModelConfig:
    defaults = {
        "transformer": _make_transformer_config(),
        "vocab_size": 128,
        "position_embedding_type": "rope",
    }
    defaults.update(overrides)
    return DualChunkGPTModelConfig(**defaults)


def test_dca_layer_spec_patches_self_attention_and_core_attention() -> None:
    config = _make_gpt_config()
    spec = get_dca_gpt_layer_spec(config)

    assert len(spec.layer_specs) == 2
    for layer_spec in spec.layer_specs:
        self_attention_spec = layer_spec.submodules.self_attention
        assert self_attention_spec.module is DualChunkSelfAttention
        assert self_attention_spec.params["attn_mask_type"] == AttnMaskType.causal
        assert self_attention_spec.submodules.core_attention.module is DualChunkAttention
        assert self_attention_spec.submodules.core_attention.params == {
            "dca_chunk_size": 16,
            "dca_local_size": 4,
        }


def test_modern_gpt_config_uses_dca_layer_spec_by_default() -> None:
    config = _make_gpt_config()
    assert config.transformer_layer_spec is get_dca_gpt_layer_spec
    assert get_dca_gpt_layer_spec(config).layer_specs[0].submodules.self_attention.module is DualChunkSelfAttention


def test_modern_gpt_config_supports_yarn_settings() -> None:
    transformer = _make_transformer_config(
        yarn_rotary_scaling_factor=4.0,
        yarn_original_max_position_embeddings=4096,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_mscale=1.0,
        yarn_mscale_all_dim=1.0,
        yarn_correction_range_round_to_int=True,
    )
    config = _make_gpt_config(transformer=transformer, position_embedding_type="yarn")

    validate_dual_chunk_gpt_config(config)
    assert config.transformer.yarn_rotary_scaling_factor == 4.0


def test_legacy_provider_uses_dca_layer_spec_by_default() -> None:
    provider = DualChunkGPTModelProvider(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=4,
        vocab_size=128,
        position_embedding_type="rope",
        dca_chunk_size=16,
        dca_local_size=4,
        apply_rope_fusion=False,
        use_cpu_initialization=True,
        transformer_impl="local",
    )

    assert provider.transformer_layer_spec is get_dca_gpt_layer_spec
    spec = get_dca_gpt_layer_spec(provider)
    assert spec.layer_specs[0].submodules.self_attention.module is DualChunkSelfAttention


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"position_embedding_type": "learned_absolute"}, "position_embedding_type"),
        ({"position_embedding_type": "yarn"}, "DCA with YARN requires"),
        ({"restore_modelopt_state": True}, "restore_modelopt_state"),
    ],
)
def test_gpt_config_validation_rejects_unsupported_model_settings(
    override: dict[str, object],
    message: str,
) -> None:
    config = _make_gpt_config(**override)
    with pytest.raises(ValueError, match=message):
        validate_dual_chunk_gpt_config(config)


def test_gpt_config_validation_rejects_local_attention_backend() -> None:
    transformer = _make_transformer_config(attention_backend=AttnBackend.local)
    config = _make_gpt_config(transformer=transformer)
    with pytest.raises(ValueError, match="attention_backend=local"):
        validate_dual_chunk_gpt_config(config)
