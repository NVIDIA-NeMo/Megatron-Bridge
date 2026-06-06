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

import pytest
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gpt.dca_attention import (
    DualChunkAttention,
    DualChunkSelfAttention,
    validate_dual_chunk_attention_config,
)
from megatron.bridge.models.gpt.dca_layer_spec import get_dca_gpt_layer_spec
from megatron.bridge.models.gpt.gpt_builder import GPTModelConfig, default_layer_spec
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.transformer_config import TransformerConfig


pytestmark = pytest.mark.unit


def _make_transformer_config(**kwargs) -> TransformerConfig:
    defaults = dict(
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=4,
        apply_rope_fusion=False,
        use_cpu_initialization=True,
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def _make_gpt_config(**kwargs) -> GPTModelConfig:
    defaults = dict(
        transformer=_make_transformer_config(),
        vocab_size=128,
        position_embedding_type="rope",
        use_dual_chunk_attention=True,
        dca_chunk_size=16,
        dca_local_size=4,
    )
    defaults.update(kwargs)
    return GPTModelConfig(**defaults)


def test_dca_layer_spec_patches_self_attention_and_core_attention() -> None:
    config = _make_gpt_config()

    spec = get_dca_gpt_layer_spec(config, use_transformer_engine=False)

    assert len(spec.layer_specs) == 2
    for layer_spec in spec.layer_specs:
        self_attention_spec = layer_spec.submodules.self_attention
        assert self_attention_spec.module is DualChunkSelfAttention
        assert self_attention_spec.params["dca_chunk_size"] == 16
        assert self_attention_spec.params["dca_local_size"] == 4
        assert self_attention_spec.submodules.core_attention.module is DualChunkAttention
        assert self_attention_spec.submodules.core_attention.params == {
            "dca_chunk_size": 16,
            "dca_local_size": 4,
        }


def test_gpt_model_config_default_layer_spec_uses_dca_when_enabled() -> None:
    config = _make_gpt_config()

    spec = default_layer_spec(config)

    assert spec.layer_specs[0].submodules.self_attention.module is DualChunkSelfAttention


def test_legacy_provider_default_layer_spec_uses_dca_when_enabled() -> None:
    provider = GPTModelProvider(
        num_layers=1,
        hidden_size=32,
        num_attention_heads=4,
        num_query_groups=4,
        vocab_size=128,
        position_embedding_type="rope",
        use_dual_chunk_attention=True,
        dca_chunk_size=16,
        dca_local_size=4,
        apply_rope_fusion=False,
        use_cpu_initialization=True,
    )

    spec = provider.transformer_layer_spec(provider)

    assert spec.layer_specs[0].submodules.self_attention.module is DualChunkSelfAttention


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"dca_chunk_size": 4, "dca_local_size": 4}, "dca_chunk_size must be greater"),
        ({"position_embedding_type": "learned_absolute"}, "position_embedding_type"),
    ],
)
def test_dca_config_validation_rejects_invalid_model_fields(override: dict, message: str) -> None:
    config = _make_gpt_config(**override)

    with pytest.raises(ValueError, match=message):
        validate_dual_chunk_attention_config(config)


def test_dca_config_validation_rejects_local_attention_backend() -> None:
    config = _make_gpt_config()
    config.attention_backend = AttnBackend.local

    with pytest.raises(ValueError, match="attention_backend=local"):
        validate_dual_chunk_attention_config(config)


def test_dca_config_validation_rejects_context_parallelism() -> None:
    config = _make_gpt_config(transformer=_make_transformer_config(context_parallel_size=2))

    with pytest.raises(ValueError, match="context_parallel_size"):
        validate_dual_chunk_attention_config(config)


@pytest.mark.parametrize(
    ("transformer_override", "message"),
    [
        ({"apply_rope_fusion": True}, "apply_rope_fusion"),
        ({"fused_single_qkv_rope": True}, "fused_single_qkv_rope"),
        ({"attention_output_gate": True}, "attention_output_gate"),
        ({"cuda_graph_impl": "full_iteration"}, "CUDA graphs"),
        ({"recompute_granularity": "selective", "recompute_modules": ["core_attn"]}, "core attention recompute"),
    ],
)
def test_dca_config_validation_rejects_invalid_transformer_fields(
    transformer_override: dict,
    message: str,
) -> None:
    config = _make_gpt_config(transformer=_make_transformer_config(**transformer_override))

    with pytest.raises(ValueError, match=message):
        validate_dual_chunk_attention_config(config)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_transformer_engine_full_layer_spec": True}, "full_layer_spec"),
        ({"restore_modelopt_state": True}, "restore_modelopt_state"),
    ],
)
def test_dca_config_validation_rejects_unsupported_layer_spec_modes(override: dict, message: str) -> None:
    config = _make_gpt_config(**override)

    with pytest.raises(ValueError, match=message):
        validate_dual_chunk_attention_config(config)
