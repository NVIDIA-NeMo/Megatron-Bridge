# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import patch

import pytest
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnBackend
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

import megatron.bridge.models.qwen_omni.qwen3_omni_provider as qwen3_omni_provider
from megatron.bridge.models.qwen_omni import Qwen3OmniModelProvider
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention


pytestmark = pytest.mark.unit


class TestQwen3OmniModelProvider:
    def test_qwen3_omni_model_provider_initialization(self):
        provider = Qwen3OmniModelProvider(
            num_layers=48,
            hidden_size=2048,
            num_attention_heads=32,
        )

        assert provider.num_layers == 48
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 32
        assert provider.position_embedding_type == "mrope"
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.qk_layernorm is True
        assert provider.masked_softmax_fusion is False
        assert provider.apply_rotary_pos_emb_in_fp32 is False
        assert provider.image_token_id == 151655
        assert provider.video_token_id == 151656
        assert provider.audio_token_id == 151646
        assert provider.vision_start_token_id == 151652

    def test_qwen3_omni_custom_thinker_config(self):
        thinker_config = Qwen3OmniMoeThinkerConfig(
            text_config={
                "num_hidden_layers": 2,
                "hidden_size": 128,
                "intermediate_size": 256,
                "moe_intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "num_experts": 8,
                "num_experts_per_tok": 2,
                "vocab_size": 1000,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-6,
            }
        )
        provider = Qwen3OmniModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            thinker_config=thinker_config,
        )

        assert provider.thinker_config.text_config.hidden_size == 128

    def test_qwen3_omni_freeze_flags(self):
        provider = Qwen3OmniModelProvider(
            num_layers=48,
            hidden_size=2048,
            num_attention_heads=32,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_audio_model=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_audio_model is True

    def test_provide_defaults_to_single_stage_pre_post_process(self):
        provider = Qwen3OmniModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=32,
            vocab_size=1024,
            use_cpu_initialization=True,
            bf16=False,
        )

        with patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.Qwen3OmniModel") as mock_model_cls:
            provider.provide()

        _, kwargs = mock_model_cls.call_args
        assert kwargs["pre_process"] is True
        assert kwargs["post_process"] is True

    def test_provide_uses_qwen3_vl_attention_with_local_backend(self):
        provider = Qwen3OmniModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=32,
            vocab_size=1024,
            use_cpu_initialization=True,
            bf16=False,
            attention_backend=AttnBackend.local,
        )

        with patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.Qwen3OmniModel") as mock_model_cls:
            provider.provide()

        _, kwargs = mock_model_cls.call_args
        language_spec = kwargs["language_transformer_layer_spec"]
        self_attention_spec = language_spec.submodules.self_attention
        assert self_attention_spec.module is Qwen3VLSelfAttention
        assert self_attention_spec.submodules.core_attention is DotProductAttention

    def test_string_local_backend_uses_local_spec_when_te_is_available(self):
        provider = Qwen3OmniModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=32,
            vocab_size=1024,
            use_cpu_initialization=True,
            bf16=False,
            attention_backend="local",
        )

        with (
            patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.HAVE_TE", True),
            patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.Qwen3OmniModel") as mock_model_cls,
        ):
            provider.provide()

        _, kwargs = mock_model_cls.call_args
        self_attention_spec = kwargs["language_transformer_layer_spec"].submodules.self_attention
        assert self_attention_spec.module is Qwen3VLSelfAttention
        assert self_attention_spec.submodules.core_attention is DotProductAttention

    def test_default_backend_keeps_te_core_attention_when_available(self):
        if not qwen3_omni_provider.HAVE_TE:
            pytest.skip("Transformer Engine is not available")

        provider = Qwen3OmniModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=32,
            vocab_size=1024,
            use_cpu_initialization=True,
            bf16=False,
        )

        with patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.Qwen3OmniModel") as mock_model_cls:
            provider.provide()

        _, kwargs = mock_model_cls.call_args
        self_attention_spec = kwargs["language_transformer_layer_spec"].submodules.self_attention
        assert self_attention_spec.module is Qwen3VLSelfAttention
        assert self_attention_spec.submodules.core_attention.__name__ == "TEDotProductAttention"

    def test_default_backend_falls_back_to_local_spec_without_te(self):
        provider = Qwen3OmniModelProvider(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=32,
            vocab_size=1024,
            use_cpu_initialization=True,
            bf16=False,
        )

        with (
            patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.HAVE_TE", False),
            patch("megatron.bridge.models.qwen_omni.qwen3_omni_provider.Qwen3OmniModel") as mock_model_cls,
        ):
            provider.provide()

        _, kwargs = mock_model_cls.call_args
        self_attention_spec = kwargs["language_transformer_layer_spec"].submodules.self_attention
        assert self_attention_spec.module is Qwen3VLSelfAttention
        assert self_attention_spec.submodules.core_attention is DotProductAttention
