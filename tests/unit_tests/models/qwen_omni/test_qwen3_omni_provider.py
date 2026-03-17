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

from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig,
)

from megatron.bridge.models.qwen_omni import Qwen3OmniModelProvider


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
