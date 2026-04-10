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

"""
Qwen2.5 Omni Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen2.5 Omni multimodal models
(audio+vision+text), compatible with HuggingFace's Qwen2.5-Omni model configurations.
Reference: https://huggingface.co/Qwen/Qwen2.5-Omni-7B
"""

from dataclasses import dataclass, field

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.model import Qwen25OmniModel


@dataclass
class Qwen25OmniModelProvider(GPTModelProvider):
    """
    Base model provider for Qwen2.5 Omni Models.

    Key differences from Qwen3OmniMoeModelProvider:
    - Dense LLM (Qwen2), not MoE
    - Has QKV bias (Qwen2 specific), no QK layernorm
    - mrope_section: [16, 24, 24] (not [24, 20, 20])
    - position_id_per_seconds: 25 (not 13)
    - seconds_per_chunk: 2 for audio-in-video
    - patch_size: 14 (not 16)
    - Uses HF vision model directly (ReplicatedMapping)
    """

    thinker_config: Qwen2_5OmniThinkerConfig = field(default_factory=lambda: Qwen2_5OmniThinkerConfig())
    talker_config: Qwen2_5OmniTalkerConfig | None = None
    token2wav_config: Qwen2_5OmniToken2WavConfig | None = None

    pretrained_model_name: str = "Qwen/Qwen2.5-Omni-7B"

    # Token IDs matching Qwen2.5-Omni configuration
    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151646
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    head_dim: int = 128
    add_qkv_bias: bool = True
    qk_layernorm: bool = False
    attention_softmax_in_fp32: bool = True
    attention_dropout: float = 0.0

    position_embedding_type: str = "mrope"
    apply_rotary_pos_emb_in_fp32: bool = False
    mrope_section: list[int] = field(default_factory=lambda: [16, 24, 24])
    rotary_base: float = 1000000
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    patch_size: int = 14

    scatter_embedding_sequence_parallel: bool = False

    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_audio_model: bool = False
    language_max_sequence_length: int = 2048

    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = False
    deallocate_pipeline_outputs: bool = True
    async_tensor_model_parallel_allreduce: bool = True
    distribute_saved_activations: bool = False
    cp_comm_type: str = "p2p"
    gradient_accumulation_fusion: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Provide a Qwen2.5 Omni model instance with vision, audio, and language components."""
        language_transformer_config = self
        thinker_config = self.thinker_config
        talker_config = self.talker_config
        token2wav_config = self.token2wav_config

        # Dense GPT layer spec (no MoE, no QK layernorm for Qwen2)
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=self.qk_layernorm,
            fp8=False,
        )

        model = Qwen25OmniModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            thinker_transformer_config=thinker_config,
            talker_transformer_config=talker_config,
            token2wav_transformer_config=token2wav_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_audio_model:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_audio_model=self.freeze_audio_model,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Provide just the language model component without vision/audio."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
