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
Qwen3 Omni MoE Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen3 Omni MoE (Mixture of Experts) multimodal models,
compatible with HuggingFace's Qwen3-Omni-MoE model configurations.
Reference: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Thinking
"""

from dataclasses import dataclass, field
from typing import List

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTextConfig,
    Qwen3OmniMoeThinkerConfig,
)

from megatron.bridge.models import Qwen3MoEModelProvider
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniMoeModel


@dataclass
class Qwen3OmniMoeModelProvider(Qwen3MoEModelProvider):
    """
    Base model provider for Qwen3 Omni Moe Models.
    Inherits language model configuration from Qwen3MoeModelProvider.

    Key MoE Parameters (inherited from Qwen3MoEModelProvider):
    - num_moe_experts: Number of total experts (default 128)
    - moe_router_topk: Number of experts selected per token (default 8)
    - moe_router_load_balancing_type: Load balancing strategy (default "aux_loss")
    - moe_aux_loss_coeff: Auxiliary loss coefficient (default 1e-3)
    - moe_grouped_gemm: Use grouped GEMM for efficiency (default True)

    Note: num_query_groups in parent class corresponds to num_key_value_heads in HF config.
    """

    thinker_config: Qwen3OmniMoeThinkerConfig = field(default_factory=lambda: Qwen3OmniMoeThinkerConfig())
    talker_config: Qwen3OmniMoeTalkerConfig | None = None
    code2wav_config: Qwen3OmniMoeCode2WavConfig | None = None
    hf_text_config: Qwen3OmniMoeTextConfig | None = None

    pretrained_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    # Vision-specific token IDs matching Qwen3-Omni-MoE configuration
    # Based on HuggingFace Qwen3-Omni-MoE configs
    # Token ID for image placeholder in text
    image_token_id: int = 151655
    # Token ID for video placeholder in text
    video_token_id: int = 151656
    # Token ID for audio placeholder in text
    audio_token_id: int = 151675
    # Token ID marking start of vision content
    vision_start_token_id: int = 151652
    # Token ID marking end of vision content
    vision_end_token_id: int = 151653
    # Token ID marking start of audio content
    audio_start_token_id: int = 151669
    # Token ID marking end of audio content
    audio_end_token_id: int = 151670
    # BOS token ID for Qwen3-Omni models
    bos_token_id: int = 151643
    # EOS token ID for Qwen3-Omni models
    eos_token_id: int = 151645
    tts_bos_token_id: int = 151672
    tts_eos_token_id: int = 151673
    tts_pad_token_id: int = 151671

    head_dim: int = 128
    qk_layernorm: bool = True
    attention_softmax_in_fp32: bool = True
    attention_dropout: float = 0.0

    # Override position embedding for multimodal rope
    position_embedding_type: str = "mrope"

    apply_rotary_pos_emb_in_fp32: bool = False

    # Multimodal rope section for [temporal, height, width] dimensions
    # Based on HuggingFace Qwen3-Omni config: mrope_section: [24, 20, 20]
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])

    # RoPE theta value specific to Qwen3-Omni models
    # From HuggingFace config: rope_theta: 1000000
    rotary_base: float = 1000000
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    patch_size: int = 16

    # Override to disable scattering embeddings for vision insertion
    scatter_embedding_sequence_parallel: bool = False

    # Router configuration
    moe_router_pre_softmax: bool = False  # Qwen3 specific
    moe_router_dtype: str = "fp32"  # Use FP32 for router computations
    moe_router_score_function: str = "softmax"  # Softmax scoring
    moe_router_bias_update_rate: float = 0.001  # Router bias update rate

    # MoE optimization settings
    moe_permute_fusion: bool = True  # Fuse permutation operations
    moe_token_dispatcher_type: str = "alltoall"  # All-to-all communication

    # Dense layers configuration (some layers may not use MoE)
    # Empty list means all layers use MoE, otherwise specify layer indices
    mlp_only_layers: List[int] = field(default_factory=list)

    # Decoder sparse step (frequency of MoE layers)
    decoder_sparse_step: int = 1  # Every layer is MoE by default

    # Freeze options for fine-tuning scenarios
    # Whether to freeze language model weights
    freeze_language_model: bool = False
    # Whether to freeze vision encoder weights
    freeze_vision_model: bool = False
    # Whether to freeze vision-to-language projection weights
    freeze_vision_projection: bool = False
    # Whether ro freeze audio model weights
    freeze_audio_model: bool = False
    language_max_sequence_length: int = 2048

    # QK layernorm is already True in Qwen3MoEModelProvider, no need to redefine

    # These are typically set in the base class but documented here for clarity
    persist_layer_norm: bool = True  # Persist layer norm for efficiency
    bias_activation_fusion: bool = True  # Fuse bias and activation
    bias_dropout_fusion: bool = True  # Fuse bias and dropout
    masked_softmax_fusion: bool = False  # Don't fuse masked softmax (Qwen specific)
    deallocate_pipeline_outputs: bool = True  # Deallocate pipeline outputs to save memory
    async_tensor_model_parallel_allreduce: bool = True  # Async tensor parallel
    distribute_saved_activations: bool = False  # Don't distribute saved activations
    cp_comm_type: str = "p2p"  # Point-to-point communication for context parallel
    position_id_per_seconds: int = 13

    use_hf_vision_model: bool = False
    vision_dp_when_cp: bool = False

    def finalize(self) -> None:
        if self.tensor_model_parallel_size > 1:
            self.sequence_parallel = True

        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """
        Provide a Qwen3 Omni MoE model instance with vision and language components.
        """
        language_transformer_config = self

        # Create vision transformer config - placeholder for future use
        # vision_transformer_config = deepcopy(self)
        thinker_config = self.thinker_config
        talker_config = self.talker_config
        code2wav_config = self.code2wav_config

        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.num_moe_experts,
            moe_grouped_gemm=True,
            qk_layernorm=self.qk_layernorm,
            fp8=False,
        )

        # reuse Qwen3OmniMoeAudioEncoder and Qwen3OmniMoeVisionEncoder for MoE model but replace the language model with MoE language model
        model = Qwen3OmniMoeModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            thinker_transformer_config=thinker_config,
            talker_transformer_config=talker_config,
            code2wav_transformer_config=code2wav_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
        )

        # Apply freeze options if any are enabled for fine-tuning
        if (
            self.freeze_language_model
            or self.freeze_vision_model
            or self.freeze_vision_projection
            or self.freeze_audio_model
        ):
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
                freeze_audio_model=self.freeze_audio_model,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """
        Provide just the language MoE model component without vision.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            MCoreGPTModel instance (MoE language model only)
        """
        # Use parent class to create standard MoE language model
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
