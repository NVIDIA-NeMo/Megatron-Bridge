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
Qwen3 VL Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen3-VL multimodal models,
compatible with HuggingFace's Qwen3-VL model configurations.
Reference: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
"""

from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy
from functools import partial
import torch.nn.functional as F

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.bridge.models import Qwen3ModelProvider
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from megatron.bridge.models.qwen_3_vl.model import Qwen3VLModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


@dataclass
class Qwen3VLModelProvider(Qwen3ModelProvider):
    """
    Base model provider for Qwen 3 VL Models.
    Inherits language model configuration from Qwen3ModelProvider.
    
    Note: num_query_groups in parent class corresponds to num_key_value_heads in HF config.
    Default value of 8 is used for GQA (Grouped Query Attention).
    """
    head_dim: int = 128
    hidden_size: int = 2048
    
    # Fields from Qwen3VLTransformerConfig
    language_max_sequence_length: int = 2048
    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2304
    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    fp16_lm_cross_entropy: bool = False
    rotary_percent: float = 1.0
    apply_rope_fusion: bool = False
    
    # Vision configuration using the transformers Qwen3VLVisionConfig
    # Default configuration matches the standard Qwen3VL vision encoder
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig())
    
    
    # Vision-specific token IDs matching Qwen3VL configuration
    # Based on https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
    # Token ID for image placeholder in text
    image_token_id: int = 151655
    # Token ID for video placeholder in text
    video_token_id: int = 151656
    # Token ID marking start of vision content
    vision_start_token_id: int = 151652
    # Token ID marking end of vision content
    vision_end_token_id: int = 151653
    # BOS token ID for Qwen3-VL models
    bos_token_id: int = 151643
    # EOS token ID for Qwen3-VL models
    eos_token_id: int = 151645
    
    # Override position embedding for multimodal rope
    position_embedding_type: str = "mrope"
    
    # Multimodal rope section for [temporal, height, width] dimensions
    # Based on HuggingFace Qwen3-VL config: mrope_section: [24, 20, 20]
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    
    # RoPE theta value specific to Qwen3-VL models
    # From HuggingFace config: rope_theta: 5000000
    rotary_base: float = 5000000.0
    
    # Override to disable scattering embeddings for vision insertion
    scatter_embedding_sequence_parallel: bool = False
    
    # Freeze options for fine-tuning scenarios
    # Whether to freeze language model weights
    freeze_language_model: bool = False
    # Whether to freeze vision encoder weights
    freeze_vision_model: bool = False
    # Whether to freeze vision-to-language projection weights
    freeze_vision_projection: bool = False

    sequence_parallel: bool = False 
    
    # QK layernorm is already True in Qwen3ModelProvider, no need to redefine
    # qk_layernorm: bool = True  # Already defined in parent
    
    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """
        Provide a Qwen3VL model instance with vision and language components.
        """
        language_transformer_config = self

        hf_vision_config = self.vision_config
            
        # Spec for the Qwen3VLTransformerLayer
        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
        )
        
        model = Qwen3VLModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=hf_vision_config,
            pre_process=pre_process,
            post_process=post_process
        )
        
        # Apply freeze options if any are enabled for fine-tuning
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )
        
        return model
    
    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """
        Provide just the language model component without vision.
        
        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism  
            vp_stage: Virtual pipeline stage number
            
        Returns:
            MCoreGPTModel instance (language model only)
        """
        # Use parent class to create standard language model
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)