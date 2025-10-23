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
    
    pretrained_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    
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

    sequence_parallel: bool = True
    
    # QK layernorm is already True in Qwen3ModelProvider, no need to redefine
    # qk_layernorm: bool = True  # Already defined in parent
    
    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """
        Provide a Qwen3VL model instance with vision and language components.
        """
        language_transformer_config = self

        # Create vision transformer config
        vision_transformer_config = deepcopy(self)
        hf_config = self.vision_config

        # Set vision-specific parameters from hf_config
        vision_transformer_config.num_layers = hf_config.depth
        vision_transformer_config.ffn_hidden_size = hf_config.intermediate_size
        vision_transformer_config.num_attention_heads = hf_config.num_heads
        vision_transformer_config.hidden_size = hf_config.hidden_size
        if hasattr(hf_config, 'patch_size'):
            vision_transformer_config.patch_size = hf_config.patch_size
        if hasattr(hf_config, 'temporal_patch_size'):
            vision_transformer_config.temporal_patch_size = hf_config.temporal_patch_size
        if hasattr(hf_config, 'in_channels'):
            vision_transformer_config.in_channels = hf_config.in_channels
        if hasattr(hf_config, 'spatial_merge_size'):
            vision_transformer_config.spatial_merge_size = hf_config.spatial_merge_size
        if hasattr(hf_config, 'num_position_embeddings'):
            vision_transformer_config.num_position_embeddings = hf_config.num_position_embeddings
        if hasattr(hf_config, 'out_hidden_size'):
            vision_transformer_config.out_hidden_size = hf_config.out_hidden_size
        if hasattr(hf_config, 'deepstack_visual_indexes'):
            vision_transformer_config.deepstack_visual_indexes = deepcopy(
                hf_config.deepstack_visual_indexes
            )

        # Set other vision-specific settings
        vision_transformer_config.add_bias_linear = True
        vision_transformer_config.add_qkv_bias = True
        vision_transformer_config.hidden_dropout = 0.0
        vision_transformer_config.attention_dropout = 0.0
        vision_transformer_config.layernorm_epsilon = 1e-6
        vision_transformer_config.apply_rotary_pos_emb_in_fp32 = True
        vision_transformer_config.gated_linear_unit = False
        vision_transformer_config.activation_func = partial(F.gelu, approximate="tanh")
        vision_transformer_config.kv_channels = (
            vision_transformer_config.hidden_size // vision_transformer_config.num_attention_heads
        )
        vision_transformer_config.num_query_groups = vision_transformer_config.num_attention_heads
        vision_transformer_config.layernorm_zero_centered_gamma = False
        vision_transformer_config.apply_query_key_layer_scaling = False
        vision_transformer_config.bias_activation_fusion = False
        vision_transformer_config.bias_dropout_fusion = False
        vision_transformer_config.attention_softmax_in_fp32 = True
        vision_transformer_config.normalization = "LayerNorm"

        # Disable MoE for vision encoder
        if hasattr(vision_transformer_config, 'num_moe_experts'):
            vision_transformer_config.num_moe_experts = None
        if hasattr(vision_transformer_config, 'expert_model_parallel_size'):
            vision_transformer_config.expert_model_parallel_size = 1
        if hasattr(vision_transformer_config, 'moe_ffn_hidden_size'):
            vision_transformer_config.moe_ffn_hidden_size = None

        # Set parallelism settings for vision
        vision_transformer_config.tp_comm_overlap = False
        vision_transformer_config.sequence_parallel = False
        vision_transformer_config.context_parallel_size = 1
        vision_transformer_config.pipeline_model_parallel_size = 1
        vision_transformer_config.num_layers_in_first_pipeline_stage = None
        vision_transformer_config.num_layers_in_last_pipeline_stage = None
        
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
            vision_transformer_config=vision_transformer_config,
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



@dataclass
class Qwen3VLModelProvider600M(Qwen3VLModelProvider):
    """
    Config for Qwen 3 VL 0.6B model.
    Language config from Qwen3ModelProvider600M.
    """
    # Language model configuration from Qwen3ModelProvider600M
    num_layers: int = 28
    hidden_size: int = 1024
    num_attention_heads: int = 16
    # num_query_groups defaults to 8 from parent (corresponds to num_key_value_heads)
    ffn_hidden_size: int = 3072
    share_embeddings_and_output_weights: bool = True
    
    # Vision configuration adjusted for smaller model
    # Using transformers Qwen3VLVisionConfig with custom parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=18,  # Reduced depth for smaller model
        hidden_size=768,  # Smaller vision hidden size
        intermediate_size=3072,  # Smaller intermediate size
        num_heads=12,  # Fewer attention heads
        out_hidden_size=1024,  # Match language hidden size
    ))


@dataclass
class Qwen3VLModelProvider1P7B(Qwen3VLModelProvider):
    """
    Config for Qwen 3 VL 1.7B model.
    Language config from Qwen3ModelProvider1P7B.
    """
    # Language model configuration from Qwen3ModelProvider1P7B
    num_layers: int = 28
    hidden_size: int = 2048
    num_attention_heads: int = 16
    # num_query_groups defaults to 8 from parent (corresponds to num_key_value_heads)
    ffn_hidden_size: int = 6144
    share_embeddings_and_output_weights: bool = True
    
    # Vision configuration adjusted for 1.7B model
    # Using transformers Qwen3VLVisionConfig with custom parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=24,  # Medium depth
        hidden_size=1024,  # Medium vision hidden size
        intermediate_size=4096,  # Medium intermediate size
        num_heads=16,  # Standard attention heads
        out_hidden_size=2048,  # Match language hidden size
    ))


@dataclass
class Qwen3VLModelProvider4B(Qwen3VLModelProvider):
    """
    Config for Qwen 3 VL 4B model.
    Language config from Qwen3ModelProvider4B.
    """
    # Language model configuration from Qwen3ModelProvider4B
    num_layers: int = 36
    hidden_size: int = 2560
    num_attention_heads: int = 32
    # num_query_groups defaults to 8 from parent (corresponds to num_key_value_heads)
    ffn_hidden_size: int = 9728
    share_embeddings_and_output_weights: bool = True
    
    # Vision configuration for 4B model
    # Using transformers Qwen3VLVisionConfig with custom parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=27,  # Full depth as in base config
        hidden_size=1152,  # Standard vision hidden size
        intermediate_size=4304,  # Standard intermediate size
        num_heads=16,  # Standard attention heads
        out_hidden_size=2560,  # Match language hidden size
    ))


@dataclass
class Qwen3VLModelProvider8B(Qwen3VLModelProvider):
    """
    Config for Qwen 3 VL 8B model.
    Language config from Qwen3ModelProvider8B.
    """
    # Language model configuration from Qwen3ModelProvider8B
    # Verified against https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json
    num_layers: int = 36
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8  # num_key_value_heads in HF config (GQA configuration)
    ffn_hidden_size: int = 12288
    # Note: share_embeddings_and_output_weights defaults to False for 8B

    pretrained_model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    
    # Vision configuration for 8B model
    # Using transformers Qwen3VLVisionConfig with default parameters
    # Default values match HF config: depth=27, hidden_size=1152, intermediate_size=4304, etc.
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        out_hidden_size=4096,  # Match language hidden size
    ))


@dataclass
class Qwen3VLModelProvider14B(Qwen3VLModelProvider):
    """
    Config for Qwen 3 VL 14B model.
    Language config from Qwen3ModelProvider14B.
    """
    # Language model configuration from Qwen3ModelProvider14B
    num_layers: int = 40
    hidden_size: int = 5120
    num_attention_heads: int = 40
    # num_query_groups defaults to 8 from parent (corresponds to num_key_value_heads)
    ffn_hidden_size: int = 17408
    
    # Vision configuration for 14B model
    # Using transformers Qwen3VLVisionConfig with custom parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=32,  # Increased depth for larger model
        hidden_size=1280,  # Larger vision hidden size
        intermediate_size=5120,  # Larger intermediate size
        num_heads=20,  # More attention heads
        out_hidden_size=5120,  # Match language hidden size
    ))


@dataclass
class Qwen3VLModelProvider32B(Qwen3VLModelProvider):
    """
    Config for Qwen 3 VL 32B model.
    Language config from Qwen3ModelProvider32B.
    """
    # Language model configuration from Qwen3ModelProvider32B
    num_layers: int = 64
    hidden_size: int = 5120
    num_attention_heads: int = 64
    # num_query_groups defaults to 8 from parent (corresponds to num_key_value_heads)
    ffn_hidden_size: int = 25600
    
    # Vision configuration for 32B model
    # Using transformers Qwen3VLVisionConfig with custom parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=32,  # Increased depth for larger model
        hidden_size=1280,  # Larger vision hidden size
        intermediate_size=5120,  # Larger intermediate size
        num_heads=20,  # More attention heads
        out_hidden_size=5120,  # Match language hidden size
    ))
