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
Qwen3 VL MoE Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen3-VL MoE (Mixture of Experts) multimodal models,
compatible with HuggingFace's Qwen3-VL-MoE model configurations.
Reference: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
"""

from dataclasses import dataclass, field
from typing import List, Optional
from copy import deepcopy
from functools import partial
import torch.nn.functional as F

import torch

# Import from Megatron-Core (available at runtime in the Megatron environment)
from megatron.core.models.gpt import GPTModel as MCoreGPTModel

# Import the Qwen3 MoE base model provider
from megatron.bridge.models import Qwen3MoEModelProvider

# Import vision config from transformers library
# This requires transformers to be installed with Qwen3VL support
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

# Placeholder for the Qwen3VL MoE model class - implement similarly to Qwen25VLModel
# from .modeling_qwen3_vl_moe import Qwen3VLMoEModel
from megatron.bridge.models.qwen_3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_3_vl.transformer_block import Qwen3VLTransformerBlock as Qwen3VLMoETransformerLayer  #fix me
from megatron.core.transformer.spec_utils import ModuleSpec


# =============================================================================
# Qwen 3 VL MoE Model Providers
# =============================================================================


@dataclass
class Qwen3VLMoEModelProvider(Qwen3MoEModelProvider):
    """
    Base model provider for Qwen 3 VL MoE Models.
    Inherits language model MoE configuration from Qwen3MoEModelProvider.
    
    Key MoE Parameters (inherited from Qwen3MoEModelProvider):
    - num_moe_experts: Number of total experts (default 128)
    - moe_router_topk: Number of experts selected per token (default 8)
    - moe_router_load_balancing_type: Load balancing strategy (default "aux_loss")
    - moe_aux_loss_coeff: Auxiliary loss coefficient (default 1e-3)
    - moe_grouped_gemm: Use grouped GEMM for efficiency (default True)
    
    Note: num_query_groups in parent class corresponds to num_key_value_heads in HF config.
    """

    # Vision configuration using the transformers Qwen3VLVisionConfig
    # Default configuration matches the standard Qwen3VL vision encoder
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig())
    
    # Vision-specific token IDs matching Qwen3VL MoE configuration
    # Based on HuggingFace Qwen3-VL-MoE configs
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
    
    # MoE-specific configurations from yan-mbridge
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
    
    # QK layernorm is already True in Qwen3MoEModelProvider, no need to redefine
    
    # Additional MoE optimizations from yan-mbridge
    # These are typically set in the base class but documented here for clarity
    persist_layer_norm: bool = True  # Persist layer norm for efficiency
    bias_activation_fusion: bool = True  # Fuse bias and activation
    bias_dropout_fusion: bool = True  # Fuse bias and dropout
    masked_softmax_fusion: bool = False  # Don't fuse masked softmax (Qwen specific)
    deallocate_pipeline_outputs: bool = True  # Deallocate pipeline outputs to save memory
    async_tensor_model_parallel_allreduce: bool = True  # Async tensor parallel
    distribute_saved_activations: bool = False  # Don't distribute saved activations
    cp_comm_type: str = "p2p"  # Point-to-point communication for context parallel
    
    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """
        Provide a Qwen3VL MoE model instance with vision and language components.
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
        
        # Spec for the Qwen3VLMoETransformerLayer
        language_transformer_layer_spec = ModuleSpec(module=Qwen3VLMoETransformerLayer)
        
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


# =============================================================================
# Qwen 3 VL MoE Model Size Configurations
# =============================================================================


@dataclass
class Qwen3VLMoEModelProvider30B_A3B(Qwen3VLMoEModelProvider):
    """
    Config for Qwen 3 VL 30B-A3B MoE model.
    Language config from Qwen3MoEModelProvider30B_A3B.
    Reference: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct
    
    This model has 30B total parameters with 3B active parameters per token.
    """
    # Language model configuration from Qwen3MoEModelProvider30B_A3B
    # Verified against HuggingFace config
    num_layers: int = 48
    hidden_size: int = 2048
    num_attention_heads: int = 32
    num_query_groups: int = 4  # num_key_value_heads=4 in HF config (GQA)
    ffn_hidden_size: int = 6144  # Dense FFN size
    moe_ffn_hidden_size: int = 768  # Expert FFN size (moe_intermediate_size)
    
    # MoE specific parameters for 30B-A3B model
    # Based on configuration_qwen3_vl_moe.py defaults
    num_moe_experts: int = 60  # Total number of experts
    moe_router_topk: int = 4  # num_experts_per_tok in HF config
    decoder_sparse_step: int = 1  # Every layer is MoE
    
    # Shared embeddings for smaller model
    share_embeddings_and_output_weights: bool = True
    
    # Vision configuration adjusted for 30B-A3B model
    # Using transformers Qwen3VLVisionConfig with custom parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=27,  # Standard depth
        hidden_size=1152,  # Standard vision hidden size
        intermediate_size=4304,  # Standard intermediate size
        num_heads=16,  # Standard attention heads
        out_hidden_size=2048,  # Match language hidden size
    ))
    
    # Override some MoE defaults for this model size
    moe_aux_loss_coeff: float = 1e-3  # Auxiliary loss for load balancing


@dataclass
class Qwen3VLMoEModelProvider235B_A22B(Qwen3VLMoEModelProvider):
    """
    Config for Qwen 3 VL 235B-A22B MoE model.
    Language config from Qwen3MoEModelProvider235B_A22B.
    
    This model has 235B total parameters with 22B active parameters per token.
    Note: This is a hypothetical VL variant as no official model exists yet.
    """
    # Language model configuration from Qwen3MoEModelProvider235B_A22B
    num_layers: int = 94
    hidden_size: int = 4096
    num_attention_heads: int = 64
    num_query_groups: int = 4  # num_key_value_heads=4 in base config (GQA)
    ffn_hidden_size: int = 12288  # Dense FFN size
    moe_ffn_hidden_size: int = 1536  # Expert FFN size
    
    # MoE specific parameters for 235B-A22B model
    num_moe_experts: int = 128  # Total number of experts (default from base)
    moe_router_topk: int = 8  # num_experts_per_tok (default from base)
    decoder_sparse_step: int = 1  # Every layer is MoE
    
    # No shared embeddings for large model
    share_embeddings_and_output_weights: bool = False
    
    # Vision configuration for large model
    # Using transformers Qwen3VLVisionConfig with enhanced parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=32,  # Increased depth for larger model
        hidden_size=1536,  # Larger vision hidden size
        intermediate_size=6144,  # Larger intermediate size
        num_heads=24,  # More attention heads
        out_hidden_size=4096,  # Match language hidden size
    ))
    
    # Override some MoE defaults for this model size
    moe_aux_loss_coeff: float = 1e-3  # Auxiliary loss for load balancing


@dataclass
class Qwen3VLMoEModelProvider48B_A8B(Qwen3VLMoEModelProvider):
    """
    Config for a hypothetical Qwen 3 VL 48B-A8B MoE model.
    Custom configuration between 30B and 235B models.
    
    This model would have 48B total parameters with 8B active parameters per token.
    """
    # Custom language model configuration
    num_layers: int = 56
    hidden_size: int = 3072
    num_attention_heads: int = 48
    num_query_groups: int = 6  # num_key_value_heads for GQA
    ffn_hidden_size: int = 8192  # Dense FFN size
    moe_ffn_hidden_size: int = 1024  # Expert FFN size
    
    # MoE specific parameters
    num_moe_experts: int = 64  # Total number of experts
    moe_router_topk: int = 4  # num_experts_per_tok
    decoder_sparse_step: int = 1  # Every layer is MoE
    
    # Shared embeddings for medium model
    share_embeddings_and_output_weights: bool = True
    
    # Vision configuration for medium model
    # Using transformers Qwen3VLVisionConfig with balanced parameters
    vision_config: Qwen3VLVisionConfig = field(default_factory=lambda: Qwen3VLVisionConfig(
        depth=30,  # Medium depth
        hidden_size=1280,  # Medium vision hidden size
        intermediate_size=5120,  # Medium intermediate size
        num_heads=20,  # Balanced attention heads
        out_hidden_size=3072,  # Match language hidden size
    ))
    
    # Override some MoE defaults for this model size
    moe_aux_loss_coeff: float = 1e-3  # Auxiliary loss for load balancing
