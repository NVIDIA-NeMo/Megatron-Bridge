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

from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.bridge.models import Qwen3MoEModelProvider    
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl_moe.configuration_qwen3_vl_moe import Qwen3VLMoETextConfig

from megatron.bridge.models.qwen_3_vl.model import Qwen3VLModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

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
    

    hf_text_config: Optional[Qwen3VLMoETextConfig] = None
    
    pretrained_model_name: str = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    
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

    head_dim: int = 128
    qk_layernorm: bool = True
    
    # Override position embedding for multimodal rope
    position_embedding_type: str = "mrope"
    
    # Multimodal rope section for [temporal, height, width] dimensions
    # Based on HuggingFace Qwen3-VL config: mrope_section: [24, 20, 20]
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    
    # RoPE theta value specific to Qwen3-VL models
    # From HuggingFace config: rope_theta: 5000000
    rotary_base: float = 5000000.0
    sequence_parallel: bool = False
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    patch_size: int = 16
    
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
    language_max_sequence_length: int = 2048
    
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

        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.num_moe_experts,
            moe_grouped_gemm=False,
            qk_layernorm=self.qk_layernorm,
            fp8=False,
        )
        
        model = Qwen3VLMoEModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=hf_config,
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
