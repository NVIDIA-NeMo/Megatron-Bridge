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
TransformerConfig for the ERNIE 4.5 VL vision encoder (DFN-style ViT with 2D RoPE).

This config inherits from Megatron-Core's TransformerConfig and adds
vision-specific fields (patch_size, spatial_merge_size, etc.).  It is
constructed from the HF vision config via ``get_ernie_vision_config()``.
"""

from dataclasses import dataclass

import torch
from megatron.core.transformer.transformer_config import TransformerConfig


@dataclass
class ErnieVisionTransformerConfig(TransformerConfig):
    """TransformerConfig for ERNIE 4.5 VL vision encoder.

    Extends Megatron-Core TransformerConfig with ERNIE vision-specific fields.

    Architecture constants from HF DFNRopeVisionTransformerConfig:
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        patch_size=14, in_channels=3, spatial_merge_size=2,
        hidden_act="quick_gelu"
    """

    patch_size: int = 14
    """Vision patch size (pixels per side)."""

    in_channels: int = 3
    """Number of input image channels."""

    spatial_merge_size: int = 2
    """Spatial merge factor for the resampler (2x2 pooling)."""


def _quick_gelu(x):
    """Quick GELU activation: x * sigmoid(1.702 * x).

    This is the activation function used by ERNIE 4.5 VL ViT (and OpenAI CLIP).
    It is a fast approximation of GELU but is NOT equivalent to
    ``F.gelu(x, approximate="tanh")``, which uses a different formula:
        0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    The two differ by up to ~2% per element.
    """
    return x * torch.sigmoid(1.702 * x)


def get_ernie_vision_config(
    hf_vision_config,
    megatron_config=None,
) -> ErnieVisionTransformerConfig:
    """Construct an ErnieVisionTransformerConfig from a HF vision config.

    Args:
        hf_vision_config: HF DFNRopeVisionTransformerConfig or equivalent
            with fields: embed_dim, depth, num_heads, mlp_ratio, patch_size,
            in_channels, spatial_merge_size, hidden_act.
        megatron_config: Optional language model TransformerConfig to copy
            recompute / CUDA-graph / TP settings from.

    Returns:
        ErnieVisionTransformerConfig ready for ErnieVLVisionModel.
    """
    embed_dim = getattr(hf_vision_config, "embed_dim",
                        getattr(hf_vision_config, "hidden_size", 1280))
    num_heads = getattr(hf_vision_config, "num_heads",
                        getattr(hf_vision_config, "num_attention_heads", 16))
    mlp_ratio = getattr(hf_vision_config, "mlp_ratio", 4)
    depth = getattr(hf_vision_config, "depth",
                    getattr(hf_vision_config, "num_hidden_layers", 32))

    config = ErnieVisionTransformerConfig(
        num_layers=depth,
        hidden_size=embed_dim,
        num_attention_heads=num_heads,
        ffn_hidden_size=int(embed_dim * mlp_ratio),
        add_bias_linear=True,     # ERNIE ViT: all linear layers have bias=True
        add_qkv_bias=True,        # ERNIE ViT: QKV projection has bias=True
    )

    # Copy parallelism / recompute settings from language model config
    if megatron_config is not None:
        config.recompute_granularity = megatron_config.recompute_granularity
        config.recompute_method = megatron_config.recompute_method
        config.recompute_num_layers = megatron_config.recompute_num_layers
        config.tensor_model_parallel_size = megatron_config.tensor_model_parallel_size
        config.enable_cuda_graph = megatron_config.enable_cuda_graph
        config.cuda_graph_use_single_mempool = megatron_config.cuda_graph_use_single_mempool
        config.cuda_graph_retain_backward_graph = megatron_config.cuda_graph_retain_backward_graph
        config.cuda_graph_warmup_steps = megatron_config.cuda_graph_warmup_steps
        config.external_cuda_graph = megatron_config.external_cuda_graph
        config.cuda_graph_impl = megatron_config.cuda_graph_impl
        config.cuda_graph_scope = megatron_config.cuda_graph_scope

    # Vision encoder specific: no MoE, no EP
    config.num_moe_experts = None
    config.expert_model_parallel_size = 1
    config.moe_ffn_hidden_size = None

    # No dropout in vision encoder
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0

    # LayerNorm with eps=1e-6 (matching HF DFNRopeVisionBlock)
    config.layernorm_epsilon = 1e-6
    config.normalization = "LayerNorm"

    # ERNIE ViT uses quick_gelu: x * sigmoid(1.702 * x)
    # Note: quick_gelu is NOT the same as F.gelu(x, approximate="tanh").
    # F.gelu(tanh) uses: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    # The two differ by up to ~2% per element, which compounds over 32 layers.
    config.activation_func = _quick_gelu
    config.gated_linear_unit = False  # No gated MLP in ViT

    # Derived head dimension
    config.kv_channels = embed_dim // num_heads
    config.num_query_groups = num_heads  # No GQA in ViT

    # Disable various fusions/features not needed for ViT
    config.layernorm_zero_centered_gamma = False
    config.apply_query_key_layer_scaling = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.apply_rope_fusion = False

    # No TP comm overlap or SP for vision encoder
    config.tp_comm_overlap = False
    config.sequence_parallel = False

    # No pipeline parallelism for vision encoder
    config.context_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    config.virtual_pipeline_model_parallel_size = 1
    config.pipeline_model_parallel_layout = None
    config.account_for_embedding_in_pipeline_split = None
    config.account_for_loss_in_pipeline_split = None

    # Vision-specific fields
    config.patch_size = getattr(hf_vision_config, "patch_size", 14)
    config.in_channels = getattr(hf_vision_config, "in_channels", 3)
    config.spatial_merge_size = getattr(hf_vision_config, "spatial_merge_size", 2)

    return config
