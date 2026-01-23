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

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
import torch.nn.functional as F
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig


@dataclass
class Qwen3VLTransformerConfig(TransformerConfig):
    """Configuration for Qwen3-VL transformer with vision and language components."""

    vocab_size: int = 64000
    language_max_sequence_length: int = 4096

    patch_size: int = 14
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    num_position_embeddings: int = 2304
    out_hidden_size: int = 2304

    apply_rotary_pos_emb_in_fp32: bool = False
    deepstack_visual_indexes: List[int] = field(default_factory=lambda: [8, 16, 24])
    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0
    rotary_base: float = 10000

    # Multimodal rope section for [temporal, height, width] dimensions
    mrope_section: List[int] = field(default_factory=lambda: [24, 20, 20])
    apply_rope_fusion: bool = False

    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    hf_text_config: Optional[Qwen3VLTextConfig] = None


def get_vision_model_config(config: Qwen3VLTransformerConfig, hf_config):
    config.num_moe_experts = None
    config.expert_model_parallel_size = 1
    config.moe_ffn_hidden_size = None

    config.num_layers = hf_config.depth
    config.ffn_hidden_size = hf_config.intermediate_size
    config.num_attention_heads = hf_config.num_heads  # num_heads
    config.add_bias_linear = True  # all nn.Linear has bias (MLP, attn)
    config.add_qkv_bias = True  # qkv_proj in attn has bias
    config.hidden_size = hf_config.hidden_size  # hidden_size
    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.layernorm_epsilon = 1e-6
    config.apply_rotary_pos_emb_in_fp32 = True

    config.patch_size = hf_config.patch_size
    config.temporal_patch_size = hf_config.temporal_patch_size
    config.in_channels = hf_config.in_channels
    config.spatial_merge_size = hf_config.spatial_merge_size
    config.num_position_embeddings = hf_config.num_position_embeddings
    config.out_hidden_size = hf_config.out_hidden_size
    config.deepstack_visual_indexes = deepcopy(hf_config.deepstack_visual_indexes)

    config.gated_linear_unit = False  # no gated
    config.activation_func = partial(F.gelu, approximate="tanh")  # hidden_act
    config.kv_channels = config.hidden_size // config.num_attention_heads
    config.num_query_groups = config.num_attention_heads  # no GQA
    config.layernorm_zero_centered_gamma = False  # False
    config.apply_query_key_layer_scaling = False  # factor=math.sqrt(head_dim)
    config.bias_activation_fusion = False  # no swiglu, set false
    config.bias_dropout_fusion = False  # no dropout, set false
    config.attention_softmax_in_fp32 = True  # use True
    config.normalization = "LayerNorm"

    config.tp_comm_overlap = False
    config.sequence_parallel = False
    config.context_parallel_size = 1
    config.pipeline_model_parallel_size = 1
    config.num_layers_in_first_pipeline_stage = None
    config.num_layers_in_last_pipeline_stage = None
    config.virtual_pipeline_model_parallel_size = 1
    config.pipeline_model_parallel_layout = None
    config.account_for_embedding_in_pipeline_split = None
    config.account_for_loss_in_pipeline_split = None
    return config
