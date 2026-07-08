# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import List, Optional

import torch
from megatron.core.transformer.transformer_config import TransformerConfig
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config


@dataclass
class Exaone45TransformerConfig(TransformerConfig):
    """Configuration for the EXAONE 4.5 transformer with vision and language components."""

    vocab_size: int = 153600
    language_max_sequence_length: int = 131072

    patch_size: int = 16
    temporal_patch_size: int = 2
    in_channels: int = 3
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048

    apply_rotary_pos_emb_in_fp32: bool = False
    fullatt_block_indexes: List[int] = field(default_factory=lambda: [6, 13, 20, 27])

    fp16_lm_cross_entropy: bool = False
    share_embeddings_and_output_weights: bool = False
    rotary_percent: float = 1.0

    apply_rope_fusion: bool = False

    image_token_id: int = 67
    video_token_id: int = 68
    vision_start_token_id: int = 73
    hf_text_config: Optional[Exaone4_5_Config] = None
    use_hf_vision_model: bool = False


def get_vision_model_config(hf_config, megatron_config=None):
    """
    Get the vision model config for Exaone45 vision model.
    """
    # init config from scratch to avoid deepcopy of parallel_state
    config = Exaone45TransformerConfig(
        num_layers=hf_config.depth,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_heads,
        ffn_hidden_size=hf_config.intermediate_size,
        add_bias_linear=True,
        add_qkv_bias=True,
    )

    # apply text model config to vision model config
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

    config.num_moe_experts = None
    config.expert_model_parallel_size = 1
    config.moe_ffn_hidden_size = None

    config.hidden_dropout = 0.0
    config.attention_dropout = 0.0
    config.layernorm_epsilon = 1e-5
    config.apply_rotary_pos_emb_in_fp32 = True

    config.patch_size = hf_config.patch_size
    config.temporal_patch_size = hf_config.temporal_patch_size
    config.in_channels = hf_config.in_channels
    config.spatial_merge_size = hf_config.spatial_merge_size

    config.out_hidden_size = hf_config.out_hidden_size

    config.apply_rope_fusion = False
    config.gated_linear_unit = True
    config.activation_func = torch.nn.functional.silu

    config.kv_channels = hf_config.hidden_size // hf_config.num_heads
    config.num_query_groups = hf_config.num_key_value_heads
    config.apply_query_key_layer_scaling = False
    config.bias_activation_fusion = False
    config.bias_dropout_fusion = False
    config.attention_softmax_in_fp32 = True
    config.normalization = "RMSNorm"

    config.fullatt_block_indexes = deepcopy(getattr(hf_config, "fullatt_block_indexes", [6, 13, 20, 27]))

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
