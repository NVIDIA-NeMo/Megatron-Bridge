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

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

LLM_VOCAB_SIZE_MAP = {
    "gpt3": 51200,
    "llama2": 32000,
    "llama3": 128256,
    "nemotronh": 256000,
}


@dataclass
class FLOPSConfig:
    """Contains the model hparams needed for FLOPS computations"""
    gbs: int
    enc_seq_len: Optional[int] = None
    hs: Optional[int] = None
    layers: Optional[int] = None
    ffn_hs: Optional[int] = None
    attention_heads: Optional[int] = None
    moe_router_topk: Optional[int] = None
    query_groups: Optional[int] = None
    kv_channels: Optional[int] = None
    img_seq_len: Optional[int] = None
    img_h: Optional[int] = None
    img_w: Optional[int] = None
    in_channels: Optional[int] = None
    patch_dim: Optional[int] = None
    class_token_len: Optional[int] = None
    projector_type: Optional[str] = None
    inp_s: Optional[int] = None
    model_pattern: Optional[str] = None
    vocab_size: Optional[int] = None
    model_channels: Optional[int] = None
    vec_in_dim: Optional[int] = None
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_head_dim: Optional[int] = None
    qk_pos_emb_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    moe_layer_freq: Union[int, List[int]] = None
    moe_shared_expert_intermediate_size: Optional[int] = None
    moe_ffn_hidden_size: Optional[int] = None
    mtp_num_layers: Optional[int] = None
    causal_self_attn: Optional[bool] = None
    is_hybrid_model: bool = False
    hybrid_override_pattern: Optional[str] = None
    mamba_state_dim: Optional[int] = None
    mamba_head_dim: Optional[int] = None
    mamba_num_groups: Optional[int] = None
    mamba_num_heads: Optional[int] = None
    # SWA configs
    window_attn_skip_freq: Optional[Union[int, List[int]]] = None
    window_size: Optional[Tuple[int, int]] = (128, 0)


def gpt3(config: FLOPSConfig):
    """Model FLOPs for GPT3 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["gpt3"]
    causal_self_attn = True

    return (
        24 * config.gbs * config.enc_seq_len * config.hs * config.hs
        + 4 * config.gbs * config.enc_seq_len * config.enc_seq_len * config.hs * (0.5 if causal_self_attn else 1)
    ) * (3 * config.layers) + (6 * config.gbs * config.enc_seq_len * config.hs * vocab_size)


def llama2(config: FLOPSConfig):
    """Model FLOPs for llama2 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama2"]
    causal_self_attn = True

    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def llama3(config: FLOPSConfig):
    """Model FLOPs for llama3 family"""
    vocab_size = LLM_VOCAB_SIZE_MAP["llama3"]
    causal_self_attn = True

    return (
        config.gbs
        * config.enc_seq_len
        * config.layers
        * config.hs
        * config.hs
        * (
            12
            + (12 * config.query_groups / config.attention_heads)
            + (18 * config.ffn_hs / config.hs)
            + (12 * config.enc_seq_len / config.hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (config.layers * config.hs))
        )
    )


def qwen3(config: FLOPSConfig):
    """Model FLOPs for Qwen3 family"""
    causal_self_attn = True
    seq_len = config.enc_seq_len
    hidden_size = config.hs
    gated_linear_multiplier = 2

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * hidden_size
        * (
            (config.query_groups / config.attention_heads * 2 + 1)  # QKV gemm
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))  # attention
            + 1  # attention proj gemm
        )
    )

    # mlp flops
    mlp_flops = (
        3
        * 2
        * config.gbs
        * config.layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (config.moe_ffn_hidden_size * config.moe_router_topk)  # MoE layers
    )

    # vocab flops
    vocab_flops = 3 * 2 * config.gbs * seq_len * hidden_size * config.vocab_size

    return attention_flops + mlp_flops + vocab_flops


def deepseekv3(config: FLOPSConfig):
    """Model FLOPs for DeepSeek V3"""

    # self-attention flops
    bmm1_flops = (
        0.5 * (config.qk_head_dim + config.qk_pos_emb_head_dim) * config.attention_heads * (config.enc_seq_len**2)
    )
    bmm2_flops = 0.5 * config.v_head_dim * config.attention_heads * (config.enc_seq_len**2)
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * config.layers
    if config.mtp_num_layers is not None:
        per_input_attention_flops += 6 * (bmm1_flops + bmm2_flops) * config.mtp_num_layers

    # linear layer flops
    per_layer_mla_params = config.hs * config.q_lora_rank + config.q_lora_rank * (
        (config.qk_head_dim + config.qk_pos_emb_head_dim) * config.attention_heads
    )  # Q
    per_layer_mla_params += config.hs * config.qk_pos_emb_head_dim  # K^R
    per_layer_mla_params += config.hs * config.kv_lora_rank + config.kv_lora_rank * (
        (config.qk_head_dim + config.v_head_dim) * config.attention_heads
    )  # K^C and V^C
    per_layer_mla_params += config.v_head_dim * config.attention_heads * config.hs  # Proj
    mla_params = per_layer_mla_params * config.layers
    if config.mtp_num_layers is not None:
        mla_params += per_layer_mla_params * config.mtp_num_layers

    dense_layer_ffn_params = config.hs * config.ffn_hs * 3  # gated linear unit
    per_shared_expert_params = config.hs * config.moe_shared_expert_intermediate_size * 3
    per_selected_expert_params = config.hs * config.moe_ffn_hidden_size * 3
    ffn_params = 0

    if isinstance(config.moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % config.moe_layer_freq == 0) else 0 for i in range(config.layers)]
    else:
        moe_layer_pattern = config.moe_layer_freq
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * config.moe_router_topk)
    if config.mtp_num_layers is not None:
        for i in range(config.mtp_num_layers):
            ffn_params += per_shared_expert_params + (per_selected_expert_params * config.moe_router_topk)
    per_input_params = mla_params + ffn_params
    per_input_linear_flops = 6 * per_input_params * config.enc_seq_len

    # vocab flops
    per_input_vocab_flops = 6 * config.vocab_size * config.hs * config.enc_seq_len
    if config.mtp_num_layers is not None:
        for i in range(config.mtp_num_layers):
            per_input_vocab_flops += 6 * config.vocab_size * config.hs * config.enc_seq_len
            per_input_vocab_flops += 6 * config.hs * 2 * config.hs * config.enc_seq_len

    return (per_input_attention_flops + per_input_linear_flops + per_input_vocab_flops) * config.gbs


def _nemotronh_mlp_layer_flops(config: FLOPSConfig):
    """Model FLOPs for MLP layer. Assume gated linear unit."""
    return 6 * config.gbs * config.enc_seq_len * config.hs * config.ffn_hs * 3


def _non_mla_attn_layer_flops(config: FLOPSConfig):
    """Model FLOPs for attention layer"""
    return (
        6
        * config.gbs
        * config.enc_seq_len
        * config.hs
        * (
            config.hs  # Q
            + config.query_groups / config.attention_heads * config.hs * 2  # KV
            + config.enc_seq_len / 2 * 2
            + config.hs
        )
    )


def _hybrid_model_flops(config: FLOPSConfig):
    """Model FLOPs for hybrid model"""
    assert config.is_hybrid_model == True
    assert config.hybrid_override_pattern is not None

    num_attn_layers, num_mamba_layers, num_mlp_layers = 0, 0, 0
    for c in config.hybrid_override_pattern:
        if c == 'M':
            num_mamba_layers += 1
        elif c == '-':
            num_mlp_layers += 1
        elif c == '*':
            num_attn_layers += 1
    return (
        num_attn_layers * _non_mla_attn_layer_flops(config)
        + num_mamba_layers * _mamba_layer_flops(config)
        + num_mlp_layers * _nemotronh_mlp_layer_flops(config)
        + 6 * config.gbs * config.enc_seq_len * config.hs * config.vocab_size
    )


def nemotronh(config: FLOPSConfig):
    """Model FLOPs for NemotronH"""
    return _hybrid_model_flops(config)


def attention_flops_calculator(
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    kv_channels: Optional[int] = None,
    is_swa: bool = False,
    swa_window_size: int = 128,
):
    """Calculate the flops for the attention part."""
    kv_channels = kv_channels or (hidden_size // num_attention_heads)

    linear_qkv = seqlen * hidden_size * (kv_channels * (num_attention_heads + num_query_groups * 2))

    linear_proj = seqlen * hidden_size * (kv_channels * num_attention_heads)

    if is_swa:
        attention_mask_nz_elem = (
            swa_window_size * (swa_window_size + 1) / 2 + (seqlen - swa_window_size) * swa_window_size
        )
        attention = num_attention_heads * (attention_mask_nz_elem * kv_channels) * 2
    else:
        bmm_k = kv_channels
        bmm_b = num_attention_heads
        attention_mask_nz_elem = seqlen * (seqlen + 1) / 2
        attention = bmm_b * attention_mask_nz_elem * bmm_k * 2

    return (linear_qkv + linear_proj + attention) * 6


def moe_mlp_flops_calculator(
    seqlen,
    hidden_size,
    moe_ffn_hidden_size,
    moe_router_topk,
    gated_linear_unit: bool = True,
):
    """Calculate the flops for the MLP"""
    total_num_tokens = seqlen * moe_router_topk
    linear_fc1 = total_num_tokens * hidden_size * moe_ffn_hidden_size * (2 if gated_linear_unit else 1)
    linear_fc2 = total_num_tokens * moe_ffn_hidden_size * hidden_size
    return (linear_fc1 + linear_fc2) * 6


def loss_flops_calculator(
    seqlen,
    hidden_size,
    vocab_size,
):
    """Calculate the flops for the loss"""
    return (seqlen * hidden_size * vocab_size) * 6
