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

"""Shensi provider / conversion tests shared fixtures.

The tiny geometry below is the one the shensi recipes use for smoke runs: four layers,
hidden 128, 8 experts.  Keeping it inline makes the tests independent of any external
checkpoint or Hub download.
"""

import torch
from transformers import ShensiConfig


TINY_SHENSI_CONFIG = {
    "architectures": ["ShensiForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "attn_res_block_size": 4,
    "attn_res_read_heads": 8,
    "bos_token_id": 0,
    "compress_rates": {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
    "compress_rope_theta": 160000.0,
    "eos_token_id": 1,
    "erc_loss_alpha": 0.5,
    "erc_loss_coef": 1.0,
    "hc_active_streams": 4,
    "hc_conv_kernels": [4, 8, 12],
    "hc_fixed_streams": 2,
    "hc_mult": 16,
    "head_dim": 512,
    "hidden_act": "silu",
    "hidden_size": 128,
    "index_head_dim": 128,
    "index_n_heads": 16,
    "index_topk": 64,
    "initializer_range": 0.02,
    "layer_types": [
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ],
    "max_position_embeddings": 4096,
    "mlp_bias": False,
    "mlp_layer_types": ["hash_moe", "moe", "moe", "moe"],
    "moe_intermediate_size": 32,
    "n_routed_experts": 8,
    "norm_topk_prob": True,
    "num_attention_heads": 8,
    "num_experts_per_tok": 2,
    "num_hidden_layers": 4,
    "num_key_value_heads": 1,
    "num_nextn_predict_layers": 0,
    "o_groups": 4,
    "o_lora_rank": 32,
    "output_router_logits": False,
    "partial_rotary_factor": 0.125,
    "q_lora_rank": 64,
    "qk_rope_head_dim": 64,
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "compress": {"partial_rotary_factor": 0.125, "rope_theta": 160000.0, "rope_type": "default"},
        "main": {"partial_rotary_factor": 0.125, "rope_theta": 10000.0, "rope_type": "default"},
    },
    "rope_theta": 10000.0,
    "routed_expert_hidden_size": 32,
    "routed_scaling_factor": 1.5,
    "router_aux_loss_coef": 0.001,
    "router_jitter_noise": 0.0,
    "scoring_func": "sqrtsoftplus",
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "tie_word_embeddings": False,
    "vocab_size": 128,
}


def make_tiny_config() -> ShensiConfig:
    """Shensi config for the tiny geometry (float32 keeps CPU-side tests fast)."""
    return ShensiConfig(**TINY_SHENSI_CONFIG)


def make_tiny_hf_model(mtp_layers: int = 0):
    """Build a randomly initialised tiny HF model to convert from."""
    from transformers import ShensiForCausalLM

    cfg_dict = dict(TINY_SHENSI_CONFIG)
    cfg_dict["num_nextn_predict_layers"] = mtp_layers
    model = ShensiForCausalLM(ShensiConfig(**cfg_dict))
    torch.manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.02)
    return model
