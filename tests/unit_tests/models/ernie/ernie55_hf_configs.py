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

"""ERNIE 5.5 Hugging Face ``config.json`` contents shared by the CPU tests.

* :data:`PROXY_CONFIG`: the 4-layer proxy the reference runs train (H, D, H, D decoder with a dense first
  layer, one shared-body MTP depth, 32 experts).
* :func:`full_size_config`: a full-size test layout (43 layers, 21 H and 22 D, one shared-body MTP depth, 256
  experts) with the ERNIE 5.5 vocabulary and D-layer geometry. It is not the released checkpoint's config.
"""

from __future__ import annotations

import copy
from typing import Any


PROXY_CONFIG: dict[str, Any] = {
    "architectures": ["Ernie5_5_MoeForCausalLM"],
    "model_type": "ernie5_5_moe",
    "torch_dtype": "bfloat16",
    "vocab_size": 200027,
    "hidden_size": 4096,
    "num_hidden_layers": 4,
    "intermediate_size": 8192,
    "first_k_dense_replace": 1,
    "max_position_embeddings": 262144,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "attention_dropout": 0.0,
    "initializer_range": 0.02,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "v_head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "rope_theta": 8000000.0,
    "compress_rope_theta": 160000.0,
    "sliding_window": 128,
    "compress_ratios": [128, -2, 128, -2, -2],
    "csa_dense_mode": True,
    "hca_rope_type": "rope",
    "rotary_scaling_factor": 1.0,
    "original_max_position_embeddings": 4096,
    "hybrid_mla_attention": "mqa_dsa",
    "hybrid_mla_num_attention_heads": 64,
    "hybrid_mla_q_lora_rank": 1024,
    "hybrid_mla_kv_lora_rank": 512,
    "hybrid_mla_qk_nope_head_dim": 448,
    "hybrid_mla_qk_rope_head_dim": 64,
    "hybrid_mla_v_head_dim": 256,
    "mqa_latent_rope_adjacent_pairing": True,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 2048,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "n_routed_experts": 32,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 2048,
    "n_shared_experts": 1,
    "scoring_func": "sigmoid",
    "norm_topk_prob": True,
    "routed_scaling_factor": 1.0,
    "latent_moe_use_norm": False,
    "num_nextn_predict_layers": 1,
    "mtp_shared_last_layer": True,
    "sigmoid_gate_fusion": False,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "pad_token_id": 151643,
    "use_qk_norm": True,
    "qk_nope_head_dim": 448,
}
"""The 4-layer proxy's ``config.json``."""

PROXY_PATTERN = "H-DEHEDE/DE"
"""Hybrid pattern of :data:`PROXY_CONFIG`."""


def full_size_config() -> dict[str, Any]:
    """The assumed full-size ``config.json``: 21 x (H, D), one more D layer, one D MTP depth, 256 experts."""
    config = copy.deepcopy(PROXY_CONFIG)
    config.update(
        num_hidden_layers=43,
        intermediate_size=12288,
        first_k_dense_replace=0,
        compress_ratios=[128, -2] * 21 + [-2, -2],
        n_routed_experts=256,
        num_experts_per_tok=6,
        routed_scaling_factor=1.5,
    )
    return config


FULL_SIZE_PATTERN = "HEDE" * 21 + "DE" + "/DE"
"""Hybrid pattern of :func:`full_size_config`."""


def proxy_config(*, drop: tuple[str, ...] = (), **overrides: Any) -> dict[str, Any]:
    """A copy of :data:`PROXY_CONFIG` without the ``drop`` keys and with ``overrides`` applied."""
    config = copy.deepcopy(PROXY_CONFIG)
    for key in drop:
        config.pop(key)
    config.update(overrides)
    return config
