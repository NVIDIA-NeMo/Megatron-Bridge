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

"""Functional toy-model conversion test scaffold for DeepSeek V4.

The test is skipped until the required DeepSeek V4 modeling classes are
available from Transformers and Megatron-Core in the test environment.
"""

import importlib.util

import pytest


def _has_dsv4_in_transformers() -> bool:
    try:
        from transformers import DeepseekV4ForCausalLM  # noqa: F401

        return True
    except Exception:
        return False


def _has_dsv4_in_mcore() -> bool:
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in (
            "megatron.core.transformer.hyper_connection",
            "megatron.core.transformer.experimental_attention_variant.csa",
            "megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention",
        )
    )


pytestmark = [
    pytest.mark.skipif(
        not _has_dsv4_in_transformers(),
        reason="transformers does not yet ship DeepseekV4ForCausalLM (HF hub only via trust_remote_code).",
    ),
    pytest.mark.skipif(
        not _has_dsv4_in_mcore(),
        reason="megatron-core does not yet ship DSv4 prerequisites (PRs #3430 / #4458 / #4481 / #4518).",
    ),
]


# Toy config tuned to satisfy DSv4 invariants at minimum size:
#   - len(compress_ratios) == num_hidden_layers + num_nextn_predict_layers
#   - sliding_window <= max_position_embeddings
#   - vocab_size large enough for hash routing in the first dense_replace block
#   - n_routed_experts divisible by num_experts_per_tok and the EP sizes we test
HF_DEEPSEEK_V4_TOY_MODEL_CONFIG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "model_type": "deepseek_v4",
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "head_dim": 256,
    "qk_rope_head_dim": 32,
    "intermediate_size": 2048,
    "max_position_embeddings": 4096,
    "moe_intermediate_size": 512,
    "n_routed_experts": 8,
    "n_shared_experts": 1,
    "num_attention_heads": 16,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 4,
    "num_key_value_heads": 4,
    "num_nextn_predict_layers": 0,  # disable MTP for the toy
    "q_lora_rank": 256,
    "o_lora_rank": 256,
    "o_groups": 4,
    "compress_ratios": [0, 4, 4, 4],  # 4 entries == num_hidden_layers (mtp=0)
    "sliding_window": 64,
    "index_n_heads": 4,
    "index_head_dim": 32,
    "index_topk": 32,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 4,
    "norm_topk_prob": True,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.0,
    "rope_theta": 10000,
    "rope_scaling": {
        "beta_fast": 32,
        "beta_slow": 1,
        "factor": 16,
        "original_max_position_embeddings": 4096,
        "type": "yarn",
    },
    "vocab_size": 8192,
    "torch_dtype": "bfloat16",
}


class TestDeepSeekV4Conversion:
    """Toy HF-to-Megatron roundtrip coverage for DeepSeek V4."""

    def test_placeholder(self):
        """Sentinel test that fails closed once the skip conditions no longer apply."""
        pytest.fail(
            "DSv4 prerequisites are now available — replace this placeholder with a real toy "
            "HF-to-Megatron roundtrip using HF_DEEPSEEK_V4_TOY_MODEL_CONFIG."
        )
