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

"""Unit tests for the DeepSeek MLA attention spec helpers."""

from types import SimpleNamespace

import pytest
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

from megatron.bridge.models.deepseek.attention import MLASelfAttentionWithoutQueryNorm


def _mla_submodules():
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=8, moe_grouped_gemm=False, qk_layernorm=True, multi_latent_attention=True
    )
    return spec.submodules.self_attention.submodules


def _config(q_lora_rank):
    return SimpleNamespace(
        q_lora_rank=q_lora_rank,
        qk_layernorm=True,
        qk_l2_norm=False,
        multi_latent_attention=True,
        experimental_attention_variant=None,
        normalization="RMSNorm",
        transformer_impl="transformer_engine",
    )


def _resolve(q_lora_rank):
    # `_resolve_qk_norm_config` only reads `self.config`, so bypass __init__ (which would
    # need process groups) but keep a real instance so its zero-arg `super()` resolves.
    attention = MLASelfAttentionWithoutQueryNorm.__new__(MLASelfAttentionWithoutQueryNorm)
    object.__setattr__(attention, "config", _config(q_lora_rank))
    return attention._resolve_qk_norm_config(_mla_submodules())


class TestMLASelfAttentionWithoutQueryNorm:
    """DeepSeek MLA must not gain a query norm the HF architecture does not define."""

    def test_no_query_lora_drops_the_fused_query_norm(self):
        """With q_lora_rank=None, HF has no query norm, so linear_q_proj must stay unfused."""
        resolved = _resolve(q_lora_rank=None)
        assert "LayerNorm" not in resolved["linear_q_proj"].__name__

    def test_no_query_lora_keeps_the_kv_norm(self):
        """kv_a_layernorm exists in every DeepSeek checkpoint and must still be built."""
        resolved = _resolve(q_lora_rank=None)
        assert "LayerNorm" in resolved["linear_kv_up_proj"].__name__

    def test_query_lora_is_untouched(self):
        """With a query LoRA the norm belongs on linear_q_up_proj and maps to q_a_layernorm."""
        resolved = _resolve(q_lora_rank=1536)
        assert "LayerNorm" in resolved["linear_q_up_proj"].__name__
        assert "LayerNorm" in resolved["linear_kv_up_proj"].__name__

    @pytest.mark.parametrize("q_lora_rank", [None, 1536])
    def test_standalone_norms_stay_disabled(self, q_lora_rank):
        """Norms stay fused into the projections; no standalone q/kv norm modules appear."""
        resolved = _resolve(q_lora_rank)
        assert resolved["q_layernorm"].__name__ == "IdentityOp"
        assert resolved["kv_layernorm"].__name__ == "IdentityOp"
