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

"""Unit tests for Gemma4ModelProvider (text-only LLM provider)."""

import torch
import pytest

from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.gpt_provider import GPTModelProvider


class TestGemma4ModelProviderDefaults:
    """Verify default values of Gemma4ModelProvider as a standalone dataclass."""

    @pytest.fixture
    def provider(self):
        return Gemma4ModelProvider()

    def test_inherits_from_gpt_provider(self):
        assert issubclass(Gemma4ModelProvider, GPTModelProvider)

    # --- Normalization ---

    def test_uses_rms_norm(self, provider):
        assert provider.normalization == "RMSNorm"

    def test_not_zero_centered_gamma(self, provider):
        """Gemma 4 uses STANDARD RMSNorm (x*w/rms), not zero-centered (Gemma 1/2/3 style)."""
        assert provider.layernorm_zero_centered_gamma is False

    def test_layernorm_epsilon(self, provider):
        assert provider.layernorm_epsilon == 1e-6

    # --- Attention ---

    def test_kv_channels_default(self, provider):
        assert provider.kv_channels == 256

    def test_qk_layernorm_enabled(self, provider):
        assert provider.qk_layernorm is True

    def test_softmax_scale_is_one(self, provider):
        assert provider.softmax_scale == 1.0

    def test_window_size_default(self, provider):
        assert provider.window_size == 1024

    def test_interleaved_attn_pattern(self, provider):
        assert provider.interleaved_attn_pattern == (5, 1)

    def test_global_head_dim(self, provider):
        assert provider.global_head_dim == 512

    def test_num_global_key_value_heads(self, provider):
        assert provider.num_global_key_value_heads == 2

    def test_global_rotary_percent(self, provider):
        assert provider.global_rotary_percent == 0.25

    def test_rotary_base_is_tuple(self, provider):
        """Dual RoPE: (local_base, global_base)."""
        assert isinstance(provider.rotary_base, tuple)
        assert len(provider.rotary_base) == 2
        local, global_ = provider.rotary_base
        assert local == 10_000
        assert global_ == 1_000_000

    # --- Embedding ---

    def test_position_embedding_rope(self, provider):
        assert provider.position_embedding_type == "rope"

    def test_shared_embeddings(self, provider):
        assert provider.share_embeddings_and_output_weights is True

    # --- MoE ---

    def test_num_moe_experts(self, provider):
        assert provider.num_moe_experts == 128

    def test_moe_router_topk(self, provider):
        assert provider.moe_router_topk == 8

    def test_moe_ffn_hidden_size(self, provider):
        assert provider.moe_ffn_hidden_size == 704

    def test_moe_shared_expert_intermediate_size(self, provider):
        assert provider.moe_shared_expert_intermediate_size == 2112

    def test_moe_shared_expert_overlap_false(self, provider):
        """Shared expert overlap must be False; Gemma 4 needs separate pre/post norms."""
        assert provider.moe_shared_expert_overlap is False

    def test_moe_shared_expert_gate_false(self, provider):
        assert provider.moe_shared_expert_gate is False

    def test_moe_layer_freq_all_layers(self, provider):
        assert provider.moe_layer_freq == 1

    def test_moe_grouped_gemm(self, provider):
        assert provider.moe_grouped_gemm is True

    def test_moe_router_pre_softmax(self, provider):
        """HF applies softmax before topk selection."""
        assert provider.moe_router_pre_softmax is True

    # --- Logit softcapping ---

    def test_final_logit_softcapping(self, provider):
        assert provider.final_logit_softcapping == 30.0

    # --- Data type ---

    def test_default_bf16(self, provider):
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_fp16_disabled(self, provider):
        assert provider.fp16 is False

    # --- No bias ---

    def test_no_bias_linear(self, provider):
        assert provider.add_bias_linear is False

    # --- Activation ---

    def test_gated_linear_unit(self, provider):
        assert provider.gated_linear_unit is True

    # --- Seq length ---

    def test_seq_length(self, provider):
        assert provider.seq_length == 262_144

    # --- Dropout ---

    def test_attention_dropout(self, provider):
        assert provider.attention_dropout == 0.0

    def test_hidden_dropout(self, provider):
        assert provider.hidden_dropout == 0.0


class TestGemma4ModelProviderOverride:
    """Test that Gemma4ModelProvider fields can be overridden at construction."""

    def test_override_num_layers(self):
        p = Gemma4ModelProvider(num_layers=32)
        assert p.num_layers == 32

    def test_override_hidden_size(self):
        p = Gemma4ModelProvider(hidden_size=4096)
        assert p.hidden_size == 4096

    def test_override_num_moe_experts(self):
        p = Gemma4ModelProvider(num_moe_experts=64)
        assert p.num_moe_experts == 64

    def test_override_window_size(self):
        p = Gemma4ModelProvider(window_size=512)
        assert p.window_size == 512

    def test_override_vocab_size(self):
        p = Gemma4ModelProvider(vocab_size=300000)
        assert p.vocab_size == 300000
