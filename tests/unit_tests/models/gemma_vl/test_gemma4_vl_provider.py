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

"""Unit tests for all Gemma 4 providers: Gemma4ModelProvider (MoE),
Gemma4DenseProvider (Dense), Gemma4VLModelProvider, and Gemma4DenseVLProvider."""

import pytest
import torch

from megatron.bridge.models.gemma_vl.gemma4_vl_provider import (
    Gemma4DenseVLProvider,
    Gemma4ModelProvider,
    Gemma4VLModelProvider,
    _install_tied_kv,
)
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4DenseProvider
from megatron.bridge.models.gpt_provider import GPTModelProvider


# ===========================================================================
# Gemma4ModelProvider (MoE) tests
# ===========================================================================


class TestGemma4ModelProviderDefaults:
    """Verify default values of Gemma4ModelProvider (MoE) as a standalone dataclass."""

    @pytest.fixture
    def provider(self):
        return Gemma4ModelProvider()

    def test_inherits_from_gpt_provider(self):
        assert issubclass(Gemma4ModelProvider, GPTModelProvider)

    # --- Normalization ---

    def test_uses_rms_norm(self, provider):
        assert provider.normalization == "RMSNorm"

    def test_not_zero_centered_gamma(self, provider):
        """Gemma 4 uses STANDARD RMSNorm (x*w/rms), NOT zero-centered (Gemma 1/2/3 style)."""
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
        assert provider.moe_shared_expert_overlap is False

    def test_moe_shared_expert_gate_false(self, provider):
        assert provider.moe_shared_expert_gate is False

    def test_moe_layer_freq_all_layers(self, provider):
        assert provider.moe_layer_freq == 1

    def test_moe_grouped_gemm(self, provider):
        assert provider.moe_grouped_gemm is True

    def test_moe_router_pre_softmax(self, provider):
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

    # --- Other ---

    def test_no_bias_linear(self, provider):
        assert provider.add_bias_linear is False

    def test_gated_linear_unit(self, provider):
        assert provider.gated_linear_unit is True

    def test_seq_length(self, provider):
        assert provider.seq_length == 262_144

    def test_attention_dropout(self, provider):
        assert provider.attention_dropout == 0.0

    def test_hidden_dropout(self, provider):
        assert provider.hidden_dropout == 0.0


class TestGemma4ModelProviderOverride:
    def test_override_num_layers(self):
        assert Gemma4ModelProvider(num_layers=32).num_layers == 32

    def test_override_hidden_size(self):
        assert Gemma4ModelProvider(hidden_size=4096).hidden_size == 4096

    def test_override_num_moe_experts(self):
        assert Gemma4ModelProvider(num_moe_experts=64).num_moe_experts == 64

    def test_override_window_size(self):
        assert Gemma4ModelProvider(window_size=512).window_size == 512

    def test_override_vocab_size(self):
        assert Gemma4ModelProvider(vocab_size=300000).vocab_size == 300000


# ===========================================================================
# Gemma4DenseProvider (Dense E4B) tests
# ===========================================================================


class TestGemma4DenseProviderDefaults:
    """Verify default values of Gemma4DenseProvider (Dense 3.8B) as a standalone dataclass."""

    @pytest.fixture
    def provider(self):
        return Gemma4DenseProvider()

    def test_inherits_from_gpt_provider(self):
        assert issubclass(Gemma4DenseProvider, GPTModelProvider)

    def test_not_moe_subclass(self):
        assert not issubclass(Gemma4DenseProvider, Gemma4ModelProvider)

    # --- Architecture defaults for E4B ---

    def test_num_layers(self, provider):
        assert provider.num_layers == 42

    def test_hidden_size(self, provider):
        assert provider.hidden_size == 2560

    def test_ffn_hidden_size(self, provider):
        assert provider.ffn_hidden_size == 10240

    def test_num_attention_heads(self, provider):
        assert provider.num_attention_heads == 8

    def test_num_query_groups(self, provider):
        assert provider.num_query_groups == 2

    def test_kv_channels(self, provider):
        assert provider.kv_channels == 256

    def test_global_kv_channels(self, provider):
        assert provider.global_kv_channels == 512

    def test_num_global_query_groups(self, provider):
        assert provider.num_global_query_groups == 2

    # --- Sequence ---

    def test_seq_length(self, provider):
        assert provider.seq_length == 131_072

    def test_vocab_size(self, provider):
        assert provider.vocab_size == 262_143

    # --- Normalization ---

    def test_normalization(self, provider):
        assert provider.normalization == "RMSNorm"

    def test_layernorm_epsilon(self, provider):
        assert provider.layernorm_epsilon == 1e-6

    def test_no_bias_linear(self, provider):
        assert provider.add_bias_linear is False

    def test_gated_linear_unit(self, provider):
        assert provider.gated_linear_unit is True

    # --- RoPE ---

    def test_sliding_window_rope_base(self, provider):
        assert provider.sliding_window_rope_base == 10_000.0

    def test_full_attention_rope_base(self, provider):
        assert provider.full_attention_rope_base == 1_000_000.0

    def test_full_attention_rope_partial_factor(self, provider):
        assert provider.full_attention_rope_partial_factor == 0.25

    # --- Per-Layer Embeddings (PLE) ---

    def test_per_layer_embed_vocab_size(self, provider):
        assert provider.per_layer_embed_vocab_size == 262_144

    def test_per_layer_embed_dim(self, provider):
        assert provider.per_layer_embed_dim == 256

    # --- Shared KV ---

    def test_num_kv_shared_layers(self, provider):
        assert provider.num_kv_shared_layers == 18

    # --- Window attention ---

    def test_window_attn_skip_freq(self, provider):
        assert provider.window_attn_skip_freq == 6

    def test_window_size(self, provider):
        assert provider.window_size == (511, 0)

    # --- Data type ---

    def test_default_bf16(self, provider):
        assert provider.bf16 is True
        assert provider.params_dtype == torch.bfloat16

    def test_fp16_disabled(self, provider):
        assert provider.fp16 is False

    # --- Dropout ---

    def test_attention_dropout(self, provider):
        assert provider.attention_dropout == 0.0

    def test_hidden_dropout(self, provider):
        assert provider.hidden_dropout == 0.0

    # --- Embeddings ---

    def test_scale_embeddings_by_hidden_size(self, provider):
        assert provider.scale_embeddings_by_hidden_size is True

    def test_shared_embeddings(self, provider):
        assert provider.share_embeddings_and_output_weights is True

    def test_rope_position_embedding(self, provider):
        assert provider.position_embedding_type == "rope"


class TestGemma4DenseProviderOverride:
    def test_override_num_layers(self):
        assert Gemma4DenseProvider(num_layers=10).num_layers == 10

    def test_override_hidden_size(self):
        assert Gemma4DenseProvider(hidden_size=1024).hidden_size == 1024

    def test_override_kv_shared_layers(self):
        assert Gemma4DenseProvider(num_kv_shared_layers=0).num_kv_shared_layers == 0

    def test_override_per_layer_embed_dim(self):
        assert Gemma4DenseProvider(per_layer_embed_dim=128).per_layer_embed_dim == 128

    def test_override_vocab_size(self):
        assert Gemma4DenseProvider(vocab_size=100000).vocab_size == 100000

    def test_override_seq_length(self):
        assert Gemma4DenseProvider(seq_length=4096).seq_length == 4096


# ===========================================================================
# Gemma4VLModelProvider (MoE VL) tests
# ===========================================================================


class TestGemma4VLModelProviderDefaults:
    def test_initialization(self):
        p = Gemma4VLModelProvider(num_layers=62, hidden_size=2816, num_attention_heads=8)
        assert isinstance(p, Gemma4VLModelProvider)
        assert isinstance(p, Gemma4ModelProvider)

    def test_vl_defaults(self):
        p = Gemma4VLModelProvider(num_layers=62, hidden_size=2816, num_attention_heads=8)
        assert p.scatter_embedding_sequence_parallel is False
        assert p.vision_soft_tokens_per_image == 280
        assert p.bos_token_id == 2
        assert p.eos_token_id == 1
        assert p.image_token_id == 258_880
        assert p.video_token_id == 258_884
        assert p.audio_token_id == 258_881

    def test_audio_config_defaults_to_none(self):
        p = Gemma4VLModelProvider(num_layers=62, hidden_size=2816, num_attention_heads=8)
        assert p.audio_config is None

    def test_freeze_defaults(self):
        p = Gemma4VLModelProvider(num_layers=62, hidden_size=2816, num_attention_heads=8)
        assert p.freeze_language_model is False
        assert p.freeze_vision_model is False
        assert p.freeze_vision_projection is False

    def test_vision_config_defaults_to_none(self):
        p = Gemma4VLModelProvider(num_layers=62, hidden_size=2816, num_attention_heads=8)
        assert p.vision_config is None
        assert p.text_config is None

    def test_inherited_gemma4_defaults(self):
        p = Gemma4VLModelProvider(num_layers=62, hidden_size=2816, num_attention_heads=8)
        assert p.normalization == "RMSNorm"
        assert p.gated_linear_unit is True
        assert p.position_embedding_type == "rope"
        assert p.add_bias_linear is False
        assert p.attention_dropout == 0.0
        assert p.hidden_dropout == 0.0
        assert p.share_embeddings_and_output_weights is True

    def test_custom_token_ids(self):
        p = Gemma4VLModelProvider(
            num_layers=62, hidden_size=2816, num_attention_heads=8,
            image_token_id=99999, video_token_id=99998,
        )
        assert p.image_token_id == 99999
        assert p.video_token_id == 99998

    def test_custom_vision_tokens_per_image(self):
        p = Gemma4VLModelProvider(
            num_layers=62, hidden_size=2816, num_attention_heads=8,
            vision_soft_tokens_per_image=560,
        )
        assert p.vision_soft_tokens_per_image == 560

    def test_freeze_options_configurable(self):
        p = Gemma4VLModelProvider(
            num_layers=62, hidden_size=2816, num_attention_heads=8,
            freeze_language_model=True, freeze_vision_model=True,
        )
        assert p.freeze_language_model is True
        assert p.freeze_vision_model is True
        assert p.freeze_vision_projection is False

    def test_different_hidden_sizes(self):
        for hs in [1152, 2048, 2816, 4096]:
            p = Gemma4VLModelProvider(num_layers=28, hidden_size=hs, num_attention_heads=8)
            assert p.hidden_size == hs

    def test_different_layer_counts(self):
        for nl in [18, 28, 46, 62]:
            p = Gemma4VLModelProvider(num_layers=nl, hidden_size=2816, num_attention_heads=8)
            assert p.num_layers == nl


# ===========================================================================
# Gemma4DenseVLProvider (Dense VL) tests
# ===========================================================================


class TestGemma4DenseVLProviderDefaults:
    def test_initialization(self):
        p = Gemma4DenseVLProvider()
        assert isinstance(p, Gemma4DenseVLProvider)
        assert isinstance(p, Gemma4DenseProvider)

    def test_inherits_dense_defaults(self):
        p = Gemma4DenseVLProvider()
        assert p.num_layers == 42
        assert p.hidden_size == 2560
        assert p.num_attention_heads == 8
        assert p.num_kv_shared_layers == 18
        assert p.per_layer_embed_dim == 256

    def test_vl_defaults(self):
        p = Gemma4DenseVLProvider()
        assert p.scatter_embedding_sequence_parallel is False
        assert p.vision_soft_tokens_per_image == 280
        assert p.bos_token_id == 2
        assert p.eos_token_id == 1
        assert p.image_token_id == 258_880
        assert p.audio_token_id == 258_881

    def test_audio_config_defaults_to_none(self):
        assert Gemma4DenseVLProvider().audio_config is None

    def test_vision_config_defaults_to_none(self):
        p = Gemma4DenseVLProvider()
        assert p.vision_config is None
        assert p.text_config is None

    def test_freeze_defaults(self):
        p = Gemma4DenseVLProvider()
        assert p.freeze_language_model is False
        assert p.freeze_vision_model is False
        assert p.freeze_vision_projection is False

    def test_override_vl_fields(self):
        p = Gemma4DenseVLProvider(image_token_id=12345, audio_token_id=99999)
        assert p.image_token_id == 12345
        assert p.audio_token_id == 99999


# ===========================================================================
# _install_tied_kv helper tests
# ===========================================================================


class TestInstallTiedKV:
    def test_skips_when_attention_k_eq_v_false(self):
        provider = Gemma4ModelProvider(
            num_layers=6, hidden_size=64, num_attention_heads=4, attention_k_eq_v=False,
        )
        provider.num_moe_experts = None

        class FakeLayer:
            layer_number = 1

        class FakeModel:
            class decoder:
                layers = [FakeLayer()]

        _install_tied_kv(FakeModel(), provider)
        assert not getattr(FakeLayer, "_tied_kv", False)

    def test_marks_global_layers_only(self):
        import torch.nn as nn

        provider = Gemma4ModelProvider(
            num_layers=6, hidden_size=64, num_attention_heads=4,
            num_global_key_value_heads=2, global_head_dim=16,
            interleaved_attn_pattern=(5, 1),
            num_moe_experts=4, attention_k_eq_v=True,
        )

        class FakeLinear(nn.Module):
            def forward(self, x):
                return x, None

        class FakeAttn:
            def __init__(self):
                self.linear_qkv = FakeLinear()

        class FakeLayer:
            def __init__(self, number):
                self.layer_number = number
                self.self_attention = FakeAttn()

        class FakeDecoder:
            def __init__(self):
                self.layers = [FakeLayer(i) for i in range(1, 7)]

        class FakeModel:
            def __init__(self):
                self.decoder = FakeDecoder()

        model = FakeModel()
        _install_tied_kv(model, provider)

        for layer in model.decoder.layers:
            is_global = layer.layer_number == 6
            has_flag = getattr(layer.self_attention, "_tied_kv", False)
            assert has_flag == is_global, (
                f"Layer {layer.layer_number}: expected _tied_kv={is_global}, got {has_flag}"
            )
