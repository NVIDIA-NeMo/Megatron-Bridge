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
Unit tests for Sarvam provider classes.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.sarvam.sarvam_provider import (
    SarvamMLAModelProvider,
    SarvamMoEModelProvider,
    _get_sarvam_moe_pipeline_layout,
)
from megatron.bridge.models.transformer_config import MLATransformerConfig, TransformerConfig


class TestSarvamProviderDefaults:
    def test_sarvam_moe_provider_defaults(self):
        provider = SarvamMoEModelProvider()

        # Core architecture flags
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.share_embeddings_and_output_weights is False

        # DType defaults
        assert provider.bf16 is True
        # Sarvam providers default params_dtype to float32; bridges override this based on HF config.
        assert provider.params_dtype == torch.float32

        # Defaults that should stay consistent with Sarvam MoE reference configs
        assert provider.num_query_groups == 4
        assert provider.kv_channels == 64
        assert provider.seq_length == 131072
        assert provider.rotary_base == 8_000_000.0
        assert provider.vocab_size == 262144

        # MoE routing defaults
        assert provider.moe_grouped_gemm is True
        assert provider.moe_aux_loss_coeff == 0
        assert provider.moe_router_pre_softmax is True
        assert provider.moe_router_enable_expert_bias is True
        assert provider.moe_router_bias_update_rate == 1e-3
        assert provider.moe_router_dtype == "fp32"
        assert provider.moe_router_score_function == "sigmoid"
        assert provider.moe_router_topk_scaling_factor == 2.5
        assert provider.moe_token_dispatcher_type == "alltoall"
        assert provider.moe_permute_fusion is True
        assert provider.recompute_granularity == "selective"

    def test_sarvam_mla_provider_defaults(self):
        provider = SarvamMLAModelProvider()

        assert provider.multi_latent_attention is True
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == F.silu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_qkv_bias is False
        assert provider.qk_layernorm is True
        assert provider.make_vocab_size_divisible_by == 128

        # DType defaults
        assert provider.bf16 is True
        assert provider.params_dtype == torch.float32

        # MLA-related defaults
        assert provider.kv_lora_rank == 512
        assert provider.qk_head_dim == 128
        assert provider.qk_pos_emb_head_dim == 64
        assert provider.v_head_dim == 128
        assert provider.rotary_scaling_factor == 40
        assert provider.mscale == 1.0
        assert provider.mscale_all_dim == 1.0

        # Shape/config defaults that must stay coherent
        assert provider.seq_length == 131072
        assert provider.rotary_base == 10_000.0
        assert provider.vocab_size == 262144
        assert provider.recompute_granularity == "selective"

    def test_default_factories_are_not_shared_between_instances(self):
        """Guard against accidental shared mutable defaults (list fields)."""
        p1 = SarvamMoEModelProvider()
        p2 = SarvamMoEModelProvider()

        assert p1.recompute_modules is not p2.recompute_modules
        assert p1.moe_layer_freq is not p2.moe_layer_freq

        # Mutating one must not affect the other
        p1.recompute_modules.append("sentinel")
        assert "sentinel" not in p2.recompute_modules

        p1.moe_layer_freq.append(999)
        assert 999 not in p2.moe_layer_freq

    def test_override_core_dims_does_not_break_defaults(self):
        """Light sanity check that overriding core sizes works and doesn't silently reset flags."""
        provider = SarvamMoEModelProvider(
            num_layers=1,
            hidden_size=1024,
            num_attention_heads=8,
            vocab_size=32000,
            seq_length=2048,
        )

        assert provider.num_layers == 1
        assert provider.hidden_size == 1024
        assert provider.num_attention_heads == 8
        assert provider.vocab_size == 32000
        assert provider.seq_length == 2048

        # Flags should remain consistent unless explicitly changed
        assert provider.position_embedding_type == "rope"
        assert provider.qk_layernorm is True
        assert provider.add_qkv_bias is False

    def test_provider_inheritance_and_provide_available(self):
        """Match Qwen provider tests: providers should inherit from GPTModelProvider and expose provide()."""
        assert issubclass(SarvamMoEModelProvider, GPTModelProvider)
        assert issubclass(SarvamMLAModelProvider, GPTModelProvider)
        assert issubclass(SarvamMLAModelProvider, MLATransformerConfig)

        moe = SarvamMoEModelProvider()
        mla = SarvamMLAModelProvider()
        assert hasattr(moe, "provide") and callable(moe.provide)
        assert hasattr(mla, "provide") and callable(mla.provide)

    def test_moe_layer_freq_matches_num_layers(self):
        """moe_layer_freq should be a per-layer schedule; ensure length matches num_layers and values are valid."""
        moe = SarvamMoEModelProvider()
        assert isinstance(moe.moe_layer_freq, list)
        assert len(moe.moe_layer_freq) == moe.num_layers
        assert set(moe.moe_layer_freq).issubset({0, 1})
        # defaults defined as [0] + [1] * (num_layers-1)
        assert moe.moe_layer_freq[0] == 0
        assert all(x == 1 for x in moe.moe_layer_freq[1:])

        mla = SarvamMLAModelProvider()
        assert isinstance(mla.moe_layer_freq, list)
        assert len(mla.moe_layer_freq) == mla.num_layers
        assert set(mla.moe_layer_freq).issubset({0, 1})
        assert mla.moe_layer_freq[0] == 0
        assert all(x == 1 for x in mla.moe_layer_freq[1:])

    def test_kv_channels_coherence_with_head_dim(self):
        """With Sarvam defaults, kv_channels should match head_dim = hidden_size / num_attention_heads."""
        moe = SarvamMoEModelProvider()
        assert moe.kv_channels is not None
        assert moe.hidden_size % moe.num_attention_heads == 0
        assert moe.kv_channels == moe.hidden_size // moe.num_attention_heads

        mla = SarvamMLAModelProvider()
        assert mla.kv_channels is not None
        assert mla.hidden_size % mla.num_attention_heads == 0
        assert mla.kv_channels == mla.hidden_size // mla.num_attention_heads

    def test_moe_configuration_validity(self):
        """Basic MoE validity check (mirrors Qwen provider edge-case tests)."""
        moe = SarvamMoEModelProvider()
        assert moe.num_moe_experts >= 1
        assert moe.moe_router_topk >= 1
        assert moe.moe_router_topk <= moe.num_moe_experts

        mla = SarvamMLAModelProvider()
        assert mla.num_moe_experts >= 1
        assert mla.moe_router_topk >= 1
        assert mla.moe_router_topk <= mla.num_moe_experts

    def test_dtype_configuration_overrides(self):
        """Ensure fp16/bf16 flags and params_dtype can be overridden explicitly."""
        moe = SarvamMoEModelProvider(fp16=True, bf16=False, params_dtype=torch.float16)
        assert moe.fp16 is True
        assert moe.bf16 is False
        assert moe.params_dtype == torch.float16

        mla = SarvamMLAModelProvider(fp16=True, bf16=False, params_dtype=torch.float16)
        assert mla.fp16 is True
        assert mla.bf16 is False
        assert mla.params_dtype == torch.float16


class TestSarvamMoEFinalize:
    """Tests for SarvamMoEModelProvider.finalize() uneven pipeline logic."""

    @pytest.mark.parametrize(
        ("pp", "expected_layout"),
        [
            (2, [["embedding"] + ["decoder"] * 10, ["decoder"] * 9 + ["loss"]]),
            (4, [["embedding"] + ["decoder"] * 5, ["decoder"] * 5, ["decoder"] * 5, ["decoder"] * 4 + ["loss"]]),
            (
                8,
                [
                    ["embedding"] + ["decoder"] * 3,
                    ["decoder"] * 3,
                    ["decoder"] * 3,
                    ["decoder"] * 2,
                    ["decoder"] * 2,
                    ["decoder"] * 2,
                    ["decoder"] * 2,
                    ["decoder"] * 2 + ["loss"],
                ],
            ),
        ],
    )
    def test_finalize_sets_pipeline_layout_when_pp_uneven(self, pp, expected_layout):
        """19 layers use explicit layouts for the supported uneven PP sizes."""
        provider = SarvamMoEModelProvider(pipeline_model_parallel_size=pp)

        with patch.object(TransformerConfig, "finalize", autospec=True) as mock_super:
            provider.finalize()

        assert provider.pipeline_model_parallel_layout == expected_layout
        assert provider.num_layers_in_first_pipeline_stage is None
        mock_super.assert_called_once_with(provider)

    def test_finalize_no_change_when_pp_is_one(self):
        """PP=1 (default) should not set a custom pipeline layout."""
        provider = SarvamMoEModelProvider(pipeline_model_parallel_size=1)

        with patch.object(TransformerConfig, "finalize", autospec=True) as mock_super:
            provider.finalize()

        assert provider.pipeline_model_parallel_layout is None
        mock_super.assert_called_once_with(provider)

    def test_finalize_no_change_when_pp_evenly_divides(self):
        """When num_layers is divisible by PP, no special layout is needed."""
        provider = SarvamMoEModelProvider(
            num_layers=20,
            moe_layer_freq=[0] + [1] * 19,
            pipeline_model_parallel_size=4,
        )

        with patch.object(TransformerConfig, "finalize", autospec=True) as mock_super:
            provider.finalize()

        assert provider.pipeline_model_parallel_layout is None
        mock_super.assert_called_once_with(provider)

    def test_finalize_raises_for_unsupported_uneven_pp(self):
        """Unsupported uneven PP sizes should fail fast with a clear error."""
        provider = SarvamMoEModelProvider(pipeline_model_parallel_size=3)

        with patch.object(TransformerConfig, "finalize", autospec=True) as mock_super:
            with pytest.raises(ValueError, match="Unsupported PP size 3"):
                provider.finalize()

        mock_super.assert_not_called()

    def test_finalize_respects_explicit_pipeline_layout(self):
        """Caller-supplied layouts should not be overwritten."""
        custom_layout = [["embedding"] + ["decoder"] * 7, ["decoder"] * 6, ["decoder"] * 6 + ["loss"]]
        provider = SarvamMoEModelProvider(
            pipeline_model_parallel_size=3,
            pipeline_model_parallel_layout=custom_layout,
        )

        with patch.object(TransformerConfig, "finalize", autospec=True) as mock_super:
            provider.finalize()

        assert provider.pipeline_model_parallel_layout == custom_layout
        mock_super.assert_called_once_with(provider)


class TestSarvamMoEPipelineLayout:
    """Tests for the Sarvam MoE PP layout helper."""

    @pytest.mark.parametrize(
        ("pp", "expected_layout"),
        [
            (1, None),
            (2, [["embedding"] + ["decoder"] * 10, ["decoder"] * 9 + ["loss"]]),
            (4, [["embedding"] + ["decoder"] * 5, ["decoder"] * 5, ["decoder"] * 5, ["decoder"] * 4 + ["loss"]]),
            (
                8,
                [
                    ["embedding"] + ["decoder"] * 3,
                    ["decoder"] * 3,
                    ["decoder"] * 3,
                    ["decoder"] * 2,
                    ["decoder"] * 2,
                    ["decoder"] * 2,
                    ["decoder"] * 2,
                    ["decoder"] * 2 + ["loss"],
                ],
            ),
        ],
    )
    def test_get_pipeline_layout_supported_sizes(self, pp, expected_layout):
        """The helper should return the known-good layouts for supported PP sizes."""
        assert _get_sarvam_moe_pipeline_layout(pp) == expected_layout

    def test_get_pipeline_layout_invalid_pp(self):
        """Unsupported PP sizes should raise a helpful error."""
        with pytest.raises(ValueError, match="Unsupported PP size 16"):
            _get_sarvam_moe_pipeline_layout(16)
