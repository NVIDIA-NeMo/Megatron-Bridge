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

import math
from unittest.mock import Mock, patch

import pytest
import torch
from megatron.core.activations import fast_gelu
from megatron.core.transformer.enums import AttnMaskType
from transformers import Gemma2Config

from megatron.bridge.models.gemma.gemma2_provider import (
    Gemma2DotProductAttention,
    Gemma2FlexDotProductAttention,
    Gemma2ModelProvider,
)
from megatron.bridge.utils.fusions import can_enable_gradient_accumulation_fusion


class TestGemma2ModelProvider:
    """Test cases for base Gemma2ModelProvider class."""

    def test_gemma2_model_provider_initialization(self):
        """Test Gemma2ModelProvider can be initialized with default values."""
        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        # Check required transformer config fields
        assert provider.num_layers == 26
        assert provider.hidden_size == 2304
        assert provider.num_attention_heads == 8

        # Check Gemma2-specific defaults
        assert provider.normalization == "RMSNorm"
        assert provider.activation_func == fast_gelu
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.seq_length == 8192
        assert provider.kv_channels == 256
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True
        assert provider.layernorm_zero_centered_gamma is True

        # Check Gemma2-specific parameters
        assert provider.layernorm_epsilon == 1e-6
        assert provider.rotary_base == 10000
        assert provider.window_size == (4095, 0)
        assert provider.vocab_size == 256000
        assert provider.gradient_accumulation_fusion is can_enable_gradient_accumulation_fusion()
        assert provider.query_pre_attn_scalar == 224
        assert provider.attn_logit_softcapping == 50.0
        assert provider.final_logit_softcapping == 30.0

    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_last_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_last_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.extend_instance")
    def test_gemma2_provider_provide_with_embedding_scaling(self, mock_extend_instance, *_):
        """Test that provide method applies embedding scaling when appropriate."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()

        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called for embedding scaling
            assert mock_extend_instance.call_count == 1
            args = mock_extend_instance.call_args_list[0][0]
            assert args[0] == mock_model.embedding

    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_first_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_first_stage", return_value=False)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.extend_instance")
    def test_gemma2_provider_provide_with_output_layer_scaling(self, mock_extend_instance, *_):
        """Test that provide method applies output layer modifications when appropriate."""
        # Mock the parent provide method
        mock_model = Mock()
        mock_model.embedding = Mock()
        mock_model.output_layer = Mock()

        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            # Use vp_stage=0 to satisfy vp_size None assertion in helpers
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called for output layer modifications
            assert mock_extend_instance.call_count == 1
            args = mock_extend_instance.call_args_list[0][0]
            assert args[0] == mock_model.output_layer

    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_pp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_first_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.is_vp_last_stage", return_value=True)
    @patch("megatron.bridge.models.gemma.gemma2_provider.extend_instance")
    def test_gemma2_provider_provide_both_stages(self, mock_extend_instance, *_):
        """Test provide method when model is both first and last stage."""
        mock_model = Mock()
        mock_model.embedding = Mock()
        mock_model.output_layer = Mock()

        provider = Gemma2ModelProvider(
            num_layers=26,
            hidden_size=2304,
            num_attention_heads=8,
        )

        provider._pg_collection = type("PG", (), {"pp": object()})()

        with patch.object(provider.__class__.__bases__[0], "provide", return_value=mock_model):
            result = provider.provide(vp_stage=0)

            # Verify that parent provide was called
            assert result == mock_model

            # Verify that extend_instance was called twice (embedding + output layer)
            assert mock_extend_instance.call_count == 2


class TestGemma2ModelProviderIntegration:
    """Integration tests for Gemma2 model providers."""

    def test_provider_accepts_explicit_architecture_values(self):
        """Test that architecture values can be supplied without size subclasses."""
        providers = [
            Gemma2ModelProvider(
                num_layers=26,
                hidden_size=2304,
                num_attention_heads=8,
                num_query_groups=4,
                ffn_hidden_size=9216,
                query_pre_attn_scalar=256,
            ),
            Gemma2ModelProvider(
                num_layers=42,
                hidden_size=3584,
                num_attention_heads=16,
                num_query_groups=8,
                ffn_hidden_size=14336,
                query_pre_attn_scalar=256,
            ),
            Gemma2ModelProvider(
                num_layers=46,
                hidden_size=4608,
                num_attention_heads=32,
                num_query_groups=16,
                kv_channels=128,
                ffn_hidden_size=36864,
                query_pre_attn_scalar=144,
            ),
        ]

        for provider in providers:
            assert isinstance(provider, Gemma2ModelProvider)
            assert hasattr(provider, "provide")
            assert callable(getattr(provider, "provide"))
            assert provider.normalization == "RMSNorm"
            assert provider.activation_func == fast_gelu
            assert provider.gated_linear_unit is True


def _make_attention(
    context_parallel_size: int = 1,
    window_size: tuple = (4095, 0),
    layer_number: int = 1,
) -> Gemma2DotProductAttention:
    """Build a Gemma2DotProductAttention with minimal mock config."""
    config = Mock()
    config.context_parallel_size = context_parallel_size
    config.window_size = window_size
    config.kv_channels = 32  # matches head_dim=32 used in forward() tests
    config.num_attention_heads = 8
    config.num_query_groups = 8
    config.tensor_model_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.query_pre_attn_scalar = 224
    config.fp16 = False
    config.bf16 = True
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.sequence_parallel = False
    config.attn_logit_softcapping = 0.0  # disable softcapping in unit tests
    return Gemma2DotProductAttention(
        config=config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
    )


class TestGemma2DotProductAttention:
    """Tests for Gemma2DotProductAttention fixes."""

    @pytest.mark.parametrize("layer_number", [1, 2])
    def test_layer_attention_pattern_matches_hf_config(self, layer_number):
        """MCore's one-based layers must preserve HF's zero-based attention pattern."""
        hf_config = Gemma2Config(num_hidden_layers=2)

        attention = _make_attention(layer_number=layer_number)

        hf_layer_type = hf_config.layer_types[layer_number - 1]
        expected_window = (4095, 0) if hf_layer_type == "sliding_attention" else None
        assert attention.window_size == expected_window

    def test_cp_greater_than_1_raises_value_error(self):
        """CP > 1 must raise ValueError, not bare AssertionError."""
        with pytest.raises(ValueError, match="Context parallelism"):
            _make_attention(context_parallel_size=2)

    def test_packed_seq_raises_value_error(self):
        """packed_seq_params != None must raise ValueError."""
        attn = _make_attention()
        dummy = torch.zeros(4, 8, 8)
        with pytest.raises(ValueError, match="Packed sequence"):
            attn.forward(
                query=dummy,
                key=dummy,
                value=dummy,
                attention_mask=None,
                packed_seq_params=Mock(),
            )

    def test_swa_applied_when_attention_mask_is_none(self):
        """SWA mask must be generated even when attention_mask=None (the pretrain path).

        Prior to the fix, the gate was:
            if attention_mask is not None and self.window_size is not None:
        which was never True on the pretrain path (MCore passes attention_mask=None).
        The gate is now simply:
            if self.window_size is not None:
        The mask is always built for SWA layers — omitting it when the window covers the
        full sequence would drop causal masking entirely because attn_mask_type=arbitrary
        routes through ScaledSoftmax (plain softmax, no causal mask) when mask=None.
        get_swa() degenerates to a pure causal mask when the window covers all positions.
        """
        seq, batch, heads, head_dim = 4, 1, 8, 32
        attn = _make_attention(window_size=(2, 0))
        assert attn.window_size == (2, 0), "SWA layer must have window_size set"

        # Use a real bool tensor so the unsqueeze chain produces a verifiable result.
        swa_tensor = torch.zeros(seq, seq, dtype=torch.bool)
        # scale_mask_softmax is a registered nn.Module submodule; patch its forward method
        # rather than replacing the whole module (which PyTorch rejects with TypeError).
        attn.attention_dropout = torch.nn.Identity()
        mock_softmax_fwd = Mock(return_value=torch.zeros(1, 8, 4, 4))
        q = torch.zeros(seq, batch, heads, head_dim)
        k = torch.zeros(seq, batch, heads, head_dim)
        v = torch.zeros(seq, batch, heads, head_dim)
        buf = torch.zeros(batch * heads, seq, seq)
        with (
            patch("megatron.bridge.models.gemma.gemma2_provider.get_swa", return_value=swa_tensor) as mock_get_swa,
            patch.object(attn.scale_mask_softmax, "forward", mock_softmax_fwd),
            patch("megatron.bridge.models.gemma.gemma2_provider.parallel_state") as mock_ps,
            patch("megatron.bridge.models.gemma.gemma2_provider.tensor_parallel") as mock_tp,
        ):
            mock_ps.get_global_memory_buffer.return_value.get_tensor.return_value = buf
            mock_tp.get_cuda_rng_tracker.return_value.fork.return_value.__enter__ = lambda s: None
            mock_tp.get_cuda_rng_tracker.return_value.fork.return_value.__exit__ = Mock(return_value=False)
            attn.forward(query=q, key=k, value=v, attention_mask=None)

        mock_get_swa.assert_called_once_with(seq, seq, (2, 0))
        # The SWA mask is unsqueezed to [1, 1, sq, sk] before being passed to scale_mask_softmax.
        call_args = mock_softmax_fwd.call_args
        expected_mask = swa_tensor.unsqueeze(0).unsqueeze(0)
        assert torch.equal(call_args[0][1], expected_mask), (
            "scale_mask_softmax must receive the SWA mask unsqueezed to [1, 1, sq, sk]"
        )

    def test_even_layer_has_no_swa(self):
        """Even-numbered layers must not have a window_size (full attention)."""
        config = Mock()
        config.context_parallel_size = 1
        config.window_size = (4095, 0)
        config.kv_channels = 256
        config.num_attention_heads = 8
        config.num_query_groups = 8
        config.tensor_model_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.query_pre_attn_scalar = 224
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.0
        config.sequence_parallel = False
        even_attn = Gemma2DotProductAttention(
            config=config,
            layer_number=2,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
        )
        assert even_attn.window_size is None
        assert even_attn.attn_mask_type == AttnMaskType.causal
        assert even_attn.scale_mask_softmax.attn_mask_type == AttnMaskType.causal

    def test_swa_layer_uses_arbitrary_mask_type(self):
        """Odd-numbered (SWA) layers must override attn_mask_type to arbitrary.

        FusedScaleMaskSoftmax with AttnMaskType.causal takes the ScaledUpperTriangMaskedSoftmax
        path which silently ignores the mask argument. Switching to arbitrary routes through
        ScaledMaskedSoftmax, which correctly applies the externally generated SWA mask.
        Even-numbered layers must keep AttnMaskType.causal to retain the fast fused path.
        """
        swa_attn = _make_attention(window_size=(4095, 0))
        assert swa_attn.attn_mask_type == AttnMaskType.arbitrary, (
            "SWA layers must use AttnMaskType.arbitrary so FusedScaleMaskSoftmax "
            "routes through ScaledMaskedSoftmax and applies the mask"
        )
        # Also verify the FusedScaleMaskSoftmax instance stored the right type
        assert swa_attn.scale_mask_softmax.attn_mask_type == AttnMaskType.arbitrary

    def test_swa_combined_with_padding_mask(self):
        """When a padding mask is present, forward() must OR it with the SWA mask.

        Prior to this fix, the forward() code was:
            attention_mask = get_swa(...)
        which silently discarded any incoming padding mask. The correct behaviour is:
            attention_mask = swa_mask if attention_mask is None else (swa_mask | attention_mask)
        Both masks use True=masked-out, so logical OR gives the union of blocked positions.
        """
        attn = _make_attention(window_size=(2, 0))

        seq, batch, heads, head_dim = 4, 2, 8, 32
        # Padding mask [b, 1, sq, sk]: block last key-position for the first sample only.
        padding_mask = torch.zeros(batch, 1, seq, seq, dtype=torch.bool)
        padding_mask[0, 0, :, -1] = True

        # SWA mask returned by the patched get_swa: first column masked to give a non-trivial OR.
        swa_mask_val = torch.zeros(seq, seq, dtype=torch.bool)
        swa_mask_val[:, 0] = True  # mask first key-position for all queries

        captured: dict = {}

        def fake_forward(scores, mask):
            captured["mask"] = mask
            return torch.zeros(batch, heads, seq, seq)

        # scale_mask_softmax is a registered nn.Module submodule; patch its forward method
        # rather than replacing the whole module (which PyTorch rejects with TypeError).
        attn.attention_dropout = torch.nn.Identity()
        q = torch.zeros(seq, batch, heads, head_dim)
        k = torch.zeros(seq, batch, heads, head_dim)
        v = torch.zeros(seq, batch, heads, head_dim)
        buf = torch.zeros(batch * heads, seq, seq)
        with (
            patch("megatron.bridge.models.gemma.gemma2_provider.get_swa", return_value=swa_mask_val) as mock_get_swa,
            patch.object(attn.scale_mask_softmax, "forward", side_effect=fake_forward),
            patch("megatron.bridge.models.gemma.gemma2_provider.parallel_state") as mock_ps,
            patch("megatron.bridge.models.gemma.gemma2_provider.tensor_parallel") as mock_tp,
        ):
            mock_ps.get_global_memory_buffer.return_value.get_tensor.return_value = buf
            mock_tp.get_cuda_rng_tracker.return_value.fork.return_value.__enter__ = lambda s: None
            mock_tp.get_cuda_rng_tracker.return_value.fork.return_value.__exit__ = Mock(return_value=False)
            attn.forward(query=q, key=k, value=v, attention_mask=padding_mask)

        mock_get_swa.assert_called_once_with(seq, seq, (2, 0))
        expected = swa_mask_val | padding_mask  # [sq, sk] | [b, 1, sq, sk] → [b, 1, sq, sk]
        assert torch.equal(captured["mask"], expected), (
            "scale_mask_softmax must receive swa_mask | padding_mask, not just swa_mask"
        )

    def test_window_size_default_is_4095(self):
        """Gemma2ModelProvider.window_size default must be (4095, 0) to match gemma2_bridge convention."""
        provider = Gemma2ModelProvider(
            num_layers=42,
            hidden_size=3584,
            num_attention_heads=16,
        )
        assert provider.window_size == (4095, 0)


def _make_flex_attention(layer_number: int = 1) -> Gemma2FlexDotProductAttention:
    """Build a Gemma2FlexDotProductAttention with minimal mock config."""
    config = Mock()
    config.context_parallel_size = 1
    config.window_size = (4095, 0)
    config.kv_channels = 256
    config.num_attention_heads = 8
    config.num_query_groups = 8
    config.tensor_model_parallel_size = 1
    config.apply_query_key_layer_scaling = False
    config.query_pre_attn_scalar = 224
    config.fp16 = False
    config.bf16 = True
    config.masked_softmax_fusion = False
    config.attention_softmax_in_fp32 = True
    config.attention_dropout = 0.0
    config.sequence_parallel = False
    config.attn_logit_softcapping = 50.0
    return Gemma2FlexDotProductAttention(
        config=config,
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
    )


class TestGemma2FlexDotProductAttention:
    """Tests for Gemma2FlexDotProductAttention FlexAttention fast path and fallback behavior."""

    def test_flex_path_used_when_available(self):
        """When _HAVE_FLEX_ATTN=True and mask=None, _flex_attn_func must be called, not baddbmm."""
        seq, batch, heads, head_dim = 4, 2, 8, 32
        # FlexAttention output: [b, np, sq, hn]
        flex_out = torch.zeros(batch, heads, seq, head_dim)
        mock_flex = Mock(return_value=flex_out)
        mock_block_mask = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=mock_block_mask,
            ),
        ):
            attn = _make_flex_attention(layer_number=1)
            q = torch.zeros(seq, batch, heads, head_dim)
            k = torch.zeros(seq, batch, heads, head_dim)
            v = torch.zeros(seq, batch, heads, head_dim)
            out = attn.forward(query=q, key=k, value=v, attention_mask=None)

        mock_flex.assert_called_once()
        assert out.shape == (seq, batch, heads * head_dim)

    def test_fallback_when_no_flex(self):
        """When _HAVE_FLEX_ATTN=False, forward must delegate entirely to the unfused parent path."""
        attn = _make_flex_attention(layer_number=1)
        mock_flex = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", False),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
        ):
            with patch.object(
                Gemma2DotProductAttention, "forward", return_value=torch.zeros(4, 2, 256)
            ) as mock_parent:
                q = torch.zeros(4, 2, 8, 32)
                k = torch.zeros(4, 2, 8, 32)
                v = torch.zeros(4, 2, 8, 32)
                attn.forward(query=q, key=k, value=v, attention_mask=None)

        mock_flex.assert_not_called()
        mock_parent.assert_called_once()

    def test_fallback_when_dropout_nonzero(self):
        """Non-zero dropout must trigger unfused fallback: FlexAttention has no dropout_p param."""
        config = Mock()
        config.context_parallel_size = 1
        config.window_size = (4095, 0)
        config.kv_channels = 256
        config.num_attention_heads = 8
        config.num_query_groups = 8
        config.tensor_model_parallel_size = 1
        config.apply_query_key_layer_scaling = False
        config.query_pre_attn_scalar = 224
        config.fp16 = False
        config.bf16 = True
        config.masked_softmax_fusion = False
        config.attention_softmax_in_fp32 = True
        config.attention_dropout = 0.1  # non-zero dropout
        config.sequence_parallel = False
        config.attn_logit_softcapping = 50.0
        mock_flex = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
        ):
            attn = Gemma2FlexDotProductAttention(
                config=config, layer_number=1, attn_mask_type=AttnMaskType.causal, attention_type="self"
            )
            attn.train()  # ensure self.training=True so dropout_p > 0
            with patch.object(
                Gemma2DotProductAttention, "forward", return_value=torch.zeros(4, 2, 256)
            ) as mock_parent:
                q = torch.zeros(4, 2, 8, 32)
                k = torch.zeros(4, 2, 8, 32)
                v = torch.zeros(4, 2, 8, 32)
                attn.forward(query=q, key=k, value=v, attention_mask=None)

        mock_flex.assert_not_called()
        mock_parent.assert_called_once()

    def test_flex_call_has_no_dropout_p_kwarg(self):
        """FlexAttention call must NOT include a dropout_p kwarg — it has no such parameter."""
        seq, batch, heads, head_dim = 4, 2, 8, 32
        flex_out = torch.zeros(batch, heads, seq, head_dim)
        mock_flex = Mock(return_value=flex_out)
        mock_block_mask = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=mock_block_mask,
            ),
        ):
            attn = _make_flex_attention(layer_number=1)
            q = torch.zeros(seq, batch, heads, head_dim)
            k = torch.zeros(seq, batch, heads, head_dim)
            v = torch.zeros(seq, batch, heads, head_dim)
            attn.forward(query=q, key=k, value=v, attention_mask=None)

        assert "dropout_p" not in mock_flex.call_args.kwargs, (
            "dropout_p must not be passed to FlexAttention: it has no such parameter."
        )

    def test_fallback_when_attention_mask_not_none(self):
        """Non-None attention_mask (fine-tuning) must trigger unfused fallback even with FlexAttention present."""
        attn = _make_flex_attention(layer_number=1)
        mock_flex = Mock()
        padding_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
        ):
            with patch.object(
                Gemma2DotProductAttention, "forward", return_value=torch.zeros(4, 2, 256)
            ) as mock_parent:
                q = torch.zeros(4, 2, 8, 32)
                k = torch.zeros(4, 2, 8, 32)
                v = torch.zeros(4, 2, 8, 32)
                attn.forward(query=q, key=k, value=v, attention_mask=padding_mask)

        mock_flex.assert_not_called()
        mock_parent.assert_called_once()

    def test_score_mod_encodes_softcap(self):
        """FlexAttention must be called with a score_mod that applies the Gemma2 softcap of 50.0.

        Softcap is now passed via score_mod (a callable closure), not as a raw kwarg.
        A regression that omits the score_mod would silently produce uncapped logits and
        diverge from the reference Gemma2 implementation.
        """
        seq, batch, heads, head_dim = 4, 2, 8, 32
        flex_out = torch.zeros(batch, heads, seq, head_dim)
        mock_flex = Mock(return_value=flex_out)
        mock_block_mask = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=mock_block_mask,
            ),
        ):
            attn = _make_flex_attention(layer_number=1)
            q = torch.zeros(seq, batch, heads, head_dim)
            k = torch.zeros(seq, batch, heads, head_dim)
            v = torch.zeros(seq, batch, heads, head_dim)
            attn.forward(query=q, key=k, value=v, attention_mask=None)

        score_mod = mock_flex.call_args.kwargs["score_mod"]
        assert callable(score_mod), "score_mod must be a callable"
        test_score = torch.tensor(1.0)
        result = score_mod(test_score, None, None, None, None)
        expected = 50.0 * torch.tanh(test_score / 50.0)
        assert torch.allclose(result, expected), (
            f"score_mod must apply 50.0 * tanh(score / 50.0), got {result:.6f} expected {expected:.6f}"
        )

    def test_score_mod_shared_across_layers(self):
        """All layers with the same softcap must share one score_mod object (lru_cache invariant).

        torch.compile guards on id(fn): if each layer has a unique score_mod object, compile
        recompiles N times at startup. This test verifies the sharing invariant so a regression
        (e.g. moving the factory inside __init__) is caught immediately.
        """
        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=Mock(),
            ),
        ):
            attn1 = _make_flex_attention(layer_number=1)
            attn2 = _make_flex_attention(layer_number=3)
        assert attn1._flex_score_mod is attn2._flex_score_mod, (
            "Layers with the same softcap must share one score_mod object to avoid torch.compile recompilations"
        )

    def test_output_shape_flex_path(self):
        """FlexAttention path output shape must be [sq, b, num_heads * head_dim]."""
        seq, batch, heads, head_dim = 6, 3, 8, 32
        flex_out = torch.zeros(batch, heads, seq, head_dim)
        mock_flex = Mock(return_value=flex_out)
        mock_block_mask = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=mock_block_mask,
            ),
        ):
            attn = _make_flex_attention(layer_number=1)
            q = torch.zeros(seq, batch, heads, head_dim)
            k = torch.zeros(seq, batch, heads, head_dim)
            v = torch.zeros(seq, batch, heads, head_dim)
            out = attn.forward(query=q, key=k, value=v, attention_mask=None)

        assert out.shape == (seq, batch, heads * head_dim)

    def test_softmax_scale_passed_to_flex(self):
        """FlexAttention must receive scale=1/sqrt(224), not the default 1/sqrt(head_dim).

        Gemma2 uses query_pre_attn_scalar=224 for the attention scale, not the usual
        head_dim (256 for kv_channels=256). The wrong scale silently produces incorrect
        attention weights and loss spikes.
        """
        seq, batch, heads, head_dim = 4, 1, 8, 32
        flex_out = torch.zeros(batch, heads, seq, head_dim)
        mock_flex = Mock(return_value=flex_out)
        mock_block_mask = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=mock_block_mask,
            ),
        ):
            attn = _make_flex_attention(layer_number=1)
            q = torch.zeros(seq, batch, heads, head_dim)
            k = torch.zeros(seq, batch, heads, head_dim)
            v = torch.zeros(seq, batch, heads, head_dim)
            attn.forward(query=q, key=k, value=v, attention_mask=None)

        expected_scale = 1.0 / math.sqrt(224)
        actual_scale = mock_flex.call_args.kwargs["scale"]
        assert abs(actual_scale - expected_scale) < 1e-9, (
            f"scale must be 1/sqrt(224)={expected_scale:.6f}, got {actual_scale:.6f}. "
            "FlexAttention's default 1/sqrt(head_dim) would be wrong for Gemma2."
        )

    def test_swa_layer_passes_block_mask(self):
        """For odd-numbered (SWA) layers, FlexAttention must receive a block_mask.

        SWA is encoded in the block_mask built from _flex_window_size=(4095, 0). A regression
        that omits block_mask would silently compute full causal attention instead of SWA.
        """
        seq, batch, heads, head_dim = 4, 2, 8, 32
        flex_out = torch.zeros(batch, heads, seq, head_dim)
        mock_flex = Mock(return_value=flex_out)
        mock_block_mask = Mock()

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
            patch(
                "megatron.bridge.models.gemma.gemma2_provider._create_flex_block_mask",
                return_value=mock_block_mask,
            ),
        ):
            attn = _make_flex_attention(layer_number=1)
            assert attn._flex_window_size == (4095, 0), "SWA layer must have _flex_window_size=(4095, 0)"
            q = torch.zeros(seq, batch, heads, head_dim)
            k = torch.zeros(seq, batch, heads, head_dim)
            v = torch.zeros(seq, batch, heads, head_dim)
            attn.forward(query=q, key=k, value=v, attention_mask=None)

        mock_flex.assert_called_once()
        assert mock_flex.call_args.kwargs.get("block_mask") is mock_block_mask, (
            "SWA layer must pass block_mask to FlexAttention — omitting it silently disables SWA."
        )

    def test_swa_layer_fallback_with_padding_mask(self):
        """An odd-numbered (SWA) layer receiving a non-None mask must fall back to the
        unfused parent, even when FlexAttention is available.

        The unfused parent OR-combines the SWA mask with the padding mask via get_swa().
        The FlexAttention path only runs when attention_mask is None.
        """
        attn = _make_flex_attention(layer_number=1)
        mock_flex = Mock()
        padding_mask = torch.zeros(2, 1, 4, 4, dtype=torch.bool)

        with (
            patch("megatron.bridge.models.gemma.gemma2_provider._HAVE_FLEX_ATTN", True),
            patch("megatron.bridge.models.gemma.gemma2_provider._flex_attn_func", mock_flex),
        ):
            with patch.object(
                Gemma2DotProductAttention, "forward", return_value=torch.zeros(4, 2, 256)
            ) as mock_parent:
                q = torch.zeros(4, 2, 8, 32)
                k = torch.zeros(4, 2, 8, 32)
                v = torch.zeros(4, 2, 8, 32)
                attn.forward(query=q, key=k, value=v, attention_mask=padding_mask)

        mock_flex.assert_not_called()
        mock_parent.assert_called_once()

    def test_packed_seq_raises_value_error_on_flex_subclass(self):
        """Gemma2FlexDotProductAttention must raise ValueError for packed_seq_params,
        independent of the parent class check.

        Removing this guard would let packed sequences silently reach the FlexAttention
        path, which does not support them.
        """
        attn = _make_flex_attention(layer_number=1)
        dummy = torch.zeros(4, 2, 8, 32)
        with pytest.raises(ValueError, match="Packed sequence"):
            attn.forward(
                query=dummy,
                key=dummy,
                value=dummy,
                attention_mask=None,
                packed_seq_params=Mock(),
            )
