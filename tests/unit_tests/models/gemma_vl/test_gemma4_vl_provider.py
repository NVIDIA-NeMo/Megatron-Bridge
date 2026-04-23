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

import pytest

from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider


class TestGemma4VLModelProviderDefaults:
    """Test Gemma4VLModelProvider default values and inheritance."""

    def test_initialization(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
        )
        assert isinstance(provider, Gemma4VLModelProvider)
        assert isinstance(provider, Gemma4ModelProvider)

    def test_vl_defaults(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
        )
        # VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.vision_soft_tokens_per_image == 280
        assert provider.bos_token_id == 2
        assert provider.eos_token_id == 1
        assert provider.image_token_id == 258_880
        assert provider.video_token_id == 258_884

    def test_freeze_defaults(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
        )
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    def test_vision_config_defaults_to_none(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
        )
        assert provider.vision_config is None
        assert provider.text_config is None

    def test_inherited_gemma4_defaults(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
        )
        # Inherited from Gemma4ModelProvider
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.position_embedding_type == "rope"
        assert provider.add_bias_linear is False
        assert provider.attention_dropout == 0.0
        assert provider.hidden_dropout == 0.0
        assert provider.share_embeddings_and_output_weights is True

    def test_custom_token_ids(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
            image_token_id=99999,
            video_token_id=99998,
        )
        assert provider.image_token_id == 99999
        assert provider.video_token_id == 99998

    def test_custom_vision_tokens_per_image(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
            vision_soft_tokens_per_image=560,
        )
        assert provider.vision_soft_tokens_per_image == 560

    def test_freeze_options_configurable(self):
        provider = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
            freeze_language_model=True,
            freeze_vision_model=True,
        )
        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is False

    def test_different_hidden_sizes(self):
        for hidden_size in [1152, 2048, 2816, 4096]:
            provider = Gemma4VLModelProvider(
                num_layers=28,
                hidden_size=hidden_size,
                num_attention_heads=8,
            )
            assert provider.hidden_size == hidden_size

    def test_different_layer_counts(self):
        for num_layers in [18, 28, 46, 62]:
            provider = Gemma4VLModelProvider(
                num_layers=num_layers,
                hidden_size=2816,
                num_attention_heads=8,
            )
            assert provider.num_layers == num_layers


class TestGemma4TiedKVMixin:
    """Tests for Gemma4TiedKVMixin K=V weight tying behavior."""

    def _make_fake_linear(self, q_total, kv_total, hidden):
        """Create a minimal fake ColumnParallelLinear stub for testing."""
        import torch
        import torch.nn as nn

        class FakeLinear(nn.Module):
            def __init__(self, q_total, kv_total, hidden):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(q_total + 2 * kv_total, hidden))

            def forward(self, x):
                out = x @ self.weight.T
                return out, None  # mimic TE (tensor, bias) tuple

        return FakeLinear(q_total, kv_total, hidden)

    def test_mixin_ties_v_to_k_tp1(self):
        """K slice in output equals V slice (same tensor object)."""
        import torch
        from megatron.bridge.models.gemma.gemma4_provider import Gemma4TiedKVMixin
        from megatron.bridge.models.gemma.modules import extend_instance

        q_total, kv_total, hidden = 16, 4, 8
        linear = self._make_fake_linear(q_total, kv_total, hidden)
        linear._tied_kv_q_size = q_total
        linear._tied_kv_kv_size = kv_total
        extend_instance(linear, Gemma4TiedKVMixin)

        x = torch.randn(2, hidden)
        out, bias = linear(x)

        assert out.shape[-1] == q_total + 2 * kv_total
        # K and V slices must be equal (K is reused)
        k_slice = out[..., q_total : q_total + kv_total]
        v_slice = out[..., q_total + kv_total :]
        assert torch.allclose(k_slice, v_slice)

    def test_mixin_gradient_accumulates_in_k(self):
        """dL/dW_K accumulates from both K and V paths; V rows get zero grad."""
        import torch
        import torch.nn as nn
        from megatron.bridge.models.gemma.gemma4_provider import Gemma4TiedKVMixin
        from megatron.bridge.models.gemma.modules import extend_instance

        q_total, kv_total, hidden = 8, 4, 6
        linear = self._make_fake_linear(q_total, kv_total, hidden)
        linear._tied_kv_q_size = q_total
        linear._tied_kv_kv_size = kv_total
        extend_instance(linear, Gemma4TiedKVMixin)

        # Run a forward + backward
        x = torch.randn(3, hidden, requires_grad=True)
        out, _ = linear(x)
        loss = out.sum()
        loss.backward()

        assert linear.weight.grad is not None

        k_rows = linear.weight.grad[q_total : q_total + kv_total]
        v_rows = linear.weight.grad[q_total + kv_total :]

        # K rows must have non-zero gradient (they serve double duty)
        assert k_rows.abs().max() > 0, "K rows should have non-zero gradient"
        # V rows are never used in forward → zero gradient
        assert v_rows.abs().max() == 0, "V rows should have zero gradient"

    def test_mixin_output_shape_consistent(self):
        """Output shape equals q_total + 2 * kv_total regardless of input shape."""
        import torch
        from megatron.bridge.models.gemma.gemma4_provider import Gemma4TiedKVMixin
        from megatron.bridge.models.gemma.modules import extend_instance

        q_total, kv_total, hidden = 12, 4, 8
        linear = self._make_fake_linear(q_total, kv_total, hidden)
        linear._tied_kv_q_size = q_total
        linear._tied_kv_kv_size = kv_total
        extend_instance(linear, Gemma4TiedKVMixin)

        for seq_len in [1, 5, 64]:
            x = torch.randn(seq_len, hidden)
            out, _ = linear(x)
            assert out.shape == (seq_len, q_total + 2 * kv_total)

    def test_mixin_handles_layernorm_fused_output(self):
        """Mixin works when TE returns ((qkv, ln_out), bias) — the LayerNormColumnParallelLinear pattern."""
        import torch
        import torch.nn as nn
        from megatron.bridge.models.gemma.gemma4_provider import Gemma4TiedKVMixin
        from megatron.bridge.models.gemma.modules import extend_instance

        q_total, kv_total, hidden = 16, 4, 8

        class LayerNormFusedLinear(nn.Module):
            """Mimics TE LayerNormColumnParallelLinear: returns ((out, ln_out), bias)."""
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(q_total + 2 * kv_total, hidden))

            def forward(self, x):
                out = x @ self.weight.T
                ln_out = x  # dummy layernorm output
                return (out, ln_out), None  # pattern 3: ((tensor, ln_out), bias)

        linear = LayerNormFusedLinear()
        linear._tied_kv_q_size = q_total
        linear._tied_kv_kv_size = kv_total
        extend_instance(linear, Gemma4TiedKVMixin)

        x = torch.randn(2, hidden)
        result = linear(x)

        # Must return a tuple: ((tied, ln_out), bias)
        assert isinstance(result, tuple), "Expected tuple output"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
        inner, bias = result
        assert isinstance(inner, tuple), "Expected inner tuple (tied, ln_out)"
        assert len(inner) == 2, f"Expected 2-element inner tuple, got {len(inner)}"
        tied, ln_out = inner

        # Shape and K=V correctness
        assert tied.shape[-1] == q_total + 2 * kv_total
        k_slice = tied[..., q_total : q_total + kv_total]
        v_slice = tied[..., q_total + kv_total :]
        assert torch.allclose(k_slice, v_slice)
        # layernorm_output should be passed through unchanged
        assert ln_out is x

    def test_install_tied_kv_skips_dense_model(self):
        """_install_tied_kv does nothing when num_moe_experts is None."""
        import torch.nn as nn
        from megatron.bridge.models.gemma.gemma4_provider import (
            Gemma4ModelProvider,
            Gemma4TiedKVMixin,
            _install_tied_kv,
        )

        provider = Gemma4ModelProvider(
            num_layers=6,
            hidden_size=64,
            num_attention_heads=4,
        )
        provider.num_moe_experts = None  # Dense model

        # Minimal fake model with one layer
        class FakeLayer:
            layer_number = 1

        class FakeModel:
            class decoder:
                layers = [FakeLayer()]

        _install_tied_kv(FakeModel(), provider)
        # No mixin should be installed since it's a dense model
        assert not isinstance(FakeLayer, Gemma4TiedKVMixin)

    def test_install_tied_kv_skips_sliding_layers(self):
        """_install_tied_kv only patches global attention layers."""
        import torch.nn as nn
        from megatron.bridge.models.gemma.gemma4_provider import (
            Gemma4ModelProvider,
            Gemma4TiedKVMixin,
            _install_tied_kv,
        )

        provider = Gemma4ModelProvider(
            num_layers=6,
            hidden_size=64,
            num_attention_heads=4,
            num_global_key_value_heads=2,
            global_head_dim=16,
            interleaved_attn_pattern=(5, 1),  # layers 1-5 sliding, layer 6 global
            num_moe_experts=4,
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
            is_global = layer.layer_number == 6  # pattern (5,1): layer 6 is global
            has_mixin = isinstance(
                layer.self_attention.linear_qkv, Gemma4TiedKVMixin
            )
            assert has_mixin == is_global, (
                f"Layer {layer.layer_number}: expected mixin={is_global}, got {has_mixin}"
            )
