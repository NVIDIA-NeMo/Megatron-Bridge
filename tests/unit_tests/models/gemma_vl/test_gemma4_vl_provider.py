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


class TestInstallTiedKV:
    """Tests for _install_tied_kv layer marking behavior."""

    def test_install_tied_kv_skips_dense_model(self):
        """_install_tied_kv does nothing when num_moe_experts is None."""
        from megatron.bridge.models.gemma.gemma4_provider import (
            Gemma4ModelProvider,
            _install_tied_kv,
        )

        provider = Gemma4ModelProvider(
            num_layers=6,
            hidden_size=64,
            num_attention_heads=4,
        )
        provider.num_moe_experts = None  # Dense model

        class FakeLayer:
            layer_number = 1

        class FakeModel:
            class decoder:
                layers = [FakeLayer()]

        _install_tied_kv(FakeModel(), provider)
        # No _tied_kv flag should be set since it's a dense model
        assert not getattr(FakeLayer, "_tied_kv", False)

    def test_install_tied_kv_marks_global_layers(self):
        """_install_tied_kv sets _tied_kv=True on global attention modules only."""
        import torch.nn as nn

        from megatron.bridge.models.gemma.gemma4_provider import (
            Gemma4ModelProvider,
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
            has_flag = getattr(layer.self_attention, "_tied_kv", False)
            assert has_flag == is_global, f"Layer {layer.layer_number}: expected _tied_kv={is_global}, got {has_flag}"
