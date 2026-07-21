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

"""Unit tests for Gemma 4 vision-language providers."""

from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import (
    Gemma4DenseVLProvider,
    Gemma4VLModelProvider,
)


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
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
            image_token_id=99999,
            video_token_id=99998,
        )
        assert p.image_token_id == 99999
        assert p.video_token_id == 99998

    def test_custom_vision_tokens_per_image(self):
        p = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
            vision_soft_tokens_per_image=560,
        )
        assert p.vision_soft_tokens_per_image == 560

    def test_freeze_options_configurable(self):
        p = Gemma4VLModelProvider(
            num_layers=62,
            hidden_size=2816,
            num_attention_heads=8,
            freeze_language_model=True,
            freeze_vision_model=True,
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
