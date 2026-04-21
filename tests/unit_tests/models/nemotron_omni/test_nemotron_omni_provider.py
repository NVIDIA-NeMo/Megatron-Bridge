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

from megatron.bridge.models.nemotron_omni.nemotron_omni_provider import (
    NemotronNano3Bv3OmniModelProvider,
    NemotronNano12Bv2OmniModelProvider,
    NemotronOmniModelProvider,
)


class TestNemotronNano12Bv2OmniModelProvider:
    def test_provider_inherits_vl_defaults(self):
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
        )

        assert provider.num_layers == 28
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40

        # VL defaults inherited
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.attention_softmax_in_fp32 is True
        assert provider.vision_model_type == "radio"
        assert provider.language_model_type == "nemotron5-hybrid-12b"
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

        # Temporal compression defaults
        assert provider.video_temporal_patch_size == 1
        assert provider.separate_video_embedder is False

    def test_sound_fields_default_values(self):
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
        )

        assert provider.has_sound is False
        assert provider.sound_model_type == "parakeet"
        assert provider.sound_hidden_size == 1024
        assert provider.sound_projection_hidden_size == 4096
        assert provider.sound_context_token_id == 0
        assert provider.sound_config is None
        assert provider.freeze_sound_encoder is False
        assert provider.freeze_sound_projection is False

        # Temporal fields are independent of sound fields
        assert provider.video_temporal_patch_size == 1
        assert provider.separate_video_embedder is False

    def test_sound_fields_custom_values(self):
        sound_cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "intermediate_size": 2048,
            "num_mel_bins": 80,
            "subsampling_factor": 4,
        }
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
            has_sound=True,
            sound_model_type="parakeet",
            sound_hidden_size=512,
            sound_projection_hidden_size=2048,
            sound_context_token_id=29,
            sound_config=sound_cfg,
        )

        assert provider.has_sound is True
        assert provider.sound_hidden_size == 512
        assert provider.sound_projection_hidden_size == 2048
        assert provider.sound_context_token_id == 29
        assert provider.sound_config == sound_cfg

    def test_freeze_sound_overrides(self):
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
            freeze_sound_encoder=True,
            freeze_sound_projection=True,
        )

        assert provider.freeze_sound_encoder is True
        assert provider.freeze_sound_projection is True
        # VL freeze defaults unchanged
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False

    def test_has_provide_method(self):
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
        )
        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_has_sound_builder_methods(self):
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
        )
        assert hasattr(provider, "_build_sound_projection_config")
        assert callable(provider._build_sound_projection_config)
        assert hasattr(provider, "_build_sound_encoder")
        assert callable(provider._build_sound_encoder)


class TestNemotronNano3Bv3OmniModelProvider:
    def test_provider_initialization(self):
        provider = NemotronNano3Bv3OmniModelProvider(
            num_layers=52,
            hidden_size=2688,
            num_attention_heads=28,
        )

        assert provider.num_layers == 52
        assert provider.hidden_size == 2688
        assert provider.num_attention_heads == 28
        assert provider.language_model_type == "nemotron6-moe"

    def test_sound_fields_present(self):
        provider = NemotronNano3Bv3OmniModelProvider(
            num_layers=52,
            hidden_size=2688,
            num_attention_heads=28,
        )

        assert provider.has_sound is False
        assert provider.sound_model_type == "parakeet"
        assert provider.sound_hidden_size == 1024

        # Temporal compression defaults
        assert provider.video_temporal_patch_size == 1
        assert provider.separate_video_embedder is False


class TestTemporalCompressionFields:
    def test_temporal_fields_custom_values(self):
        provider = NemotronNano3Bv3OmniModelProvider(
            num_layers=52,
            hidden_size=2688,
            num_attention_heads=28,
            video_temporal_patch_size=2,
            separate_video_embedder=True,
        )
        assert provider.video_temporal_patch_size == 2
        assert provider.separate_video_embedder is True

    def test_temporal_fields_independent_of_sound(self):
        provider = NemotronNano12Bv2OmniModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
            video_temporal_patch_size=4,
            separate_video_embedder=True,
            has_sound=False,
        )
        assert provider.video_temporal_patch_size == 4
        assert provider.separate_video_embedder is True
        assert provider.has_sound is False

    def test_temporal_with_sound_combined(self):
        sound_cfg = {
            "hidden_size": 512,
            "num_hidden_layers": 12,
            "num_attention_heads": 4,
            "intermediate_size": 2048,
            "num_mel_bins": 80,
            "subsampling_factor": 4,
        }
        provider = NemotronNano3Bv3OmniModelProvider(
            num_layers=52,
            hidden_size=2688,
            num_attention_heads=28,
            has_sound=True,
            sound_config=sound_cfg,
            video_temporal_patch_size=2,
            separate_video_embedder=True,
        )
        assert provider.has_sound is True
        assert provider.video_temporal_patch_size == 2
        assert provider.separate_video_embedder is True


class TestOmniModelProviderIsAbstract:
    def test_base_class_is_abstract(self):
        assert issubclass(NemotronOmniModelProvider, NemotronOmniModelProvider)
        # NemotronOmniModelProvider has ABC in MRO
        from abc import ABC
        assert ABC in NemotronOmniModelProvider.__mro__
