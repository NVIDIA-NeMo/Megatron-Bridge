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

from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import (
    NemotronNano12Bv2VLModelProvider,
)


class TestNemotronNano12Bv2VLModelProvider:
    def test_provider_initialization_minimal(self):
        provider = NemotronNano12Bv2VLModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
        )

        # Core fields
        assert provider.num_layers == 28
        assert provider.hidden_size == 5120
        assert provider.num_attention_heads == 40

        # VL-specific defaults
        assert provider.scatter_embedding_sequence_parallel is False
        assert provider.attention_softmax_in_fp32 is True
        assert provider.vision_model_type == "radio"
        assert provider.language_model_type == "nemotron5-hybrid-12b"

        # Freeze defaults
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

        # API presence
        assert hasattr(provider, "provide")
        assert callable(provider.provide)
        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)

    def test_provider_freeze_overrides(self):
        provider = NemotronNano12Bv2VLModelProvider(
            num_layers=28,
            hidden_size=5120,
            num_attention_heads=40,
            freeze_language_model=True,
            freeze_vision_model=True,
            freeze_vision_projection=True,
        )

        assert provider.freeze_language_model is True
        assert provider.freeze_vision_model is True
        assert provider.freeze_vision_projection is True
