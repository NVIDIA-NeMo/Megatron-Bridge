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

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.mistral import (
    MistralModelProvider,
)


HF_MODEL_ID_TO_EXPECTED_PROVIDER_FIELDS = {
    # Mistral models
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "num_layers": 32,
        "hidden_size": 4096,
        "ffn_hidden_size": 14336,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "vocab_size": 32768,
    },
    # Mistral Small3 24B models
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "num_layers": 40,
        "hidden_size": 5120,
        "ffn_hidden_size": 32768,
        "num_attention_heads": 32,
        "num_query_groups": 8,
        "vocab_size": 131072,
    },
}


class TestMistralModelProviderMapping:
    """Test that bridge provider configs are derived from HF config."""

    @pytest.mark.parametrize("hf_model_id,expected_fields", list(HF_MODEL_ID_TO_EXPECTED_PROVIDER_FIELDS.items()))
    def test_bridge_provider_config_matches_expected_hf_fields(self, hf_model_id, expected_fields):
        """Test that bridge converted provider config matches expected HF-derived fields."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)
        converted_provider.finalize()

        assert isinstance(converted_provider, MistralModelProvider)
        for name, expected_value in expected_fields.items():
            assert getattr(converted_provider, name) == expected_value
