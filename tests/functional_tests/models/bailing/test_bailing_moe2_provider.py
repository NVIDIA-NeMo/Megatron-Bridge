# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from megatron.bridge.models.bailing import (
    Ling1TModelProvider,
    LingFlash2ModelProvider,
    LingMini2ModelProvider,
)
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from tests.functional_tests.utils import compare_provider_configs


HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    "inclusionAI/Ling-mini-2.0": LingMini2ModelProvider,
    "inclusionAI/Ling-flash-2.0": LingFlash2ModelProvider,
    "inclusionAI/Ling-1T": Ling1TModelProvider,
}


class TestBailingMoeV2ModelProviderMapping:
    """Test that bridge provider configs are equivalent to predefined provider configs."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_equivalence(self, hf_model_id, provider_class):
        """Test that bridge converted provider config matches predefined provider config."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Create predefined provider
        predefined_provider = provider_class()

        # Compare configs
        # Skip fields that may differ due to HF config differences or dynamic computation
        skip_fields = {
            "transformer_layer_spec",  # May differ in function identity
            "generation_config",  # Not in predefined providers
            "hf_model_id",  # Not in predefined providers
            "mtp_num_layers",  # May differ if HF model doesn't have num_nextn_predict_layers set
        }
        compare_provider_configs(converted_provider, predefined_provider, hf_model_id, skip_fields=skip_fields)
