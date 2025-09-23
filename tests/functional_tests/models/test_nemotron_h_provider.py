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
from megatron.bridge.models.mamba import (
    NemotronHModel8BProvider,
    NemotronHModel47BProvider,
    NemotronHModel56BProvider,
    NemotronNano9Bv2Provider,
    NemotronNano12Bv2Provider,
)
from tests.functional_tests.utils import compare_provider_configs


HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    # NemotronH models
    "nvidia/Nemotron-H-8B-Base-8K": NemotronHModel8BProvider,
    "nvidia/Nemotron-H-47B-Base-8K": NemotronHModel47BProvider,
    "nvidia/Nemotron-H-56B-Base-8K": NemotronHModel56BProvider,
    "nvidia/NVIDIA-Nemotron-Nano-9B-v2": NemotronNano9Bv2Provider,
    "nvidia/NVIDIA-Nemotron-Nano-12B-v2": NemotronNano12Bv2Provider,
}


class TestNemotronHModelProviderMapping:
    """Test that bridge provider configs are equivalent to predefined provider configs."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_equivalence(self, hf_model_id, provider_class):
        """Test that bridge converted provider config matches predefined provider config."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
        converted_provider = bridge.to_megatron_provider(load_weights=False)
        converted_provider.finalize()

        # Create predefined provider
        predefined_provider = provider_class()
        predefined_provider.finalize()

        # Compare configs
        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)
