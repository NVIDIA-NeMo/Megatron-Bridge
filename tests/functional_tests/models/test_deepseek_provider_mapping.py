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
from megatron.bridge.models.deepseek.deepseek_provider import (
    DeepSeekV2LiteProvider,
    DeepSeekV2Provider,
    DeepSeekV3Provider,
    MoonlightProvider,
)
from tests.functional_tests.utils import compare_provider_configs


HF_MODEL_ID_TO_PROVIDER = {
    "deepseek-ai/DeepSeek-V2": DeepSeekV2Provider,
    "deepseek-ai/DeepSeek-V2-Chat": DeepSeekV2Provider,
    "deepseek-ai/DeepSeek-V2-Lite": DeepSeekV2LiteProvider,
    "deepseek-ai/DeepSeek-V3": DeepSeekV3Provider,
    "deepseek-ai/DeepSeek-V3-Base": DeepSeekV3Provider,
    "moonshotai/Moonlight-16B-A3B": MoonlightProvider,
}


class TestDeepSeekProviderMapping:
    """Test that bridge provider configs match predefined DeepSeek providers."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config(self, hf_model_id, provider_class):
        bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Finalize the converted provider to ensure computed fields are set
        converted_provider.finalize()

        predefined_provider = provider_class()
        # Also finalize the predefined provider for fair comparison
        predefined_provider.finalize()

        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)
