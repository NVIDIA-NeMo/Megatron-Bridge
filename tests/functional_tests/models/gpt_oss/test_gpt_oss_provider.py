#!/usr/bin/env python3
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
from megatron.bridge.models.gpt_oss import (
    GPTOSSProvider20B,
    GPTOSSProvider120B,
)
from tests.functional_tests.utils import compare_provider_configs


# These HF IDs are placeholders for tests in environments with pre-downloaded models.
# For CI or local runs without actual HF downloads, we allow mapping using a config-only load.
HF_MODEL_ID_TO_PROVIDER = {
    # If your environment has these, point to actual repo ids or local cache paths
    "openai/gpt-oss-20b": GPTOSSProvider20B,
    "openai/gpt-oss-120b": GPTOSSProvider120B,
}


class TestGptOssProviderMapping:
    """Test that bridge provider configs match predefined GPT-OSS providers."""

    @pytest.mark.parametrize("provider_class", [GPTOSSProvider20B, GPTOSSProvider120B])
    def test_bridge_vs_predefined_provider_config_from_config_only(self, provider_class):
        # Skip if transformers lacks GPT-OSS
        transformers = pytest.importorskip("transformers")
        GptOssConfig = getattr(transformers, "GptOssConfig", None)
        if GptOssConfig is None:
            pytest.skip("transformers installation does not include GptOssConfig")

        # Create a minimal config aligned with GPT-OSS; values don't need to match a real HF repo
        # because we compare converted vs predefined providers for equality, not against a specific model ID.
        cfg = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_size=provider_class.hidden_size if hasattr(provider_class, "hidden_size") else 2880,
            num_hidden_layers=getattr(provider_class, "num_layers", 24),
            num_attention_heads=getattr(provider_class, "num_attention_heads", 64),
            num_key_value_heads=getattr(provider_class, "num_key_value_heads", 8),
            num_local_experts=getattr(provider_class, "num_moe_experts", 32),
            vocab_size=201088,
        )

        bridge = AutoBridge.from_hf_config(cfg)
        converted_provider = bridge.to_megatron_provider(load_weights=False)
        converted_provider.finalize()

        predefined_provider = provider_class()
        predefined_provider.finalize()

        compare_provider_configs(converted_provider, predefined_provider, "gpt-oss-config-only")

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_hf(self, hf_model_id, provider_class):
        # Optional mapping test that uses from_hf_pretrained if available in the environment
        if not HF_MODEL_ID_TO_PROVIDER:
            pytest.skip("No HF model ids configured for GPT-OSS mapping test")

        bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
        converted_provider = bridge.to_megatron_provider(load_weights=False)
        converted_provider.finalize()

        predefined_provider = provider_class()
        predefined_provider.finalize()

        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)
