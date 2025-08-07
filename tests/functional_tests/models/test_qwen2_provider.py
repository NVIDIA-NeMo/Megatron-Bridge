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
from megatron.bridge.models.qwen import (
    Qwen2ModelProvider1P5B,
    Qwen2ModelProvider7B,
    Qwen2ModelProvider72B,
    Qwen2ModelProvider500M,
    Qwen25ModelProvider1P5B,
    Qwen25ModelProvider3B,
    Qwen25ModelProvider7B,
    Qwen25ModelProvider14B,
    Qwen25ModelProvider32B,
    Qwen25ModelProvider72B,
    Qwen25ModelProvider500M,
)


HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    # Qwen2.5 models
    "Qwen/Qwen2.5-0.5B-Instruct": Qwen25ModelProvider500M,
    "Qwen/Qwen2.5-1.5B-Instruct": Qwen25ModelProvider1P5B,
    "Qwen/Qwen2.5-3B-Instruct": Qwen25ModelProvider3B,
    "Qwen/Qwen2.5-7B-Instruct": Qwen25ModelProvider7B,
    "Qwen/Qwen2.5-14B-Instruct": Qwen25ModelProvider14B,
    "Qwen/Qwen2.5-32B-Instruct": Qwen25ModelProvider32B,
    "Qwen/Qwen2.5-72B-Instruct": Qwen25ModelProvider72B,
    # # Qwen2 models
    "Qwen/Qwen2-0.5B-Instruct": Qwen2ModelProvider500M,
    "Qwen/Qwen2-1.5B-Instruct": Qwen2ModelProvider1P5B,
    "Qwen/Qwen2-7B-Instruct": Qwen2ModelProvider7B,
    "Qwen/Qwen2-72B-Instruct": Qwen2ModelProvider72B,
}


def compare_provider_configs(converted_provider, predefined_provider, model_id):
    """Compare ALL configuration attributes between converted and predefined providers."""

    # Get all attributes from both providers
    converted_attrs = vars(converted_provider)
    predefined_attrs = vars(predefined_provider)

    # First check that both providers have the same set of attributes
    converted_keys = set(converted_attrs.keys())
    predefined_keys = set(predefined_attrs.keys())

    missing_in_converted = predefined_keys - converted_keys
    missing_in_predefined = converted_keys - predefined_keys

    if missing_in_converted:
        raise AssertionError(f"Converted provider for {model_id} is missing attributes: {missing_in_converted}")

    if missing_in_predefined:
        raise AssertionError(f"Predefined provider for {model_id} is missing attributes: {missing_in_predefined}")

    # Compare all attribute values
    mismatched_attrs = []
    excluded_attrs = set()

    for attr_name in sorted(converted_keys):
        # Skip excluded attributes
        if "init_method" in attr_name or attr_name == "generation_config":
            excluded_attrs.add(attr_name)
            continue

        converted_value = converted_attrs[attr_name]
        predefined_value = predefined_attrs[attr_name]

        # Handle special comparison cases for different types
        if converted_value != predefined_value:
            # For functions, compare by name/identity since they might be the same function
            # but not pass == comparison
            if callable(converted_value) and callable(predefined_value):
                if (
                    hasattr(converted_value, "__name__")
                    and hasattr(predefined_value, "__name__")
                    and converted_value.__name__ == predefined_value.__name__
                ):
                    continue
                elif converted_value is predefined_value:
                    continue

            mismatched_attrs.append(f"  {attr_name}: converted={converted_value} vs predefined={predefined_value}")

    if mismatched_attrs:
        raise AssertionError(f"Configuration mismatch for {model_id}:\n" + "\n".join(mismatched_attrs))


class TestQwenModelProviderMapping:
    """Test that bridge provider configs are equivalent to predefined provider configs."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_equivalence(self, hf_model_id, provider_class):
        """Test that bridge converted provider config matches predefined provider config."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Create predefined provider
        predefined_provider = provider_class()

        # Compare configs
        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)
