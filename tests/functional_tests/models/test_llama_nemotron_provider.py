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
from megatron.bridge.models.llama_nemotron import (
    Llama31Nemotron70BProvider,
    Llama31NemotronUltra253BProvider,
    Llama33NemotronSuper49BProvider,
)
from tests.functional_tests.utils import compare_provider_configs


HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    # Available HuggingFace Llama-Nemotron models with standard LlamaForCausalLM architecture
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": Llama31Nemotron70BProvider,
    # Note: Nano model may not be publicly available
    # "nvidia/Llama-3.1-Nemotron-Nano-8B-v1": Llama31NemotronNano8BProvider,
}

# Heterogeneous models use custom DeciLMForCausalLM architecture - tested separately
HETEROGENEOUS_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    "nvidia/Llama-3_3-Nemotron-Super-49B-v1": Llama33NemotronSuper49BProvider,
    "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1": Llama31NemotronUltra253BProvider,
}


class TestLlamaNemotronModelProviderMapping:
    """Test that bridge provider configs are equivalent to predefined provider configs."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_equivalence(self, hf_model_id, provider_class):
        """Test that bridge converted provider config matches predefined provider config."""
        # Create bridge from HF model (with trust_remote_code for heterogeneous models)
        bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
        converted_provider = bridge.to_megatron_provider(load_weights=False)
        converted_provider.finalize()

        # Create predefined provider
        predefined_provider = provider_class()
        predefined_provider.finalize()

        # Compare configs
        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)

    def test_llama_nemotron_has_correct_architecture(self):
        """Test that Llama-Nemotron models are detected as Llama architecture."""
        for model_id in HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.keys():
            bridge = AutoBridge.from_hf_pretrained(model_id, trust_remote_code=True)

            # Should be able to create provider without errors
            provider = bridge.to_megatron_provider(load_weights=False)

            # Should have Llama-specific settings
            assert provider.gated_linear_unit is True  # SwiGLU
            assert provider.position_embedding_type == "rope"
            assert provider.kv_channels == 128  # Nemotron-specific

            # Should have rope scaling for Llama 3.1
            assert hasattr(provider, "scale_factor")
            assert provider.scale_factor == 8.0

    def test_heterogeneous_models_have_special_fields(self):
        """Test that heterogeneous models have the required heterogeneous fields."""
        heterogeneous_models = [
            "nvidia/Llama-3_3-Nemotron-Super-49B-v1",
            "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
        ]

        for model_id in heterogeneous_models:
            if model_id in HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER:
                provider_class = HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER[model_id]
                provider = provider_class()

                # Check heterogeneous-specific fields
                assert hasattr(provider, "heterogeneous_layers_config_encoded_json")
                assert hasattr(provider, "transformer_layer_spec")
                assert provider.heterogeneous_layers_config_encoded_json != ""

                # Should be able to finalize without errors
                provider.finalize()

                print(f"✓ {model_id}: Heterogeneous config validated")

    def test_super_vs_ultra_differences(self):
        """Test that Super 49B and Ultra 253B have different configurations."""
        super_provider = Llama33NemotronSuper49BProvider()
        ultra_provider = Llama31NemotronUltra253BProvider()

        # Should have different layer counts and sizes
        assert super_provider.num_layers != ultra_provider.num_layers
        assert super_provider.hidden_size != ultra_provider.hidden_size
        assert super_provider.num_attention_heads != ultra_provider.num_attention_heads

        # But both should have Nemotron-specific kv_channels
        assert super_provider.kv_channels == 128
        assert ultra_provider.kv_channels == 128

        # Both should have heterogeneous configs
        assert super_provider.heterogeneous_layers_config_encoded_json != ""
        assert ultra_provider.heterogeneous_layers_config_encoded_json != ""
        assert (
            super_provider.heterogeneous_layers_config_encoded_json
            != ultra_provider.heterogeneous_layers_config_encoded_json
        )

    @pytest.mark.parametrize(
        "hf_model_id,provider_class", list(HETEROGENEOUS_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items())
    )
    def test_heterogeneous_models_bridge_conversion(self, hf_model_id, provider_class):
        """Test bridge conversion for heterogeneous models with custom DeciLMForCausalLM architecture."""
        try:
            # These models use custom DeciLMForCausalLM architecture and require trust_remote_code
            bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
            converted_provider = bridge.to_megatron_provider(load_weights=False)
            converted_provider.finalize()

            # Create predefined provider
            predefined_provider = provider_class()
            predefined_provider.finalize()

            # Compare configs - heterogeneous models may have some differences due to block configs
            # so we'll do basic validation rather than full config comparison
            assert type(converted_provider).__name__ == type(predefined_provider).__name__
            assert converted_provider.num_layers == predefined_provider.num_layers
            assert converted_provider.hidden_size == predefined_provider.hidden_size
            assert converted_provider.num_attention_heads == predefined_provider.num_attention_heads
            assert converted_provider.kv_channels == predefined_provider.kv_channels

            # Both should have heterogeneous configs
            assert hasattr(converted_provider, "heterogeneous_layers_config_encoded_json")
            assert hasattr(predefined_provider, "heterogeneous_layers_config_encoded_json")

            print(f"✓ {hf_model_id}: Heterogeneous bridge conversion successful")

        except Exception as e:
            # Should not fail now that AutoBridge handles custom architectures
            raise AssertionError(f"Unexpected failure for {hf_model_id}: {e}") from e

    def test_heterogeneous_providers_can_finalize(self):
        """Test that heterogeneous providers can finalize without HuggingFace models."""
        # Test the provider classes directly without requiring HF model loading
        for model_id, provider_class in HETEROGENEOUS_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items():
            provider = provider_class()

            # Should be able to finalize without errors
            provider.finalize()

            # Should have heterogeneous-specific fields
            assert hasattr(provider, "heterogeneous_layers_config_encoded_json")
            assert hasattr(provider, "transformer_layer_spec")
            assert provider.heterogeneous_layers_config_encoded_json != ""
            assert provider.kv_channels == 128

            print(f"✓ {provider_class.__name__}: Provider finalize successful")
