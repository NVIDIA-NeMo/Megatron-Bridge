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
from megatron.bridge.models.qwen_vl import Qwen25VLModelProvider
from tests.functional_tests.utils import compare_provider_configs


# Mapping of HuggingFace model IDs to predefined providers
# Note: Since we only have one generic Qwen25VLModelProvider class,
# we'll test against it for all model sizes
HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER = {
    "Qwen/Qwen2.5-VL-3B-Instruct": Qwen25VLModelProvider,
    "Qwen/Qwen2.5-VL-7B-Instruct": Qwen25VLModelProvider,
    # Add more model IDs as they become available
    # "Qwen/Qwen2.5-VL-1.5B-Instruct": Qwen25VLModelProvider,
    # "Qwen/Qwen2.5-VL-14B-Instruct": Qwen25VLModelProvider,
}


class TestQwen25VLModelProviderMapping:
    """Test that bridge provider configs are equivalent to predefined provider configs."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_bridge_vs_predefined_provider_config_equivalence(self, hf_model_id, provider_class):
        """Test that bridge converted provider config matches predefined provider config."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # For VL models, we create a predefined provider with the same configuration
        # as the converted one since we don't have size-specific predefined providers
        predefined_provider = provider_class(
            num_layers=converted_provider.num_layers,
            hidden_size=converted_provider.hidden_size,
            ffn_hidden_size=converted_provider.ffn_hidden_size,
            num_attention_heads=converted_provider.num_attention_heads,
            num_query_groups=converted_provider.num_query_groups,
            vocab_size=converted_provider.vocab_size,
            seq_length=converted_provider.seq_length,
            vision_config=converted_provider.vision_config,
        )

        # Compare configs
        compare_provider_configs(converted_provider, predefined_provider, hf_model_id)

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_vl_specific_config_consistency(self, hf_model_id, provider_class):
        """Test VL-specific configuration consistency between bridge and predefined."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Create predefined provider with same core config
        predefined_provider = provider_class(
            num_layers=converted_provider.num_layers,
            hidden_size=converted_provider.hidden_size,
            num_attention_heads=converted_provider.num_attention_heads,
        )

        # Compare VL-specific attributes
        assert converted_provider.position_embedding_type == predefined_provider.position_embedding_type
        assert converted_provider.mrope_section == predefined_provider.mrope_section
        assert converted_provider.scatter_embedding_sequence_parallel == predefined_provider.scatter_embedding_sequence_parallel

        # Compare token IDs (predefined should have defaults, converted should have HF values)
        # For token IDs, we expect the converted provider to have values from HF model
        # while predefined provider has defaults
        assert isinstance(converted_provider.bos_token_id, int)
        assert isinstance(converted_provider.eos_token_id, int)
        assert isinstance(converted_provider.vision_start_token_id, int)
        assert isinstance(converted_provider.vision_end_token_id, int)

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_freeze_options_consistency(self, hf_model_id, provider_class):
        """Test freeze options consistency."""
        # Create bridge from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Create predefined provider
        predefined_provider = provider_class(
            num_layers=converted_provider.num_layers,
            hidden_size=converted_provider.hidden_size,
            num_attention_heads=converted_provider.num_attention_heads,
        )

        # Freeze options should default to False for both
        assert converted_provider.freeze_language_model == predefined_provider.freeze_language_model
        assert converted_provider.freeze_vision_model == predefined_provider.freeze_vision_model
        assert converted_provider.freeze_vision_projection == predefined_provider.freeze_vision_projection


class TestQwen25VLProviderInheritance:
    """Test inheritance and method consistency for Qwen25VL providers."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_provider_methods_exist(self, hf_model_id, provider_class):
        """Test that all required methods exist."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Check that provide methods exist
        assert hasattr(converted_provider, "provide")
        assert callable(converted_provider.provide)
        
        assert hasattr(converted_provider, "provide_language_model")
        assert callable(converted_provider.provide_language_model)

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_qwen2_inheritance_preserved(self, hf_model_id, provider_class):
        """Test that Qwen2 model characteristics are preserved."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Should inherit Qwen2 characteristics
        assert converted_provider.normalization == "RMSNorm"
        assert converted_provider.gated_linear_unit is True
        assert converted_provider.add_bias_linear is False
        assert converted_provider.add_qkv_bias is True


class TestQwen25VLProviderSpecificSizes:
    """Test specific model size configurations."""

    def test_qwen25_vl_3b_provider_configuration(self):
        """Test 3B model specific configuration."""
        if "Qwen/Qwen2.5-VL-3B-Instruct" not in HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER:
            pytest.skip("Qwen2.5-VL 3B model not available for testing")

        bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        provider = bridge.to_megatron_provider(load_weights=False)

        # Check that this is indeed a 3B-class model
        # Note: These are approximate checks since exact values depend on actual model config
        assert provider.hidden_size >= 2000  # Should be reasonable for 3B model
        assert provider.num_layers >= 20      # Should have sufficient layers

    def test_qwen25_vl_7b_provider_configuration(self):
        """Test 7B model specific configuration."""
        if "Qwen/Qwen2.5-VL-7B-Instruct" not in HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER:
            pytest.skip("Qwen2.5-VL 7B model not available for testing")

        bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        provider = bridge.to_megatron_provider(load_weights=False)

        # Check that this is indeed a 7B-class model
        assert provider.hidden_size >= 3000  # Should be larger for 7B model
        assert provider.num_layers >= 25      # Should have more layers than 3B


class TestQwen25VLProviderVisionConfiguration:
    """Test vision-specific configuration for Qwen25VL providers."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_vision_config_preservation(self, hf_model_id, provider_class):
        """Test that vision configuration is properly preserved from HF model."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # Vision config should be present and valid
        assert converted_provider.vision_config is not None
        assert hasattr(converted_provider.vision_config, "hidden_size")
        assert hasattr(converted_provider.vision_config, "intermediate_size")
        assert hasattr(converted_provider.vision_config, "num_hidden_layers")
        assert hasattr(converted_provider.vision_config, "num_attention_heads")

        # Values should be reasonable
        assert converted_provider.vision_config.hidden_size > 0
        assert converted_provider.vision_config.intermediate_size > 0
        assert converted_provider.vision_config.num_hidden_layers > 0
        assert converted_provider.vision_config.num_attention_heads > 0

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_mrope_configuration(self, hf_model_id, provider_class):
        """Test mRoPE configuration for VL models."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        converted_provider = bridge.to_megatron_provider(load_weights=False)

        # mRoPE should be configured for VL models
        assert converted_provider.position_embedding_type == "mrope"
        assert hasattr(converted_provider, "mrope_section")
        assert isinstance(converted_provider.mrope_section, list)
        assert len(converted_provider.mrope_section) == 3


class TestQwen25VLProviderEdgeCases:
    """Test edge cases for Qwen25VL provider functionality."""

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_provider_reproducibility(self, hf_model_id, provider_class):
        """Test that provider creation is reproducible."""
        # Create provider twice
        bridge1 = AutoBridge.from_hf_pretrained(hf_model_id)
        provider1 = bridge1.to_megatron_provider(load_weights=False)

        bridge2 = AutoBridge.from_hf_pretrained(hf_model_id)
        provider2 = bridge2.to_megatron_provider(load_weights=False)

        # Core configurations should be identical
        assert provider1.num_layers == provider2.num_layers
        assert provider1.hidden_size == provider2.hidden_size
        assert provider1.num_attention_heads == provider2.num_attention_heads
        assert provider1.vocab_size == provider2.vocab_size

        # VL-specific configurations should be identical
        assert provider1.bos_token_id == provider2.bos_token_id
        assert provider1.eos_token_id == provider2.eos_token_id
        assert provider1.position_embedding_type == provider2.position_embedding_type

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_provider_without_weight_loading(self, hf_model_id, provider_class):
        """Test that provider works correctly without loading weights."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)

        # Should work without any issues
        assert isinstance(provider, Qwen25VLModelProvider)
        assert provider.num_layers > 0
        assert provider.hidden_size > 0

    @pytest.mark.parametrize("hf_model_id,provider_class", list(HF_MODEL_ID_TO_BRIDGE_MODEL_PROVIDER.items()))
    def test_provider_custom_overrides(self, hf_model_id, provider_class):
        """Test that provider can be created with custom overrides."""
        # Get base configuration from HF model
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        base_provider = bridge.to_megatron_provider(load_weights=False)

        # Create provider with custom overrides
        custom_provider = provider_class(
            num_layers=base_provider.num_layers,
            hidden_size=base_provider.hidden_size,
            num_attention_heads=base_provider.num_attention_heads,
            seq_length=8192,  # Custom sequence length
            freeze_vision_model=True,  # Custom freeze option
        )

        # Custom values should be applied
        assert custom_provider.seq_length == 8192
        assert custom_provider.freeze_vision_model is True

        # Other values should remain as defaults
        assert custom_provider.num_layers == base_provider.num_layers
        assert custom_provider.hidden_size == base_provider.hidden_size
