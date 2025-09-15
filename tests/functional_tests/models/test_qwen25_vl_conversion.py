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


# HuggingFace Qwen2.5-VL model IDs that should be testable
HF_QWEN25_VL_MODEL_IDS = [
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2.5-VL-7B-Instruct",
    # Note: Adding more model IDs as they become available
    # "Qwen/Qwen2.5-VL-1.5B-Instruct",
    # "Qwen/Qwen2.5-VL-14B-Instruct",
]


class TestQwen25VLConversion:
    """Test conversion from HuggingFace Qwen2.5-VL models to Megatron."""

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_bridge_from_hf_pretrained(self, hf_model_id):
        """Test that AutoBridge can load Qwen2.5-VL models from HF."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        
        assert bridge is not None
        assert hasattr(bridge, "to_megatron_provider")

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_to_megatron_provider(self, hf_model_id):
        """Test conversion to Megatron provider."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        assert isinstance(provider, Qwen25VLModelProvider)

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provider_basic_config(self, hf_model_id):
        """Test that converted provider has correct basic configuration."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Check basic transformer config is present
        assert hasattr(provider, "num_layers")
        assert hasattr(provider, "hidden_size")
        assert hasattr(provider, "num_attention_heads")
        assert hasattr(provider, "vocab_size")
        
        # Check values are reasonable
        assert provider.num_layers > 0
        assert provider.hidden_size > 0
        assert provider.num_attention_heads > 0
        assert provider.vocab_size > 0

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provider_vl_specific_config(self, hf_model_id):
        """Test that converted provider has VL-specific configuration."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Check VL-specific attributes
        assert hasattr(provider, "vision_config")
        assert hasattr(provider, "bos_token_id")
        assert hasattr(provider, "eos_token_id")
        assert hasattr(provider, "vision_start_token_id")
        assert hasattr(provider, "vision_end_token_id")
        assert hasattr(provider, "vision_token_id")
        assert hasattr(provider, "image_token_id")
        assert hasattr(provider, "video_token_id")
        
        # Check position embedding type
        assert provider.position_embedding_type == "mrope"
        assert hasattr(provider, "mrope_section")
        assert isinstance(provider.mrope_section, list)

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provider_token_ids(self, hf_model_id):
        """Test that converted provider has correct token IDs."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Check token IDs are reasonable values
        assert isinstance(provider.bos_token_id, int)
        assert isinstance(provider.eos_token_id, int)
        assert isinstance(provider.vision_start_token_id, int)
        assert isinstance(provider.vision_end_token_id, int)
        assert isinstance(provider.vision_token_id, int)
        assert isinstance(provider.image_token_id, int)
        assert isinstance(provider.video_token_id, int)
        
        # Token IDs should be within vocabulary range
        assert 0 <= provider.bos_token_id < provider.vocab_size
        assert 0 <= provider.eos_token_id < provider.vocab_size

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provider_qwen2_inheritance(self, hf_model_id):
        """Test that converted provider inherits Qwen2 characteristics."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Check Qwen2-specific configuration
        assert provider.normalization == "RMSNorm"
        assert provider.gated_linear_unit is True
        assert provider.add_bias_linear is False
        assert provider.add_qkv_bias is True
        assert provider.layernorm_epsilon == 1e-6

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provider_vision_config(self, hf_model_id):
        """Test that converted provider has valid vision configuration."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Check vision config exists and has required attributes
        assert provider.vision_config is not None
        assert hasattr(provider.vision_config, "hidden_size")
        assert hasattr(provider.vision_config, "intermediate_size")
        assert hasattr(provider.vision_config, "num_hidden_layers")
        assert hasattr(provider.vision_config, "num_attention_heads")

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provider_freeze_options(self, hf_model_id):
        """Test that converted provider has freeze options."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Check freeze options exist and default to False
        assert hasattr(provider, "freeze_language_model")
        assert hasattr(provider, "freeze_vision_model")
        assert hasattr(provider, "freeze_vision_projection")
        
        # Default should be False
        assert provider.freeze_language_model is False
        assert provider.freeze_vision_model is False
        assert provider.freeze_vision_projection is False

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_provide_method(self, hf_model_id):
        """Test that provider can create models."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Should have provide method
        assert hasattr(provider, "provide")
        assert callable(provider.provide)
        
        # Should have provide_language_model method
        assert hasattr(provider, "provide_language_model")
        assert callable(provider.provide_language_model)


class TestQwen25VLModelSizes:
    """Test different Qwen2.5-VL model sizes have expected configurations."""

    def test_qwen25_vl_3b_configuration(self):
        """Test Qwen2.5-VL 3B model specific configuration."""
        if "Qwen/Qwen2.5-VL-3B-Instruct" not in HF_QWEN25_VL_MODEL_IDS:
            pytest.skip("Qwen2.5-VL 3B model not available for testing")
            
        bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # 3B model should have specific architecture
        # Note: These values might need adjustment based on actual model config
        assert provider.hidden_size > 1000  # Should be reasonable size
        assert provider.num_layers > 20     # Should have multiple layers

    def test_qwen25_vl_7b_configuration(self):
        """Test Qwen2.5-VL 7B model specific configuration."""
        if "Qwen/Qwen2.5-VL-7B-Instruct" not in HF_QWEN25_VL_MODEL_IDS:
            pytest.skip("Qwen2.5-VL 7B model not available for testing")
            
        bridge = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # 7B model should be larger than 3B
        assert provider.hidden_size > 2000  # Should be larger
        assert provider.num_layers > 25     # Should have more layers


class TestQwen25VLSpecialCases:
    """Test special cases and edge conditions for Qwen2.5-VL conversion."""

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_scatter_embedding_sequence_parallel(self, hf_model_id):
        """Test that scatter_embedding_sequence_parallel is False for VL models."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # VL models should not scatter embeddings across sequence parallel regions
        # because vision embeddings are inserted into language embeddings
        assert provider.scatter_embedding_sequence_parallel is False

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_position_embedding_mrope(self, hf_model_id):
        """Test that position embedding type is mrope for VL models."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # VL models should use mrope position embedding
        assert provider.position_embedding_type == "mrope"
        assert hasattr(provider, "mrope_section")
        assert len(provider.mrope_section) == 3

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_generation_config(self, hf_model_id):
        """Test that generation config is preserved."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Should have generation config
        assert hasattr(provider, "generation_config")
        # Generation config might be None, which is acceptable

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_qwen25_vl_conversion_reproducible(self, hf_model_id):
        """Test that conversion is reproducible."""
        # Convert twice and check consistency
        bridge1 = AutoBridge.from_hf_pretrained(hf_model_id)
        provider1 = bridge1.to_megatron_provider(load_weights=False)
        
        bridge2 = AutoBridge.from_hf_pretrained(hf_model_id)
        provider2 = bridge2.to_megatron_provider(load_weights=False)
        
        # Basic config should be identical
        assert provider1.num_layers == provider2.num_layers
        assert provider1.hidden_size == provider2.hidden_size
        assert provider1.num_attention_heads == provider2.num_attention_heads
        assert provider1.vocab_size == provider2.vocab_size
        
        # VL-specific config should be identical
        assert provider1.bos_token_id == provider2.bos_token_id
        assert provider1.eos_token_id == provider2.eos_token_id
        assert provider1.vision_start_token_id == provider2.vision_start_token_id


class TestQwen25VLErrorHandling:
    """Test error handling for Qwen2.5-VL conversion."""

    def test_invalid_model_id(self):
        """Test that invalid model ID raises appropriate error."""
        with pytest.raises((ValueError, OSError, Exception)):
            AutoBridge.from_hf_pretrained("invalid/qwen25-vl-model")

    @pytest.mark.parametrize("hf_model_id", HF_QWEN25_VL_MODEL_IDS)
    def test_conversion_without_weights_loading(self, hf_model_id):
        """Test that conversion works without loading weights."""
        bridge = AutoBridge.from_hf_pretrained(hf_model_id)
        provider = bridge.to_megatron_provider(load_weights=False)
        
        # Should work without errors
        assert isinstance(provider, Qwen25VLModelProvider)
