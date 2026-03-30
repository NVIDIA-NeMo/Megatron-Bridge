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

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig

from megatron.bridge.models import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.nemotronh.nemotron_h_bridge import (
    NemotronHBridge,
    _MTPFlatteningMapping,
    _MTPFlatteningQKVMapping,
    _replace_wildcards,
)


class TestNemotronHBridge:
    """Test cases for NemotronHBridge class."""

    @pytest.fixture
    def nemotronh_8b_config_dict(self):
        """Create a sample NemotronH configuration."""
        return {
            "architectures": ["NemotronHForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_head_dim": 128,
            "auto_map": {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            },
            "bos_token_id": 1,
            "chunk_size": 128,
            "conv_kernel": 4,
            "eos_token_id": 2,
            "expand": 2,
            "hidden_act": "relu2",  # Required for base class activation mapping
            "hidden_dropout": 0.0,
            "hidden_size": 4096,
            "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            "initializer_range": 0.02,
            "intermediate_size": 21504,
            "layer_norm_epsilon": 1e-05,
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 128,
            "mamba_proj_bias": False,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "model_type": "nemotron_h",
            "n_groups": 8,
            "num_attention_heads": 32,
            "num_hidden_layers": 52,
            "num_key_value_heads": 8,
            "num_logits_to_keep": 1,
            "pad_token_id": 0,
            "rescale_prenorm_residual": True,
            "residual_in_fp32": False,
            "rms_norm_eps": 1e-05,
            "ssm_state_size": 128,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.48.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "use_mamba_kernels": True,
            "vocab_size": 131072,
            # Explicitly set to None to disable MoE; Mock objects return Mock for any attr access,
            # so hasattr() always returns True.
            "n_routed_experts": None,
        }

    @pytest.fixture
    def mock_nemotronh_config(self, nemotronh_8b_config_dict):
        """Create mock config instance.

        Uses spec=[] to make getattr return None for undefined attributes
        instead of Mock objects, which would incorrectly be passed to the provider.
        """
        cfg = Mock(spec=[])
        for k, v in nemotronh_8b_config_dict.items():
            setattr(cfg, k, v)
        return cfg

    @pytest.fixture
    def mock_pretrained_nemotronh(self, mock_nemotronh_config):
        """Create a mock PreTrainedCausalLM with NemotronH model."""

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_nemotronh_config
        return mock_pretrained

    def test_bridge_registration(self):
        """Test that NemotronHBridge is properly registered."""
        assert issubclass(NemotronHBridge, MegatronModelBridge)

    def test_provider_bridge_basic(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test basic provider_bridge functionality."""
        bridge = NemotronHBridge()

        # Call provider_bridge
        result = bridge.provider_bridge(mock_pretrained_nemotronh)
        result.finalize()

        # Check that it returns a MambaModelProvider instance
        assert isinstance(result, MambaModelProvider)

        # Check basic configuration mapping
        assert result.num_layers == mock_nemotronh_config.num_hidden_layers
        assert result.hybrid_layer_pattern == mock_nemotronh_config.hybrid_override_pattern
        assert result.hidden_size == mock_nemotronh_config.hidden_size
        assert result.add_bias_linear == mock_nemotronh_config.use_bias
        assert result.num_attention_heads == mock_nemotronh_config.num_attention_heads
        assert result.seq_length == mock_nemotronh_config.max_position_embeddings

    def test_provider_bridge_vocabulary(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test vocabulary size mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check vocabulary configuration
        assert result.vocab_size == mock_nemotronh_config.vocab_size
        assert result.share_embeddings_and_output_weights == mock_nemotronh_config.tie_word_embeddings

    def test_provider_bridge_attention_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test attention configuration mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check attention configuration
        assert result.num_attention_heads == mock_nemotronh_config.num_attention_heads
        assert result.num_query_groups == mock_nemotronh_config.num_key_value_heads

    def test_provider_bridge_mamba_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test Mamba-specific configuration mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)
        result.finalize()

        # Check Mamba-specific configuration
        assert result.mamba_state_dim == mock_nemotronh_config.ssm_state_size
        assert result.mamba_head_dim == mock_nemotronh_config.mamba_head_dim
        assert result.mamba_num_heads == mock_nemotronh_config.mamba_num_heads
        assert result.mamba_num_groups == mock_nemotronh_config.n_groups
        assert result.hybrid_layer_pattern == mock_nemotronh_config.hybrid_override_pattern

    def test_provider_bridge_mlp_config(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test MLP configuration mapping."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check MLP configuration
        assert result.ffn_hidden_size == mock_nemotronh_config.intermediate_size
        assert result.gated_linear_unit == False  # Mamba doesn't use gated linear units

    def test_provider_bridge_normalization(self, mock_pretrained_nemotronh, mock_nemotronh_config):
        """Test normalization configuration."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # Check normalization settings
        assert result.layernorm_epsilon == mock_nemotronh_config.layer_norm_epsilon

    def test_provider_bridge_dtype_handling(self, mock_nemotronh_config):
        """Test dtype handling in provider_bridge."""
        # Create model with specific dtype
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_nemotronh_config.torch_dtype = "bfloat16"
        mock_pretrained.config = mock_nemotronh_config

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # The provider should respect the model's dtype
        assert result.params_dtype == torch.bfloat16
        assert result.bf16 == True
        assert result.fp16 == False

    def test_mapping_registry_implementation(self, mock_pretrained_nemotronh):
        """Test that mapping_registry returns a proper MegatronMappingRegistry."""
        bridge = NemotronHBridge()

        # Get the mapping registry
        mapping_registry = bridge.mapping_registry()

        # Check it's not None
        assert mapping_registry is not None
        assert any([isinstance(m, AutoMapping) for m in mapping_registry.mappings])
        # assert any([isinstance(m, PrunedVocabMapping) for m in mapping_registry.mappings])
        assert any([isinstance(m, QKVMapping) for m in mapping_registry.mappings])

    def test_provider_bridge_fixed_settings(self, mock_pretrained_nemotronh):
        """Test fixed settings that should always be set regardless of config."""
        bridge = NemotronHBridge()

        result = bridge.provider_bridge(mock_pretrained_nemotronh)

        # These should always be set to these values for Mamba
        assert result.position_embedding_type == "none"  # Mamba doesn't use position embeddings
        assert result.rotary_percent == 1.0
        assert result.rotary_base == 10000

    def test_provider_bridge_moe_config(self, nemotronh_8b_config_dict):
        """Test MoE configuration mapping when n_routed_experts > 0."""
        # Add MoE-specific configurations to the base config
        moe_config_dict = {
            **nemotronh_8b_config_dict,
            "n_routed_experts": 64,
            "moe_intermediate_size": 2048,
            "moe_shared_expert_intermediate_size": 8192,
            "num_experts_per_tok": 8,
            "n_group": 4,
            "topk_group": 2,
            "routed_scaling_factor": 2.0,
        }

        cfg = Mock(spec=[])
        for k, v in moe_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check MoE configuration mappings
        assert result.num_moe_experts == cfg.n_routed_experts
        assert result.moe_ffn_hidden_size == cfg.moe_intermediate_size
        assert result.moe_shared_expert_intermediate_size == cfg.moe_shared_expert_intermediate_size
        assert result.moe_router_topk == cfg.num_experts_per_tok
        assert result.moe_router_num_groups == cfg.n_group
        assert result.moe_router_group_topk == cfg.topk_group
        assert result.moe_router_topk_scaling_factor == cfg.routed_scaling_factor

    def test_provider_bridge_no_moe_when_n_routed_experts_zero(self, nemotronh_8b_config_dict):
        """Test that MoE configs are not added when n_routed_experts is 0."""
        # Add MoE config with n_routed_experts = 0
        moe_config_dict = {
            **nemotronh_8b_config_dict,
            "n_routed_experts": 0,
            "moe_intermediate_size": 2048,
            "moe_shared_expert_intermediate_size": 8192,
            "num_experts_per_tok": 8,
            "n_group": 4,
            "topk_group": 2,
            "routed_scaling_factor": 2.0,
        }

        cfg = Mock(spec=[])
        for k, v in moe_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # When n_routed_experts is 0, num_moe_experts should be 0 or None
        assert result.num_moe_experts in (0, None)

    def test_provider_bridge_moe_latent_size(self, nemotronh_8b_config_dict):
        """Test moe_latent_size mapping for Super model."""
        moe_config_dict = {
            **nemotronh_8b_config_dict,
            "n_routed_experts": 512,
            "moe_intermediate_size": 2688,
            "moe_shared_expert_intermediate_size": 5376,
            "num_experts_per_tok": 22,
            "n_group": 1,
            "topk_group": 1,
            "routed_scaling_factor": 5.0,
            "moe_latent_size": 1024,
            "moe_shared_expert_overlap": False,
        }

        cfg = Mock(spec=[])
        for k, v in moe_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check Super-specific MoE configuration
        assert result.moe_latent_size == 1024
        assert result.moe_shared_expert_overlap is False

    def test_provider_bridge_mtp_config(self, nemotronh_8b_config_dict):
        """Test MTP configuration mapping for Super model."""
        mtp_config_dict = {
            **nemotronh_8b_config_dict,
            "n_routed_experts": 0,
            "num_nextn_predict_layers": 2,
            "mtp_hybrid_override_pattern": "*E",
            "keep_mtp_spec_in_bf16": True,
        }

        cfg = Mock(spec=[])
        for k, v in mtp_config_dict.items():
            setattr(cfg, k, v)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Check MTP configuration mappings
        assert result.mtp_num_layers == 2
        assert result.mtp_hybrid_override_pattern == "*E"
        assert result.keep_mtp_spec_in_bf16 is True

    def test_provider_bridge_no_moe_when_attribute_missing(self, nemotronh_8b_config_dict):
        """Test that MoE configs are not added when n_routed_experts attribute is missing."""
        from types import SimpleNamespace

        # Create config without n_routed_experts using SimpleNamespace (hasattr returns False for missing attrs)
        config_dict = {k: v for k, v in nemotronh_8b_config_dict.items() if k != "n_routed_experts"}
        cfg = SimpleNamespace(**config_dict)

        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = cfg

        bridge = NemotronHBridge()
        result = bridge.provider_bridge(mock_pretrained)

        # Should work without MoE configs - provider should still be created
        assert isinstance(result, MambaModelProvider)
        assert not hasattr(result, "num_moe_experts") or result.num_moe_experts is None

    def test_mapping_registry_contains_moe_mappings(self):
        """Test that mapping_registry contains MoE parameter mappings."""
        bridge = NemotronHBridge()
        mapping_registry = bridge.mapping_registry()

        # Get all megatron params from mappings
        megatron_params = [m.megatron_param for m in mapping_registry.mappings if hasattr(m, "megatron_param")]

        # Check MoE router mappings exist
        assert "decoder.layers.*.mlp.router.weight" in megatron_params
        assert "decoder.layers.*.mlp.router.expert_bias" in megatron_params

        # Check MoE expert mappings exist
        assert "decoder.layers.*.mlp.experts.linear_fc1.weight*" in megatron_params
        assert "decoder.layers.*.mlp.experts.linear_fc2.weight*" in megatron_params

        # Check shared expert mappings exist
        assert "decoder.layers.*.mlp.shared_experts.linear_fc1.weight" in megatron_params
        assert "decoder.layers.*.mlp.shared_experts.linear_fc2.weight" in megatron_params

        # Check pre_mlp_layernorm mapping exists
        assert "decoder.layers.*.pre_mlp_layernorm.weight" in megatron_params

    def test_mapping_registry_contains_latent_proj_mappings(self):
        """Test that mapping_registry contains latent projection mappings for Super model."""
        bridge = NemotronHBridge()
        mapping_registry = bridge.mapping_registry()

        # Get all megatron params from mappings
        megatron_params = [m.megatron_param for m in mapping_registry.mappings if hasattr(m, "megatron_param")]

        # Check latent projection mappings exist (used by Super model)
        assert "decoder.layers.*.mlp.fc1_latent_proj.weight" in megatron_params
        assert "decoder.layers.*.mlp.fc2_latent_proj.weight" in megatron_params

    def test_mapping_registry_moe_hf_params(self):
        """Test that MoE mappings have correct HF parameter names."""
        bridge = NemotronHBridge()
        mapping_registry = bridge.mapping_registry()

        # Create a lookup dict of megatron -> hf params
        param_map = {
            m.megatron_param: m.hf_param
            for m in mapping_registry.mappings
            if hasattr(m, "megatron_param") and hasattr(m, "hf_param")
        }

        # Check MoE HF param mappings are correct
        assert param_map.get("decoder.layers.*.mlp.router.weight") == "backbone.layers.*.mixer.gate.weight"
        assert (
            param_map.get("decoder.layers.*.mlp.router.expert_bias")
            == "backbone.layers.*.mixer.gate.e_score_correction_bias"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.experts.linear_fc1.weight*")
            == "backbone.layers.*.mixer.experts.*.up_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.experts.linear_fc2.weight*")
            == "backbone.layers.*.mixer.experts.*.down_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.shared_experts.linear_fc1.weight")
            == "backbone.layers.*.mixer.shared_experts.up_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.shared_experts.linear_fc2.weight")
            == "backbone.layers.*.mixer.shared_experts.down_proj.weight"
        )

    def test_mapping_registry_latent_proj_hf_params(self):
        """Test that latent projection mappings have correct HF parameter names."""
        bridge = NemotronHBridge()
        mapping_registry = bridge.mapping_registry()

        # Create a lookup dict of megatron -> hf params
        param_map = {
            m.megatron_param: m.hf_param
            for m in mapping_registry.mappings
            if hasattr(m, "megatron_param") and hasattr(m, "hf_param")
        }

        # Check latent projection HF param mappings
        assert (
            param_map.get("decoder.layers.*.mlp.fc1_latent_proj.weight")
            == "backbone.layers.*.mixer.fc1_latent_proj.weight"
        )
        assert (
            param_map.get("decoder.layers.*.mlp.fc2_latent_proj.weight")
            == "backbone.layers.*.mixer.fc2_latent_proj.weight"
        )


class TestNemotronHBridgeTokenizerKwargs:
    """Test get_hf_tokenizer_kwargs method."""

    def test_tokenizer_kwargs_returns_dict(self):
        """Test get_hf_tokenizer_kwargs returns a dict."""
        kwargs = NemotronHBridge.get_hf_tokenizer_kwargs()
        assert isinstance(kwargs, dict)

    def test_tokenizer_kwargs_use_fast(self):
        """Test get_hf_tokenizer_kwargs returns use_fast=True."""
        kwargs = NemotronHBridge.get_hf_tokenizer_kwargs()
        assert kwargs.get("use_fast") is True


class TestAutoBridgeIntegration:
    """Integration tests for AutoBridge with NemotronH models."""

    @pytest.fixture
    def nemotronh_config_dict(self):
        """Create a sample NemotronH configuration."""
        return {
            "architectures": ["NemotronHForCausalLM"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "attention_head_dim": 128,
            "auto_map": {
                "AutoConfig": "configuration_nemotron_h.NemotronHConfig",
                "AutoModelForCausalLM": "modeling_nemotron_h.NemotronHForCausalLM",
            },
            "bos_token_id": 1,
            "chunk_size": 128,
            "conv_kernel": 4,
            "eos_token_id": 2,
            "expand": 2,
            "hidden_act": "relu2",  # Required for base class activation mapping
            "hidden_dropout": 0.0,
            "hidden_size": 4096,
            "hybrid_override_pattern": "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-",
            "initializer_range": 0.02,
            "intermediate_size": 21504,
            "layer_norm_epsilon": 1e-05,
            "mamba_head_dim": 64,
            "mamba_hidden_act": "silu",
            "mamba_num_heads": 128,
            "mamba_proj_bias": False,
            "max_position_embeddings": 8192,
            "mlp_bias": False,
            "mlp_hidden_act": "relu2",
            "model_type": "nemotron_h",
            "n_groups": 8,
            "num_attention_heads": 32,
            "num_hidden_layers": 52,
            "num_key_value_heads": 8,
            "num_logits_to_keep": 1,
            "pad_token_id": 0,
            "rescale_prenorm_residual": True,
            "residual_in_fp32": False,
            "rms_norm_eps": 1e-05,
            "ssm_state_size": 128,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.48.0.dev0",
            "use_bias": False,
            "use_cache": True,
            "use_conv_bias": True,
            "use_mamba_kernels": True,
            "vocab_size": 131072,
        }

    @pytest.fixture
    def nemotronh_config(self, nemotronh_config_dict):
        """Create mock config instance."""
        cfg = Mock(spec=PretrainedConfig)
        for k, v in nemotronh_config_dict.items():
            setattr(cfg, k, v)
        return cfg

    def create_mock_model_files(self, config_dict, save_dir):
        """Create mock model files in a directory."""
        import json

        # Save config
        config_path = Path(save_dir) / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        # Create a dummy safetensors index file
        index_path = Path(save_dir) / "model.safetensors.index.json"
        index_data = {
            "metadata": {"total_size": 1000000},
            "weight_map": {
                "backbone.embeddings.weight": "model-00001-of-00001.safetensors",
                "backbone.layers.0.mixer.in_proj.weight": "model-00001-of-00001.safetensors",
            },
        }
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        # Create tokenizer files
        tokenizer_config = {
            "model_max_length": config_dict["max_position_embeddings"],
        }
        tokenizer_path = Path(save_dir) / "tokenizer_config.json"
        with open(tokenizer_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        # Create dummy tokenizer.json
        tokenizer_json_path = Path(save_dir) / "tokenizer.json"
        tokenizer_data = {
            "version": "1.0",
            "model": {"type": "BPE"},
        }
        with open(tokenizer_json_path, "w") as f:
            json.dump(tokenizer_data, f, indent=2)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_with_temp_dir(
        self, mock_autoconfig, mock_pretrained, nemotronh_config_dict, nemotronh_config
    ):
        """Test AutoBridge.from_hf_pretrained with temporary directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = nemotronh_config_dict
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = nemotronh_config
            mock_autoconfig.return_value = config

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_model.model_name_or_path = temp_dir
            mock_pretrained.return_value = mock_model

            # Create bridge from the temp directory
            bridge = AutoBridge.from_hf_pretrained(temp_dir)

            # Verify
            assert isinstance(bridge, AutoBridge)
            assert bridge.hf_pretrained == mock_model
            mock_autoconfig.assert_called_once_with(temp_dir, trust_remote_code=False)
            mock_pretrained.assert_called_once_with(temp_dir)

    @patch("megatron.bridge.models.conversion.auto_bridge.PreTrainedCausalLM.from_pretrained")
    @patch("transformers.AutoConfig.from_pretrained")
    def test_from_pretrained_with_kwargs(
        self, mock_autoconfig, mock_pretrained, nemotronh_config_dict, nemotronh_config
    ):
        """Test AutoBridge.from_hf_pretrained with various kwargs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = nemotronh_config_dict
            self.create_mock_model_files(config_dict, temp_dir)

            # Mock the config loading
            config = nemotronh_config
            mock_autoconfig.return_value = config

            # Mock the pretrained model
            mock_model = Mock(spec=PreTrainedCausalLM)
            mock_model.config = config
            mock_pretrained.return_value = mock_model

            # Test with various kwargs
            kwargs = {
                "torch_dtype": torch.bfloat16,
                "device_map": "auto",
                "trust_remote_code": True,
            }

            _ = AutoBridge.from_hf_pretrained(temp_dir, **kwargs)

            # Verify kwargs were passed through
            mock_pretrained.assert_called_once_with(temp_dir, **kwargs)

    def test_supports_nemotronh_architectures(self, nemotronh_config):
        """Test that AutoBridge.supports correctly identifies NemotronH models."""
        config = nemotronh_config
        assert AutoBridge.supports(config) == True

        # Test non-causal LM architecture
        non_causal_config = Mock()
        non_causal_config.architectures = ["NemotronHModel"]  # Not ForCausalLM
        assert AutoBridge.supports(non_causal_config) == False


class TestReplaceWildcards:
    """Test _replace_wildcards helper function."""

    def test_single_star(self):
        result = _replace_wildcards("decoder.layers.*.weight", ("5",))
        assert result == "decoder.layers.5.weight"

    def test_two_stars(self):
        result = _replace_wildcards("mtp.layers.*.mtp_model_layer.layers.*.weight", ("0", "1"))
        assert result == "mtp.layers.0.mtp_model_layer.layers.1.weight"

    def test_double_star_then_single(self):
        result = _replace_wildcards("mtp.layers.**.experts.*.weight", ("3", "7"))
        assert result == "mtp.layers.3.experts.7.weight"

    def test_fewer_captures_than_wildcards(self):
        result = _replace_wildcards("a.*.b.*.c", ("1",))
        assert result == "a.1.b.*.c"

    def test_no_wildcards(self):
        result = _replace_wildcards("decoder.final_norm.weight", ("0",))
        assert result == "decoder.final_norm.weight"

    def test_empty_captures(self):
        result = _replace_wildcards("decoder.layers.*.weight", ())
        assert result == "decoder.layers.*.weight"


class TestMTPFlatteningMapping:
    """Test _MTPFlatteningMapping class."""

    def test_init_basic(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=2,
        )
        assert m._mtp_layers_per_block == 2
        assert m._inner_override is None
        assert m._megatron_wc == 2
        assert m._hf_wc == 1

    def test_init_with_inner_override(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.eh_proj.weight",
            hf_param="mtp.layers.*.eh_proj.weight",
            mtp_layers_per_block=3,
            inner_override=0,
        )
        assert m._inner_override == 0

    def test_init_invalid_mtp_layers(self):
        with pytest.raises(ValueError, match="mtp_layers_per_block must be > 0"):
            _MTPFlatteningMapping(
                megatron_param="mtp.layers.*.weight",
                hf_param="mtp.layers.*.weight",
                mtp_layers_per_block=0,
            )

    def test_resolve_megatron_nested(self):
        """Test resolve with Megatron captures (outer, inner) for nested params."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=3,
        )
        result = m.resolve(("1", "2"))  # outer=1, inner=2
        assert isinstance(result, AutoMapping)
        # flat = 1*3 + 2 = 5
        assert result.megatron_param == "mtp.layers.1.mtp_model_layer.layers.2.mlp.weight"
        assert result.hf_param == "mtp.layers.5.mixer.up_proj.weight"

    def test_resolve_megatron_with_inner_override(self):
        """Test resolve with inner_override (outer-only Megatron params like eh_proj)."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.eh_proj.weight",
            hf_param="mtp.layers.*.eh_proj.weight",
            mtp_layers_per_block=2,
            inner_override=0,
        )
        result = m.resolve(("3",))  # outer=3, inner_override=0
        assert isinstance(result, AutoMapping)
        # flat = 3*2 + 0 = 6
        assert result.megatron_param == "mtp.layers.3.eh_proj.weight"
        assert result.hf_param == "mtp.layers.6.eh_proj.weight"

    def test_resolve_megatron_last_inner_override(self):
        """Test resolve with inner_override pointing to last inner layer."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.final_layernorm.weight",
            hf_param="mtp.layers.*.final_layernorm.weight",
            mtp_layers_per_block=4,
            inner_override=3,  # last inner = L-1
        )
        result = m.resolve(("2",))  # outer=2
        # flat = 2*4 + 3 = 11
        assert result.hf_param == "mtp.layers.11.final_layernorm.weight"

    def test_resolve_hf_path(self):
        """Test resolve when captures come from HF reverse lookup (1 capture for 1 HF wildcard)."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=3,
        )
        # HF lookup: 1 capture (flat idx) != 2 megatron wildcards → treat_as_hf=True
        result = m.resolve(("7",))  # flat=7 → outer=2, inner=1
        assert isinstance(result, AutoMapping)
        assert result.hf_param == "mtp.layers.7.mixer.up_proj.weight"
        assert result.megatron_param == "mtp.layers.2.mtp_model_layer.layers.1.mlp.weight"

    def test_resolve_hf_path_with_inner_override(self):
        """Test HF resolve with inner_override."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.eh_proj.weight",
            hf_param="mtp.layers.*.eh_proj.weight",
            mtp_layers_per_block=2,
            inner_override=0,
        )
        # Both patterns have 1 wildcard, so hf_wc == megatron_wc → treat_as_hf=False
        # This will take the Megatron path with inner_override
        result = m.resolve(("4",))  # outer=4, inner_override=0
        assert isinstance(result, AutoMapping)
        # flat = 4*2 + 0 = 8
        assert result.hf_param == "mtp.layers.8.eh_proj.weight"

    def test_resolve_megatron_with_expert_captures(self):
        """Test resolve with extra captures (e.g., expert index)."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.experts.linear_fc1.weight*",
            hf_param="mtp.layers.*.mixer.experts.*.up_proj.weight",
            mtp_layers_per_block=2,
        )
        # 3 megatron wildcards vs 2 hf wildcards
        result = m.resolve(("1", "0", "5"))  # outer=1, inner=0, expert=5
        assert isinstance(result, AutoMapping)
        # flat = 1*2 + 0 = 2
        assert result.hf_param == "mtp.layers.2.mixer.experts.5.up_proj.weight"

    def test_resolve_megatron_missing_inner_raises(self):
        """Test that resolve raises when inner capture is missing for nested mapping."""
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.mlp.weight",
            hf_param="mtp.layers.*.mixer.up_proj.weight",
            mtp_layers_per_block=2,
        )
        # 1 capture but no inner_override and needs 2 captures
        # Note: 1 capture == hf_wc (1) and hf_wc != megatron_wc (2), so this takes HF path
        # To trigger the Megatron path error, need captures count != hf_wc
        # With 0 captures and hf_wc=1, megatron_wc=2: len(captures)=0 != hf_wc=1, so Megatron path
        with pytest.raises(ValueError, match="Expected at least 1 capture"):
            m.resolve(())

    def test_hf_to_megatron_raises(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.weight",
            hf_param="mtp.layers.*.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.hf_to_megatron(torch.tensor([1.0]), torch.nn.Linear(1, 1))

    def test_megatron_to_hf_raises(self):
        m = _MTPFlatteningMapping(
            megatron_param="mtp.layers.*.weight",
            hf_param="mtp.layers.*.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.megatron_to_hf(torch.tensor([1.0]), torch.nn.Linear(1, 1))


class TestMTPFlatteningQKVMapping:
    """Test _MTPFlatteningQKVMapping class."""

    def test_init_basic(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.weight",
            q="mtp.layers.*.mixer.q_proj.weight",
            k="mtp.layers.*.mixer.k_proj.weight",
            v="mtp.layers.*.mixer.v_proj.weight",
            mtp_layers_per_block=2,
        )
        assert m._mtp_layers_per_block == 2
        assert m.hf_param == {
            "q": "mtp.layers.*.mixer.q_proj.weight",
            "k": "mtp.layers.*.mixer.k_proj.weight",
            "v": "mtp.layers.*.mixer.v_proj.weight",
        }

    def test_init_invalid_mtp_layers(self):
        with pytest.raises(ValueError, match="mtp_layers_per_block must be > 0"):
            _MTPFlatteningQKVMapping(
                megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
                q="mtp.layers.*.q.weight",
                k="mtp.layers.*.k.weight",
                v="mtp.layers.*.v.weight",
                mtp_layers_per_block=0,
            )

    def test_resolve(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.self_attention.linear_qkv.weight",
            q="mtp.layers.*.mixer.q_proj.weight",
            k="mtp.layers.*.mixer.k_proj.weight",
            v="mtp.layers.*.mixer.v_proj.weight",
            mtp_layers_per_block=3,
        )
        result = m.resolve(("2", "1"))  # outer=2, inner=1 → flat=7
        assert isinstance(result, QKVMapping)
        assert result.megatron_param == "mtp.layers.2.mtp_model_layer.layers.1.self_attention.linear_qkv.weight"
        assert result.hf_param["q"] == "mtp.layers.7.mixer.q_proj.weight"
        assert result.hf_param["k"] == "mtp.layers.7.mixer.k_proj.weight"
        assert result.hf_param["v"] == "mtp.layers.7.mixer.v_proj.weight"

    def test_resolve_insufficient_captures_raises(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=2,
        )
        with pytest.raises(ValueError, match="Expected \\(outer, inner\\) captures"):
            m.resolve(("0",))

    def test_hf_to_megatron_raises(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.hf_to_megatron({"q": torch.tensor([1.0])}, torch.nn.Linear(1, 1))

    def test_megatron_to_hf_raises(self):
        m = _MTPFlatteningQKVMapping(
            megatron_param="mtp.layers.*.mtp_model_layer.layers.*.weight",
            q="mtp.layers.*.q.weight",
            k="mtp.layers.*.k.weight",
            v="mtp.layers.*.v.weight",
            mtp_layers_per_block=1,
        )
        with pytest.raises(NotImplementedError):
            m.megatron_to_hf(torch.tensor([1.0]), torch.nn.Linear(1, 1))


class TestNemotronHBridgeMTPIntegration:
    """Test NemotronHBridge methods related to MTP layer handling."""

    def test_build_conversion_tasks_caches_mtp_pattern(self):
        """Test that build_conversion_tasks caches mtp_layers_per_block from config."""
        bridge = NemotronHBridge()

        mock_config = Mock(spec=[])
        mock_config.mtp_hybrid_override_pattern = "*E*"
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_config

        # build_conversion_tasks calls super() which needs megatron_model — mock it
        with patch.object(MegatronModelBridge, "build_conversion_tasks", return_value=[]):
            bridge.build_conversion_tasks(mock_pretrained, Mock())

        assert bridge._mtp_layers_per_block == 3  # len("*E*")

    def test_build_conversion_tasks_no_mtp_pattern(self):
        """Test build_conversion_tasks when mtp_hybrid_override_pattern is missing."""
        bridge = NemotronHBridge()

        mock_config = Mock(spec=[])  # spec=[] means hasattr returns False for unset attrs
        mock_pretrained = Mock(spec=PreTrainedCausalLM)
        mock_pretrained.config = mock_config

        with patch.object(MegatronModelBridge, "build_conversion_tasks", return_value=[]):
            bridge.build_conversion_tasks(mock_pretrained, Mock())

        assert bridge._mtp_layers_per_block == 0

    def test_mapping_registry_with_mtp(self):
        """Test mapping_registry includes MTP flattening mappings when mtp is configured."""
        bridge = NemotronHBridge()
        bridge._mtp_layers_per_block = 2

        registry = bridge.mapping_registry()

        # Should contain _MTPFlatteningMapping instances
        mtp_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningMapping)]
        assert len(mtp_mappings) > 0

        # Check inner_override=0 mappings exist (eh_proj, enorm, hnorm)
        inner0 = [m for m in mtp_mappings if m._inner_override == 0]
        assert len(inner0) == 3

        # Check inner_override=L-1 mapping exists (final_layernorm)
        inner_last = [m for m in mtp_mappings if m._inner_override == 1]  # L-1 = 2-1 = 1
        assert len(inner_last) == 1

        # Check nested MTP mappings (inner_override=None)
        nested = [m for m in mtp_mappings if m._inner_override is None]
        assert len(nested) > 0

        # Should contain _MTPFlatteningQKVMapping
        qkv_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningQKVMapping)]
        assert len(qkv_mappings) == 1

    def test_mapping_registry_without_mtp_logs_warning(self):
        """Test mapping_registry logs warning when mtp_layers_per_block is 0."""
        bridge = NemotronHBridge()
        bridge._mtp_layers_per_block = 0

        with patch("megatron.bridge.models.nemotronh.nemotron_h_bridge.logger") as mock_logger:
            registry = bridge.mapping_registry()
            mock_logger.warning.assert_called_once()

        # Should NOT contain any MTP flattening mappings
        mtp_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningMapping)]
        assert len(mtp_mappings) == 0
        qkv_mappings = [m for m in registry.mappings if isinstance(m, _MTPFlatteningQKVMapping)]
        assert len(qkv_mappings) == 0
