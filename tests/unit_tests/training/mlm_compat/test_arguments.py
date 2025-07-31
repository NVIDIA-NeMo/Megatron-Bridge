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

import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.heterogeneous.heterogeneous_config import HeterogeneousTransformerConfig

from megatron.bridge.training.mlm_compat.arguments import (
    _load_args_from_checkpoint,
    _tokenizer_config_from_args,
    _transformer_config_from_args,
)
from megatron.bridge.training.tokenizers.config import TokenizerConfig


class TestLoadArgsFromCheckpoint:
    """Test behavior when loading MLM args from checkpoint."""

    @pytest.fixture
    def mock_args(self):
        """Mock args namespace for testing."""
        args = argparse.Namespace()
        setattr(args, "tensor_model_parallel_size", 1)
        setattr(args, "num_attention_heads", 8)
        return args

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_args_present(self, mock_dist_ckpt, mock_args):
        """Test behavior when state dict contains args."""
        mock_dist_ckpt.load_common_state_dict.return_value = {"args": mock_args}

        loaded_args = _load_args_from_checkpoint("/test/checkpoint")

        mock_dist_ckpt.load_common_state_dict.assert_called_once_with("/test/checkpoint")
        assert loaded_args == mock_args

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_args_not_present(self, mock_dist_ckpt):
        """Test behavior when state dict does not contain args."""
        mock_dist_ckpt.load_common_state_dict.return_value = {}

        with pytest.raises(AssertionError) as exc_info:
            _load_args_from_checkpoint("/test/checkpoint")

        mock_dist_ckpt.load_common_state_dict.assert_called_once_with("/test/checkpoint")
        assert "Provided checkpoint does not have arguments saved" in str(exc_info.value)

    @patch("megatron.bridge.training.checkpointing.dist_checkpointing")
    def test_no_state_dict(self, mock_dist_ckpt):
        """Test behavior when state dict not returned."""
        mock_dist_ckpt.load_common_state_dict.return_value = None

        with pytest.raises(AssertionError) as exc_info:
            _load_args_from_checkpoint("/test/checkpoint")

        mock_dist_ckpt.load_common_state_dict.assert_called_once_with("/test/checkpoint")
        assert "Could not load state from checkpoint" in str(exc_info.value)


class TestTokenizerConfigFromArgs:
    """Test extracting TokenizerConfig from argparse args."""

    @pytest.fixture
    def mock_args(self):
        """Mock args namespace for testing."""
        args = argparse.Namespace()
        setattr(args, "vocab_size", 32000)
        setattr(args, "tokenizer_type", "SentencePieceTokenizer")
        setattr(args, "tokenizer_model", "/path/to/mock/tokenizer.model")
        setattr(args, "tensor_model_parallel_size", 1)
        setattr(args, "num_attention_heads", 8)
        return args

    def test_with_complete_args(self, mock_args):
        """Test an expected successful build from args."""
        cfg = _tokenizer_config_from_args(mock_args)

        assert isinstance(cfg, TokenizerConfig)
        assert cfg.vocab_size == 32000
        assert cfg.tokenizer_type == "SentencePieceTokenizer"
        assert cfg.tokenizer_model == "/path/to/mock/tokenizer.model"

    def test_with_empty_args(self):
        """Test with no relevant args provided."""
        fn_cfg = _tokenizer_config_from_args(argparse.Namespace())
        default_cfg = TokenizerConfig()

        assert fn_cfg == default_cfg


class TestTransformerConfigFromArgs:
    """Test extracting TransformerConfig from argparse args."""

    @pytest.fixture
    def basic_args(self):
        """Mock basic args namespace for testing."""
        args = argparse.Namespace()
        # Basic transformer args
        setattr(args, "num_layers", 12)
        setattr(args, "hidden_size", 768)
        setattr(args, "num_attention_heads", 12)
        setattr(args, "max_position_embeddings", 2048)
        setattr(args, "params_dtype", torch.float32)

        # Layer norm args
        setattr(args, "no_persist_layer_norm", False)
        setattr(args, "apply_layernorm_1p", False)
        setattr(args, "norm_epsilon", 1e-5)

        # Pipeline args
        setattr(args, "overlap_p2p_comm", False)
        setattr(args, "decoder_first_pipeline_num_layers", 4)
        setattr(args, "decoder_last_pipeline_num_layers", 4)

        # Other args
        setattr(args, "num_experts", 8)
        setattr(args, "rotary_interleaved", False)
        setattr(args, "fp8_param_gather", False)
        setattr(args, "swiglu", False)
        setattr(args, "bias_gelu_fusion", False)
        setattr(args, "squared_relu", False)
        setattr(args, "init_method_xavier_uniform", False)
        setattr(args, "group_query_attention", False)
        setattr(args, "config_logger_dir", None)
        setattr(args, "cp_comm_type", ["ring"])
        setattr(args, "is_hybrid_model", False)

        # Multi-latent attention args
        setattr(args, "multi_latent_attention", False)
        setattr(args, "heterogeneous_layers_config_path", None)

        return args

    def test_basic_transformer_config(self, basic_args):
        """Test basic transformer config creation."""
        cfg = _transformer_config_from_args(basic_args)

        assert isinstance(cfg, TransformerConfig)
        assert cfg.num_layers == 12
        assert cfg.hidden_size == 768
        assert cfg.num_attention_heads == 12
        assert cfg.max_position_embeddings == 2048
        assert cfg.persist_layer_norm is True  # not no_persist_layer_norm
        assert cfg.layernorm_zero_centered_gamma is False
        assert cfg.layernorm_epsilon == 1e-5
        assert cfg.deallocate_pipeline_outputs is True
        assert cfg.pipeline_dtype == torch.float32
        assert cfg.batch_p2p_comm is True  # not overlap_p2p_comm
        assert cfg.num_moe_experts == 8
        assert cfg.rotary_interleaved is False
        assert cfg.num_layers_in_first_pipeline_stage == 4
        assert cfg.num_layers_in_last_pipeline_stage == 4
        assert cfg.fp8_param is False
        assert cfg.bias_activation_fusion is False
        assert cfg.num_query_groups is None
        assert cfg.cp_comm_type == "ring"
        assert cfg.is_hybrid_model is False
        # Should not have use_kitchen or quant_recipe attributes
        assert not hasattr(cfg, "use_kitchen")
        assert not hasattr(cfg, "quant_recipe")

    def test_multi_latent_attention_config(self, basic_args):
        """Test multi-latent attention config creation."""
        basic_args.multi_latent_attention = True

        cfg = _transformer_config_from_args(basic_args)

        assert isinstance(cfg, MLATransformerConfig)
        assert cfg.multi_latent_attention is True

    def test_heterogeneous_transformer_config(self, basic_args):
        """Test heterogeneous transformer config creation."""
        basic_args.heterogeneous_layers_config_path = "/path/to/config.json"

        cfg = _transformer_config_from_args(basic_args)

        assert isinstance(cfg, HeterogeneousTransformerConfig)
        assert cfg.heterogeneous_layers_config_path == "/path/to/config.json"

    def test_heterogeneous_with_multi_latent_error(self, basic_args):
        """Test error when both heterogeneous and multi-latent attention are enabled."""
        basic_args.multi_latent_attention = True
        basic_args.heterogeneous_layers_config_path = "/path/to/config.json"

        with pytest.raises(AssertionError) as exc_info:
            _transformer_config_from_args(basic_args)

        assert "Multi latent attention with heterogeneous layers is not supported" in str(exc_info.value)

    def test_swiglu_activation(self, basic_args):
        """Test SWiGLU activation configuration."""
        basic_args.swiglu = True
        basic_args.bias_swiglu_fusion = True

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.activation_func == F.silu
        assert cfg.gated_linear_unit is True
        assert cfg.bias_activation_fusion is True

    def test_squared_relu_activation(self, basic_args):
        """Test squared ReLU activation configuration."""
        basic_args.squared_relu = True

        cfg = _transformer_config_from_args(basic_args)

        # Check that activation_func is set to squared_relu function
        assert cfg.activation_func is not None
        # We can't directly compare functions, but we can check it's not the default

    def test_squared_relu_with_swiglu_error(self, basic_args):
        """Test error when both squared_relu and swiglu are enabled."""
        basic_args.squared_relu = True
        basic_args.swiglu = True

        with pytest.raises(AssertionError) as exc_info:
            _transformer_config_from_args(basic_args)

        assert "assert not args.swiglu" in str(exc_info.value)

    def test_xavier_uniform_init_method(self, basic_args):
        """Test Xavier uniform initialization method."""
        basic_args.init_method_xavier_uniform = True

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.init_method == torch.nn.init.xavier_uniform_
        assert cfg.scaled_init_method == torch.nn.init.xavier_uniform_

    def test_group_query_attention(self, basic_args):
        """Test group query attention configuration."""
        basic_args.group_query_attention = True
        basic_args.num_query_groups = 4

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.num_query_groups == 4

    def test_group_query_attention_disabled(self, basic_args):
        """Test group query attention when disabled."""
        basic_args.group_query_attention = False

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.num_query_groups is None

    def test_cp_comm_type_single_element(self, basic_args):
        """Test cp_comm_type with single element list."""
        basic_args.cp_comm_type = ["nccl"]

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.cp_comm_type == "nccl"

    def test_cp_comm_type_multiple_elements(self, basic_args):
        """Test cp_comm_type with multiple elements (should remain as list)."""
        basic_args.cp_comm_type = ["nccl", "ring"]

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.cp_comm_type == ["nccl", "ring"]

    def test_hybrid_model(self, basic_args):
        """Test hybrid model configuration."""
        basic_args.is_hybrid_model = True

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.is_hybrid_model is True

    @patch("megatron.bridge.training.mlm_compat.arguments.load_quantization_recipe")
    def test_kitchen_config_file(self, mock_load_recipe, basic_args):
        """Test kitchen quantization with config file."""
        basic_args.kitchen_config_file = "/path/to/kitchen_config.json"
        mock_recipe = MagicMock()
        mock_load_recipe.return_value = mock_recipe

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.use_kitchen is True
        assert cfg.quant_recipe == mock_recipe
        mock_load_recipe.assert_called_once_with("/path/to/kitchen_config.json")

    @patch("megatron.bridge.training.mlm_compat.arguments.kitchen_quantization_recipe_config")
    def test_kitchen_recipe_number(self, mock_kitchen_config, basic_args):
        """Test kitchen quantization with recipe number."""
        basic_args.kitchen_recipe_number = 42
        mock_recipe = MagicMock()
        mock_kitchen_config.return_value = mock_recipe

        cfg = _transformer_config_from_args(basic_args)

        assert cfg.use_kitchen is True
        assert cfg.quant_recipe == mock_recipe
        mock_kitchen_config.assert_called_once_with(42)

    def test_custom_config_class(self, basic_args):
        """Test using a custom config class."""

        class CustomTransformerConfig(TransformerConfig):
            custom_field: int = 42

        basic_args.custom_field = 100

        cfg = _transformer_config_from_args(basic_args, CustomTransformerConfig)

        assert isinstance(cfg, CustomTransformerConfig)
        assert cfg.custom_field == 100

    def test_args_with_missing_fields(self, basic_args):
        """Test behavior when args are missing some fields."""
        # Remove some fields from basic_args
        delattr(basic_args, "num_layers")
        delattr(basic_args, "hidden_size")

        cfg = _transformer_config_from_args(basic_args)

        # Should still create a valid config with default values
        assert isinstance(cfg, TransformerConfig)
        # Fields not in args should use default values from TransformerConfig
