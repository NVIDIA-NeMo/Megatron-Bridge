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
from megatron.core.transformer import TransformerConfig

from megatron.bridge.training.mlm_compat.model import _get_transformer_layer_spec, _gpt_provider, _mamba_provider


def common_mock_args() -> argparse.Namespace:
    args = argparse.Namespace()
    # Basic model parameters
    args.padded_vocab_size = 32000
    args.max_position_embeddings = 2048
    args.fp16_lm_cross_entropy = False
    args.untie_embeddings_and_output_weights = False
    args.position_embedding_type = "rope"
    args.rotary_percent = 1.0
    args.rotary_base = 10000

    # Transformer config parameters (needed for _transformer_config_from_args)
    args.num_layers = 3
    args.hidden_size = 256
    args.num_attention_heads = 4
    args.use_cpu_initialization = True
    args.mtp_num_layers = None
    args.sequence_parallel = False
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.expert_model_parallel_size = 1
    args.sequence_parallel_size = 1
    args.distribute_saved_activations = False
    args.overlap_p2p_comm = False
    args.batch_p2p_comm = True
    args.num_moe_experts = None
    args.squared_relu = False
    args.bias_gelu_fusion = False
    args.bias_swiglu_fusion = False
    args.init_method_xavier_uniform = False
    args.group_query_attention = False
    args.num_query_groups = None
    args.is_hybrid_model = False
    args.no_persist_layer_norm = False
    args.apply_layernorm_1p = False
    args.norm_epsilon = 1e-5
    args.params_dtype = torch.float32
    args.rotary_interleaved = False
    args.fp8_param_gather = False
    args.normalization = "LayerNorm"
    args.multi_latent_attention = False
    args.swiglu = False
    args.use_kitchen = False
    args.kitchen_config_file = None
    args.kitchen_recipe_number = None

    # MoE and expert parameters
    args.num_experts = None
    args.moe_grouped_gemm = False
    args.moe_use_legacy_grouped_gemm = False
    args.qk_layernorm = False
    args.qk_l2_norm = False

    return args


def common_mock_transformer_cfg():
    return TransformerConfig(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
    )


class TestTransformerLayerSpecRouter:
    """Test function that decides between local and TE spec."""

    @pytest.fixture
    def mock_args(self):
        """Mock args."""
        return common_mock_args()

    @patch("megatron.bridge.training.mlm_compat.model.get_gpt_layer_with_transformer_engine_spec")
    def test_te_spec(self, mock_te_spec_func, mock_args):
        """Test TE layer spec branch."""
        _get_transformer_layer_spec(mock_args, True, False)

        mock_te_spec_func.assert_called_once_with(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            multi_latent_attention=False,
            linear_attention_type=None,
            moe_use_legacy_grouped_gemm=False,
            qk_l2_norm=False,
            use_kitchen=False,
        )

    @patch("megatron.bridge.training.mlm_compat.model.get_gpt_layer_local_spec")
    def test_local_spec(self, mock_local_spec_func, mock_args):
        """Test local layer spec branch."""
        _get_transformer_layer_spec(mock_args, False, True)

        mock_local_spec_func.assert_called_once_with(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            multi_latent_attention=False,
            linear_attention_type=None,
            moe_use_legacy_grouped_gemm=False,
            normalization="LayerNorm",
            use_kitchen=True,
        )


class TestGPTProvider:
    """Test GPT model provider function."""

    @pytest.fixture
    def mock_gpt_args(self):
        """Create mock args namespace with required attributes for GPT model."""
        args = common_mock_args()

        args.use_rope_scaling = False

        # Transformer implementation
        args.transformer_impl = "local"

        # Heterogeneous layers
        args.heterogeneous_layers_config_path = None

        return args

    @pytest.fixture
    def mock_transformer_config(self):
        """Create a mock TransformerConfig for testing."""
        return common_mock_transformer_cfg()

    @pytest.fixture
    def mock_transformer_layer_spec(self):
        """Create a mock transformer layer spec."""
        mock_spec = MagicMock()
        mock_spec.name = "transformer_layer_spec"
        return mock_spec

    @patch("megatron.bridge.training.mlm_compat.model._get_transformer_layer_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_basic_local_impl(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_layer_spec_func,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test basic GPT model creation with local transformer implementation."""
        mock_config_func.return_value = mock_transformer_config
        mock_layer_spec_func.return_value = mock_transformer_layer_spec
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args)

        mock_config_func.assert_called_once_with(mock_gpt_args)
        mock_layer_spec_func.assert_called_once_with(mock_gpt_args, False, False)

        mock_gpt_model_class.assert_called_once_with(
            config=mock_transformer_config,
            transformer_layer_spec=mock_transformer_layer_spec,
            vocab_size=32000,
            max_sequence_length=2048,
            pre_process=True,
            post_process=True,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
            rope_scaling=False,
            mtp_block_spec=None,
            vp_stage=None,
        )

    @patch("megatron.bridge.training.mlm_compat.model._get_transformer_layer_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_transformer_engine_impl(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_layer_spec_func,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test GPT model creation with transformer engine implementation."""
        mock_gpt_args.transformer_impl = "transformer_engine"
        mock_config_func.return_value = mock_transformer_config
        mock_layer_spec_func.return_value = mock_transformer_layer_spec
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args)

        mock_layer_spec_func.assert_called_once_with(mock_gpt_args, True, False)

    @patch("megatron.bridge.training.mlm_compat.model._get_transformer_layer_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_with_custom_config(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_layer_spec_func,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test GPT model creation with custom config parameter."""
        mock_layer_spec_func.return_value = mock_transformer_layer_spec
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args, config=mock_transformer_config)

        mock_config_func.assert_not_called()
        call_args = mock_gpt_model_class.call_args
        assert call_args[1]["config"] == mock_transformer_config

    @patch("megatron.bridge.training.mlm_compat.model.get_gpt_decoder_block_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_with_moe_experts(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_decoder_block_spec,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test GPT model creation with MoE experts."""
        mock_gpt_args.num_experts = 8
        mock_config_func.return_value = mock_transformer_config
        mock_decoder_block_spec.return_value = mock_transformer_layer_spec
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args)

        mock_decoder_block_spec.assert_called_once_with(
            mock_transformer_config,
            use_transformer_engine=False,
            normalization="LayerNorm",
            qk_l2_norm=False,
            vp_stage=None,
        )

    @patch("megatron.bridge.training.mlm_compat.model.get_gpt_heterogeneous_layer_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_with_heterogeneous_layers(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_heterogeneous_spec,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test GPT model creation with heterogeneous layers."""
        mock_gpt_args.heterogeneous_layers_config_path = "/path/to/config"
        mock_config_func.return_value = mock_transformer_config
        mock_heterogeneous_spec.return_value = mock_transformer_layer_spec
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args)

        mock_heterogeneous_spec.assert_called_once_with(mock_transformer_config, False)

    @patch("megatron.bridge.training.mlm_compat.model.get_gpt_mtp_block_spec")
    @patch("megatron.bridge.training.mlm_compat.model._get_transformer_layer_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_with_mtp_block_spec(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_layer_spec_func,
        mock_mtp_spec,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test GPT model creation with MTP block spec."""
        mock_gpt_args.mtp_num_layers = 4
        mock_config_func.return_value = mock_transformer_config
        mock_layer_spec_func.return_value = mock_transformer_layer_spec
        mock_mtp_spec.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args)

        mock_mtp_spec.assert_called_once_with(
            mock_transformer_config,
            mock_transformer_layer_spec,
            use_transformer_engine=False,
            vp_stage=None,
        )

    @patch("megatron.bridge.training.mlm_compat.model.get_gpt_mtp_block_spec")
    @patch("megatron.bridge.training.mlm_compat.model._get_transformer_layer_spec")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.GPTModel")
    def test_gpt_provider_with_mtp_and_empty_layer_specs(
        self,
        mock_gpt_model_class,
        mock_config_func,
        mock_layer_spec_func,
        mock_mtp_spec,
        mock_gpt_args,
        mock_transformer_config,
        mock_transformer_layer_spec,
    ):
        """Test GPT model creation with MTP and empty layer specs."""
        mock_gpt_args.mtp_num_layers = 4
        mock_config_func.return_value = mock_transformer_config

        # Mock layer spec with empty layer_specs
        mock_layer_spec = MagicMock()
        mock_layer_spec.layer_specs = []
        mock_layer_spec_func.return_value = mock_layer_spec

        mock_mtp_spec.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_gpt_model_class.return_value = mock_model_instance

        _gpt_provider(mock_gpt_args)

        # Should call _get_transformer_layer_spec twice: once for the main spec and once for MTP
        assert mock_layer_spec_func.call_count == 2


class TestMambaModelProvider:
    """Test Mamba model provider function."""

    @pytest.fixture
    def mock_args(self):
        """Create mock args namespace with required attributes for Mamba model."""
        args = common_mock_args()

        args.spec = "megatron.core.models.mamba.mamba_layer_specs.mamba_stack_spec"

        # Hybrid model parameters
        args.hybrid_attention_ratio = 0.3
        args.hybrid_mlp_ratio = 0.3
        args.hybrid_override_pattern = None

        return args

    @pytest.fixture
    def mock_transformer_config(self):
        """Create a mock TransformerConfig for testing."""
        return common_mock_transformer_cfg()

    @pytest.fixture
    def mock_mamba_stack_spec(self):
        """Create a mock mamba stack spec."""
        mock_spec = MagicMock()
        mock_spec.name = "mamba_stack_spec"
        return mock_spec

    @patch("megatron.bridge.training.mlm_compat.model.import_module")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.MambaModel")
    def test_mamba_provider_basic(
        self,
        mock_mamba_model_class,
        mock_config_func,
        mock_import,
        mock_args,
        mock_transformer_config,
        mock_mamba_stack_spec,
    ):
        """Test basic Mamba model creation with default parameters."""
        mock_import.return_value = mock_mamba_stack_spec
        mock_config_func.return_value = mock_transformer_config
        mock_model_instance = MagicMock()
        mock_mamba_model_class.return_value = mock_model_instance

        _mamba_provider(mock_args)

        mock_import.assert_called_once_with(mock_args.spec)
        mock_config_func.assert_called_once_with(mock_args)

        mock_mamba_model_class.assert_called_once_with(
            config=mock_transformer_config,
            mamba_stack_spec=mock_mamba_stack_spec,
            vocab_size=32000,
            max_sequence_length=2048,
            pre_process=True,
            hybrid_attention_ratio=0.3,
            hybrid_mlp_ratio=0.3,
            hybrid_override_pattern=None,
            post_process=True,
            fp16_lm_cross_entropy=False,
            parallel_output=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type="rope",
            rotary_percent=1.0,
            rotary_base=10000,
        )

    @patch("megatron.bridge.training.mlm_compat.model.import_module")
    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    @patch("megatron.bridge.training.mlm_compat.model.MambaModel")
    def test_mamba_provider_transformer_config_not_called_when_provided(
        self,
        mock_mamba_model_class,
        mock_config_func,
        mock_import,
        mock_args,
        mock_transformer_config,
        mock_mamba_stack_spec,
    ):
        """Test that _transformer_config_from_args is not called when config is provided."""
        mock_import.return_value = mock_mamba_stack_spec

        _mamba_provider(mock_args, config=mock_transformer_config)

        mock_config_func.assert_not_called()
        call_args = mock_mamba_model_class.call_args
        assert call_args[1]["config"] == mock_transformer_config

    @patch("megatron.bridge.training.mlm_compat.model._transformer_config_from_args")
    def test_mamba_no_stack_spec(
        self,
        mock_config_func,
        mock_args,
        mock_transformer_config,
        mock_mamba_stack_spec,
    ):
        """Test failure without stack spec."""
        mock_config_func.return_value = mock_transformer_config
        mock_args.spec = None

        with pytest.raises(AssertionError, match="You must provide a valid Mamba layer spec!"):
            _mamba_provider(mock_args)
