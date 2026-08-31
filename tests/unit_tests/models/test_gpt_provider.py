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

import inspect
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.models import gpt_provider
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.hybridep import HYBRIDEP_PADDING_FIELDS, register_hybridep_thd_padding
from megatron.bridge.models.transformer_config import (
    HeterogeneousTransformerConfig,
    MLATransformerConfig,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.instantiate_utils import instantiate


class _ConfigModule(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


class _ConfigModel(torch.nn.Module):
    def __init__(self, config, child_configs=()):
        super().__init__()
        self.config = config
        self.config_modules = torch.nn.ModuleList(_ConfigModule(child_config) for child_config in child_configs)

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        decoder_input=None,
        labels=None,
        inference_context=None,
        packed_seq_params=None,
    ):
        return self.config


def _make_hybridep_config(config_type=GPTModelProvider, **kwargs):
    defaults = dict(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=8,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
    )
    padding_field = next(field for field in HYBRIDEP_PADDING_FIELDS if field in config_type.__dataclass_fields__)
    return config_type(**defaults, **{padding_field: False}, **kwargs), padding_field


def _make_hybridep_manager(config):
    from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager

    manager = object.__new__(_HybridEPManager)
    manager.group = object()
    manager.num_local_experts = 1
    manager.num_experts = config.num_moe_experts
    manager.config = config
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.capacity = None
    manager.num_permuted_tokens = None
    manager.moe_expert_rank_capacity_factor = None
    manager._original_num_tokens = None
    manager._padded_num_tokens = None
    manager.handle = None
    manager.pad_multiple = None
    return manager


class TestHybridEPLayoutAwarePadding:
    @pytest.mark.parametrize(
        "config_type",
        [GPTModelProvider, MLATransformerConfig, HeterogeneousTransformerConfig],
    )
    def test_automatic_padding_follows_runtime_layout(self, config_type):
        config, padding_field = _make_hybridep_config(config_type)
        model = _ConfigModel(config)
        register_hybridep_thd_padding(model, config)

        model(packed_seq_params=SimpleNamespace(qkv_format="thd"))
        assert getattr(config, padding_field) is True

        model(packed_seq_params=SimpleNamespace(qkv_format="bshd"))
        assert getattr(config, padding_field) is False

        model()
        assert getattr(config, padding_field) is False

    def test_heterogeneous_layer_configs_follow_runtime_layout(self):
        config, padding_field = _make_hybridep_config(HeterogeneousTransformerConfig)
        layer_config = SimpleNamespace(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            **{padding_field: False},
        )
        model = _ConfigModel(config, child_configs=(layer_config,))
        register_hybridep_thd_padding(model, config)

        model(packed_seq_params=SimpleNamespace(qkv_format="thd"))
        assert getattr(config, padding_field) is True
        assert getattr(layer_config, padding_field) is True

        model()
        assert getattr(config, padding_field) is False
        assert getattr(layer_config, padding_field) is False

    def test_thd_uses_mcore_safe_path_and_bshd_skips_it(self):
        config, padding_field = _make_hybridep_config()
        model = _ConfigModel(config)
        register_hybridep_thd_padding(model, config)
        routing_map = torch.zeros((100, config.num_moe_experts), dtype=torch.bool)
        routing_map[:, 0] = True
        probs = routing_map.float()

        model(packed_seq_params=SimpleNamespace(qkv_format="thd"))
        thd_manager = _make_hybridep_manager(config)

        def set_group_max(token_count, **_kwargs):
            token_count.fill_(500)

        with patch("torch.distributed.all_reduce", side_effect=set_group_max) as mock_all_reduce:
            thd_manager.setup_metadata(routing_map, probs)

        mock_all_reduce.assert_called_once()
        assert thd_manager._original_num_tokens == 100
        assert thd_manager._padded_num_tokens == 512
        assert thd_manager.routing_map.shape == (512, config.num_moe_experts)
        assert thd_manager.token_probs.shape == (512, config.num_moe_experts)
        assert not thd_manager.routing_map[100:].any()
        assert not thd_manager.token_probs[100:].any()

        thd_manager.handle = object()
        combined = torch.ones((512, config.hidden_size))
        with patch(
            "megatron.core.transformer.moe.token_dispatcher.hybrid_ep_combine",
            return_value=combined,
        ):
            restored = thd_manager.combine(combined)
        assert restored.shape == (100, config.hidden_size)

        model()
        assert getattr(config, padding_field) is False
        bshd_manager = _make_hybridep_manager(config)
        with patch("torch.distributed.all_reduce") as mock_all_reduce:
            bshd_manager.setup_metadata(routing_map, probs)

        mock_all_reduce.assert_not_called()
        assert bshd_manager._padded_num_tokens == 100
        assert bshd_manager.routing_map.shape == (100, config.num_moe_experts)
        assert bshd_manager.token_probs.shape == (100, config.num_moe_experts)

    def test_explicit_padding_setting_is_preserved_for_bshd(self):
        config, padding_field = _make_hybridep_config()
        setattr(config, padding_field, True)
        model = _ConfigModel(config)

        register_hybridep_thd_padding(model, config)
        model()

        assert not model._forward_pre_hooks
        assert getattr(config, padding_field) is True

    def test_cuda_graph_setting_disables_automatic_padding(self):
        config, padding_field = _make_hybridep_config(cuda_graph_impl="local")
        model = _ConfigModel(config)

        register_hybridep_thd_padding(model, config)
        model(packed_seq_params=SimpleNamespace(qkv_format="thd"))

        assert not model._forward_pre_hooks
        assert getattr(config, padding_field) is False

    def test_mcore_dev_padding_field_follows_runtime_layout(self):
        config = SimpleNamespace(
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_hybridep_pad_variable_tokens=False,
            cuda_graph_impl="none",
        )
        model = _ConfigModel(config)
        register_hybridep_thd_padding(model, config)

        model(packed_seq_params=SimpleNamespace(qkv_format="thd"))
        assert config.moe_hybridep_pad_variable_tokens is True

        model()
        assert config.moe_hybridep_pad_variable_tokens is False

    def test_positional_packed_seq_params_follows_runtime_layout(self):
        config, padding_field = _make_hybridep_config()
        model = _ConfigModel(config)
        register_hybridep_thd_padding(model, config)

        model(None, None, None, None, None, None, SimpleNamespace(qkv_format="thd"))

        assert getattr(config, padding_field) is True

    def test_hybrid_provider_registers_layout_aware_padding(self):
        padding_field = next(
            field for field in HYBRIDEP_PADDING_FIELDS if field in HybridModelProvider.__dataclass_fields__
        )
        provider = HybridModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            num_moe_experts=8,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            **{padding_field: False},
        )
        provider._pg_collection = type("PG", (), {"pp": object()})()
        model = _ConfigModel(provider)

        with patch("megatron.bridge.models.hybrid.hybrid_provider.MCoreHybridModel", return_value=model):
            result = provider.provide(pre_process=True, post_process=True)

        result(packed_seq_params=SimpleNamespace(qkv_format="thd"))
        assert getattr(provider, padding_field) is True

        result()
        assert getattr(provider, padding_field) is False


class TestGPTModelProvider:
    """Test cases for GPTModelProvider class."""

    def test_gpt_model_provider_initialization(self):
        """Test GPTModelProvider can be initialized with default values."""
        provider = GPTModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
        )

        # Check required transformer config fields
        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 12

        # Check GPT-specific defaults
        assert provider.fp16_lm_cross_entropy is False
        assert provider.parallel_output is True
        assert provider.share_embeddings_and_output_weights is True
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.position_embedding_type == "learned_absolute"
        assert provider.rotary_base == 10000
        assert provider.rotary_percent == 1.0
        assert provider.seq_length == 1024
        assert provider.mtp_enabled is False
        assert provider.logit_dtype is None

    @pytest.mark.skipif(
        "logit_dtype" not in inspect.signature(gpt_provider.MCoreGPTModel).parameters,
        reason="Installed MCore predates logit_dtype",
    )
    def test_provide_propagates_requested_logit_dtype(self):
        """Test the requested output-logit dtype reaches MCore."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            logit_dtype=torch.float32,
        )
        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel", autospec=True) as mock_model:
            provider.provide(pre_process=True, post_process=True)

        assert mock_model.call_args.kwargs["logit_dtype"] is torch.float32

    @pytest.mark.skipif(
        "logit_dtype" in inspect.signature(gpt_provider.MCoreGPTModel).parameters,
        reason="Installed MCore supports logit_dtype",
    )
    def test_requested_logit_dtype_fails_clearly_on_old_mcore(self):
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            logit_dtype=torch.float32,
        )
        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        with pytest.raises(RuntimeError, match="Megatron-LM PR #6252"):
            provider.provide(pre_process=True, post_process=True)

    def test_logit_dtype_survives_provider_serialization(self):
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            logit_dtype=torch.float32,
        )

        restored = instantiate(ConfigContainer._convert_value_to_dict(provider))

        assert restored.logit_dtype is torch.float32

    def test_gpt_model_provider_with_rope(self):
        """Test GPTModelProvider with RoPE embeddings."""
        provider = GPTModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            position_embedding_type="rope",
            rotary_percent=0.5,
            seq_len_interpolation_factor=2.0,
        )

        assert provider.position_embedding_type == "rope"
        assert provider.rotary_percent == 0.5
        assert provider.seq_len_interpolation_factor == 2.0

    def test_provide_method_basic(self):
        """Test the provide method creates a GPT model."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        # Provide minimal pg_collection for provider
        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        # Mock dependencies
        with patch("megatron.bridge.models.gpt_provider.calculate_padded_vocab_size", return_value=1024):
            with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance

                result = provider.provide(pre_process=True, post_process=True)

                assert result == mock_instance
                mock_model.assert_called_once()
                assert "logit_dtype" not in mock_model.call_args.kwargs

    def test_provide_registers_layout_aware_hybridep_padding(self):
        provider, _ = _make_hybridep_config(vocab_size=1000)
        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_model:
            mock_instance = Mock()
            mock_instance.config = provider
            mock_instance.modules.return_value = [mock_instance]
            mock_model.return_value = mock_instance

            result = provider.provide(pre_process=True, post_process=True)

        assert result is mock_instance
        mock_instance.register_forward_pre_hook.assert_called_once()
        assert mock_instance.register_forward_pre_hook.call_args.kwargs == {"with_kwargs": True}

    def test_provide_method_with_vocab_padding(self):
        """Test provide method calculates padded vocab size when padding is enabled."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=True,  # Enable padding
        )

        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        with patch(
            "megatron.bridge.models.gpt_provider.calculate_padded_vocab_size", return_value=50176
        ) as mock_calc_vocab:
            with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance

                _ = provider.provide(pre_process=True, post_process=True)

                # Verify calculate_padded_vocab_size was called with correct parameters
                mock_calc_vocab.assert_called_once_with(50000, 128, 8)
                # Verify model was created with padded vocab size
                call_kwargs = mock_model.call_args.kwargs
                assert call_kwargs["vocab_size"] == 50176

    def test_provide_method_no_vocab_padding(self):
        """Test provide method uses original vocab size when padding is disabled."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=False,  # Disable padding
        )

        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        with patch("megatron.bridge.models.gpt_provider.calculate_padded_vocab_size") as mock_calc_vocab:
            with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_model:
                mock_instance = Mock()
                mock_model.return_value = mock_instance

                _ = provider.provide(pre_process=True, post_process=True)

                # Verify calculate_padded_vocab_size was NOT called
                mock_calc_vocab.assert_not_called()
                # Verify model was created with original vocab size
                call_kwargs = mock_model.call_args.kwargs
                assert call_kwargs["vocab_size"] == 50000

    def test_provide_method_pipeline_stages(self):
        """Test provide method respects pipeline stage arguments."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        provider._pg_collection = type("PG", (), {"pp": object(), "tp": object(), "cp": object()})()

        with patch("megatron.bridge.models.gpt_provider.calculate_padded_vocab_size", return_value=1024):
            with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_gpt:
                mock_instance = Mock()
                mock_gpt.return_value = mock_instance

                provider.provide(pre_process=False, post_process=True)

                # Check the model was called with provided pipeline stages
                call_kwargs = mock_gpt.call_args.kwargs
                assert call_kwargs["pre_process"] is False
                assert call_kwargs["post_process"] is True

    def test_fp8_configuration(self):
        """Test GPTModelProvider with FP8 configuration."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            fp8="e4m3",
            fp8_margin=2,
            fp8_interval=100,
            fp8_amax_history_len=512,
            fp8_amax_compute_algo="max",
        )

        assert provider.fp8 == "e4m3"
        assert provider.fp8_margin == 2
        assert provider.fp8_interval == 100
        assert provider.fp8_amax_history_len == 512
        assert provider.fp8_amax_compute_algo == "max"

    def test_fusion_settings(self):
        """Test fusion configuration defaults."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )

        # These should be set by default factories or explicit values
        assert isinstance(provider.masked_softmax_fusion, bool)
        assert provider.cross_entropy_loss_fusion is True
        assert isinstance(provider.gradient_accumulation_fusion, bool)
        assert provider.bias_activation_fusion is False
        assert provider.persist_layer_norm is False
        assert isinstance(provider.bias_dropout_fusion, bool)
        assert isinstance(provider.apply_rope_fusion, bool)

    def test_communication_overlap_config(self):
        """Test tensor parallel communication overlap configuration."""
        tp_config = {"method": "ring", "num_splits": 4}

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            tp_comm_overlap_cfg=tp_config,
        )

        assert provider.tp_comm_overlap_cfg == tp_config

    def test_minimal_configuration(self):
        """Test that minimal configuration works."""
        # GPTModelProvider should work with minimal required fields
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )
        assert provider.num_layers == 2
        assert provider.hidden_size == 128
        assert provider.num_attention_heads == 4

    def test_multi_token_prediction(self):
        """Test MTP (multi-token prediction) configuration."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            mtp_enabled=True,
        )

        assert provider.mtp_enabled is True

    def test_scatter_embedding_config(self):
        """Test scatter embedding sequence parallel configuration."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            scatter_embedding_sequence_parallel=False,
        )

        assert provider.scatter_embedding_sequence_parallel is False

    def test_attention_softmax_fp32(self):
        """Test attention softmax in FP32 configuration."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            attention_softmax_in_fp32=True,
        )

        assert provider.attention_softmax_in_fp32 is True

    @patch("megatron.core.parallel_state")
    @patch("megatron.bridge.models.gpt_provider.get_gpt_modelopt_spec")
    def test_modelopt_transformer_layer_spec(self, mock_get_gpt_modelopt_spec, mock_parallel_state):
        """Test modelopt_transformer_layer_spec function."""
        from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

        # Mock context parallel world size to return 1 (use_arbitrary_attention_mask will be True)
        mock_parallel_state.get_context_parallel_world_size.return_value = 1

        # Create a mock provider
        provider = GPTModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
        )

        # Mock the return value
        mock_spec = Mock()
        mock_get_gpt_modelopt_spec.return_value = mock_spec

        # Call the function
        result = modelopt_transformer_layer_spec(provider)

        # Verify the mock was called with correct parameters
        mock_get_gpt_modelopt_spec.assert_called_once_with(
            config=provider,
            local_core_attention=False,
            remap_te_layernorm=True,
            real_quant_cfg="None",
            use_arbitrary_attention_mask=True,
        )

        # Verify the result
        assert result is mock_spec

    @patch("megatron.bridge.models.gpt_provider.transformer_engine_layer_spec")
    @patch("megatron.bridge.models.gpt_provider.transformer_engine_full_layer_spec")
    def test_default_layer_spec_with_restore_modelopt_state(self, mock_te_full_spec, mock_te_spec):
        """Test default_layer_spec when restore_modelopt_state is True uses TE spec."""
        from megatron.bridge.models.gpt_provider import default_layer_spec

        # Create a provider with restore_modelopt_state=True
        provider = GPTModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            restore_modelopt_state=True,
        )

        # Mock return values
        mock_te_full_spec.return_value = "te_full_spec"
        mock_te_spec.return_value = "te_spec"

        # Call the function
        result = default_layer_spec(provider)

        # Should use TE spec even when restore_modelopt_state is True (all models support TE spec)
        mock_te_full_spec.assert_not_called()
        mock_te_spec.assert_called_once_with(provider)
        assert result == "te_spec"

    @patch("megatron.bridge.models.gpt_provider.transformer_engine_layer_spec")
    @patch("megatron.bridge.models.gpt_provider.transformer_engine_full_layer_spec")
    def test_default_layer_spec_with_te_full_layer_spec(self, mock_te_full_spec, mock_te_spec):
        """Test default_layer_spec when use_transformer_engine_full_layer_spec is True."""
        from megatron.bridge.models.gpt_provider import default_layer_spec

        # Create a provider with use_transformer_engine_full_layer_spec=True
        provider = GPTModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            restore_modelopt_state=False,
            use_transformer_engine_full_layer_spec=True,
        )

        # Mock return values
        mock_te_full_spec.return_value = "te_full_spec"
        mock_te_spec.return_value = "te_spec"

        # Call the function
        result = default_layer_spec(provider)

        # Should use TE full spec when use_transformer_engine_full_layer_spec is True
        mock_te_full_spec.assert_called_once_with(provider)
        mock_te_spec.assert_not_called()
        assert result == "te_full_spec"

    @patch("megatron.bridge.models.gpt_provider.transformer_engine_layer_spec")
    @patch("megatron.bridge.models.gpt_provider.transformer_engine_full_layer_spec")
    def test_default_layer_spec_default_case(self, mock_te_full_spec, mock_te_spec):
        """Test default_layer_spec default case (regular TE spec)."""
        from megatron.bridge.models.gpt_provider import default_layer_spec

        # Create a provider with default settings
        provider = GPTModelProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=12,
            restore_modelopt_state=False,
            use_transformer_engine_full_layer_spec=False,
        )

        # Mock return values
        mock_te_full_spec.return_value = "te_full_spec"
        mock_te_spec.return_value = "te_spec"

        # Call the function
        result = default_layer_spec(provider)

        # Should use regular TE spec by default
        mock_te_full_spec.assert_not_called()
        mock_te_spec.assert_called_once_with(provider)
        assert result == "te_spec"

    def test_mtp_block_spec_returns_none_when_mtp_disabled(self):
        """mtp_block_spec returns None when mtp_num_layers is unset."""
        from megatron.bridge.models.gpt_provider import mtp_block_spec

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )

        assert mtp_block_spec(provider) is None

    def test_mtp_checkpointed_forward_accepts_padding_mask(self):
        """Bridge MCore compatibility patch keeps MTP recompute aligned with MCore forward."""
        import inspect

        from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionLayer

        params = inspect.signature(MultiTokenPredictionLayer._checkpointed_forward).parameters
        assert "padding_mask" in params or any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
        )

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_mtp_block_spec_uses_callable_spec_directly_when_layer_specs_nonempty(self, mock_get_mtp):
        """When the callable spec returns a non-empty block spec, use it as-is."""
        from megatron.bridge.models.gpt_provider import mtp_block_spec

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            mtp_num_layers=1,
        )

        block_spec = Mock()
        block_spec.layer_specs = ["layer_a", "layer_b"]
        provider.transformer_layer_spec = lambda config: block_spec

        mock_get_mtp.return_value = "mtp_spec"

        result = mtp_block_spec(provider, vp_stage=None)

        mock_get_mtp.assert_called_once_with(provider, block_spec, use_transformer_engine=True, vp_stage=None)
        assert result == "mtp_spec"

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_decoder_layer_specs")
    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_mtp_block_spec_re_derives_last_decoder_spec_when_layer_specs_empty(
        self, mock_get_mtp, mock_get_decoder_specs
    ):
        """When the last-stage spec has empty layer_specs (MoE block spec on the last PP stage),
        re-derive all decoder layer specs and pass the last one to get_gpt_mtp_block_spec."""
        from megatron.bridge.models.gpt_provider import mtp_block_spec

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            mtp_num_layers=1,
        )

        empty_block_spec = Mock()
        empty_block_spec.layer_specs = []
        provider.transformer_layer_spec = lambda config: empty_block_spec

        dense_layer_spec = Mock(name="dense_layer_spec")
        moe_layer_spec = Mock(name="moe_layer_spec")
        mock_get_decoder_specs.return_value = [dense_layer_spec, moe_layer_spec]
        mock_get_mtp.return_value = "mtp_spec"

        result = mtp_block_spec(provider, vp_stage=2)

        mock_get_decoder_specs.assert_called_once_with(
            provider,
            use_transformer_engine=True,
            normalization=provider.normalization,
            qk_l2_norm=provider.qk_l2_norm,
        )
        mock_get_mtp.assert_called_once_with(provider, moe_layer_spec, use_transformer_engine=True, vp_stage=2)
        assert result == "mtp_spec"

    @patch("megatron.core.models.gpt.gpt_layer_specs.get_gpt_mtp_block_spec")
    def test_mtp_block_spec_passes_vp_stage_to_callable_spec(self, mock_get_mtp):
        """When the transformer_layer_spec callable accepts vp_stage, it is forwarded."""
        from megatron.bridge.models.gpt_provider import mtp_block_spec

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            mtp_num_layers=1,
        )

        block_spec = Mock()
        block_spec.layer_specs = ["layer_a"]
        received_vp_stage = {}

        def spec_fn(config, vp_stage=None):
            received_vp_stage["vp_stage"] = vp_stage
            return block_spec

        provider.transformer_layer_spec = spec_fn
        mock_get_mtp.return_value = "mtp_spec"

        result = mtp_block_spec(provider, vp_stage=3)

        assert received_vp_stage["vp_stage"] == 3
        mock_get_mtp.assert_called_once_with(provider, block_spec, use_transformer_engine=True, vp_stage=3)
        assert result == "mtp_spec"

    def test_dense_grouped_gemm_defaults_to_false(self):
        """GPTModelProvider.dense_grouped_gemm defaults to False."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )
        assert provider.dense_grouped_gemm is False

    def test_dense_grouped_gemm_can_be_enabled(self):
        """GPTModelProvider.dense_grouped_gemm is a settable bool attribute."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            dense_grouped_gemm=True,
        )
        assert provider.dense_grouped_gemm is True

    def test_transformer_engine_layer_spec_forwards_dense_grouped_gemm_when_supported(self):
        """Forward dense_grouped_gemm to the current Megatron-Core dense MLP kwarg."""
        from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec

        captured: dict = {}

        def fake_spec_supported(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
            use_grouped_gemm_for_dense_mlp=False,
        ):
            captured["num_experts"] = num_experts
            captured["moe_grouped_gemm"] = moe_grouped_gemm
            captured["qk_layernorm"] = qk_layernorm
            captured["fp8"] = fp8
            captured["use_grouped_gemm_for_dense_mlp"] = use_grouped_gemm_for_dense_mlp
            return "te_spec_supported"

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            dense_grouped_gemm=True,
        )

        with patch(
            "megatron.bridge.models.gpt_provider.get_gpt_layer_with_transformer_engine_spec",
            new=fake_spec_supported,
        ):
            result = transformer_engine_layer_spec(provider)

        assert result == "te_spec_supported"
        assert captured["use_grouped_gemm_for_dense_mlp"] is True

    def test_transformer_engine_layer_spec_omits_dense_grouped_gemm_when_unsupported(self):
        """When the upstream spec function does not expose a dense grouped GEMM
        parameter (older Megatron-Core), transformer_engine_layer_spec must not
        pass the kwarg — otherwise the call would raise TypeError at runtime."""
        from megatron.bridge.models.gpt_provider import transformer_engine_layer_spec

        captured: dict = {}

        # Signature intentionally excludes dense grouped GEMM args. If the production
        # code were to forward it, the call below would raise TypeError.
        def fake_spec_unsupported(
            num_experts=None,
            moe_grouped_gemm=False,
            qk_layernorm=False,
            fp8=False,
        ):
            captured["num_experts"] = num_experts
            captured["moe_grouped_gemm"] = moe_grouped_gemm
            captured["qk_layernorm"] = qk_layernorm
            captured["fp8"] = fp8
            return "te_spec_unsupported"

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            dense_grouped_gemm=True,
        )

        with patch(
            "megatron.bridge.models.gpt_provider.get_gpt_layer_with_transformer_engine_spec",
            new=fake_spec_unsupported,
        ):
            result = transformer_engine_layer_spec(provider)

        assert result == "te_spec_unsupported"
        assert "use_grouped_gemm_for_dense_mlp" not in captured
