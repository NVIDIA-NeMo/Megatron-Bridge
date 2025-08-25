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

from unittest.mock import Mock, patch

import torch

from megatron.bridge.models.mamba.mamba_provider import (
    MambaProvider,
    MambaProvider1_3B,
    MambaProvider2_7B,
    MambaProvider130M,
    MambaProvider370M,
    MambaProvider780M,
    NVIDIAMambaHybridProvider8B,
    NVIDIAMambaProvider8B,
)


class TestMambaProvider:
    """Test cases for MambaProvider class."""

    def test_mamba_provider_initialization(self):
        """Test MambaProvider can be initialized with default values."""
        provider = MambaProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=1,
        )

        # Check required transformer config fields
        assert provider.num_layers == 12
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 1

        # Check Mamba-specific defaults
        assert provider.fp16_lm_cross_entropy is False
        assert provider.parallel_output is True
        assert provider.share_embeddings_and_output_weights is False
        assert provider.params_dtype == torch.bfloat16
        assert provider.fp16 is False
        assert provider.bf16 is True
        assert provider.mamba_num_groups == 8
        assert provider.hybrid_attention_ratio == 0.0
        assert provider.hybrid_mlp_ratio == 0.0
        assert provider.hybrid_override_pattern is None
        assert provider.seq_length == 8192
        assert provider.position_embedding_type == "none"
        assert provider.rotary_percent == 1.0
        assert provider.rotary_base == 10000
        assert provider.seq_len_interpolation_factor is None
        assert provider.apply_rope_fusion is True
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.gated_linear_unit is False
        assert provider.normalization == "RMSNorm"
        assert provider.add_bias_linear is False
        assert provider.hidden_dropout == 0.0
        assert provider.attention_dropout == 0.0
        assert provider.layernorm_epsilon == 1e-5
        assert provider.deallocate_pipeline_outputs is True
        assert provider.bias_dropout_fusion is True
        assert provider.cross_entropy_loss_fusion is True
        assert provider.vocab_size is None

    def test_mamba_provider_with_hybrid_configuration(self):
        """Test MambaProvider with hybrid attention/MLP configuration."""
        provider = MambaProvider(
            num_layers=12,
            hidden_size=768,
            num_attention_heads=8,
            hybrid_attention_ratio=0.25,
            hybrid_mlp_ratio=0.1,
            hybrid_override_pattern="M-M-M*-M-M-M-M*-M-M-M-M-",
        )

        assert provider.hybrid_attention_ratio == 0.25
        assert provider.hybrid_mlp_ratio == 0.1
        assert provider.hybrid_override_pattern == "M-M-M*-M-M-M-M*-M-M-M-M-"

    def test_provide_method_basic(self):
        """Test the provide method creates a Mamba model."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        # Mock dependencies
        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.mamba.mamba_provider.calculate_padded_vocab_size", return_value=1024):
                with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    result = provider.provide()

                    assert result == mock_instance
                    mock_model.assert_called_once()

    def test_provide_method_with_vocab_padding(self):
        """Test provide method calculates padded vocab size when padding is enabled."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=True,  # Enable padding
        )

        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state") as mock_ps:
            with patch(
                "megatron.bridge.models.mamba.mamba_provider.calculate_padded_vocab_size", return_value=50176
            ) as mock_calc_vocab:
                with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    _ = provider.provide()

                    # Verify calculate_padded_vocab_size was called with correct parameters
                    mock_calc_vocab.assert_called_once_with(50000, 128, 8)
                    # Verify model was created with padded vocab size
                    call_kwargs = mock_model.call_args.kwargs
                    assert call_kwargs["vocab_size"] == 50176

    def test_provide_method_no_vocab_padding(self):
        """Test provide method uses original vocab size when padding is disabled."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            vocab_size=50000,
            tensor_model_parallel_size=8,
            make_vocab_size_divisible_by=128,
            should_pad_vocab=False,  # Disable padding
        )

        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.mamba.mamba_provider.calculate_padded_vocab_size") as mock_calc_vocab:
                with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    _ = provider.provide()

                    # Verify calculate_padded_vocab_size was NOT called
                    mock_calc_vocab.assert_not_called()
                    # Verify model was created with original vocab size
                    call_kwargs = mock_model.call_args.kwargs
                    assert call_kwargs["vocab_size"] == 50000

    def test_provide_method_pipeline_stages(self):
        """Test provide method respects pipeline stage arguments."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.mamba.mamba_provider.calculate_padded_vocab_size", return_value=1024):
                with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel") as mock_mamba:
                    # Test default behavior - uses parallel_state
                    mock_ps.is_pipeline_first_stage.return_value = False
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_mamba.return_value = mock_instance

                    provider.provide()

                    # Check the model was called with pipeline stages from parallel_state
                    call_kwargs = mock_mamba.call_args.kwargs
                    assert call_kwargs["pre_process"] is False
                    assert call_kwargs["post_process"] is True

    def test_provide_method_with_preset_vocab_size(self):
        """Test provide method with preset vocab_size calculates padding correctly."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=2000,
            should_pad_vocab=True,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
        )

        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state") as mock_ps:
            with patch(
                "megatron.bridge.models.mamba.mamba_provider.calculate_padded_vocab_size", return_value=2048
            ) as mock_calc:
                with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel") as mock_mamba:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_mamba.return_value = mock_instance

                    provider.provide()

                    mock_calc.assert_called_once_with(2000, 128, 1)
                    call_kwargs = mock_mamba.call_args.kwargs
                    assert call_kwargs["vocab_size"] == 2048

    def test_provide_method_virtual_pipeline_error(self):
        """Test provide method raises error for virtual pipeline."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
        )
        provider.virtual_pipeline_model_parallel_size = 2  # Set virtual pipeline

        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state"):
            with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel"):
                # Should raise AssertionError for virtual pipeline
                try:
                    provider.provide(vp_stage=0)
                    assert False, "Expected AssertionError for virtual pipeline"
                except AssertionError as e:
                    assert "Virtual pipeline model parallelism is temporarily unsupported" in str(e)

    def test_mamba_stack_spec_callable(self):
        """Test that mamba_stack_spec can be a callable."""

        def custom_stack_spec():
            spec = Mock()
            spec.info = "custom spec"
            return spec

        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            vocab_size=1000,
            tensor_model_parallel_size=1,
            make_vocab_size_divisible_by=128,
            mamba_stack_spec=custom_stack_spec,
        )

        with patch("megatron.bridge.models.mamba.mamba_provider.parallel_state"):
            with patch("megatron.bridge.models.mamba.mamba_provider.calculate_padded_vocab_size", return_value=1024):
                with patch("megatron.bridge.models.mamba.mamba_provider.MCoreMambaModel") as mock_mamba:
                    mock_instance = Mock()
                    mock_mamba.return_value = mock_instance

                    provider.provide()

                    # The custom_stack_spec should have been called
                    assert provider.mamba_stack_spec == custom_stack_spec
                    spec_call_kwarg = mock_mamba.call_args.kwargs["mamba_stack_spec"]
                    assert isinstance(spec_call_kwarg, Mock)
                    assert spec_call_kwarg.info == "custom spec"

    def test_minimal_configuration(self):
        """Test that minimal configuration works."""
        # MambaProvider should work with minimal required fields
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
        )
        assert provider.num_layers == 2
        assert provider.hidden_size == 128
        assert provider.num_attention_heads == 1

    def test_mamba_specific_configuration(self):
        """Test Mamba-specific configuration parameters."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            mamba_num_groups=16,
            gated_linear_unit=True,
            normalization="LayerNorm",
            add_bias_linear=True,
        )

        assert provider.mamba_num_groups == 16
        assert provider.gated_linear_unit is True
        assert provider.normalization == "LayerNorm"
        assert provider.add_bias_linear is True

    def test_dropout_configuration(self):
        """Test dropout configuration."""
        provider = MambaProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=1,
            hidden_dropout=0.1,
            attention_dropout=0.2,
            layernorm_epsilon=1e-6,
        )

        assert provider.hidden_dropout == 0.1
        assert provider.attention_dropout == 0.2
        assert provider.layernorm_epsilon == 1e-6


class TestMambaProvider130M:
    """Test cases for MambaProvider130M class."""

    def test_mamba_130m_default_configuration(self):
        """Test Mamba 130M model has correct default configuration."""
        provider = MambaProvider130M()

        # Check Mamba 130M specific configuration
        assert provider.num_layers == 24
        assert provider.hidden_size == 768
        assert provider.num_attention_heads == 1
        assert provider.mamba_num_groups == 1
        assert provider.ffn_hidden_size == 768
        assert provider.seq_length == 2048
        assert provider.make_vocab_size_divisible_by == 16
        assert provider.hybrid_override_pattern == "M" * 24

    def test_mamba_130m_override_configuration(self):
        """Test Mamba 130M model with overridden configuration."""
        provider = MambaProvider130M(
            seq_length=4096,
            hidden_dropout=0.1,
        )

        # Check overridden values
        assert provider.seq_length == 4096
        assert provider.hidden_dropout == 0.1

        # Check defaults remain
        assert provider.num_layers == 24
        assert provider.hidden_size == 768
        assert provider.mamba_num_groups == 1


class TestMambaProvider370M:
    """Test cases for MambaProvider370M class."""

    def test_mamba_370m_default_configuration(self):
        """Test Mamba 370M model has correct default configuration."""
        provider = MambaProvider370M()

        # Check Mamba 370M specific configuration
        assert provider.num_layers == 48
        assert provider.hidden_size == 1024
        assert provider.num_attention_heads == 1
        assert provider.mamba_num_groups == 1
        assert provider.ffn_hidden_size == 1024
        assert provider.seq_length == 2048
        assert provider.make_vocab_size_divisible_by == 16
        assert provider.hybrid_override_pattern == "M" * 48


class TestMambaProvider780M:
    """Test cases for MambaProvider780M class."""

    def test_mamba_780m_default_configuration(self):
        """Test Mamba 780M model has correct default configuration."""
        provider = MambaProvider780M()

        # Check Mamba 780M specific configuration
        assert provider.num_layers == 48
        assert provider.hidden_size == 1536
        assert provider.num_attention_heads == 1
        assert provider.mamba_num_groups == 1
        assert provider.ffn_hidden_size == 1536
        assert provider.seq_length == 2048
        assert provider.make_vocab_size_divisible_by == 16
        assert provider.hybrid_override_pattern == "M" * 48


class TestMambaProvider1_3B:
    """Test cases for MambaProvider1_3B class."""

    def test_mamba_1_3b_default_configuration(self):
        """Test Mamba 1.3B model has correct default configuration."""
        provider = MambaProvider1_3B()

        # Check Mamba 1.3B specific configuration
        assert provider.num_layers == 48
        assert provider.hidden_size == 2048
        assert provider.num_attention_heads == 1
        assert provider.mamba_num_groups == 1
        assert provider.ffn_hidden_size == 2048
        assert provider.seq_length == 2048
        assert provider.make_vocab_size_divisible_by == 16
        assert provider.hybrid_override_pattern == "M" * 48


class TestMambaProvider2_7B:
    """Test cases for MambaProvider2_7B class."""

    def test_mamba_2_7b_default_configuration(self):
        """Test Mamba 2.7B model has correct default configuration."""
        provider = MambaProvider2_7B()

        # Check Mamba 2.7B specific configuration
        assert provider.num_layers == 64
        assert provider.hidden_size == 2560
        assert provider.num_attention_heads == 1
        assert provider.mamba_num_groups == 1
        assert provider.ffn_hidden_size == 2560
        assert provider.seq_length == 2048
        assert provider.make_vocab_size_divisible_by == 16
        assert provider.hybrid_override_pattern == "M" * 64


class TestNVIDIAMambaProvider8B:
    """Test cases for NVIDIAMambaProvider8B class."""

    def test_nvidia_mamba_8b_default_configuration(self):
        """Test NVIDIA Mamba 8B model has correct default configuration."""
        provider = NVIDIAMambaProvider8B()

        # Check NVIDIA Mamba 8B specific configuration
        assert provider.num_layers == 56
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.mamba_num_groups == 8
        assert provider.ffn_hidden_size == 4096
        assert provider.seq_length == 4096
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.hybrid_override_pattern == "M" * 56


class TestNVIDIAMambaHybridProvider8B:
    """Test cases for NVIDIAMambaHybridProvider8B class."""

    def test_nvidia_mamba_hybrid_8b_default_configuration(self):
        """Test NVIDIA Mamba Hybrid 8B model has correct default configuration."""
        provider = NVIDIAMambaHybridProvider8B()

        # Check NVIDIA Mamba Hybrid 8B specific configuration
        assert provider.num_layers == 56
        assert provider.hidden_size == 4096
        assert provider.num_attention_heads == 32
        assert provider.num_query_groups == 8
        assert provider.mamba_num_groups == 8
        assert provider.ffn_hidden_size == 16384
        assert provider.seq_length == 4096
        assert provider.make_vocab_size_divisible_by == 128
        assert provider.hybrid_override_pattern == "M-M-M--M-M*-M-M-M-M--M*-M-M-M-M-M*--M-M-M-M-M*-M--M-M-M-"

    def test_nvidia_mamba_hybrid_8b_hybrid_pattern(self):
        """Test NVIDIA Mamba Hybrid 8B hybrid pattern configuration."""
        provider = NVIDIAMambaHybridProvider8B()

        # Check that the hybrid pattern contains both Mamba and Attention layers
        pattern = provider.hybrid_override_pattern
        assert "M" in pattern  # Mamba layers
        assert "*" in pattern  # Attention layers
        assert len(pattern) > 0


class TestMambaProviderInheritance:
    """Test inheritance relationships between Mamba providers."""

    def test_all_providers_inherit_from_base(self):
        """Test all Mamba providers inherit from MambaProvider."""
        providers = [
            MambaProvider130M,
            MambaProvider370M,
            MambaProvider780M,
            MambaProvider1_3B,
            MambaProvider2_7B,
            NVIDIAMambaProvider8B,
            NVIDIAMambaHybridProvider8B,
        ]

        for provider_class in providers:
            assert issubclass(provider_class, MambaProvider)

    def test_provide_method_inherited(self):
        """Test that provide method works correctly in inherited classes."""
        # Test with Mamba 130M
        provider = MambaProvider130M()

        # The provide method should be inherited from MambaProvider
        assert hasattr(provider, "provide")
        assert callable(provider.provide)

    def test_hybrid_patterns_consistency(self):
        """Test that hybrid patterns are consistent across providers."""
        # Pure Mamba models should have only "M" in their pattern
        pure_mamba_providers = [
            MambaProvider130M(),
            MambaProvider370M(),
            MambaProvider780M(),
            MambaProvider1_3B(),
            MambaProvider2_7B(),
            NVIDIAMambaProvider8B(),
        ]

        for provider in pure_mamba_providers:
            pattern = provider.hybrid_override_pattern
            assert "M" in pattern  # Mamba layers
            assert "*" not in pattern  # No attention layers

        # Hybrid models should have both "M" and "*" in their pattern
        hybrid_provider = NVIDIAMambaHybridProvider8B()
        pattern = hybrid_provider.hybrid_override_pattern
        assert "M" in pattern  # Mamba layers
        assert "*" in pattern  # Attention layers
