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

from megatron.bridge.models import GPTModelProvider


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
        )

        # Mock dependencies
        with patch("megatron.bridge.models.gpt_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.gpt_provider.get_vocab_size", return_value=1000):
                with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_model:
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True
                    mock_instance = Mock()
                    mock_model.return_value = mock_instance

                    # Mock tokenizer
                    mock_tokenizer = Mock()
                    mock_tokenizer.vocab_size = 1000

                    result = provider.provide(tokenizer=mock_tokenizer)

                    assert result == mock_instance
                    mock_model.assert_called_once()

    def test_provide_method_with_tokenizer(self):
        """Test provide method with tokenizer provided."""
        mock_tokenizer = Mock()
        mock_tokenizer.vocab_size = 50000

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
        )

        with patch("megatron.bridge.models.gpt_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.gpt_provider.get_vocab_size", return_value=50000) as mock_get_vocab:
                with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel"):
                    mock_ps.is_pipeline_first_stage.return_value = True
                    mock_ps.is_pipeline_last_stage.return_value = True

                    provider.provide(tokenizer=mock_tokenizer)

                    # Verify get_vocab_size was called with tokenizer vocab size
                    mock_get_vocab.assert_called_once_with(provider, 50000, 128)

    def test_provide_method_pipeline_stages(self):
        """Test provide method respects pipeline stage arguments."""
        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            vocab_size=1000,
        )

        with patch("megatron.bridge.models.gpt_provider.parallel_state") as mock_ps:
            with patch("megatron.bridge.models.gpt_provider.get_vocab_size", return_value=1000):
                with patch("megatron.bridge.models.gpt_provider.MCoreGPTModel") as mock_gpt:
                    # Test default behavior - uses parallel_state
                    mock_ps.is_pipeline_first_stage.return_value = False
                    mock_ps.is_pipeline_last_stage.return_value = True

                    mock_tokenizer = Mock(vocab_size=1000)
                    provider.provide(tokenizer=mock_tokenizer)

                    # Check the model was called with pipeline stages from parallel_state
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

    @patch("megatron.bridge.models.gpt_provider.parallel_state")
    def test_provide_with_generation_config(self, mock_parallel_state):
        """Test provide method with generation configuration."""
        mock_parallel_state.is_pipeline_first_stage.return_value = True
        mock_parallel_state.is_pipeline_last_stage.return_value = True

        generation_config = {"max_length": 100, "temperature": 0.7}

        provider = GPTModelProvider(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            vocab_size=1000,
            generation_config=generation_config,
        )

        assert provider.generation_config == generation_config
