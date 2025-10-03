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

from unittest.mock import MagicMock, patch

from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer


class TestTokenizerConfig:
    """Test cases for TokenizerConfig dataclass."""

    def test_tokenizer_config_default_hf_kwargs(self):
        """Test that hf_tokenizer_kwargs defaults to empty dict."""
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="bert-base-uncased",
        )
        assert config.hf_tokenizer_kwargs == {}

    def test_tokenizer_config_with_hf_kwargs(self):
        """Test that hf_tokenizer_kwargs can be set."""
        custom_kwargs = {
            "use_fast": True,
            "trust_remote_code": True,
            "chat_template": "custom_template",
        }
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-chat-hf",
            hf_tokenizer_kwargs=custom_kwargs,
        )
        assert config.hf_tokenizer_kwargs == custom_kwargs


class TestBuildTokenizer:
    """Test cases for build_tokenizer function."""

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_with_config_kwargs(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that hf_tokenizer_kwargs from config are passed to HuggingFaceTokenizer."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        custom_kwargs = {
            "use_fast": True,
            "trust_remote_code": False,
            "padding_side": "left",
        }
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("gpt2", **custom_kwargs)
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_kwargs_override(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that passed kwargs override config hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config_kwargs = {
            "use_fast": True,
            "trust_remote_code": False,
        }
        passed_kwargs = {
            "use_fast": False,  # This should override
            "padding_side": "right",  # This should be added
        }
        expected_kwargs = {
            "use_fast": False,  # Overridden
            "trust_remote_code": False,  # From config
            "padding_side": "right",  # From passed kwargs
        }

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs=config_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config, **passed_kwargs)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("gpt2", **expected_kwargs)
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_no_config_kwargs(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that HuggingFaceTokenizer works without hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            # hf_tokenizer_kwargs not set, should default to {}
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("gpt2")
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_hf_tokenizer_with_chat_template(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that chat_template can be passed via hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        custom_kwargs = {
            "chat_template": chat_template,
        }
        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="meta-llama/Llama-2-7b-chat-hf",
            hf_tokenizer_kwargs=custom_kwargs,
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify
        mock_hf_tokenizer_class.assert_called_once_with("meta-llama/Llama-2-7b-chat-hf", chat_template=chat_template)
        assert tokenizer == mock_tokenizer_instance

    @patch("megatron.bridge.training.tokenizers.tokenizer._SentencePieceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_build_non_hf_tokenizer_ignores_hf_kwargs(self, mock_get_rank, mock_sp_tokenizer_class):
        """Test that non-HuggingFace tokenizers don't use hf_tokenizer_kwargs."""
        # Setup
        mock_tokenizer_instance = MagicMock()
        mock_sp_tokenizer_class.return_value = mock_tokenizer_instance

        # Even if hf_tokenizer_kwargs is set, it shouldn't affect SentencePiece tokenizer
        config = TokenizerConfig(
            tokenizer_type="SentencePieceTokenizer",
            tokenizer_model="tokenizer.model",
            hf_tokenizer_kwargs={"use_fast": True},  # Should be ignored
        )

        # Execute
        tokenizer = build_tokenizer(config)

        # Verify - SentencePiece should be called without hf_tokenizer_kwargs
        mock_sp_tokenizer_class.assert_called_once_with("tokenizer.model", vocab_extra_ids=0)
        assert tokenizer == mock_tokenizer_instance


class TestHuggingFaceTokenizerIntegration:
    """Integration tests for HuggingFace tokenizer with mocked transformers."""

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_hf_tokenizer_with_use_fast_integration(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test that use_fast parameter flows through correctly in a realistic scenario."""
        # Setup a realistic mock that behaves like a real HF tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_underlying_tokenizer = MagicMock()
        mock_underlying_tokenizer.__class__.__name__ = "GPT2Tokenizer"  # Not "Fast"
        mock_tokenizer_instance._tokenizer = mock_underlying_tokenizer
        mock_tokenizer_instance.vocab_size = 50257
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            hf_tokenizer_kwargs={"use_fast": False},
        )

        tokenizer = build_tokenizer(config)

        # Verify the kwargs were passed
        mock_hf_tokenizer_class.assert_called_once_with("gpt2", use_fast=False)
        assert tokenizer is not None
        assert hasattr(tokenizer, "_tokenizer")
        # Verify it's not a fast tokenizer
        assert "Fast" not in type(tokenizer._tokenizer).__name__

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_hf_tokenizer_backward_compatibility_integration(self, mock_get_rank, mock_hf_tokenizer_class):
        """Test backward compatibility with mocked tokenizer."""
        # Setup a realistic mock
        mock_tokenizer_instance = MagicMock()
        mock_underlying_tokenizer = MagicMock()
        mock_underlying_tokenizer.__class__.__name__ = "GPT2TokenizerFast"
        mock_tokenizer_instance._tokenizer = mock_underlying_tokenizer
        mock_tokenizer_instance.vocab_size = 50257
        mock_hf_tokenizer_class.return_value = mock_tokenizer_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
            # No hf_tokenizer_kwargs specified
        )

        tokenizer = build_tokenizer(config)

        # Verify no extra kwargs were passed (backward compatible)
        mock_hf_tokenizer_class.assert_called_once_with("gpt2")
        assert tokenizer is not None
        assert hasattr(tokenizer, "vocab_size")
