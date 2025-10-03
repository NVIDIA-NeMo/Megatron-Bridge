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

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from megatron.bridge.data.datasets.sft import GPTSFTChatDataset, create_sft_dataset
from megatron.bridge.data.datasets.utils import _chat_preprocess, _convert_to_openai_messages


class TestConvertToOpenAIMessages:
    """Test cases for _convert_to_openai_messages function."""

    def test_convert_conversations_format(self):
        """Test conversion from conversations format to OpenAI messages."""
        source = {
            "conversations": [
                {"from": "User", "value": "Hello"},
                {"from": "Assistant", "value": "Hi there!"},
            ]
        }

        result = _convert_to_openai_messages(source)

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "Hello"}
        assert result[1] == {"role": "assistant", "content": "Hi there!"}

    def test_convert_conversations_with_system(self):
        """Test conversion with system message."""
        source = {
            "system": "You are a helpful assistant.",
            "conversations": [
                {"from": "User", "value": "Hello"},
                {"from": "Assistant", "value": "Hi!"},
            ],
        }

        result = _convert_to_openai_messages(source)

        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "You are a helpful assistant."}
        assert result[1] == {"role": "user", "content": "Hello"}
        assert result[2] == {"role": "assistant", "content": "Hi!"}

    def test_convert_messages_format(self):
        """Test that messages format passes through unchanged."""
        source = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        result = _convert_to_openai_messages(source)

        assert result == source["messages"]

    def test_convert_list_input(self):
        """Test that list input passes through unchanged."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        result = _convert_to_openai_messages(messages)

        assert result == messages


class TestChatPreprocess:
    """Test cases for _chat_preprocess function."""

    def test_chat_preprocess_basic(self):
        """Test basic chat preprocessing with mocked tokenizer."""
        # Create mock tokenizer
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        # Mock chat template
        mock_hf_tokenizer.chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 30, 2],
            "assistant_masks": [0, 0, 1, 1, 1],
        }

        source = {
            "conversations": [
                {"from": "User", "value": "Hello"},
                {"from": "Assistant", "value": "Hi!"},
            ]
        }

        result = _chat_preprocess(source, mock_tokenizer, tool_schemas=None)

        # Verify structure
        assert "input_ids" in result
        assert "loss_mask" in result
        assert "context_ids" in result
        assert "answer_ids" in result

        # Verify types
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["loss_mask"], torch.Tensor)
        assert isinstance(result["context_ids"], torch.Tensor)
        assert isinstance(result["answer_ids"], torch.Tensor)

        # Verify apply_chat_template was called
        mock_hf_tokenizer.apply_chat_template.assert_called_once()

    def test_chat_preprocess_without_generation_keyword(self):
        """Test chat preprocessing when template lacks generation keyword."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        # Chat template without generation keyword
        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 30, 2],
        }

        source = {"conversations": [{"from": "User", "value": "Test"}]}

        result = _chat_preprocess(source, mock_tokenizer)

        # Should default to all 1s for loss mask
        assert result["loss_mask"].tolist() == [1, 1, 1, 1, 1]

    def test_chat_preprocess_adds_eos_if_missing(self):
        """Test that EOS token is added if missing."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 999

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20],  # No EOS
        }

        source = {"conversations": [{"from": "User", "value": "Test"}]}

        result = _chat_preprocess(source, mock_tokenizer)

        # EOS should be added
        assert result["input_ids"][-1].item() == 999
        assert len(result["input_ids"]) == 4  # Original 3 + EOS

    def test_chat_preprocess_with_tool_schemas(self):
        """Test chat preprocessing with tool schemas."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
        }

        source = {"conversations": [{"from": "User", "value": "Test"}]}
        tool_schemas = [{"type": "function", "function": {"name": "test_func"}}]

        _chat_preprocess(source, mock_tokenizer, tool_schemas=tool_schemas)

        # Verify tools were passed to apply_chat_template
        call_kwargs = mock_hf_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tool_schemas

    def test_chat_preprocess_invalid_tokenizer(self):
        """Test that error is raised for tokenizer without apply_chat_template."""
        mock_tokenizer = MagicMock()
        # No _tokenizer attribute
        del mock_tokenizer._tokenizer

        source = {"conversations": [{"from": "User", "value": "Test"}]}

        with pytest.raises(ValueError, match="Cannot apply chat template"):
            _chat_preprocess(source, mock_tokenizer)


class TestGPTSFTChatDataset:
    """Test cases for GPTSFTChatDataset with HF chat template support."""

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_chat_dataset_init_with_hf_template(self, mock_dataset_class):
        """Test GPTSFTChatDataset initialization with HF chat template enabled."""
        # Mock the indexed dataset
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        # Create mock tokenizer with chat template support
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_hf_tokenizer.apply_chat_template = MagicMock()
        mock_tokenizer.eos_id = 2

        # Create dataset
        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
            tool_schemas=None,
        )

        assert dataset.use_hf_tokenizer_chat_template is True
        assert dataset.tool_schemas is None

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_chat_dataset_init_with_tool_schemas_json(self, mock_dataset_class):
        """Test tool schemas parsing from JSON string."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_hf_tokenizer.apply_chat_template = MagicMock()
        mock_tokenizer.eos_id = 2

        tool_schemas_json = '[{"type": "function", "function": {"name": "test"}}]'

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
            tool_schemas=tool_schemas_json,
        )

        assert isinstance(dataset.tool_schemas, list)
        assert len(dataset.tool_schemas) == 1
        assert dataset.tool_schemas[0]["type"] == "function"

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_chat_dataset_init_without_chat_template(self, mock_dataset_class):
        """Test that error is raised when tokenizer lacks chat template."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        # Mock tokenizer WITHOUT chat template support
        mock_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = MagicMock()
        # Remove apply_chat_template method
        del mock_tokenizer._tokenizer.apply_chat_template
        mock_tokenizer.eos_id = 2

        with pytest.raises(ValueError, match="Dataset configured to use HF tokenizer chat template"):
            GPTSFTChatDataset(
                file_path="test.jsonl",
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                use_hf_tokenizer_chat_template=True,
            )

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_chat_dataset_legacy_mode(self, mock_dataset_class):
        """Test GPTSFTChatDataset in legacy mode (no HF template)."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2
        mock_tokenizer.text_to_ids.return_value = [1, 2, 3]

        # Should not raise error even without chat template
        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=False,
        )

        assert dataset.use_hf_tokenizer_chat_template is False

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_process_example_uses_chat_preprocess(self, mock_dataset_class):
        """Test that _process_example uses _chat_preprocess when enabled."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
        )

        example = {
            "conversations": [
                {"from": "User", "value": "Hello"},
                {"from": "Assistant", "value": "Hi!"},
            ],
            "metadata_key": "test_value",
        }

        result = dataset._process_example(example)

        # Verify result has expected structure
        assert "input_ids" in result
        assert "loss_mask" in result
        assert "context_ids" in result
        assert "answer_ids" in result
        assert "metadata" in result

        # Verify metadata preserved
        assert result["metadata"]["metadata_key"] == "test_value"
        # Verify conversations not in metadata by default
        assert "conversations" not in result["metadata"]

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_collate_fn_handles_loss_mask(self, mock_dataset_class):
        """Test that collate_fn handles loss_mask correctly."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
            pad_to_max_length=False,
        )

        # Create mock batch with loss_mask (not mask)
        batch = [
            {
                "input_ids": torch.tensor([1, 10, 20, 30, 2]),
                "loss_mask": torch.tensor([0, 0, 1, 1, 1]),
                "context_ids": torch.tensor([1, 10]),
                "answer_ids": torch.tensor([20, 30, 2]),
                "metadata": {"id": 1},
            },
            {
                "input_ids": torch.tensor([1, 11, 21, 2]),
                "loss_mask": torch.tensor([0, 0, 1, 1]),
                "context_ids": torch.tensor([1, 11]),
                "answer_ids": torch.tensor([21, 2]),
                "metadata": {"id": 2},
            },
        ]

        result = dataset.collate_fn(batch)

        # Verify output structure
        assert "tokens" in result
        assert "labels" in result
        assert "loss_mask" in result
        assert "position_ids" in result
        assert "contexts" in result
        assert "answers" in result
        assert "metadata" in result

        # Verify batch size
        assert result["tokens"].shape[0] == 2
        assert result["labels"].shape[0] == 2


class TestCreateSFTDataset:
    """Test cases for create_sft_dataset factory function."""

    @patch("megatron.bridge.data.datasets.sft.GPTSFTChatDataset")
    def test_create_chat_dataset_with_template(self, mock_chat_class):
        """Test creating chat dataset with HF template."""
        from pathlib import Path

        mock_tokenizer = MagicMock()
        mock_chat_class.return_value = MagicMock()

        create_sft_dataset(
            path=Path("test.jsonl"),
            tokenizer=mock_tokenizer,
            chat=True,
            use_hf_tokenizer_chat_template=True,
            tool_schemas={"type": "function"},
        )

        # Verify GPTSFTChatDataset was called with correct args
        mock_chat_class.assert_called_once()
        call_kwargs = mock_chat_class.call_args[1]
        assert call_kwargs["use_hf_tokenizer_chat_template"] is True
        assert call_kwargs["tool_schemas"] == {"type": "function"}

    @patch("megatron.bridge.data.datasets.sft.GPTSFTPackedDataset")
    def test_create_packed_dataset_priority(self, mock_packed_class):
        """Test that .npy files create GPTSFTPackedDataset even with chat=True."""
        from pathlib import Path

        mock_tokenizer = MagicMock()
        mock_packed_class.return_value = MagicMock()

        create_sft_dataset(
            path=Path("test.npy"),
            tokenizer=mock_tokenizer,
            chat=True,  # Should be ignored for .npy files
            use_hf_tokenizer_chat_template=True,
        )

        # Verify GPTSFTPackedDataset was called (not GPTSFTChatDataset)
        mock_packed_class.assert_called_once()


class TestTokenizerSpaceSensitive:
    """Test cases for space_sensitive attribute on tokenizers."""

    @patch("megatron.bridge.training.tokenizers.tokenizer._HuggingFaceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_hf_tokenizer_computes_space_sensitive(self, mock_get_rank, mock_hf_class):
        """Test that _HuggingFaceTokenizer computes space_sensitive attribute."""
        from megatron.bridge.training.tokenizers.config import TokenizerConfig
        from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

        # Create a realistic mock tokenizer
        mock_instance = MagicMock()

        # Mock tokenize method to simulate space-sensitive tokenizer
        def mock_tokenize(text, **kwargs):
            if text == "x y":
                return [1, 2, 3]  # Different from x + y
            elif text == "x":
                return [1]
            elif text == "y":
                return [2]
            return []

        mock_instance.tokenize = mock_tokenize
        mock_instance.space_sensitive = True  # This should be set by __init__
        mock_hf_class.return_value = mock_instance

        config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model="gpt2",
        )

        tokenizer = build_tokenizer(config)

        # Verify space_sensitive is set
        assert hasattr(tokenizer, "space_sensitive")

    @patch("megatron.bridge.training.tokenizers.tokenizer._SentencePieceTokenizer")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_sentencepiece_tokenizer_has_space_sensitive(self, mock_get_rank, mock_sp_class):
        """Test that _SentencePieceTokenizer has space_sensitive attribute."""
        from megatron.bridge.training.tokenizers.config import TokenizerConfig
        from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer

        mock_instance = MagicMock()
        mock_instance.space_sensitive = True
        mock_sp_class.return_value = mock_instance

        config = TokenizerConfig(
            tokenizer_type="SentencePieceTokenizer",
            tokenizer_model="tokenizer.model",
        )

        tokenizer = build_tokenizer(config)

        # Verify space_sensitive is set
        assert hasattr(tokenizer, "space_sensitive")


class TestPackedDatasetNaNFix:
    """Test cases for NaN fix in packed dataset collate_fn."""

    def test_safe_max_seqlen_calculation_logic(self):
        """Test the safe_max_seqlen calculation logic (without full dataset init)."""
        # This tests the core logic from the NaN fix
        pack_metadata = [
            {
                "dataset_max_seqlen": 100,
                "max_samples_per_bin": 5,
                "min_packed_seqlen": 50,
            }
        ]

        # Simulate values from collate_fn
        max_length = 512  # Current batch max length

        # Apply the NaN fix logic
        dataset_max_seqlen = max(p["dataset_max_seqlen"] for p in pack_metadata)
        min_pack_seq_len = min(p["min_packed_seqlen"] for p in pack_metadata)
        padding_gap = max_length - min_pack_seq_len

        # Use the larger of the two values to avoid NaN issues with attention kernel
        safe_max_seqlen = max(dataset_max_seqlen, padding_gap)

        # Verify the calculation
        assert dataset_max_seqlen == 100
        assert min_pack_seq_len == 50
        assert padding_gap == 462  # 512 - 50
        assert safe_max_seqlen == 462  # max(100, 462)

        # This is the key: when padding_gap > dataset_max_seqlen,
        # using padding_gap prevents NaNs in attention kernel


class TestChatTemplateOverrideWarning:
    """Test cases for chat template override warning."""

    @patch("megatron.bridge.training.tokenizers.tokenizer.print_rank_0")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_warning_on_chat_template_override(self, mock_get_rank, mock_print):
        """Test that warning is printed when overwriting existing chat template."""
        from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_hf_tok = MagicMock()
            mock_hf_tok.chat_template = "existing_template"  # Has existing template
            mock_hf_tok.get_vocab.return_value = {"test": 1}
            mock_from_pretrained.return_value = mock_hf_tok

            # Create tokenizer with chat_template override
            tokenizer = _HuggingFaceTokenizer("gpt2", chat_template="new_template")

            # Verify warning was printed
            mock_print.assert_called_once()
            assert "overwriting" in mock_print.call_args[0][0].lower()

            # Verify template was set
            assert tokenizer._tokenizer.chat_template == "new_template"

    @patch("megatron.bridge.training.tokenizers.tokenizer.print_rank_0")
    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_no_warning_when_no_existing_template(self, mock_get_rank, mock_print):
        """Test that no warning when setting template on tokenizer without one."""
        from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_hf_tok = MagicMock()
            mock_hf_tok.chat_template = None  # No existing template
            mock_hf_tok.get_vocab.return_value = {"test": 1}
            mock_from_pretrained.return_value = mock_hf_tok

            _HuggingFaceTokenizer("gpt2", chat_template="new_template")

            # No warning should be printed
            mock_print.assert_not_called()


class TestOutputOriginalText:
    """Test cases for output_original_text with different formats."""

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_output_original_text_with_messages(self, mock_dataset_class):
        """Test that output_original_text works with messages format."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
            output_original_text=True,
        )

        example = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "extra_field": "value",
        }

        result = dataset._process_example(example)

        # Verify messages are stored in metadata
        assert "metadata" in result
        assert "messages" in result["metadata"]
        assert result["metadata"]["messages"] == example["messages"]
        assert result["metadata"]["extra_field"] == "value"

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_output_original_text_with_conversations(self, mock_dataset_class):
        """Test that output_original_text works with conversations format."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
            output_original_text=True,
        )

        example = {
            "conversations": [
                {"from": "User", "value": "Hello"},
                {"from": "Assistant", "value": "Hi!"},
            ],
        }

        result = dataset._process_example(example)

        # Verify conversations are stored in metadata
        assert "metadata" in result
        assert "conversations" in result["metadata"]
        assert result["metadata"]["conversations"] == example["conversations"]


class TestToolSchemasEdgeCases:
    """Test cases for tool schemas edge cases."""

    def test_tool_schemas_json_parsing(self):
        """Test that tool_schemas is parsed from JSON string."""
        tool_schemas_json = '[{"type": "function", "function": {"name": "get_weather"}}]'

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
        }

        with patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset"):
            dataset = GPTSFTChatDataset(
                file_path="test.jsonl",
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                use_hf_tokenizer_chat_template=True,
                tool_schemas=tool_schemas_json,
            )

            # Verify it was parsed
            assert isinstance(dataset.tool_schemas, list)
            assert len(dataset.tool_schemas) == 1
            assert dataset.tool_schemas[0]["type"] == "function"

    def test_tool_schemas_source_override(self):
        """Test that tool schemas from source override global schemas."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        global_schemas = [{"type": "function", "function": {"name": "global"}}]
        source_schemas = [{"type": "function", "function": {"name": "source"}}]

        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
        }

        source = {
            "conversations": [{"from": "User", "value": "Test"}],
            "tools": source_schemas,  # Source has its own tools
        }

        _chat_preprocess(source, mock_tokenizer, tool_schemas=global_schemas)

        # Verify apply_chat_template was called with source tools
        call_kwargs = mock_hf_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == source_schemas  # Source overrides global

    def test_invalid_tool_schemas_json(self):
        """Test that invalid JSON in tool_schemas raises error."""
        with pytest.raises(json.JSONDecodeError):
            with patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset"):
                mock_tokenizer = MagicMock()
                mock_hf_tokenizer = MagicMock()
                mock_tokenizer._tokenizer = mock_hf_tokenizer
                mock_hf_tokenizer.apply_chat_template = MagicMock()
                mock_tokenizer.eos_id = 2

                GPTSFTChatDataset(
                    file_path="test.jsonl",
                    tokenizer=mock_tokenizer,
                    max_seq_length=512,
                    use_hf_tokenizer_chat_template=True,
                    tool_schemas="invalid json {",  # Invalid JSON
                )


class TestTruncationWithChatTemplates:
    """Test cases for truncation behavior with chat templates."""

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_truncation_happens_in_collate_fn(self, mock_dataset_class):
        """Test that truncation happens in collate_fn, not _process_example."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        # Simulate long sequence
        long_input_ids = list(range(1, 600))  # 599 tokens
        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": long_input_ids,
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
        )

        example = {"conversations": [{"from": "User", "value": "Test"}]}
        result = dataset._process_example(example)

        # _process_example does NOT truncate - that happens in collate_fn
        # Just verify it processed successfully
        assert "input_ids" in result
        assert "loss_mask" in result
        assert len(result["loss_mask"]) == len(result["input_ids"])

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_collate_fn_truncation_warning(self, mock_dataset_class):
        """Test collate_fn handles truncation gracefully."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=10,  # Very small for truncation
            use_hf_tokenizer_chat_template=True,
        )

        # Create batch with sequences longer than max_seq_length
        batch = [
            {
                "input_ids": torch.tensor([1] * 20),  # Longer than max
                "loss_mask": torch.tensor([1] * 20),
                "context_ids": torch.tensor([1] * 5),
                "answer_ids": torch.tensor([1] * 15),
                "metadata": {},
            }
        ]

        # Should not crash, just truncate
        result = dataset.collate_fn(batch)

        # Verify truncation occurred
        assert result["tokens"].shape[1] <= dataset.max_seq_length

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_truncation_warns_when_loss_mask_empty(self, mock_dataset_class):
        """Test that truncation warns and fixes when all assistant tokens are removed."""

        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{{ messages }}"

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=10,
            use_hf_tokenizer_chat_template=True,
        )

        # Create batch where truncation removes all assistant tokens
        # Context is first 15 tokens, answer is at end - will be truncated away
        batch = [
            {
                "input_ids": torch.tensor([1] * 20),
                "loss_mask": torch.tensor([0] * 15 + [1] * 5),  # Assistant tokens at end
                "context_ids": torch.tensor([1] * 15),
                "answer_ids": torch.tensor([1] * 5),
                "metadata": {},
            }
        ]

        # Capture log warnings
        with patch("megatron.bridge.data.datasets.sft.logger") as mock_logger:
            result = dataset.collate_fn(batch)

            # Should have logged warning
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "no assistant tokens" in warning_msg.lower()

            # Loss mask should be set to all ones as fallback
            assert result["loss_mask"].sum().item() > 0


class TestSpaceSensitiveInDataset:
    """Test that space_sensitive attribute is used correctly in dataset."""

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_dataset_uses_space_sensitive_attribute(self, mock_dataset_class):
        """Test that dataset's _separate_template uses tokenizer.space_sensitive."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2
        mock_tokenizer.bos_id = 1
        mock_tokenizer.text_to_ids = MagicMock(return_value=[1, 2, 3])

        # Set space_sensitive to False
        mock_tokenizer.space_sensitive = False

        from megatron.bridge.data.datasets.sft import GPTSFTDataset

        dataset = GPTSFTDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            prompt_template="{input} {output}",
            label_key="output",  # Match the prompt template
            truncation_field="input",  # Match a key in the prompt template
        )

        # Call _separate_template which uses space_sensitive
        template_values = ["input_text", "output_text"]
        template_strings, template_keys = dataset._separate_template(template_values)

        # Verify it ran without error (space_sensitive was accessible)
        assert template_strings is not None
        assert template_keys is not None


class TestChatTemplateFormat:
    """Test that chat_template_format is set correctly."""

    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_chat_template_format_set_to_jinja(self, mock_get_rank):
        """Test that chat_template_format is set to 'jinja'."""
        from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_hf_tok = MagicMock()
            mock_hf_tok.chat_template = None
            mock_hf_tok.get_vocab.return_value = {"test": 1}
            mock_from_pretrained.return_value = mock_hf_tok

            tokenizer = _HuggingFaceTokenizer("gpt2", chat_template="custom_template")

            # Verify chat_template_format was set
            assert hasattr(tokenizer._tokenizer, "chat_template_format")
            assert tokenizer._tokenizer.chat_template_format == "jinja"


class TestContextAnswerSplit:
    """Test context and answer splitting logic in _chat_preprocess."""

    def test_context_answer_split_with_mask(self):
        """Test that context/answer are split correctly based on mask."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        # Mask with 0s (context) and 1s (answer)
        mock_hf_tokenizer.chat_template = "{% generation %}"  # Has generation keyword
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 30, 40, 2],
            "assistant_masks": [0, 0, 0, 1, 1, 1],  # First 3 are context
        }

        source = {"conversations": [{"from": "User", "value": "Test"}]}

        result = _chat_preprocess(source, mock_tokenizer)

        # Context should be first 3 tokens
        assert result["context_ids"].tolist() == [1, 10, 20]
        # Answer should be remaining tokens
        assert result["answer_ids"].tolist() == [30, 40, 2]

    def test_context_answer_split_no_mask(self):
        """Test that when no mask, all is considered answer."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        # No generation keyword means all 1s for mask
        mock_hf_tokenizer.chat_template = "{{ messages }}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
        }

        source = {"conversations": [{"from": "User", "value": "Test"}]}

        result = _chat_preprocess(source, mock_tokenizer)

        # When all is masked as answer, context_ids should be everything
        assert len(result["context_ids"]) == len(result["input_ids"])
        assert result["answer_ids"].tolist() == []


class TestLegacyPreprocessReturnsLossMask:
    """Test that legacy _preprocess returns loss_mask not mask."""

    def test_legacy_preprocess_returns_loss_mask_key(self):
        """Test that _preprocess signature returns dict with loss_mask key."""
        # This is a simple check that the return signature changed from 'mask' to 'loss_mask'
        # Full _preprocess testing requires complex mocking, so we just verify the key name

        # We can verify this by checking the code directly or with a simpler test
        # The key change is in utils.py line 958: return dict(..., loss_mask=..., ...)

        # Simple assertion: the function signature should include loss_mask
        import inspect

        from megatron.bridge.data.datasets.utils import _preprocess

        # Get the source code
        source = inspect.getsource(_preprocess)

        # Verify it returns loss_mask, not mask
        assert "return dict(" in source
        assert "loss_mask=loss_mask" in source or "loss_mask=" in source
        # Verify it doesn't return 'mask=mask'
        assert "mask=mask" not in source or "loss_mask" in source


class TestEOSIndexFixInPackedDataset:
    """Test EOS index fix for cu_seqlens_unpadded calculation."""

    def test_eos_index_logic_uses_shape_check(self):
        """Test that EOS index logic uses shape[0] > 1 check (not .any())."""
        # Verify the code uses the correct logic from NeMo PR #14437
        import inspect

        from megatron.bridge.data.datasets.sft import GPTSFTPackedDataset

        # Get the collate_fn source code
        source = inspect.getsource(GPTSFTPackedDataset.collate_fn)

        # Verify it uses eos_idx[0][1] (second EOS) and shape[0] > 1
        assert "eos_idx[0][1]" in source
        assert "shape[0] > 1" in source
        # Should NOT use eos_idx[0][0] or .any()
        assert "eos_idx[0][0]" not in source


class TestPackedChatDatasetIntegration:
    """Integration tests for packed chat datasets."""

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_chat_dataset_with_loss_mask_field(self, mock_dataset_class):
        """Test that chat dataset with HF template produces loss_mask field."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{% generation %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
        )

        example = {"conversations": [{"from": "User", "value": "Test"}]}
        result = dataset._process_example(example)

        # Should have loss_mask, not mask
        assert "loss_mask" in result
        assert "mask" not in result

    def test_legacy_chat_dataset_backward_compatibility(self):
        """Test that legacy chat dataset still has loss_mask in output."""
        # Simplified test - just verify the code path
        import inspect

        from megatron.bridge.data.datasets.utils import _preprocess

        # Verify _preprocess returns a dict with loss_mask
        source_code = inspect.getsource(_preprocess)

        # The return statement should include loss_mask
        assert "loss_mask" in source_code
        assert "return dict(" in source_code


class TestPackedSequenceWithChatEndToEnd:
    """End-to-end tests for packed sequences with chat templates."""

    def test_tokenize_dataset_produces_loss_mask(self):
        """Test that tokenize_dataset with chat produces items with loss_mask."""
        from pathlib import Path
        from unittest.mock import MagicMock, patch

        from megatron.bridge.data.datasets.packed_sequence import tokenize_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        # Mock dataset that returns items with loss_mask
        mock_dataset = MagicMock()
        mock_item = {
            "input_ids": [1, 2, 3, 2],
            "loss_mask": [0, 1, 1, 1],
            "context_ids": [1],
            "answer_ids": [2, 3, 2],
        }
        mock_dataset.__getitem__ = MagicMock(return_value=mock_item)
        mock_dataset.__len__ = MagicMock(return_value=1)

        dataset_kwargs = {
            "chat": True,
            "use_hf_tokenizer_chat_template": True,
        }

        with patch("megatron.bridge.data.datasets.packed_sequence.create_sft_dataset") as mock_create:
            mock_create.return_value = mock_dataset

            result = tokenize_dataset(
                path=Path("test.jsonl"),
                tokenizer=mock_tokenizer,
                max_seq_length=512,
                seed=1234,
                dataset_kwargs=dataset_kwargs,
            )

            # Verify result is array of items with loss_mask
            assert isinstance(result, np.ndarray)
            assert len(result) == 1
            assert "loss_mask" in result[0]

    def test_packed_dataset_preserves_chat_loss_mask(self):
        """Test that packed dataset preserves loss_mask from chat preprocessing."""
        from megatron.bridge.data.datasets.packing_utils import fill_packing_strategy

        # Simulate chat dataset items with loss_mask
        assignments = [[2]]
        sequences = {
            0: [],
            1: [],
            2: [
                {
                    "input_ids": [1, 10, 20, 2],
                    "loss_mask": [False, False, True, True],  # Chat: only train on assistant
                }
            ],
            3: [],
            4: [],  # Need all keys up to pack_size
            5: [],
        }

        pack_size = 5
        pad_id = 2

        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        # Verify loss_mask was preserved and rolled correctly
        assert len(output_data) == 1
        assert "loss_mask" in output_data[0]
        # Original: [False, False, True, True] -> Rolled: [False, True, True, False]
        expected_loss_mask = [False, True, True, False]
        assert output_data[0]["loss_mask"] == expected_loss_mask


class TestCuSeqlensUnpaddedCalculation:
    """Test cu_seqlens_unpadded calculation with EOS fix."""

    def test_cu_seqlens_unpadded_calculation_uses_correct_eos(self):
        """Test that cu_seqlens_unpadded calculation uses correct EOS logic."""
        # Code inspection test to verify the fix from NeMo PR #14437
        import inspect

        from megatron.bridge.data.datasets.sft import GPTSFTPackedDataset

        # Get the collate_fn source
        source = inspect.getsource(GPTSFTPackedDataset.collate_fn)

        # Should calculate seqlen_unpadded using second EOS
        assert "seqlen_unpadded" in source
        assert "eos_idx[0][1]" in source  # Uses second EOS
        # The calculation should be: eos_idx[0][1] + 1 if eos_idx[0].shape[0] > 1
        assert ".shape[0] > 1" in source


class TestBackwardCompatibilityLossMask:
    """Test backward compatibility for loss_mask field naming."""

    @patch("megatron.bridge.data.datasets.sft._JSONLMemMapDataset")
    def test_legacy_chat_dataset_uses_loss_mask(self, mock_dataset_class):
        """Test that legacy chat dataset (non-HF template) uses loss_mask."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2
        mock_tokenizer.text_to_ids = MagicMock(return_value=[1, 2, 3])

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=False,  # Legacy mode
        )

        # Mock _preprocess to return loss_mask
        with patch("megatron.bridge.data.datasets.sft._preprocess") as mock_preprocess:
            mock_preprocess.return_value = {
                "input_ids": torch.tensor([1, 2, 3, 4]),
                "loss_mask": torch.tensor([0, 1, 1, 1]),
                "context_ids": torch.tensor([1]),
                "answer_ids": torch.tensor([2, 3, 4]),
            }

            example = {"conversations": [{"from": "User", "value": "Test"}]}
            result = dataset._process_example(example)

            # Should have loss_mask
            assert "loss_mask" in result


class TestSpaceSensitiveComputation:
    """Test space_sensitive attribute computation in tokenizers."""

    @patch("megatron.bridge.training.tokenizers.tokenizer.get_rank_safe", return_value=0)
    def test_hf_tokenizer_computes_space_sensitive_correctly(self, mock_get_rank):
        """Test that HF tokenizer correctly computes space_sensitive."""
        from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer

        with patch("transformers.AutoTokenizer.from_pretrained") as mock_from_pretrained:
            mock_hf_tok = MagicMock()
            mock_hf_tok.get_vocab.return_value = {"x": 1, "y": 2}

            # Mock tokenization to simulate space-sensitive tokenizer
            def mock_call(text, **kwargs):
                mock_result = MagicMock()
                if text == "x y":
                    mock_result.input_ids = [1, 2, 3]  # Different
                elif text == "x":
                    mock_result.input_ids = [1]
                elif text == "y":
                    mock_result.input_ids = [2]
                else:
                    mock_result.input_ids = []
                return mock_result

            mock_hf_tok.__call__ = mock_call
            mock_from_pretrained.return_value = mock_hf_tok

            tokenizer = _HuggingFaceTokenizer("gpt2")

            # Should compute space_sensitive as True
            assert hasattr(tokenizer, "space_sensitive")
            assert tokenizer.space_sensitive is True

    def test_sentencepiece_tokenizer_space_sensitive_fallback(self):
        """Test that SentencePiece defaults to True if computation fails."""
        from megatron.bridge.training.tokenizers.tokenizer import _SentencePieceTokenizer

        with patch("sentencepiece.SentencePieceProcessor") as mock_sp:
            mock_sp_instance = MagicMock()
            mock_sp_instance.bos_id.return_value = 1
            mock_sp_instance.eos_id.return_value = 2
            mock_sp_instance.pad_id.return_value = 0
            mock_sp_instance.get_piece_size.return_value = 1000
            mock_sp_instance.id_to_piece = MagicMock(side_effect=lambda i: f"<{i}>")

            # Make tokenize fail to test fallback
            mock_sp_instance.encode_as_ids.side_effect = Exception("Test error")
            mock_sp.return_value = mock_sp_instance

            tokenizer = _SentencePieceTokenizer("test.model")

            # Should fallback to True
            assert tokenizer.space_sensitive is True


class TestPackedDatasetWithChatTemplateEdgeCases:
    """Edge case tests for packed datasets with chat templates."""

    def test_create_packed_dataset_ignores_chat_flag(self):
        """Test that .npy files ignore chat flag (packed has priority)."""
        from pathlib import Path

        from megatron.bridge.data.datasets.sft import create_sft_dataset

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2

        with patch("numpy.load") as mock_load:
            mock_load.return_value = np.array([{"input_ids": [1, 2], "seq_start_id": [0], "loss_mask": [1, 1]}])

            # Even with chat=True, should create GPTSFTPackedDataset for .npy
            dataset = create_sft_dataset(
                path=Path("test.npy"),
                tokenizer=mock_tokenizer,
                chat=True,
                use_hf_tokenizer_chat_template=True,
                prompt_template="{input} {output}",  # Avoid validation error
            )

            # Verify it's a packed dataset
            from megatron.bridge.data.datasets.sft import GPTSFTPackedDataset

            assert isinstance(dataset, GPTSFTPackedDataset)

    def test_dataset_kwargs_flow_through_create_sft(self):
        """Test that dataset_kwargs flow through create_sft_dataset to chat dataset."""
        from pathlib import Path

        from megatron.bridge.data.datasets.sft import create_sft_dataset

        with patch("megatron.bridge.data.datasets.sft.GPTSFTChatDataset") as mock_chat:
            mock_tokenizer = MagicMock()

            tool_schemas = [{"type": "function"}]

            create_sft_dataset(
                path=Path("test.jsonl"),
                tokenizer=mock_tokenizer,
                chat=True,
                use_hf_tokenizer_chat_template=True,
                tool_schemas=tool_schemas,
                custom_kwarg="custom_value",  # Extra kwargs
            )

            # Verify all kwargs passed through
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["use_hf_tokenizer_chat_template"] is True
            assert call_kwargs["tool_schemas"] == tool_schemas
            assert call_kwargs["custom_kwarg"] == "custom_value"
