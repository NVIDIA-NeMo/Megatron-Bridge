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
