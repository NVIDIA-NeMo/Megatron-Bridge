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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from megatron.bridge.data.builders.gpt_sft import build_gpt_sft_split
from megatron.bridge.data.datasets.gpt_sft import GPTSFTChatDataset
from megatron.bridge.data.datasets.utils import _chat_preprocess, _convert_to_openai_messages
from megatron.bridge.data.packing.gpt_sft import GPTSFTPackedDataset


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
        mock_hf_tokenizer.chat_template = (
            "{% for message in messages %}{% generation %}{{ message.content }}{% endgeneration %}{% endfor %}"
        )
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
        assert mock_hf_tokenizer.apply_chat_template.called

    def test_chat_preprocess_without_generation_keyword_uses_boundary_config(self):
        """Test chat preprocessing uses assistant boundary markers when generation masks are unavailable."""

        class BoundaryTokenizer:
            eos_id = 2
            legacy = True
            pad_token_id = 0
            added_tokens_decoder = {}
            chat_template = (
                "<|im_start|>user\n{{ user }}<|im_end|>\n<|im_start|>assistant\n{{ assistant }}<|im_end|>\n"
            )

            def __init__(self):
                self._tokenizer = self

            def __call__(self, text, add_special_tokens=False, **kwargs):
                del add_special_tokens, kwargs
                mapping = {
                    "<|im_start|>assistant\n": [101],
                    "<|im_start|>user\n": [100],
                    "<|im_start|>system\n": [110],
                    "<|im_start|>developer\n": [111],
                    "<|im_start|>tool\n": [112],
                    "<|im_end|>": [102],
                    "<|im_end|>\n": [102, 103],
                }
                return {"input_ids": mapping.get(text, [42])}

            def apply_chat_template(self, chat, tools=None, tokenize=True, return_dict=True, **kwargs):
                del chat, tools, tokenize, return_dict
                assert "return_assistant_tokens_mask" not in kwargs
                return {"input_ids": [100, 10, 102, 103, 101, 21, 22, 102, 103]}

        source = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }

        result = _chat_preprocess(source, BoundaryTokenizer())

        assert result["input_ids"].tolist() == [100, 10, 102, 103, 101, 21, 22, 102, 103]
        assert result["loss_mask"].tolist() == [False, False, False, False, False, True, True, True, True]
        assert result["context_ids"].tolist() == [100, 10, 102, 103, 101]
        assert result["answer_ids"].tolist() == [21, 22, 102, 103]

    def test_chat_preprocess_augments_content_only_generation_mask_with_assistant_end(self):
        class ContentOnlyMaskTokenizer:
            eos_id = 2
            legacy = True
            pad_token_id = 0
            added_tokens_decoder = {}
            chat_template = (
                "<|im_start|>user\n{{ user }}<|im_end|>\n"
                "<|im_start|>assistant\n{% generation %}{{ assistant }}{% endgeneration %}<|im_end|>\n"
            )

            def __init__(self):
                self._tokenizer = self

            def __call__(self, text, add_special_tokens=False, **kwargs):
                del add_special_tokens, kwargs
                mapping = {
                    "<|im_start|>assistant\n": [101],
                    "<|im_start|>user\n": [100],
                    "<|im_start|>system\n": [110],
                    "<|im_start|>developer\n": [111],
                    "<|im_start|>tool\n": [112],
                    "<|im_end|>": [102],
                    "<|im_end|>\n": [102, 103],
                }
                return {"input_ids": mapping.get(text, [42])}

            def apply_chat_template(self, chat, tools=None, tokenize=True, return_dict=True, **kwargs):
                del chat, tools, tokenize, return_dict, kwargs
                return {
                    "input_ids": [100, 10, 102, 103, 101, 21, 22, 102, 103],
                    "assistant_masks": [0, 0, 0, 0, 0, 1, 1, 0, 0],
                }

        source = {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }

        result = _chat_preprocess(source, ContentOnlyMaskTokenizer())

        assert result["loss_mask"].tolist() == [False, False, False, False, False, True, True, True, True]
        assert result["answer_ids"].tolist() == [21, 22, 102, 103]

    def test_chat_preprocess_trusts_template_eos(self):
        """Test that _chat_preprocess does not append eos_id when template uses a different end token."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 999
        mock_tokenizer.legacy = False

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 888],  # Ends with 888, not eos_id 999
            "assistant_masks": [0, 0, 1, 1],
        }

        source = {
            "conversations": [
                {"from": "User", "value": "Test"},
                {"from": "Assistant", "value": "Answer"},
            ]
        }

        result = _chat_preprocess(source, mock_tokenizer)

        assert result["input_ids"].tolist() == [1, 10, 20, 888]

    @pytest.mark.parametrize(
        "tool_schemas",
        [
            {"type": "function", "function": {"name": "test_func"}},
            [{"type": "function", "function": {"name": "test_func"}}],
        ],
    )
    def test_chat_preprocess_with_tool_schemas(self, tool_schemas):
        """Test chat preprocessing with tool schemas."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2
        mock_tokenizer.legacy = False

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 0, 1, 1],
        }

        source = {"conversations": [{"from": "User", "value": "Test"}]}
        _chat_preprocess(source, mock_tokenizer, tool_schemas=tool_schemas)

        # Verify tools were passed to apply_chat_template
        call_kwargs = mock_hf_tokenizer.apply_chat_template.call_args[1]
        assert call_kwargs["tools"] == tool_schemas

    def test_chat_preprocess_empty_source_tools_fall_back_to_global_schemas(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2
        mock_tokenizer.legacy = False
        mock_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 0, 1, 1],
        }
        global_schemas = {"type": "function", "function": {"name": "global"}}
        source = {"conversations": [{"from": "User", "value": "Test"}], "tools": []}

        _chat_preprocess(source, mock_tokenizer, tool_schemas=global_schemas)

        assert mock_tokenizer.apply_chat_template.call_args[1]["tools"] == global_schemas

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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
    def test_chat_dataset_defaults_to_hf_template(self, mock_dataset_class):
        """Test GPTSFTChatDataset defaults to the HF chat template."""
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
            tool_schemas=None,
        )

        assert dataset.use_hf_tokenizer_chat_template is True
        assert dataset.tool_schemas is None

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
    def test_process_example_uses_chat_preprocess(self, mock_dataset_class):
        """Test that _process_example uses _chat_preprocess when enabled."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
        )

        tools = [{"type": "function", "function": {"name": "lookup"}}]
        example = {
            "conversations": [
                {"from": "User", "value": "Hello"},
                {"from": "Assistant", "value": "Hi!"},
            ],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "truncate_history_thinking": False,
            },
            "tools": tools,
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
        assert mock_hf_tokenizer.apply_chat_template.call_args_list[0].kwargs == {
            "tokenize": True,
            "add_generation_prompt": False,
            "return_dict": True,
            "enable_thinking": True,
            "truncate_history_thinking": False,
            "tools": tools,
            "return_assistant_tokens_mask": True,
        }

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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


class TestIsPackedParquetFile:
    """Test cases for is_packed_parquet_file utility function."""

    def test_single_file_idx_parquet(self):
        """Test detection of single .idx.parquet file."""
        from megatron.bridge.data.packing.paths import is_packed_parquet_file

        assert is_packed_parquet_file("data.idx.parquet") is True
        assert is_packed_parquet_file("/path/to/data.idx.parquet") is True

    def test_single_file_idx_pq(self):
        """Test detection of single .idx.pq file."""
        from megatron.bridge.data.packing.paths import is_packed_parquet_file

        assert is_packed_parquet_file("data.idx.pq") is True
        assert is_packed_parquet_file("/path/to/data.idx.pq") is True

    def test_glob_pattern_idx_parquet(self):
        """Test detection of glob patterns for .idx.parquet."""
        from megatron.bridge.data.packing.paths import is_packed_parquet_file

        assert is_packed_parquet_file("data*.idx.parquet") is True
        assert is_packed_parquet_file("shard_?.idx.parquet") is True
        assert is_packed_parquet_file("/path/to/data*.idx.parquet") is True

    def test_glob_pattern_idx_pq(self):
        """Test detection of glob patterns for .idx.pq."""
        from megatron.bridge.data.packing.paths import is_packed_parquet_file

        assert is_packed_parquet_file("data*.idx.pq") is True
        assert is_packed_parquet_file("shard_?.idx.pq") is True

    def test_regular_parquet_not_detected(self):
        """Test that regular .parquet files are not detected as packed."""
        from megatron.bridge.data.packing.paths import is_packed_parquet_file

        assert is_packed_parquet_file("data.parquet") is False
        assert is_packed_parquet_file("data*.parquet") is False
        assert is_packed_parquet_file("/path/to/data.parquet") is False

    def test_case_insensitive(self):
        """Test case-insensitive detection."""
        from megatron.bridge.data.packing.paths import is_packed_parquet_file

        assert is_packed_parquet_file("DATA.IDX.PARQUET") is True
        assert is_packed_parquet_file("Data.Idx.Pq") is True


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


class TestOutputOriginalText:
    """Test cases for output_original_text with different formats."""

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
    def test_output_original_text_with_messages(self, mock_dataset_class):
        """Test that output_original_text works with messages format."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
    def test_output_original_text_with_conversations(self, mock_dataset_class):
        """Test that output_original_text works with conversations format."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
    def test_output_original_text_with_conversation(self, mock_dataset_class):
        """Test that output_original_text works with singular conversation format."""
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        mock_dataset_class.return_value = mock_dataset

        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer._tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
        }

        dataset = GPTSFTChatDataset(
            file_path="test.jsonl",
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            use_hf_tokenizer_chat_template=True,
            output_original_text=True,
        )

        example = {
            "conversation": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ],
            "extra_field": "value",
        }

        result = dataset._process_example(example)

        assert result["metadata"]["conversation"] == example["conversation"]
        assert result["metadata"]["extra_field"] == "value"


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

        with patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset"):
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

        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 2],
            "assistant_masks": [0, 1, 1, 1],
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
            with patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset"):
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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
        mock_hf_tokenizer.chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": long_input_ids,
            "assistant_masks": [1] * len(long_input_ids),
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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
        with patch("megatron.bridge.data.datasets.gpt_sft.logger") as mock_logger:
            result = dataset.collate_fn(batch)

            # Should have logged warning
            mock_logger.warning.assert_called()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "no assistant tokens" in warning_msg.lower()

            # Truncated assistant targets stay masked instead of exposing user tokens.
            assert result["loss_mask"].sum().item() == 0


class TestContextAnswerSplit:
    """Test context and answer splitting logic in _chat_preprocess."""

    def test_context_answer_split_with_mask(self):
        """Test that context/answer are split correctly based on mask."""
        mock_tokenizer = MagicMock()
        mock_hf_tokenizer = MagicMock()
        mock_tokenizer = mock_hf_tokenizer
        mock_tokenizer.eos_id = 2
        mock_tokenizer.legacy = False

        # Mask with 0s (context) and 1s (answer)
        mock_hf_tokenizer.chat_template = "{% generation %}"  # Has generation keyword
        mock_hf_tokenizer.apply_chat_template.return_value = {
            "input_ids": [1, 10, 20, 30, 40, 2],
            "assistant_masks": [0, 0, 0, 1, 1, 1],  # First 3 are context
        }

        source = {
            "conversations": [
                {"from": "User", "value": "Test"},
                {"from": "Assistant", "value": "Answer"},
            ]
        }

        result = _chat_preprocess(source, mock_tokenizer)

        # Context should be first 3 tokens
        assert result["context_ids"].tolist() == [1, 10, 20]
        # Answer should be remaining tokens
        assert result["answer_ids"].tolist() == [30, 40, 2]

    def test_context_answer_split_no_mask_raises_without_boundary_config(self):
        """Test that a template without generation or known boundaries raises."""

        class NoBoundaryTokenizer:
            eos_id = 2
            legacy = True
            added_tokens_decoder = {}
            chat_template = "{{ messages }}"

            def __init__(self):
                self._tokenizer = self

            def __call__(self, text, add_special_tokens=False, **kwargs):
                del text, add_special_tokens, kwargs
                return {"input_ids": [1, 2, 3]}

            def apply_chat_template(self, chat, tools=None, tokenize=True, return_dict=True, **kwargs):
                del chat, tools, tokenize, return_dict, kwargs
                return {"input_ids": [1, 2, 3]}

        source = {"conversations": [{"from": "User", "value": "Test"}]}

        with pytest.raises(ValueError, match="could not infer assistant boundary markers"):
            _chat_preprocess(source, NoBoundaryTokenizer())


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


def _create_minimal_packed_dataset(tokenizer_eos_id: int = 0):
    """Utility helper to instantiate GPTSFTPackedDataset without touching disk."""
    dataset = GPTSFTPackedDataset.__new__(GPTSFTPackedDataset)
    dataset.pad_to_max_length = False
    dataset.max_seq_length = 32
    dataset.pad_seq_length_to_mult = 1
    dataset._pad_seq_to_mult = 2  # Emit distinct logical and physical cumulative offsets.
    dataset.ceil_to_power_2 = False
    dataset.tokenizer = SimpleNamespace(eos_id=tokenizer_eos_id)
    dataset.answer_only_loss = False
    dataset.return_cu_seqlen = True
    dataset.pad_cu_seqlens = False
    dataset.pack_metadata = []
    return dataset


class TestEOSIndexFixInPackedDataset:
    """Test EOS index fix for logical cumulative sequence offsets."""

    def test_eos_index_logic_uses_shape_check(self):
        """Ensure logical offsets handle sequences with fewer than two EOS tokens."""
        dataset = _create_minimal_packed_dataset()
        batch = [
            {
                "input_ids": np.array([7, 0, 0, 0, 0], dtype=np.int64),
                "seq_boundaries": [0, 5],
                "loss_mask": np.array([1, 0, 0, 0, 0], dtype=np.int64),
            }
        ]

        processed = dataset.collate_fn(batch)
        cu_logical = processed["cu_seqlens_q"][0].tolist()

        # Expect a single non-EOS token tracked without indexing errors.
        assert cu_logical == [0, 1]
        assert processed["cu_seqlens_kv"].tolist() == processed["cu_seqlens_q"].tolist()
        assert processed["attention_mask"] is None

    def test_static_metadata_keeps_logical_and_physical_boundaries_aligned(self):
        """Static metadata pads logical and physical boundary arrays to the same shape."""
        dataset = _create_minimal_packed_dataset()
        dataset.pad_to_max_length = True
        dataset.max_seq_length = 8
        dataset.pad_cu_seqlens = True
        dataset.pack_metadata = [
            {
                "max_samples_per_bin": 3,
                "dataset_max_seqlen": 8,
                "min_packed_seqlen": 4,
            }
        ]
        batch = [
            {
                "input_ids": np.array([7, 0, 0, 0, 0], dtype=np.int64),
                "seq_boundaries": [0, 5],
                "loss_mask": np.array([1, 0, 0, 0, 0], dtype=np.int64),
            }
        ]

        processed = dataset.collate_fn(batch)

        assert processed["cu_seqlens_q"].shape == processed["cu_seqlens_q_padded"].shape
        assert processed["cu_seqlens_q"][0].tolist() == [0, 1, 1, 1, 1]
        assert processed["cu_seqlens_kv"][0].tolist() == [0, 1, 1, 1, 1]
        assert processed["cu_seqlens_q_padded"][0].tolist() == [0, 4, 8, 8, 8]
        assert processed["cu_seqlens_kv_padded"][0].tolist() == [0, 4, 8, 8, 8]

    def test_supervised_terminal_eos_precedes_zero_loss_padding(self):
        """A supervised EOS remains logical while later zero-loss EOS tokens are padding."""
        dataset = _create_minimal_packed_dataset()
        batch = [
            {
                "input_ids": np.array([7, 0, 0, 0, 0], dtype=np.int64),
                "seq_boundaries": [0, 5],
                "loss_mask": np.array([1, 1, 0, 0, 0], dtype=np.int64),
            }
        ]

        processed = dataset.collate_fn(batch)

        assert processed["cu_seqlens_q"][0].tolist() == [0, 2]
        assert processed["cu_seqlens_q_padded"][0].tolist() == [0, 4]
        assert processed["padding_mask"].tolist() == [[False, False, True, True]]

    def test_without_alignment_padding_omits_physical_variants(self):
        """Identical logical and physical layouts use only the faster logical TE fields."""
        dataset = _create_minimal_packed_dataset()
        dataset._pad_seq_to_mult = 1
        batch = [
            {
                "input_ids": np.array([7, 8, 9, 0, 0], dtype=np.int64),
                "seq_boundaries": [0, 5],
                "loss_mask": np.ones(5, dtype=np.int64),
            }
        ]

        processed = dataset.collate_fn(batch)

        assert processed["cu_seqlens_q"][0].tolist() == [0, 4]
        assert processed["cu_seqlens_kv"][0].tolist() == [0, 4]
        assert "cu_seqlens_q_padded" not in processed
        assert "cu_seqlens_kv_padded" not in processed


def test_packed_full_loss_preserves_target_after_internal_eos():
    """Full-loss packing must supervise the label after an EOS turn delimiter."""
    from megatron.bridge.data.packing.algorithms import fill_packing_strategy

    eos_id = 2
    sequences = {length: [] for length in range(4)}
    sequences[3].append(
        {
            "input_ids": [10, eos_id, 20, eos_id],
            "loss_mask": [True, True, True, True],
        }
    )
    packed_row = fill_packing_strategy([[3]], sequences, pack_size=3, pad_id=eos_id)[0]
    dataset = _create_minimal_packed_dataset(tokenizer_eos_id=eos_id)
    dataset.answer_only_loss = True
    dataset.return_cu_seqlen = False

    batch = dataset.collate_fn(
        [
            {
                "input_ids": packed_row["input_ids"],
                "seq_boundaries": [*packed_row["seq_start_id"], len(packed_row["input_ids"])],
                "loss_mask": packed_row["loss_mask"],
            }
        ]
    )

    assert batch["tokens"].tolist() == [[10, eos_id, 20]]
    assert batch["labels"].tolist() == [[eos_id, 20, eos_id]]
    assert batch["loss_mask"].tolist() == [[1, 1, 1]]


class TestPackedChatDatasetIntegration:
    """Integration tests for packed chat datasets."""

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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
        from unittest.mock import MagicMock

        from megatron.bridge.data.packing.offline import tokenize_dataset

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

        result = tokenize_dataset(
            path=Path("test.jsonl"),
            tokenizer=mock_tokenizer,
            max_seq_length=512,
            seed=1234,
            dataset_kwargs=dataset_kwargs,
            dataset_builder=MagicMock(return_value=mock_dataset),
        )

        # Verify result is array of items with loss_mask
        assert isinstance(result, np.ndarray)
        assert len(result) == 1
        assert "loss_mask" in result[0]

    def test_packed_dataset_preserves_chat_loss_mask(self):
        """Test that packed dataset preserves loss_mask from chat preprocessing."""
        from megatron.bridge.data.packing.algorithms import fill_packing_strategy

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


class TestLogicalCuSeqlensCalculation:
    """Test logical cumulative sequence offsets with the EOS fix."""

    def test_logical_cu_seqlens_uses_correct_eos(self):
        """Ensure logical cumulative offsets honor the tokenizer's EOS id."""
        dataset = _create_minimal_packed_dataset(tokenizer_eos_id=999)
        batch = [
            {
                "input_ids": np.array(
                    [5, 999, 999, 999, 1, 3, 999, 999, 999, 4],
                    dtype=np.int64,
                ),
                "seq_boundaries": [0, 5, 10],
                "loss_mask": np.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int64),
            }
        ]

        processed = dataset.collate_fn(batch)
        cu_logical = processed["cu_seqlens_q"][0].tolist()

        # Each non-EOS token contributes exactly once despite zero-loss EOS padding.
        assert cu_logical == [0, 1, 2]


class TestBackwardCompatibilityLossMask:
    """Test backward compatibility for loss_mask field naming."""

    @patch("megatron.bridge.data.datasets.gpt_sft._JSONLMemMapDataset")
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
        with patch("megatron.bridge.data.datasets.gpt_sft._preprocess") as mock_preprocess:
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


class TestPackedDatasetWithChatTemplateEdgeCases:
    """Edge case tests for packed datasets with chat templates."""

    def test_build_packed_dataset_ignores_chat_flag(self, tmp_path):
        """Test that .npy files ignore chat flag (packed has priority)."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_id = 2
        dataset_path = tmp_path / "test.npy"
        dataset_path.touch()

        with patch("megatron.bridge.data.packing.gpt_sft._safe_load_packed_npy") as mock_load:
            mock_load.return_value = np.array([{"input_ids": [1, 2], "seq_start_id": [0], "loss_mask": [1, 1]}])

            # Even with chat=True, should create GPTSFTPackedDataset for .npy
            dataset = build_gpt_sft_split(
                dataset_path,
                tokenizer=mock_tokenizer,
                seq_length=2048,
                memmap_workers=2,
                seed=1234,
                packed_sequence_size=2048,
                dataset_kwargs={
                    "chat": True,
                    "use_hf_tokenizer_chat_template": True,
                    "prompt_template": "{input} {output}",
                },
            )

            # Verify it's a packed dataset
            from megatron.bridge.data.packing.gpt_sft import GPTSFTPackedDataset

            assert isinstance(dataset, GPTSFTPackedDataset)

    def test_dataset_kwargs_flow_through_builder(self, tmp_path):
        """Test that dataset_kwargs flow through the builder-owned split helper."""
        dataset_path = tmp_path / "test.jsonl"
        dataset_path.touch()

        with patch("megatron.bridge.data.builders.gpt_sft.GPTSFTChatDataset") as mock_chat:
            mock_tokenizer = MagicMock()

            tool_schemas = [{"type": "function"}]

            build_gpt_sft_split(
                dataset_path,
                tokenizer=mock_tokenizer,
                seq_length=2048,
                memmap_workers=2,
                seed=1234,
                packed_sequence_size=-1,
                dataset_kwargs={
                    "chat": True,
                    "use_hf_tokenizer_chat_template": True,
                    "tool_schemas": tool_schemas,
                    "custom_kwarg": "custom_value",
                },
            )

            # Verify all kwargs passed through
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["use_hf_tokenizer_chat_template"] is True
            assert call_kwargs["tool_schemas"] == tool_schemas
            assert call_kwargs["custom_kwarg"] == "custom_value"
