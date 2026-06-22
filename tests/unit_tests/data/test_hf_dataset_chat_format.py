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

from datasets import Dataset, DatasetDict

from megatron.bridge.data.builders.hf_dataset import preprocess_and_split_data


CHAT_EXAMPLES = [
    {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "tools": [{"name": "search", "description": "Search the web"}],
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        "tools": [],
    },
]


def process_chat_example(example, tokenizer=None):
    """Extract messages and tools from a chat-format example."""
    return {"messages": example["messages"], "tools": example["tools"]}


def process_input_output_example(example, tokenizer=None):
    """Extract input/output from chat messages for backward-compat testing."""
    return {
        "input": example["messages"][0]["content"],
        "output": example["messages"][1]["content"],
    }


class TestPreprocessChatFormat:
    """Tests that preprocess_and_split_data writes arbitrary process_example_fn output to JSONL."""

    def test_chat_format_writes_messages_and_tools(self, tmp_path):
        """Chat-format process functions should write messages/tools keys to JSONL."""
        dset = DatasetDict({"train": Dataset.from_list(CHAT_EXAMPLES)})

        preprocess_and_split_data(
            dset=dset,
            dataset_name="test-chat",
            dataset_root=tmp_path,
            tokenizer=None,
            process_example_fn=process_chat_example,
            val_proportion=None,
            do_validation=False,
            do_test=False,
            rewrite=True,
        )

        output_file = tmp_path / "training.jsonl"
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        for i, line in enumerate(lines):
            data = json.loads(line)
            assert "messages" in data, f"Line {i} missing 'messages' key"
            assert "tools" in data, f"Line {i} missing 'tools' key"
            assert data["messages"] == CHAT_EXAMPLES[i]["messages"]
            assert data["tools"] == CHAT_EXAMPLES[i]["tools"]
            assert "input" not in data
            assert "output" not in data

    def test_input_output_format_still_works(self, tmp_path):
        """Backward compat: input/output process functions should still produce correct JSONL."""
        dset = DatasetDict({"train": Dataset.from_list(CHAT_EXAMPLES)})

        preprocess_and_split_data(
            dset=dset,
            dataset_name="test-io",
            dataset_root=tmp_path,
            tokenizer=None,
            process_example_fn=process_input_output_example,
            val_proportion=None,
            do_validation=False,
            do_test=False,
            rewrite=True,
        )

        output_file = tmp_path / "training.jsonl"
        assert output_file.exists()

        lines = output_file.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            data = json.loads(line)
            assert "input" in data
            assert "output" in data
            assert "messages" not in data
