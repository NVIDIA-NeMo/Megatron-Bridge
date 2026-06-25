# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from megatron.bridge.data.hf_datasets.text_collate import text_chat_collate_fn


pytestmark = pytest.mark.unit


class _TextChatTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 2
    added_tokens_decoder = {}
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def __init__(self):
        self.conversations = []

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.conversations.append(conversation)
        if tokenize:
            assert kwargs.get("return_assistant_tokens_mask") is True
            if conversation[-1]["content"] == "bye":
                return {"input_ids": [11, 12, 21, 22], "assistant_masks": [0, 0, 1, 1]}
            return {"input_ids": [11, 21, 22], "assistant_masks": [0, 1, 1]}
        return "bye" if conversation[-1]["content"] == "bye" else "hello"

    def __call__(self, text, padding=True, truncation=False, return_tensors="pt", max_length=None, **kwargs):
        texts = text if isinstance(text, list) else [text]
        tokenized = [[11, 12, 21, 22] if item == "bye" else [11, 21, 22] for item in texts]
        if truncation and max_length is not None:
            tokenized = [ids[:max_length] for ids in tokenized]
        if padding == "max_length" and max_length is not None:
            max_len = max_length
        else:
            max_len = max(len(ids) for ids in tokenized) if padding else None
        input_ids = []
        attention_mask = []
        for ids in tokenized:
            row = list(ids)
            mask = [1] * len(row)
            if max_len is not None:
                pad_len = max_len - len(row)
                row.extend([self.pad_token_id] * pad_len)
                mask.extend([0] * pad_len)
            input_ids.append(row)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def test_text_chat_collate_fn_builds_shifted_assistant_labels_from_messages():
    tokenizer = _TextChatTokenizer()
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "later"},
                {"role": "assistant", "content": "bye"},
            ]
        },
    ]

    batch = text_chat_collate_fn(examples, tokenizer)

    assert batch["tokens"].tolist() == [[11, 21, 22, 0], [11, 12, 21, 22]]
    assert batch["input_ids"].data_ptr() == batch["tokens"].data_ptr()
    assert batch["labels"].tolist() == [[21, 22, -100, -100], [-100, 21, 22, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3], [0, 1, 2, 3]]
    assert batch["token_count"] == [3, 4]


def test_text_chat_collate_fn_accepts_legacy_conversations_and_max_length():
    tokenizer = _TextChatTokenizer()
    examples = [
        {
            "conversations": [
                {"from": "User", "value": "hi"},
                {"from": "Assistant", "value": "hello"},
            ]
        }
    ]

    batch = text_chat_collate_fn(examples, tokenizer, max_length=4, pad_to_max_length=True)

    expected_conversation = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert tokenizer.conversations == [expected_conversation, expected_conversation]
    assert batch["tokens"].tolist() == [[11, 21, 22, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0]]
    assert batch["labels"].tolist() == [[21, 22, -100, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0]]
    assert batch["token_count"] == [3]


def test_text_chat_collate_fn_packs_sequences_for_gpt_step():
    tokenizer = _TextChatTokenizer()
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "later"},
                {"role": "assistant", "content": "bye"},
            ]
        },
    ]

    batch = text_chat_collate_fn(examples, tokenizer, enable_in_batch_packing=True)

    assert batch["tokens"].tolist() == [[11, 21, 22, 11, 12, 21, 22]]
    assert batch["input_ids"].data_ptr() == batch["tokens"].data_ptr()
    assert batch["labels"].tolist() == [[21, 22, -100, -100, 21, 22, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens"].tolist() == [[0, 3, 7]]
    assert batch["cu_seqlens_argmin"].item() == 3
    assert batch["max_seqlen"].tolist() == [[4]]
    assert "cu_seqlens_unpadded" not in batch
    assert "cu_seqlens_unpadded_argmin" not in batch


def test_text_chat_collate_fn_pads_packed_sequences_to_multiple():
    tokenizer = _TextChatTokenizer()
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "later"},
                {"role": "assistant", "content": "bye"},
            ]
        },
    ]

    batch = text_chat_collate_fn(
        examples, tokenizer, enable_in_batch_packing=True, in_batch_packing_pad_to_multiple_of=4
    )

    assert batch["tokens"].tolist() == [[11, 21, 22, 0, 11, 12, 21, 22]]
    assert batch["labels"].tolist() == [[21, 22, -100, -100, -100, 21, 22, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3, 0, 1, 2, 3]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens"].tolist() == [[0, 4, 8]]
    assert batch["cu_seqlens_argmin"].item() == 3
    assert batch["max_seqlen"].tolist() == [[4]]
    assert batch["cu_seqlens_unpadded"].tolist() == [[0, 3, 7]]
    assert batch["cu_seqlens_unpadded_argmin"].item() == 3
