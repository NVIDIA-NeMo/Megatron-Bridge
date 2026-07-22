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

from megatron.bridge.data.collators.sft import text_chat_collate_fn, text_prompt_completion_collate_fn
from megatron.bridge.data.sft_processing import PromptCompletionSFTPreprocessingConfig


pytestmark = pytest.mark.unit


class _TextChatTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    padding_side = "right"
    eos_token_id = 2
    added_tokens_decoder = {}
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def __init__(self):
        self.conversations = []
        self.padding_values = []
        self.template_kwargs = []

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        self.conversations.append(conversation)
        self.template_kwargs.append(kwargs)
        if tokenize:
            assert kwargs.get("return_assistant_tokens_mask") is True
            if conversation[-1]["content"] == "bye":
                input_ids = [11, 12, 21, 22]
                assistant_masks = [0, 0, 1, 1]
            else:
                input_ids = [11, 21, 22]
                assistant_masks = [0, 1, 1]
            if kwargs.get("truncation") and kwargs.get("max_length") is not None:
                input_ids = input_ids[: kwargs["max_length"]]
                assistant_masks = assistant_masks[: kwargs["max_length"]]
            return {"input_ids": input_ids, "assistant_masks": assistant_masks}
        return "bye" if conversation[-1]["content"] == "bye" else "hello"

    def __call__(
        self,
        text,
        padding=True,
        truncation=False,
        return_tensors="pt",
        max_length=None,
        pad_to_multiple_of=None,
        **kwargs,
    ):
        texts = text if isinstance(text, list) else [text]
        if isinstance(text, list):
            self.padding_values.append(padding)
        tokenized = [[11, 12, 21, 22] if item == "bye" else [11, 21, 22] for item in texts]
        if truncation and max_length is not None:
            tokenized = [ids[:max_length] for ids in tokenized]
        if padding == "max_length" and max_length is not None:
            max_len = max_length
        else:
            max_len = max(len(ids) for ids in tokenized) if padding else None
            if max_len is not None and pad_to_multiple_of is not None and pad_to_multiple_of > 1:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        input_ids = []
        attention_mask = []
        for ids in tokenized:
            row = list(ids)
            mask = [1] * len(row)
            if max_len is not None:
                pad_len = max_len - len(row)
                if self.padding_side == "left":
                    row = [self.pad_token_id] * pad_len + row
                    mask = [0] * pad_len + mask
                else:
                    row.extend([self.pad_token_id] * pad_len)
                    mask.extend([0] * pad_len)
            input_ids.append(row)
            attention_mask.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) for character in text]


class _ChatMLBoundaryTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    added_tokens_decoder = {102: "<|im_end|>"}
    chat_template = "<|im_start|>user\n{{ user }}<|im_end|>\n<|im_start|>assistant\n{{ assistant }}<|im_end|>\n"

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False, **kwargs):
        if tokenize:
            input_ids = [100, 10, 102, 103, 101, 21, 22, 102, 103]
            if kwargs.get("truncation") and kwargs.get("max_length") is not None:
                input_ids = input_ids[: kwargs["max_length"]]
            return {"input_ids": input_ids}
        return "rendered-boundary"

    def __call__(
        self,
        text,
        padding=True,
        truncation=False,
        return_tensors="pt",
        max_length=None,
        add_special_tokens=True,
        **kwargs,
    ):
        del kwargs, add_special_tokens
        if text == "<|im_start|>assistant\n":
            return {"input_ids": [101]}
        if text == "<|im_end|>":
            return {"input_ids": [102]}
        if text == "<|im_end|>\n":
            return {"input_ids": [102, 103]}

        texts = text if isinstance(text, list) else [text]
        tokenized = [[100, 10, 102, 103, 101, 21, 22, 102, 103] for _ in texts]
        if truncation and max_length is not None:
            tokenized = [ids[:max_length] for ids in tokenized]
        max_len = (
            max_length if padding == "max_length" and max_length is not None else max(len(ids) for ids in tokenized)
        )
        input_ids = []
        attention_mask = []
        for ids in tokenized:
            row = list(ids)
            mask = [1] * len(row)
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


def test_text_chat_collate_fn_pads_non_packed_sequences_to_multiple():
    tokenizer = _TextChatTokenizer()
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]

    batch = text_chat_collate_fn(examples, tokenizer, pad_to_multiple_of=4)

    assert batch["tokens"].tolist() == [[11, 21, 22, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 3]]


def test_text_chat_collate_fn_caps_non_packed_padding_multiple_at_max_length():
    tokenizer = _TextChatTokenizer()
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
    ]

    batch = text_chat_collate_fn(examples, tokenizer, max_length=3, pad_to_multiple_of=128)

    assert batch["tokens"].shape == (1, 3)
    assert batch["tokens"].tolist() == [[11, 21, 22]]


def test_text_chat_collate_fn_uses_chatml_boundary_mask_without_generation_template():
    tokenizer = _ChatMLBoundaryTokenizer()
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "answer"},
            ]
        }
    ]

    batch = text_chat_collate_fn(examples, tokenizer)

    assert batch["tokens"].tolist() == [[100, 10, 102, 103, 101, 21, 22, 102, 103]]
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 21, 22, 102, 103, -100]]
    assert batch["loss_mask"].tolist() == [[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]


@pytest.mark.parametrize("enable_in_batch_packing", [False, True])
def test_text_chat_collate_fn_forwards_tools_to_render_and_mask_templates(enable_in_batch_packing):
    tokenizer = _TextChatTokenizer()
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    examples = [
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "tools": tools,
        }
    ]

    text_chat_collate_fn(examples, tokenizer, enable_in_batch_packing=enable_in_batch_packing)

    assert tokenizer.template_kwargs
    assert all(template_kwargs.get("tools") == tools for template_kwargs in tokenizer.template_kwargs)


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

    batch = text_chat_collate_fn(examples, tokenizer, max_length=2, pad_to_max_length=True)

    expected_conversation = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert tokenizer.conversations == [expected_conversation]
    assert batch["tokens"].tolist() == [[11, 21]]
    assert batch["attention_mask"].tolist() == [[1, 1]]
    assert batch["labels"].tolist() == [[21, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 0.0]]
    assert batch["token_count"] == [2]


def test_text_chat_collate_fn_packs_sequences_for_gpt_step():
    tokenizer = _TextChatTokenizer()
    tokenizer.padding_side = "left"
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
    assert tokenizer.padding_values == []
    assert tokenizer.padding_side == "left"
    assert batch["input_ids"].data_ptr() == batch["tokens"].data_ptr()
    assert batch["labels"].tolist() == [[21, 22, -100, -100, 21, 22, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3]]
    assert batch["attention_mask"] is None
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 7]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3, 7]
    assert batch["max_seqlen_q"].item() == 4
    assert batch["max_seqlen_kv"].item() == 4
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_argmin" not in batch
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
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 7]
    assert batch["cu_seqlens_kv"].tolist() == [0, 3, 7]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 8]
    assert batch["cu_seqlens_kv_padded"].tolist() == [0, 4, 8]
    assert batch["max_seqlen_q"].item() == 4
    assert batch["max_seqlen_kv"].item() == 4
    assert "cu_seqlens" not in batch
    assert "cu_seqlens_argmin" not in batch
    assert "cu_seqlens_unpadded" not in batch
    assert "cu_seqlens_unpadded_argmin" not in batch


def test_text_chat_collate_fn_packed_width_is_emergent_when_pad_to_max_length_is_set():
    tokenizer = _TextChatTokenizer()
    examples = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "later"}, {"role": "assistant", "content": "bye"}]},
    ]

    batch = text_chat_collate_fn(
        examples,
        tokenizer,
        max_length=16,
        pad_to_max_length=True,
        enable_in_batch_packing=True,
        in_batch_packing_pad_to_multiple_of=4,
    )

    assert batch["tokens"].shape == (1, 8)
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 7]
    assert batch["cu_seqlens_q_padded"].tolist() == [0, 4, 8]
    assert batch["total_tokens"] == 8


def test_text_chat_collate_fn_allows_packed_aggregate_over_per_row_max_length():
    tokenizer = _TextChatTokenizer()
    examples = [
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
        {"messages": [{"role": "user", "content": "later"}, {"role": "assistant", "content": "bye"}]},
    ]

    batch = text_chat_collate_fn(
        examples,
        tokenizer,
        max_length=6,
        enable_in_batch_packing=True,
    )

    assert batch["tokens"].shape == (1, 7)
    assert batch["cu_seqlens_q"].tolist() == [0, 3, 7]
    assert batch["total_tokens"] == 7


@pytest.mark.parametrize("enable_in_batch_packing", [False, True])
def test_text_prompt_completion_collate_masks_prompt_without_chat_template(enable_in_batch_packing):
    tokenizer = _TextChatTokenizer()
    tokenizer.apply_chat_template = lambda *args, **kwargs: pytest.fail("chat template must not be called")
    preprocessing = PromptCompletionSFTPreprocessingConfig(
        prompt_column="question",
        completion_column="answer",
        separator=" ",
    )

    batch = text_prompt_completion_collate_fn(
        [{"question": "Q", "answer": "A", "id": 7}],
        tokenizer,
        preprocessing=preprocessing,
        enable_in_batch_packing=enable_in_batch_packing,
    )

    assert batch["tokens"].tolist() == [[ord("Q"), ord(" "), ord("A"), tokenizer.eos_token_id]]
    assert batch["labels"].tolist() == [[ord(" "), ord("A"), tokenizer.eos_token_id, -100]]
    assert batch["loss_mask"].tolist() == [[1.0, 1.0, 1.0, 0.0]]
    assert batch["metadata"] == [{"id": 7}]
