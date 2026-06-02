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

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.vlm_processing import (
    HFProcessorVLMDataProcessor,
    apply_assistant_labels_to_batch,
    build_assistant_loss_mask,
    build_shifted_labels_and_loss_mask,
    convert_media_placeholders_to_content_parts,
    gather_assistant_text_segments,
)


pytestmark = pytest.mark.unit


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99
    added_tokens_decoder = {}

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "answer": [3, 4],
            "answer\n": [3, 4, 99],
            "ok": [7],
        }
        return {"input_ids": mapping.get(text, [42])}


class _Processor:
    image_token_id = 10

    def __init__(self):
        self.tokenizer = _Tokenizer()
        self.template_inputs = []
        self.processor_inputs = []

    def apply_chat_template(self, conversation, tokenize=False):
        self.template_inputs.append((conversation, tokenize))
        return "prompt"

    def __call__(self, **kwargs):
        self.processor_inputs.append(kwargs)
        output = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
        if kwargs.get("images") is not None:
            output["pixel_values"] = torch.ones(len(kwargs["images"]), 3, 4, 4)
        return output


def test_gather_assistant_text_segments_handles_structured_and_string_content():
    example = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}, {"type": "image"}]},
            {"role": "assistant", "content": "ok"},
        ]
    }

    assert gather_assistant_text_segments(example) == ["answer", "ok"]


def test_build_assistant_loss_mask_uses_current_search_variants_and_skipped_tokens():
    example = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }
    input_ids = torch.tensor([1, 2, 3, 4, 99])

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), skipped_tokens=torch.tensor([4]))

    assert mask.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]


def test_build_shifted_labels_and_loss_mask_aligns_next_token_labels():
    input_ids = torch.tensor([1, 2, 3, 4, 5])
    assistant_mask = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])

    labels, shifted_mask = build_shifted_labels_and_loss_mask(
        input_ids, assistant_mask, skipped_tokens=torch.tensor([5])
    )

    assert shifted_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert labels.tolist() == [IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX]


def test_apply_assistant_labels_to_batch_mutates_batch_with_shared_masking():
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    apply_assistant_labels_to_batch(batch, examples, _Processor(), skipped_tokens=torch.tensor([]))

    assert batch["loss_mask"].tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX]]


def test_convert_media_placeholders_to_content_parts_preserves_text_order():
    conversation = [{"role": "user", "content": "before <image> between <video> after"}]

    converted = convert_media_placeholders_to_content_parts(conversation)

    assert converted == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image"},
                {"type": "text", "text": "between"},
                {"type": "video"},
                {"type": "text", "text": "after"},
            ],
        }
    ]


def test_hf_processor_vlm_data_processor_encodes_one_sample_and_collects_visual_tensors():
    processor = _Processor()
    data_processor = HFProcessorVLMDataProcessor(
        processor,
        seq_length=8,
        visual_keys=("pixel_values",),
    )
    conversation = [
        {"role": "user", "content": "describe <image>"},
        {"role": "assistant", "content": "answer"},
    ]
    image = object()

    encoded = data_processor.encode(conversation, images=[image])

    assert encoded.input_ids.tolist() == [1, 2, 3, 4, 5]
    assert encoded.loss_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert encoded.labels.tolist() == [IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX]
    assert encoded.visual_tensors["pixel_values"].shape == (1, 3, 4, 4)
    templated_conversation = processor.template_inputs[0][0]
    assert templated_conversation[0]["content"][1] == {"type": "image"}
    assert processor.processor_inputs[0]["images"] == [image]
