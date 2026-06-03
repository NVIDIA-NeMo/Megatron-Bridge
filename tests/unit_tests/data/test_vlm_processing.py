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

import json

import pytest
import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.energon.metadata import sample_metadata_kwargs
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample
from megatron.bridge.data.vlm_processing import (
    HFProcessorVLMDataProcessor,
    apply_assistant_labels_to_batch,
    build_assistant_loss_mask,
    build_shifted_labels_and_loss_mask,
    collect_media_from_conversation,
    convert_media_placeholders_to_content_parts,
    gather_assistant_text_segments,
    normalize_energon_vlm_sample,
    normalize_hf_vlm_example,
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


def test_build_assistant_loss_mask_require_matches_raises_for_missing_answer():
    example = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "missing"}]},
        ]
    }
    input_ids = torch.tensor([1, 2, 3, 4, 99])

    with pytest.raises(AssertionError, match="Not found valid answer"):
        build_assistant_loss_mask(
            example,
            input_ids,
            _Processor(),
            require_matches=True,
            warn_on_all_masked=False,
        )


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


def test_apply_assistant_labels_to_batch_unmask_last_token_affects_shifted_loss_mask():
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    apply_assistant_labels_to_batch(
        batch,
        examples,
        _Processor(),
        skipped_tokens=torch.tensor([]),
        unmask_last_token=True,
    )

    assert batch["loss_mask"].tolist() == [[0.0, 1.0, 1.0, 1.0, 0.0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 3, 4, 5, IGNORE_INDEX]]


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


def test_collect_media_from_conversation_handles_inline_image_and_video_payloads():
    image = object()
    video = object()
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "watch"},
                {"type": "video", "video": video},
            ],
        }
    ]

    images, videos = collect_media_from_conversation(conversation)

    assert images == [image]
    assert videos == [video]


def test_normalize_energon_vlm_sample_decodes_chatml_and_converts_images():
    sample = ChatMLSample(
        **sample_metadata_kwargs(key="sample-1", restore_key=(), subflavors={}),
        conversation=json.dumps(
            [
                {"from": "human", "value": "describe <image>"},
                {"from": "gpt", "value": "answer"},
            ]
        ),
        imgs=[torch.ones(3, 2, 2)],
    )

    normalized = normalize_energon_vlm_sample(sample)

    assert normalized.conversation == [
        {"role": "user", "content": "describe <image>"},
        {"role": "assistant", "content": "answer"},
    ]
    assert normalized.images is not None
    assert len(normalized.images) == 1
    assert normalized.images[0].size == (2, 2)
    assert normalized.videos is None


def test_normalize_hf_vlm_example_keeps_structured_conversation_and_top_level_media():
    image = object()
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "describe"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]
    example = {
        "conversation": conversation,
        "image": image,
        "audio": "audio-payload",
    }

    normalized = normalize_hf_vlm_example(example)

    assert normalized.conversation == conversation
    assert normalized.conversation is not conversation
    assert normalized.images == [image]
    assert normalized.videos is None
    assert normalized.audio == "audio-payload"


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
    normalized = normalize_hf_vlm_example({"conversation": conversation, "image": image})

    encoded = data_processor.encode_normalized(normalized)

    assert encoded.input_ids.tolist() == [1, 2, 3, 4, 5]
    assert encoded.loss_mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert encoded.labels.tolist() == [IGNORE_INDEX, 3, 4, IGNORE_INDEX, IGNORE_INDEX]
    assert encoded.visual_tensors["pixel_values"].shape == (1, 3, 4, 4)
    templated_conversation = processor.template_inputs[0][0]
    assert templated_conversation[0]["content"][1] == {"type": "image"}
    assert processor.processor_inputs[0]["images"] == [image]


def test_hf_processor_vlm_data_processor_truncates_partial_image_blocks_and_visual_tensors():
    class _ImageBlockProcessor(_Processor):
        def __call__(self, **kwargs):
            self.processor_inputs.append(kwargs)
            return {
                "input_ids": torch.tensor([[1, 10, 10, 3, 4]]),
                "pixel_values": torch.ones(1, 3, 4, 4),
            }

    processor = _ImageBlockProcessor()
    data_processor = HFProcessorVLMDataProcessor(
        processor,
        seq_length=2,
        visual_keys=("pixel_values",),
    )
    conversation = [
        {"role": "user", "content": "describe <image>"},
        {"role": "assistant", "content": "answer"},
    ]
    image = object()
    normalized = normalize_hf_vlm_example({"conversation": conversation, "image": image})

    encoded = data_processor.encode_normalized(normalized)

    assert encoded.input_ids.tolist() == [1, processor.tokenizer.pad_token_id]
    assert encoded.labels.tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert encoded.loss_mask.tolist() == [0.0, 0.0]
    assert encoded.visual_tensors == {}
