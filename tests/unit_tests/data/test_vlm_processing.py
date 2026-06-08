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
    AssistantMaskBoundaryConfig,
    NormalizedVLMSample,
    apply_assistant_labels_to_batch,
    build_assistant_loss_mask,
    build_shifted_labels_and_loss_mask,
    gather_assistant_text_segments,
    normalize_energon_vlm_sample,
    normalize_hf_vlm_example,
    normalized_vlm_sample_to_hf_example,
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


class _NonTokenizingProcessor:
    class _Tok:
        pad_token_id = 0
        eos_token_id = 99

    tokenizer = _Tok()


class _GenerationMaskTokenizer(_Tokenizer):
    chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        assert return_assistant_tokens_mask is True
        assert conversation[-1]["role"] == "assistant"
        return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 0, 1, 0]}


class _GenerationMaskProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _GenerationMaskTokenizer()


class _ChatMLTokenizer:
    pad_token_id = 0
    eos_token_id = 99
    added_tokens_decoder = {}

    _encoding = {
        "<|im_start|>": [10],
        "<|im_end|>": [11],
        "assistant": [12],
        "answer": [13],
        "answer\n": [13, 99],
        "user": [14],
        "\n": [15],
        "question": [16],
        "\nanswer": [17, 13],
    }
    _decoding = {
        10: "<|im_start|>",
        11: "<|im_end|>",
        12: "assistant",
        13: "answer",
        14: "user",
        15: "\n",
        16: "question",
        17: "\n\n",
    }

    def encode(self, text, add_special_tokens=False):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encoding.get(text, [42])}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self._decoding[token_id] for token_id in token_ids)


class _ChatMLProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _ChatMLTokenizer()


class _RoleTemplateTokenizer(_Tokenizer):
    _encoding = {
        "<user>": [20],
        "<assistant>": [21],
        "</turn>": [22],
        "\n": [23],
        "answer": [24],
        "question": [25],
        "__MEGATRON_BRIDGE_ASSISTANT_CONTENT_SENTINEL__": [26],
    }

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, return_dict=False):
        assert tokenize is True
        assert add_generation_prompt is False
        ids = []
        for turn in conversation:
            ids.extend(self._encoding[f"<{turn['role']}>"])
            ids.extend(self._encoding["\n"])
            content = turn.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) and part.get("type") == "text" else ""
                    for part in content
                )
            ids.extend(self._encoding.get(content, [42]))
            ids.extend(self._encoding["</turn>"])
        if return_dict:
            return {"input_ids": ids}
        return ids


class _RoleTemplateProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _RoleTemplateTokenizer()


def test_gather_assistant_text_segments_handles_structured_and_string_content():
    example = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}, {"type": "image"}]},
            {"role": "assistant", "content": "ok"},
        ]
    }

    assert gather_assistant_text_segments(example) == ["answer", "ok"]


def test_build_assistant_loss_mask_prefers_hf_generation_mask_when_supported():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([1, 2, 3, 4])

    mask = build_assistant_loss_mask(example, input_ids, _GenerationMaskProcessor())

    assert mask.tolist() == [0.0, 0.0, 1.0, 0.0]


def test_build_assistant_loss_mask_uses_chatml_boundaries_instead_of_user_text_search():
    example = {
        "conversation": [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor(
        [
            10,
            14,
            15,
            13,
            11,
            10,
            12,
            15,
            13,
            11,
        ]
    )

    mask = build_assistant_loss_mask(example, input_ids, _ChatMLProcessor())

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_build_assistant_loss_mask_uses_chatml_boundaries_for_multi_turn():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor(
        [
            10,
            14,
            15,
            16,
            11,
            10,
            12,
            15,
            13,
            11,
            10,
            14,
            15,
            16,
            11,
            10,
            12,
            15,
            13,
            11,
        ]
    )

    mask = build_assistant_loss_mask(example, input_ids, _ChatMLProcessor())

    assert mask.tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ]


def test_build_assistant_loss_mask_masks_chatml_terminator_for_empty_assistant():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ]
    }
    input_ids = torch.tensor([10, 14, 15, 16, 11, 10, 12, 15, 11])

    mask = build_assistant_loss_mask(example, input_ids, _ChatMLProcessor())

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def test_build_assistant_loss_mask_skips_chatml_delimiter_newlines_before_content():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "\nanswer"},
        ]
    }
    input_ids = torch.tensor([10, 14, 15, 16, 11, 10, 12, 17, 13, 11])

    mask = build_assistant_loss_mask(example, input_ids, _ChatMLProcessor())

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_build_assistant_loss_mask_text_fallback_does_not_search_newline_for_empty_assistant():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": ""},
        ]
    }
    input_ids = torch.tensor([1, 42, 2, 3])

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), warn_on_all_masked=False)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_build_assistant_loss_mask_handles_non_tokenizing_tokenizer():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([1, 2, 3])

    mask = build_assistant_loss_mask(example, input_ids, _NonTokenizingProcessor(), warn_on_all_masked=False)

    assert mask.tolist() == [0.0, 0.0, 0.0]


def test_build_assistant_loss_mask_uses_template_diff_for_non_chatml_same_text():
    example = {
        "conversation": [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([20, 23, 24, 22, 21, 23, 24, 22])

    mask = build_assistant_loss_mask(example, input_ids, _RoleTemplateProcessor())

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_build_assistant_loss_mask_template_diff_handles_structured_content():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }
    input_ids = torch.tensor([20, 23, 25, 22, 21, 23, 24, 22])

    mask = build_assistant_loss_mask(example, input_ids, _RoleTemplateProcessor())

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]


def test_build_assistant_loss_mask_uses_explicit_boundary_config_before_text_search():
    example = {
        "conversation": [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 4, 101, 102, 3, 4, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"user": [100], "assistant": [102]},
        end_tokens=[101],
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_splits_nested_parts():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 8, 101, 102, 200, 30, 31, 201, 32, 202, 40, 41, 203, 33, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"user": [100], "assistant": [102]},
        end_tokens=[101],
        masked_roles=("assistant", "tool_call"),
        part_start_tokens={"reasoning": [200], "tool_call": [202]},
        part_end_tokens={"reasoning": [201], "tool_call": [203]},
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_splits_parts_only_in_assistant_by_default():
    example = {
        "conversation": [
            {"role": "system", "content": "tool schema"},
            {"role": "assistant", "content": "tool call"},
        ]
    }
    input_ids = torch.tensor([99, 202, 50, 203, 101, 102, 202, 60, 203, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"system": [99], "assistant": [102]},
        end_tokens=[101],
        masked_roles=("tool_call",),
        part_start_tokens={"tool_call": [202]},
        part_end_tokens={"tool_call": [203]},
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_can_match_omni_whole_assistant_message():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "tool call"},
        ]
    }
    input_ids = torch.tensor([100, 8, 101, 102, 202, 60, 203, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"user": [100], "assistant": [102]},
        end_tokens=[101],
        masked_roles=("assistant",),
        include_start_tokens_for_roles=("assistant",),
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_can_include_part_boundary_tokens():
    example = {
        "conversation": [
            {"role": "assistant", "content": "tool call"},
        ]
    }
    input_ids = torch.tensor([102, 202, 60, 203, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"assistant": [102]},
        end_tokens=[101],
        masked_roles=("tool_call",),
        part_start_tokens={"tool_call": [202]},
        part_end_tokens={"tool_call": [203]},
        include_start_tokens_for_parts=("tool_call",),
        include_end_tokens_for_parts=("tool_call",),
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_trims_leading_delimiters():
    example = {
        "conversation": [
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([102, 55, 3, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"assistant": [102]},
        end_tokens=[101],
        trim_leading_token_ids=(55,),
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]


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


def test_normalized_vlm_sample_to_hf_example_expands_placeholders_and_threads_media():
    image = object()
    video = object()
    sample = NormalizedVLMSample(
        conversation=[
            {"role": "user", "content": "before <image> between <video> after"},
            {"role": "assistant", "content": "answer"},
        ],
        images=[image],
        videos=[video],
        audio="audio-payload",
    )

    example = normalized_vlm_sample_to_hf_example(sample)

    assert example["conversation"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image", "image": image},
                {"type": "text", "text": "between"},
                {"type": "video", "video": video},
                {"type": "text", "text": "after"},
            ],
        },
        {"role": "assistant", "content": "answer"},
    ]
    assert example["images"] == [image]
    assert example["videos"] == [video]
    assert example["audio"] == "audio-payload"


def test_normalized_vlm_sample_to_hf_example_can_emit_media_first_content():
    image = object()
    sample = NormalizedVLMSample(
        conversation=[{"role": "user", "content": "describe <image>"}],
        images=[image],
    )

    example = normalized_vlm_sample_to_hf_example(sample, media_first=True)

    assert example["conversation"][0]["content"] == [
        {"type": "image", "image": image},
        {"type": "text", "text": "describe"},
    ]


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
