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

from megatron.bridge.data.collators.sft import text_chat_collate_fn
from megatron.bridge.data.conversation_processing import (
    _MAX_SCANNED_REPR_CHARS,
    AssistantMaskBoundaryConfig,
    NormalizedVLMSample,
    _conversation_contains_boundary_tokens,
    _value_contains_token_sequence,
    apply_assistant_labels_to_batch,
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    build_shifted_labels_and_loss_mask,
    chat_template_kwargs_from_example,
    gather_assistant_text_segments,
    get_processor_tokenizer,
    infer_assistant_mask_boundary_config,
    normalize_chat_conversation,
    normalize_energon_vlm_sample,
    normalize_hf_vlm_example,
    normalized_vlm_sample_to_hf_example,
    shared_chat_template_kwargs_from_examples,
    tokenize_chat_example,
)
from megatron.bridge.data.datasets.gpt_sft import GPTSFTChatDataset
from megatron.bridge.data.datasets.utils import IGNORE_INDEX, _chat_preprocess
from megatron.bridge.data.energon.metadata import sample_metadata_kwargs
from megatron.bridge.data.energon.task_encoder_utils import ChatMLSample


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


def test_get_processor_tokenizer_unwraps_megatron_layers_but_keeps_hf_backend_private():
    class RawHFTokenizer:
        added_tokens_decoder = {}

        def __init__(self):
            self._tokenizer = object()

        def __call__(self, text, **kwargs):
            return {"input_ids": [1, 2, 3]}

    raw_tokenizer = RawHFTokenizer()

    class MegatronHFTokenizerWrapper:
        tokenizer = raw_tokenizer

    class MegatronTokenizerTextWrapper:
        _tokenizer = MegatronHFTokenizerWrapper()

    assert get_processor_tokenizer(MegatronTokenizerTextWrapper()) is raw_tokenizer


def test_get_processor_tokenizer_does_not_probe_dynamic_wrapper_attributes():
    class DynamicWrapper:
        def __init__(self, tokenizer):
            self._tokenizer = tokenizer

        def __getattr__(self, name):
            raise AssertionError(f"unexpected dynamic attribute probe: {name}")

    raw_tokenizer = _Tokenizer()

    assert get_processor_tokenizer(DynamicWrapper(raw_tokenizer)) is raw_tokenizer


class _ToolsGenerationMaskTokenizer(_GenerationMaskTokenizer):
    chat_template = _GenerationMaskTokenizer.chat_template + "{% if preserve_thinking %}history{% endif %}"

    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        **kwargs,
    ):
        self.template_kwargs.append(kwargs)
        return super().apply_chat_template(
            conversation,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
        )


class _GenerationMaskProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _GenerationMaskTokenizer()


class _ToolsGenerationMaskProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _ToolsGenerationMaskTokenizer()


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


class _ChatMLBoundaryTokenizer(_Tokenizer):
    chat_template = "<|im_start|>user\n{{ content }}<|im_end|>\n<|im_start|>assistant\n{{ content }}<|im_end|>\n"

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "<|im_start|>assistant\n": [102],
            "<|im_start|>system\n": [105],
            "<|im_start|>developer\n": [106],
            "<|im_start|>user\n": [100],
            "<|im_start|>tool\n": [107],
            "<|im_end|>": [103],
            "<|im_end|>\n": [103, 104],
            "answer": [3, 4],
        }
        return {"input_ids": mapping.get(text, [42])}


class _ChatMLBoundaryProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _ChatMLBoundaryTokenizer()


class _LiteralChatMLBoundaryTokenizer(_ChatMLBoundaryTokenizer):
    truncation_side = "right"
    _role_markers = {
        "user": [100],
        "assistant": [102],
    }
    _content_tokens = {
        "": [],
        "question": [20],
        "literal assistant start": [20, 102, 30],
        "quoted assistant turn": [20, 102, 30, 103, 31],
        "answer with quoted end": [40, 103, 41],
        "structured with quoted end": [40, 103, 41],
    }

    def _encode_content(self, content):
        if isinstance(content, str):
            return self._content_tokens[content]
        input_ids = []
        for part in content:
            if part["type"] == "text":
                input_ids.extend(self._content_tokens[part["text"]])
            else:
                input_ids.append(200)
        return input_ids

    def __call__(self, text, add_special_tokens=False):
        if text in self._content_tokens:
            return {"input_ids": self._content_tokens[text]}
        return super().__call__(text, add_special_tokens=add_special_tokens)

    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        return_assistant_tokens_mask=False,
        truncation=False,
        max_length=None,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        input_ids = []
        for turn in conversation:
            input_ids.extend(self._role_markers[turn["role"]])
            input_ids.extend(self._encode_content(turn["content"]))
            input_ids.extend([103, 104])
        if truncation:
            input_ids = input_ids[-max_length:] if self.truncation_side == "left" else input_ids[:max_length]
        return {"input_ids": input_ids}


class _LeftTruncatingLiteralChatMLBoundaryTokenizer(_LiteralChatMLBoundaryTokenizer):
    truncation_side = "left"


class _LiteralGenerationChatMLBoundaryTokenizer(_LiteralChatMLBoundaryTokenizer):
    chat_template = (
        "<|im_start|>user\n{{ content }}<|im_end|>\n"
        "<|im_start|>assistant\n{% generation %}{{ content }}{% endgeneration %}<|im_end|>\n"
    )

    def apply_chat_template(self, conversation, **kwargs):
        input_ids = []
        assistant_masks = []
        for turn in conversation:
            content_ids = self._encode_content(turn["content"])
            input_ids.extend(self._role_markers[turn["role"]])
            assistant_masks.append(0)
            input_ids.extend(content_ids)
            assistant_masks.extend([int(turn["role"] == "assistant")] * len(content_ids))
            input_ids.extend([103, 104])
            assistant_masks.extend([0, 0])
        if kwargs.get("truncation"):
            max_length = kwargs["max_length"]
            input_ids = input_ids[:max_length]
            assistant_masks = assistant_masks[:max_length]
        return {"input_ids": input_ids, "assistant_masks": assistant_masks}


class _PrefixUnsupportedLiteralChatMLBoundaryTokenizer(_LiteralChatMLBoundaryTokenizer):
    def apply_chat_template(self, conversation, **kwargs):
        if len(conversation) != 2:
            raise ValueError("prefix rendering is unavailable")
        return {"input_ids": [100, 20, 102, 30, 103, 104]}


class _PrefixUnsupportedLiteralGenerationChatMLBoundaryTokenizer(_LiteralGenerationChatMLBoundaryTokenizer):
    def apply_chat_template(self, conversation, **kwargs):
        if len(conversation) != 2:
            raise ValueError("prefix rendering is unavailable")
        return super().apply_chat_template(conversation, **kwargs)


class _MoonlightBoundaryTokenizer(_Tokenizer):
    chat_template = (
        "{%- for message in messages -%}"
        "{%- if message['role'] == 'system' -%}<|im_system|>{%- endif -%}"
        "{%- if message['role'] == 'user' -%}<|im_user|>{%- endif -%}"
        "{%- if message['role'] == 'assistant' -%}<|im_assistant|>{%- endif -%}"
        "{{ message['role'] }}<|im_middle|>{{ message['content'] }}<|im_end|>"
        "{%- endfor -%}"
    )

    _role_markers = {
        "system": [405, 415, 401],
        "user": [400, 414, 401],
        "assistant": [402, 412, 401],
    }

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "<|im_system|>system<|im_middle|>": self._role_markers["system"],
            "<|im_user|>user<|im_middle|>": self._role_markers["user"],
            "<|im_assistant|>assistant<|im_middle|>": self._role_markers["assistant"],
            "<|im_end|>": [403],
            "question": [16],
            "answer": [3, 4],
        }
        return {"input_ids": mapping.get(text, [42])}

    def apply_chat_template(self, conversation, tokenize=True, add_generation_prompt=False, return_dict=False):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        input_ids = []
        for turn in conversation:
            input_ids.extend(self._role_markers[turn["role"]])
            input_ids.extend(self(turn["content"])["input_ids"])
            input_ids.append(403)
        return {"input_ids": input_ids}


class _MoonlightBoundaryProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _MoonlightBoundaryTokenizer()


class _ProcessorTemplateBoundaryProcessor(_ChatMLBoundaryProcessor):
    chat_template = "<|turn>model\n{{ content }}<turn|>"

    class _Tok(_Tokenizer):
        chat_template = ""

        def __call__(self, text, add_special_tokens=False):
            mapping = {
                "<|turn>model\n": [202],
                "<turn|>": [203],
                "answer": [3, 4],
            }
            return {"input_ids": mapping.get(text, [42])}

    def __init__(self):
        super().__init__()
        self.tokenizer = self._Tok()


class _JinjaSeparatedChatMLBoundaryProcessor(_ChatMLBoundaryProcessor):
    class _Tok(_ChatMLBoundaryTokenizer):
        chat_template = "<|im_start|>assistant\n{{ content }}<|im_end|>{% if not loop.last %}{{ '\\n' }}{% endif %}"

    def __init__(self):
        super().__init__()
        self.tokenizer = self._Tok()


class _LlamaBoundaryProcessor(_ChatMLBoundaryProcessor):
    class _Tok(_ChatMLBoundaryTokenizer):
        chat_template = (
            "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' }}"
            "{{ message['content'] }}{{ '<|eot_id|>' }}"
        )

        def __call__(self, text, add_special_tokens=False):
            mapping = {
                "<|start_header_id|>assistant<|end_header_id|>\n\n": [302],
                "<|start_header_id|>system<|end_header_id|>\n\n": [305],
                "<|start_header_id|>developer<|end_header_id|>\n\n": [306],
                "<|start_header_id|>user<|end_header_id|>\n\n": [300],
                "<|start_header_id|>tool<|end_header_id|>\n\n": [307],
                "<|eot_id|>": [303],
            }
            return {"input_ids": mapping.get(text, [42])}

    def __init__(self):
        super().__init__()
        self.tokenizer = self._Tok()


class _LlamaPreprocessingTokenizer(_LlamaBoundaryProcessor._Tok):
    def apply_chat_template(self, conversation, tokenize=True, **kwargs):
        assert tokenize is True
        if [turn["role"] for turn in conversation] == ["user"]:
            return {"input_ids": [300, 42, 303]}
        assert [turn["role"] for turn in conversation] == ["user", "assistant"]
        return {"input_ids": [300, 42, 303, 302, 42, 303]}


class _ZeroGenerationMaskTokenizer(_ChatMLBoundaryTokenizer):
    chat_template = (
        "<|im_start|>user\n{{ content }}<|im_end|>\n"
        "<|im_start|>assistant\n{% generation %}{{ content }}<|im_end|>\n{% endgeneration %}"
    )

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
        return {
            "input_ids": [100, 3, 103, 104, 102, 3, 4, 103, 104],
            "assistant_masks": [0] * 9,
        }


class _ZeroGenerationMaskProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _ZeroGenerationMaskTokenizer()


class _ContentOnlyGenerationMaskTokenizer(_ZeroGenerationMaskTokenizer):
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
        return {
            "input_ids": [100, 3, 103, 104, 102, 3, 4, 103, 104],
            "assistant_masks": [0, 0, 0, 0, 0, 1, 1, 0, 0],
        }


class _ContentOnlyGenerationMaskProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _ContentOnlyGenerationMaskTokenizer()


class _TruncatedZeroGenerationMaskTokenizer(_ZeroGenerationMaskTokenizer):
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
        return {"input_ids": [100, 3], "assistant_masks": [0, 0]}


class _TruncatedZeroGenerationMaskProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _TruncatedZeroGenerationMaskTokenizer()


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


@pytest.mark.parametrize(
    ("input_ids", "expected_mask"),
    [
        (torch.tensor([1, 2, 3, 4, 0, 0]), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        (torch.tensor([0, 0, 1, 2, 3, 4]), [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ],
)
def test_build_assistant_loss_mask_aligns_hf_generation_mask_to_batch_padding(input_ids, expected_mask):
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    mask = build_assistant_loss_mask(example, input_ids, _GenerationMaskProcessor())

    assert mask.tolist() == expected_mask


@pytest.mark.parametrize(
    ("truncate_history_thinking", "native_preserve_thinking"),
    [(True, False), (False, True)],
)
def test_build_assistant_loss_mask_adapts_history_thinking_kwarg_to_native_template(
    truncate_history_thinking,
    native_preserve_thinking,
):
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        "chat_template_kwargs": {
            "enable_thinking": True,
            "truncate_history_thinking": truncate_history_thinking,
        },
        "tools": tools,
    }

    processor = _ToolsGenerationMaskProcessor()

    mask = build_assistant_loss_mask(example, torch.tensor([1, 2, 3, 4]), processor)

    assert mask.tolist() == [0.0, 0.0, 1.0, 0.0]
    assert processor.tokenizer.template_kwargs == [
        {"enable_thinking": True, "preserve_thinking": native_preserve_thinking, "tools": tools}
    ]


def test_chat_template_kwargs_from_example_rejects_invalid_or_pipeline_controlled_values():
    with pytest.raises(ValueError, match="string-keyed mapping"):
        chat_template_kwargs_from_example({"chat_template_kwargs": ["truncate_history_thinking"]})
    with pytest.raises(ValueError, match="pipeline-controlled arguments: tokenize, tools"):
        chat_template_kwargs_from_example({"chat_template_kwargs": {"tokenize": False, "tools": []}})
    with pytest.raises(ValueError, match="truncate_history_thinking must be a boolean"):
        chat_template_kwargs_from_example({"chat_template_kwargs": {"truncate_history_thinking": "yes"}})


def test_shared_chat_template_kwargs_from_examples_requires_shared_values():
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    kwargs = {"truncate_history_thinking": False}

    assert shared_chat_template_kwargs_from_examples(
        [
            {"chat_template_kwargs": kwargs, "tools": tools},
            {"chat_template_kwargs": kwargs, "tools": tools},
        ]
    ) == {"truncate_history_thinking": False, "tools": tools}
    with pytest.raises(ValueError, match="same chat-template kwargs and tools"):
        shared_chat_template_kwargs_from_examples(
            [
                {"chat_template_kwargs": {"truncate_history_thinking": True}, "tools": tools},
                {"chat_template_kwargs": {"truncate_history_thinking": False}, "tools": tools},
            ]
        )


def test_build_assistant_loss_mask_raises_without_template_or_boundary_config():
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

    with pytest.raises(ValueError, match="Unable to build assistant loss mask"):
        build_assistant_loss_mask(example, input_ids, _ChatMLProcessor())


def test_infer_assistant_mask_boundary_config_from_chatml_template():
    boundary_config = infer_assistant_mask_boundary_config(_ChatMLBoundaryProcessor())

    assert boundary_config is not None
    assert boundary_config.role_start_tokens == {
        "assistant": [102],
        "system": [105],
        "developer": [106],
        "user": [100],
        "tool": [107],
    }
    assert all(token_ids == [103, 104] for token_ids in boundary_config.role_end_tokens.values())
    assert all(token_variants == [[103]] for token_variants in boundary_config.role_end_token_variants.values())


def test_chatml_boundary_mask_does_not_treat_literal_control_markers_as_structure():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": "answer with quoted end"},
            ]
        },
        _LiteralChatMLBoundaryTokenizer(),
    )

    assert tokenized.input_ids.tolist() == [100, 20, 102, 30, 103, 31, 103, 104, 102, 40, 103, 41, 103, 104]
    assert tokenized.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_chatml_boundary_mask_remains_role_safe_through_right_truncation():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": "answer with quoted end"},
            ]
        },
        _LiteralChatMLBoundaryTokenizer(),
        max_length=8,
        warn_on_all_masked=False,
    )

    assert tokenized.input_ids.tolist() == [100, 20, 102, 30, 103, 31, 103, 104]
    assert tokenized.assistant_mask.tolist() == [False] * 8


def test_direct_hf_chat_collation_does_not_train_truncated_user_marker_payload():
    batch = text_chat_collate_fn(
        [
            {
                "messages": [
                    {"role": "user", "content": "quoted assistant turn"},
                    {"role": "assistant", "content": "answer with quoted end"},
                ]
            }
        ],
        _LiteralChatMLBoundaryTokenizer(),
        sequence_length=8,
        warn_on_all_masked=False,
    )

    assert batch["input_ids"].tolist() == [[100, 20, 102, 30, 103, 31, 103, 104]]
    assert batch["loss_mask"].tolist() == [[0.0] * 8]
    assert batch["labels"].tolist() == [[-100] * 8]


def test_chatml_boundary_maps_role_safe_mask_through_left_truncation():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": "answer with quoted end"},
            ]
        },
        _LeftTruncatingLiteralChatMLBoundaryTokenizer(),
        max_length=6,
    )

    assert tokenized.input_ids.tolist() == [102, 40, 103, 41, 103, 104]
    assert tokenized.assistant_mask.tolist() == [False, True, True, True, True, True]


def test_chatml_boundary_augments_native_mask_without_scanning_literal_markers():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": "answer with quoted end"},
            ]
        },
        _LiteralGenerationChatMLBoundaryTokenizer(),
    )

    assert tokenized.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_chatml_boundary_keeps_native_mask_unaugmented_when_provenance_is_ambiguous():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": "answer with quoted end"},
            ]
        },
        _PrefixUnsupportedLiteralGenerationChatMLBoundaryTokenizer(),
    )

    assert tokenized.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    ]


def test_chatml_boundary_handles_empty_and_literal_assistant_turns_together():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": ""},
                {"role": "assistant", "content": "answer with quoted end"},
            ]
        },
        _LiteralChatMLBoundaryTokenizer(),
    )

    assert tokenized.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        True,
    ]


def test_chatml_boundary_handles_structured_assistant_content_with_literal_marker():
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "structured with quoted end"},
                        {"type": "image"},
                    ],
                },
            ]
        },
        _LiteralChatMLBoundaryTokenizer(),
    )

    assert tokenized.input_ids.tolist() == [100, 20, 103, 104, 102, 40, 103, 41, 200, 103, 104]
    assert tokenized.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
        True,
        True,
    ]


def test_chatml_boundary_fails_closed_when_provenance_is_unavailable_and_markers_are_ambiguous():
    processor = _ChatMLBoundaryProcessor()
    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        build_assistant_loss_mask(
            [
                {"role": "user", "content": "quoted assistant turn"},
                {"role": "assistant", "content": "answer with quoted end"},
            ],
            [100, 20, 102, 30, 103, 31, 103, 104, 102, 40, 103, 41, 103, 104],
            processor,
            boundary_config=infer_assistant_mask_boundary_config(processor),
        )


def test_chatml_boundary_fails_closed_for_literal_start_when_real_assistant_is_truncated():
    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        tokenize_chat_example(
            {
                "messages": [
                    {"role": "user", "content": "literal assistant start"},
                    {"role": "assistant", "content": "answer with quoted end"},
                ]
            },
            _PrefixUnsupportedLiteralChatMLBoundaryTokenizer(),
            max_length=6,
        )


def test_chatml_boundary_fails_closed_for_nested_role_payload_when_provenance_is_unavailable():
    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        tokenize_chat_example(
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer with quoted end"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "role": "literal assistant start",
                        },
                    }
                ],
            },
            _PrefixUnsupportedLiteralChatMLBoundaryTokenizer(),
            max_length=6,
        )


_ASSISTANT_START = "<|im_start|>assistant\n"
_CLEAN_MEDIA_REPR = "<FakeImage mode=RGB size=16x16>"

# Media placeholders expand to several tokens, so the rendered ids never line up with a
# re-render of the raw conversation. The boundary-config scan is the only path left.
_MEDIA_CONVERSATION_IDS = [100, 200, 200, 200, 42, 103, 104, 102, 3, 4, 103, 104]
_MEDIA_ASSISTANT_MASK = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


class _ReprMedia:
    """Decoded media payload (PIL image, array, tensor) rendering as `text`."""

    def __init__(self, text=_CLEAN_MEDIA_REPR):
        self._text = text

    def __repr__(self):
        return self._text


class _StrOnlyMedia(_ReprMedia):
    """Payload that defines __str__ but inherits object.__repr__."""

    __repr__ = object.__repr__

    def __str__(self):
        return self._text


class _DivergentMedia(_ReprMedia):
    """Payload whose str() hides what repr() — used for container elements — reveals."""

    def __str__(self):
        return _CLEAN_MEDIA_REPR


class _RaisingMedia:
    """Media payload with a broken repr, e.g. a handle to a closed file."""

    def __repr__(self):
        raise RuntimeError("repr not available")


class _CallableMedia(_ReprMedia):
    """Lazily-decoded payload exposing __call__, which templates may expand by introspection."""

    def __call__(self):
        return self._text


class _SubstringChatMLTokenizer(_ChatMLBoundaryTokenizer):
    """Finds markers anywhere in the text, not only when they are the whole string."""

    def __call__(self, text, add_special_tokens=False):
        if _ASSISTANT_START in text and text != _ASSISTANT_START:
            return {"input_ids": [42, 102, 42]}
        return super().__call__(text, add_special_tokens=add_special_tokens)


def _media_conversation(*media, assistant_media=()):
    user_content = [{"type": "image", "image": item} for item in media]
    user_content.append({"type": "text", "text": "question"})
    assistant_content = [{"type": "image", "image": item} for item in assistant_media]
    assistant_content.append({"type": "text", "text": "answer"})
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _boundary_scan(example, processor):
    return _conversation_contains_boundary_tokens(
        example,
        processor.tokenizer,
        infer_assistant_mask_boundary_config(processor),
    )


def test_chatml_boundary_scans_conversations_carrying_media_payloads():
    processor = _ChatMLBoundaryProcessor()
    example = {"conversation": _media_conversation(_ReprMedia())}

    # False, not None: a media repr that carries no marker is known-clean provenance, which is
    # what re-enables the boundary-config fallback.
    assert _boundary_scan(example, processor) is False

    mask = build_assistant_loss_mask(
        example,
        _MEDIA_CONVERSATION_IDS,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
    )

    assert mask.tolist() == _MEDIA_ASSISTANT_MASK


@pytest.mark.parametrize("media_field", ["media", "assistant_media"])
def test_chatml_boundary_fails_closed_when_media_payload_renders_a_literal_marker(media_field):
    processor = _ChatMLBoundaryProcessor()
    marker_media = _ReprMedia(_ASSISTANT_START)
    example = {
        "conversation": (
            _media_conversation(marker_media)
            if media_field == "media"
            else _media_conversation(assistant_media=[marker_media])
        )
    }

    assert _boundary_scan(example, processor) is True

    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        build_assistant_loss_mask(
            example,
            _MEDIA_CONVERSATION_IDS,
            processor,
            boundary_config=infer_assistant_mask_boundary_config(processor),
        )


@pytest.mark.parametrize("sibling", [_ReprMedia(), object()], ids=["clean", "unknown"])
def test_chatml_boundary_scan_reports_a_marker_beside_other_media(sibling):
    # A detected marker outranks both a clean sibling and an unscannable one.
    example = {"conversation": _media_conversation(sibling, _ReprMedia(_ASSISTANT_START))}

    assert _boundary_scan(example, _ChatMLBoundaryProcessor()) is True


@pytest.mark.parametrize(
    "media",
    [_RaisingMedia(), _ReprMedia("x" * (_MAX_SCANNED_REPR_CHARS + 1)), object(), _CallableMedia()],
    ids=["raising", "oversized", "opaque", "callable"],
)
def test_chatml_boundary_fails_closed_for_unrenderable_media_without_propagating(media):
    processor = _ChatMLBoundaryProcessor()
    example = {"conversation": _media_conversation(media)}

    assert _boundary_scan(example, processor) is None

    # The collator recovers from ValueError; anything else escapes and kills the dataloader.
    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        build_assistant_loss_mask(
            example,
            _MEDIA_CONVERSATION_IDS,
            processor,
            boundary_config=infer_assistant_mask_boundary_config(processor),
        )


def test_chatml_boundary_fails_closed_for_callable_template_kwargs():
    def lookup(city):
        """Templates expand tools by introspection, so str() hides <|im_start|>assistant."""

    example = {"conversation": _media_conversation(_ReprMedia()), "tools": [lookup]}

    assert _boundary_scan(example, _ChatMLBoundaryProcessor()) is None


def test_chatml_boundary_scan_is_unchanged_for_text_only_conversations():
    processor = _ChatMLBoundaryProcessor()
    clean = {"conversation": [{"role": "user", "content": "question"}]}
    marked = {"conversation": [{"role": "user", "content": _ASSISTANT_START}]}

    assert _boundary_scan(clean, processor) is False
    assert _boundary_scan(marked, processor) is True


@pytest.mark.parametrize(
    "value,expected",
    [
        # Untouched branches, pinned so the media branch cannot bleed into them.
        (None, False),
        ("question", False),
        ({"text": _ASSISTANT_START}, True),
        # Buffers render as an escaped literal, so a marker inside one is undetectable.
        (b"raw", None),
        (bytearray(b"raw"), None),
        (memoryview(b"raw"), None),
        # Media payloads: cleared, flagged, or failed closed.
        (_ReprMedia(), False),
        (_ReprMedia(_ASSISTANT_START), True),
        (_ReprMedia(f"<Image path='/tmp/{_ASSISTANT_START}.png'>"), True),
        (_StrOnlyMedia(), False),
        (_StrOnlyMedia(_ASSISTANT_START), True),
        (_DivergentMedia(_ASSISTANT_START), True),
        (_ReprMedia("x" * _MAX_SCANNED_REPR_CHARS), False),
        (_ReprMedia("x" * (_MAX_SCANNED_REPR_CHARS + 1)), None),
        (_RaisingMedia(), None),
        (_CallableMedia(), None),
        (object(), None),
        (len, None),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, _ReprMedia) else None,
)
def test_value_contains_token_sequence_classifies_payload_leaves(value, expected):
    # Substring-aware so a marker embedded in a longer repr is detected, as in the real scan.
    tokenizer = _SubstringChatMLTokenizer()

    assert _value_contains_token_sequence(value, tokenizer, [[102]]) is expected


def test_infer_assistant_mask_boundary_config_from_moonlight_template():
    processor = _MoonlightBoundaryProcessor()
    assert "<|im_assistant|>assistant<|im_middle|>" not in processor.tokenizer.chat_template
    boundary_config = infer_assistant_mask_boundary_config(processor)

    assert boundary_config is not None
    assert boundary_config.role_start_tokens == {
        "assistant": [402, 412, 401],
        "system": [405, 415, 401],
        "user": [400, 414, 401],
    }
    assert all(token_ids == [403] for token_ids in boundary_config.role_end_tokens.values())

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        processor,
    )
    assert tokenized.input_ids.tolist() == [400, 414, 401, 16, 403, 402, 412, 401, 3, 4, 403]
    assert tokenized.assistant_mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_infer_assistant_mask_boundary_config_handles_jinja_separated_chatml_newline():
    boundary_config = infer_assistant_mask_boundary_config(_JinjaSeparatedChatMLBoundaryProcessor())

    assert boundary_config is not None
    assert boundary_config.role_end_tokens["assistant"] == [103, 104]


def test_infer_assistant_mask_boundary_config_from_llama_template():
    boundary_config = infer_assistant_mask_boundary_config(_LlamaBoundaryProcessor())

    assert boundary_config is not None
    assert boundary_config.role_start_tokens == {
        "assistant": [302],
        "system": [305],
        "developer": [306],
        "user": [300],
        "tool": [307],
    }
    assert all(token_ids == [303] for token_ids in boundary_config.role_end_tokens.values())


class _HarmonyBoundaryTokenizer(_Tokenizer):
    """gpt-oss / OpenAI Harmony tokenizer double.

    A single assistant turn renders as two channel segments (``analysis`` then
    ``final``), each wrapped in its own ``<|start|>assistant<|channel|>...
    <|message|>...`` header. ``<|start|>`` and the role are emitted separately,
    so the literal string ``<|start|>assistant`` never appears in the template.
    """

    chat_template = (
        "{% for message in messages %}"
        "<|start|>{{ message.role }}<|channel|>{{ channel }}<|message|>{{ message.content }}<|end|>"
        "{% endfor %}<|return|>"
    )

    _role_ids = {"assistant": 210, "user": 211, "system": 212, "developer": 213, "tool": 214}

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "<|start|>assistant": [200, 210],
            "<|start|>user": [200, 211],
            "<|start|>system": [200, 212],
            "<|start|>developer": [200, 213],
            "<|start|>tool": [200, 214],
            "<|return|>": [204],
            "<|end|>": [203],
            "<|call|>": [205],
            "question": [20],
            "reasoning": [30, 31],
            "answer": [40, 41],
        }
        return {"input_ids": mapping.get(text, [42])}

    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        input_ids = []
        for turn in conversation:
            role_id = self._role_ids[turn["role"]]
            content_ids = self(turn["content"])["input_ids"]
            if turn["role"] == "assistant":
                # Chain-of-thought in the analysis channel, then the user-facing final channel.
                input_ids.extend([200, role_id, 201, 220, 202] + self("reasoning")["input_ids"] + [203])
                input_ids.extend([200, role_id, 201, 221, 202] + content_ids + [204])
            else:
                input_ids.extend([200, role_id, 202] + content_ids + [203])
        return {"input_ids": input_ids}


class _HarmonyBoundaryProcessor(_Processor):
    def __init__(self):
        super().__init__()
        self.tokenizer = _HarmonyBoundaryTokenizer()


def test_infer_assistant_mask_boundary_config_from_gpt_oss_harmony_template():
    processor = _HarmonyBoundaryProcessor()
    # The concatenated assistant header is not a literal substring of the template;
    # detection must rely on the standalone structural specials.
    assert "<|start|>assistant" not in processor.tokenizer.chat_template
    boundary_config = infer_assistant_mask_boundary_config(processor)

    assert boundary_config is not None
    # Start markers stop at the role header so the loss span begins at <|channel|>,
    # covering the analysis channel as well as the final channel.
    assert boundary_config.role_start_tokens == {
        "assistant": [200, 210],
        "system": [200, 212],
        "developer": [200, 213],
        "user": [200, 211],
        "tool": [200, 214],
    }
    assert all(token_ids == [204] for token_ids in boundary_config.role_end_tokens.values())
    # <|end|> (message) and <|call|> (tool call) terminators back up the <|return|> primary.
    assert all(variants == [[203], [205]] for variants in boundary_config.role_end_token_variants.values())


def test_gpt_oss_harmony_assistant_mask_covers_analysis_and_final_channels():
    processor = _HarmonyBoundaryProcessor()
    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        processor,
    )

    user_block = [200, 211, 202, 20, 203]  # <|start|>user<|message|>question<|end|>
    analysis_block = [200, 210, 201, 220, 202, 30, 31, 203]  # ...<|channel|>analysis<|message|>reasoning<|end|>
    final_block = [200, 210, 201, 221, 202, 40, 41, 204]  # ...<|channel|>final<|message|>answer<|return|>
    assert tokenized.input_ids.tolist() == user_block + analysis_block + final_block

    mask = tokenized.assistant_mask.tolist()
    assert len(mask) == len(user_block + analysis_block + final_block)

    # The user turn and the assistant's priming role header (<|start|>assistant) carry no loss.
    priming_header_len = 2
    assert not any(mask[: len(user_block) + priming_header_len])
    # Everything from the first <|channel|> through the final <|return|> is trained, so the
    # analysis chain-of-thought and the final answer both contribute to the loss.
    assert all(mask[len(user_block) + priming_header_len :])

    # Pin the reasoning (analysis channel) and answer (final channel) token positions so a
    # regression that drops the analysis channel from the loss fails loudly here.
    reasoning_positions = [len(user_block) + 5, len(user_block) + 6]
    answer_positions = [len(user_block) + len(analysis_block) + 5, len(user_block) + len(analysis_block) + 6]
    assert tokenized.input_ids[reasoning_positions].tolist() == [30, 31]
    assert tokenized.input_ids[answer_positions].tolist() == [40, 41]
    assert all(mask[position] for position in reasoning_positions + answer_positions)


class _Gemma4ToolCallBoundaryTokenizer(_Tokenizer):
    chat_template = (
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
        "{% for message in messages %}<|turn>{{ message.role }}{{ message.content }}"
        "{% if message.tool_calls %}<|tool_call>{{ message.tool_calls }}<tool_call|><|tool_response>{% endif %}"
        "<turn|>{% endfor %}"
    )

    def __call__(self, text, add_special_tokens=False):
        mapping = {
            "<|turn>model\n": [10],
            "<turn|>": [11],
            "<|tool_response>": [12],
            "<tool_response|>": [13],
        }
        return {"input_ids": mapping.get(text, [42])}

    def apply_chat_template(
        self,
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=False,
        **kwargs,
    ):
        assert tokenize is True
        assert add_generation_prompt is False
        assert return_dict is True
        input_ids = []
        previous_non_tool_role = None
        for turn_index, turn in enumerate(conversation):
            role = turn["role"]
            if role == "user":
                input_ids.extend([20, 21, 11])
            elif role == "assistant":
                if previous_non_tool_role != "assistant":
                    input_ids.append(10)
                if turn.get("tool_calls"):
                    input_ids.append(30)
                    following_turn = conversation[turn_index + 1] if turn_index + 1 < len(conversation) else None
                    if following_turn is not None and following_turn["role"] == "tool":
                        input_ids.extend([12, 40, 13])
                    else:
                        input_ids.append(12)
                if turn.get("content"):
                    input_ids.append(50)
                if not turn.get("tool_calls"):
                    input_ids.append(11)
                previous_non_tool_role = role
        return {"input_ids": input_ids}


@pytest.mark.parametrize(
    ("trailing_messages", "expected_input_ids", "expected_mask", "expected_final_start"),
    [
        ([], [20, 21, 11, 10, 30, 12], [False, False, False, False, True, True], 4),
        (
            [
                {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature":"72F"}'},
                {"role": "assistant", "content": "It is 72F."},
            ],
            [20, 21, 11, 10, 30, 12, 40, 13, 50, 11],
            [False, False, False, False, True, True, False, False, True, True],
            8,
        ),
    ],
)
def test_gemma4_tool_call_assistant_mask_and_generation_boundary(
    trailing_messages,
    expected_input_ids,
    expected_mask,
    expected_final_start,
):
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"city":"Denver"}'},
                }
            ],
        },
        *trailing_messages,
    ]

    tokenized = tokenize_chat_example(
        {"messages": messages},
        _Gemma4ToolCallBoundaryTokenizer(),
        return_final_assistant_start=True,
    )

    assert tokenized.input_ids.tolist() == expected_input_ids
    assert tokenized.assistant_mask.tolist() == expected_mask
    assert tokenized.final_assistant_start == expected_final_start


@pytest.mark.parametrize("column", ["messages", "conversation", "conversations"])
def test_shared_chat_preprocessing_supports_all_declared_conversation_columns(column):
    turns = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"},
    ]
    if column == "conversations":
        row = {column: [{"from": turn["role"], "value": turn["content"]} for turn in turns]}
    else:
        row = {column: turns}

    assert normalize_chat_conversation(row) == turns


def test_shared_chat_preprocessing_normalizes_sharegpt_roles_before_templating():
    tokenized = tokenize_chat_example(
        {
            "conversations": [
                {"from": "human", "value": "question"},
                {"from": "gpt", "value": "answer"},
            ]
        },
        _LlamaPreprocessingTokenizer(),
    )

    assert [turn["role"] for turn in tokenized.conversation] == ["user", "assistant"]
    assert tokenized.assistant_mask.any()


def test_normalize_chat_conversation_normalizes_openai_tool_calls_without_mutating_input():
    row = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"city":"Seattle"}'},
                    }
                ],
            },
        ]
    }

    normalized = normalize_chat_conversation(row)

    assert normalized[1]["content"] == ""
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == {"city": "Seattle"}
    assert row["messages"][1]["content"] is None
    assert row["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"city":"Seattle"}'


@pytest.mark.parametrize("arguments", ["not JSON", "[]", '"Seattle"'])
def test_normalize_chat_conversation_rejects_invalid_openai_tool_call_arguments(arguments):
    row = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": arguments}}],
            }
        ]
    }

    with pytest.raises(ValueError, match=r"function\.arguments must be (valid JSON|a JSON object)"):
        normalize_chat_conversation(row)


def test_ultrachat_style_row_has_matching_gpt_sft_and_direct_hf_collation():
    tokenizer = _LlamaPreprocessingTokenizer()
    row = {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }

    shared = tokenize_chat_example(row, tokenizer)

    class _MegatronTokenizerWrapper:
        _tokenizer = tokenizer
        eos_id = tokenizer.eos_token_id

    megatron_tokenizer = _MegatronTokenizerWrapper()
    gpt_dataset = GPTSFTChatDataset.__new__(GPTSFTChatDataset)
    gpt_dataset.use_hf_tokenizer_chat_template = True
    gpt_dataset.loss_mode = "assistant"
    gpt_dataset.tool_schemas = None
    gpt_dataset.tokenizer = megatron_tokenizer
    gpt_dataset.output_original_text = False
    gpt_dataset.max_seq_length = 16
    gpt_dataset.tokens_to_generate = 0
    gpt_dataset.pad_to_max_length = False
    gpt_dataset.pad_seq_length_to_mult = 1
    gpt_dataset.ceil_to_power_2 = False
    gpt_dataset.get_attention_mask_from_fusion = True

    gpt_sft = gpt_dataset._process_example(row)
    gpt_batch = gpt_dataset.collate_fn([gpt_sft])
    direct_hf = text_chat_collate_fn([row], tokenizer)

    assert shared.input_ids.tolist() == [300, 42, 303, 302, 42, 303]
    assert shared.assistant_mask.tolist() == [False, False, False, False, True, True]
    assert gpt_sft["input_ids"].tolist() == shared.input_ids.tolist()
    assert gpt_sft["loss_mask"].tolist() == shared.assistant_mask.tolist()
    assert direct_hf["input_ids"].tolist() == [shared.input_ids.tolist()]
    assert direct_hf["loss_mask"].tolist() == [[False, False, False, True, True, False]]

    pair_count = shared.input_ids.numel() - 1
    assert gpt_batch["tokens"][0, :pair_count].tolist() == direct_hf["tokens"][0, :pair_count].tolist()
    gpt_trainable = gpt_batch["loss_mask"][0, :pair_count].bool()
    direct_trainable = direct_hf["loss_mask"][0, :pair_count].bool()
    assert gpt_trainable.tolist() == direct_trainable.tolist()
    assert (
        gpt_batch["labels"][0, :pair_count][gpt_trainable].tolist()
        == direct_hf["labels"][0, :pair_count][direct_trainable].tolist()
    )


def test_chat_full_loss_does_not_require_assistant_markers():
    class _FullLossTokenizer:
        chat_template = "{{ messages }}"

        def apply_chat_template(self, conversation, **kwargs):
            del conversation, kwargs
            return {"input_ids": [1, 2, 3]}

    tokenized = tokenize_chat_example(
        {"messages": [{"role": "user", "content": "plain text"}]},
        _FullLossTokenizer(),
        loss_mode="full",
    )

    assert tokenized.assistant_mask.tolist() == [True, True, True]


def test_chat_last_turn_loss_keeps_final_assistant_span():
    class _MultiTurnTokenizer:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, **kwargs):
            del kwargs
            if len(conversation) == 3:
                return {
                    "input_ids": [1, 2, 3],
                    "assistant_masks": [0, 1, 0],
                }
            return {
                "input_ids": [1, 2, 3, 4, 5],
                "assistant_masks": [0, 1, 0, 1, 1],
            }

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _MultiTurnTokenizer(),
        loss_mode="last_turn",
    )

    assert tokenized.assistant_mask.tolist() == [False, False, False, True, True]


def test_chat_last_turn_selects_turn_before_removing_skipped_tokens():
    class _MultiTurnTokenizer:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, **kwargs):
            del kwargs
            if len(conversation) == 2:
                return {
                    "input_ids": [1, 2, 3],
                    "assistant_masks": [0, 1, 0],
                }
            return {
                "input_ids": [1, 2, 3, 99, 5],
                "assistant_masks": [0, 1, 0, 1, 1],
            }

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _MultiTurnTokenizer(),
        skipped_tokens=torch.tensor([99]),
        loss_mode="last_turn",
    )

    assert tokenized.assistant_mask.tolist() == [False, False, False, False, True]


def test_chat_last_turn_uses_conversation_boundary_across_mask_gaps():
    class _MultiTurnTokenizer:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, **kwargs):
            del kwargs
            if len(conversation) == 3:
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0]}
            if conversation[-1]["content"] == "":
                return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 1, 0, 0]}
            return {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "assistant_masks": [0, 1, 0, 1, 0, 1],
            }

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _MultiTurnTokenizer(),
        loss_mode="last_turn",
    )

    assert tokenized.assistant_mask.tolist() == [False, False, False, True, False, True]


def test_chat_last_turn_does_not_fall_back_when_final_turn_is_entirely_skipped():
    class _MultiTurnTokenizer:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"

        def apply_chat_template(self, conversation, **kwargs):
            del kwargs
            if len(conversation) == 3:
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0]}
            return {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "assistant_masks": [0, 1, 0, 1, 0, 1],
            }

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _MultiTurnTokenizer(),
        skipped_tokens=torch.tensor([4, 6]),
        loss_mode="last_turn",
        warn_on_all_masked=False,
    )

    assert tokenized.assistant_mask.tolist() == [False, False, False, False, False, False]


def test_chat_last_turn_right_truncation_does_not_train_earlier_turn():
    class _RightTruncatingTokenizer:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        truncation_side = "right"

        def apply_chat_template(self, conversation, **kwargs):
            if len(conversation) == 3:
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0]}
            if kwargs.get("truncation"):
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0]}
            return {"input_ids": [1, 2, 3, 4, 5, 6], "assistant_masks": [0, 1, 0, 1, 1, 1]}

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _RightTruncatingTokenizer(),
        max_length=3,
        loss_mode="last_turn",
        warn_on_all_masked=False,
    )

    assert tokenized.assistant_mask.tolist() == [False, False, False]
    assert tokenized.final_assistant_start is None


def test_chat_last_turn_maps_boundary_through_left_truncation():
    class _LeftTruncatingTokenizer:
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        truncation_side = "left"

        def apply_chat_template(self, conversation, **kwargs):
            if len(conversation) == 3:
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0]}
            if kwargs.get("truncation"):
                return {"input_ids": [4, 5, 6], "assistant_masks": [1, 0, 1]}
            return {"input_ids": [1, 2, 3, 4, 5, 6], "assistant_masks": [0, 1, 0, 1, 0, 1]}

    tokenized = tokenize_chat_example(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _LeftTruncatingTokenizer(),
        max_length=3,
        loss_mode="last_turn",
    )

    assert tokenized.assistant_mask.tolist() == [True, False, True]
    assert tokenized.final_assistant_start is None


@pytest.mark.parametrize("loss_mode", ["assistant", "full"])
def test_gpt_chat_context_answer_split_is_independent_of_loss_mask(loss_mode):
    class _MultiTurnTokenizer:
        eos_id = 6
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        added_tokens_decoder = {4: "<image>"}

        def __init__(self):
            self._tokenizer = self

        def apply_chat_template(self, conversation, **kwargs):
            del kwargs
            if len(conversation) == 3:
                return {"input_ids": [1, 2, 3], "assistant_masks": [0, 1, 0]}
            if conversation[-1]["content"] == "":
                return {"input_ids": [1, 2, 3, 4], "assistant_masks": [0, 1, 0, 0]}
            return {
                "input_ids": [1, 2, 3, 4, 5, 6],
                "assistant_masks": [0, 1, 0, 0, 1, 1],
            }

    result = _chat_preprocess(
        {
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "second answer"},
            ]
        },
        _MultiTurnTokenizer(),
        loss_mode=loss_mode,
    )

    assert result["context_ids"].tolist() == [1, 2, 3, 4]
    assert result["answer_ids"].tolist() == [5, 6]


def test_gpt_chat_row_ending_in_user_content_has_no_answer_split():
    class _TrailingUserTokenizer:
        eos_id = 5
        chat_template = "{% generation %}{{ messages }}{% endgeneration %}"
        added_tokens_decoder = {}

        def __init__(self):
            self._tokenizer = self

        def apply_chat_template(self, conversation, **kwargs):
            del conversation, kwargs
            return {
                "input_ids": [1, 2, 3, 4, 5],
                "assistant_masks": [0, 1, 0, 0, 0],
            }

    result = _chat_preprocess(
        {
            "messages": [
                {"role": "assistant", "content": "earlier answer"},
                {"role": "user", "content": "new question"},
            ]
        },
        _TrailingUserTokenizer(),
        loss_mode="last_turn",
    )

    assert result["loss_mask"].tolist() == [False, True, False, False, False]
    assert result["context_ids"].tolist() == [1, 2, 3, 4, 5]
    assert result["answer_ids"].tolist() == []


def test_assistant_mask_boundary_config_from_markers_tokenizes_declared_markers():
    boundary_config = assistant_mask_boundary_config_from_markers(
        _ChatMLBoundaryProcessor(),
        assistant_start="<|im_start|>assistant\n",
        assistant_end="<|im_end|>",
    )

    assert boundary_config.role_start_tokens == {"assistant": [102]}
    assert boundary_config.role_end_tokens == {"assistant": [103]}


def test_infer_assistant_mask_boundary_config_uses_processor_template_when_tokenizer_template_is_empty():
    boundary_config = infer_assistant_mask_boundary_config(_ProcessorTemplateBoundaryProcessor())

    assert boundary_config is not None
    assert boundary_config.role_start_tokens == {"assistant": [202]}
    assert boundary_config.role_end_tokens == {"assistant": [203]}


def test_assistant_mask_boundary_config_from_markers_raises_when_markers_cannot_tokenize():
    with pytest.raises(ValueError, match="Unable to tokenize assistant loss-mask boundary markers"):
        assistant_mask_boundary_config_from_markers(
            _NonTokenizingProcessor(),
            assistant_start="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
        )


def test_build_assistant_loss_mask_uses_inferred_boundary_config():
    example = {
        "conversation": [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 4, 103, 104, 102, 3, 4, 103, 104])
    processor = _ChatMLBoundaryProcessor()

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_falls_back_when_hf_generation_mask_is_all_zero():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 103, 104, 102, 3, 4, 103, 104])
    processor = _ZeroGenerationMaskProcessor()

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_augments_nonzero_hf_mask_with_assistant_end_tokens():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 103, 104, 102, 3, 4, 103, 104])
    processor = _ContentOnlyGenerationMaskProcessor()

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_falls_back_to_end_without_newline_before_right_padding():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 103, 104, 102, 3, 4, 103, 0, 0])
    processor = _ChatMLBoundaryProcessor()

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_build_assistant_loss_mask_uses_earliest_end_variant_before_later_user_turn():
    input_ids = torch.tensor([102, 3, 103, 100, 16, 103, 104])
    processor = _ChatMLBoundaryProcessor()

    mask = build_assistant_loss_mask(
        [
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
        ],
        input_ids,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
    )

    assert mask.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_build_assistant_loss_mask_fails_closed_when_no_end_boundary_precedes_padding():
    input_ids = torch.tensor([102, 3, 4, 0, 0])
    processor = _ChatMLBoundaryProcessor()

    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        build_assistant_loss_mask(
            [{"role": "assistant", "content": "answer"}],
            input_ids,
            processor,
            boundary_config=infer_assistant_mask_boundary_config(processor),
        )


def test_build_assistant_loss_mask_keeps_valid_all_zero_hf_mask_when_assistant_is_truncated():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3])
    processor = _TruncatedZeroGenerationMaskProcessor()

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        boundary_config=infer_assistant_mask_boundary_config(processor),
        warn_on_all_masked=False,
    )

    assert mask.tolist() == [0.0, 0.0]


def test_build_assistant_loss_mask_uses_marker_boundary_config():
    example = {
        "conversation": [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 4, 101, 102, 3, 4, 103])
    processor = _ChatMLBoundaryProcessor()

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        boundary_config=assistant_mask_boundary_config_from_markers(
            processor,
            assistant_start="<|im_start|>assistant\n",
            assistant_end="<|im_end|>",
        ),
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_handles_non_tokenizing_tokenizer():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([1, 2, 3])

    with pytest.raises(ValueError, match="Unable to build assistant loss mask"):
        build_assistant_loss_mask(example, input_ids, _NonTokenizingProcessor(), warn_on_all_masked=False)


def test_build_assistant_loss_mask_uses_explicit_boundary_config():
    example = {
        "conversation": [
            {"role": "user", "content": "answer"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 3, 4, 101, 102, 3, 4, 103])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"user": [100], "assistant": [102]},
        role_end_tokens={"user": [101], "assistant": [103]},
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_raises_for_incomplete_boundary_config():
    example = {
        "conversation": [
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([102, 3, 4, 103])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"assistant": [102]},
        role_end_tokens={},
    )

    with pytest.raises(ValueError, match="role_start_tokens, role_end_tokens"):
        build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)


def test_build_assistant_loss_mask_raises_when_boundary_config_does_not_match():
    example = {
        "conversation": [
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([1, 3, 4, 2])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"assistant": [102]},
        role_end_tokens={"assistant": [103]},
    )

    with pytest.raises(ValueError, match="did not match any loss-contributing spans"):
        build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)


def test_build_assistant_loss_mask_boundary_config_trains_full_loss_role_content():
    example = {
        "conversation": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([100, 8, 101, 102, 200, 30, 31, 201, 32, 202, 40, 41, 203, 33, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"user": [100], "assistant": [102]},
        role_end_tokens={"user": [101], "assistant": [101]},
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_ignores_tool_like_content_in_non_loss_roles():
    example = {
        "conversation": [
            {"role": "system", "content": "tool schema"},
            {"role": "assistant", "content": "tool call"},
        ]
    }
    input_ids = torch.tensor([99, 202, 50, 203, 101, 102, 202, 60, 203, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"system": [99], "assistant": [102]},
        role_end_tokens={"system": [101], "assistant": [101]},
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]


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
        role_end_tokens={"user": [101], "assistant": [101]},
        loss_roles=("assistant",),
        include_start_tokens_for_roles=("assistant",),
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_trims_leading_delimiters():
    example = {
        "conversation": [
            {"role": "assistant", "content": "answer"},
        ]
    }
    input_ids = torch.tensor([102, 55, 3, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"assistant": [102]},
        role_end_tokens={"assistant": [101]},
        trim_leading_token_ids=(55,),
    )

    mask = build_assistant_loss_mask(example, input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_build_assistant_loss_mask_boundary_config_trims_leading_sequences_only_when_complete():
    input_ids = torch.tensor([102, 55, 3, 56, 4, 101, 102, 55, 56, 5, 101])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"assistant": [102]},
        role_end_tokens={"assistant": [101]},
        trim_leading_token_sequences=([55, 56],),
    )

    mask = build_assistant_loss_mask([], input_ids, _Processor(), boundary_config=boundary_config)

    assert mask.tolist() == [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_build_assistant_loss_mask_applies_skipped_tokens_to_boundary_mask():
    example = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
        ]
    }
    input_ids = torch.tensor([100, 1, 2, 101, 102, 3, 4, 103])
    boundary_config = AssistantMaskBoundaryConfig(
        role_start_tokens={"user": [100], "assistant": [102]},
        role_end_tokens={"user": [101], "assistant": [103]},
    )

    mask = build_assistant_loss_mask(
        example,
        input_ids,
        _Processor(),
        skipped_tokens=torch.tensor([4]),
        boundary_config=boundary_config,
    )

    assert mask.tolist() == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]


def test_build_assistant_loss_mask_raises_without_valid_mask_source():
    example = {
        "conversation": [
            {"role": "user", "content": [{"type": "text", "text": "question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "missing"}]},
        ]
    }
    input_ids = torch.tensor([1, 2, 3, 4, 99])

    with pytest.raises(ValueError, match="Unable to build assistant loss mask"):
        build_assistant_loss_mask(example, input_ids, _Processor(), warn_on_all_masked=False)


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
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    apply_assistant_labels_to_batch(batch, examples, _GenerationMaskProcessor(), skipped_tokens=torch.tensor([]))

    assert batch["loss_mask"].tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX]]


def test_apply_assistant_labels_to_batch_unmask_last_token_affects_shifted_loss_mask():
    examples = [
        {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": "question"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
            ]
        }
    ]
    batch = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    apply_assistant_labels_to_batch(
        batch,
        examples,
        _GenerationMaskProcessor(),
        skipped_tokens=torch.tensor([]),
        unmask_last_token=True,
    )

    assert batch["loss_mask"].tolist() == [[0.0, 1.0, 1.0, 0.0]]
    assert batch["labels"].tolist() == [[IGNORE_INDEX, 3, 4, IGNORE_INDEX]]


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


def test_normalize_energon_vlm_sample_preserves_tools_and_tool_calls():
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    tool_calls = [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    sample = ChatMLSample(
        **sample_metadata_kwargs(key="sample-tools", restore_key=(), subflavors={}),
        conversation=json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {"role": "assistant", "content": None, "tool_calls": tool_calls},
                ],
                "tools": tools,
            }
        ),
    )

    normalized = normalize_energon_vlm_sample(sample)
    example = normalized_vlm_sample_to_hf_example(normalized)

    assert normalized.tools == tools
    assert normalized.conversation[1]["tool_calls"] == tool_calls
    assert example["tools"] == tools
    assert example["conversation"][1]["tool_calls"] == tool_calls


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
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
    }

    normalized = normalize_hf_vlm_example(example)

    assert normalized.conversation == conversation
    assert normalized.conversation is not conversation
    assert normalized.images == [image]
    assert normalized.videos is None
    assert normalized.audio == "audio-payload"
    assert normalized.tools == example["tools"]
