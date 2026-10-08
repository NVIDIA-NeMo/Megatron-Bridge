# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch

from megatron.bridge.models.deepseek.data.collate_fn import (
    deepseek_v4_collate_fn,
    tokenize_deepseek_v4_example,
)
from megatron.bridge.models.deepseek.data.encoding_v4 import (
    ASSISTANT_TOKEN,
    BOS_TOKEN,
    DSML_TOKEN,
    EOS_TOKEN,
    THINKING_END_TOKEN,
    THINKING_START_TOKEN,
    USER_TOKEN,
    encode_deepseek_v4_messages,
)


pytestmark = pytest.mark.unit


class _DeepSeekV4CharacterTokenizer:
    name_or_path = "deepseek-ai/DeepSeek-V4-Flash"
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token_id = 1
    truncation_side = "right"
    added_tokens_decoder: dict[int, Any] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(character) for character in text]

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


class _DeepSeekV4ReferenceTokenizer(_DeepSeekV4CharacterTokenizer):
    def __init__(self, expected_prompt_sha256: str, expected_input_ids: list[int]):
        self.expected_prompt_sha256 = expected_prompt_sha256
        self.expected_input_ids = expected_input_ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        assert hashlib.sha256(text.encode()).hexdigest() == self.expected_prompt_sha256
        return self.expected_input_ids


ORDINARY_MESSAGES = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "First?"},
    {"role": "assistant", "content": "One."},
    {"role": "user", "content": "Second?"},
    {"role": "assistant", "content": "Two."},
]

REASONING_MESSAGES = [
    {"role": "system", "content": "Solve carefully."},
    {"role": "user", "content": "1+1?"},
    {"role": "assistant", "reasoning_content": "Add one and one.", "content": "Two."},
]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# Generated with the DeepSeek-V4-Flash tokenizer snapshot 60d8d70770c6776ff598c94bb586a859a38244f1.  # pragma: allowlist secret
# The prompt also matches SGLang 2b4381956f2dfc302ddb4c48a9ab30be41958838 exactly.  # pragma: allowlist secret
# It matches vLLM 7f7a32cfec0f1bc5b73c37200b86631523a1ea8f with low reasoning effort.  # pragma: allowlist secret
DSV4_THINKING_TOOL_PROMPT_SHA256 = (
    "56f4dfb17a268aa7477ec0b804db0cd0aae9f2961afa9b9e97124348055c0f5f"  # pragma: allowlist secret
)
DSV4_THINKING_TOOL_INPUT_IDS = [
    int(token_id)
    for token_id in (
        "0 271 372 27193 271 3476 611 3278 304 260 1341 294 6704 304 1694 3287 270 3967 734 3417 16 2042 588 "
        "34756 6704 513 4985 260 38322 128825 72461 4941 12548 48902 5603 1277 270 2502 979 30 128825 72461 4941 "
        "12548 1018 30 128825 40148 5406 2329 84476 6792 5896 35044 3816 30 128825 41523 2329 84476 46630 4130 "
        "61729 35044 4 3418 1281 11476 94 19836 3320 6 46630 4130 61729 56909 1718 128825 41523 1018 7835 1718 "
        "128825 40148 5406 1018 30 128825 40148 5406 2329 84476 6792 5896 35044 20 3816 7835 1718 128825 40148 "
        "5406 1018 1718 128825 72461 4941 12548 4697 3524 8252 1531 366 12038 412 344 305 1341 3608 4463 1281 "
        "11476 4 37419 1884 710 915 4815 343 62896 14 2631 2065 634 14 31939 14 8435 754 2281 270 1990 295 26639 "
        "8786 305 1341 3608 4463 1281 19836 4 108526 3575 6892 71144 344 22104 343 89944 284 513 223 128821 754 "
        "440 74366 5238 782 5553 22805 6352 223 128821 1613 128822 119907 1117 4105 10699 469 4087 4256 339 13079 "
        "6922 14 5238 6578 1561 223 128822 418 4105 10699 469 4087 4256 339 795 17829 28249 13178 8380 271 24313 "
        "2852 3362 582 1133 65 50219 1760 582 20855 3362 582 6287 9670 66910 582 46172 3362 28612 4611 3362 582 "
        "10325 1760 582 68838 3362 28612 37399 3362 28612 4611 3362 582 4463 4 55695 582 24486 3362 20584 37399 "
        "13747 33236 3476 74366 29851 1605 270 3554 6428 4105 2329 305 10767 5815 8380 304 34756 4105 10699 603 "
        "128803 58565 33 128804 128821 50249 1499 16 128822 271 30 128825 72461 4941 12548 1018 30 128825 40148 "
        "5406 2329 1281 1133 65 50219 3816 30 128825 41523 2329 1281 37399 4 3418 1281 11476 3320 4374 6829 1718 "
        "128825 41523 1018 1718 128825 40148 5406 1018 1718 128825 72461 4941 12548 32 1 128803 30 72461 46148 32 "
        "24313 88634 3362 736 24568 72461 46148 32 128804 128821 2107 344 223 736 16 128822 736 2614 37 16 1"
    ).split()
]


def _decode_character_ids(input_ids: list[int]) -> str:
    return "".join(chr(token_id) for token_id in input_ids if token_id != 0)


def test_deepseek_v4_encoder_matches_official_ordinary_modes():
    chat = encode_deepseek_v4_messages(ORDINARY_MESSAGES, thinking_mode="chat", truncate_history_thinking=False)
    thinking = encode_deepseek_v4_messages(
        ORDINARY_MESSAGES, thinking_mode="thinking", truncate_history_thinking=False
    )

    assert chat == (
        f"{BOS_TOKEN}You are concise.{USER_TOKEN}First?{ASSISTANT_TOKEN}{THINKING_END_TOKEN}One.{EOS_TOKEN}"
        f"{USER_TOKEN}Second?{ASSISTANT_TOKEN}{THINKING_END_TOKEN}Two.{EOS_TOKEN}"
    )
    assert thinking == (
        f"{BOS_TOKEN}You are concise.{USER_TOKEN}First?{ASSISTANT_TOKEN}"
        f"{THINKING_START_TOKEN}{THINKING_END_TOKEN}One.{EOS_TOKEN}"
        f"{USER_TOKEN}Second?{ASSISTANT_TOKEN}{THINKING_START_TOKEN}{THINKING_END_TOKEN}Two.{EOS_TOKEN}"
    )


def test_deepseek_v4_encoder_preserves_reasoning_and_formats_tools():
    messages = [
        {"role": "system", "content": "Use tools.", "tools": [WEATHER_TOOL]},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "reasoning_content": "Need data.",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Seattle"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature":12}'},
        {"role": "assistant", "reasoning_content": "It is 12.", "content": "12°C."},
    ]

    rendered = encode_deepseek_v4_messages(messages, thinking_mode="thinking")

    assert "Need data.</think>" in rendered
    assert "It is 12.</think>12°C." in rendered
    assert f"<{DSML_TOKEN}tool_calls>" in rendered
    assert f'<{DSML_TOKEN}parameter name="city" string="true">Seattle</{DSML_TOKEN}parameter>' in rendered
    assert f'{USER_TOKEN}<tool_result>{{"temperature":12}}</tool_result>{ASSISTANT_TOKEN}' in rendered


def test_deepseek_v4_encoder_formats_non_thinking_tool_followup():
    messages = [
        {"role": "system", "content": "Use tools.", "tools": [WEATHER_TOOL]},
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Seattle"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature":12}'},
        {"role": "assistant", "content": "12°C."},
    ]

    rendered = encode_deepseek_v4_messages(messages, thinking_mode="chat")

    assert f"{ASSISTANT_TOKEN}{THINKING_END_TOKEN}\n\n<{DSML_TOKEN}tool_calls>" in rendered
    assert (
        f'{USER_TOKEN}<tool_result>{{"temperature":12}}</tool_result>'
        f"{ASSISTANT_TOKEN}{THINKING_END_TOKEN}12°C.{EOS_TOKEN}"
    ) in rendered


def test_deepseek_v4_thinking_tool_prompt_and_ids_match_serving_references():
    tokenizer = _DeepSeekV4ReferenceTokenizer(
        DSV4_THINKING_TOOL_PROMPT_SHA256,
        DSV4_THINKING_TOOL_INPUT_IDS,
    )
    example = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "reasoning_content": "Need data.",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Seattle"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature":12}'},
            {"role": "assistant", "reasoning_content": "It is 12.", "content": "12°C."},
        ],
        "tools": [WEATHER_TOOL],
        "thinking_mode": "thinking",
        "truncate_history_thinking": False,
    }

    tokenized = tokenize_deepseek_v4_example(example, tokenizer, loss_mode="full")

    assert tokenized.input_ids.tolist() == DSV4_THINKING_TOOL_INPUT_IDS
    assert tokenized.assistant_mask.all()


@pytest.mark.parametrize(
    ("mode", "expected_assistant_text"),
    [
        ("chat", f"{THINKING_END_TOKEN}Two.{EOS_TOKEN}"),
        ("thinking", f"{THINKING_START_TOKEN}Add one and one.{THINKING_END_TOKEN}Two.{EOS_TOKEN}"),
    ],
)
def test_deepseek_v4_tokenizer_builds_content_plus_eos_mask(mode, expected_assistant_text):
    tokenizer = _DeepSeekV4CharacterTokenizer()
    tokenized = tokenize_deepseek_v4_example(
        {"messages": REASONING_MESSAGES, "thinking_mode": mode, "truncate_history_thinking": False},
        tokenizer,
    )

    rendered = _decode_character_ids(tokenized.input_ids.tolist())
    supervised = _decode_character_ids(tokenized.input_ids[tokenized.assistant_mask].tolist())
    assert rendered.endswith(f"{ASSISTANT_TOKEN}{expected_assistant_text}")
    assert supervised == expected_assistant_text


def test_deepseek_v4_tokenizer_accepts_tool_call_without_content():
    tokenizer = _DeepSeekV4CharacterTokenizer()
    example = {
        "messages": [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Seattle"}'}}],
            },
        ],
        "tools": [WEATHER_TOOL],
        "enable_thinking": False,
    }

    tokenized = tokenize_deepseek_v4_example(example, tokenizer)

    assert tokenized.conversation[2]["content"] == ""
    assert f'<{DSML_TOKEN}invoke name="get_weather">' in _decode_character_ids(tokenized.input_ids.tolist())


@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_deepseek_v4_tokenizer_truncates_ids_and_mask_together(truncation_side):
    tokenizer = _DeepSeekV4CharacterTokenizer()
    tokenizer.truncation_side = truncation_side
    example = {"messages": ORDINARY_MESSAGES, "thinking_mode": "chat"}
    full = tokenize_deepseek_v4_example(example, tokenizer, loss_mode="last_turn")
    max_length = full.input_ids.numel() - 1

    truncated = tokenize_deepseek_v4_example(
        example,
        tokenizer,
        max_length=max_length,
        loss_mode="last_turn",
        warn_on_all_masked=False,
    )

    expected_slice = slice(1, None) if truncation_side == "left" else slice(None, -1)
    assert torch.equal(truncated.input_ids, full.input_ids[expected_slice])
    assert torch.equal(truncated.assistant_mask, full.assistant_mask[expected_slice])


def test_deepseek_v4_collator_shifts_labels_and_pads_rows():
    tokenizer = _DeepSeekV4CharacterTokenizer()
    examples = [
        {"messages": ORDINARY_MESSAGES, "enable_thinking": False},
        {"messages": REASONING_MESSAGES, "enable_thinking": True, "truncate_history_thinking": False},
    ]

    batch = deepseek_v4_collate_fn(examples, tokenizer, pad_to_multiple_of=8)

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"][1, -1].item() == 0
    assert batch["labels"][1, -1].item() == -100
    assert batch["loss_mask"][1, -1].item() == 0
    assert batch["tokens"].data_ptr() == batch["input_ids"].data_ptr()


def test_deepseek_v4_collator_requires_explicit_thinking_mode():
    with pytest.raises(ValueError, match="require thinking_mode"):
        deepseek_v4_collate_fn([{"messages": ORDINARY_MESSAGES}], _DeepSeekV4CharacterTokenizer())


@pytest.mark.parametrize("legacy_key", ["drop_thinking", "preserve_thinking"])
def test_deepseek_v4_collator_rejects_legacy_history_thinking_options(legacy_key):
    example = {"messages": ORDINARY_MESSAGES, "enable_thinking": True, legacy_key: True}

    with pytest.raises(ValueError, match="use truncate_history_thinking"):
        deepseek_v4_collate_fn([example], _DeepSeekV4CharacterTokenizer())
