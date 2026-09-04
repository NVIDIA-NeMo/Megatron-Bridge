# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from typing import Any

import pytest

from megatron.bridge.data.collators.sft import text_chat_collate_fn
from megatron.bridge.data.conversation_processing import tokenize_chat_example


pytestmark = pytest.mark.unit


IM_START = 10
IM_END = 11
THINK_OPEN = 12
THINK_CLOSE = 13
SYSTEM_ROLE = 20
USER_ROLE = 21
ASSISTANT_ROLE = 22
NEWLINE = 30
TOOL_RESPONSE_OPEN = 40
TOOL_RESPONSE_CLOSE = 41
TOOL_CALL_OPEN = 42
TOOL_CALL_CLOSE = 43
TOOL_NAME = 44
TOOL_ARGUMENT = 45

TEXT_IDS = {
    "system": 50,
    "question one": 51,
    "answer one": 52,
    "reason one": 53,
    "question two": 54,
    "answer two": 55,
    "reason two": 56,
    "weather?": 57,
    "12 C": 58,
    "Seattle is 12 C.": 59,
}

_CHATML_TEMPLATE = (
    "<|im_start|>system\n{{ content }}<|im_end|>\n"
    "<|im_start|>user\n{{ content }}<|im_end|>\n"
    "<|im_start|>assistant\n{{ content }}<|im_end|>\n"
)


class _Qwen35ChatTokenizer:
    """Small offline renderer with Qwen 3.5 completed-turn semantics."""

    name_or_path = "Qwen/Qwen3.5-27B"
    pad_token_id = 0
    eos_token_id = IM_END
    truncation_side = "right"
    added_tokens_decoder: dict[int, Any] = {}
    chat_template = _CHATML_TEMPLATE

    def __init__(self) -> None:
        self.template_kwargs: list[dict[str, Any]] = []
        self.tool_call_arguments: list[dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        marker_ids = {
            "<|im_start|>system\n": [IM_START, SYSTEM_ROLE, NEWLINE],
            "<|im_start|>user\n": [IM_START, USER_ROLE, NEWLINE],
            "<|im_start|>assistant\n": [IM_START, ASSISTANT_ROLE, NEWLINE],
            "<|im_end|>\n": [IM_END, NEWLINE],
            "<|im_end|>": [IM_END],
            "\n": [NEWLINE],
        }
        return marker_ids.get(text, [TEXT_IDS.get(text, 99)])

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[int]] | list[int]:
        assert tokenize is True
        assert add_generation_prompt is False
        self.template_kwargs.append(dict(kwargs))

        last_user_index = max(
            (index for index, message in enumerate(conversation) if message["role"] == "user"),
            default=-1,
        )
        preserve_thinking = kwargs.get("preserve_thinking", False)
        input_ids: list[int] = []
        for index, message in enumerate(conversation):
            role = message["role"]
            if role == "assistant":
                input_ids.extend([IM_START, ASSISTANT_ROLE, NEWLINE])
                if preserve_thinking or index > last_user_index:
                    input_ids.extend([THINK_OPEN, NEWLINE])
                    reasoning = message.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        input_ids.append(TEXT_IDS[reasoning])
                    input_ids.extend([NEWLINE, THINK_CLOSE, NEWLINE, NEWLINE])
                content = message.get("content")
                if isinstance(content, str) and content:
                    input_ids.append(TEXT_IDS[content])
                for tool_call in message.get("tool_calls", []):
                    arguments = tool_call["function"]["arguments"]
                    self.tool_call_arguments.append(dict(arguments.items()))
                    input_ids.extend([TOOL_CALL_OPEN, TOOL_NAME, TOOL_ARGUMENT, TOOL_CALL_CLOSE])
                input_ids.extend([IM_END, NEWLINE])
            elif role == "tool":
                input_ids.extend(
                    [
                        IM_START,
                        USER_ROLE,
                        NEWLINE,
                        TOOL_RESPONSE_OPEN,
                        TEXT_IDS[message["content"]],
                        TOOL_RESPONSE_CLOSE,
                        IM_END,
                        NEWLINE,
                    ]
                )
            else:
                role_id = SYSTEM_ROLE if role == "system" else USER_ROLE
                input_ids.extend([IM_START, role_id, NEWLINE, TEXT_IDS[message["content"]], IM_END, NEWLINE])

        return {"input_ids": input_ids} if return_dict else input_ids


class _Qwen36ChatTokenizer(_Qwen35ChatTokenizer):
    """Qwen 3.6 adds the native preserve_thinking template control."""

    name_or_path = "Qwen/Qwen3.6-35B-A3B"
    chat_template = _CHATML_TEMPLATE + "{% if preserve_thinking %}history{% endif %}"


def _tokenizer(version: str) -> _Qwen35ChatTokenizer:
    return _Qwen35ChatTokenizer() if version == "3.5" else _Qwen36ChatTokenizer()


def _thinking_ids(reasoning_id: int | None) -> list[int]:
    reasoning = [] if reasoning_id is None else [reasoning_id]
    return [THINK_OPEN, NEWLINE, *reasoning, NEWLINE, THINK_CLOSE, NEWLINE, NEWLINE]


def _supervised_ids(tokenized: Any) -> list[int]:
    return tokenized.input_ids[tokenized.assistant_mask].tolist()


@pytest.mark.parametrize("version", ["3.5", "3.6"])
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen35_36_no_tool_thinking_and_mask_boundaries(version: str, enable_thinking: bool) -> None:
    tokenizer = _tokenizer(version)
    if enable_thinking:
        messages = [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "reasoning_content": "reason one", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "reasoning_content": "reason two", "content": "answer two"},
        ]
        historical_targets = (
            [*_thinking_ids(TEXT_IDS["reason one"]), TEXT_IDS["answer one"], IM_END, NEWLINE]
            if version == "3.6"
            else [TEXT_IDS["answer one"], IM_END, NEWLINE]
        )
        expected_targets = [
            *historical_targets,
            *_thinking_ids(TEXT_IDS["reason two"]),
            TEXT_IDS["answer two"],
            IM_END,
            NEWLINE,
        ]
    else:
        messages = [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one"},
        ]
        expected_targets = [*_thinking_ids(None), TEXT_IDS["answer one"], IM_END, NEWLINE]
    example = {
        "messages": messages,
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "truncate_history_thinking": False,
        },
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)

    semantic_kwargs = [kwargs for kwargs in tokenizer.template_kwargs if "enable_thinking" in kwargs]
    assert semantic_kwargs
    assert all(kwargs["enable_thinking"] is enable_thinking for kwargs in semantic_kwargs)
    if version == "3.6":
        assert all(kwargs["preserve_thinking"] is True for kwargs in semantic_kwargs)
        assert all("truncate_history_thinking" not in kwargs for kwargs in semantic_kwargs)
    else:
        assert all(kwargs["truncate_history_thinking"] is False for kwargs in semantic_kwargs)
        assert all("preserve_thinking" not in kwargs for kwargs in semantic_kwargs)
    assert _supervised_ids(tokenized) == expected_targets


@pytest.mark.parametrize("version", ["3.5", "3.6"])
@pytest.mark.parametrize("enable_thinking", [False, True])
def test_qwen35_36_tool_calls_are_supervised_and_tool_responses_are_masked(
    version: str,
    enable_thinking: bool,
) -> None:
    tokenizer = _tokenizer(version)
    first_reasoning = TEXT_IDS["reason one"] if enable_thinking else None
    final_reasoning = TEXT_IDS["reason two"] if enable_thinking else None
    example = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "reasoning_content": "reason one" if enable_thinking else None,
                "content": None,
                "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Seattle"}'}}],
            },
            {"role": "tool", "content": "12 C"},
            {
                "role": "assistant",
                "reasoning_content": "reason two" if enable_thinking else None,
                "content": "Seattle is 12 C.",
            },
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "chat_template_kwargs": {
            "enable_thinking": enable_thinking,
            "truncate_history_thinking": False,
        },
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)
    batch = text_chat_collate_fn([example], tokenizer, warn_on_all_masked=False)

    expected_targets = [
        *_thinking_ids(first_reasoning),
        TOOL_CALL_OPEN,
        TOOL_NAME,
        TOOL_ARGUMENT,
        TOOL_CALL_CLOSE,
        IM_END,
        NEWLINE,
        *_thinking_ids(final_reasoning),
        TEXT_IDS["Seattle is 12 C."],
        IM_END,
        NEWLINE,
    ]
    assert _supervised_ids(tokenized) == expected_targets
    assert tokenizer.tool_call_arguments
    assert all(arguments == {"city": "Seattle"} for arguments in tokenizer.tool_call_arguments)
    assert TOOL_RESPONSE_OPEN not in expected_targets
    assert batch["labels"][0][batch["loss_mask"][0].bool()].tolist() == expected_targets
