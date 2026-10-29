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


class _NemotronLightningTokenizer:
    """Small offline renderer with the official Lightning chat-template semantics."""

    name_or_path = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    pad_token_id = 0
    eos_token_id = IM_END
    truncation_side = "right"
    added_tokens_decoder: dict[int, Any] = {}
    chat_template = (
        "{% set truncate_history_thinking = true %}"
        "<|im_start|>system\n{{ content }}<|im_end|>\n"
        "<|im_start|>user\n{{ content }}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n{{ reasoning_content }}</think>{{ content }}<|im_end|>\n"
        "<|im_start|>tool\n{{ content }}<|im_end|>\n"
    )

    def __init__(self) -> None:
        self.template_kwargs: list[dict[str, Any]] = []
        self.tool_call_arguments: list[dict[str, Any]] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        marker_ids = {
            "<|im_start|>system\n": [IM_START, SYSTEM_ROLE, NEWLINE],
            "<|im_start|>user\n": [IM_START, USER_ROLE, NEWLINE],
            "<|im_start|>assistant\n": [IM_START, ASSISTANT_ROLE, NEWLINE],
            "<|im_start|>tool\n": [IM_START, USER_ROLE, NEWLINE],
            "<|im_end|>\n": [IM_END, NEWLINE],
            "<|im_end|>": [IM_END],
            "<think>": [THINK_OPEN],
            "<think>\n": [THINK_OPEN, NEWLINE],
            "<think></think>": [THINK_OPEN, THINK_CLOSE],
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

        truncate_history = kwargs.get("truncate_history_thinking", True)
        last_user_index = max(
            (index for index, message in enumerate(conversation) if message["role"] == "user"),
            default=-1,
        )
        input_ids: list[int] = []
        for index, message in enumerate(conversation):
            role = message["role"]
            if role == "assistant":
                input_ids.extend([IM_START, ASSISTANT_ROLE, NEWLINE])
                reasoning = message.get("reasoning_content")
                if truncate_history and index < last_user_index:
                    reasoning = None
                input_ids.append(THINK_OPEN)
                if isinstance(reasoning, str) and reasoning:
                    input_ids.extend([NEWLINE, TEXT_IDS[reasoning]])
                input_ids.append(THINK_CLOSE)
                content = message.get("content")
                if isinstance(content, str) and content:
                    input_ids.append(TEXT_IDS[content])
                for tool_call in message.get("tool_calls", []):
                    arguments = tool_call["function"]["arguments"]
                    self.tool_call_arguments.append(dict(arguments.items()))
                    input_ids.extend([NEWLINE, TOOL_CALL_OPEN, TOOL_NAME, TOOL_ARGUMENT, TOOL_CALL_CLOSE])
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


def _supervised_ids(tokenized: Any) -> list[int]:
    return tokenized.input_ids[tokenized.assistant_mask].tolist()


@pytest.mark.parametrize(
    ("truncate_history", "expected_first_thinking"),
    [(True, []), (False, [TEXT_IDS["reason one"], THINK_CLOSE])],
)
def test_nemotron_lightning_forwards_thinking_controls_and_masks_assistant_turns(
    truncate_history: bool,
    expected_first_thinking: list[int],
) -> None:
    tokenizer = _NemotronLightningTokenizer()
    example = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question one"},
            {"role": "assistant", "reasoning_content": "reason one", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "reasoning_content": "reason two", "content": "answer two"},
        ],
        "chat_template_kwargs": {
            "enable_thinking": True,
            "truncate_history_thinking": truncate_history,
        },
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)

    semantic_kwargs = [kwargs for kwargs in tokenizer.template_kwargs if "enable_thinking" in kwargs]
    assert semantic_kwargs
    assert all(kwargs["enable_thinking"] is True for kwargs in semantic_kwargs)
    assert all(kwargs["truncate_history_thinking"] is truncate_history for kwargs in semantic_kwargs)
    assert _supervised_ids(tokenized) == [
        *expected_first_thinking,
        TEXT_IDS["answer one"],
        IM_END,
        NEWLINE,
        TEXT_IDS["reason two"],
        THINK_CLOSE,
        TEXT_IDS["answer two"],
        IM_END,
        NEWLINE,
    ]


def test_nemotron_lightning_without_reasoning_masks_empty_thinking_boundary() -> None:
    tokenizer = _NemotronLightningTokenizer()
    example = {
        "messages": [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "answer one"},
        ],
        "chat_template_kwargs": {
            "enable_thinking": False,
            "truncate_history_thinking": True,
        },
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)

    semantic_kwargs = [kwargs for kwargs in tokenizer.template_kwargs if "enable_thinking" in kwargs]
    assert semantic_kwargs
    assert all(kwargs["enable_thinking"] is False for kwargs in semantic_kwargs)
    assert _supervised_ids(tokenized) == [
        TEXT_IDS["answer one"],
        IM_END,
        NEWLINE,
    ]


def test_nemotron_lightning_tool_reasoning_and_calls_are_supervised() -> None:
    tokenizer = _NemotronLightningTokenizer()
    example = {
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "reasoning_content": "reason one",
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city":"Seattle"}'}}],
            },
            {"role": "tool", "content": "12 C"},
            {"role": "assistant", "reasoning_content": "reason two", "content": "Seattle is 12 C."},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
        "chat_template_kwargs": {"enable_thinking": True, "truncate_history_thinking": True},
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)
    batch = text_chat_collate_fn([example], tokenizer, warn_on_all_masked=False)

    expected_targets = [
        TEXT_IDS["reason one"],
        THINK_CLOSE,
        NEWLINE,
        TOOL_CALL_OPEN,
        TOOL_NAME,
        TOOL_ARGUMENT,
        TOOL_CALL_CLOSE,
        IM_END,
        NEWLINE,
        TEXT_IDS["reason two"],
        THINK_CLOSE,
        TEXT_IDS["Seattle is 12 C."],
        IM_END,
        NEWLINE,
    ]
    assert _supervised_ids(tokenized) == expected_targets
    assert tokenizer.tool_call_arguments
    assert all(arguments == {"city": "Seattle"} for arguments in tokenizer.tool_call_arguments)
    assert TOOL_RESPONSE_OPEN not in expected_targets
    assert batch["labels"][0][batch["loss_mask"][0].bool()].tolist() == expected_targets
