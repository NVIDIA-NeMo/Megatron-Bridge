# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from typing import Any

import pytest

from megatron.bridge.data.collators.sft import text_chat_collate_fn
from megatron.bridge.data.conversation_processing import tokenize_chat_example


pytestmark = pytest.mark.unit


USER_START = 10
ASSISTANT_START = 11
TURN_END = 12
THINK_OPEN = 13
THINK_CLOSE = 14
RESPONSE_OPEN = 15
RESPONSE_CLOSE = 16
TOOL_CALL = 17
TOOL_RESULT = 18

TEXT_IDS = {
    "question one": 30,
    "answer one": 31,
    "reason one": 32,
    "question two": 33,
    "answer two": 34,
    "reason two": 35,
    "weather?": 36,
    "12 C": 37,
    "Seattle is 12 C.": 38,
}


class _ModelChatTokenizer:
    """Small semantic renderer for three official chat-template boundary formats."""

    pad_token_id = 0
    eos_token_id = TURN_END
    truncation_side = "right"
    added_tokens_decoder: dict[int, Any] = {}

    def __init__(self, model: str) -> None:
        self.model = model
        self.template_kwargs: list[dict[str, Any]] = []
        self.tool_arguments: list[dict[str, Any]] = []
        if model == "glm":
            self.chat_template = "<|assistant|> enable_thinking clear_thinking <think>"
        elif model == "minimax":
            self.chat_template = "]~!b[ ]~b] [e~[ <mm:think>"
        else:
            self.chat_template = None

    def get_vocab(self) -> dict[str, int]:
        if self.model != "kimi":
            raise AssertionError("Template-backed tokenizers must not materialize their vocabulary during masking.")
        return {
            "<|open|>": 100,
            "<|close|>": 101,
            "<|sep|>": 102,
            "<|end_of_msg|>": 103,
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        marker_ids = {
            "<|assistant|>": [ASSISTANT_START],
            "<|system|>": [40],
            "<|user|>": [USER_START],
            "<|observation|>": [TOOL_RESULT],
            '<|open|>message role="assistant"<|sep|>': [ASSISTANT_START],
            "<|close|>message<|sep|><|end_of_msg|>": [TURN_END],
            "]~b]ai\n": [ASSISTANT_START],
            "[e~[\n": [TURN_END],
            "]~!b[]~b]system\n": [40],
            "]~b]developer\n": [41],
            "]~b]user\n": [USER_START],
            "]~b]tool": [TOOL_RESULT],
        }
        return marker_ids.get(text, [TEXT_IDS.get(text, 99)])

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    def _assistant_tokens(
        self, message: dict[str, Any], kwargs: dict[str, Any], index: int, last_user: int
    ) -> list[int]:
        result = [ASSISTANT_START]
        reasoning = message.get("reasoning_content")
        content = message.get("content")
        if self.model == "glm":
            clear_thinking = kwargs.get("clear_thinking", True)
            visible_reasoning = reasoning if (not clear_thinking or index > last_user) else None
            result.extend([THINK_OPEN, *([TEXT_IDS[visible_reasoning]] if visible_reasoning else []), THINK_CLOSE])
        elif self.model == "kimi":
            if kwargs.get("thinking", True):
                result.extend([THINK_OPEN, *([TEXT_IDS[reasoning]] if reasoning else []), THINK_CLOSE])
            result.append(RESPONSE_OPEN)
        else:
            if reasoning:
                result.extend([THINK_OPEN, TEXT_IDS[reasoning], THINK_CLOSE])
            else:
                result.append(THINK_CLOSE)

        if isinstance(content, str) and content:
            result.append(TEXT_IDS[content])
        if self.model == "kimi":
            result.append(RESPONSE_CLOSE)
        for tool_call in message.get("tool_calls", []):
            arguments = tool_call["function"]["arguments"]
            self.tool_arguments.append(dict(arguments.items()))
            result.append(TOOL_CALL)
        if self.model != "glm":
            result.append(TURN_END)
        return result

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        thinking: bool = True,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> dict[str, list[int]] | list[int]:
        assert tokenize is True
        assert add_generation_prompt is False
        if self.model == "kimi":
            kwargs["thinking"] = thinking
        truncation = kwargs.pop("truncation", False)
        max_length = kwargs.pop("max_length", None)
        self.template_kwargs.append(dict(kwargs))
        last_user = max(
            (index for index, message in enumerate(conversation) if message["role"] == "user"),
            default=-1,
        )
        input_ids: list[int] = []
        for index, message in enumerate(conversation):
            role = message["role"]
            if role == "assistant":
                input_ids.extend(self._assistant_tokens(message, kwargs, index, last_user))
            elif role == "tool":
                input_ids.extend([TOOL_RESULT, TEXT_IDS[message["content"]], TURN_END])
            else:
                input_ids.extend([USER_START, TEXT_IDS[message["content"]], TURN_END])
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return {"input_ids": input_ids} if return_dict else input_ids


def _messages(*, tools: bool, thinking: bool) -> list[dict[str, Any]]:
    if tools:
        return [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "reasoning_content": "reason one" if thinking else None,
                "content": None,
                "tool_calls": [{"function": {"name": "weather", "arguments": '{"city":"Seattle"}'}}],
            },
            {"role": "tool", "content": "12 C"},
            {
                "role": "assistant",
                "reasoning_content": "reason two" if thinking else None,
                "content": "Seattle is 12 C.",
            },
        ]
    if thinking:
        return [
            {"role": "user", "content": "question one"},
            {"role": "assistant", "reasoning_content": "reason one", "content": "answer one"},
            {"role": "user", "content": "question two"},
            {"role": "assistant", "reasoning_content": "reason two", "content": "answer two"},
        ]
    return [
        {"role": "user", "content": "question one"},
        {"role": "assistant", "content": "answer one"},
    ]


def _assistant_prefix(model: str, reasoning: str | None) -> list[int]:
    if model == "glm":
        return [THINK_OPEN, *([TEXT_IDS[reasoning]] if reasoning else []), THINK_CLOSE]
    if model == "kimi":
        if reasoning:
            return [THINK_OPEN, TEXT_IDS[reasoning], THINK_CLOSE, RESPONSE_OPEN]
        return [RESPONSE_OPEN]
    if reasoning:
        return [THINK_OPEN, TEXT_IDS[reasoning], THINK_CLOSE]
    return [THINK_CLOSE]


def _assistant_suffix(model: str) -> list[int]:
    if model == "glm":
        return []
    if model == "kimi":
        return [RESPONSE_CLOSE, TURN_END]
    return [TURN_END]


@pytest.mark.parametrize("model", ["glm", "kimi", "minimax"])
@pytest.mark.parametrize("tools", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_model_chat_four_way_rendering_and_loss_boundaries(model: str, tools: bool, thinking: bool) -> None:
    tokenizer = _ModelChatTokenizer(model)
    kwargs: dict[str, Any]
    if model == "minimax":
        kwargs = {"thinking_mode": "enabled" if thinking else "disabled"}
    else:
        kwargs = {"enable_thinking": thinking}
    if model == "glm" and thinking:
        kwargs["truncate_history_thinking"] = False
    example = {
        "messages": _messages(tools=tools, thinking=thinking),
        "chat_template_kwargs": kwargs,
    }
    if tools:
        example["tools"] = [{"type": "function", "function": {"name": "weather"}}]

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)
    batch = text_chat_collate_fn([example], tokenizer, warn_on_all_masked=False)

    reasoning_one = "reason one" if thinking else None
    reasoning_two = "reason two" if thinking else None
    if tools:
        expected = [
            *_assistant_prefix(model, reasoning_one),
            *([RESPONSE_CLOSE] if model == "kimi" else []),
            TOOL_CALL,
            *([] if model == "glm" else [TURN_END]),
            *_assistant_prefix(model, reasoning_two),
            TEXT_IDS["Seattle is 12 C."],
            *_assistant_suffix(model),
        ]
    elif thinking:
        expected = [
            *_assistant_prefix(model, "reason one"),
            TEXT_IDS["answer one"],
            *_assistant_suffix(model),
            *_assistant_prefix(model, "reason two"),
            TEXT_IDS["answer two"],
            *_assistant_suffix(model),
        ]
    else:
        expected = [*_assistant_prefix(model, None), TEXT_IDS["answer one"], *_assistant_suffix(model)]

    supervised_ids = tokenized.input_ids[tokenized.assistant_mask].tolist()
    assert supervised_ids == expected
    assert batch["labels"][0][batch["loss_mask"][0].bool()].tolist() == expected
    assert ASSISTANT_START not in supervised_ids
    assert TOOL_RESULT not in supervised_ids
    if tools:
        assert tokenizer.tool_arguments
        assert all(arguments == {"city": "Seattle"} for arguments in tokenizer.tool_arguments)

    semantic_kwargs = [entry for entry in tokenizer.template_kwargs if entry]
    if model == "glm":
        assert semantic_kwargs
        assert all(entry["enable_thinking"] is thinking for entry in semantic_kwargs)
    if model == "glm" and thinking:
        assert all(entry["clear_thinking"] is False for entry in semantic_kwargs)
        assert all("truncate_history_thinking" not in entry for entry in semantic_kwargs)
    if model == "kimi":
        assert semantic_kwargs
        assert all(entry["thinking"] is thinking for entry in semantic_kwargs)
        assert all("enable_thinking" not in entry for entry in semantic_kwargs)


def test_glm_history_thinking_truncation_preserves_historical_answer_loss() -> None:
    tokenizer = _ModelChatTokenizer("glm")
    example = {
        "messages": _messages(tools=False, thinking=True),
        "chat_template_kwargs": {
            "enable_thinking": True,
            "truncate_history_thinking": True,
        },
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)

    assert tokenized.input_ids[tokenized.assistant_mask].tolist() == [
        THINK_OPEN,
        THINK_CLOSE,
        TEXT_IDS["answer one"],
        THINK_OPEN,
        TEXT_IDS["reason two"],
        THINK_CLOSE,
        TEXT_IDS["answer two"],
    ]


@pytest.mark.parametrize("model", ["glm", "kimi", "minimax"])
def test_model_chat_right_truncation_and_padding_preserve_loss_alignment(model: str) -> None:
    tokenizer = _ModelChatTokenizer(model)
    thinking_kwargs: dict[str, Any]
    plain_kwargs: dict[str, Any]
    if model == "minimax":
        thinking_kwargs = {"thinking_mode": "enabled"}
        plain_kwargs = {"thinking_mode": "disabled"}
    else:
        thinking_kwargs = {"enable_thinking": True}
        plain_kwargs = {"enable_thinking": False}
    if model == "glm":
        thinking_kwargs["truncate_history_thinking"] = False

    thinking_example = {
        "messages": _messages(tools=False, thinking=True),
        "chat_template_kwargs": thinking_kwargs,
    }
    plain_example = {
        "messages": _messages(tools=False, thinking=False),
        "chat_template_kwargs": plain_kwargs,
    }
    full = tokenize_chat_example(thinking_example, tokenizer, warn_on_all_masked=False)
    max_length = full.input_ids.numel() - 1
    truncated = tokenize_chat_example(
        thinking_example,
        tokenizer,
        max_length=max_length,
        warn_on_all_masked=False,
    )
    plain = tokenize_chat_example(plain_example, tokenizer, warn_on_all_masked=False)
    batch = text_chat_collate_fn(
        [thinking_example, plain_example],
        tokenizer,
        max_length=max_length,
        pad_to_max_length=True,
        warn_on_all_masked=False,
    )

    assert truncated.input_ids.tolist() == full.input_ids[:-1].tolist()
    assert truncated.assistant_mask.tolist() == full.assistant_mask[:-1].tolist()
    assert batch["input_ids"].shape == (2, max_length)
    assert (
        batch["labels"][0][batch["loss_mask"][0].bool()].tolist()
        == truncated.input_ids[truncated.assistant_mask].tolist()
    )
    assert batch["attention_mask"][1].sum().item() == plain.input_ids.numel()
    assert not batch["loss_mask"][1, plain.input_ids.numel() :].any()


def test_minimax_adaptive_thinking_mode_is_forwarded_without_boolean_coercion() -> None:
    tokenizer = _ModelChatTokenizer("minimax")
    example = {
        "messages": _messages(tools=False, thinking=True),
        "chat_template_kwargs": {"thinking_mode": "adaptive"},
    }

    tokenized = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)

    assert tokenizer.template_kwargs
    assert all(kwargs["thinking_mode"] == "adaptive" for kwargs in tokenizer.template_kwargs)
    assert tokenized.input_ids[tokenized.assistant_mask].tolist() == [
        THINK_OPEN,
        TEXT_IDS["reason one"],
        THINK_CLOSE,
        TEXT_IDS["answer one"],
        TURN_END,
        THINK_OPEN,
        TEXT_IDS["reason two"],
        THINK_CLOSE,
        TEXT_IDS["answer two"],
        TURN_END,
    ]
