from __future__ import annotations

from typing import Any

import pytest

from megatron.bridge.data.collators.registry import (
    model_collate_required_for_processor,
    resolve_model_collate_for_processor,
)
from megatron.bridge.data.datasets.direct_sft import DirectSFTDataset
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


def _decode_character_ids(input_ids: list[int]) -> str:
    return "".join(chr(token_id) for token_id in input_ids if token_id != 0)


def test_deepseek_v4_encoder_matches_official_ordinary_modes():
    chat = encode_deepseek_v4_messages(ORDINARY_MESSAGES, thinking_mode="chat", drop_thinking=False)
    thinking = encode_deepseek_v4_messages(ORDINARY_MESSAGES, thinking_mode="thinking", drop_thinking=False)

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
        {"messages": REASONING_MESSAGES, "thinking_mode": mode, "preserve_thinking": True},
        tokenizer,
    )

    rendered = _decode_character_ids(tokenized.input_ids.tolist())
    supervised = _decode_character_ids(tokenized.input_ids[tokenized.assistant_mask].tolist())
    assert rendered.endswith(f"{ASSISTANT_TOKEN}{expected_assistant_text}")
    assert supervised == expected_assistant_text


def test_deepseek_v4_collator_shifts_labels_and_pads_rows():
    tokenizer = _DeepSeekV4CharacterTokenizer()
    examples = [
        {"messages": ORDINARY_MESSAGES, "enable_thinking": False},
        {"messages": REASONING_MESSAGES, "enable_thinking": True, "preserve_thinking": True},
    ]

    batch = deepseek_v4_collate_fn(examples, tokenizer, pad_to_multiple_of=8)

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] % 8 == 0
    assert batch["attention_mask"][1, -1].item() == 0
    assert batch["labels"][1, -1].item() == -100
    assert batch["loss_mask"][1, -1].item() == 0
    assert batch["tokens"].data_ptr() == batch["input_ids"].data_ptr()


def test_deepseek_v4_direct_sft_auto_selects_model_collator():
    tokenizer = _DeepSeekV4CharacterTokenizer()
    example = {"messages": REASONING_MESSAGES, "enable_thinking": True, "preserve_thinking": True}

    assert model_collate_required_for_processor(tokenizer)
    assert resolve_model_collate_for_processor(tokenizer) is deepseek_v4_collate_fn

    dataset = DirectSFTDataset([example], target_length=1, processor=tokenizer, pad_to_multiple_of=1)
    batch = dataset.collate_fn([dataset[0]])
    assert batch["input_ids"].shape[0] == 1
    assert batch["loss_mask"].sum().item() > 0


def test_deepseek_v4_collator_requires_explicit_thinking_mode():
    with pytest.raises(ValueError, match="requires thinking_mode"):
        deepseek_v4_collate_fn([{"messages": ORDINARY_MESSAGES}], _DeepSeekV4CharacterTokenizer())
