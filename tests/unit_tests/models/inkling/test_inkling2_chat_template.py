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

"""The release's Inkling2 chat template through Bridge's chat SFT pipeline: turn markers, masks, reasoning, tools."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from megatron.bridge.data.conversation_processing import (
    infer_assistant_mask_boundary_config,
    normalize_chat_conversation,
    tokenize_chat_example,
)
from megatron.bridge.data.sources.hf_adapters import _openmathinstruct2_thinking_adapter


pytestmark = pytest.mark.unit

# The release's template (thinkingmachines/Inkling-2-Small, revision 78cb6c0e, Apache-2.0), unchanged.
RELEASE_TEMPLATE = (Path(__file__).parent / "fixtures" / "inkling2_small_78cb6c0e_chat_template.jinja").read_text()
FRAMING = (
    "<|message_user|>",
    "<|message_model|>",
    "<|message_system|>",
    "<|message_tool|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_image|>",
    "<|content_xml|>",
    "<|content_invoke_tool_json|>",
    "<|content_model_end_sampling|>",
    "<|end_message|>",
    "<|unused_200054|>",
)
SUPERVISED = (
    "<|message_model|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_model_end_sampling|>",
)
UNSUPERVISED = ("<|message_user|>", "<|message_system|>", "<|message_tool|>", "<|content_xml|>", "<|content_image|>")
EFFORT = "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Wetter für eine Stadt",
            "parameters": {"type": "object", "required": ["city"], "properties": {"city": {"type": "string"}}},
        },
    },
    {"type": "function", "function": {"name": "finish", "parameters": None}},
]
CONVERSATION = [
    {"role": "system", "content": "You are a weather agent."},
    {"role": "user", "content": "What should I wear in SF and Zürich?"},
    {
        "role": "assistant",
        "reasoning_content": "Check both cities.",
        "content": "",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "SF"}}},
            {
                "id": "call_2",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"units": {"temp": "C", "speed": "km/h"}, "city": "Zürich"},
                },
            },
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": '{"temperature":58}'},
    {"role": "tool", "name": "get_weather", "content": "12 C, rain"},
    {"role": "assistant", "content": "A light jacket in SF, a raincoat in Zürich."},
]
CONVERSATIONS = {
    "tool_use": CONVERSATION,
    "plain": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
    "content_parts": [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is it?"}, "loose text"]},
        {"role": "assistant", "content": [{"type": "text", "text": "A"}, {"text": "B"}]},
    ],
    "empty_and_missing_content": [
        {"role": "system", "content": ""},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": None},
        {"role": "tool", "name": "noop", "content": ""},
    ],
    "named_assistant": [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "name": "f",
            "content": "",
            "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": {}}}],
        },
    ],
    "prompt_only": [{"role": "system", "content": "s"}, {"role": "user", "content": "What is 2 + 2?"}],
    "system_only": [{"role": "system", "content": "s"}],
}


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    """A word-level tokenizer that knows the Inkling2 framing tokens and carries the release's template."""
    vocab = {token: index for index, token in enumerate(("<unk>", "<pad>", *FRAMING))}
    model = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=model, unk_token="<unk>", pad_token="<pad>")
    fast.add_special_tokens({"additional_special_tokens": list(FRAMING)})
    fast.chat_template = RELEASE_TEMPLATE
    return fast


def _render(tokenizer: PreTrainedTokenizerFast, messages: list[dict], **kwargs) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def _expected_mask(tokens: list[str]) -> list[int]:
    """Supervise each assistant turn from its first ``<|message_model|>`` through ``<|content_model_end_sampling|>``."""
    mask, inside = [], False
    for token in tokens:
        inside = inside or token == "<|message_model|>"
        # An assistant turn without content renders only its end-of-sampling token, which is still the model's.
        mask.append(int(inside or token == "<|content_model_end_sampling|>"))
        if token == "<|content_model_end_sampling|>":
            inside = False
    return mask


class TestInkling2ChatTemplate:
    def test_bridge_finds_the_inkling2_turn_markers(self, tokenizer: PreTrainedTokenizerFast) -> None:
        config = infer_assistant_mask_boundary_config(tokenizer)

        assert config is not None
        assert tokenizer.convert_ids_to_tokens(list(config.role_start_tokens["assistant"])) == ["<|message_model|>"]
        assert tokenizer.convert_ids_to_tokens(list(config.role_end_tokens["assistant"])) == [
            "<|content_model_end_sampling|>"
        ]
        assert "assistant" in config.include_start_tokens_for_roles
        assert "assistant" in config.include_end_tokens_for_roles

    @pytest.mark.parametrize("effort", [None, "none", 0.7])
    @pytest.mark.parametrize(
        "name", sorted(name for name in CONVERSATIONS if name not in ("prompt_only", "system_only"))
    )
    def test_masks_cover_whole_assistant_turns(self, tokenizer: PreTrainedTokenizerFast, name: str, effort) -> None:
        example = {"messages": CONVERSATIONS[name], "tools": TOOLS if name == "tool_use" else None}
        if effort is not None:
            example["chat_template_kwargs"] = {"reasoning_effort": effort}
        encoded = tokenize_chat_example(example, tokenizer, warn_on_all_masked=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids.tolist())

        assert [int(value) for value in encoded.assistant_mask.tolist()] == _expected_mask(tokens)

    def test_tool_use_is_supervised_and_the_rest_is_not(self, tokenizer: PreTrainedTokenizerFast) -> None:
        encoded = tokenize_chat_example({"messages": CONVERSATION, "tools": TOOLS}, tokenizer)
        tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids.tolist())
        masks = dict.fromkeys(SUPERVISED + UNSUPERVISED, set())
        for token, mask in zip(tokens, encoded.assistant_mask.tolist(), strict=True):
            if token in masks:
                masks[token] = masks[token] | {int(mask)}

        assert all(masks[token] == {1} for token in SUPERVISED)
        assert all(masks[token] == {0} for token in UNSUPERVISED if token != "<|content_image|>")
        supervised = [token for token, mask in zip(tokens, encoded.assistant_mask.tolist(), strict=True) if mask]
        assert supervised.count("<|content_model_end_sampling|>") == 2
        assert supervised[0] == "<|message_model|>" and supervised[-1] == "<|content_model_end_sampling|>"
        # Tools first, then the system prompt, then the thinking-effort line before the first non-system message.
        rendered = _render(tokenizer, CONVERSATION, tools=TOOLS)
        assert rendered.index("tool_declare") < rendered.index("You are a weather agent.") < rendered.index(EFFORT)

    def test_openmath_thinking_rows_render_their_reasoning(self, tokenizer: PreTrainedTokenizerFast) -> None:
        row = _openmathinstruct2_thinking_adapter(
            {"problem": "What is 2 + 2?", "generated_solution": "Add them: $\\boxed{4}$", "expected_answer": "4"}, {}
        )
        encoded = tokenize_chat_example(row, tokenizer)
        tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids.tolist())

        assert _render(tokenizer, row["messages"]) == (
            EFFORT + "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
            "<|message_model|><|content_thinking|>Add them:<|end_message|>"
            "<|message_model|><|content_text|>#### 4<|end_message|><|content_model_end_sampling|>"
        )
        assert [int(value) for value in encoded.assistant_mask.tolist()] == _expected_mask(tokens)

    def test_openai_string_arguments_are_parsed_by_the_pipeline(self, tokenizer: PreTrainedTokenizerFast) -> None:
        call = {"id": "c", "type": "function", "function": {"name": "f", "arguments": '{"b": 1, "a": [2, 3]}'}}
        messages = normalize_chat_conversation(
            {"messages": [{"role": "assistant", "content": None, "tool_calls": [call]}]}
        )

        assert _render(tokenizer, messages) == (
            EFFORT
            + '<|message_model|>f<|content_invoke_tool_json|>{"name":"f","args":{"a":[2,3],"b":1}}<|end_message|>'
            "<|content_model_end_sampling|>"
        )
        with pytest.raises(Exception, match="must be a parsed object"):
            _render(tokenizer, [{"role": "assistant", "content": "", "tool_calls": [call]}])
