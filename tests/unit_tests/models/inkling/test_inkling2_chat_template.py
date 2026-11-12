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

"""Bridge's Inkling2 chat template against the release's chat_template.jinja: framing, tools, reasoning, masks."""

from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from megatron.bridge.data.conversation_processing import normalize_chat_conversation
from megatron.bridge.data.sources.hf_adapters import _openmathinstruct2_thinking_adapter
from megatron.bridge.models.inkling.inkling2_chat_template import INKLING2_CHAT_TEMPLATE


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
    """A word-level tokenizer that knows the Inkling2 framing tokens."""
    vocab = {token: index for index, token in enumerate(("<unk>", "<pad>", *FRAMING))}
    model = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    model.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=model, unk_token="<unk>", pad_token="<pad>")
    fast.add_special_tokens({"additional_special_tokens": list(FRAMING)})
    fast.chat_template = INKLING2_CHAT_TEMPLATE
    return fast


def _render(tokenizer: PreTrainedTokenizerFast, messages: list[dict], template: str | None = None, **kwargs) -> str:
    return tokenizer.apply_chat_template(messages, chat_template=template, tokenize=False, **kwargs)


class TestInkling2ChatTemplate:
    @pytest.mark.parametrize("effort", [None, "none", "low", 0.7, 0])
    @pytest.mark.parametrize("name", sorted(CONVERSATIONS))
    def test_renders_as_the_release_template(self, tokenizer: PreTrainedTokenizerFast, name: str, effort) -> None:
        kwargs = {"tools": TOOLS if name == "tool_use" else None}
        if effort is not None:
            kwargs["reasoning_effort"] = effort

        assert _render(tokenizer, CONVERSATIONS[name], **kwargs) == _render(
            tokenizer, CONVERSATIONS[name], RELEASE_TEMPLATE, **kwargs
        )

    def test_tool_use_framing(self, tokenizer: PreTrainedTokenizerFast) -> None:
        rendered = _render(tokenizer, CONVERSATION, tools=TOOLS)

        declaration = (
            '<|message_system|>tool_declare<|content_xml|>[{"description":"Wetter für eine Stadt","name":"get_weather",'
            '"parameters":{"properties":{"city":{"type":"string"}},"required":["city"],"type":"object"},'
            '"type":"function"},{"description":"","name":"finish","parameters":{},"type":"function"}]<|end_message|>'
        )
        # Tools first, then the system prompt, then the thinking-effort line before the first non-system message.
        assert rendered.startswith(
            declaration + "<|message_system|><|content_text|>You are a weather agent.<|end_message|>" + EFFORT
        )
        assert (
            '<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":'
            '{"city":"Zürich","units":{"speed":"km/h","temp":"C"}}}<|end_message|><|content_model_end_sampling|>'
        ) in rendered
        # A result without a name takes the name of the call it answers.
        assert '<|message_tool|>get_weather<|content_text|>{"temperature":58}<|end_message|>' in rendered

    def test_reasoning_reads_the_thinking_field(self, tokenizer: PreTrainedTokenizerFast) -> None:
        row = _openmathinstruct2_thinking_adapter(
            {"problem": "What is 2 + 2?", "generated_solution": "Add them: $\\boxed{4}$", "expected_answer": "4"}, {}
        )
        as_reasoning_content = [
            row["messages"][0],
            {"role": "assistant", "reasoning_content": "Add them:", "content": "#### 4"},
        ]

        assert _render(tokenizer, row["messages"]) == (
            EFFORT + "<|message_user|><|content_text|>What is 2 + 2?<|end_message|>"
            "<|message_model|><|content_thinking|>Add them:<|end_message|>"
            "<|message_model|><|content_text|>#### 4<|end_message|><|content_model_end_sampling|>"
        )
        assert _render(tokenizer, row["messages"]) == _render(tokenizer, as_reasoning_content, RELEASE_TEMPLATE)

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

    def test_reasoning_content_takes_precedence_over_the_aliases(self, tokenizer: PreTrainedTokenizerFast) -> None:
        both = [{"role": "assistant", "content": "4", "reasoning_content": "a", "thinking": "b", "reasoning": "c"}]
        alias = [{"role": "assistant", "content": "4", "reasoning": "c"}]

        assert "<|content_thinking|>a<|end_message|>" in _render(tokenizer, both)
        assert "<|content_thinking|>c<|end_message|>" in _render(tokenizer, alias)

    @pytest.mark.parametrize("effort", ["huge", 1.5])
    def test_unknown_efforts_and_roles_are_rejected(self, tokenizer: PreTrainedTokenizerFast, effort) -> None:
        with pytest.raises(Exception, match="reasoning_effort"):
            _render(tokenizer, CONVERSATIONS["plain"], reasoning_effort=effort)
        with pytest.raises(Exception, match="Unknown message role: developer"):
            _render(tokenizer, [{"role": "developer", "content": "x"}])

    def test_assistant_turns_are_supervised_and_the_rest_is_not(self, tokenizer: PreTrainedTokenizerFast) -> None:
        encoded = tokenizer.apply_chat_template(
            CONVERSATION, tools=TOOLS, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
        )
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
        masks = dict.fromkeys(SUPERVISED + UNSUPERVISED, set())
        for token, mask in zip(tokens, encoded["assistant_masks"], strict=True):
            if token in masks:
                masks[token] = masks[token] | {mask}

        assert all(masks[token] == {1} for token in SUPERVISED)
        assert all(masks[token] == {0} for token in UNSUPERVISED if token != "<|content_image|>")
        # Each supervised span runs from an assistant turn's first block to its end-of-sampling token.
        supervised = [token for token, mask in zip(tokens, encoded["assistant_masks"], strict=True) if mask]
        assert supervised.count("<|content_model_end_sampling|>") == 2
        assert supervised[0] == "<|message_model|>" and supervised[-1] == "<|content_model_end_sampling|>"
