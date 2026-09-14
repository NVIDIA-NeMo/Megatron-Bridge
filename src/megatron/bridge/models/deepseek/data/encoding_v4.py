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
#
# Portions are adapted from DeepSeek-V4's encoding_dsv4.py:
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/encoding_dsv4.py
#
# MIT License
# Copyright (c) 2023 DeepSeek
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DeepSeek-V4 conversation encoding for SFT.

This module implements the public encoding contract released with
``deepseek-ai/DeepSeek-V4-Flash`` and ``DeepSeek-V4-Pro``. The upstream model
repository is MIT licensed; this implementation is adapted to Bridge's typing,
validation, and OpenAI-style dataset contracts.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal


BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
THINKING_START_TOKEN = "<think>"
THINKING_END_TOKEN = "</think>"
DSML_TOKEN = "｜DSML｜"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"

_TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
_TOOL_CALLS_BLOCK_NAME = "tool_calls"
_REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root "
    "cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial "
    "scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered "
    "alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)
_TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""


def _to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _openai_tools(tools: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [tool["function"] if "function" in tool else tool for tool in tools]


def _openai_tool_calls(tool_calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for tool_call in tool_calls:
        function = tool_call.get("function", tool_call)
        normalized.append({"name": function["name"], "arguments": function.get("arguments", {})})
    return normalized


def _encode_arguments_to_dsml(tool_call: Mapping[str, Any]) -> str:
    raw_arguments = tool_call.get("arguments", {})
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError:
            arguments = {"arguments": raw_arguments}
    else:
        arguments = raw_arguments
    if not isinstance(arguments, Mapping):
        arguments = {"arguments": arguments}

    parameters = []
    for key, value in arguments.items():
        parameters.append(
            '<{dsml_token}parameter name="{key}" string="{is_string}">{value}</{dsml_token}parameter>'.format(
                dsml_token=DSML_TOKEN,
                key=key,
                is_string="true" if isinstance(value, str) else "false",
                value=value if isinstance(value, str) else _to_json(value),
            )
        )
    return "\n".join(parameters)


def _render_tools(tools: Sequence[Mapping[str, Any]]) -> str:
    return _TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(_to_json(tool) for tool in tools),
        dsml_token=DSML_TOKEN,
        thinking_start_token=THINKING_START_TOKEN,
        thinking_end_token=THINKING_END_TOKEN,
    )


def _last_user_index(messages: Sequence[Mapping[str, Any]]) -> int:
    return next(
        (index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") in {"user", "developer"}),
        -1,
    )


def _render_message(
    index: int,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: Literal["chat", "thinking"],
    truncate_history_thinking: bool,
    reasoning_effort: Literal["high", "max"] | None,
) -> str:
    if not 0 <= index < len(messages):
        raise IndexError(f"DeepSeek-V4 message index {index} is out of range.")

    message = messages[index]
    role = message.get("role")
    content = message.get("content")
    tools = message.get("tools")
    response_format = message.get("response_format")
    tool_calls = message.get("tool_calls")
    reasoning_content = message.get("reasoning_content")
    without_eos = bool(message.get("wo_eos", False))
    prompt = ""

    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
        prompt += _REASONING_EFFORT_MAX

    if role == "system":
        prompt += str(content or "")
        if tools:
            prompt += "\n\n" + _render_tools(_openai_tools(tools))
        if response_format:
            prompt += "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
            prompt += _to_json(response_format)
    elif role == "developer":
        if not content:
            raise ValueError("DeepSeek-V4 developer messages require non-empty content.")
        developer_content = USER_TOKEN + str(content)
        if tools:
            developer_content += "\n\n" + _render_tools(_openai_tools(tools))
        if response_format:
            developer_content += (
                "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n"
                + _to_json(response_format)
            )
        prompt += developer_content
    elif role == "user":
        prompt += USER_TOKEN
        content_blocks = message.get("content_blocks")
        if content_blocks:
            parts = []
            for block in content_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    parts.append(str(block.get("text", "")))
                elif block_type == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n\n".join(
                            str(item.get("text", ""))
                            if item.get("type") == "text"
                            else f"[Unsupported {item.get('type')}]"
                            for item in tool_content
                        )
                    parts.append(f"<tool_result>{tool_content}</tool_result>")
                else:
                    parts.append(f"[Unsupported {block_type}]")
            prompt += "\n\n".join(parts)
        else:
            prompt += str(content or "")
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_TOKEN + str(content or "")
    elif role == "tool":
        raise ValueError("DeepSeek-V4 tool messages must be merged before rendering.")
    elif role == "assistant":
        thinking_part = ""
        if thinking_mode == "thinking" and not (index > 0 and messages[index - 1].get("task") is not None):
            if not truncate_history_thinking or index > _last_user_index(messages):
                thinking_part = str(reasoning_content or "") + THINKING_END_TOKEN

        tool_call_content = ""
        if tool_calls:
            rendered_calls = [
                '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'.format(
                    dsml_token=DSML_TOKEN,
                    name=tool_call["name"],
                    arguments=_encode_arguments_to_dsml(tool_call),
                )
                for tool_call in _openai_tool_calls(tool_calls)
            ]
            tool_call_content = (
                f"\n\n<{DSML_TOKEN}{_TOOL_CALLS_BLOCK_NAME}>\n"
                + "\n".join(rendered_calls)
                + f"\n</{DSML_TOKEN}{_TOOL_CALLS_BLOCK_NAME}>"
            )

        prompt += thinking_part + str(content or "") + tool_call_content
        if not without_eos:
            prompt += EOS_TOKEN
    else:
        raise ValueError(f"Unsupported DeepSeek-V4 chat role: {role!r}.")

    if index + 1 < len(messages) and messages[index + 1].get("role") not in {"assistant", "latest_reminder"}:
        return prompt

    task = message.get("task")
    if task is not None:
        if task not in _TASK_TOKENS:
            raise ValueError(f"Unsupported DeepSeek-V4 quick-instruction task: {task!r}.")
        if task == "action":
            prompt += ASSISTANT_TOKEN
            prompt += THINKING_START_TOKEN if thinking_mode == "thinking" else THINKING_END_TOKEN
        prompt += _TASK_TOKENS[task]
    elif role in {"user", "developer"}:
        prompt += ASSISTANT_TOKEN
        if thinking_mode == "thinking" and (not truncate_history_thinking or index >= _last_user_index(messages)):
            prompt += THINKING_START_TOKEN
        else:
            prompt += THINKING_END_TOKEN
    return prompt


def _merge_tool_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for source_message in messages:
        message = copy.deepcopy(dict(source_message))
        role = message.get("role")
        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content", ""),
            }
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif role == "user":
            text_block = {"type": "text", "text": message.get("content", "")}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(text_block)
            else:
                normalized = {
                    "role": "user",
                    "content": message.get("content", ""),
                    "content_blocks": [text_block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in message:
                        normalized[key] = message[key]
                merged.append(normalized)
        else:
            merged.append(message)
    return merged


def _sort_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_call_order: dict[str, int] = {}
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            last_call_order = {
                str(tool_call.get("id") or tool_call.get("function", {}).get("id", "")): index
                for index, tool_call in enumerate(message["tool_calls"])
            }
        elif message.get("role") == "user" and message.get("content_blocks"):
            tool_blocks = [block for block in message["content_blocks"] if block.get("type") == "tool_result"]
            if len(tool_blocks) <= 1 or not last_call_order:
                continue
            sorted_blocks = iter(
                sorted(tool_blocks, key=lambda block: last_call_order.get(block.get("tool_use_id", ""), 0))
            )
            message["content_blocks"] = [
                next(sorted_blocks) if block.get("type") == "tool_result" else block
                for block in message["content_blocks"]
            ]
    return messages


def _drop_historical_thinking(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    last_user_index = _last_user_index(messages)
    retained = []
    for index, source_message in enumerate(messages):
        message = copy.deepcopy(dict(source_message))
        role = message.get("role")
        if role in {"user", "system", "tool", "latest_reminder", "direct_search_results"} or index >= last_user_index:
            retained.append(message)
        elif role == "assistant":
            message.pop("reasoning_content", None)
            retained.append(message)
    return retained


def encode_deepseek_v4_messages(
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking_mode: Literal["chat", "thinking"],
    truncate_history_thinking: bool = True,
    reasoning_effort: Literal["high", "max"] | None = None,
    add_bos: bool = True,
) -> str:
    """Encode OpenAI-style messages with the official DeepSeek-V4 contract.

    Args:
        messages: Structured chat turns. Tool definitions belong on a system or
            developer message; standalone tool results are merged automatically.
        thinking_mode: ``chat`` emits ``</think>`` immediately after the
            assistant marker; ``thinking`` emits explicit reasoning spans.
        truncate_history_thinking: Remove reasoning from historical assistant
            turns. The official contract disables this automatically when tools
            are present.
        reasoning_effort: Optional ``high`` or ``max`` reasoning policy.
        add_bos: Prepend the DeepSeek beginning-of-sequence token.

    Returns:
        Fully rendered DeepSeek-V4 conversation text ready for tokenization.
    """
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError("DeepSeek-V4 thinking_mode must be 'chat' or 'thinking'.")
    if reasoning_effort not in {None, "high", "max"}:
        raise ValueError("DeepSeek-V4 reasoning_effort must be 'high', 'max', or None.")

    normalized = _sort_tool_results(_merge_tool_messages(messages))
    effective_truncate_history = truncate_history_thinking and not any(message.get("tools") for message in normalized)
    if thinking_mode == "thinking" and effective_truncate_history:
        normalized = _drop_historical_thinking(normalized)

    prompt = BOS_TOKEN if add_bos else ""
    for index in range(len(normalized)):
        prompt += _render_message(
            index,
            normalized,
            thinking_mode=thinking_mode,
            truncate_history_thinking=effective_truncate_history,
            reasoning_effort=reasoning_effort,
        )
    return prompt
