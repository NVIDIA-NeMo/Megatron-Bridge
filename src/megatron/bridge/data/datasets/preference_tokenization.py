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
from typing import Any, Mapping, Sequence

from megatron.bridge.data.conversation_processing import tokenize_chat_example


def tokenize_conversation(
    tokenizer,
    messages: list[dict],
    max_seq_length: int,
    tools: Sequence[Mapping[str, Any]] | None = None,
    all_assistant_turns: bool = False,
) -> tuple[list[int], list[bool]] | str:
    """Tokenize one conversation into ``(input_ids, loss_mask)``, or return a drop reason.

    Scores the final assistant turn (header included) through its EOS; ``all_assistant_turns``
    also scores every earlier assistant turn. ``tools`` are rendered into the prompt, never scored.
    """
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return "no_assistant_completion"
    if not (str(messages[-1].get("content") or "").strip() or messages[-1].get("tool_calls")):
        return "empty_completion"

    tokenized = tokenize_chat_example(
        messages,
        tokenizer,
        tool_schemas=tools,
        loss_mode="assistant" if all_assistant_turns else "full",
        return_final_assistant_start=True,
        final_assistant_span="turn",
    )
    input_ids = tokenized.input_ids.tolist()
    context_len = tokenized.final_assistant_start
    if context_len is None or not 1 <= context_len < len(input_ids):
        return "empty_completion"
    loss_mask = [position >= context_len for position in range(len(input_ids))]
    if all_assistant_turns:
        loss_mask = [a or b for a, b in zip(loss_mask, tokenized.assistant_mask.tolist())]

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        eos_end = _rendered_eos_end(tokenizer, input_ids, context_len, eos_token_id)
        if eos_end is not None:
            # Templates whose turn terminator is the EOS (ChatML <|im_end|>) already rendered it; drop the
            # trailing newline after it instead of appending a second EOS that generation never produces.
            input_ids, loss_mask = input_ids[:eos_end], loss_mask[:eos_end]
        else:
            input_ids.append(eos_token_id)
            loss_mask.append(True)

    if len(input_ids) > max_seq_length:
        return "over_length"

    return input_ids, loss_mask


def _rendered_eos_end(tokenizer, input_ids: list[int], context_len: int, eos_token_id: int) -> int | None:
    """End (exclusive) of a final-turn EOS followed by nothing but whitespace, or None if there is none."""
    for position in range(len(input_ids) - 1, context_len - 1, -1):
        if input_ids[position] == eos_token_id:
            return position + 1
        if tokenizer.decode(input_ids[position:]).strip():
            return None
    return None


def _as_messages(value: str | Sequence[Mapping[str, Any]], role: str) -> list[dict]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return [_decode_tool_calls(message) for message in value]


def _decode_tool_calls(message: Mapping[str, Any]) -> dict[str, Any]:
    """HF ``datasets`` stores ``tool_calls`` as a JSON string when message schemas differ; the template needs the list."""
    message = dict(message)
    if isinstance(message.get("tool_calls"), str):
        message["tool_calls"] = json.loads(message["tool_calls"])
    if message.get("tool_calls"):
        message["tool_calls"] = [_decode_arguments(call) for call in message["tool_calls"]]
    return message


def _decode_arguments(call: Mapping[str, Any]) -> dict[str, Any]:
    function = call.get("function")
    if not isinstance(function, Mapping) or not isinstance(function.get("arguments"), str):
        return dict(call)
    return {**call, "function": {**function, "arguments": json.loads(function["arguments"])}}


def build_pair_conversations(
    row: Mapping[str, Any],
    *,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    prompt_key: str | None = None,
) -> tuple[list[dict], list[dict]] | str:
    """Build the chosen/rejected conversations for one row, or return a drop reason.

    Full trajectories (no ``prompt_key``) must first differ at an assistant turn; the rest is shared context.
    """
    chosen = _as_messages(row[chosen_key], "assistant")
    rejected = _as_messages(row[rejected_key], "assistant")

    if prompt_key is not None:
        prompt = _as_messages(row[prompt_key], "user")
        return prompt + chosen, prompt + rejected

    split = next((i for i, (c, r) in enumerate(zip(chosen, rejected)) if c != r), None)
    if split is None or chosen[split].get("role") != "assistant" or rejected[split].get("role") != "assistant":
        return "context_mismatch"

    return chosen, rejected
