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

from typing import Any, Mapping

from megatron.bridge.data.conversation_processing import tokenize_chat_example


def tokenize_conversation(tokenizer, messages: list[dict], max_seq_length: int) -> tuple[list[int], int] | str:
    """Tokenize one conversation into ``(input_ids, context_len)``, or return a drop reason."""
    if len(messages) < 2 or messages[-1].get("role") != "assistant":
        return "no_assistant_completion"
    if not str(messages[-1].get("content") or "").strip():
        return "empty_completion"

    tokenized = tokenize_chat_example(
        messages,
        tokenizer,
        loss_mode="full",
        return_final_assistant_start=True,
        final_assistant_span="turn",
    )
    input_ids = tokenized.input_ids.tolist()
    context_len = tokenized.final_assistant_start

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None and input_ids and input_ids[-1] != eos_token_id:
        input_ids.append(eos_token_id)

    if context_len is None or not 1 <= context_len < len(input_ids):
        return "empty_completion"

    if len(input_ids) > max_seq_length:
        return "over_length"

    return input_ids, context_len


def _as_messages(value: str | list[dict], role: str) -> list[dict]:
    if isinstance(value, str):
        return [{"role": role, "content": value}]
    return list(value)


def build_pair_conversations(
    row: Mapping[str, Any],
    *,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    prompt_key: str | None = None,
) -> tuple[list[dict], list[dict]] | str:
    """Build the chosen/rejected conversations for one row, or return a drop reason."""
    chosen = _as_messages(row[chosen_key], "assistant")
    rejected = _as_messages(row[rejected_key], "assistant")

    if prompt_key is not None:
        prompt = _as_messages(row[prompt_key], "user")
        return prompt + chosen, prompt + rejected

    if chosen[:-1] != rejected[:-1]:
        return "context_mismatch"

    return chosen, rejected
