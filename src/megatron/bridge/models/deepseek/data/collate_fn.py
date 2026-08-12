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

"""DeepSeek text-chat collators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from megatron.bridge.data.collators.sft import text_chat_collate_fn
from megatron.bridge.data.conversation_processing import (
    AssistantMaskBoundaryConfig,
    TokenizedConversation,
    apply_chat_loss_mode,
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    get_processor_tokenizer,
    normalize_chat_conversation,
)
from megatron.bridge.models.deepseek.data.encoding_v4 import (
    ASSISTANT_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    encode_deepseek_v4_messages,
)


def _deepseek_v4_options(
    example: Mapping[str, Any],
) -> tuple[Literal["chat", "thinking"], bool, Literal["high", "max"] | None]:
    legacy_history_keys = sorted({"drop_thinking", "preserve_thinking"}.intersection(example))
    if legacy_history_keys:
        joined_keys = ", ".join(legacy_history_keys)
        raise ValueError(f"DeepSeek-V4 rows use truncate_history_thinking instead of legacy option(s): {joined_keys}.")
    thinking_mode = example.get("thinking_mode")
    if thinking_mode is None:
        enable_thinking = example.get("enable_thinking")
        if not isinstance(enable_thinking, bool):
            raise ValueError("DeepSeek-V4 rows require thinking_mode or enable_thinking.")
        thinking_mode = "thinking" if enable_thinking else "chat"
    truncate_history_thinking = example.get("truncate_history_thinking", True)
    if not isinstance(truncate_history_thinking, bool):
        raise ValueError("DeepSeek-V4 truncate_history_thinking must be a boolean.")
    reasoning_effort = example.get("reasoning_effort")
    return thinking_mode, truncate_history_thinking, reasoning_effort


def _normalize_deepseek_v4_conversation(
    example_or_conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Add the omitted content field used by OpenAI tool-call-only turns."""
    if isinstance(example_or_conversation, Mapping):
        normalized_source = dict(example_or_conversation)
        for key in ("messages", "conversation", "conversations"):
            if normalized_source.get(key) is not None:
                messages = list(normalized_source[key])
                normalized_source[key] = messages
                break
        else:
            return normalize_chat_conversation(normalized_source)
    else:
        messages = list(example_or_conversation)
        normalized_source = messages
    for index, message in enumerate(messages):
        if isinstance(message, Mapping) and message.get("role") == "assistant" and "content" not in message:
            messages[index] = {**message, "content": None}
    return normalize_chat_conversation(normalized_source)


def _attach_tools(
    conversation: list[dict[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return conversation
    owner = next((message for message in conversation if message.get("role") in {"system", "developer"}), None)
    if owner is None:
        owner = {"role": "system", "content": ""}
        conversation.insert(0, owner)
    existing_tools = owner.get("tools")
    if existing_tools is not None and existing_tools != tools:
        raise ValueError("DeepSeek-V4 top-level tools conflict with tools already attached to the conversation.")
    owner["tools"] = list(tools)
    return conversation


def tokenize_deepseek_v4_example(
    example_or_conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    max_length: int | None = None,
    skipped_tokens: torch.Tensor | None = None,
    boundary_config: AssistantMaskBoundaryConfig | None = None,
    warn_on_all_masked: bool = True,
    loss_mode: Literal["assistant", "last_turn", "full"] = "assistant",
    **_: Any,
) -> TokenizedConversation:
    """Render and tokenize one DeepSeek-V4 chat row with the official encoder."""
    source = example_or_conversation if isinstance(example_or_conversation, Mapping) else {}
    conversation = _normalize_deepseek_v4_conversation(example_or_conversation)
    tools = source.get("tools")
    if tools is not None and (
        not isinstance(tools, Sequence)
        or isinstance(tools, (str, bytes))
        or not all(isinstance(tool, Mapping) for tool in tools)
    ):
        raise TypeError("DeepSeek-V4 tools must be a sequence of OpenAI-format tool dictionaries.")
    conversation = _attach_tools(conversation, tools)
    thinking_mode, truncate_history_thinking, reasoning_effort = _deepseek_v4_options(source)
    rendered = encode_deepseek_v4_messages(
        conversation,
        thinking_mode=thinking_mode,
        truncate_history_thinking=truncate_history_thinking,
        reasoning_effort=reasoning_effort,
    )

    tokenizer = get_processor_tokenizer(processor)
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)

    if loss_mode == "full":
        assistant_mask = torch.ones_like(input_tensor, dtype=torch.bool)
    else:
        boundary_config = boundary_config or assistant_mask_boundary_config_from_markers(
            processor,
            assistant_start=ASSISTANT_TOKEN,
            assistant_end=EOS_TOKEN,
            role_start_markers={"user": USER_TOKEN},
        )
        assistant_mask = build_assistant_loss_mask(
            {"conversation": conversation},
            input_tensor,
            processor,
            skipped_tokens,
            boundary_config=boundary_config,
            warn_on_all_masked=warn_on_all_masked,
        ).to(dtype=torch.bool)
    assistant_mask = apply_chat_loss_mode(
        assistant_mask,
        input_tensor,
        loss_mode=loss_mode,
        skipped_tokens=skipped_tokens,
    )
    if max_length is not None and input_tensor.numel() > max_length:
        if getattr(tokenizer, "truncation_side", "right") == "left":
            start = input_tensor.numel() - max_length
            input_tensor = input_tensor[start:]
            assistant_mask = assistant_mask[start:]
        else:
            input_tensor = input_tensor[:max_length]
            assistant_mask = assistant_mask[:max_length]
    return TokenizedConversation(
        input_ids=input_tensor,
        assistant_mask=assistant_mask,
        conversation=conversation,
    )


def deepseek_v4_collate_fn(
    examples: list[Mapping[str, Any]],
    processor: Any,
    *,
    sequence_length: int | None = None,
    max_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 1,
    warn_on_all_masked: bool = True,
    loss_mode: Literal["assistant", "last_turn", "full"] = "assistant",
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate DeepSeek-V4 chats without synthesizing a Jinja template."""
    return text_chat_collate_fn(
        examples,
        processor,
        max_length=max_length,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        warn_on_all_masked=warn_on_all_masked,
        loss_mode=loss_mode,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        tokenize_impl=tokenize_deepseek_v4_example,
        **kwargs,
    )
