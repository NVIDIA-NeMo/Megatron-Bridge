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

import copy
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


_DEEPSEEK_V4_TEMPLATE_OPTIONS = {
    "thinking_mode",
    "enable_thinking",
    "drop_thinking",
    "preserve_thinking",
    "reasoning_effort",
}


def _deepseek_v4_options(
    example: Mapping[str, Any],
    defaults: Mapping[str, Any] | None,
) -> tuple[Literal["chat", "thinking"], bool, Literal["high", "max"] | None]:
    options = dict(defaults or {})
    row_options = example.get("chat_template_kwargs")
    if row_options is not None:
        if not isinstance(row_options, Mapping):
            raise TypeError("DeepSeek-V4 chat_template_kwargs must be a mapping.")
        options.update(row_options)
    options.update({key: example[key] for key in _DEEPSEEK_V4_TEMPLATE_OPTIONS if key in example})

    unknown = set(options) - _DEEPSEEK_V4_TEMPLATE_OPTIONS
    if unknown:
        raise ValueError(f"Unsupported DeepSeek-V4 chat-template options: {sorted(unknown)}.")

    thinking_mode = options.get("thinking_mode")
    enable_thinking = options.get("enable_thinking")
    if enable_thinking is not None and not isinstance(enable_thinking, bool):
        raise TypeError("DeepSeek-V4 enable_thinking must be a boolean.")
    if thinking_mode is None and enable_thinking is not None:
        thinking_mode = "thinking" if enable_thinking else "chat"
    if thinking_mode not in {"chat", "thinking"}:
        raise ValueError(
            "DeepSeek-V4 chat preprocessing requires thinking_mode='chat'/'thinking' or enable_thinking=true/false."
        )
    if enable_thinking is not None and (thinking_mode == "thinking") != enable_thinking:
        raise ValueError("DeepSeek-V4 thinking_mode and enable_thinking disagree.")

    drop_thinking = options.get("drop_thinking")
    preserve_thinking = options.get("preserve_thinking")
    if drop_thinking is not None and not isinstance(drop_thinking, bool):
        raise TypeError("DeepSeek-V4 drop_thinking must be a boolean.")
    if preserve_thinking is not None and not isinstance(preserve_thinking, bool):
        raise TypeError("DeepSeek-V4 preserve_thinking must be a boolean.")
    if drop_thinking is None:
        drop_thinking = not preserve_thinking if preserve_thinking is not None else True
    elif preserve_thinking is not None and drop_thinking == preserve_thinking:
        raise ValueError("DeepSeek-V4 drop_thinking and preserve_thinking disagree.")

    reasoning_effort = options.get("reasoning_effort")
    if reasoning_effort not in {None, "high", "max"}:
        raise ValueError("DeepSeek-V4 reasoning_effort must be 'high', 'max', or None.")
    return thinking_mode, drop_thinking, reasoning_effort


def _attach_tools(
    conversation: list[dict[str, Any]],
    tools: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not tools:
        return conversation
    result = copy.deepcopy(conversation)
    owner = next((message for message in result if message.get("role") in {"system", "developer"}), None)
    if owner is None:
        owner = {"role": "system", "content": ""}
        result.insert(0, owner)
    existing_tools = owner.get("tools")
    if existing_tools is not None and existing_tools != tools:
        raise ValueError("DeepSeek-V4 top-level tools conflict with tools already attached to the conversation.")
    owner["tools"] = copy.deepcopy(list(tools))
    return result


def tokenize_deepseek_v4_example(
    example_or_conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    max_length: int | None = None,
    skipped_tokens: torch.Tensor | None = None,
    boundary_config: AssistantMaskBoundaryConfig | None = None,
    warn_on_all_masked: bool = True,
    loss_mode: Literal["assistant", "last_turn", "full"] = "assistant",
    chat_template_kwargs: Mapping[str, Any] | None = None,
    **_: Any,
) -> TokenizedConversation:
    """Render and tokenize one DeepSeek-V4 chat row with the official encoder."""
    if loss_mode not in {"assistant", "last_turn", "full"}:
        raise ValueError("Chat SFT loss_mode must be assistant, last_turn, or full.")
    source = example_or_conversation if isinstance(example_or_conversation, Mapping) else {}
    conversation = normalize_chat_conversation(example_or_conversation)
    tools = source.get("tools")
    if tools is not None and (
        not isinstance(tools, Sequence)
        or isinstance(tools, (str, bytes))
        or not all(isinstance(tool, Mapping) for tool in tools)
    ):
        raise TypeError("DeepSeek-V4 tools must be a sequence of OpenAI-format tool dictionaries.")
    conversation = _attach_tools(conversation, tools)
    thinking_mode, drop_thinking, reasoning_effort = _deepseek_v4_options(source, chat_template_kwargs)
    rendered = encode_deepseek_v4_messages(
        conversation,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
        reasoning_effort=reasoning_effort,
    )

    tokenizer = get_processor_tokenizer(processor)
    input_ids = tokenizer.encode(rendered, add_special_tokens=False)
    if max_length is not None and len(input_ids) > max_length:
        if getattr(tokenizer, "truncation_side", "right") == "left":
            input_ids = input_ids[-max_length:]
        else:
            input_ids = input_ids[:max_length]
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
    if loss_mode == "full" and skipped_tokens is not None and skipped_tokens.numel() > 0:
        assistant_mask &= ~torch.isin(input_tensor, skipped_tokens.to(dtype=torch.long))
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
    chat_template_kwargs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate DeepSeek-V4 chats without synthesizing a Jinja template."""

    def _tokenize(example: Mapping[str, Any], tokenizer: Any, **tokenize_kwargs: Any) -> TokenizedConversation:
        return tokenize_deepseek_v4_example(
            example,
            tokenizer,
            chat_template_kwargs=chat_template_kwargs,
            **tokenize_kwargs,
        )

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
        tokenize_impl=_tokenize,
        **kwargs,
    )
