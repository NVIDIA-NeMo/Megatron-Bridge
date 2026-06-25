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

"""Generic text-only HF chat collator for the conversation dataset path."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX, _convert_to_openai_messages
from megatron.bridge.data.hf_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.data.sequence_packing import _pack_padded_sequence
from megatron.bridge.data.vlm_processing import (
    build_assistant_loss_mask,
    build_shifted_labels_and_loss_mask,
    ensure_position_ids,
    get_processor_tokenizer,
)


_CONVERSATION_KEYS = ("conversation", "messages", "conversations")


def _normalize_text_conversation(example: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "conversation" in example:
        conversation = example["conversation"]
    elif "messages" in example:
        conversation = example["messages"]
    elif "conversations" in example:
        conversation = _convert_to_openai_messages(dict(example))
    else:
        raise ValueError("Text chat examples must contain 'conversation', 'messages', or 'conversations'.")

    if not isinstance(conversation, Sequence) or isinstance(conversation, str):
        raise ValueError("Text chat conversation must be a list of message dictionaries.")

    normalized = []
    for turn in copy.deepcopy(list(conversation)):
        if not isinstance(turn, Mapping):
            raise ValueError("Text chat conversation turns must be dictionaries.")
        normalized.append(dict(turn))
    return normalized


def _render_chat(conversation: list[dict[str, Any]], processor: Any, tokenizer: Any) -> str:
    seen: set[int] = set()
    for template_owner in (tokenizer, processor):
        if id(template_owner) in seen:
            continue
        seen.add(id(template_owner))
        apply_chat_template = getattr(template_owner, "apply_chat_template", None)
        if apply_chat_template is None:
            continue
        try:
            return apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
        except TypeError:
            try:
                return apply_chat_template(conversation, tokenize=False)
            except TypeError:
                continue
    raise ValueError("Text chat collate requires a processor or tokenizer with apply_chat_template.")


def _call_tokenizer(
    tokenizer_or_processor: Any, texts: list[str], tokenizer_kwargs: dict[str, Any]
) -> Mapping[str, Any]:
    if not callable(tokenizer_or_processor):
        raise TypeError("tokenizer_or_processor is not callable.")
    try:
        return tokenizer_or_processor(text=texts, **tokenizer_kwargs)
    except TypeError:
        return tokenizer_or_processor(texts, **tokenizer_kwargs)


def _tokenize_texts(
    texts: list[str],
    processor: Any,
    tokenizer: Any,
    *,
    max_length: int | None,
    pad_to_max_length: bool,
) -> dict[str, Any]:
    tokenizer_kwargs: dict[str, Any] = {
        "padding": "max_length" if pad_to_max_length and max_length is not None else True,
        "return_tensors": "pt",
    }
    if max_length is not None:
        tokenizer_kwargs["max_length"] = max_length
        tokenizer_kwargs["truncation"] = True

    seen: set[int] = set()
    for tokenizer_or_processor in (tokenizer, processor):
        if id(tokenizer_or_processor) in seen:
            continue
        seen.add(id(tokenizer_or_processor))
        try:
            tokenized = _call_tokenizer(tokenizer_or_processor, texts, tokenizer_kwargs)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if "input_ids" in tokenized:
            return dict(tokenized)

    raise ValueError("Text chat collate could not tokenize rendered chat text.")


def _as_2d_long_tensor(value: Any) -> torch.Tensor:
    tensor = value if isinstance(value, torch.Tensor) else torch.tensor(value, dtype=torch.long)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.to(dtype=torch.long).contiguous()


def _tensorize_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    tensorized: dict[str, Any] = {}
    for key, value in batch.items():
        if key in {"input_ids", "attention_mask"}:
            tensorized[key] = _as_2d_long_tensor(value)
        elif isinstance(value, torch.Tensor):
            tensorized[key] = value.contiguous()
        else:
            tensorized[key] = value
    return tensorized


def _ensure_attention_mask(batch: dict[str, Any], tokenizer: Any) -> None:
    if "attention_mask" in batch:
        batch["attention_mask"] = _as_2d_long_tensor(batch["attention_mask"])
        return
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = 0
    batch["attention_mask"] = (batch["input_ids"] != int(pad_token_id)).to(dtype=torch.long)


def _metadata_from_example(example: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in example.items() if key not in _CONVERSATION_KEYS}


def text_chat_collate_fn(
    examples: list[Mapping[str, Any]],
    processor: Any,
    *,
    max_length: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    warn_on_all_masked: bool = True,
    ignore_index: int = IGNORE_INDEX,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate text-only HF chat examples using the shared assistant-mask path.

    Args:
        examples: HF-style chat rows containing ``messages``, ``conversation``,
            or legacy ``conversations``.
        processor: A HF tokenizer or processor. It must expose
            ``apply_chat_template`` directly or through ``processor.tokenizer``.
        max_length: Optional tokenizer truncation length.
        sequence_length: Optional tokenizer truncation length used by
            conversation-dataset providers.
        pad_to_max_length: If set with ``max_length``, pad every row to
            ``max_length`` instead of the longest row in the batch.
        warn_on_all_masked: Forwarded to assistant-mask construction.
        ignore_index: Label ignore value for masked targets.
        enable_in_batch_packing: If True, flatten the padded microbatch and emit
            packed-sequence metadata for GPT-style training steps.
        in_batch_packing_pad_to_multiple_of: Optional per-sequence length multiple
            used when ``enable_in_batch_packing`` inserts padding for CP/SP constraints.
        **kwargs: Additional common collate kwargs accepted for parity with
            VLM collate functions and ignored by the text-only path.

    Returns:
        Batch dictionary with VLM-style ``input_ids`` and GPT-style ``tokens``
        aliases, shifted ``labels`` and ``loss_mask``, ``position_ids``, and
        optional tokenizer fields such as ``attention_mask``.
    """
    del kwargs

    max_length = max_length if max_length is not None else sequence_length
    tokenizer = get_processor_tokenizer(processor)
    conversations = [_normalize_text_conversation(example) for example in examples]
    rendered_texts = [_render_chat(conversation, processor, tokenizer) for conversation in conversations]
    batch = _tensorize_batch(
        _tokenize_texts(
            rendered_texts,
            processor,
            tokenizer,
            max_length=max_length,
            pad_to_max_length=pad_to_max_length,
        )
    )
    batch["input_ids"] = _as_2d_long_tensor(batch["input_ids"])
    _ensure_attention_mask(batch, tokenizer)

    skipped_tokens = extract_skipped_token_ids(processor)
    loss_masks = [
        build_assistant_loss_mask(
            conversation,
            input_ids,
            processor,
            skipped_tokens,
            warn_on_all_masked=warn_on_all_masked,
        )
        for conversation, input_ids in zip(conversations, batch["input_ids"])
    ]
    loss_mask_t = torch.stack(loss_masks).to(device=batch["input_ids"].device, dtype=torch.float32)
    labels, shifted_loss_mask = build_shifted_labels_and_loss_mask(
        batch["input_ids"],
        loss_mask_t,
        skipped_tokens,
        ignore_index=ignore_index,
    )

    ensure_position_ids(batch)
    batch["tokens"] = batch["input_ids"]
    batch["labels"] = labels
    batch["loss_mask"] = shifted_loss_mask
    batch["metadata"] = [_metadata_from_example(example) for example in examples]
    batch["token_count"] = [int(count) for count in batch["attention_mask"].sum(dim=1).tolist()]
    if enable_in_batch_packing:
        # Transitional path: tokenizer output is already padded here. Future
        # text collates should construct packed layout directly.
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        _pack_padded_sequence(
            batch,
            pad_token_id=int(pad_token_id),
            ignore_index=ignore_index,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
    return batch
