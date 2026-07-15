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

"""Shared text-only collators for direct SFT."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch

from megatron.bridge.data.conversation_processing import (
    build_shifted_labels_and_loss_mask,
    ensure_position_ids,
    get_processor_tokenizer,
    infer_assistant_mask_boundary_config,
    tokenize_chat_example,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.sft_processing import (
    PromptCompletionSFTPreprocessingConfig,
    sft_example_metadata,
    tokenize_prompt_completion_example,
)
from megatron.bridge.data.token_utils import extract_skipped_token_ids


_CONVERSATION_KEYS = ("conversation", "messages", "conversations")


@dataclass(frozen=True, kw_only=True)
class _TokenizedTextSFTRow:
    """Backend-neutral tokenized text row consumed by direct-HF batching."""

    input_ids: torch.Tensor
    loss_mask: torch.Tensor


def _pad_tokenized_rows(
    rows: list[_TokenizedTextSFTRow],
    *,
    pad_token_id: int,
    max_length: int | None,
    pad_to_max_length: bool,
    pad_to_multiple_of: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad shared row preprocessing outputs for direct-HF batching."""
    target_length = max(row.input_ids.numel() for row in rows)
    if pad_to_max_length and max_length is not None:
        target_length = max_length
    elif pad_to_multiple_of > 1:
        target_length = ((target_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
        if max_length is not None:
            target_length = min(target_length, max_length)

    input_ids = torch.full((len(rows), target_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), target_length), dtype=torch.long)
    loss_mask = torch.zeros((len(rows), target_length), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        row_length = min(row.input_ids.numel(), target_length)
        input_ids[row_index, :row_length] = row.input_ids[:row_length]
        attention_mask[row_index, :row_length] = 1
        loss_mask[row_index, :row_length] = row.loss_mask[:row_length]
    return input_ids.contiguous(), attention_mask.contiguous(), loss_mask.contiguous()


def _metadata_from_example(example: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in example.items() if key not in _CONVERSATION_KEYS}


def _build_text_sft_batch(
    examples: list[Mapping[str, Any]],
    tokenized_rows: list[_TokenizedTextSFTRow],
    tokenizer: Any,
    skipped_tokens: torch.Tensor,
    metadata: list[dict[str, Any]],
    *,
    max_length: int | None,
    pad_to_max_length: bool,
    pad_to_multiple_of: int,
    ignore_index: int,
    enable_in_batch_packing: bool,
    in_batch_packing_pad_to_multiple_of: int,
) -> dict[str, Any]:
    """Build one padded or in-batch-packed direct-HF text batch.

    Packed batches use their emergent aggregate width; ``pad_to_max_length``
    applies only to the non-packed path.
    """
    if not examples:
        raise ValueError("Text SFT collators require at least one example.")
    if enable_in_batch_packing:
        sequence_rows = []
        token_counts = []
        for tokenized_row in tokenized_rows:
            input_ids = tokenized_row.input_ids
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
            labels, shifted_loss_mask = build_shifted_labels_and_loss_mask(
                input_ids,
                tokenized_row.loss_mask,
                skipped_tokens,
                ignore_index=ignore_index,
            )
            sequence_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "position_ids": torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long),
                    "labels": labels,
                    "loss_mask": shifted_loss_mask,
                }
            )
            token_counts.append(int(attention_mask.sum().item()))

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        batch = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=max_length,
            pad_token_id=0 if pad_token_id is None else int(pad_token_id),
            ignore_index=ignore_index,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        batch["tokens"] = batch["input_ids"]
        batch["metadata"] = metadata
        batch["token_count"] = token_counts
        return batch

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    input_ids, attention_mask, loss_mask = _pad_tokenized_rows(
        tokenized_rows,
        pad_token_id=0 if pad_token_id is None else int(pad_token_id),
        max_length=max_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    labels, shifted_loss_mask = build_shifted_labels_and_loss_mask(
        input_ids,
        loss_mask.to(dtype=torch.float32),
        skipped_tokens,
        ignore_index=ignore_index,
    )
    ensure_position_ids(batch)
    batch["tokens"] = input_ids
    batch["labels"] = labels
    batch["loss_mask"] = shifted_loss_mask
    batch["metadata"] = metadata
    batch["token_count"] = [int(count) for count in attention_mask.sum(dim=1).tolist()]
    return batch


def text_chat_collate_fn(
    examples: list[Mapping[str, Any]],
    processor: Any,
    *,
    max_length: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 1,
    warn_on_all_masked: bool = True,
    loss_mode: Literal["assistant", "last_turn", "full"] = "assistant",
    ignore_index: int = IGNORE_INDEX,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate text-only HF chat examples using the shared assistant-mask path.

    Args:
        examples: HF-style chat rows containing ``messages``, ``conversation``,
            or legacy ``conversations``. Optional top-level ``tools`` are
            forwarded to the chat template for rendering and assistant masks.
        processor: A HF tokenizer or processor. It must expose
            ``apply_chat_template`` directly or through ``processor.tokenizer``.
        max_length: Optional tokenizer truncation length.
        sequence_length: Optional tokenizer truncation length used by
            Direct Hugging Face SFT builders.
        pad_to_max_length: On non-packed batches, pad every row to ``max_length``
            instead of the longest row. Packed batches use emergent width.
        pad_to_multiple_of: Optional non-packed padding multiple. The HF
            SFT builder uses this to keep CP/SP slices shape-compatible.
        warn_on_all_masked: Forwarded to assistant-mask construction.
        loss_mode: Chat tokens that contribute to loss.
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
    boundary_config = infer_assistant_mask_boundary_config(processor)
    skipped_tokens = extract_skipped_token_ids(processor)
    tokenized_rows = []
    for example in examples:
        tokenized = tokenize_chat_example(
            example,
            processor,
            max_length=max_length,
            skipped_tokens=skipped_tokens,
            boundary_config=boundary_config,
            warn_on_all_masked=warn_on_all_masked,
            loss_mode=loss_mode,
        )
        tokenized_rows.append(
            _TokenizedTextSFTRow(
                input_ids=tokenized.input_ids,
                loss_mask=tokenized.assistant_mask,
            )
        )

    return _build_text_sft_batch(
        examples,
        tokenized_rows,
        tokenizer,
        skipped_tokens,
        [_metadata_from_example(example) for example in examples],
        max_length=max_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        ignore_index=ignore_index,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
    )


def text_prompt_completion_collate_fn(
    examples: list[Mapping[str, Any]],
    processor: Any,
    *,
    preprocessing: PromptCompletionSFTPreprocessingConfig,
    max_length: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 1,
    ignore_index: int = IGNORE_INDEX,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
    **kwargs: Any,
) -> dict[str, Any]:
    """Collate prompt-completion rows without applying a chat template."""
    del kwargs

    max_length = max_length if max_length is not None else sequence_length
    tokenizer = get_processor_tokenizer(processor)
    skipped_tokens = extract_skipped_token_ids(processor)
    tokenized_rows = []
    for example in examples:
        tokenized = tokenize_prompt_completion_example(
            example,
            processor,
            preprocessing,
            max_length=max_length,
            skipped_tokens=skipped_tokens,
        )
        tokenized_rows.append(
            _TokenizedTextSFTRow(
                input_ids=tokenized.input_ids,
                loss_mask=tokenized.loss_mask,
            )
        )
    metadata = [sft_example_metadata(example, preprocessing) for example in examples]
    return _build_text_sft_batch(
        examples,
        tokenized_rows,
        tokenizer,
        skipped_tokens,
        metadata,
        max_length=max_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        ignore_index=ignore_index,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
    )
