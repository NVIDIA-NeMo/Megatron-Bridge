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

"""Gemma VL collator implementations."""

from collections.abc import Sequence
from typing import Any

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.hf_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.data.sequence_batching import pad_or_pack_sequence
from megatron.bridge.data.vlm_datasets.collate_utils import PASSTHROUGH_VISUAL_KEYS, THW_GRID_VISUAL_KEYS
from megatron.bridge.data.vlm_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    infer_assistant_mask_boundary_config,
)
from megatron.bridge.models.ministral3.data.collate_fn import ministral3_collate_fn
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def gemma3_vl_collate_fn(
    examples: list,
    processor,
    *,
    visual_keys: Sequence[str] = THW_GRID_VISUAL_KEYS,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Gemma3 VL models."""
    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = infer_assistant_mask_boundary_config(processor)

    # If pad_token remains unset after the eos_token fallback, disable padding to
    # avoid a ValueError from apply_chat_template.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    can_pad = tokenizer is not None and tokenizer.pad_token is not None

    tokenizer_for_padding = getattr(processor, "tokenizer", processor)
    saved_padding_side = getattr(tokenizer_for_padding, "padding_side", None)
    if tokenizer_for_padding is not None:
        tokenizer_for_padding.padding_side = "right"
    try:
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "padding": can_pad,
            "truncation": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if min_pixels is not None:
            template_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            template_kwargs["max_pixels"] = max_pixels
        batch = processor.apply_chat_template([example["conversation"] for example in examples], **template_kwargs)
    finally:
        if tokenizer_for_padding is not None and saved_padding_side is not None:
            tokenizer_for_padding.padding_side = saved_padding_side

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
            .contiguous()
        )

    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
            )
            for example, input_ids in zip(examples, batch["input_ids"])
        ]
    ).to(device=batch["input_ids"].device, dtype=torch.float32)
    labels = batch["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, :1])], dim=1)
    batch["labels"] = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask

    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

    visual_kwargs = {}
    for key in visual_keys:
        if key in batch:
            visual_kwargs[key] = batch[key]
    visual_inputs = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None
    for key in PASSTHROUGH_VISUAL_KEYS:
        batch.pop(key, None)
    batch["visual_inputs"] = visual_inputs
    pad_or_pack_sequence(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
    )
    return batch


def gemma4_vl_collate_fn(
    examples: list,
    processor,
    *,
    visual_keys: object = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Gemma4 VL models."""
    return ministral3_collate_fn(
        examples,
        processor,
        visual_keys=visual_keys,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        assistant_mask_boundary_config=assistant_mask_boundary_config_from_markers(
            processor,
            assistant_start="<|turn>model\n",
            assistant_end="<turn|>",
        ),
    )
