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

"""Kimi VL collator implementations."""

import warnings
from typing import Any

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.conversation_processing import (
    AssistantMaskBoundaryConfig,
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
    get_processor_tokenizer,
    tokenize_text_without_special_tokens,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


KIMI_ASSISTANT_START = "<|im_assistant|>assistant<|im_middle|>"
KIMI_ASSISTANT_END = "<|im_end|>"
KIMI_THINK_OPEN = "<think>"
KIMI_THINK_CLOSE = "</think>"


def _kimi_assistant_mask_boundary_config(processor: Any) -> AssistantMaskBoundaryConfig:
    """Build Kimi assistant loss boundaries and trim only empty thinking blocks."""
    tokenizer = get_processor_tokenizer(processor)
    think_open_tokens = tokenize_text_without_special_tokens(tokenizer, KIMI_THINK_OPEN)
    think_close_tokens = tokenize_text_without_special_tokens(tokenizer, KIMI_THINK_CLOSE)
    empty_think_tokens = [*think_open_tokens, *think_close_tokens]
    trim_leading_token_sequences = (empty_think_tokens,) if think_open_tokens and think_close_tokens else ()

    return assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=KIMI_ASSISTANT_START,
        assistant_end=KIMI_ASSISTANT_END,
        trim_leading_token_sequences=trim_leading_token_sequences,
    )


def _expand_image_tokens_and_aligned_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor | None,
    grid_thws: torch.Tensor,
    media_token_id: int,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Expand image placeholder tokens and any aligned per-token loss mask."""
    merge_h, merge_w = merge_kernel_size

    # Calculate number of image tokens for each image: t * (h // merge_h) * (w // merge_w)
    feature_counts = []
    for grid_thw in grid_thws:
        t, h, w = (int(x) for x in grid_thw.tolist())
        feature_counts.append(t * (h // merge_h) * (w // merge_w))

    # Find placeholder positions
    placeholder_positions = (input_ids == media_token_id).nonzero(as_tuple=True)[0]
    if len(placeholder_positions) == 0:
        # No placeholder found, return as-is
        return input_ids, attention_mask, loss_mask

    if len(placeholder_positions) != len(feature_counts):
        warnings.warn(
            "Mismatch between image placeholder count and grid_thws rows during Kimi token expansion; "
            "expanding as many placeholders as have corresponding grid metadata.",
            stacklevel=2,
        )

    expanded_input_ids = []
    expanded_attention_mask = []
    expanded_loss_mask = [] if loss_mask is not None else None
    feature_idx = 0

    loss_mask_values = loss_mask.tolist() if loss_mask is not None else [None] * input_ids.shape[0]
    for token_id, mask_value, loss_mask_value in zip(input_ids.tolist(), attention_mask.tolist(), loss_mask_values):
        if token_id == media_token_id and feature_idx < len(feature_counts):
            expanded_input_ids.extend([media_token_id] * feature_counts[feature_idx])
            expanded_attention_mask.extend([1] * feature_counts[feature_idx])
            if expanded_loss_mask is not None:
                expanded_loss_mask.extend([loss_mask_value] * feature_counts[feature_idx])
            feature_idx += 1
            continue

        expanded_input_ids.append(token_id)
        expanded_attention_mask.append(mask_value)
        if expanded_loss_mask is not None:
            expanded_loss_mask.append(loss_mask_value)

    expanded_input_ids = torch.tensor(
        expanded_input_ids,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    expanded_attention_mask = torch.tensor(
        expanded_attention_mask,
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    expanded_loss_mask_tensor = (
        torch.tensor(
            expanded_loss_mask,
            dtype=loss_mask.dtype,
            device=loss_mask.device,
        )
        if expanded_loss_mask is not None and loss_mask is not None
        else None
    )

    return expanded_input_ids, expanded_attention_mask, expanded_loss_mask_tensor


def _expand_image_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    grid_thws: torch.Tensor,
    media_token_id: int,
    merge_kernel_size: tuple[int, int] = (2, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand image placeholder tokens to the correct count based on grid_thws.

    For PP, this ensures the sequence length is fixed BEFORE the model forward pass,
    eliminating dynamic sequence expansion inside the model.

    Args:
        input_ids: (seq_len,) tensor with one placeholder per image
        attention_mask: (seq_len,) tensor
        grid_thws: (num_images, 3) tensor with [t, h, w] for each image
        media_token_id: Token ID of the image placeholder
        merge_kernel_size: Vision tower's patch merge kernel, default (2, 2)

    Returns:
        expanded_input_ids: Input IDs with placeholder expanded to N tokens
        expanded_attention_mask: Attention mask expanded accordingly
    """
    expanded_input_ids, expanded_attention_mask, _ = _expand_image_tokens_and_aligned_mask(
        input_ids,
        attention_mask,
        None,
        grid_thws,
        media_token_id,
        merge_kernel_size,
    )
    return expanded_input_ids, expanded_attention_mask


def kimi_k25_vl_collate_fn(
    examples: list[dict[str, Any]],
    processor,
    max_length: int | None = None,
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
    """Collate function for Kimi K2.5 VL processors with pre-expanded image tokens.

    For pipeline parallelism, this function:
    1. Processes each sample to get input_ids with 1 placeholder per image
    2. Pre-expands each placeholder to N tokens (N = t*(h//2)*(w//2) from grid_thws)
    3. Pads all sequences to fixed max_length
    This ensures the model forward pass doesn't change sequence length dynamically.
    """
    del visual_keys, min_pixels, max_pixels

    # Kimi SFT supervision is defined by the assistant-span loss mask. Do not
    # globally drop special tokens such as <|im_end|>, which the assistant must
    # learn to emit.
    skipped_tokens = torch.empty(0, dtype=torch.long)
    boundary_config = _kimi_assistant_mask_boundary_config(processor)

    # Get media token ID
    media_token_id = getattr(processor, "media_placeholder_token_id", None)
    if media_token_id is None and hasattr(processor, "tokenizer"):
        media_token_id = processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")
    if media_token_id is None:
        media_token_id = 163605  # Default for Kimi K2.5

    pad_token_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0

    # Get actual merge_kernel_size from processor's vision config
    merge_kernel_size = (2, 2)  # default fallback
    if hasattr(processor, "config") and hasattr(processor.config, "vision_config"):
        merge_kernel_size = getattr(processor.config.vision_config, "merge_kernel_size", (2, 2))
    elif hasattr(processor, "vision_config"):
        merge_kernel_size = getattr(processor.vision_config, "merge_kernel_size", (2, 2))

    # Process each sample individually
    all_expanded = []
    all_pixel_values = []
    all_grid_thws = []

    for i, example in enumerate(examples):
        conversation = example["conversation"]
        # Collect medias for this conversation
        medias = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        medias.append({"type": "image", "image": item.get("image")})

        template_kwargs = chat_template_kwargs_from_example(example)
        template_kwargs.setdefault("preserve_thinking", True)
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=False,
            tokenize=False,
            **template_kwargs,
        )

        processor_kwargs = {
            "text": text,
            "medias": medias,
            "return_tensors": "pt",
        }

        sample_batch = processor(**processor_kwargs)

        input_ids = sample_batch["input_ids"][0]
        attention_mask = sample_batch["attention_mask"][0]
        loss_mask = build_assistant_loss_mask(
            examples[i],
            input_ids,
            processor,
            skipped_tokens,
            boundary_config=boundary_config,
        )

        # Pre-expand image tokens if we have grid_thws
        if "grid_thws" in sample_batch and sample_batch["grid_thws"] is not None:
            grid_thws = sample_batch["grid_thws"]

            input_ids, attention_mask, loss_mask = _expand_image_tokens_and_aligned_mask(
                input_ids, attention_mask, loss_mask, grid_thws, media_token_id, merge_kernel_size
            )
            all_grid_thws.append(grid_thws)

        if "pixel_values" in sample_batch:
            all_pixel_values.append(sample_batch["pixel_values"])

        all_expanded.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "loss_mask": loss_mask,
            }
        )

    if enable_in_batch_packing:
        sequence_rows = []
        for sample in all_expanded:
            input_ids = sample["input_ids"]
            loss_mask = sample["loss_mask"].to(device=input_ids.device, dtype=torch.float32)
            shifted_loss_mask = torch.cat([loss_mask[1:], loss_mask.new_zeros(1)])
            labels = torch.cat([input_ids[1:], input_ids.new_full((1,), IGNORE_INDEX)])
            sequence_rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": sample["attention_mask"],
                    "position_ids": torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long),
                    "labels": labels.masked_fill(shifted_loss_mask == 0, IGNORE_INDEX),
                    "loss_mask": shifted_loss_mask,
                }
            )

        result = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=max_length if max_length is not None else sequence_length,
            pad_token_id=pad_token_id,
            ignore_index=IGNORE_INDEX,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        result["visual_inputs"] = GenericVisualInputs(
            pixel_values=torch.cat(all_pixel_values, dim=0) if all_pixel_values else None,
            image_grid_thw=torch.cat(all_grid_thws, dim=0) if all_grid_thws else None,
        )
        return result

    # Determine target length for padding
    expanded_lens = [b["input_ids"].shape[0] for b in all_expanded]
    batch_max = max(expanded_lens)

    if max_length is not None:
        target_len = max_length
    else:
        target_len = batch_max

    if batch_max > target_len:
        raise ValueError(
            f"Kimi VL collate refuses to truncate: max length {batch_max} > target {target_len}. "
            "Filter oversized records before collation."
        )

    # Pad to target_len
    padded_input_ids = []
    padded_attention_mask = []
    padded_loss_mask = []

    for batch in all_expanded:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        loss_mask = batch["loss_mask"]
        seq_len = input_ids.shape[0]

        if seq_len < target_len:
            # Pad
            pad_len = target_len - seq_len
            input_ids = torch.cat(
                [input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)]
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype, device=attention_mask.device)]
            )
            loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype, device=loss_mask.device)])

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)
        padded_loss_mask.append(loss_mask)

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
    }

    if all_pixel_values:
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
    if all_grid_thws:
        result["grid_thws"] = torch.cat(all_grid_thws, dim=0)  # (N, 3) with [t, h, w]

    if "position_ids" not in result:
        batch_size, seq_len = result["input_ids"].shape
        result["position_ids"] = (
            torch.arange(seq_len, device=result["input_ids"].device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
            .contiguous()
        )

    loss_mask = torch.stack(padded_loss_mask).to(device=result["input_ids"].device, dtype=torch.float32)
    labels = result["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, :1])], dim=1)
    result["labels"] = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    result["loss_mask"] = loss_mask

    visual_inputs = GenericVisualInputs(
        pixel_values=result.get("pixel_values"),
        pixel_values_videos=result.get("pixel_values_videos"),
        image_grid_thw=result.get("grid_thws"),
        video_grid_thw=result.get("video_grid_thw"),
    )
    for key in ("pixel_values", "pixel_values_videos", "grid_thws", "video_grid_thw"):
        result.pop(key, None)
    result["visual_inputs"] = visual_inputs
    prepare_sequence_batch(
        result,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
    )
    return result
