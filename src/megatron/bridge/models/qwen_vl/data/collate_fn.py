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

"""Qwen VL collator implementations."""

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.collators.visual import THW_GRID_VISUAL_KEYS
from megatron.bridge.data.conversation_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


MISSING_QWEN_VL_UTILS_MSG = (
    "qwen_vl_utils is required for Qwen2.5 VL processing. Please `pip install qwen-vl-utils` or"
    " provide compatible vision preprocessing."
)
QWEN_VL_MIN_PIXELS = 200704
QWEN_VL_MAX_PIXELS = 1003520
CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_ASSISTANT_END = "<|im_end|>\n"
CHATML_OTHER_ROLE_STARTS = {role: f"<|im_start|>{role}\n" for role in ("system", "developer", "user", "tool")}
QWEN_VISUAL_KEYS = (*THW_GRID_VISUAL_KEYS, "second_per_grid_ts")

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False


def _normalize_qwen_video_paths(example: dict[str, Any]) -> dict[str, Any]:
    """Map path-based video parts to the inline schema expected by Qwen processors."""
    conversation = example.get("conversation")
    if not isinstance(conversation, list):
        return example

    normalized_conversation = []
    example_changed = False
    for turn in conversation:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("content"), list):
            normalized_conversation.append(turn)
            continue

        normalized_content = []
        turn_changed = False
        for part in turn["content"]:
            if isinstance(part, Mapping) and part.get("type") == "video" and "video" not in part and "path" in part:
                normalized_part = dict(part)
                normalized_part["video"] = normalized_part.pop("path")
                normalized_content.append(normalized_part)
                turn_changed = True
            else:
                normalized_content.append(part)

        if turn_changed:
            normalized_turn = dict(turn)
            normalized_turn["content"] = normalized_content
            normalized_conversation.append(normalized_turn)
            example_changed = True
        else:
            normalized_conversation.append(turn)

    if not example_changed:
        return example
    normalized_example = dict(example)
    normalized_example["conversation"] = normalized_conversation
    return normalized_example


def qwen2_5_collate_fn(
    examples: list,
    processor,
    min_pixels: int | None = QWEN_VL_MIN_PIXELS,
    max_pixels: int | None = QWEN_VL_MAX_PIXELS,
    visual_keys: object = None,
    require_assistant_matches: bool = False,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    del visual_keys

    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    examples = [_normalize_qwen_video_paths(example) for example in examples]
    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_ASSISTANT_END,
        assistant_end_fallbacks=("<|im_end|>",),
        role_start_markers=CHATML_OTHER_ROLE_STARTS,
    )

    texts = [
        processor.apply_chat_template(
            example["conversation"],
            tokenize=False,
            **chat_template_kwargs_from_example(example),
        )
        for example in examples
    ]
    # Build per-example media (list) and split by presence.  Qwen processors accept
    # nested per-example image/video lists; splitting avoids passing empty media
    # kwargs for text-only rows.
    per_example_images = []
    per_example_videos = []
    has_media = []

    for example in examples:
        imgs, videos = process_vision_info(example["conversation"])
        if imgs is None:
            imgs = []
        elif not isinstance(imgs, list):
            imgs = [imgs]
        if videos is None:
            videos = []
        elif not isinstance(videos, list):
            videos = [videos]
        per_example_images.append(imgs)
        per_example_videos.append(videos)
        has_media.append(len(imgs) > 0 or len(videos) > 0)

    if enable_in_batch_packing:
        sequence_rows = []
        visual_values: dict[str, list[torch.Tensor]] = {key: [] for key in QWEN_VISUAL_KEYS}
        with use_processor_right_padding(processor):
            for example, text, images, videos in zip(
                examples, texts, per_example_images, per_example_videos, strict=True
            ):
                processor_kwargs = {
                    "text": [text],
                    "padding": False,
                    "return_tensors": "pt",
                }
                if min_pixels is not None:
                    processor_kwargs["min_pixels"] = min_pixels
                if max_pixels is not None:
                    processor_kwargs["max_pixels"] = max_pixels
                if images:
                    processor_kwargs["images"] = images
                if videos:
                    processor_kwargs["videos"] = videos

                sample_batch = {
                    key: value.contiguous() if isinstance(value, torch.Tensor) else value
                    for key, value in processor(**processor_kwargs).items()
                }
                input_ids = sample_batch["input_ids"][0]
                attention_mask = sample_batch.get("attention_mask")
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids)
                else:
                    attention_mask = attention_mask[0]
                position_ids = sample_batch.get("position_ids")
                if position_ids is None:
                    position_ids = torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long)
                else:
                    position_ids = position_ids[0]

                loss_mask = build_assistant_loss_mask(
                    example,
                    input_ids,
                    processor,
                    skipped_tokens,
                    boundary_config=boundary_config,
                    warn_on_all_masked=not require_assistant_matches,
                ).to(device=input_ids.device, dtype=torch.float32)
                labels = torch.cat([input_ids[1:], input_ids.new_full((1,), IGNORE_INDEX)])
                if skipped_tokens.numel() > 0:
                    labels = labels.masked_fill(
                        torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX
                    )
                shifted_loss_mask = torch.cat([loss_mask[1:], loss_mask.new_zeros(1)])
                sequence_rows.append(
                    {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "labels": labels.masked_fill(shifted_loss_mask == 0, IGNORE_INDEX),
                        "loss_mask": shifted_loss_mask,
                    }
                )

                for key in QWEN_VISUAL_KEYS:
                    value = sample_batch.get(key)
                    if not isinstance(value, torch.Tensor):
                        continue
                    if key in {"pixel_values", "pixel_values_videos"} and value.dim() == 5:
                        value = value.flatten(0, 1)
                    elif key in {"image_grid_thw", "video_grid_thw"} and value.dim() == 3:
                        value = value.flatten(0, 1)
                    visual_values[key].append(value)

        packed_batch = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=sequence_length,
            pad_token_id=0,
            ignore_index=IGNORE_INDEX,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        packed_batch["visual_inputs"] = GenericVisualInputs(
            **{key: torch.cat(values, dim=0) for key, values in visual_values.items() if values}
        )
        return packed_batch

    idx_with = [i for i, h in enumerate(has_media) if h]
    idx_without = [i for i, h in enumerate(has_media) if not h]

    batch_with = None
    batch_without = None

    with use_processor_right_padding(processor):
        if idx_with:
            texts_with = [texts[i] for i in idx_with]
            images_with = [per_example_images[i] for i in idx_with]
            videos_with = [per_example_videos[i] for i in idx_with]
            processor_kwargs = {
                "text": texts_with,
                "padding": True,
                "return_tensors": "pt",
            }
            if min_pixels is not None:
                processor_kwargs["min_pixels"] = min_pixels
            if max_pixels is not None:
                processor_kwargs["max_pixels"] = max_pixels
            if any(images_with):
                processor_kwargs["images"] = images_with
            if any(videos_with):
                processor_kwargs["videos"] = videos_with
            batch_with = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(**processor_kwargs).items()
            }

        if idx_without:
            texts_without = [texts[i] for i in idx_without]
            batch_without = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(
                    text=texts_without,
                    padding=True,
                    return_tensors="pt",
                ).items()
            }
    # Merge batches back to original order
    if batch_with is not None and batch_without is None:
        batch = batch_with
    elif batch_with is None and batch_without is not None:
        batch = batch_without
    else:
        # Both exist: pad to common max length and interleave rows
        pad_id = getattr(processor.tokenizer, "pad_token_id", 0) or 0
        in_with = batch_with["input_ids"]
        in_without = batch_without["input_ids"]
        max_len = max(in_with.shape[1], in_without.shape[1])

        def pad_to(x, tgt_len, value):
            if x.shape[1] == tgt_len:
                return x
            pad_len = tgt_len - x.shape[1]
            return F.pad(x, (0, pad_len), value=value)

        in_with = pad_to(in_with, max_len, pad_id)
        in_without = pad_to(in_without, max_len, pad_id)

        input_ids = torch.full((len(examples), max_len), pad_id, dtype=in_with.dtype)
        # Place rows
        for row, i in enumerate(idx_with):
            input_ids[i] = in_with[row]
        for row, i in enumerate(idx_without):
            input_ids[i] = in_without[row]

        batch = {"input_ids": input_ids}
        if "attention_mask" in batch_with and "attention_mask" in batch_without:
            attn_with = pad_to(batch_with["attention_mask"], max_len, 0)
            attn_without = pad_to(batch_without["attention_mask"], max_len, 0)
            attention_mask = torch.zeros((len(examples), max_len), dtype=attn_with.dtype)
            for row, i in enumerate(idx_with):
                attention_mask[i] = attn_with[row]
            for row, i in enumerate(idx_without):
                attention_mask[i] = attn_without[row]
            batch["attention_mask"] = attention_mask
        # Carry over vision tensors if present
        for key in QWEN_VISUAL_KEYS:
            if key in batch_with:
                batch[key] = batch_with[key]

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
                warn_on_all_masked=not require_assistant_matches,
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

    visual_inputs = GenericVisualInputs(
        pixel_values=batch.get("pixel_values"),
        pixel_values_videos=batch.get("pixel_values_videos"),
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        second_per_grid_ts=batch.get("second_per_grid_ts"),
    )
    for key in QWEN_VISUAL_KEYS:
        batch.pop(key, None)
    batch["visual_inputs"] = visual_inputs
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
    )
    return batch
