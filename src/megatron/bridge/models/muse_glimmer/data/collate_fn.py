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

"""Muse Glimmer multimodal SFT collator."""

from typing import Any

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.collators.visual import THW_GRID_VISUAL_KEYS
from megatron.bridge.data.conversation_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
    shared_chat_template_kwargs_from_examples,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


MUSE_GLIMMER_ASSISTANT_START = "<|start|>assistant to=user<|message|>"
MUSE_GLIMMER_TURN_END = "<|eot|>"


def muse_glimmer_collate_fn(
    examples: list[dict[str, Any]],
    processor: Any,
    *,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, Any]:
    """Collate Muse Glimmer conversations and their THW-grid media tensors."""
    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=MUSE_GLIMMER_ASSISTANT_START,
        assistant_end=MUSE_GLIMMER_TURN_END,
    )

    if enable_in_batch_packing:
        sequence_rows = []
        visual_values: dict[str, list[torch.Tensor]] = {key: [] for key in THW_GRID_VISUAL_KEYS}
        with use_processor_right_padding(processor):
            for example in examples:
                sample_batch = dict(
                    processor.apply_chat_template(
                        [example["conversation"]],
                        tokenize=True,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        return_dict=True,
                        **chat_template_kwargs_from_example(example),
                    )
                )
                input_ids = sample_batch["input_ids"][0]
                attention_mask = sample_batch.get("attention_mask")
                attention_mask = attention_mask[0] if attention_mask is not None else torch.ones_like(input_ids)
                position_ids = sample_batch.get("position_ids")
                position_ids = (
                    position_ids[0]
                    if position_ids is not None
                    else torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long)
                )
                loss_mask = build_assistant_loss_mask(
                    example,
                    input_ids,
                    processor,
                    skipped_tokens,
                    boundary_config=boundary_config,
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
                for key in THW_GRID_VISUAL_KEYS:
                    value = sample_batch.get(key)
                    if isinstance(value, torch.Tensor):
                        visual_values[key].append(value)

        packed_batch = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=sequence_length,
            pad_token_id=int(getattr(processor.tokenizer, "pad_token_id", 0) or 0),
            ignore_index=IGNORE_INDEX,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        visual_kwargs = {key: torch.cat(values, dim=0) for key, values in visual_values.items() if values}
        packed_batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None
        return packed_batch

    with use_processor_right_padding(processor):
        batch = dict(
            processor.apply_chat_template(
                [example["conversation"] for example in examples],
                tokenize=True,
                padding=True,
                truncation=True,
                return_tensors="pt",
                return_dict=True,
                **shared_chat_template_kwargs_from_examples(examples),
            )
        )

    input_ids = batch["input_ids"]
    if "position_ids" not in batch:
        batch_size, seq_len = input_ids.shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1).clone().contiguous()
        )

    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                row_input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
            )
            for example, row_input_ids in zip(examples, input_ids, strict=True)
        ]
    ).to(device=input_ids.device, dtype=torch.float32)
    labels = torch.cat(
        [input_ids[:, 1:].contiguous(), input_ids.new_full((input_ids.size(0), 1), IGNORE_INDEX)], dim=1
    )
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    loss_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    batch["labels"] = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask

    visual_kwargs = {key: batch.pop(key) for key in THW_GRID_VISUAL_KEYS if key in batch}
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=False,
        in_batch_packing_pad_to_multiple_of=1,
        ignore_index=IGNORE_INDEX,
    )
    return batch


__all__ = ["muse_glimmer_collate_fn"]
