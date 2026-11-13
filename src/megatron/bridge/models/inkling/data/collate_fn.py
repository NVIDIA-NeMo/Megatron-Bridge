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

"""Inkling2 image-text SFT collator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.conversation_processing import chat_template_kwargs_from_example
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.models.inkling.data.processor import Inkling2Processor
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


@dataclass
class Inkling2VisualInputs(GenericVisualInputs):
    """Inkling2 image inputs: ``pixel_values`` ``[patches, 2, 40, 40, 3]`` and ``image_grid_thw`` ``[images, 3]``.

    The patches are five-dimensional but already flat over images, so they pass to the model unchanged instead
    of being reshaped as a batch of per-sample image stacks.
    """

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        """Return the non-None fields unchanged."""
        return self.as_model_kwargs()


def inkling2_collate_fn(
    examples: list[dict[str, Any]],
    processor: Inkling2Processor,
    *,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, Any]:
    """Collate Inkling2 conversations with images into a padded training batch.

    Args:
        examples: Rows with a ``conversation`` message list, and optionally the ``tools`` and
            ``chat_template_kwargs`` (``reasoning_effort``) it is rendered with.
        processor: The checkpoint's :class:`Inkling2Processor`.
        sequence_length: Longest sample, and the padded width with ``pad_to_max_length``.
        pad_to_max_length: Pad every batch to ``sequence_length``, which keeps one compiled attention shape.
        pad_to_multiple_of: Width multiple when not padding to ``sequence_length``.
        enable_in_batch_packing: Not supported; must be False.
        in_batch_packing_pad_to_multiple_of: Unused without in-batch packing.

    Returns:
        ``input_ids``, ``attention_mask``, ``position_ids``, ``labels`` and ``loss_mask`` over the assistant
        turns, and ``visual_inputs``.

    Raises:
        ValueError: If in-batch packing is requested, or a row sets template arguments the chat template does
            not read.
    """
    if enable_in_batch_packing:
        raise ValueError("Inkling2 image-text SFT does not support in-batch packing")
    del in_batch_packing_pad_to_multiple_of
    rendered = processor(
        [example["conversation"] for example in examples],
        max_length=sequence_length,
        chat_template_kwargs=[chat_template_kwargs_from_example(example) for example in examples],
    )
    input_ids = rendered["input_ids"]
    batch_size, width = input_ids.shape
    # Each position predicts the next token; the mask follows the predicted token.
    loss_mask = torch.cat(
        [rendered["assistant_masks"][:, 1:], rendered["assistant_masks"].new_zeros((batch_size, 1))], dim=1
    ).to(torch.float32)
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((batch_size, 1), IGNORE_INDEX)], dim=1)
    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": rendered["attention_mask"],
        "position_ids": torch.arange(width).unsqueeze(0).expand(batch_size, -1).clone(),
        "labels": labels.masked_fill(loss_mask == 0, IGNORE_INDEX),
        "loss_mask": loss_mask,
    }
    batch["visual_inputs"] = (
        Inkling2VisualInputs(pixel_values=rendered["pixel_values"], image_grid_thw=rendered["image_grid_thw"])
        if rendered["pixel_values"] is not None
        else None
    )
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0,
        ignore_index=IGNORE_INDEX,
    )
    return batch


__all__ = ["Inkling2VisualInputs", "inkling2_collate_fn"]
