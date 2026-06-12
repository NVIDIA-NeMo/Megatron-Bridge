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

"""GLM VL collator implementations."""

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.hf_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.data.vlm_batching import prepare_vlm_batch_for_training
from megatron.bridge.data.vlm_datasets.collate_utils import THW_GRID_VISUAL_KEYS
from megatron.bridge.data.vlm_processing import build_assistant_loss_mask
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def glm4v_collate_fn(
    examples: list,
    processor,
    *,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    pack_sequences: bool = False,
    pack_sequences_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for GLM-4.5V model.

    GLM-4.5V requires ``mm_token_type_ids`` to distinguish image (1) and video (2)
    tokens from text (0) when computing 3D MRoPE positions.  The processor returns
    this field by default (``return_mm_token_type_ids=True`` in Glm4vProcessor
    defaults).  We wrap all visual tensors — including ``mm_token_type_ids`` — in
    :class:`GenericVisualInputs` so they flow through ``vlm_step.py`` to the model.
    """
    skipped_tokens = extract_skipped_token_ids(processor)

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

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
            build_assistant_loss_mask(example, input_ids, processor, skipped_tokens)
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

    visual_kwargs = {}
    for key in (*THW_GRID_VISUAL_KEYS, "mm_token_type_ids"):
        if key in batch:
            visual_kwargs[key] = batch.pop(key)
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None
    prepare_vlm_batch_for_training(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        pack_sequences=pack_sequences,
        pack_sequences_pad_to_multiple_of=pack_sequences_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
    )

    return batch
