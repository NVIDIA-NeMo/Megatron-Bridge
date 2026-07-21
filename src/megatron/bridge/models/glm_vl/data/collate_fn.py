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

from typing import Any

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.collators.visual import THW_GRID_VISUAL_KEYS
from megatron.bridge.data.conversation_processing import (
    AssistantMaskBoundaryConfig,
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
    get_processor_tokenizer,
    shared_chat_template_kwargs_from_examples,
    tokenize_text_without_special_tokens,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


GLM4V_ASSISTANT_START = "<|assistant|>\n"
GLM4V_ASSISTANT_END = "<|endoftext|>"
GLM4V_EMPTY_THINK = "<think></think>\n"
GLM4V_NEXT_ROLE_MARKERS = ("<|system|>\n", "<|user|>\n", "<|observation|>\n")


def _glm4v_assistant_mask_boundary_config(processor: Any) -> AssistantMaskBoundaryConfig:
    """Build GLM-4.5V role boundaries and exclude its empty thinking prefix."""
    tokenizer = get_processor_tokenizer(processor)
    empty_think_tokens = tokenize_text_without_special_tokens(tokenizer, GLM4V_EMPTY_THINK)
    trim_leading_token_sequences = (empty_think_tokens,) if empty_think_tokens else ()

    return assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=GLM4V_ASSISTANT_START,
        assistant_end=GLM4V_ASSISTANT_END,
        assistant_end_fallbacks=GLM4V_NEXT_ROLE_MARKERS,
        include_end_tokens_for_roles=(),
        trim_leading_token_sequences=trim_leading_token_sequences,
    )


def _build_glm4v_assistant_loss_mask(
    example: dict,
    input_ids: torch.Tensor,
    processor: Any,
    skipped_tokens: torch.Tensor,
    boundary_config: AssistantMaskBoundaryConfig,
) -> torch.Tensor:
    """Build GLM-4.5V loss spans using a virtual final turn terminator."""
    tokenizer = get_processor_tokenizer(processor)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        raise ValueError("GLM-4.5V assistant masking requires tokenizer.eos_token_id.")

    # GLM role markers terminate intermediate turns, but its final assistant
    # turn ends directly at the sequence boundary. Add EOS only to delimit that
    # final span for masking; it is neither returned nor trained as a target.
    terminated_input_ids = torch.cat([input_ids, input_ids.new_tensor([int(eos_token_id)])])
    return build_assistant_loss_mask(
        example,
        terminated_input_ids,
        processor,
        skipped_tokens,
        boundary_config=boundary_config,
    )[:-1]


def glm4v_collate_fn(
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
    """Collate function for GLM-4.5V model.

    GLM-4.5V requires ``mm_token_type_ids`` to distinguish image (1) and video (2)
    tokens from text (0) when computing 3D MRoPE positions.  The processor returns
    this field by default (``return_mm_token_type_ids=True`` in Glm4vProcessor
    defaults).  We wrap all visual tensors — including ``mm_token_type_ids`` — in
    :class:`GenericVisualInputs` so they flow through ``vlm_step.py`` to the model.
    """
    del visual_keys, min_pixels, max_pixels

    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = _glm4v_assistant_mask_boundary_config(processor)

    if enable_in_batch_packing:
        sequence_rows = []
        visual_values: dict[str, list[torch.Tensor]] = {key: [] for key in THW_GRID_VISUAL_KEYS}
        with use_processor_right_padding(processor):
            for example in examples:
                sample_batch = processor.apply_chat_template(
                    [example["conversation"]],
                    tokenize=True,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                    return_dict=True,
                    **chat_template_kwargs_from_example(example),
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
                loss_mask = _build_glm4v_assistant_loss_mask(
                    example,
                    input_ids,
                    processor,
                    skipped_tokens,
                    boundary_config,
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
                        "mm_token_type_ids": sample_batch["mm_token_type_ids"][0],
                    }
                )
                for key in THW_GRID_VISUAL_KEYS:
                    value = sample_batch.get(key)
                    if isinstance(value, torch.Tensor):
                        visual_values[key].append(value)

        packed_batch = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=sequence_length,
            pad_token_id=int(getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0) or 0),
            ignore_index=IGNORE_INDEX,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            sequence_tensor_pad_values={"mm_token_type_ids": 0},
        )
        visual_kwargs = {key: torch.cat(values, dim=0) for key, values in visual_values.items() if values}
        visual_kwargs["mm_token_type_ids"] = packed_batch.pop("mm_token_type_ids")
        packed_batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs)
        return packed_batch

    with use_processor_right_padding(processor):
        batch = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
            **shared_chat_template_kwargs_from_examples(examples),
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
            _build_glm4v_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config,
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

    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        ignore_index=IGNORE_INDEX,
        sequence_tensor_pad_values={"mm_token_type_ids": 0},
    )

    visual_kwargs = {}
    for key in (*THW_GRID_VISUAL_KEYS, "mm_token_type_ids"):
        if key in batch:
            visual_kwargs[key] = batch.pop(key)
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None

    return batch
