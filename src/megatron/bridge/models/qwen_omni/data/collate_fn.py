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

"""Qwen3-Omni thinker collator implementation."""

from typing import Any

import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.conversation_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    shared_chat_template_kwargs_from_examples,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.token_utils import extract_skipped_token_ids


CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_ASSISTANT_END = "<|im_end|>\n"
CHATML_OTHER_ROLE_STARTS = {role: f"<|im_start|>{role}\n" for role in ("system", "developer", "user", "tool")}


def qwen3_omni_collate_fn(
    examples: list[dict[str, Any]],
    processor: Any,
    *,
    visual_keys: object = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, Any]:
    """Collate typed Qwen3-Omni conversations with image, video, and audio inputs.

    Media resolution is delegated to the Hugging Face processor's chat-template
    path so local paths and URLs follow the processor's native conversation
    schema. Qwen3-Omni training currently uses dense right-padded batches; its
    model step rejects in-batch packing.
    """
    del visual_keys, min_pixels, max_pixels
    if enable_in_batch_packing:
        raise ValueError("Qwen3-Omni does not support in-batch packing.")

    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_ASSISTANT_END,
        assistant_end_fallbacks=("<|im_end|>",),
        role_start_markers=CHATML_OTHER_ROLE_STARTS,
    )
    conversations = [example["conversation"] for example in examples]
    template_kwargs = shared_chat_template_kwargs_from_examples(examples)
    with use_processor_right_padding(processor):
        processed = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True, "padding_side": "right"},
            **template_kwargs,
        )
    batch = {key: value.contiguous() if isinstance(value, torch.Tensor) else value for key, value in processed.items()}
    if "audio_feature_lengths" not in batch and isinstance(batch.get("feature_attention_mask"), torch.Tensor):
        batch["audio_feature_lengths"] = batch["feature_attention_mask"].sum(dim=-1).to(dtype=torch.long)

    input_ids = batch["input_ids"]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        batch["input_ids"] = input_ids
    if input_ids.dim() != 2:
        raise ValueError(f"Qwen3-Omni processor returned input_ids with shape {tuple(input_ids.shape)}; expected 2D.")

    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                row,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
            )
            for example, row in zip(examples, input_ids, strict=True)
        ]
    ).to(device=input_ids.device, dtype=torch.float32)
    labels = torch.cat([input_ids[:, 1:], input_ids.new_full((input_ids.size(0), 1), IGNORE_INDEX)], dim=1)
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    loss_mask = torch.cat([loss_mask[:, 1:], loss_mask.new_zeros((loss_mask.size(0), 1))], dim=1)
    batch["labels"] = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask

    if "attention_mask" not in batch:
        batch["attention_mask"] = torch.ones_like(input_ids)
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(input_ids.size(1), device=input_ids.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(input_ids.size(0), -1)
            .clone()
        )

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=False,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        pad_token_id=pad_token_id,
        ignore_index=IGNORE_INDEX,
    )
    return batch
