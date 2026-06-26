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

"""Ministral 3 VL collator implementations."""

import torch
from PIL import Image

from megatron.bridge.data.datasets.utils import GENERATION_REGEX, IGNORE_INDEX
from megatron.bridge.data.hf_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.data.sequence_batching import pad_or_pack_sequence
from megatron.bridge.data.vlm_datasets.collate_utils import PASSTHROUGH_VISUAL_KEYS
from megatron.bridge.data.vlm_processing import (
    AssistantMaskBoundaryConfig,
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    infer_assistant_mask_boundary_config,
)
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


MISTRAL3_ASSISTANT_START = "[/INST]"
MISTRAL3_ASSISTANT_END = "</s>"


def _has_generation_chat_template(processor) -> bool:
    tokenizer = getattr(processor, "tokenizer", None)
    for template_owner in (processor, tokenizer):
        template = getattr(template_owner, "chat_template", None)
        if isinstance(template, str) and GENERATION_REGEX.search(template) is not None:
            return True
    return False


def _default_ministral3_assistant_mask_boundary_config(processor) -> AssistantMaskBoundaryConfig:
    tokenizer = getattr(processor, "tokenizer", processor)
    assistant_end = getattr(tokenizer, "eos_token", None) or MISTRAL3_ASSISTANT_END
    return assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=MISTRAL3_ASSISTANT_START,
        assistant_end=assistant_end,
    )


def ministral3_collate_fn(
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
    assistant_mask_boundary_config: AssistantMaskBoundaryConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Collate function for Ministral 3 VL model."""
    del visual_keys, min_pixels, max_pixels

    skipped_tokens = extract_skipped_token_ids(processor)
    if assistant_mask_boundary_config is not None:
        boundary_config = assistant_mask_boundary_config
    elif _has_generation_chat_template(processor):
        boundary_config = infer_assistant_mask_boundary_config(processor)
    else:
        boundary_config = _default_ministral3_assistant_mask_boundary_config(processor)

    if processor.chat_template is not None:
        batch = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=True,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        texts = []
        for example in examples:
            conv_text = []
            for msg in example["conversation"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Handle multimodal content (list of items)
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            elif item.get("type") == "image":
                                text_parts.append("[IMG]")
                        elif isinstance(item, str):
                            text_parts.append(item)
                    content = " ".join(text_parts)

                conv_text.append(f"{role.capitalize()}: {content}")
            texts.append("\n".join(conv_text))

        images = []
        for example in examples:
            ex_images = []
            for msg in example.get("conversation", []):
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "image":
                            if "image" in item:
                                ex_images.append(item["image"])
                            elif "path" in item:
                                ex_images.append(Image.open(item["path"]))
            images.append(ex_images if ex_images else None)
        batch = processor(
            text=texts,
            images=[img if img else [] for img in images],
            padding=True,
            truncation=True,
            return_tensors="pt",
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
    if loss_mask.numel() > 0:
        loss_mask[:, -1] = 1.0
    labels = batch["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    loss_mask = torch.cat([loss_mask[:, 1:], torch.zeros_like(loss_mask[:, :1])], dim=1)
    batch["labels"] = labels.masked_fill(loss_mask == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
            .contiguous()
        )

    visual_kwargs = {}
    for key in PASSTHROUGH_VISUAL_KEYS:
        if key in batch:
            visual_kwargs[key] = batch.pop(key)
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None
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
