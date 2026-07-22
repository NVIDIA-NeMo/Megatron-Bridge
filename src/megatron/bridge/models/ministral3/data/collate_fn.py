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

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.collators.visual import PASSTHROUGH_VISUAL_KEYS
from megatron.bridge.data.conversation_processing import (
    AssistantMaskBoundaryConfig,
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
    infer_assistant_mask_boundary_config,
    shared_chat_template_kwargs_from_examples,
)
from megatron.bridge.data.datasets.utils import GENERATION_REGEX, IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.token_utils import extract_skipped_token_ids
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

    if enable_in_batch_packing:
        sequence_rows = []
        visual_values: dict[str, list[torch.Tensor]] = {key: [] for key in PASSTHROUGH_VISUAL_KEYS}
        with use_processor_right_padding(processor):
            for example in examples:
                if processor.chat_template is not None:
                    sample_batch = processor.apply_chat_template(
                        [example["conversation"]],
                        tokenize=True,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        return_dict=True,
                        **chat_template_kwargs_from_example(example),
                    )
                else:
                    conversation_text = []
                    images = []
                    for message in example["conversation"]:
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        if isinstance(content, list):
                            text_parts = []
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                                elif isinstance(item, dict) and item.get("type") == "image":
                                    text_parts.append("[IMG]")
                                    if "image" in item:
                                        images.append(item["image"])
                                    elif "path" in item:
                                        images.append(Image.open(item["path"]))
                                elif isinstance(item, str):
                                    text_parts.append(item)
                            content = " ".join(text_parts)
                        conversation_text.append(f"{role.capitalize()}: {content}")
                    sample_batch = processor(
                        text=["\n".join(conversation_text)],
                        images=images,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
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
                if loss_mask.numel() > 0:
                    loss_mask[-1] = 1.0
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
                for key in PASSTHROUGH_VISUAL_KEYS:
                    value = sample_batch.get(key)
                    if isinstance(value, torch.Tensor):
                        visual_values[key].append(value)

        packed_batch = build_mcore_thd_sequence_batch_from_rows(
            sequence_rows,
            sequence_length=sequence_length,
            pad_token_id=int(getattr(getattr(processor, "tokenizer", None), "pad_token_id", 0) or 0),
            ignore_index=IGNORE_INDEX,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        )
        visual_kwargs = {key: torch.cat(values, dim=0) for key, values in visual_values.items() if values}
        packed_batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None
        return packed_batch

    if processor.chat_template is not None:
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
        with use_processor_right_padding(processor):
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
        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2:
            last_token_indices = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
        else:
            last_token_indices = torch.full(
                (batch["input_ids"].size(0),),
                batch["input_ids"].size(1) - 1,
                dtype=torch.long,
                device=batch["input_ids"].device,
            )
        loss_mask[torch.arange(loss_mask.size(0), device=loss_mask.device), last_token_indices] = 1.0
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
