# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Collation utilities for building VLM training batches from conversation examples.
"""

import warnings

import torch
import torch.nn.functional as F
from PIL import Image  # noqa: F401  # may be used downstream by processors

from megatron.bridge.data.datasets.utils import (
    create_multiturn_loss_mask_by_search,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs


# Local message used when optional qwen_vl_utils dependency is missing
MISSING_QWEN_VL_UTILS_MSG = (
    "qwen_vl_utils is required for Qwen2.5 VL processing. Please `pip install qwen-vl-utils` or"
    " provide compatible vision preprocessing."
)

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False


# moved helper functions to datasets.utils


def phi4_mm_collate_fn(examples, processor):
    """Collate function for Phi-4 MM model audio input"""

    # Extract conversations and audio data
    conversations = [example["conversation"] for example in examples]
    audios = [example["audio"] for example in examples]
    texts = [processor.apply_chat_template(conversation, tokenize=False) for conversation in conversations]
    audio_inputs = [(audio["array"], audio["sampling_rate"]) if isinstance(audio, dict) else audio for audio in audios]
    batch = processor(
        text=texts, audios=audio_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024
    )
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)

    loss_masks = []
    for i, conversation in enumerate(conversations):
        input_ids = batch["input_ids"][i].tolist()

        assistant_content = conversation[1]["content"]
        assistant_tokens = processor.tokenizer(assistant_content, add_special_tokens=False)["input_ids"]

        loss_mask = [0] * len(input_ids)
        for start_idx in range(len(input_ids) - len(assistant_tokens) + 1):
            if input_ids[start_idx : start_idx + len(assistant_tokens)] == assistant_tokens:
                for j in range(len(assistant_tokens)):
                    loss_mask[start_idx + j] = 1
                break
        loss_masks.append(loss_mask)

    max_len = max(len(mask) for mask in loss_masks)
    padded_loss_masks = [mask + [0] * (max_len - len(mask)) for mask in loss_masks]
    batch["loss_mask"] = torch.tensor(padded_loss_masks, dtype=torch.float)

    labels[batch["loss_mask"] == 0] = -100
    batch["labels"] = labels

    # Remove specified batch features if present
    for key in ["input_image_embeds", "image_sizes", "image_attention_mask"]:
        if key in batch:
            del batch[key]
    return batch


def qwen2_5_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    texts = [processor.apply_chat_template(example["conversation"], tokenize=False) for example in examples]
    # Build per-example images (list) and split by presence
    per_example_images = []
    has_images = []
    for example in examples:
        imgs = process_vision_info(example["conversation"])[0]
        if imgs is None:
            imgs = []
        elif not isinstance(imgs, list):
            imgs = [imgs]
        per_example_images.append(imgs)
        has_images.append(len(imgs) > 0)

    idx_with = [i for i, h in enumerate(has_images) if h]
    idx_without = [i for i, h in enumerate(has_images) if not h]

    batch_with = None
    batch_without = None

    if idx_with:
        texts_with = [texts[i] for i in idx_with]
        images_with = [per_example_images[i] for i in idx_with]
        batch_with = processor(
            text=texts_with,
            images=images_with,
            padding=True,
            return_tensors="pt",
            min_pixels=200704,  # 256*28*28
            max_pixels=1003520,  # 1280*28*28
        )

        batch_with = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch_with.items()}

    if idx_without:
        texts_without = [texts[i] for i in idx_without]
        batch_without = processor(
            text=texts_without,
            padding=True,
            return_tensors="pt",
        )

        batch_without = {k: v.contiguous() if isinstance(v, torch.Tensor) else v for k, v in batch_without.items()}

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

        def pad_to(x, tgt_len):
            if x.shape[1] == tgt_len:
                return x
            pad_len = tgt_len - x.shape[1]
            return F.pad(x, (0, pad_len), value=pad_id)

        in_with = pad_to(in_with, max_len)
        in_without = pad_to(in_without, max_len)

        input_ids = torch.full((len(examples), max_len), pad_id, dtype=in_with.dtype)
        # Place rows
        for row, i in enumerate(idx_with):
            input_ids[i] = in_with[row]
        for row, i in enumerate(idx_without):
            input_ids[i] = in_without[row]

        batch = {"input_ids": input_ids}
        # Carry over vision tensors if present
        if "pixel_values" in batch_with:
            batch["pixel_values"] = batch_with["pixel_values"]
        if "image_grid_thw" in batch_with:
            batch["image_grid_thw"] = batch_with["image_grid_thw"]

    labels = batch["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    # Ensure position_ids exist for the model
    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .clone()
            .contiguous()
        )
    # Prefer general search-based masking using structured example content (not template-specific)
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    # Enforce label masking to match shifted loss_mask
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    # Build Qwen2VL visual inputs object and attach to batch; remove raw keys
    visual_inputs = Qwen2_5_VLVisualInputs(
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
    )
    if "pixel_values" in batch:
        del batch["pixel_values"]
    if "image_grid_thw" in batch:
        del batch["image_grid_thw"]
    batch["visual_inputs"] = visual_inputs
    return batch


def nemotron_nano_v2_vl_collate_fn(examples: list, processor, start_of_response_token=None) -> dict[str, torch.Tensor]:
    """Collate function for Nemotron Nano V2 VL model."""
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

    skipped_tokens = extract_skipped_token_ids(processor)
    # this assumes the first message in conversation is the video message
    is_video = examples[0]["conversation"][0]["content"][0]["type"] == "video"
    if is_video:
        from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
            maybe_path_or_url_to_data_urls,
            pil_image_from_base64,
        )

        assert len(examples) == 1, "Nemotron Nano V2 VL processor only supports batch size == 1"
        frames = []
        video_fps = -1
        video_nframe = 10
        video_nframe_max = -1

        for example in examples:
            video_path = example["conversation"][0]["content"][0]["path"]
            image_urls, metadata = maybe_path_or_url_to_data_urls(
                video_path,
                fps=max(0, int(video_fps)),
                nframe=max(0, int(video_nframe)),
                nframe_max=int(video_nframe_max),
            )
            frames.append([pil_image_from_base64(image_url) for image_url in image_urls])

        prompt = processor.apply_chat_template([example["conversation"] for example in examples], tokenize=False)
        batch = processor(
            text=prompt,
            videos=frames,
            videos_kwargs={"video_metadata": metadata},
            return_tensors="pt",
        )
    else:
        batch = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=True,
            padding=processor.tokenizer.pad_token is not None,
            truncation=True,
            return_tensors="pt",
            return_dict=True,
        )
    loss_mask = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]

    img_start_token_id = 131073  # tokenizer.convert_tokens_to_ids("<img>")
    img_end_token_id = 131074  # tokenizer.convert_tokens_to_ids("</img>")
    adjusted_batch = adjust_image_tokens(
        {
            "input_ids": batch["input_ids"],
            "loss_mask": torch.tensor(loss_mask),
        },
        batch["num_patches"],
        img_start_token_id,
        img_end_token_id,
    )

    if is_video:
        video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        adjusted_batch["input_ids"] = torch.where(
            adjusted_batch["input_ids"] == video_token_id, image_token_id, adjusted_batch["input_ids"]
        )

    batch["input_ids"] = adjusted_batch["input_ids"]
    loss_mask = adjusted_batch["loss_mask"]

    if "position_ids" not in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1)
        )

    key = "pixel_values_videos" if is_video else "pixel_values"
    batch["pixel_values"] = batch[key].to(torch.bfloat16)
    # roll label by 1 and fill last token with IGNORE_INDEX
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = IGNORE_INDEX
    batch["labels"] = labels

    loss_mask_t = torch.tensor(loss_mask, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, IGNORE_INDEX)
    batch["loss_mask"] = loss_mask_t
    return batch


def ministral3_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Ministral 3 VL model."""
    skipped_tokens = extract_skipped_token_ids(processor)

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

    if "input_ids" in batch:
        labels = batch["input_ids"].clone()[:, 1:]
        labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
        labels[torch.isin(labels, skipped_tokens)] = -100
        batch["labels"] = labels

        # Create loss mask using search-based masking for assistant turns
        loss_masks = [
            create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
            for example, input_ids in zip(examples, batch["input_ids"])
        ]
        loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
        # Unmask the last token (EOS) so the model learns when to stop generating
        loss_mask_t[:, -1] = 1
        # Shift loss mask to align with next-token labels timeline
        loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
        # Enforce label masking to match shifted loss_mask
        batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
        batch["loss_mask"] = loss_mask_t

    if "position_ids" not in batch and "input_ids" in batch:
        batch_size, seq_len = batch["input_ids"].shape
        batch["position_ids"] = (
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1).clone()
        )

    return batch


def default_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Default collate function for VLM models."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

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
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1).clone()
        )

    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t
    # Build Qwen2VL visual inputs object and attach to batch; remove raw keys
    visual_inputs = Qwen2_5_VLVisualInputs(
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
    )
    if "pixel_values" in batch:
        del batch["pixel_values"]
    if "image_grid_thw" in batch:
        del batch["image_grid_thw"]
    batch["visual_inputs"] = visual_inputs
    return batch


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3VLProcessor": qwen2_5_collate_fn,
    "NemotronNanoVLV2Processor": nemotron_nano_v2_vl_collate_fn,
    "PixtralProcessor": ministral3_collate_fn,  # Ministral3 uses PixtralProcessor
    "default": default_collate_fn,
}
