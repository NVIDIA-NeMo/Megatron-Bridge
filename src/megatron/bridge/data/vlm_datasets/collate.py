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
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image  # noqa: F401  # may be used downstream by processors

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.vlm_datasets.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs, Qwen2_5_VLVisualInputs, Qwen2AudioInputs


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


def _gather_assistant_text_segments(example: dict) -> list[str]:
    """Extract assistant text segments from the structured conversation example.

    The example schema is expected to be {"conversation": [{"role": ..., "content": [...]} ...]} where
    content is a list of items like {"type": "text"|"image"|..., "text": "..."}.
    Returns a list of concatenated text strings, one per assistant turn.
    """
    texts: list[str] = []
    for turn in example.get("conversation", []):
        if turn.get("role") != "assistant":
            continue
        parts = turn.get("content", [])
        buf = []
        if isinstance(parts, list):
            for p in parts:
                if isinstance(p, dict) and p.get("type") == "text" and isinstance(p.get("text"), str):
                    buf.append(p["text"])
        elif isinstance(parts, str):
            buf.append(parts)
        if buf:
            texts.append("".join(buf))
    return texts


def create_multiturn_loss_mask_by_search(
    example: dict, input_ids, processor, skipped_tokens: torch.Tensor
) -> list[int]:
    """Tokenizer-agnostic masking via substring search of assistant texts.

    - Tokenize full conversation with processor already done -> input_ids
    - Extract assistant text strings from the structured example
    - For each assistant text, tokenize without special tokens and search sequentially
    - On success, unmask that span; otherwise leave masked
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    ids = input_ids.tolist()
    mask = [0] * len(ids)

    def try_mark(span_text: str, start_from: int) -> int:
        """Tokenize a span and mark its occurrence if found. Returns new search start index."""
        variants = [span_text, span_text + "\n", span_text.strip(), span_text.strip() + "\n"]
        for text in variants:
            span_tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not span_tokens:
                continue
            # naive sequential search from start_from
            for i in range(start_from, len(ids) - len(span_tokens) + 1):
                if ids[i : i + len(span_tokens)] == span_tokens:
                    for j in range(i, i + len(span_tokens)):
                        mask[j] = 1
                    return i + len(span_tokens)
        return start_from

    search_start = 0
    for asst_text in _gather_assistant_text_segments(example):
        search_start = try_mark(asst_text, search_start)

    if sum(mask) == 0:
        warnings.warn("*" * 100)
        warnings.warn(f"All tokens are masked for example:\n{example}.")
        warnings.warn("*" * 100)

    # Ensure pad/skipped tokens are masked
    ids_t = torch.tensor(ids)
    for k, t in enumerate(ids_t):
        if t in skipped_tokens:
            mask[k] = 0
    return mask


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
        # Ensure a pad_token is set so padding can produce uniform-length tensors.
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        batch = processor.apply_chat_template(
            [example["conversation"] for example in examples],
            tokenize=True,
            padding=True,
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
    pv = batch[key].to(torch.bfloat16)
    batch[key] = pv
    batch["visual_inputs"] = GenericVisualInputs(pixel_values=pv)
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

    # Wrap visual tensors in GenericVisualInputs so vlm_step.py picks them up
    visual_kwargs = {}
    for vk in (
        "pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw",
        "image_sizes", "image_position_ids",
    ):
        if vk in batch:
            visual_kwargs[vk] = batch.pop(vk)
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None

    return batch


def glm4v_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
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
            torch.arange(seq_len, device=batch["input_ids"].device).unsqueeze(0).expand(batch_size, -1).clone()
        )

    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    batch["labels"] = labels

    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=batch["input_ids"].device)
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    batch["labels"] = batch["labels"].masked_fill(loss_mask_t == 0, -100)
    batch["loss_mask"] = loss_mask_t

    # Wrap visual tensors in GenericVisualInputs (includes mm_token_type_ids for GLM)
    visual_kwargs = {}
    for vk in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw", "mm_token_type_ids"):
        if vk in batch:
            visual_kwargs[vk] = batch.pop(vk)
    batch["visual_inputs"] = GenericVisualInputs(**visual_kwargs) if visual_kwargs else None

    return batch


def default_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Default collate function for VLM models."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    skipped_tokens = extract_skipped_token_ids(processor)

    # Ensure a pad_token is set so padding can produce uniform-length tensors.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # If pad_token is still unset after the eos_token fallback, disable padding
    # to avoid a ValueError from apply_chat_template.
    can_pad = tokenizer is not None and tokenizer.pad_token is not None

    # Force right-padding for training collation.  Some tokenizers (e.g. Gemma3)
    # default to left-padding which breaks downstream sequence packing: the packer
    # copies tokens[seq_idx, :length] from position 0, so left-padded content gets
    # replaced by padding tokens and image/special tokens are lost.
    saved_padding_side = getattr(tokenizer, "padding_side", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"

    batch = processor.apply_chat_template(
        [example["conversation"] for example in examples],
        tokenize=True,
        padding=can_pad,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    )

    # Restore original padding side so generation paths are unaffected.
    if tokenizer is not None and saved_padding_side is not None:
        tokenizer.padding_side = saved_padding_side

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


# Gemma4 VL uses apply_chat_template and returns image_position_ids — same path as ministral3
gemma4_vl_collate_fn = ministral3_collate_fn
def qwen2_audio_collate_fn(examples: list, processor) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2-Audio model.

    Uses HF-compatible label construction:
    - Backward search for assistant text spans (matching HF Trainer convention)
    - No skipped_tokens masking on labels (model learns to predict EOS/im_end)
    - Loss mask derived directly from active label positions
    """
    texts = []
    audio_inputs = []
    for example in examples:
        text = processor.apply_chat_template(example["conversation"], tokenize=False)
        texts.append(text)
        audio = example.get("audio")
        if audio is not None:
            if isinstance(audio, tuple):
                audio_inputs.append(audio[0])  # (array, sr) -> array
            elif isinstance(audio, dict):
                audio_inputs.append(audio["array"])
            else:
                audio_inputs.append(audio)

    # Megatron's packing and padding utilities assume right-padding
    # (tokens[:length] extracts real content). Override the tokenizer's
    # default padding_side which may be "left" (e.g. Qwen2Audio).
    tokenizer = getattr(processor, "tokenizer", processor)
    orig_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"

    batch = processor(
        text=texts,
        audio=audio_inputs if audio_inputs else None,
        return_tensors="pt",
        padding=True,
    )

    tokenizer.padding_side = orig_padding_side
    input_ids = batch["input_ids"]
    batch_size, seq_len = input_ids.shape
    pad_token_id = tokenizer.pad_token_id

    # --- HF-compatible label construction ---
    # Step 1: Build unshifted labels (same convention as HF Trainer)
    hf_labels = input_ids.clone()

    for i, example in enumerate(examples):
        ids = input_ids[i].tolist()
        assistant_texts = _gather_assistant_text_segments(example)

        # Find assistant span using backward search (like HF's Qwen2AudioCollator)
        found = -1
        for asst_text in assistant_texts:
            asst_token_ids = tokenizer(asst_text, add_special_tokens=False)["input_ids"]
            span_len = len(asst_token_ids)
            if span_len == 0:
                continue
            for start in range(len(ids) - span_len, -1, -1):
                if ids[start : start + span_len] == asst_token_ids:
                    found = start
                    break
            if found >= 0:
                break

        if found >= 0:
            # Mask everything before the assistant span (prompt + special tokens)
            hf_labels[i, :found] = IGNORE_INDEX
        else:
            warnings.warn(f"Could not find assistant span for example {i}, masking all labels", stacklevel=2)
            hf_labels[i, :] = IGNORE_INDEX

        # Mask padding tokens
        if pad_token_id is not None:
            hf_labels[i][input_ids[i] == pad_token_id] = IGNORE_INDEX

    # Step 2: Shift labels for Megatron (labels[j] = hf_labels[j+1])
    labels = hf_labels[:, 1:]
    labels = torch.cat([labels, IGNORE_INDEX * torch.ones_like(labels[:, :1])], dim=1)
    batch["labels"] = labels

    # Step 3: Derive loss_mask from active label positions
    batch["loss_mask"] = (labels != IGNORE_INDEX).float()

    # Ensure position_ids exist
    if "position_ids" not in batch:
        batch["position_ids"] = (
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1).clone().contiguous()
        )

    # Wrap audio tensors in Qwen2AudioInputs and attach as audio_inputs
    audio_inputs = Qwen2AudioInputs(
        input_features=batch.get("input_features"),
        feature_attention_mask=batch.get("feature_attention_mask"),
    )
    for key in ("input_features", "feature_attention_mask"):
        if key in batch:
            del batch[key]
    batch["audio_inputs"] = audio_inputs

    return batch


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
        return input_ids, attention_mask

    if len(placeholder_positions) != len(feature_counts):
        warnings.warn(
            "Mismatch between image placeholder count and grid_thws rows during Kimi token expansion; "
            "expanding as many placeholders as have corresponding grid metadata.",
            stacklevel=2,
        )

    expanded_input_ids = []
    expanded_attention_mask = []
    feature_idx = 0

    for token_id, mask_value in zip(input_ids.tolist(), attention_mask.tolist()):
        if token_id == media_token_id and feature_idx < len(feature_counts):
            expanded_input_ids.extend([media_token_id] * feature_counts[feature_idx])
            expanded_attention_mask.extend([1] * feature_counts[feature_idx])
            feature_idx += 1
            continue

        expanded_input_ids.append(token_id)
        expanded_attention_mask.append(mask_value)

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

    return expanded_input_ids, expanded_attention_mask


def kimi_k25_vl_collate_fn(
    examples: list[dict[str, Any]],
    processor,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Collate function for Kimi K2.5 VL processors with pre-expanded image tokens.

    For pipeline parallelism, this function:
    1. Processes each sample to get input_ids with 1 placeholder per image
    2. Pre-expands each placeholder to N tokens (N = t*(h//2)*(w//2) from grid_thws)
    3. Pads all sequences to fixed max_length
    This ensures the model forward pass doesn't change sequence length dynamically.
    """
    skipped_tokens = extract_skipped_token_ids(processor)
    conversations = [example["conversation"] for example in examples]

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

    for i, conversation in enumerate(conversations):
        # Collect medias for this conversation
        medias = []
        for message in conversation:
            content = message.get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        medias.append({"type": "image", "image": item.get("image")})

        text = processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)

        processor_kwargs = {
            "text": text,
            "return_tensors": "pt",
        }
        if medias:
            processor_kwargs["medias"] = medias

        sample_batch = processor(**processor_kwargs)

        input_ids = sample_batch["input_ids"][0]
        attention_mask = sample_batch["attention_mask"][0]

        # Pre-expand image tokens if we have grid_thws
        if "grid_thws" in sample_batch and sample_batch["grid_thws"] is not None:
            # print(f"using grid_thws: {sample_batch['grid_thws']} and pre-expand mode")
            grid_thws = sample_batch["grid_thws"]
            # print(f"before expand input_ids: {input_ids.shape}")

            input_ids, attention_mask = _expand_image_tokens(
                input_ids, attention_mask, grid_thws, media_token_id, merge_kernel_size
            )
            all_grid_thws.append(grid_thws)
            # print(f"after expand input_ids: {input_ids.shape}")

        if "pixel_values" in sample_batch:
            all_pixel_values.append(sample_batch["pixel_values"])

        all_expanded.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

    # Determine target length for padding
    expanded_lens = [b["input_ids"].shape[0] for b in all_expanded]
    batch_max = max(expanded_lens)

    if max_length is not None:
        target_len = max_length
    else:
        target_len = batch_max

    # Pad/truncate to target_len
    padded_input_ids = []
    padded_attention_mask = []

    for batch in all_expanded:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        seq_len = input_ids.shape[0]

        if seq_len < target_len:
            # Pad
            pad_len = target_len - seq_len
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token_id, dtype=input_ids.dtype)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        elif seq_len > target_len:
            # Truncate
            input_ids = input_ids[:target_len]
            attention_mask = attention_mask[:target_len]

        padded_input_ids.append(input_ids)
        padded_attention_mask.append(attention_mask)

    result = {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
    }

    if all_pixel_values:
        result["pixel_values"] = torch.cat(all_pixel_values, dim=0)
    if all_grid_thws:
        result["grid_thws"] = torch.cat(all_grid_thws, dim=0)  # (N, 3) with [t, h, w]

    labels = result["input_ids"].clone()[:, 1:].contiguous()
    labels = torch.cat([labels, -100 * torch.ones_like(labels[:, :1])], dim=1)
    labels[torch.isin(labels, skipped_tokens)] = -100
    result["labels"] = labels
    # Ensure position_ids exist for the model
    if "position_ids" not in result:
        result_size, seq_len = result["input_ids"].shape
        result["position_ids"] = (
            torch.arange(seq_len, device=result["input_ids"].device)
            .unsqueeze(0)
            .expand(result_size, -1)
            .clone()
            .contiguous()
        )
    # Prefer general search-based masking using structured example content (not template-specific)
    loss_masks = [
        create_multiturn_loss_mask_by_search(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, result["input_ids"])  # type: ignore[arg-type]
    ]
    loss_mask_t = torch.tensor(loss_masks, dtype=torch.float, device=result["input_ids"].device)
    # Shift loss mask to align with next-token labels timeline
    loss_mask_t = torch.cat([loss_mask_t[:, 1:], torch.zeros_like(loss_mask_t[:, :1])], dim=1)
    # Enforce label masking to match shifted loss_mask
    result["labels"] = result["labels"].masked_fill(loss_mask_t == 0, -100)
    result["loss_mask"] = loss_mask_t.contiguous()

    # Build visual inputs object and attach to batch; remove raw keys.
    # Kimi processor outputs "grid_thws" (N, 3); pass full [t, h, w] as image_grid_thw.
    visual_inputs = Qwen2_5_VLVisualInputs(
        pixel_values=result.get("pixel_values"),
        image_grid_thw=result.get("grid_thws"),
    )
    for k in ("pixel_values", "grid_thws"):
        result.pop(k, None)
    result["visual_inputs"] = visual_inputs
    return result


# Mapping of processor types to their collate functions
COLLATE_FNS = {
    "Qwen2_5_VLProcessor": qwen2_5_collate_fn,
    "Qwen3VLProcessor": qwen2_5_collate_fn,
    "NemotronNanoVLV2Processor": nemotron_nano_v2_vl_collate_fn,
    "PixtralProcessor": ministral3_collate_fn,  # Ministral3 uses PixtralProcessor
    "Gemma4Processor": gemma4_vl_collate_fn,   # Gemma4 VL
    "Qwen2AudioProcessor": qwen2_audio_collate_fn,
    "Glm4vProcessor": glm4v_collate_fn,
    "KimiK25Processor": kimi_k25_vl_collate_fn,
    "default": default_collate_fn,
}
