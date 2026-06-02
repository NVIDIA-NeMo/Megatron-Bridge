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

"""Shared VLM processing helpers for Energon and HF dataset paths."""

from __future__ import annotations

import copy
import logging
import re
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


logger = logging.getLogger(__name__)


def get_processor_tokenizer(processor: Any) -> Any:
    """Return the tokenizer attached to a processor, or the object itself."""
    return getattr(processor, "tokenizer", processor)


def find_token_span(sequence: Sequence[int] | torch.Tensor, pattern: Sequence[int], start: int = 0) -> tuple[int, int]:
    """Find the first ``[start, end)`` token span matching ``pattern``.

    Args:
        sequence: Token id sequence to search.
        pattern: Token id pattern to locate.
        start: Index to begin searching from.

    Returns:
        ``(start, end)`` for the first match, or ``(-1, -1)`` when no match exists.
    """
    if isinstance(sequence, torch.Tensor):
        ids = sequence.detach().cpu().tolist()
    else:
        ids = list(sequence)
    pat = list(pattern)
    if not pat or start >= len(ids):
        return -1, -1

    end_limit = len(ids) - len(pat) + 1
    for idx in range(start, max(end_limit, start)):
        if ids[idx : idx + len(pat)] == pat:
            return idx, idx + len(pat)
    return -1, -1


def gather_assistant_text_segments(
    example_or_conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[str]:
    """Extract assistant text segments from a structured VLM conversation."""
    conversation = (
        example_or_conversation.get("conversation", [])
        if isinstance(example_or_conversation, Mapping)
        else example_or_conversation
    )
    texts: list[str] = []
    for turn in conversation:
        if turn.get("role") != "assistant":
            continue
        content = turn.get("content", "")
        text_parts: list[str] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, Mapping) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
        elif isinstance(content, str):
            text_parts.append(content)
        if text_parts:
            texts.append("".join(text_parts))
    return texts


def tokenize_text_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    """Tokenize text using a HF-like tokenizer without adding special tokens."""
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
    else:
        tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if isinstance(tokens, torch.Tensor):
        return tokens.detach().cpu().tolist()
    return list(tokens)


def _assistant_text_variants(text: str, *, include_search_variants: bool) -> list[str]:
    if not include_search_variants:
        return [text]
    stripped = text.strip()
    variants = [text, text + "\n", stripped, stripped + "\n"]
    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def build_assistant_loss_mask(
    example_or_conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    input_ids: Sequence[int] | torch.Tensor,
    processor: Any,
    skipped_tokens: torch.Tensor | None = None,
    *,
    include_search_variants: bool = True,
    require_matches: bool = False,
    warn_on_all_masked: bool = True,
) -> torch.Tensor:
    """Build an unshifted assistant-only loss mask using the current token-search behavior.

    This intentionally preserves the existing text-search masking strategy. Step 3
    of issue #4041 can replace this helper with a generation-tag or boundary-token
    implementation without touching every VLM collate/task encoder again.
    """
    tokenizer = get_processor_tokenizer(processor)
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.detach().cpu().tolist()
    else:
        ids = list(input_ids)

    mask = torch.zeros(len(ids), dtype=torch.float32)
    search_start = 0
    missing_segments: list[str] = []
    for assistant_text in gather_assistant_text_segments(example_or_conversation):
        found = False
        for text in _assistant_text_variants(assistant_text, include_search_variants=include_search_variants):
            tokens = tokenize_text_without_special_tokens(tokenizer, text)
            if not tokens:
                continue
            span_start, span_end = find_token_span(ids, tokens, search_start)
            if span_start >= 0:
                mask[span_start:span_end] = 1.0
                search_start = span_end
                found = True
                break
        if not found:
            missing_segments.append(assistant_text)

    if missing_segments and require_matches:
        raise AssertionError("Not found valid answer in conversation.")

    if skipped_tokens is not None and skipped_tokens.numel() > 0:
        skipped = set(skipped_tokens.detach().cpu().tolist())
        for idx, token_id in enumerate(ids):
            if token_id in skipped:
                mask[idx] = 0.0

    if warn_on_all_masked and len(mask) > 0 and float(mask.sum().item()) == 0.0:
        warnings.warn("*" * 100, stacklevel=2)
        warnings.warn(f"All tokens are masked for example:\n{example_or_conversation}.", stacklevel=2)
        warnings.warn("*" * 100, stacklevel=2)

    return mask


def build_shifted_labels_and_loss_mask(
    input_ids: torch.Tensor,
    assistant_loss_mask: torch.Tensor,
    skipped_tokens: torch.Tensor | None = None,
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build next-token labels and shifted loss mask for Megatron training."""
    squeeze = input_ids.dim() == 1
    ids = input_ids.unsqueeze(0) if squeeze else input_ids
    mask = assistant_loss_mask.unsqueeze(0) if assistant_loss_mask.dim() == 1 else assistant_loss_mask
    mask = mask.to(device=ids.device, dtype=torch.float32)

    labels = ids.clone()[:, 1:].contiguous()
    labels = torch.cat([labels, ignore_index * torch.ones_like(labels[:, :1])], dim=1)

    if skipped_tokens is not None and skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), ignore_index)

    shifted_loss_mask = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, :1])], dim=1)
    labels = labels.masked_fill(shifted_loss_mask == 0, ignore_index)

    if squeeze:
        return labels[0].contiguous(), shifted_loss_mask[0].contiguous()
    return labels.contiguous(), shifted_loss_mask.contiguous()


def apply_assistant_labels_to_batch(
    batch: MutableMapping[str, Any],
    examples: Sequence[Mapping[str, Any]],
    processor: Any,
    skipped_tokens: torch.Tensor,
    *,
    unmask_last_token: bool = False,
) -> None:
    """Attach ``labels`` and ``loss_mask`` to a collated HF VLM batch."""
    loss_masks = [
        build_assistant_loss_mask(example, input_ids, processor, skipped_tokens)
        for example, input_ids in zip(examples, batch["input_ids"])
    ]
    loss_mask_t = torch.stack(loss_masks).to(device=batch["input_ids"].device, dtype=torch.float32)
    if unmask_last_token and loss_mask_t.numel() > 0:
        loss_mask_t[:, -1] = 1.0
    labels, shifted_loss_mask = build_shifted_labels_and_loss_mask(batch["input_ids"], loss_mask_t, skipped_tokens)
    batch["labels"] = labels
    batch["loss_mask"] = shifted_loss_mask


def ensure_position_ids(batch: MutableMapping[str, Any]) -> None:
    """Ensure a collated batch has 2D position IDs."""
    if "position_ids" in batch or "input_ids" not in batch:
        return
    batch_size, seq_len = batch["input_ids"].shape
    batch["position_ids"] = (
        torch.arange(seq_len, device=batch["input_ids"].device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
        .contiguous()
    )


def pop_generic_visual_inputs(
    batch: MutableMapping[str, Any],
    visual_keys: Sequence[str],
) -> GenericVisualInputs | None:
    """Move processor visual tensor keys into ``GenericVisualInputs``."""
    visual_kwargs = {}
    for key in visual_keys:
        if key in batch:
            visual_kwargs[key] = batch.pop(key)
    return GenericVisualInputs(**visual_kwargs) if visual_kwargs else None


def convert_media_placeholders_to_content_parts(
    conversation: Sequence[Mapping[str, Any]],
    *,
    image_pattern: str = "<image>",
    video_pattern: str = "<video>",
) -> list[dict[str, Any]]:
    """Convert text ``<image>`` / ``<video>`` placeholders to processor content parts."""
    pattern = r"({image}|{video})".format(image=re.escape(image_pattern), video=re.escape(video_pattern))
    converted = copy.deepcopy(list(conversation))
    for turn in converted:
        content = turn.get("content")
        if not isinstance(content, str) or not re.search(pattern, content):
            continue
        parts = re.split(pattern, content)
        content_parts: list[dict[str, str]] = []
        for part in parts:
            if part == image_pattern:
                content_parts.append({"type": "image"})
            elif part == video_pattern:
                content_parts.append({"type": "video"})
            elif part.strip():
                content_parts.append({"type": "text", "text": part.strip()})
        turn["content"] = content_parts
    return converted


def collect_media_from_conversation(
    conversation: Sequence[Mapping[str, Any]],
) -> tuple[list[Any] | None, list[Any] | None]:
    """Collect inline image/video payloads from HF-style conversation content."""
    images: list[Any] = []
    videos: list[Any] = []
    for turn in conversation:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("type") == "image":
                if "image" in item:
                    images.append(item["image"])
                elif "path" in item:
                    images.append(item["path"])
            elif item.get("type") == "video":
                if "video" in item:
                    videos.append(item["video"])
                elif "path" in item:
                    videos.append(item["path"])
    return images or None, videos or None


@dataclass
class HFProcessorEncodedSample:
    """Per-sample tensors produced by the generic HF VLM processor path."""

    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    visual_tensors: dict[str, torch.Tensor] = field(default_factory=dict)


class HFProcessorVLMDataProcessor:
    """Generic per-sample VLM processor shared by Energon and HF-style data sources."""

    def __init__(
        self,
        processor: Any,
        *,
        seq_length: int,
        visual_keys: Sequence[str] = ("pixel_values",),
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.processor = processor
        self.seq_length = seq_length
        self.visual_keys = tuple(visual_keys)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    @property
    def tokenizer(self) -> Any:
        """Return the underlying tokenizer."""
        return get_processor_tokenizer(self.processor)

    @property
    def pad_token_id(self) -> int:
        """Return the pad token id, falling back to 0 when absent."""
        return getattr(self.tokenizer, "pad_token_id", None) or 0

    @property
    def image_token_id(self) -> int | None:
        """Resolve the image token id from processor/tokenizer metadata."""
        if hasattr(self.processor, "image_token_id"):
            return self.processor.image_token_id
        image_token = getattr(self.processor, "image_token", None)
        if image_token is not None:
            return self.tokenizer.convert_tokens_to_ids(image_token)
        return None

    @staticmethod
    def _find_contiguous_blocks(input_ids: torch.Tensor, value: int) -> list[tuple[int, int]]:
        mask = input_ids == value
        blocks: list[tuple[int, int]] = []
        idx = 0
        while idx < mask.numel():
            if bool(mask[idx]):
                start = idx
                while idx < mask.numel() and bool(mask[idx]):
                    idx += 1
                blocks.append((start, idx))
            else:
                idx += 1
        return blocks

    def encode(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        images: list[Any] | None = None,
        videos: list[Any] | None = None,
    ) -> HFProcessorEncodedSample:
        """Encode one structured VLM conversation into tensors."""
        media_images, media_videos = collect_media_from_conversation(conversation)
        images = images if images is not None else media_images
        videos = videos if videos is not None else media_videos
        proc_conversation = (
            convert_media_placeholders_to_content_parts(conversation)
            if images is not None or videos is not None
            else copy.deepcopy(list(conversation))
        )

        prompt_text = self.processor.apply_chat_template(proc_conversation, tokenize=False)
        proc_kwargs: dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
        if images is not None:
            proc_kwargs["images"] = images
        if videos is not None:
            proc_kwargs["videos"] = videos
        if self.min_pixels is not None:
            proc_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            proc_kwargs["max_pixels"] = self.max_pixels

        proc_output = self.processor(**proc_kwargs)
        input_ids = proc_output["input_ids"]
        if input_ids.dim() == 2:
            input_ids = input_ids[0]
        input_ids = input_ids.detach().cpu().to(dtype=torch.long).contiguous()

        unshifted_loss_mask = build_assistant_loss_mask(
            proc_conversation,
            input_ids,
            self.processor,
            include_search_variants=False,
            warn_on_all_masked=False,
        )
        labels, loss_mask = build_shifted_labels_and_loss_mask(input_ids, unshifted_loss_mask)

        input_ids_pre_trunc = input_ids
        input_ids = input_ids[: self.seq_length].clone()
        labels = labels[: self.seq_length].clone()
        loss_mask = loss_mask[: self.seq_length].clone()

        num_images = len(images) if images is not None else 0
        num_complete_images = num_images
        if num_images > 0 and input_ids.numel() < input_ids_pre_trunc.numel():
            image_token_id = self.image_token_id
            if image_token_id is not None:
                image_blocks = self._find_contiguous_blocks(input_ids_pre_trunc, int(image_token_id))
                num_complete_images = sum(1 for _, end in image_blocks if end <= self.seq_length)
                if num_complete_images < len(image_blocks):
                    for start, end in image_blocks[num_complete_images:]:
                        block_start = max(start, 0)
                        block_end = min(end, self.seq_length)
                        if block_start < block_end:
                            input_ids[block_start:block_end] = self.pad_token_id
                            labels[block_start:block_end] = IGNORE_INDEX
                            loss_mask[block_start:block_end] = 0.0
                    if num_complete_images < num_images:
                        logger.warning(
                            "Truncation to seq_length=%d removed %d of %d images whose token blocks did not fit.",
                            self.seq_length,
                            num_images - num_complete_images,
                            num_images,
                        )

        visual_tensors: dict[str, torch.Tensor] = {}
        for key in self.visual_keys:
            value = proc_output.get(key)
            if value is None:
                continue
            visual_tensors[key] = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)

        if num_complete_images < num_images:
            for key in list(visual_tensors):
                value = visual_tensors[key]
                if value.dim() >= 1 and value.shape[0] == num_images:
                    if num_complete_images > 0:
                        visual_tensors[key] = value[:num_complete_images]
                    else:
                        del visual_tensors[key]

        return HFProcessorEncodedSample(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            visual_tensors=visual_tensors,
        )
