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
import re
import warnings
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


@dataclass(frozen=True)
class NormalizedVLMSample:
    """Source-normalized VLM sample consumed by shared processing.

    Expected input format:
        Instances are produced by source adapters such as
        :func:`normalize_energon_vlm_sample` and
        :func:`normalize_hf_vlm_example`.  ``conversation`` must be a structured
        list of chat turns in HF processor format, for example::

            [
                {"role": "user", "content": "describe <image>"},
                {"role": "assistant", "content": "a red car"},
            ]

        ``content`` may also be a multimodal content list accepted by
        ``processor.apply_chat_template``, for example::

            [{"type": "image", "image": image_obj}, {"type": "text", "text": "describe"}]

        ``images`` and ``videos`` are optional processor-ready modality payloads.
        Energon adapters convert WDS tensors to PIL objects before populating
        these fields; HF adapters may leave them ``None`` when media already lives
        inline in ``conversation`` content.

    Output format:
        Shared processing treats this as the single boundary contract before
        model-specific tokenization and vision preprocessing.  It does not
        contain batched tensors; model-specific collators convert it into model
        input tensors.
    """

    conversation: list[dict[str, Any]]
    images: list[Any] | None = None
    videos: list[Any] | None = None
    audio: Any | None = None


def normalize_energon_vlm_sample(sample: Any) -> NormalizedVLMSample:
    """Normalize an Energon ``ChatMLSample`` into the shared VLM sample contract.

    Expected input format:
        ``sample`` is expected to expose the Energon ``ChatMLSample`` fields:

        - ``conversation``: JSON string accepted by ``cook_chatml_sample``.  The
          JSON may use either ``{"role": ..., "content": ...}`` turns or
          ``{"from": ..., "value": ...}`` turns.
        - ``imgs``: optional WDS decoded image tensor/list payload.
        - ``videos``: optional WDS decoded video tensor/list payload.
        - ``audio``: optional audio payload, passed through unchanged.

    Output format:
        Returns ``NormalizedVLMSample`` where ``conversation`` is a list of
        ``{"role": str, "content": str | list[dict]}`` turns, ``images`` are
        PIL/list processor inputs or ``None``, ``videos`` are nested PIL/list
        processor inputs or ``None``, and ``audio`` is copied from the source
        sample when present.
    """
    from megatron.bridge.data.energon.task_encoder_utils import (
        _images_to_pil,
        _videos_to_pil,
        cook_chatml_sample,
    )

    imgs = getattr(sample, "imgs", None)
    videos = getattr(sample, "videos", None)
    return NormalizedVLMSample(
        conversation=cook_chatml_sample(sample.conversation),
        images=_images_to_pil(imgs) if imgs is not None and len(imgs) > 0 else None,
        videos=_videos_to_pil(videos) if videos is not None and len(videos) > 0 else None,
        audio=getattr(sample, "audio", None),
    )


def normalize_hf_vlm_example(example: Mapping[str, Any]) -> NormalizedVLMSample:
    """Normalize a HF-style VLM dataset example into the shared sample contract.

    Expected input format:
        ``example`` must contain ``"conversation"`` as a structured list of chat
        turns already produced by an HF dataset maker, for example::

            {
                "conversation": [
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Q"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "A"}]},
                ],
                "audio": optional_audio,
            }

        Optional top-level ``images``/``image`` and ``videos``/``video`` fields
        are accepted for maker variants that do not embed media inline in the
        conversation.

    Output format:
        Returns ``NormalizedVLMSample`` with a deep-copied structured
        ``conversation`` list, optional list-valued ``images`` and ``videos``
        payloads, and optional ``audio``.  The adapter does not call
        ``cook_chatml_sample`` because HF makers have already normalized the chat
        schema.

    Raises:
        ValueError: If ``example["conversation"]`` is missing or is not a list.
    """
    conversation = example.get("conversation")
    if not isinstance(conversation, list):
        raise ValueError("HF VLM examples must contain a list-valued 'conversation' field.")

    images = example.get("images")
    if images is None and "image" in example:
        images = [example["image"]]
    videos = example.get("videos")
    if videos is None and "video" in example:
        videos = [example["video"]]

    return NormalizedVLMSample(
        conversation=copy.deepcopy(conversation),
        images=images,
        videos=videos,
        audio=example.get("audio"),
    )


def normalized_vlm_sample_to_hf_example(
    sample: NormalizedVLMSample,
    *,
    media_first: bool = False,
) -> dict[str, Any]:
    """Convert a normalized VLM sample into the HF-style collate example schema.

    Expected input format:
        ``sample`` follows ``NormalizedVLMSample``: ``conversation`` is a list of
        chat turns, and optional ``images``/``videos`` contain processor-ready
        media payloads such as PIL images or decoded video frame lists.  Text
        turns may contain literal ``<image>`` / ``<video>`` placeholders.

    Output format:
        Returns a dictionary suitable for VLM HF collate functions::

            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image_obj},
                            {"type": "text", "text": "describe"},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
                ],
                "images": [image_obj],   # present when sample.images is not None
                "videos": [video_obj],   # present when sample.videos is not None
                "audio": audio_obj,      # present when sample.audio is not None
            }

        Inline media parts are populated from ``sample.images`` and
        ``sample.videos`` in placeholder order.  When ``media_first=True``,
        media parts are moved before text parts within each turn to preserve
        Qwen Energon's legacy media-before-text ordering while still using the
        shared HF collate function.
    """
    images = list(sample.images) if sample.images is not None else []
    videos = list(sample.videos) if sample.videos is not None else []
    image_idx = 0
    video_idx = 0

    def _media_part(media_type: str) -> dict[str, Any]:
        nonlocal image_idx, video_idx
        part: dict[str, Any] = {"type": media_type}
        if media_type == "image":
            if image_idx < len(images):
                part["image"] = images[image_idx]
                image_idx += 1
        elif media_type == "video" and video_idx < len(videos):
            part["video"] = videos[video_idx]
            video_idx += 1
        return part

    def _normalize_content(content: Any) -> Any:
        nonlocal image_idx, video_idx
        if isinstance(content, list):
            parts: list[dict[str, Any] | str] = []
            for item in content:
                if not isinstance(item, Mapping):
                    parts.append(item)
                    continue
                item_copy = dict(item)
                item_type = item_copy.get("type")
                if item_type == "image" and "image" not in item_copy and image_idx < len(images):
                    item_copy["image"] = images[image_idx]
                    image_idx += 1
                elif item_type == "video" and "video" not in item_copy and video_idx < len(videos):
                    item_copy["video"] = videos[video_idx]
                    video_idx += 1
                parts.append(item_copy)
            if media_first:
                media_parts = [
                    part for part in parts if isinstance(part, Mapping) and part.get("type") in ("image", "video")
                ]
                other_parts = [part for part in parts if part not in media_parts]
                return media_parts + other_parts
            return parts

        if not isinstance(content, str) or ("<image>" not in content and "<video>" not in content):
            return copy.deepcopy(content)

        pieces = re.split(r"(<image>|<video>)", content)
        parts = []
        for piece in pieces:
            if piece == "<image>":
                parts.append(_media_part("image"))
            elif piece == "<video>":
                parts.append(_media_part("video"))
            elif piece:
                parts.append({"type": "text", "text": piece.strip(" ")})
        if media_first:
            media_parts = [part for part in parts if part.get("type") in ("image", "video")]
            text_parts = [part for part in parts if part.get("type") not in ("image", "video")]
            return media_parts + text_parts
        return parts

    conversation = []
    for turn in sample.conversation:
        turn_copy = dict(turn)
        turn_copy["content"] = _normalize_content(turn_copy.get("content", ""))
        conversation.append(turn_copy)

    example: dict[str, Any] = {"conversation": conversation}
    if sample.images is not None:
        example["images"] = images
    if sample.videos is not None:
        example["videos"] = videos
    if sample.audio is not None:
        example["audio"] = sample.audio
    return example


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
    normalized_samples = [normalize_hf_vlm_example(example) for example in examples]
    loss_masks = [
        build_assistant_loss_mask(sample.conversation, input_ids, processor, skipped_tokens)
        for sample, input_ids in zip(normalized_samples, batch["input_ids"])
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
