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

"""Qwen VL collator implementations."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.collators.visual import THW_GRID_VISUAL_KEYS
from megatron.bridge.data.conversation_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
    resolve_chat_template_kwargs,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.packing.in_batch import build_mcore_thd_sequence_batch_from_rows
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


MISSING_QWEN_VL_UTILS_MSG = (
    "qwen_vl_utils is required for Qwen2.5 VL processing. Please `pip install qwen-vl-utils` or"
    " provide compatible vision preprocessing."
)
QWEN_VL_MIN_PIXELS = 200704
QWEN_VL_MAX_PIXELS = 1003520
CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_ASSISTANT_END = "<|im_end|>\n"
CHATML_OTHER_ROLE_STARTS = {role: f"<|im_start|>{role}\n" for role in ("system", "developer", "user", "tool")}
QWEN_VISUAL_KEYS = (*THW_GRID_VISUAL_KEYS, "second_per_grid_ts")

try:
    from qwen_vl_utils import process_vision_info

    HAVE_QWEN_VL_UTILS = True
except ImportError:
    HAVE_QWEN_VL_UTILS = False


@dataclass
class QwenVLPreparedSequence:
    """One fully processed, unpadded Qwen-VL sequence and its visual payload.

    Note:
        This model-owned storage layout is provisional and may evolve as more
        VLMs adopt native packing. Callers should use ``sequence_length`` when
        they do not need Qwen-specific ``row`` or ``visual_values`` payloads.
    """

    row: dict[str, torch.Tensor]
    visual_values: dict[str, torch.Tensor]

    @property
    def sequence_length(self) -> int:
        """Return the exact post-processor token length."""
        return int(self.row["input_ids"].numel())


def _normalize_qwen_video_paths(example: dict[str, Any]) -> dict[str, Any]:
    """Map path-based video parts to the inline schema expected by Qwen processors."""
    conversation = example.get("conversation")
    if not isinstance(conversation, list):
        return example

    normalized_conversation = []
    example_changed = False
    for turn in conversation:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("content"), list):
            normalized_conversation.append(turn)
            continue

        normalized_content = []
        turn_changed = False
        for part in turn["content"]:
            if isinstance(part, Mapping) and part.get("type") == "video" and "video" not in part and "path" in part:
                normalized_part = dict(part)
                normalized_part["video"] = normalized_part.pop("path")
                normalized_content.append(normalized_part)
                turn_changed = True
            else:
                normalized_content.append(part)

        if turn_changed:
            normalized_turn = dict(turn)
            normalized_turn["content"] = normalized_content
            normalized_conversation.append(normalized_turn)
            example_changed = True
        else:
            normalized_conversation.append(turn)

    if not example_changed:
        return example
    normalized_example = dict(example)
    normalized_example["conversation"] = normalized_conversation
    return normalized_example


@dataclass(frozen=True)
class _QwenVLMedia:
    """Images and videos of one example after qwen-vl-utils decodes them."""

    images: list[Any]
    videos: list[Any]
    video_metadata: list[Any]


def _takes_default_pixel_bounds(part: Any) -> bool:
    """Return whether a content part is an image without its own pixel bounds.

    A bound with the value ``None`` is not set, as in qwen-vl-utils.
    """
    return (
        isinstance(part, Mapping)
        and ("image" in part or "image_url" in part)
        and part.get("min_pixels") is None
        and part.get("max_pixels") is None
    )


def with_default_pixel_bounds(part: Any, *, min_pixels: int | None, max_pixels: int | None) -> Any:
    """Return a content part with the default pixel bounds of ``qwen3_vl_collate_fn``.

    Only an image part without its own ``min_pixels`` and ``max_pixels`` gets the default bounds. An image part
    that sets one of them keeps its own bounds, as at inference. qwen-vl-utils gives the default of a bound that
    is not set.

    Args:
        part: One content part of a conversation turn.
        min_pixels: Default lower pixel bound of an image. ``None`` keeps the default of qwen-vl-utils.
        max_pixels: Default upper pixel bound of an image. ``None`` keeps the default of qwen-vl-utils. A
            ``max_pixels`` below ``min_pixels`` also lowers ``min_pixels``.

    Returns:
        The part itself, or a copy of the image part with the default bounds.
    """
    if not _takes_default_pixel_bounds(part):
        return part
    if min_pixels is not None and max_pixels is not None:
        # qwen-vl-utils requires min_pixels <= max_pixels. max_pixels sets the visual token budget, so a
        # max_pixels below min_pixels also lowers min_pixels.
        min_pixels = min(min_pixels, max_pixels)
    bounds = {
        key: value for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels)) if value is not None
    }
    return {**part, **bounds}


def _with_image_pixel_bounds(conversation: list[Any], *, min_pixels: int | None, max_pixels: int | None) -> list[Any]:
    """Return a copy of the conversation with default pixel bounds on its image parts."""
    if min_pixels is None and max_pixels is None:
        return conversation
    bounded_conversation = []
    for turn in conversation:
        if not isinstance(turn, Mapping) or not isinstance(turn.get("content"), list):
            bounded_conversation.append(turn)
            continue
        content = [
            with_default_pixel_bounds(part, min_pixels=min_pixels, max_pixels=max_pixels) for part in turn["content"]
        ]
        bounded_conversation.append({**turn, "content": content})
    return bounded_conversation


def _fetch_qwen_vl_media(
    conversation: list[Any],
    processor: Any,
    *,
    qwen3_vl: bool,
    min_pixels: int | None,
    max_pixels: int | None,
) -> _QwenVLMedia:
    """Decode the images and videos of one conversation with qwen-vl-utils.

    qwen-vl-utils samples the video frames. Its video metadata keeps the source fps and the indices of
    the sampled frames. Qwen3-VL writes the frame timestamps from this metadata, and Qwen2.5-VL
    calculates ``second_per_grid_ts`` from it.

    For Qwen3-VL, qwen-vl-utils also resizes the media, as in the Qwen3-VL model card, and the processor
    does not resize them again. qwen-vl-utils uses the patch size of the processor and the image pixel
    bounds of the collate.
    """
    vision_kwargs: dict[str, Any] = {"return_video_metadata": True}
    if qwen3_vl:
        conversation = _with_image_pixel_bounds(conversation, min_pixels=min_pixels, max_pixels=max_pixels)
        vision_kwargs["image_patch_size"] = processor.image_processor.patch_size
    images, videos = process_vision_info(conversation, **vision_kwargs)
    if images is None:
        images = []
    elif not isinstance(images, list):
        images = [images]
    return _QwenVLMedia(
        images=images,
        videos=[frames for frames, _ in videos or []],
        video_metadata=[metadata for _, metadata in videos or []],
    )


def _qwen_vl_media_kwargs(
    media: list[_QwenVLMedia],
    *,
    qwen3_vl: bool,
    min_pixels: int | None,
    max_pixels: int | None,
) -> dict[str, Any]:
    """Return the media arguments of one processor call on the rows in ``media``.

    A call on one row gets flat media lists. A call on more rows gets one media list for each row.
    """
    kwargs: dict[str, Any] = {}
    if qwen3_vl:
        # qwen-vl-utils resized the media already. A second resize gives sizes that inference does not use.
        kwargs["do_resize"] = False
    else:
        if min_pixels is not None:
            kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            kwargs["max_pixels"] = max_pixels
    single_row = len(media) == 1
    if any(row.images for row in media):
        kwargs["images"] = media[0].images if single_row else [row.images for row in media]
    if any(row.videos for row in media):
        kwargs["videos"] = media[0].videos if single_row else [row.videos for row in media]
        kwargs["video_metadata"] = media[0].video_metadata if single_row else [row.video_metadata for row in media]
        # qwen-vl-utils sampled the frames already. Without this flag, Qwen3-VL samples them again.
        kwargs["do_sample_frames"] = False
    return kwargs


def _raise_if_truncation_cuts_visual_tokens(
    input_ids: torch.Tensor, processor: Any, sequence_length: int | None
) -> None:
    """Raise a ValueError if a truncation to ``sequence_length`` removes visual tokens.

    The visual features stay complete, so the remaining visual tokens would not match them.
    """
    if sequence_length is None or input_ids.size(1) <= sequence_length:
        return
    token_ids = [
        token_id
        for token_id in (getattr(processor, "image_token_id", None), getattr(processor, "video_token_id", None))
        if isinstance(token_id, int)
    ]
    if not token_ids:
        return
    cut_rows = torch.isin(input_ids[:, sequence_length:], torch.tensor(token_ids, device=input_ids.device)).any(dim=1)
    if cut_rows.any():
        raise ValueError(
            f"Truncating sequence batch from {input_ids.size(1)} to {sequence_length} tokens removes visual tokens "
            f"from rows {cut_rows.nonzero().flatten().tolist()}. Increase sequence_length, or lower max_pixels or "
            "the number of video frames."
        )


def _prepare_qwen_vl_sequence(
    example: dict[str, Any],
    processor: Any,
    *,
    text: str,
    media_kwargs: dict[str, Any],
    skipped_tokens: torch.Tensor,
    boundary_config: Any,
    require_assistant_matches: bool,
) -> QwenVLPreparedSequence:
    """Process one example into the exact row consumed by THD packing."""
    processor_kwargs: dict[str, Any] = {
        "text": [text],
        "padding": False,
        "return_tensors": "pt",
        **media_kwargs,
    }

    with use_processor_right_padding(processor):
        sample_batch = {
            key: value.contiguous() if isinstance(value, torch.Tensor) else value
            for key, value in processor(**processor_kwargs).items()
        }

    input_ids = sample_batch["input_ids"][0]
    attention_mask = sample_batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask[0]
    position_ids = sample_batch.get("position_ids")
    if position_ids is None:
        position_ids = torch.arange(input_ids.numel(), device=input_ids.device, dtype=torch.long)
    else:
        position_ids = position_ids[0]

    loss_mask = build_assistant_loss_mask(
        example,
        input_ids,
        processor,
        skipped_tokens,
        boundary_config=boundary_config,
        warn_on_all_masked=not require_assistant_matches,
    ).to(device=input_ids.device, dtype=torch.float32)
    labels = torch.cat([input_ids[1:], input_ids.new_full((1,), IGNORE_INDEX)])
    if skipped_tokens.numel() > 0:
        labels = labels.masked_fill(torch.isin(labels, skipped_tokens.to(device=labels.device)), IGNORE_INDEX)
    shifted_loss_mask = torch.cat([loss_mask[1:], loss_mask.new_zeros(1)])

    visual_values = {}
    for key in QWEN_VISUAL_KEYS:
        value = sample_batch.get(key)
        if not isinstance(value, torch.Tensor):
            continue
        if key in {"pixel_values", "pixel_values_videos"} and value.dim() == 5:
            value = value.flatten(0, 1)
        elif key in {"image_grid_thw", "video_grid_thw"} and value.dim() == 3:
            value = value.flatten(0, 1)
        visual_values[key] = value

    return QwenVLPreparedSequence(
        row={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels.masked_fill(shifted_loss_mask == 0, IGNORE_INDEX),
            "loss_mask": shifted_loss_mask,
        },
        visual_values=visual_values,
    )


def prepare_qwen_vl_sequence(
    example: dict[str, Any],
    processor: Any,
    *,
    min_pixels: int | None = QWEN_VL_MIN_PIXELS,
    max_pixels: int | None = QWEN_VL_MAX_PIXELS,
    require_assistant_matches: bool = False,
    qwen3_vl: bool | None = None,
) -> QwenVLPreparedSequence:
    """Process one Qwen-VL example once for online or collate-time packing.

    Args:
        example: HF-style example with a ``conversation``.
        processor: Qwen-VL HF processor.
        min_pixels: Lower pixel bound of each image.
        max_pixels: Upper pixel bound of each image.
        require_assistant_matches: Do not warn when the loss mask has no assistant token.
        qwen3_vl: Use the media settings of ``qwen3_vl_collate_fn``. ``None`` uses them for a ``Qwen3VLProcessor``,
            as the collate registry does.

    Returns:
        The unpadded sequence and its visual tensors.
    """
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)
    if qwen3_vl is None:
        qwen3_vl = type(processor).__name__ == "Qwen3VLProcessor"

    example = _normalize_qwen_video_paths(example)
    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_ASSISTANT_END,
        assistant_end_fallbacks=("<|im_end|>",),
        role_start_markers=CHATML_OTHER_ROLE_STARTS,
    )
    text = processor.apply_chat_template(
        example["conversation"],
        tokenize=False,
        **resolve_chat_template_kwargs(processor, chat_template_kwargs_from_example(example)),
    )
    media = _fetch_qwen_vl_media(
        example["conversation"], processor, qwen3_vl=qwen3_vl, min_pixels=min_pixels, max_pixels=max_pixels
    )

    return _prepare_qwen_vl_sequence(
        example,
        processor,
        text=text,
        media_kwargs=_qwen_vl_media_kwargs([media], qwen3_vl=qwen3_vl, min_pixels=min_pixels, max_pixels=max_pixels),
        skipped_tokens=skipped_tokens,
        boundary_config=boundary_config,
        require_assistant_matches=require_assistant_matches,
    )


def build_qwen_vl_packed_batch(
    sequences: list[QwenVLPreparedSequence],
    *,
    sequence_length: int | None,
    pad_to_multiple_of: int,
    pad_to_max_length: bool = False,
) -> dict[str, Any]:
    """Build one canonical MCore THD row from prepared Qwen-VL sequences.

    When ``pad_to_max_length`` is true, the last physical segment is padded so
    the row reaches ``sequence_length`` while the unpadded THD boundaries keep
    describing only real tokens.
    """
    packed_batch = build_mcore_thd_sequence_batch_from_rows(
        [sequence.row for sequence in sequences],
        sequence_length=sequence_length,
        pad_token_id=0,
        ignore_index=IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple_of,
        pad_to_max_length=pad_to_max_length,
    )
    visual_values: dict[str, list[torch.Tensor]] = {key: [] for key in QWEN_VISUAL_KEYS}
    for sequence in sequences:
        for key, value in sequence.visual_values.items():
            visual_values[key].append(value)
    packed_batch["visual_inputs"] = GenericVisualInputs(
        **{key: torch.cat(values, dim=0) for key, values in visual_values.items() if values}
    )
    return packed_batch


def qwen2_5_collate_fn(
    examples: list,
    processor,
    min_pixels: int | None = QWEN_VL_MIN_PIXELS,
    max_pixels: int | None = QWEN_VL_MAX_PIXELS,
    visual_keys: object = None,
    require_assistant_matches: bool = False,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for Qwen2.5 VL model."""
    del visual_keys
    return _collate_qwen_vl_examples(
        examples,
        processor,
        qwen3_vl=False,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        require_assistant_matches=require_assistant_matches,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
    )


def qwen3_vl_collate_fn(
    examples: list,
    processor: Any,
    *,
    min_pixels: int | None = QWEN_VL_MIN_PIXELS,
    max_pixels: int | None = QWEN_VL_MAX_PIXELS,
    visual_keys: object = None,
    require_assistant_matches: bool = False,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    enable_in_batch_packing: bool = False,
    in_batch_packing_pad_to_multiple_of: int = 1,
) -> dict[str, torch.Tensor]:
    """Collate function for models that use ``Qwen3VLProcessor``, for example Qwen3-VL and Qwen3.5-VL.

    The media get the preprocessing of the Qwen3-VL model card. qwen-vl-utils resizes the media at the
    patch size of the processor: an image one time, a video file one time after it samples the frames,
    and a list of frames two times (each frame as an image, then the clip). The processor gets the video
    metadata, ``do_sample_frames=False``, and ``do_resize=False``. Thus each video keeps its sampled
    frames and its frame timestamps.

    ``min_pixels`` and ``max_pixels`` are the default pixel bounds of each image in that resize. An
    image part that sets its own bounds keeps them. Set them to ``None`` to use the bounds of qwen-vl-utils.
    """
    del visual_keys
    return _collate_qwen_vl_examples(
        examples,
        processor,
        qwen3_vl=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        require_assistant_matches=require_assistant_matches,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
    )


def _collate_qwen_vl_examples(
    examples: list,
    processor: Any,
    *,
    qwen3_vl: bool,
    min_pixels: int | None,
    max_pixels: int | None,
    require_assistant_matches: bool,
    sequence_length: int | None,
    pad_to_max_length: bool,
    pad_to_multiple_of: int,
    enable_in_batch_packing: bool,
    in_batch_packing_pad_to_multiple_of: int,
) -> dict[str, torch.Tensor]:
    """Collate Qwen-VL examples with the media settings of Qwen3-VL or Qwen2.5-VL."""
    if not HAVE_QWEN_VL_UTILS:
        raise ImportError(MISSING_QWEN_VL_UTILS_MSG)

    examples = [_normalize_qwen_video_paths(example) for example in examples]
    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_ASSISTANT_END,
        assistant_end_fallbacks=("<|im_end|>",),
        role_start_markers=CHATML_OTHER_ROLE_STARTS,
    )

    texts = [
        processor.apply_chat_template(
            example["conversation"],
            tokenize=False,
            **resolve_chat_template_kwargs(processor, chat_template_kwargs_from_example(example)),
        )
        for example in examples
    ]
    # Build per-example media (list) and split by presence.  Qwen processors accept
    # nested per-example image/video lists; splitting avoids passing empty media
    # kwargs for text-only rows.
    media = [
        _fetch_qwen_vl_media(
            example["conversation"], processor, qwen3_vl=qwen3_vl, min_pixels=min_pixels, max_pixels=max_pixels
        )
        for example in examples
    ]
    has_media = [bool(row.images or row.videos) for row in media]

    if enable_in_batch_packing:
        prepared_sequences = []
        for example, text, row_media in zip(examples, texts, media, strict=True):
            prepared = _prepare_qwen_vl_sequence(
                example,
                processor,
                text=text,
                media_kwargs=_qwen_vl_media_kwargs(
                    [row_media], qwen3_vl=qwen3_vl, min_pixels=min_pixels, max_pixels=max_pixels
                ),
                skipped_tokens=skipped_tokens,
                boundary_config=boundary_config,
                require_assistant_matches=require_assistant_matches,
            )
            prepared_sequences.append(prepared)

        return build_qwen_vl_packed_batch(
            prepared_sequences,
            sequence_length=sequence_length,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            pad_to_max_length=pad_to_max_length,
        )

    idx_with = [i for i, h in enumerate(has_media) if h]
    idx_without = [i for i, h in enumerate(has_media) if not h]

    batch_with = None
    batch_without = None

    with use_processor_right_padding(processor):
        if idx_with:
            processor_kwargs = {
                "text": [texts[i] for i in idx_with],
                "padding": True,
                "return_tensors": "pt",
                **_qwen_vl_media_kwargs(
                    [media[i] for i in idx_with], qwen3_vl=qwen3_vl, min_pixels=min_pixels, max_pixels=max_pixels
                ),
            }
            batch_with = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(**processor_kwargs).items()
            }

        if idx_without:
            texts_without = [texts[i] for i in idx_without]
            batch_without = {
                key: value.contiguous() if isinstance(value, torch.Tensor) else value
                for key, value in processor(
                    text=texts_without,
                    padding=True,
                    return_tensors="pt",
                ).items()
            }
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

        def pad_to(x, tgt_len, value):
            if x.shape[1] == tgt_len:
                return x
            pad_len = tgt_len - x.shape[1]
            return F.pad(x, (0, pad_len), value=value)

        in_with = pad_to(in_with, max_len, pad_id)
        in_without = pad_to(in_without, max_len, pad_id)

        input_ids = torch.full((len(examples), max_len), pad_id, dtype=in_with.dtype)
        # Place rows
        for row, i in enumerate(idx_with):
            input_ids[i] = in_with[row]
        for row, i in enumerate(idx_without):
            input_ids[i] = in_without[row]

        batch = {"input_ids": input_ids}
        if "attention_mask" in batch_with and "attention_mask" in batch_without:
            attn_with = pad_to(batch_with["attention_mask"], max_len, 0)
            attn_without = pad_to(batch_without["attention_mask"], max_len, 0)
            attention_mask = torch.zeros((len(examples), max_len), dtype=attn_with.dtype)
            for row, i in enumerate(idx_with):
                attention_mask[i] = attn_with[row]
            for row, i in enumerate(idx_without):
                attention_mask[i] = attn_without[row]
            batch["attention_mask"] = attention_mask
        # Carry over vision tensors if present
        for key in QWEN_VISUAL_KEYS:
            if key in batch_with:
                batch[key] = batch_with[key]

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
            build_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
                warn_on_all_masked=not require_assistant_matches,
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

    visual_inputs = GenericVisualInputs(
        pixel_values=batch.get("pixel_values"),
        pixel_values_videos=batch.get("pixel_values_videos"),
        image_grid_thw=batch.get("image_grid_thw"),
        video_grid_thw=batch.get("video_grid_thw"),
        second_per_grid_ts=batch.get("second_per_grid_ts"),
    )
    for key in QWEN_VISUAL_KEYS:
        batch.pop(key, None)
    batch["visual_inputs"] = visual_inputs
    _raise_if_truncation_cuts_visual_tokens(batch["input_ids"], processor, sequence_length)
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
    return batch
