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

"""Shared Nemotron Omni collation for Direct-HF and Energon datasets."""

from __future__ import annotations

import copy
import tempfile
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from megatron.bridge.data.collators.sequence import prepare_sequence_batch
from megatron.bridge.data.collators.sequence_padding import use_processor_right_padding
from megatron.bridge.data.conversation_processing import (
    assistant_mask_boundary_config_from_markers,
    build_assistant_loss_mask,
    chat_template_kwargs_from_example,
)
from megatron.bridge.data.datasets.utils import IGNORE_INDEX
from megatron.bridge.data.token_utils import extract_skipped_token_ids
from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import (
    COMPACT_IMAGE_PLACEHOLDER,
    patchify_temporal_frame,
    temporal_model_frames,
)
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_ASSISTANT_END = "<|im_end|>\n"
CHATML_OTHER_ROLE_STARTS = {role: f"<|im_start|>{role}\n" for role in ("system", "developer", "user", "tool")}
VISION_FRAME_SIZE = 512
PIXEL_SHUFFLE_FACTOR = 2
_NEMOTRON_OMNI_VISUAL_KEYS = ("pixel_values",)


def _validate_nemotron_omni_visual_keys(visual_keys: object = None) -> None:
    """Validate the model-owned visual input contract retained for API compatibility."""
    if visual_keys is None:
        return
    if isinstance(visual_keys, str):
        requested_keys = (visual_keys,)
    else:
        try:
            requested_keys = tuple(visual_keys)
        except TypeError as error:
            raise ValueError("Nemotron Omni visual_keys must contain only 'pixel_values'.") from error
    if requested_keys != _NEMOTRON_OMNI_VISUAL_KEYS:
        raise ValueError(
            "Nemotron Omni owns its visual input contract; visual_keys must be exactly ('pixel_values',)."
        )


def _pad_text_rows(
    rows: Sequence[torch.Tensor],
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad unbatched token rows without deriving padding from token values."""
    if not rows:
        raise ValueError("Nemotron Omni collation requires at least one example.")
    max_length = max(int(row.numel()) for row in rows)
    input_ids = torch.full((len(rows), max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    for row_index, row in enumerate(rows):
        row = row.to(dtype=torch.long).flatten()
        input_ids[row_index, : row.numel()] = row
        attention_mask[row_index, : row.numel()] = 1
    return input_ids, attention_mask


def _pil_images(payload: Any) -> list[Any]:
    """Normalize one image/frame payload to a flat list of PIL images."""
    if payload is None:
        return []
    if isinstance(payload, torch.Tensor):
        from PIL import Image

        if payload.dim() == 4:
            return [Image.fromarray(_tensor_image_to_uint8(image)) for image in payload]
        if payload.dim() == 3:
            return [Image.fromarray(_tensor_image_to_uint8(payload))]
        raise ValueError(f"Image tensors must have shape [C,H,W] or [N,C,H,W], got {tuple(payload.shape)}.")
    if isinstance(payload, (list, tuple)):
        result: list[Any] = []
        for item in payload:
            result.extend(_pil_images(item))
        return result
    return [payload]


def _tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert one CHW image tensor in either [0, 1] or [0, 255] range to uint8."""
    array = image.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    if array.size and array.min() >= 0 and array.max() <= 1:
        array *= 255
    return array.clip(0, 255).astype(np.uint8)


def _decode_video_path(path: str, *, video_fps: float, video_nframes: int) -> tuple[list[Any], float]:
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    image_urls, metadata = maybe_path_or_url_to_data_urls(
        path,
        fps=max(0, int(video_fps)),
        nframe=max(0, int(video_nframes)),
        nframe_max=-1,
    )
    frames = [pil_image_from_base64(image_url) for image_url in image_urls]
    sampled_fps = float(getattr(metadata, "fps", 0) or video_fps)
    return frames, sampled_fps


def _video_frames(payload: Any, *, video_fps: float, video_nframes: int) -> tuple[list[Any], float]:
    """Decode raw/path video payloads or flatten already-decoded frame payloads."""
    if isinstance(payload, (bytes, bytearray)):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as temporary_video:
            temporary_video.write(payload)
            temporary_video.flush()
            return _decode_video_path(
                temporary_video.name,
                video_fps=video_fps,
                video_nframes=video_nframes,
            )
    if isinstance(payload, str):
        return _decode_video_path(payload, video_fps=video_fps, video_nframes=video_nframes)
    return _pil_images(payload), video_fps


def _patchify_frame(frame: Any, *, height: int, width: int, patch_dim: int) -> torch.Tensor:
    """Apply the public normalization kernel on MCore's square temporal canvas.

    The public HF processor preserves video aspect ratio, but pinned MCore's
    temporal path currently stacks tubelets and pixel-shuffles them as a common
    square grid. Until MCore supports ragged non-square tubelets, Bridge keeps
    the required 512-square compatibility canvas while matching HF's
    antialiased bicubic interpolation and RADIO normalization.
    """
    _pixel_shuffled_token_count(height=height, width=width, patch_dim=patch_dim)
    return patchify_temporal_frame(frame, height=height, width=width, patch_dim=patch_dim)


def _pixel_shuffled_token_count(*, height: int, width: int, patch_dim: int) -> int:
    """Return RADIO tokens after the model's fixed 2x2 spatial pixel shuffle."""
    if patch_dim < 1 or height % patch_dim or width % patch_dim:
        raise ValueError(f"Image {height}x{width} is not divisible by patch_dim={patch_dim}.")
    patch_rows, patch_cols = height // patch_dim, width // patch_dim
    if patch_rows % PIXEL_SHUFFLE_FACTOR or patch_cols % PIXEL_SHUFFLE_FACTOR:
        raise ValueError(
            f"Image {height}x{width} produces a {patch_rows}x{patch_cols} patch grid, which is not divisible "
            f"by the {PIXEL_SHUFFLE_FACTOR}x{PIXEL_SHUFFLE_FACTOR} vision pixel shuffle."
        )
    return (patch_rows // PIXEL_SHUFFLE_FACTOR) * (patch_cols // PIXEL_SHUFFLE_FACTOR)


def _render_text_conversation(example: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Replace structured image parts with literal placeholders in source order."""
    conversation: list[dict[str, Any]] = []
    images: list[Any] = []
    for turn in example["conversation"]:
        turn_copy = copy.deepcopy(turn)
        content = turn_copy.get("content")
        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == "image":
                    image = item.get("image", item.get("path"))
                    if image is None:
                        raise ValueError("Nemotron Omni image content must provide 'image' or 'path'.")
                    text_parts.append("<image>")
                    images.extend(_pil_images(image))
                elif isinstance(item, Mapping) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    text_parts.append(item)
            turn_copy["content"] = "\n".join(text_parts)
        elif content is not None and not isinstance(content, str):
            turn_copy["content"] = str(content)
        conversation.append(turn_copy)
    return conversation, images


def _prepare_temporal_rows(
    examples: Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    temporal_patch_size: int,
    video_fps: float,
    video_nframes: int,
    patch_dim: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    """Build row-local temporal prompts and one packed all-frame vision tensor."""
    if temporal_patch_size < 1:
        raise ValueError("temporal_patch_size must be at least 1.")
    frame_height = frame_width = VISION_FRAME_SIZE
    token_rows: list[torch.Tensor] = []
    mask_examples: list[dict[str, Any]] = []
    all_patches: list[torch.Tensor] = []
    all_sizes: list[list[int]] = []
    all_num_frames: list[int] = []
    placeholder_counts: list[int] = []

    for example in examples:
        conversation: list[dict[str, Any]] = []
        row_placeholder_count = 0
        for turn in example["conversation"]:
            turn_copy = copy.deepcopy(turn)
            content = turn_copy.get("content")
            if isinstance(content, list):
                text_parts: list[str] = []
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") == "image":
                        image = item.get("image", item.get("path"))
                        images = _pil_images(image)
                        if len(images) != 1:
                            raise ValueError(
                                "Each Nemotron Omni image content part must resolve to exactly one image."
                            )
                        image = images[0]
                        text_parts.append(COMPACT_IMAGE_PLACEHOLDER)
                        all_patches.append(
                            _patchify_frame(image, height=frame_height, width=frame_width, patch_dim=patch_dim)
                        )
                        all_sizes.append([frame_height, frame_width])
                        all_num_frames.append(1)
                        row_placeholder_count += 1
                    elif isinstance(item, Mapping) and item.get("type") == "video":
                        payload = item.get("video", item.get("path"))
                        frames, sampled_fps = _video_frames(
                            payload,
                            video_fps=video_fps,
                            video_nframes=video_nframes,
                        )
                        if not frames:
                            raise ValueError("Nemotron Omni temporal video content decoded to zero frames.")
                        video_lines = ["This is a video:"]
                        for frame_start in range(0, len(frames), temporal_patch_size):
                            group = frames[frame_start : frame_start + temporal_patch_size]
                            timestamps = [
                                f"{'Frame' if offset == 0 else 'frame'} {frame_start + offset + 1} sampled at "
                                f"{(frame_start + offset) / sampled_fps:.2f} seconds"
                                for offset in range(len(group))
                            ]
                            video_lines.append(" and ".join(timestamps) + f": {COMPACT_IMAGE_PLACEHOLDER}")
                            row_placeholder_count += 1
                        text_parts.append("\n".join(video_lines))
                        model_frames = temporal_model_frames(frames, temporal_patch_size)
                        all_patches.extend(
                            _patchify_frame(frame, height=frame_height, width=frame_width, patch_dim=patch_dim)
                            for frame in model_frames
                        )
                        all_sizes.extend([[frame_height, frame_width]] * len(model_frames))
                        all_num_frames.append(len(model_frames))
                    elif isinstance(item, Mapping) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, str):
                        text_parts.append(item)
                turn_copy["content"] = "\n".join(text_parts)
            elif content is not None and not isinstance(content, str):
                turn_copy["content"] = str(content)
            conversation.append(turn_copy)

        prompt = processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs_from_example(example),
        )
        audio_token = getattr(processor.tokenizer, "audio_token", "<so_embedding>")
        prompt = prompt.replace("<|audio_1|>", audio_token)
        with use_processor_right_padding(processor):
            output = processor.tokenizer(
                [prompt],
                padding=False,
                truncation=False,
                return_tensors=None,
            )
        token_rows.append(torch.as_tensor(output["input_ids"][0], dtype=torch.long))
        mask_example = dict(example)
        mask_example["conversation"] = conversation
        mask_examples.append(mask_example)
        placeholder_counts.append(row_placeholder_count)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0
    input_ids, attention_mask = _pad_text_rows(token_rows, pad_token_id=int(pad_token_id))
    num_image_tiles = torch.tensor(
        [1 for count in placeholder_counts for _ in range(count)],
        dtype=torch.int,
    )
    batch: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    if all_patches:
        batch["visual_inputs"] = GenericVisualInputs(
            pixel_values=torch.cat(all_patches, dim=0).unsqueeze(0).to(torch.bfloat16).contiguous()
        )
        batch["imgs_sizes"] = torch.tensor(all_sizes, dtype=torch.long)
        batch["num_frames"] = torch.tensor(all_num_frames, dtype=torch.long)
        batch["num_image_tiles"] = num_image_tiles
    else:
        batch["visual_inputs"] = None
    return (
        batch,
        mask_examples,
        num_image_tiles.to(dtype=torch.long),
    )


def _prepare_standard_rows(
    examples: Sequence[Mapping[str, Any]],
    processor: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor | None]:
    """Use the HF processor for text/images while preserving row ownership."""
    text_conversations: list[list[dict[str, Any]]] = []
    images_per_example: list[list[Any]] = []
    mask_examples: list[dict[str, Any]] = []
    for example in examples:
        conversation, images = _render_text_conversation(example)
        text_conversations.append(conversation)
        images_per_example.append(images)
        mask_example = dict(example)
        mask_example["conversation"] = conversation
        mask_examples.append(mask_example)

    prompts = [
        processor.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            **chat_template_kwargs_from_example(example),
        )
        for example, conversation in zip(examples, text_conversations, strict=True)
    ]
    audio_token = getattr(processor.tokenizer, "audio_token", "<so_embedding>")
    prompts = [prompt.replace("<|audio_1|>", audio_token) for prompt in prompts]
    all_images = [image for images in images_per_example for image in images]

    if not all_images:
        with use_processor_right_padding(processor):
            batch = processor.tokenizer(
                prompts,
                padding=processor.tokenizer.pad_token is not None,
                truncation=False,
                return_tensors="pt",
            )
        batch = dict(batch)
        batch.setdefault("attention_mask", torch.ones_like(batch["input_ids"], dtype=torch.long))
        return batch, mask_examples, None

    with use_processor_right_padding(processor):
        per_example = [
            processor(
                text=[prompt],
                images=images or None,
                padding=False,
                truncation=False,
                return_tensors=None,
            )
            for prompt, images in zip(prompts, images_per_example, strict=True)
        ]
    token_rows = [torch.as_tensor(output["input_ids"][0], dtype=torch.long) for output in per_example]
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0
    input_ids, attention_mask = _pad_text_rows(token_rows, pad_token_id=int(pad_token_id))
    pixel_values: list[torch.Tensor] = []
    for output in per_example:
        values = output.get("pixel_values")
        if isinstance(values, list):
            pixel_values.extend(torch.as_tensor(value) for value in values)
        elif torch.is_tensor(values) and values.dim() == 4:
            pixel_values.extend(values.unbind(0))
        elif torch.is_tensor(values) and values.dim() == 3:
            pixel_values.append(values)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
    num_tiles = torch.ones(len(pixel_values), dtype=torch.long)
    return batch, mask_examples, num_tiles


def _audio_waveform(example: Mapping[str, Any], *, target_sampling_rate: int = 16000) -> np.ndarray | None:
    from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import load_audio

    if example.get("audio_path") is not None:
        return load_audio(str(example["audio_path"]), target_sr=target_sampling_rate)
    audio = example.get("audio")
    if audio is None:
        return None
    sampling_rate = target_sampling_rate
    if isinstance(audio, Mapping):
        sampling_rate = int(audio.get("sampling_rate", target_sampling_rate))
        audio = audio.get("array")
    elif isinstance(audio, tuple) and len(audio) == 2:
        audio, sampling_rate = audio
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)
    if sampling_rate != target_sampling_rate:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sampling_rate, target_sr=target_sampling_rate)
    return waveform.astype(np.float32)


def _add_audio_inputs(
    batch: dict[str, Any],
    examples: Sequence[Mapping[str, Any]],
    processor: Any,
    *,
    max_audio_duration: float,
    num_mel_bins: int,
) -> None:
    """Extract audio features and align each row's sound placeholder count."""
    from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import compute_mel_features

    waveforms = [_audio_waveform(example) for example in examples]
    if not any(waveform is not None for waveform in waveforms):
        return
    if not all(waveform is not None for waveform in waveforms):
        raise ValueError("Nemotron Omni collation does not support mixing audio and no-audio samples.")

    mel_features: list[torch.Tensor] = []
    token_counts: list[int] = []
    for example, waveform in zip(examples, waveforms, strict=True):
        assert waveform is not None
        duration = float(example.get("max_audio_duration", max_audio_duration))
        waveform = waveform[: int(duration * 16000)]
        mel = compute_mel_features(waveform, sampling_rate=16000, num_mel_bins=num_mel_bins)
        mel_features.append(mel)
        token_length = int(mel.shape[0])
        for _ in range(3):
            token_length = (token_length + 1) // 2
        token_counts.append(max(1, token_length))

    sound_token_id = processor.tokenizer.convert_tokens_to_ids("<so_embedding>")
    sound_start_id = processor.tokenizer.convert_tokens_to_ids("<so_start>")
    sound_end_id = processor.tokenizer.convert_tokens_to_ids("<so_end>")
    image_end_id = processor.tokenizer.convert_tokens_to_ids("</img>")
    rows: list[torch.Tensor] = []
    for row_index, token_count in enumerate(token_counts):
        row_length = int(batch["attention_mask"][row_index].sum().item())
        row = batch["input_ids"][row_index, :row_length]
        sound_positions = torch.where(row == sound_token_id)[0]
        if sound_positions.numel() > 0:
            if sound_positions.numel() > 1 and not bool(torch.all(sound_positions[1:] == sound_positions[:-1] + 1)):
                raise ValueError(
                    "Nemotron Omni supports one contiguous audio placeholder block per sample; "
                    f"row {row_index} contains disjoint <so_embedding> runs."
                )
            first_position = int(sound_positions[0].item())
            last_position = int(sound_positions[-1].item()) + 1
            has_start = first_position > 0 and int(row[first_position - 1].item()) == sound_start_id
            has_end = last_position < row.numel() and int(row[last_position].item()) == sound_end_id
            if has_start != has_end:
                raise ValueError(
                    "Nemotron Omni audio placeholders must have both <so_start> and <so_end> delimiters or neither."
                )
            replacement = torch.tensor(
                ([sound_token_id] * token_count)
                if has_start
                else [sound_start_id, *([sound_token_id] * token_count), sound_end_id],
                dtype=row.dtype,
                device=row.device,
            )
            row = torch.cat((row[:first_position], replacement, row[last_position:]))
        else:
            image_end_positions = torch.where(row == image_end_id)[0]
            insertion = int(image_end_positions[-1].item()) + 1 if image_end_positions.numel() else min(1, row.numel())
            sound_block = torch.tensor(
                [sound_start_id, *([sound_token_id] * token_count), sound_end_id],
                dtype=row.dtype,
                device=row.device,
            )
            row = torch.cat((row[:insertion], sound_block, row[insertion:]))
        rows.append(row)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0
    batch["input_ids"], batch["attention_mask"] = _pad_text_rows(rows, pad_token_id=int(pad_token_id))
    max_mel_length = max(int(mel.shape[0]) for mel in mel_features)
    sound_clips = torch.zeros(len(mel_features), max_mel_length, num_mel_bins)
    for row_index, mel in enumerate(mel_features):
        sound_clips[row_index, : mel.shape[0]] = mel
    batch["sound_clips"] = sound_clips
    batch["sound_length"] = torch.tensor([mel.shape[0] for mel in mel_features], dtype=torch.long)


def _adjust_image_placeholders(
    batch: dict[str, Any],
    loss_mask: torch.Tensor,
    processor: Any,
    num_tiles: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import adjust_image_tokens

    image_start_id = processor.tokenizer.convert_tokens_to_ids("<img>")
    image_end_id = processor.tokenizer.convert_tokens_to_ids("</img>")
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(batch["input_ids"], dtype=torch.long)
    aligned = {"input_ids": batch["input_ids"], "loss_mask": loss_mask, "attention_mask": attention_mask}
    if not bool((batch["input_ids"] == image_start_id).any()):
        return aligned, loss_mask
    if num_tiles is None:
        raise ValueError("Image-bearing Nemotron Omni batches require one tile count per image placeholder.")
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0
    adjusted = adjust_image_tokens(
        aligned,
        num_tiles,
        image_start_id,
        image_end_id,
        padding_values={"input_ids": int(pad_token_id), "loss_mask": 0, "attention_mask": 0},
    )
    return adjusted, adjusted["loss_mask"]


def _pack_dynamic_images(batch: dict[str, Any], *, patch_dim: int) -> None:
    pixel_values = batch.pop("pixel_values", None)
    if not pixel_values:
        batch["visual_inputs"] = None
        return
    images = [torch.as_tensor(image).to(torch.bfloat16) for image in pixel_values]
    patches: list[torch.Tensor] = []
    sizes: list[list[int]] = []
    tile_counts: list[int] = []
    for image in images:
        if image.dim() != 3:
            raise ValueError(f"Expected one [3,H,W] image, got {tuple(image.shape)}.")
        channels, height, width = image.shape
        token_count = _pixel_shuffled_token_count(height=height, width=width, patch_dim=patch_dim)
        patch_rows, patch_cols = height // patch_dim, width // patch_dim
        patches.append(
            image.reshape(channels, patch_rows, patch_dim, patch_cols, patch_dim)
            .permute(1, 3, 0, 2, 4)
            .reshape(patch_rows * patch_cols, channels * patch_dim * patch_dim)
            .contiguous()
        )
        sizes.append([height, width])
        tile_counts.append(token_count)
    batch["visual_inputs"] = GenericVisualInputs(pixel_values=torch.cat(patches).unsqueeze(0).contiguous())
    batch["imgs_sizes"] = torch.tensor(sizes, dtype=torch.long)
    batch["num_frames"] = torch.ones(len(images), dtype=torch.long)
    batch["num_image_tiles"] = torch.tensor(tile_counts, dtype=torch.int)


def _model_merge_row_lengths(
    batch: Mapping[str, Any],
    processor: Any,
    *,
    use_per_image_token_counts: bool,
    patch_dim: int,
) -> torch.Tensor:
    """Return per-row lengths after MCore replaces compact image placeholders."""
    attention_mask = batch["attention_mask"].to(dtype=torch.bool)
    collapsed_lengths = attention_mask.sum(dim=1)
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    image_counts = torch.stack(
        [
            ((row == image_token_id) & row_mask).sum()
            for row, row_mask in zip(batch["input_ids"], attention_mask, strict=True)
        ]
    )
    total_images = int(image_counts.sum().item())
    if total_images == 0:
        return collapsed_lengths

    if use_per_image_token_counts:
        replacement_counts = batch.get("num_image_tiles")
        if replacement_counts is None:
            raise ValueError("Dynamic-resolution image batches require one model token count per image.")
        replacement_counts = torch.as_tensor(
            replacement_counts, dtype=torch.long, device=collapsed_lengths.device
        ).flatten()
    else:
        # Nemotron Omni's temporal path fixes RADIO input frames at 512x512 and
        # uses 2x2 pixel shuffle. Each temporal tubelet therefore replaces one
        # compact <image> token with this many model embeddings.
        tokens_per_image = _pixel_shuffled_token_count(
            height=VISION_FRAME_SIZE,
            width=VISION_FRAME_SIZE,
            patch_dim=patch_dim,
        )
        replacement_counts = torch.full(
            (total_images,), tokens_per_image, dtype=torch.long, device=collapsed_lengths.device
        )

    if replacement_counts.numel() != total_images:
        raise ValueError(
            "Nemotron Omni image metadata does not match compact placeholders: "
            f"got {replacement_counts.numel()} model token counts for {total_images} <image> tokens."
        )

    expanded_lengths = collapsed_lengths.clone()
    offset = 0
    for row_index, image_count_tensor in enumerate(image_counts):
        image_count = int(image_count_tensor.item())
        row_counts = replacement_counts[offset : offset + image_count]
        expanded_lengths[row_index] += row_counts.sum() - image_count
        offset += image_count
    return expanded_lengths


def _nonpacked_multimodal_compact_width(
    batch: Mapping[str, Any],
    post_merge_row_lengths: torch.Tensor,
    *,
    sequence_length: int,
    pad_to_max_length: bool,
    pad_to_multiple_of: int,
) -> int:
    """Choose a compact width whose model-side merged width stays in bounds."""
    if pad_to_multiple_of < 1:
        raise ValueError("pad_to_multiple_of must be >= 1.")

    attention_mask = batch["attention_mask"].to(dtype=torch.bool)
    active_compact_lengths = attention_mask.sum(dim=1).to(dtype=torch.long)
    merge_deltas = post_merge_row_lengths.to(dtype=torch.long) - active_compact_lengths
    if bool((merge_deltas < 0).any()):
        raise ValueError("Nemotron Omni model-merge metadata cannot shrink a compact sequence row.")

    compact_width = int(batch["input_ids"].shape[1])
    model_row_lengths = merge_deltas + compact_width
    if bool((model_row_lengths > sequence_length).any()):
        raise ValueError(
            "Nemotron Omni cannot fit the rectangular multimodal batch before model-side merge: "
            f"compact width {compact_width} produces model row lengths {model_row_lengths.tolist()} "
            f"(active compact lengths {active_compact_lengths.tolist()}) with sequence_length={sequence_length}."
        )

    max_merge_delta = int(merge_deltas.max().item())
    if pad_to_max_length:
        target_model_width = sequence_length
    else:
        current_model_width = int(model_row_lengths.max().item())
        aligned_model_width = (
            (current_model_width + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of
        target_model_width = min(sequence_length, aligned_model_width)
    return target_model_width - max_merge_delta


def _pack_omni_rows_to_mcore_thd(
    batch: dict[str, Any],
    post_merge_row_lengths: torch.Tensor,
    *,
    sequence_length: int | None,
    pad_to_max_length: bool,
    pad_to_multiple_of: int,
    pad_token_id: int,
) -> None:
    """Pack compact Omni rows using their lengths after model-side embedding merge."""
    if pad_to_multiple_of < 1:
        raise ValueError("in_batch_packing_pad_to_multiple_of must be >= 1.")
    if pad_to_max_length:
        if sequence_length is None:
            raise ValueError("pad_to_max_length requires sequence_length for Nemotron Omni packing.")
        if sequence_length % pad_to_multiple_of != 0:
            raise ValueError(
                "Nemotron Omni fixed-width packing requires sequence_length to be divisible by "
                f"in_batch_packing_pad_to_multiple_of; got {sequence_length} and {pad_to_multiple_of}."
            )

    tokens = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if not isinstance(tokens, torch.Tensor) or tokens.dim() != 2:
        raise ValueError("Nemotron Omni packing expects 2D input_ids.")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != tokens.shape:
        raise ValueError("Nemotron Omni packing requires an attention mask matching input_ids.")

    active_mask = attention_mask.to(device=tokens.device, dtype=torch.bool)
    compact_lengths = active_mask.sum(dim=1).to(dtype=torch.long)
    positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
    if not torch.equal(active_mask, positions < compact_lengths.unsqueeze(1)):
        raise ValueError("Nemotron Omni packing requires right-padded input rows.")
    if bool((compact_lengths == 0).any()):
        raise ValueError("Cannot pack a batch containing an empty Nemotron Omni sequence row.")

    merged_lengths = torch.as_tensor(post_merge_row_lengths, dtype=torch.long, device=tokens.device).flatten()
    if merged_lengths.numel() != tokens.size(0):
        raise ValueError("Post-merge row lengths must contain one entry per Nemotron Omni input row.")
    if bool((merged_lengths < compact_lengths).any()):
        raise ValueError("Nemotron Omni model-merge metadata cannot shrink a compact sequence row.")

    padded_merged_lengths = ((merged_lengths + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    total_merged_length = int(padded_merged_lengths.sum().item())
    if sequence_length is not None and total_merged_length > sequence_length:
        raise ValueError(
            "Nemotron Omni packed rows exceed configured sequence_length after model merge and "
            f"alignment: got per-row lengths {merged_lengths.tolist()}, aligned lengths "
            f"{padded_merged_lengths.tolist()}, and total {total_merged_length} with "
            f"sequence_length={sequence_length}."
        )
    if pad_to_max_length:
        assert sequence_length is not None
        # MCore widens Omni rows to the model sequence length for pipeline
        # parallelism. Make that tail part of the last physical THD segment so
        # its tensor width, padded boundary, total_tokens, and seq_idx agree.
        padded_merged_lengths[-1] += sequence_length - total_merged_length
        total_merged_length = sequence_length

    compact_physical_lengths = compact_lengths + padded_merged_lengths - merged_lengths
    compact_total_length = int(compact_physical_lengths.sum().item())
    sequence_pad_values: dict[str, int | float] = {
        "input_ids": pad_token_id,
        "position_ids": 0,
        "labels": IGNORE_INDEX,
        "loss_mask": 0,
    }
    packed_tensors: dict[str, torch.Tensor] = {}
    for key, pad_value in sequence_pad_values.items():
        tensor = batch.get(key)
        if not isinstance(tensor, torch.Tensor) or tensor.shape != tokens.shape:
            raise ValueError(f"Nemotron Omni packing requires '{key}' to match input_ids.")
        packed_tensors[key] = torch.full(
            (1, compact_total_length),
            pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    compact_offset = 0
    for row_index, (compact_length_tensor, physical_length_tensor) in enumerate(
        zip(compact_lengths, compact_physical_lengths, strict=True)
    ):
        compact_length = int(compact_length_tensor.item())
        physical_length = int(physical_length_tensor.item())
        for key, packed_tensor in packed_tensors.items():
            packed_tensor[0, compact_offset : compact_offset + compact_length] = batch[key][row_index, :compact_length]

        gap_length = physical_length - compact_length
        if gap_length > 0:
            row_positions = batch["position_ids"][row_index]
            start_position = row_positions[compact_length - 1] + 1
            packed_tensors["position_ids"][0, compact_offset + compact_length : compact_offset + physical_length] = (
                torch.arange(
                    start_position,
                    start_position + gap_length,
                    dtype=row_positions.dtype,
                    device=row_positions.device,
                )
            )
        compact_offset += physical_length

    batch.update(packed_tensors)
    batch["attention_mask"] = None

    cu_seqlens = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=tokens.device),
            torch.cumsum(merged_lengths, dim=0),
        )
    ).to(dtype=torch.int32)
    batch["cu_seqlens_q"] = cu_seqlens
    batch["cu_seqlens_kv"] = cu_seqlens
    if pad_to_multiple_of > 1 or pad_to_max_length:
        cu_seqlens_padded = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=tokens.device),
                torch.cumsum(padded_merged_lengths, dim=0),
            )
        ).to(dtype=torch.int32)
        batch["cu_seqlens_q_padded"] = cu_seqlens_padded
        batch["cu_seqlens_kv_padded"] = cu_seqlens_padded
    else:
        batch.pop("cu_seqlens_q_padded", None)
        batch.pop("cu_seqlens_kv_padded", None)
    max_seqlen = torch.tensor(int(padded_merged_lengths.max().item()), dtype=torch.int32)
    batch["max_seqlen_q"] = max_seqlen
    batch["max_seqlen_kv"] = max_seqlen
    # The compact input is shorter when image placeholders expand. MCore builds
    # seq_idx only after that expansion, so total_tokens is the merged width.
    batch["total_tokens"] = total_merged_length


def nemotron_omni_collate_fn(
    examples: list[Mapping[str, Any]],
    processor: Any,
    start_of_response_token: Any = None,
    *,
    visual_keys: object = None,
    min_pixels: int | None = None,
    max_pixels: int | None = None,
    enable_in_batch_packing: bool = False,
    sequence_length: int | None = None,
    pad_to_max_length: bool = False,
    pad_to_multiple_of: int = 128,
    in_batch_packing_pad_to_multiple_of: int = 1,
    max_audio_duration: float = 30.0,
    num_mel_bins: int = 128,
    temporal_patch_size: int = 2,
    video_fps: float = 1.0,
    video_nframes: int = 8,
    use_temporal_video_embedder: bool = False,
    patch_dim: int = 16,
    collapse_image_tokens: bool = False,
) -> dict[str, Any]:
    """Build one model-ready Omni batch from either HF or Energon examples.

    The canonical :class:`NemotronOmniModel` consumes the processor-expanded
    token sequence, with one image placeholder for every projected feature.
    Use :func:`nemotron_omni_llava_collate_fn` for the legacy LLaVA
    collapse/expand contract.
    """
    _validate_nemotron_omni_visual_keys(visual_keys)
    del start_of_response_token, min_pixels, max_pixels
    if not examples:
        raise ValueError("Nemotron Omni collation requires at least one example.")
    if enable_in_batch_packing and not collapse_image_tokens:
        raise ValueError(
            "Canonical Nemotron Omni owns sequence packing after media insertion; "
            "collator-side in-batch packing is supported only by the explicit "
            "nemotron_omni_llava_collate_fn compatibility path."
        )
    if hasattr(processor.image_processor, "max_num_tiles"):
        raise ValueError(
            "Nemotron Omni requires its dynamic-resolution image processor; "
            "legacy fixed-tile processors are not supported."
        )
    if (
        getattr(processor.tokenizer, "pad_token", None) is None
        and getattr(processor.tokenizer, "eos_token", None) is not None
    ):
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if use_temporal_video_embedder:
        batch, mask_examples, num_tiles = _prepare_temporal_rows(
            examples,
            processor,
            temporal_patch_size=temporal_patch_size,
            video_fps=video_fps,
            video_nframes=video_nframes,
            patch_dim=patch_dim,
        )
        use_per_image_token_counts = False
    else:
        video_parts = [
            item
            for example in examples
            for turn in example["conversation"]
            if isinstance(turn.get("content"), list)
            for item in turn["content"]
            if isinstance(item, Mapping) and item.get("type") == "video"
        ]
        if video_parts:
            raise ValueError("Nemotron Omni video collation requires use_temporal_video_embedder=True.")
        else:
            batch, mask_examples, num_tiles = _prepare_standard_rows(examples, processor)
            use_per_image_token_counts = num_tiles is not None

    _add_audio_inputs(
        batch,
        examples,
        processor,
        max_audio_duration=max_audio_duration,
        num_mel_bins=num_mel_bins,
    )

    skipped_tokens = extract_skipped_token_ids(processor)
    boundary_config = assistant_mask_boundary_config_from_markers(
        processor,
        assistant_start=CHATML_ASSISTANT_START,
        assistant_end=CHATML_ASSISTANT_END,
        assistant_end_fallbacks=("<|im_end|>",),
        role_start_markers=CHATML_OTHER_ROLE_STARTS,
    )
    loss_mask = torch.stack(
        [
            build_assistant_loss_mask(
                example,
                input_ids,
                processor,
                skipped_tokens,
                boundary_config=boundary_config,
            ).to(dtype=torch.int)
            for example, input_ids in zip(mask_examples, batch["input_ids"], strict=True)
        ]
    )
    if collapse_image_tokens:
        adjusted, loss_mask = _adjust_image_placeholders(batch, loss_mask, processor, num_tiles)
        batch["input_ids"] = adjusted["input_ids"]
        batch["attention_mask"] = adjusted["attention_mask"]

    if use_per_image_token_counts:
        _pack_dynamic_images(batch, patch_dim=patch_dim)
    elif "visual_inputs" not in batch:
        pixel_values = batch.pop("pixel_values", None)
        batch["visual_inputs"] = (
            GenericVisualInputs(pixel_values=pixel_values.to(torch.bfloat16)) if pixel_values is not None else None
        )

    has_modalities = batch.get("visual_inputs") is not None or batch.get("sound_clips") is not None
    post_merge_row_lengths = None
    if collapse_image_tokens and has_modalities and (sequence_length is not None or enable_in_batch_packing):
        post_merge_row_lengths = _model_merge_row_lengths(
            batch,
            processor,
            use_per_image_token_counts=use_per_image_token_counts,
            patch_dim=patch_dim,
        )

    batch_size, sequence_width = batch["input_ids"].shape
    batch["position_ids"] = torch.arange(sequence_width, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    labels = batch["input_ids"].clone()[:, 1:]
    labels = torch.cat((labels, torch.full_like(labels[:, :1], IGNORE_INDEX)), dim=1)
    labels[torch.isin(labels, skipped_tokens)] = IGNORE_INDEX
    shifted_loss_mask = torch.cat(
        (loss_mask.to(dtype=torch.float32)[:, 1:], torch.zeros_like(loss_mask[:, :1], dtype=torch.float32)), dim=1
    )
    batch["labels"] = labels.masked_fill(shifted_loss_mask == 0, IGNORE_INDEX)
    batch["loss_mask"] = shifted_loss_mask

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id or 0
    if enable_in_batch_packing:
        if post_merge_row_lengths is None:
            post_merge_row_lengths = batch["attention_mask"].to(dtype=torch.bool).sum(dim=1)
        _pack_omni_rows_to_mcore_thd(
            batch,
            post_merge_row_lengths,
            sequence_length=sequence_length,
            pad_to_max_length=pad_to_max_length,
            pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            pad_token_id=int(pad_token_id),
        )
    else:
        compact_sequence_length = sequence_length
        compact_pad_to_max_length = pad_to_max_length
        if post_merge_row_lengths is not None and sequence_length is not None:
            compact_sequence_length = _nonpacked_multimodal_compact_width(
                batch,
                post_merge_row_lengths,
                sequence_length=sequence_length,
                pad_to_max_length=pad_to_max_length,
                pad_to_multiple_of=pad_to_multiple_of,
            )
            compact_pad_to_max_length = True
        prepare_sequence_batch(
            batch,
            sequence_length=compact_sequence_length,
            pad_to_max_length=compact_pad_to_max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            enable_in_batch_packing=enable_in_batch_packing,
            in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
            pad_token_id=int(pad_token_id),
            ignore_index=IGNORE_INDEX,
        )
    return batch


def nemotron_omni_llava_collate_fn(*args, **kwargs) -> dict[str, torch.Tensor]:
    """Collate inputs for the explicit legacy LLaVA collapse/expand path."""

    kwargs["collapse_image_tokens"] = True
    return nemotron_omni_collate_fn(*args, **kwargs)


def nemotron_omni_expanded_collate_fn(*args, **kwargs) -> dict[str, Any]:
    """Collate processor-expanded inputs for the canonical Nemotron Omni model."""

    kwargs["collapse_image_tokens"] = False
    return nemotron_omni_collate_fn(*args, **kwargs)
