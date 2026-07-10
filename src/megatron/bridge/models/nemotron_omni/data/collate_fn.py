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

import contextlib
import copy
import tempfile
from collections.abc import Iterator, Mapping, Sequence
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
from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


CHATML_ASSISTANT_START = "<|im_start|>assistant\n"
CHATML_ASSISTANT_END = "<|im_end|>\n"
CHATML_OTHER_ROLE_STARTS = {role: f"<|im_start|>{role}\n" for role in ("system", "developer", "user", "tool")}


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


@contextlib.contextmanager
def _single_tile_processor(processor: Any) -> Iterator[None]:
    """Temporarily force one processor tile per placeholder and always restore it."""
    image_processor = processor.image_processor
    tile_attr = None
    for candidate in ("max_num_tiles", "max_num_patches"):
        if hasattr(image_processor, candidate):
            tile_attr = candidate
            break
    if tile_attr is None:
        yield
        return
    original = getattr(image_processor, tile_attr)
    setattr(image_processor, tile_attr, 1)
    try:
        yield
    finally:
        setattr(image_processor, tile_attr, original)


def _pil_images(payload: Any) -> list[Any]:
    """Normalize one image/frame payload to a flat list of PIL images."""
    from megatron.bridge.data.energon.task_encoder_utils import _images_to_pil

    if payload is None:
        return []
    if isinstance(payload, torch.Tensor):
        converted = _images_to_pil(payload)
        return converted if isinstance(converted, list) else [converted]
    if isinstance(payload, (list, tuple)):
        result: list[Any] = []
        for item in payload:
            result.extend(_pil_images(item))
        return result
    return [payload]


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
    """Match the RADIO/CLIP frame normalization used by Omni inference."""
    from torchvision import transforms

    image = frame.resize((width, height))
    tensor = transforms.ToTensor()(image)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    patch_rows, patch_cols = height // patch_dim, width // patch_dim
    return (
        tensor.reshape(3, patch_rows, patch_dim, patch_cols, patch_dim)
        .permute(1, 3, 0, 2, 4)
        .reshape(patch_rows * patch_cols, 3 * patch_dim * patch_dim)
    )


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
    frame_height = frame_width = 512
    token_rows: list[torch.Tensor] = []
    mask_examples: list[dict[str, Any]] = []
    all_patches: list[torch.Tensor] = []
    all_sizes: list[list[int]] = []
    all_num_frames: list[int] = []
    all_num_image_tiles: list[int] = []
    placeholder_counts: list[int] = []

    for example in examples:
        conversation: list[dict[str, Any]] = []
        representative_images: list[Any] = []
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
                        text_parts.append("<image>")
                        representative_images.append(image)
                        all_patches.append(
                            _patchify_frame(image, height=frame_height, width=frame_width, patch_dim=patch_dim)
                        )
                        all_sizes.append([frame_height, frame_width])
                        all_num_frames.append(1)
                        all_num_image_tiles.append((frame_height // patch_dim) * (frame_width // patch_dim) // 4)
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
                                f"frame {frame_start + offset + 1} sampled at "
                                f"{(frame_start + offset) / sampled_fps:.2f} seconds"
                                for offset in range(len(group))
                            ]
                            video_lines.append(" and ".join(timestamps) + ": <image>")
                            representative_images.append(group[0])
                            row_placeholder_count += 1
                        text_parts.append("\n".join(video_lines))
                        all_patches.extend(
                            _patchify_frame(frame, height=frame_height, width=frame_width, patch_dim=patch_dim)
                            for frame in frames
                        )
                        all_sizes.extend([[frame_height, frame_width]] * len(frames))
                        all_num_frames.append(len(frames))
                        tiles_per_frame = (frame_height // patch_dim) * (frame_width // patch_dim) // 4
                        all_num_image_tiles.extend([tiles_per_frame] * len(frames))
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
        with _single_tile_processor(processor), use_processor_right_padding(processor):
            output = processor(
                text=[prompt],
                images=representative_images or None,
                padding=False,
                truncation=False,
                return_tensors="pt",
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
    batch: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention_mask}
    if all_patches:
        batch["visual_inputs"] = GenericVisualInputs(
            pixel_values=torch.cat(all_patches, dim=0).unsqueeze(0).to(torch.bfloat16).contiguous()
        )
        batch["imgs_sizes"] = torch.tensor(all_sizes, dtype=torch.long)
        batch["num_frames"] = torch.tensor(all_num_frames, dtype=torch.long)
        batch["num_image_tiles"] = torch.tensor(all_num_image_tiles, dtype=torch.int)
    else:
        batch["visual_inputs"] = None
    return (
        batch,
        mask_examples,
        torch.tensor([1 for count in placeholder_counts for _ in range(count)], dtype=torch.long),
    )


def _prepare_standard_rows(
    examples: Sequence[Mapping[str, Any]],
    processor: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor | None, bool]:
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
    is_dynamic_resolution = not hasattr(processor.image_processor, "max_num_tiles")

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
        return batch, mask_examples, None, is_dynamic_resolution

    if is_dynamic_resolution:
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
        return batch, mask_examples, num_tiles, is_dynamic_resolution

    with _single_tile_processor(processor), use_processor_right_padding(processor):
        batch = processor(
            text=prompts,
            images=all_images,
            padding=processor.tokenizer.pad_token is not None,
            truncation=False,
            return_tensors="pt",
        )
    num_tiles = batch.get("num_patches")
    if num_tiles is None:
        raise ValueError("The static Nemotron Omni image processor must provide num_patches.")
    return dict(batch), mask_examples, torch.as_tensor(num_tiles), is_dynamic_resolution


def _prepare_static_video_row(
    examples: Sequence[Mapping[str, Any]],
    processor: Any,
    video_part: Mapping[str, Any],
    *,
    video_fps: float,
    video_nframes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], torch.Tensor | None]:
    """Preserve the legacy single-row static-video processor path."""
    if len(examples) != 1:
        raise ValueError("Static Nemotron Omni video collation supports batch size 1 only.")
    if not hasattr(processor.image_processor, "max_num_tiles"):
        raise ValueError(
            "The dynamic-resolution Nemotron Omni processor does not support HF video collation; "
            "use the temporal-video recipe."
        )
    payload = video_part.get("video", video_part.get("path"))
    if not isinstance(payload, str):
        raise ValueError("Static Nemotron Omni video collation requires a video path.")

    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    image_urls, metadata = maybe_path_or_url_to_data_urls(
        payload,
        fps=max(0, int(video_fps)),
        nframe=max(0, int(video_nframes)),
        nframe_max=-1,
    )
    frames = [[pil_image_from_base64(image_url) for image_url in image_urls]]
    prompt = processor.apply_chat_template(
        [examples[0]["conversation"]],
        tokenize=False,
        **chat_template_kwargs_from_example(examples[0]),
    )
    with use_processor_right_padding(processor):
        batch = dict(
            processor(
                text=prompt,
                videos=frames,
                videos_kwargs={"video_metadata": metadata},
                return_tensors="pt",
            )
        )
    batch.setdefault("attention_mask", torch.ones_like(batch["input_ids"], dtype=torch.long))
    pixel_values = batch.pop("pixel_values_videos", None)
    if pixel_values is not None:
        batch["pixel_values"] = pixel_values
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<video>")
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    batch["input_ids"] = torch.where(batch["input_ids"] == video_token_id, image_token_id, batch["input_ids"])
    num_tiles = batch.get("num_patches")
    return batch, [dict(examples[0])], torch.as_tensor(num_tiles) if num_tiles is not None else None


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
            sound_tokens = torch.full((token_count,), sound_token_id, dtype=row.dtype)
            row = torch.cat((row[:first_position], sound_tokens, row[last_position:]))
        else:
            image_end_positions = torch.where(row == image_end_id)[0]
            insertion = int(image_end_positions[-1].item()) + 1 if image_end_positions.numel() else min(1, row.numel())
            sound_block = torch.tensor(
                [sound_start_id, *([sound_token_id] * token_count), sound_end_id],
                dtype=row.dtype,
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
        if height % patch_dim or width % patch_dim:
            raise ValueError(f"Image {height}x{width} is not divisible by patch_dim={patch_dim}.")
        patch_rows, patch_cols = height // patch_dim, width // patch_dim
        patches.append(
            image.reshape(channels, patch_rows, patch_dim, patch_cols, patch_dim)
            .permute(1, 3, 0, 2, 4)
            .reshape(patch_rows * patch_cols, channels * patch_dim * patch_dim)
            .contiguous()
        )
        sizes.append([height, width])
        tile_counts.append((patch_rows * patch_cols) // 4)
    batch["visual_inputs"] = GenericVisualInputs(pixel_values=torch.cat(patches).unsqueeze(0).contiguous())
    batch["imgs_sizes"] = torch.tensor(sizes, dtype=torch.long)
    batch["num_frames"] = torch.ones(len(images), dtype=torch.long)
    batch["num_image_tiles"] = torch.tensor(tile_counts, dtype=torch.int)


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
) -> dict[str, Any]:
    """Build one model-ready Omni batch from either HF or Energon examples."""
    del start_of_response_token, visual_keys, min_pixels, max_pixels
    if not examples:
        raise ValueError("Nemotron Omni collation requires at least one example.")
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
        is_dynamic_resolution = False
    else:
        video_parts = [
            item
            for example in examples
            for turn in example["conversation"]
            if isinstance(turn.get("content"), list)
            for item in turn["content"]
            if isinstance(item, Mapping) and item.get("type") == "video"
        ]
        if len(video_parts) > 1:
            raise ValueError("Static Nemotron Omni video collation supports exactly one video per batch.")
        if video_parts:
            batch, mask_examples, num_tiles = _prepare_static_video_row(
                examples,
                processor,
                video_parts[0],
                video_fps=video_fps,
                video_nframes=video_nframes,
            )
            is_dynamic_resolution = False
        else:
            batch, mask_examples, num_tiles, is_dynamic_resolution = _prepare_standard_rows(examples, processor)

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
    adjusted, loss_mask = _adjust_image_placeholders(batch, loss_mask, processor, num_tiles)
    batch["input_ids"] = adjusted["input_ids"]
    batch["attention_mask"] = adjusted["attention_mask"]

    if is_dynamic_resolution:
        _pack_dynamic_images(batch, patch_dim=patch_dim)
    elif "visual_inputs" not in batch:
        pixel_values = batch.pop("pixel_values", None)
        batch["visual_inputs"] = (
            GenericVisualInputs(pixel_values=pixel_values.to(torch.bfloat16)) if pixel_values is not None else None
        )

    if enable_in_batch_packing and (batch.get("visual_inputs") is not None or batch.get("sound_clips") is not None):
        raise ValueError(
            "Nemotron Omni in-batch packing does not support image, video, or audio samples because "
            "modality embeddings are merged after packed-sequence boundaries are built."
        )

    has_modalities = batch.get("visual_inputs") is not None or batch.get("sound_clips") is not None
    if sequence_length is not None and has_modalities:
        row_lengths = batch["attention_mask"].to(dtype=torch.bool).sum(dim=1)
        if bool((row_lengths > sequence_length).any()):
            raise ValueError(
                "Nemotron Omni cannot truncate image, video, or audio rows because modality metadata would no "
                f"longer align with text placeholders; got row lengths {row_lengths.tolist()} with "
                f"sequence_length={sequence_length}."
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
    prepare_sequence_batch(
        batch,
        sequence_length=sequence_length,
        pad_to_max_length=pad_to_max_length,
        pad_to_multiple_of=pad_to_multiple_of,
        enable_in_batch_packing=enable_in_batch_packing,
        in_batch_packing_pad_to_multiple_of=in_batch_packing_pad_to_multiple_of,
        pad_token_id=int(pad_token_id),
        ignore_index=IGNORE_INDEX,
    )
    return batch
