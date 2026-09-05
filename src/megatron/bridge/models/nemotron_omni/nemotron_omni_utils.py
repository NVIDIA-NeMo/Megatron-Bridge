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

import math
import warnings
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, TypeVar, Union

import numpy as np
import torch


_FrameT = TypeVar("_FrameT")
COMPACT_IMAGE_PLACEHOLDER = "<img><image></img>"


def patchify_temporal_frame(frame: Any, *, height: int, width: int, patch_dim: int) -> torch.Tensor:
    """Resize and normalize one frame for the fixed temporal RADIO policy.

    This compatibility helper intentionally places every frame on the supplied
    canvas. It is shared by fixed-policy training collation and inference so
    both paths use the same antialiased bicubic interpolation, RADIO
    normalization, and patch layout.

    Args:
        frame: PIL-compatible image with ``convert("RGB")`` support.
        height: Compatibility-canvas height.
        width: Compatibility-canvas width.
        patch_dim: Vision patch edge length.

    Returns:
        A tensor with shape ``[num_patches, 3 * patch_dim * patch_dim]``.
    """
    if patch_dim < 1 or height % patch_dim or width % patch_dim:
        raise ValueError(f"Image {height}x{width} is not divisible by patch_dim={patch_dim}.")
    image = np.asarray(frame.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
    if tensor.shape[-2:] != (height, width):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(height, width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    tensor = tensor.squeeze(0) / 255.0
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    patch_rows, patch_cols = height // patch_dim, width // patch_dim
    return (
        tensor.reshape(3, patch_rows, patch_dim, patch_cols, patch_dim)
        .permute(1, 3, 0, 2, 4)
        .reshape(patch_rows * patch_cols, 3 * patch_dim * patch_dim)
    )


def temporal_model_frames(frames: Sequence[_FrameT], temporal_patch_size: int) -> list[_FrameT]:
    """Return frames passed to MCore for one temporal-video sample.

    MCore pads incomplete groups with the final frame, so odd multi-frame
    samples retain their original metadata. A single frame needs explicit
    repetition to select the temporal embedder instead of the image embedder.

    Args:
        frames: Sampled frames in prompt order.
        temporal_patch_size: Number of frames fused into one temporal tubelet.

    Returns:
        Frames for model patchification and ``num_frames`` metadata.
    """
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be greater than 0.")
    model_frames = list(frames)
    if len(model_frames) == 1 and temporal_patch_size > 1:
        model_frames *= temporal_patch_size
    return model_frames


def temporal_video_frame_labels(
    num_frames: int,
    *,
    temporal_patch_size: int,
    source_fps: float | None,
    frame_indices: Sequence[int] | None,
) -> list[str]:
    """Build one source-timestamped label per temporal tubelet.

    Args:
        num_frames: Number of sampled frames represented in the prompt.
        temporal_patch_size: Number of consecutive frames per tubelet.
        source_fps: Source-video frame rate, when available.
        frame_indices: Source-frame index for every sampled frame, when available.

    Returns:
        Frame-label prefixes aligned one-to-one with temporal tubelets.
    """
    if num_frames < 0:
        raise ValueError("num_frames must be non-negative.")
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be greater than 0.")

    fps = float(source_fps or 0)
    frame_duration_ms = int(1000.0 / fps) if fps > 0 else None
    labels: list[str] = []
    for frame_start in range(0, num_frames, temporal_patch_size):
        parts: list[str] = []
        for offset in range(min(temporal_patch_size, num_frames - frame_start)):
            frame_position = frame_start + offset
            prefix = "Frame" if offset == 0 else "frame"
            if frame_duration_ms is not None and frame_indices is not None and frame_position < len(frame_indices):
                timestamp = int(frame_indices[frame_position]) * frame_duration_ms / 1000.0
                parts.append(f"{prefix} {frame_position + 1} sampled at {timestamp:.2f} seconds")
            elif fps > 0:
                timestamp = frame_position / fps
                parts.append(f"{prefix} {frame_position + 1} sampled at {timestamp:.2f} seconds")
            else:
                parts.append(f"{prefix} {frame_position + 1}")
        labels.append(" and ".join(parts) + ": ")
    return labels


def processor_patchify_temporal_frames(
    frames: Sequence[Any],
    *,
    image_processor: Any,
    patch_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the public processor's video resize policy and pack normalized frames.

    The Nemotron Omni remote image processor selects a patch-grid size using a
    separate aspect-preserving video policy. It exposes that policy through the
    same temporary ``_is_video_mode`` switch used by the public multimodal
    processor. Reusing the processor output here keeps Bridge pixels, grid
    metadata, and placeholder counts aligned with HF/vLLM preprocessing.

    Args:
        frames: Frames belonging to one source video, in model order.
        image_processor: Nemotron Omni dynamic-resolution image processor.
        patch_dim: Vision patch edge length.

    Returns:
        A pair containing packed patches with shape
        ``[1, total_patches, 3 * patch_dim**2]`` and one ``(height, width)``
        metadata row per frame.

    Raises:
        ValueError: If the processor contract or returned frame metadata is
            incompatible with RADIO's packed temporal input.
    """
    if not frames:
        raise ValueError("Processor-driven temporal preprocessing requires at least one frame.")
    if patch_dim <= 0:
        raise ValueError("patch_dim must be greater than 0.")
    if not hasattr(image_processor, "_is_video_mode"):
        raise ValueError(
            "Processor-driven temporal preprocessing requires a Nemotron Omni image processor "
            "with the public '_is_video_mode' video contract."
        )

    previous_video_mode = image_processor._is_video_mode
    image_processor._is_video_mode = True
    try:
        processed = image_processor(images=list(frames), return_tensors=None)
    finally:
        image_processor._is_video_mode = previous_video_mode

    pixel_values = processed.get("pixel_values")
    if isinstance(pixel_values, list):
        frame_tensors = [torch.as_tensor(value) for value in pixel_values]
    elif torch.is_tensor(pixel_values) and pixel_values.ndim == 4:
        frame_tensors = list(pixel_values.unbind(0))
    elif torch.is_tensor(pixel_values) and pixel_values.ndim == 3:
        frame_tensors = [pixel_values]
    else:
        shape = getattr(pixel_values, "shape", None)
        raise ValueError(f"Video image processor returned unsupported pixel_values shape {shape}.")

    raw_imgs_sizes = processed.get("imgs_sizes")
    if raw_imgs_sizes is None:
        raise ValueError("Video image processor must return imgs_sizes metadata.")
    imgs_sizes = torch.as_tensor(raw_imgs_sizes, dtype=torch.long)
    if imgs_sizes.ndim != 2 or imgs_sizes.shape != (len(frame_tensors), 2):
        raise ValueError(
            "Video image processor must return one (height, width) row per frame; "
            f"got {tuple(imgs_sizes.shape)} for {len(frame_tensors)} frames."
        )
    if len(frame_tensors) != len(frames):
        raise ValueError(f"Video image processor returned {len(frame_tensors)} tensors for {len(frames)} frames.")

    reported_tokens = processed.get("num_tokens")
    if reported_tokens is None or len(reported_tokens) != len(frame_tensors):
        count = None if reported_tokens is None else len(reported_tokens)
        raise ValueError(
            "Video image processor must return one num_tokens entry per frame; "
            f"got {count} entries for {len(frame_tensors)} frames."
        )

    patches = []
    for frame_index, (frame, size, reported_count) in enumerate(
        zip(frame_tensors, imgs_sizes.tolist(), reported_tokens, strict=True)
    ):
        if frame.ndim != 3:
            raise ValueError(f"Processed video frame {frame_index} must have shape [3,H,W], got {tuple(frame.shape)}.")
        channels, height, width = frame.shape
        expected_height, expected_width = (int(value) for value in size)
        if channels != 3 or (height, width) != (expected_height, expected_width):
            raise ValueError(
                f"Processed video frame {frame_index} shape {tuple(frame.shape)} does not match "
                f"imgs_sizes row {(expected_height, expected_width)}."
            )
        if height % patch_dim or width % patch_dim:
            raise ValueError(f"Video frame {height}x{width} is not divisible by patch_dim={patch_dim}.")
        patch_rows, patch_cols = height // patch_dim, width // patch_dim
        if patch_rows % 2 or patch_cols % 2:
            raise ValueError(f"Video patch grid {patch_rows}x{patch_cols} is not divisible by the 2x2 pixel shuffle.")
        expected_count = (patch_rows * patch_cols) // 4
        if int(reported_count) != expected_count:
            raise ValueError(
                f"Video image processor reported {int(reported_count)} tokens for frame {frame_index}, "
                f"but grid {patch_rows}x{patch_cols} produces {expected_count}."
            )
        patches.append(
            frame.reshape(channels, patch_rows, patch_dim, patch_cols, patch_dim)
            .permute(1, 3, 0, 2, 4)
            .reshape(patch_rows * patch_cols, channels * patch_dim * patch_dim)
            .contiguous()
        )

    return torch.cat(patches, dim=0).unsqueeze(0).contiguous(), imgs_sizes


def temporal_tubelet_feature_counts(
    imgs_sizes: torch.Tensor,
    num_frames: torch.Tensor,
    *,
    temporal_patch_size: int,
    patch_dim: int,
    pixel_shuffle_factor: int = 2,
) -> torch.Tensor:
    """Compute projected RADIO feature counts in temporal tubelet order.

    Args:
        imgs_sizes: One ``(height, width)`` row per ungrouped input frame.
        num_frames: Number of frame rows owned by each image or video item.
        temporal_patch_size: Frames fused into one video tubelet.
        patch_dim: Vision patch edge length.
        pixel_shuffle_factor: Spatial reduction factor per dimension.

    Returns:
        One feature count per image or temporal tubelet, aligned with compact
        ``<img><image></img>`` wrappers.

    Raises:
        ValueError: If metadata is inconsistent, a patch grid cannot be
            shuffled, or frames inside one tubelet use different grids.
    """
    if imgs_sizes.ndim != 2 or imgs_sizes.shape[1] != 2:
        raise ValueError(f"imgs_sizes must have shape [N, 2], got {tuple(imgs_sizes.shape)}.")
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be greater than 0.")
    if patch_dim <= 0:
        raise ValueError("patch_dim must be greater than 0.")
    if pixel_shuffle_factor <= 0:
        raise ValueError("pixel_shuffle_factor must be greater than 0.")

    frame_counts = [int(count) for count in num_frames.reshape(-1).tolist()]
    if not frame_counts or any(count <= 0 for count in frame_counts):
        raise ValueError("num_frames must contain positive entries.")
    if sum(frame_counts) != imgs_sizes.shape[0]:
        raise ValueError(
            f"num_frames accounts for {sum(frame_counts)} frames but imgs_sizes has {imgs_sizes.shape[0]} rows."
        )

    sizes = [(int(height), int(width)) for height, width in imgs_sizes.tolist()]
    feature_counts = []
    frame_offset = 0
    for media_index, frame_count in enumerate(frame_counts):
        group_width = 1 if frame_count == 1 else temporal_patch_size
        for group_start in range(0, frame_count, group_width):
            group = sizes[frame_offset + group_start : frame_offset + min(group_start + group_width, frame_count)]
            if any(size != group[0] for size in group[1:]):
                raise ValueError(
                    f"Temporal tubelet {len(feature_counts)} in media item {media_index} has inconsistent "
                    f"frame sizes {group}."
                )
            height, width = group[0]
            if height <= 0 or width <= 0 or height % patch_dim or width % patch_dim:
                raise ValueError(f"Frame size {height}x{width} is not divisible by patch_dim={patch_dim}.")
            patch_rows, patch_cols = height // patch_dim, width // patch_dim
            if patch_rows % pixel_shuffle_factor or patch_cols % pixel_shuffle_factor:
                raise ValueError(
                    f"Patch grid {patch_rows}x{patch_cols} is not divisible by the "
                    f"{pixel_shuffle_factor}x{pixel_shuffle_factor} pixel shuffle."
                )
            feature_counts.append((patch_rows * patch_cols) // (pixel_shuffle_factor**2))
        frame_offset += frame_count

    return torch.tensor(feature_counts, dtype=torch.int, device=imgs_sizes.device)


def inference_num_image_tiles(
    imgs_sizes: torch.Tensor,
    *,
    patch_dim: int,
    pixel_shuffle_factor: int = 2,
    num_frames: torch.Tensor | None = None,
    temporal_patch_size: int = 1,
) -> torch.Tensor:
    """Build image-placeholder replacement counts for pipeline inference.

    Dynamic images contribute their post-pixel-shuffle feature count per tile.
    Temporal tubelets contribute one logical count each; canonical inference
    applies the fixed tubelet width through
    :func:`inference_expanded_image_token_counts`. The deprecated LLaVA path
    instead applies that width inside the model.

    Args:
        imgs_sizes: Per-image or per-frame ``(height, width)`` metadata.
        patch_dim: Vision patch edge length.
        pixel_shuffle_factor: Spatial downsampling factor per dimension.
        num_frames: Frame counts per temporal video, or ``None`` for images.
        temporal_patch_size: Frames fused into one temporal tubelet.

    Returns:
        One integer replacement count per compact image placeholder.
    """
    if patch_dim <= 0:
        raise ValueError("patch_dim must be greater than 0.")
    if pixel_shuffle_factor <= 0:
        raise ValueError("pixel_shuffle_factor must be greater than 0.")
    if temporal_patch_size <= 0:
        raise ValueError("temporal_patch_size must be greater than 0.")
    if imgs_sizes.ndim != 2 or imgs_sizes.shape[1] != 2:
        raise ValueError(f"imgs_sizes must have shape [N, 2], got {tuple(imgs_sizes.shape)}.")

    if num_frames is not None:
        frame_counts = num_frames.reshape(-1).tolist()
        if any(int(count) <= 0 for count in frame_counts):
            raise ValueError("num_frames entries must be greater than 0.")
        if sum(int(count) for count in frame_counts) != imgs_sizes.shape[0]:
            raise ValueError("num_frames must account for every row in imgs_sizes.")
        num_tubelets = sum(math.ceil(int(count) / temporal_patch_size) for count in frame_counts)
        return torch.ones(num_tubelets, dtype=torch.int, device=imgs_sizes.device)

    grid_sizes = torch.div(imgs_sizes, patch_dim, rounding_mode="floor")
    if torch.any(grid_sizes * patch_dim != imgs_sizes):
        raise ValueError("Image dimensions must be divisible by patch_dim.")
    if torch.any(grid_sizes % pixel_shuffle_factor != 0):
        raise ValueError("Image patch grids must be divisible by pixel_shuffle_factor.")
    return (grid_sizes.prod(dim=1) // (pixel_shuffle_factor**2)).to(dtype=torch.int)


def inference_expanded_image_token_counts(
    tile_feature_counts: torch.Tensor,
    tiles_per_media: int | Sequence[int] | torch.Tensor,
    *,
    feature_multiplier: int = 1,
) -> torch.Tensor:
    """Aggregate projected feature counts for canonical inference prompts.

    Dynamic image processors can split one source image into multiple RADIO
    tiles, while each ``<img>...</img>`` region belongs to the source image.
    The canonical model needs one ``<image>`` placeholder per projected
    feature, so per-tile counts must be summed back to one count per region.
    Temporal inference uses one logical tile per tubelet and a fixed feature
    multiplier for the post-pixel-shuffle tubelet width.

    Args:
        tile_feature_counts: Number of projected feature rows produced by each
            RADIO tile or temporal tubelet.
        tiles_per_media: Number of entries in ``tile_feature_counts`` owned by
            each ``<img>...</img>`` region.
        feature_multiplier: Additional projected width per count. Use one for
            dynamic images and the tubelet feature width for temporal video.

    Returns:
        One expanded placeholder count per ``<img>...</img>`` region.

    Raises:
        ValueError: If counts are non-positive or do not account for every
            tile/tubelet.
    """
    flat_feature_counts = tile_feature_counts.reshape(-1)
    if torch.any(flat_feature_counts <= 0):
        raise ValueError("tile_feature_counts entries must be greater than 0.")
    if feature_multiplier <= 0:
        raise ValueError("feature_multiplier must be greater than 0.")

    if isinstance(tiles_per_media, int):
        media_tile_counts = [tiles_per_media]
    elif isinstance(tiles_per_media, torch.Tensor):
        media_tile_counts = [int(count) for count in tiles_per_media.detach().cpu().reshape(-1).tolist()]
    else:
        media_tile_counts = [int(count) for count in tiles_per_media]
    if not media_tile_counts or any(count <= 0 for count in media_tile_counts):
        raise ValueError("tiles_per_media entries must be greater than 0.")
    if sum(media_tile_counts) != flat_feature_counts.numel():
        raise ValueError(
            "tiles_per_media must account for every tile feature count; "
            f"got {sum(media_tile_counts)} tiles for {flat_feature_counts.numel()} counts."
        )

    offset = 0
    expanded_counts = []
    for tile_count in media_tile_counts:
        expanded_counts.append(
            int(flat_feature_counts[offset : offset + tile_count].sum().item()) * feature_multiplier
        )
        offset += tile_count
    return torch.tensor(expanded_counts, dtype=torch.int, device=tile_feature_counts.device)


def inference_merged_sequence_length(
    input_ids: torch.Tensor,
    *,
    image_token_index: int,
    num_image_tiles: torch.Tensor | None,
    image_seq_len: int,
) -> int:
    """Return the legacy unpadded length after model-owned vision expansion.

    This helper is deprecated because the canonical model consumes an already
    expanded sequence; its merged length is simply ``input_ids.shape[1]``.

    Args:
        input_ids: One inference prompt row, including generated tokens so far.
        image_token_index: Token ID replaced by vision embeddings.
        num_image_tiles: Row-major replacement metadata per image placeholder.
        image_seq_len: Embeddings contributed by each tile.

    Returns:
        The real merged sequence length before pipeline padding.
    """
    warnings.warn(
        "inference_merged_sequence_length is deprecated with the Nemotron Omni LLaVA collapse/expand path; "
        "canonical expanded-sequence inference uses input_ids.shape[1].",
        FutureWarning,
        stacklevel=2,
    )
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, S], got {tuple(input_ids.shape)}.")
    if image_seq_len <= 0:
        raise ValueError("image_seq_len must be greater than 0.")
    num_placeholders = int((input_ids == image_token_index).sum().item())
    if num_placeholders == 0:
        if num_image_tiles is not None and num_image_tiles.numel() != 0:
            raise ValueError("num_image_tiles must be empty when input_ids has no image placeholders.")
        return input_ids.shape[1]
    if num_image_tiles is None or num_image_tiles.numel() != num_placeholders:
        count = None if num_image_tiles is None else num_image_tiles.numel()
        raise ValueError(f"Expected {num_placeholders} num_image_tiles entries, got {count}.")
    replacement_length = int(num_image_tiles.sum().item()) * image_seq_len
    return input_ids.shape[1] - num_placeholders + replacement_length


def select_inference_next_token(logits: torch.Tensor, merged_sequence_length: int) -> torch.Tensor:
    """Select the next token from the last real position, excluding PP padding."""
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [B, S, V], got {tuple(logits.shape)}.")
    if merged_sequence_length <= 0 or merged_sequence_length > logits.shape[1]:
        raise ValueError(f"Merged sequence length {merged_sequence_length} is outside logits width {logits.shape[1]}.")
    return torch.argmax(logits[:, merged_sequence_length - 1], dim=-1, keepdim=True)


def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load an audio file and resample to ``target_sr`` Hz.

    Supports WAV, MP3, FLAC, and other formats handled by *soundfile*
    (with *librosa* as a fallback for MP3 and other FFmpeg-decoded formats).

    Args:
        path: Path to the audio file.
        target_sr: Target sampling rate in Hz.

    Returns:
        1-D float32 numpy array of the mono waveform at ``target_sr``.
    """
    try:
        import soundfile as sf

        waveform, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        import librosa

        waveform, sr = librosa.load(path, sr=None, mono=True)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=-1)

    if sr != target_sr:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    return waveform.astype(np.float32)


@lru_cache(maxsize=None)
def _parakeet_feature_extractor(num_mel_bins: int, sampling_rate: int) -> Any:
    """Construct one reusable feature extractor per audio configuration."""
    from transformers import ParakeetFeatureExtractor

    return ParakeetFeatureExtractor(
        feature_size=num_mel_bins,
        sampling_rate=sampling_rate,
    )


def valid_audio_feature_lengths(attention_mask: torch.Tensor, *, num_frames: int) -> torch.Tensor:
    """Convert Parakeet feature masks to contiguous-prefix frame lengths.

    Parakeet retains a padded boundary frame in ``input_features`` while its
    feature-level attention mask records the semantic frame count. Bridge's
    sound encoder accepts that mask in compressed form as ``sound_length``.

    Args:
        attention_mask: Binary mask with shape ``(batch, frames)``.
        num_frames: Physical frame width of the corresponding feature tensor.

    Returns:
        Long tensor containing one valid frame length per batch item.

    Raises:
        ValueError: If the mask is empty, non-binary, has the wrong shape, or
            is not a non-empty contiguous prefix for every batch item.
    """
    mask = torch.as_tensor(attention_mask)
    if mask.ndim != 2:
        raise ValueError(f"Parakeet feature attention_mask must be 2-D, got shape {tuple(mask.shape)}.")
    if num_frames < 1 or mask.shape[1] != num_frames:
        raise ValueError(
            "Parakeet feature attention_mask width must match the physical feature width; "
            f"got mask shape {tuple(mask.shape)} and num_frames={num_frames}."
        )
    if not bool(torch.all((mask == 0) | (mask == 1))):
        raise ValueError("Parakeet feature attention_mask must contain only binary values.")

    prefix_mask = mask.to(dtype=torch.bool)
    lengths = prefix_mask.sum(dim=1, dtype=torch.long)
    if bool(torch.any(lengths == 0)):
        raise ValueError("Parakeet feature attention_mask must contain at least one valid frame per sample.")
    expected_mask = torch.arange(num_frames, device=mask.device)[None, :] < lengths[:, None]
    if not torch.equal(prefix_mask, expected_mask):
        raise ValueError("Parakeet feature attention_mask must contain one contiguous valid prefix per sample.")
    return lengths


def compute_mel_features_with_length(
    waveform: Union[np.ndarray, list],
    sampling_rate: int = 16000,
    num_mel_bins: int = 128,
) -> tuple[torch.Tensor, int]:
    """Convert one waveform to physical mel features and its valid frame length.

    Args:
        waveform: 1-D float32 numpy array (or list) of the mono waveform.
        sampling_rate: Sampling rate of *waveform* (must match the extractor).
        num_mel_bins: Number of mel frequency bins.

    Returns:
        A ``(mel, valid_length)`` pair. ``mel`` has physical shape
        ``(frames, num_mel_bins)`` and may include padded boundary rows;
        ``valid_length`` is derived from Parakeet's feature attention mask.
    """
    extractor = _parakeet_feature_extractor(num_mel_bins, sampling_rate)
    features = extractor(
        waveform,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    input_features = torch.as_tensor(features["input_features"])
    if input_features.ndim != 3 or input_features.shape[0] != 1:
        raise ValueError(
            "compute_mel_features_with_length expects one waveform and Parakeet features with shape "
            f"(1, frames, mel_bins), got {tuple(input_features.shape)}."
        )
    mel = input_features.squeeze(0)
    lengths = valid_audio_feature_lengths(features["attention_mask"], num_frames=mel.shape[0])
    return mel, int(lengths[0].item())


def compute_mel_features(
    waveform: Union[np.ndarray, list],
    sampling_rate: int = 16000,
    num_mel_bins: int = 128,
) -> torch.Tensor:
    """Convert a raw waveform to a mel spectrogram tensor.

    Uses HF ``ParakeetFeatureExtractor`` (from ``transformers``) to produce
    mel features compatible with ``BridgeSoundEncoder`` / ``ParakeetEncoder``.

    Args:
        waveform: 1-D float32 numpy array (or list) of the mono waveform.
        sampling_rate: Sampling rate of *waveform* (must match the extractor).
        num_mel_bins: Number of mel frequency bins.

    Returns:
        Float tensor of shape ``(frames, num_mel_bins)`` -- a single clip
        ready to be batched and passed as ``sound_clips`` to the model.

        Call :func:`compute_mel_features_with_length` when the corresponding
        semantic frame length is also required.
    """
    mel, _ = compute_mel_features_with_length(
        waveform,
        sampling_rate=sampling_rate,
        num_mel_bins=num_mel_bins,
    )
    return mel


def compute_audio_token_count(
    waveform: Union[np.ndarray, list],
    hop_length: int = 160,
    subsampling_factor: int = 8,
) -> int:
    """Compute the expected number of audio tokens for a waveform.

    Uses the same Conv2D subsampling math as ``ParakeetEncoder`` /
    ``ParakeetEncoderSubsamplingConv2D``: kernel_size=3, stride=2, padding=1,
    applied log2(subsampling_factor) times to the mel frame count.

    Args:
        waveform: 1-D waveform array (only its length is used).
        hop_length: Hop length in samples for mel feature extraction.
        subsampling_factor: Subsampling factor of the conformer encoder.

    Returns:
        Number of audio tokens (at least 1).
    """
    num_frames = len(waveform) // hop_length
    # Match BridgeSoundEncoder._compute_output_lengths exactly:
    # Conv2D subsampling with kernel=3, stride=2, padding=1, ceil_mode=False
    length = float(num_frames)
    num_layers = int(math.log2(subsampling_factor))
    kernel_size = 3
    stride = 2
    padding = (kernel_size - 1) // 2
    all_paddings = padding * 2
    for _ in range(num_layers):
        length = math.floor((length + all_paddings - kernel_size) / stride + 1)
    return max(1, int(length))
