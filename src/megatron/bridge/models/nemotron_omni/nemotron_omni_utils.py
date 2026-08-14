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
    """Resize and normalize one frame for MCore's square temporal RADIO path.

    The public HF processor preserves aspect ratio, but pinned MCore requires
    temporal tubelets to share one square spatial grid. This helper is shared
    by training collation and inference so both paths use the same antialiased
    bicubic interpolation, RADIO normalization, and patch layout.

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
