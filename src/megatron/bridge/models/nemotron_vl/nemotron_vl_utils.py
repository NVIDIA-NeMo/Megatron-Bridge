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

import base64
import io
import logging
import mimetypes
import os
from collections.abc import Mapping

import torch
from PIL import Image
from transformers.video_utils import VideoMetadata

from megatron.bridge.utils.safe_url import (
    ALLOW_PRIVATE_URL_FETCH_ENV,
    is_safe_public_http_url,
    safe_url_open,
)


logger = logging.getLogger(__name__)

# Backward-compatible private aliases so existing callers and tests keep working.
_ALLOW_PRIVATE_URL_FETCH_ENV = ALLOW_PRIVATE_URL_FETCH_ENV
_is_safe_public_http_url = is_safe_public_http_url
_safe_url_open = safe_url_open


def encode_pil_to_jpeg_data_url(pil_image):
    """Encode a PIL image to a base64-encoded data URL."""
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def sample_video_frames_to_data_urls(video_path_local, fps=1, nframe=0, nframe_max=-1):
    """
    Sample frames from a video and return base64-encoded data URLs along with metadata.

    Args:
        video_path_local: Path to the video file
        fps: Target frames per second for sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample (used if fps <= 0)
        nframe_max: Maximum number of frames to sample

    Returns:
        tuple: (frame_data_urls, metadata)
        - frame_data_urls: List of base64-encoded frame images
        - metadata: VideoMetadata dataclass containing info about the sampled frames:
            - total_num_frames: Number of sampled frames
            - fps: Effective frame rate of the sampled frames
            - duration: Duration covered by the sampled frames (in seconds)
            - video_backend: Backend used for video processing ('decord')
    """
    import decord
    import numpy as np
    from PIL import Image

    vid = decord.VideoReader(video_path_local)
    total_frames = len(vid)
    video_fps = vid.get_avg_fps()
    total_duration = total_frames / max(1e-6, video_fps)

    if fps > 0:
        required_frames = int(total_duration * fps)
        desired_frames = max(1, required_frames)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))
    else:
        desired_frames = max(1, int(nframe) if nframe and nframe > 0 else 8)
        if nframe_max > 0 and desired_frames > nframe_max:
            desired_frames = nframe_max
        if desired_frames >= total_frames:
            indices = list(range(total_frames))
        elif desired_frames == 1:
            indices = [0]  # Always use first frame for single frame sampling
        else:
            # Generate evenly spaced indices and ensure uniqueness
            raw_indices = np.linspace(0, total_frames - 1, desired_frames)
            indices = list(np.unique(np.round(raw_indices).astype(int)))

    images = [Image.fromarray(vid[i].asnumpy()) for i in indices]
    frame_urls = [encode_pil_to_jpeg_data_url(im) for im in images]

    # Calculate timestamps for each sampled frame
    timestamps = [float(idx) / video_fps for idx in indices]

    # Calculate metadata for the sampled frames
    sampled_num_frames = len(indices)

    # Duration is the time span from first to last frame
    if len(timestamps) > 1:
        sampled_duration = timestamps[-1] - timestamps[0]
        sampled_fps = (sampled_num_frames - 1) / sampled_duration if sampled_duration > 0 else 1.0
    else:
        # Single frame case
        sampled_duration = None
        sampled_fps = None

    metadata = VideoMetadata(
        total_num_frames=sampled_num_frames,
        fps=sampled_fps,
        duration=sampled_duration,
        video_backend=None,
    )

    return frame_urls, metadata


def maybe_path_or_url_to_data_urls(path_or_url, fps=1, nframe=0, nframe_max=-1):
    """
    Convert a path or URL to data URLs, handling videos, images, and remote files.

    Args:
        path_or_url: Path or URL to the media file
        fps: Target frames per second for video sampling (if > 0, uses fps-based sampling)
        nframe: Number of frames to sample from video (used if fps <= 0)
        nframe_max: Maximum number of frames to sample

    Returns:
        tuple: (data_urls, metadata)
        - data_urls: List of base64-encoded data URLs
        - metadata: VideoMetadata dataclass with video metadata or None for images
    """
    val = str(path_or_url or "")
    low = val.lower()

    # Handle data URLs
    if low.startswith("data:"):
        if low.startswith("data:video/mp4"):
            header, _, b64part = val.partition(",")
            if not b64part:
                return [val], None
            import tempfile

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            try:
                tmp.write(base64.b64decode(b64part))
                tmp.flush()
                tmp.close()
                return sample_video_frames_to_data_urls(tmp.name, fps=fps, nframe=nframe, nframe_max=nframe_max)
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
        return [val], None

    # Remote URL
    if low.startswith("http://") or low.startswith("https://"):
        if low.endswith(".mp4"):
            is_safe, reason = _is_safe_public_http_url(val)
            if not is_safe:
                logger.warning("Refusing to fetch video URL (%s): %s", reason, val)
                return [val], None
            try:
                import shutil
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpf:
                    local_path = tmpf.name
                    with _safe_url_open(val) as resp:
                        shutil.copyfileobj(resp, tmpf)
                result = sample_video_frames_to_data_urls(local_path, fps=fps, nframe=nframe, nframe_max=nframe_max)
                try:
                    os.unlink(local_path)
                except Exception:
                    pass
                return result
            except Exception:
                return [val], None
        return [val], None

    # Local path
    if os.path.exists(val):
        mime, _ = mimetypes.guess_type(val)
        if mime and mime.startswith("image/"):
            with open(val, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return [f"data:{mime};base64,{b64}"], None
        if mime == "video/mp4" or (mime is None and val.endswith(".mp4")):
            return sample_video_frames_to_data_urls(val, fps=fps, nframe=nframe, nframe_max=nframe_max)
        # Fallback: treat as binary image
        with open(val, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return [f"data:image/jpeg;base64,{b64}"], None

    return [val], None


def pil_image_from_base64(b64_str: str) -> Image.Image:
    """Decode a base64-encoded image to a PIL image."""
    # Handle data URLs like "data:image/png;base64,...."
    if b64_str.startswith("data:"):
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def adjust_image_tokens(
    input_ids: torch.Tensor | dict[str, torch.Tensor],
    num_tiles: int | list[int] | torch.Tensor,
    img_start_token_id: int,
    img_end_token_id: int,
    *,
    padding_values: Mapping[str, int | float] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Adjust each image placeholder without merging samples in the batch.

    ``num_tiles`` contains one desired media-token count per image, flattened
    in row-major sample order. A text-only row consumes no entry, while a row
    with multiple images consumes one entry for each ``<img>...</img>`` region.
    Every tensor in an input dictionary is adjusted at the same positions and
    rows are right-padded back to a common length. If the dictionary contains
    ``attention_mask``, trailing masked positions are discarded before the
    adjustment so stale processor padding cannot pin the adjusted batch width.

    Example:
        input_ids decoded may look like this
        System: ...
        User:...
        Image 1: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[0]
        Image 2: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[1]
        ...
        etc

    Args:
        input_ids: A 2D token tensor, or a dictionary containing ``input_ids``
            and other sequence-aligned 2D tensors with the same shape.
        num_tiles: Desired media-token counts for all images in row-major order.
        img_start_token_id: Token ID for ``<img>``.
        img_end_token_id: Token ID for ``</img>``.
        padding_values: Per-tensor values used when adjusted rows have different
            lengths. Unspecified tensors default to zero.

    Returns:
        The adjusted tensor or dictionary, preserving the input batch size.

    Raises:
        ValueError: If tensors are not aligned, image boundaries are malformed,
            or the number of tile counts does not match the number of images.
    """
    input_is_dict = isinstance(input_ids, dict)
    if input_is_dict:
        if "input_ids" not in input_ids:
            raise ValueError("input_ids must be a dictionary containing an 'input_ids' tensor.")
        aligned_tensors = dict(input_ids)
    else:
        aligned_tensors = {"input_ids": input_ids}

    token_ids = aligned_tensors["input_ids"]
    if token_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D, got shape {tuple(token_ids.shape)}.")
    for key, tensor in aligned_tensors.items():
        if tensor.shape != token_ids.shape:
            raise ValueError(
                f"Tensor {key} has shape {tuple(tensor.shape)} but input_ids has shape {tuple(token_ids.shape)}."
            )

    valid_lengths = [token_ids.shape[1]] * token_ids.shape[0]
    attention_mask = aligned_tensors.get("attention_mask")
    if attention_mask is not None:
        valid_lengths = []
        for row in attention_mask:
            valid_positions = torch.where(row != 0)[0]
            valid_lengths.append(int(valid_positions[-1].item()) + 1 if valid_positions.numel() else 0)

    if isinstance(num_tiles, int):
        tile_counts = [num_tiles]
    elif isinstance(num_tiles, torch.Tensor):
        tile_counts = [int(count) for count in num_tiles.detach().cpu().reshape(-1).tolist()]
    else:
        tile_counts = [int(count) for count in num_tiles]
    if any(count < 0 for count in tile_counts):
        raise ValueError(f"num_tiles values must be non-negative, got {tile_counts}.")

    boundaries_by_row: list[list[tuple[int, int]]] = []
    total_images = 0
    for row_index, row in enumerate(token_ids):
        row = row[: valid_lengths[row_index]]
        starts = (row == img_start_token_id).nonzero(as_tuple=True)[0].tolist()
        ends = (row == img_end_token_id).nonzero(as_tuple=True)[0].tolist()
        if len(starts) != len(ends):
            raise ValueError(f"Mismatched image boundaries: found {len(starts)} starts and {len(ends)} ends.")

        boundaries: list[tuple[int, int]] = []
        previous_end = -1
        for start, end in zip(starts, ends, strict=True):
            if start <= previous_end or end <= start:
                raise ValueError(f"Malformed image boundaries at positions {start} and {end}.")
            media_tokens = row[start + 1 : end]
            if media_tokens.numel() == 0:
                raise ValueError(f"Image boundary at position {start} does not contain a media token.")
            if not torch.all(media_tokens == media_tokens[0]):
                raise ValueError(f"Image boundary at position {start} contains mixed media token IDs.")
            boundaries.append((start, end))
            previous_end = end

        boundaries_by_row.append(boundaries)
        total_images += len(boundaries)

    if len(tile_counts) != total_images:
        raise ValueError(f"Received {len(tile_counts)} tile counts for {total_images} image regions.")

    adjusted_rows: dict[str, list[torch.Tensor]] = {key: [] for key in aligned_tensors}
    tile_offset = 0
    for row_index, boundaries in enumerate(boundaries_by_row):
        row_tile_counts = tile_counts[tile_offset : tile_offset + len(boundaries)]
        tile_offset += len(boundaries)
        for key, tensor in aligned_tensors.items():
            row = tensor[row_index, : valid_lengths[row_index]]
            pieces: list[torch.Tensor] = []
            cursor = 0
            for (start, end), count in zip(boundaries, row_tile_counts, strict=True):
                pieces.append(row[cursor : start + 1])
                if count > 0:
                    pieces.append(row[start + 1 : start + 2].expand(count))
                pieces.append(row[end : end + 1])
                cursor = end + 1
            pieces.append(row[cursor:])
            adjusted_rows[key].append(torch.cat(pieces))

    max_length = max(row.numel() for row in adjusted_rows["input_ids"])
    padding_values = padding_values or {}
    adjusted_tensors: dict[str, torch.Tensor] = {}
    for key, rows in adjusted_rows.items():
        pad_value = padding_values.get(key, 0)
        padded_rows = [
            torch.cat([row, row.new_full((max_length - row.numel(),), pad_value)]) if row.numel() < max_length else row
            for row in rows
        ]
        adjusted_tensors[key] = torch.stack(padded_rows).contiguous()

    if input_is_dict:
        return adjusted_tensors
    return adjusted_tensors["input_ids"]
