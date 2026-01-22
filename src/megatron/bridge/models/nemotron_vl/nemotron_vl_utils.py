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
import mimetypes
import os

import torch
from PIL import Image
from transformers.video_utils import VideoMetadata


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
            try:
                import tempfile
                import urllib.request

                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpf:
                    urllib.request.urlretrieve(val, tmpf.name)
                    local_path = tmpf.name
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
    num_tiles: int | list[int],
    img_start_token_id: int,
    img_end_token_id: int,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Ensures the input_ids tensor contains the correct number of <image> tokens as specified by num_tiles.
    This adjustment is necessary to bridge the gap between from HF processor to Megatron LLaVAModel.

    Example:
        input_ids decoded may look like this
        System: ...
        User:...
        Image 1: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[0]
        Image 2: <img><image>...<image></img>  # adjust number of <image> tokens to be num_tiles[1]
        ...
        etc

    Args:
        input_ids: The input_ids tensor (output of HF processor)
            or a dictionary of tensors, one of the keys of which must be "input_ids",
            and other tensors must have the same shape as input_ids
        num_tiles: The number of <image> tokens to ensure, either a single int or a list of ints
        img_start_token_id: The token id of <img>
        img_end_token_id: The token id of </img>

    Returns:
        The input_ids tensor with the correct number of <image> tokens
        or a dictionary of tensors each with the same shape as input_ids
    """
    if isinstance(num_tiles, int):
        num_tiles = [num_tiles]
    if isinstance(input_ids, dict):
        assert "input_ids" in input_ids, "input_ids must be a dictionary with 'input_ids' as one of the keys"
        other_tensors = {key: value for key, value in input_ids.items() if key != "input_ids"}
        input_ids = input_ids["input_ids"]
    else:
        other_tensors = None

    for i, num_tile in enumerate(num_tiles):
        image_start_pos = (input_ids[0] == img_start_token_id).nonzero(as_tuple=True)[0][i].item()
        image_end_pos = (input_ids[0] == img_end_token_id).nonzero(as_tuple=True)[0][i].item()
        media_token_id = input_ids[0, image_start_pos + 1]  # this can be <image> or <video> token
        existing = image_end_pos - image_start_pos + 1

        if num_tile > existing:
            # Need to add tokens
            repeat = num_tile + 2 - existing  # +2 for <img> and </img> tokens
            repeat_tokens = torch.full((1, repeat), media_token_id, dtype=input_ids.dtype, device=input_ids.device)
            if other_tensors is not None:
                for key, tensor in other_tensors.items():
                    assert other_tensors[key].shape == input_ids.shape, (
                        f"Tensor {key} has shape {other_tensors[key].shape} but input_ids has shape {input_ids.shape}"
                    )
                    other_tensors[key] = torch.cat(
                        [tensor[:, : image_start_pos + 1], repeat_tokens, tensor[:, image_start_pos + 1 :]], dim=1
                    )
            input_ids = torch.cat(
                [input_ids[:, : image_start_pos + 1], repeat_tokens, input_ids[:, image_start_pos + 1 :]], dim=1
            )

        elif num_tile < existing:
            # Need to remove tokens (keep only the first `num_tile` occurrences)
            keep_tokens_mask = torch.ones_like(input_ids, dtype=torch.bool)
            positions = (input_ids[0][image_start_pos : image_end_pos + 1] == media_token_id).nonzero(as_tuple=True)[
                0
            ] + image_start_pos
            # positions to drop are after the first num_tile occurrences
            drop_positions = positions[num_tile:].tolist()
            keep_tokens_mask[0, drop_positions] = False
            if other_tensors is not None:
                for key, tensor in other_tensors.items():
                    assert other_tensors[key].shape == input_ids.shape, (
                        f"Tensor {key} has shape {other_tensors[key].shape} but input_ids has shape {input_ids.shape}"
                    )
                    other_tensors[key] = tensor[keep_tokens_mask].unsqueeze(0)
            input_ids = input_ids[keep_tokens_mask].unsqueeze(0)

    if other_tensors is not None:
        return {
            "input_ids": input_ids,
            **other_tensors,
        }
    else:
        return input_ids
