# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native HF preprocessing adapted to the Omni family's packed vision input.

This is an inference adapter, not the fixed-canvas training preprocessing policy.
Both backends consume one processor invocation's pixels and expanded token IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch

from megatron.bridge.models.nemotron_omni.nemotron_omni_utils import temporal_model_frames


@dataclass
class NemotronOmniInputs:
    """Native HF inputs and their equivalent packed Bridge media tensors."""

    hf: dict[str, Any]
    bridge: dict[str, torch.Tensor]


def is_nemotron_omni(config: Any) -> bool:
    """Recognize Omni and Super-VL by registered model type or architecture."""
    names = [getattr(config, "model_type", ""), *(getattr(config, "architectures", None) or [])]
    return any(name.lower().startswith("nemotron") and "omni" in name.lower() for name in names)


def load_nemotron_omni_video(path: str, *, fps: float) -> tuple[list[Any], Any]:
    """Decode frames with the maintained Omni decoder and retain source timestamps."""
    from megatron.bridge.models.nemotron_vl.nemotron_vl_utils import (
        maybe_path_or_url_to_data_urls,
        pil_image_from_base64,
    )

    if fps <= 0:
        raise ValueError("Video sampling fps must be positive.")
    urls, metadata = maybe_path_or_url_to_data_urls(path, fps=fps, nframe=0, nframe_max=-1)
    frames = [pil_image_from_base64(url) for url in urls]
    if not frames:
        raise ValueError("Omni video inference requires at least one sampled frame.")
    return frames, metadata


def prepare_nemotron_omni_inputs(
    processor: Any,
    *,
    prompt: str,
    images: list[Any] | None = None,
    video_frames: list[Any] | None = None,
    video_metadata: Any = None,
) -> NemotronOmniInputs:
    """Render native expanded media tokens and pack exactly the processed pixels.

    Images remain independent image-embedder items. Video frames form one
    temporal item; a single frame is repeated only for Bridge's temporal routing.
    Sizes are post-resize, before temporal grouping. No additional resize or
    normalization is performed, and no training prompt is synthesized.
    """
    if bool(images) == bool(video_frames):
        raise ValueError("Provide images or one video's decoded frames, exclusively.")
    is_video = video_frames is not None
    media = video_frames if is_video else images
    marker = processor.video_token if is_video else processor.image_token
    content = "\n".join([marker] * (1 if is_video else len(media))) + "\n" + prompt
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    kwargs = {"videos" if is_video else "images": media}
    if is_video and video_metadata is not None:
        kwargs["videos_kwargs"] = {"video_metadata": video_metadata}
    # BatchFeature(tensor_type="pt") attempts to stack differently sized image
    # tensors. Keep native pixels ragged; tensorize only token fields below.
    processed = processor(text=[text], return_tensors=None, **kwargs)
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    pixels = processed[pixel_key]
    frames = list(pixels.unbind(0)) if torch.is_tensor(pixels) and pixels.ndim == 4 else pixels
    if not isinstance(frames, (tuple, list)) or not frames:
        raise ValueError("Omni processor must return nonempty CHW media tensors.")
    sizes = torch.as_tensor(processed["imgs_sizes"], dtype=torch.long)
    counts = torch.as_tensor(processed["num_tokens"], dtype=torch.long).reshape(-1)
    if sizes.shape != (len(frames), 2) or counts.numel() != len(frames):
        raise ValueError("Processor sizes and token counts must describe every processed image/frame.")
    if len(frames) != len(media):
        raise ValueError("Omni inference requires one processor image/frame per source item.")
    patch_dim = int(processor.image_processor.patch_size)
    if patch_dim <= 0:
        raise ValueError("Vision patch size must be positive.")
    patches = []
    for frame, size, count in zip(frames, sizes.tolist(), counts.tolist(), strict=True):
        height, width = size
        if frame.shape != (3, height, width) or height % (2 * patch_dim) or width % (2 * patch_dim):
            raise ValueError("Processed pixels must match post-resize sizes and an even 2x2 patch grid.")
        rows, cols = height // patch_dim, width // patch_dim
        if count != rows * cols // 4:
            raise ValueError("Processor visual token count disagrees with its post-resize grid.")
        patches.append(
            frame.reshape(3, rows, patch_dim, cols, patch_dim)
            .permute(1, 3, 0, 2, 4)
            .reshape(rows * cols, 3 * patch_dim * patch_dim)
        )
    if is_video:
        temporal_size = int(processor.video_temporal_patch_dim)
        if temporal_size <= 0 or not torch.equal(sizes, sizes[:1].expand_as(sizes)):
            raise ValueError("Video requires a positive temporal size and a common post-resize frame grid.")
        expected_tokens = ((len(frames) + temporal_size - 1) // temporal_size) * int(counts[0])
        # A one-frame video must select the temporal embedder, not the image embedder.
        patches = temporal_model_frames(patches, temporal_size)
        sizes = torch.stack(temporal_model_frames(list(sizes.unbind(0)), temporal_size))
        num_frames = torch.tensor([len(patches)], dtype=torch.long)
    else:
        expected_tokens = int(counts.sum())
        num_frames = torch.ones(len(frames), dtype=torch.long)
    input_ids = torch.as_tensor(processed["input_ids"], dtype=torch.long)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Public Omni inference supports one prompt at a time.")
    if int((input_ids == processor.image_token_id).sum()) != expected_tokens:
        raise ValueError("Native expanded image-token count disagrees with the processed media grid/tubelets.")
    hf_inputs = {"input_ids": input_ids, pixel_key: pixels}
    if "attention_mask" in processed:
        hf_inputs["attention_mask"] = torch.as_tensor(processed["attention_mask"], dtype=torch.long)
    return NemotronOmniInputs(
        hf=hf_inputs,
        bridge={
            "pixel_values": torch.cat(patches).unsqueeze(0).contiguous(),
            "imgs_sizes": sizes,
            "num_frames": num_frames,
        },
    )


def nemotron_omni_reference_metadata(
    inputs: NemotronOmniInputs, *, model: str, revision: str | None, video_pruning_rate: float | None = None
) -> dict[str, Any]:
    """Bind a saved HF reference to its source and exact preprocessed media bytes."""
    if "pixel_values_videos" in inputs.hf and video_pruning_rate != 0.0:
        raise ValueError("Omni video references must explicitly record unpruned HF inference (rate 0).")
    tensors = {}
    for name, value in {**inputs.hf, **{f"bridge.{k}": v for k, v in inputs.bridge.items()}}.items():
        values = value if isinstance(value, (list, tuple)) else [value]
        tensors[name] = [
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(
                    tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                ).hexdigest(),
            }
            for tensor in values
        ]
    return {
        "schema": "nemotron-omni-native-inputs-v1",
        "model": model,
        "revision": revision,
        "video_pruning_rate": video_pruning_rate,
        "tensors": tensors,
    }
