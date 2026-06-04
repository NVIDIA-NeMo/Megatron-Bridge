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

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import torch


@dataclass
class GenericVisualInputs:
    """Container for visual modality tensors produced by HF processors.

    Expected input format:
        Optional HF processor tensor outputs. Qwen-style processors may provide
        batched image/video tensors with shape ``[B, N, C, H, W]`` and THW grid
        metadata with shape ``[B, N, 3]``. Other processors may provide already
        flat tensors such as ``[N, C, H, W]`` / ``[N, 3]`` or model-specific
        fields such as ``image_sizes`` and ``image_position_ids``.

    Output format:
        ``as_model_kwargs()`` returns all non-None fields unchanged.
        ``normalized_for_model()`` returns non-None fields with Qwen-style
        image/video tensors flattened to ``[B*N, C, H, W]`` and THW metadata
        flattened to ``[B*N, 3]``. Already-flat tensors and non-Qwen fields are
        passed through unchanged.
    """

    pixel_values: Optional[torch.Tensor] = None
    pixel_values_videos: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    video_grid_thw: Optional[torch.Tensor] = None
    image_sizes: Optional[torch.Tensor] = None
    image_position_ids: Optional[torch.Tensor] = None  # Gemma4-VL: 2D patch position coords [B, N, 2]
    mm_token_type_ids: Optional[torch.Tensor] = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """Return a mapping of non-None fields suitable for model forward kwargs."""
        result: dict[str, torch.Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        """Return non-None fields with Qwen-style batched visual tensors flattened.

        - pixel_values: [B, N, C, H, W] -> [B*N, C, H, W]
        - pixel_values_videos: [B, N, C, H, W] -> [B*N, C, H, W]
        - image_grid_thw: [B, N, 3] -> [B*N, 3]
        - video_grid_thw: [B, N, 3] -> [B*N, 3]
        """
        kwargs = self.as_model_kwargs()

        pixel_values = kwargs.get("pixel_values")
        if isinstance(pixel_values, torch.Tensor) and pixel_values.dim() == 5:
            b, n, c, h, w = pixel_values.shape
            kwargs["pixel_values"] = pixel_values.view(b * n, c, h, w)

        pixel_values_videos = kwargs.get("pixel_values_videos")
        if isinstance(pixel_values_videos, torch.Tensor) and pixel_values_videos.dim() == 5:
            b, n, c, h, w = pixel_values_videos.shape
            kwargs["pixel_values_videos"] = pixel_values_videos.view(b * n, c, h, w)

        image_grid_thw = kwargs.get("image_grid_thw")
        if isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.dim() == 3:
            kwargs["image_grid_thw"] = image_grid_thw.view(-1, image_grid_thw.size(-1))

        video_grid_thw = kwargs.get("video_grid_thw")
        if isinstance(video_grid_thw, torch.Tensor) and video_grid_thw.dim() == 3:
            kwargs["video_grid_thw"] = video_grid_thw.view(-1, video_grid_thw.size(-1))

        return kwargs


@dataclass
class Qwen2AudioInputs:
    """Container for Qwen2-Audio modality tensors.

    Fields mirror the processor outputs for Qwen2-Audio. The model expects
    ``input_features`` (mel spectrograms) and ``feature_attention_mask``.
    """

    input_features: Optional[torch.Tensor] = None
    feature_attention_mask: Optional[torch.Tensor] = None

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        """Return a mapping of non-None fields suitable for model forward kwargs."""
        result: dict[str, torch.Tensor] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None:
                result[f.name] = value
        return result

    def normalized_for_model(self) -> dict[str, torch.Tensor]:
        """Return non-None fields (no shape normalization needed for audio)."""
        return self.as_model_kwargs()
