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

import torch

from megatron.bridge.training.utils.visual_inputs import GenericVisualInputs


def test_normalized_for_model_shapes():
    # pixel_values: [B, N, C, H, W] -> [B*N, C, H, W]
    pixel_values = torch.randn(2, 3, 4, 5, 6)
    # image_grid_thw: [B, N, 3] -> [B*N, 3]
    image_grid_thw = torch.randint(0, 10, (2, 3, 3))

    vi = GenericVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    kwargs = vi.normalized_for_model()

    assert "pixel_values" in kwargs
    assert "image_grid_thw" in kwargs
    assert kwargs["pixel_values"].shape == (2 * 3, 4, 5, 6)
    assert kwargs["image_grid_thw"].shape == (2 * 3, 3)


def test_as_model_kwargs_filters_none():
    vi = GenericVisualInputs(pixel_values=None, image_grid_thw=None)
    kwargs = vi.as_model_kwargs()
    assert kwargs == {}


def test_normalized_for_model_video_shapes():
    # pixel_values_videos: [B, N, C, H, W] -> [B*N, C, H, W]
    pixel_values_videos = torch.randn(2, 4, 3, 8, 8)
    # video_grid_thw: [B, N, 3] -> [B*N, 3]
    video_grid_thw = torch.randint(0, 10, (2, 4, 3))

    vi = GenericVisualInputs(
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )
    kwargs = vi.normalized_for_model()

    assert kwargs["pixel_values_videos"].shape == (2 * 4, 3, 8, 8)
    assert kwargs["video_grid_thw"].shape == (2 * 4, 3)
    # Image-only fields stay absent when None.
    assert "pixel_values" not in kwargs
    assert "image_grid_thw" not in kwargs


def test_normalized_for_model_mixed_image_and_video():
    pixel_values = torch.randn(2, 1, 3, 4, 4)
    pixel_values_videos = torch.randn(2, 2, 3, 4, 4)
    image_grid_thw = torch.randint(0, 10, (2, 1, 3))
    video_grid_thw = torch.randint(0, 10, (2, 2, 3))

    vi = GenericVisualInputs(
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    kwargs = vi.normalized_for_model()

    assert kwargs["pixel_values"].shape == (2, 3, 4, 4)
    assert kwargs["pixel_values_videos"].shape == (4, 3, 4, 4)
    assert kwargs["image_grid_thw"].shape == (2, 3)
    assert kwargs["video_grid_thw"].shape == (4, 3)


def test_normalized_for_model_already_flat_passthrough():
    # When tensors are already in the flat shape (dim != 5 / dim != 3),
    # normalized_for_model should leave them untouched.
    pixel_values = torch.randn(6, 3, 4, 4)  # already [B*N, C, H, W]
    image_grid_thw = torch.randint(0, 10, (6, 3))  # already [B*N, 3]
    pixel_values_videos = torch.randn(8, 3, 4, 4)
    video_grid_thw = torch.randint(0, 10, (8, 3))

    vi = GenericVisualInputs(
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    kwargs = vi.normalized_for_model()

    assert kwargs["pixel_values"].shape == (6, 3, 4, 4)
    assert kwargs["pixel_values_videos"].shape == (8, 3, 4, 4)
    assert kwargs["image_grid_thw"].shape == (6, 3)
    assert kwargs["video_grid_thw"].shape == (8, 3)


def test_as_model_kwargs_includes_video_fields():
    pixel_values_videos = torch.randn(1, 1, 3, 2, 2)
    video_grid_thw = torch.randint(0, 4, (1, 1, 3))
    vi = GenericVisualInputs(
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
    )
    kwargs = vi.as_model_kwargs()
    assert set(kwargs.keys()) == {"pixel_values_videos", "video_grid_thw"}
