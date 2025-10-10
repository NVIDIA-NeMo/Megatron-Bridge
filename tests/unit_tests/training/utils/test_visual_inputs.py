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

from megatron.bridge.training.utils.visual_inputs import Qwen2_5_VLVisualInputs


def test_normalized_for_model_shapes():
    # pixel_values: [B, N, C, H, W] -> [B*N, C, H, W]
    pixel_values = torch.randn(2, 3, 4, 5, 6)
    # image_grid_thw: [B, N, 3] -> [B*N, 3]
    image_grid_thw = torch.randint(0, 10, (2, 3, 3))

    vi = Qwen2_5_VLVisualInputs(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    kwargs = vi.normalized_for_model()

    assert "pixel_values" in kwargs
    assert "image_grid_thw" in kwargs
    assert kwargs["pixel_values"].shape == (2 * 3, 4, 5, 6)
    assert kwargs["image_grid_thw"].shape == (2 * 3, 3)


def test_as_model_kwargs_filters_none():
    vi = Qwen2_5_VLVisualInputs(pixel_values=None, image_grid_thw=None)
    kwargs = vi.as_model_kwargs()
    assert kwargs == {}
