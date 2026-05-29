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

import pytest
import torch
from PIL import Image

from megatron.bridge.data.vlm_datasets.step37_flickr8k.multimodal_utils import (
    IMAGE_ITEM_TYPE,
    PATCH_ITEM_TYPE,
)
from megatron.bridge.data.vlm_datasets.step37_flickr8k.preprocess import (
    _CLIP_MEAN,
    _CLIP_STD,
    _image_to_tensor,
    _load_image,
    load_images,
)


pytestmark = pytest.mark.unit


def test_image_to_tensor_shape_and_clip_normalization():
    img = Image.new("RGB", (16, 16), (0, 0, 0))  # all-black
    out = _image_to_tensor(img, size=4)

    assert out.shape == (3, 4, 4)
    assert out.dtype == torch.float32
    # black pixel -> (0/255 - mean) / std, constant per channel.
    per_channel = torch.tensor([(0.0 - m) / s for m, s in zip(_CLIP_MEAN, _CLIP_STD)])
    expected = per_channel.view(3, 1, 1).expand(3, 4, 4)
    assert torch.allclose(out, expected, atol=1e-5)


def test_load_image_falls_back_to_zero_image(tmp_path):
    img = _load_image(str(tmp_path / "does_not_exist.jpg"))
    assert img.size == (224, 224)
    assert img.mode == "RGB"


def test_load_images_picks_size_per_type(tmp_path):
    p_img = tmp_path / "img.png"
    p_patch = tmp_path / "patch.png"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(p_img)
    Image.new("RGB", (32, 32), (40, 50, 60)).save(p_patch)

    out = load_images(
        [(str(p_img), IMAGE_ITEM_TYPE), (str(p_patch), PATCH_ITEM_TYPE)],
        image_size=8,
        patch_image_size=4,
    )

    assert len(out) == 2
    img_tensor, img_type = out[0]
    patch_tensor, patch_type = out[1]
    assert img_type == IMAGE_ITEM_TYPE
    assert patch_type == PATCH_ITEM_TYPE
    assert img_tensor.shape == (3, 8, 8)
    assert patch_tensor.shape == (3, 4, 4)
