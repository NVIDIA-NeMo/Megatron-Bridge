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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.multimodal_utils import (
    IMAGE_ITEM_TYPE,
    PATCH_ITEM_TYPE,
    build_image_for_insert,
    compute_rope_args,
)


pytestmark = pytest.mark.unit


def test_compute_rope_args_counts_patches():
    images = [torch.zeros(3, 28, 28), torch.zeros(3, 14, 14)]
    cu_seqlens, max_seq_len = compute_rope_args(images, patch_size=14, to_cuda=False)
    # (28//14)^2 = 4 patches, (14//14)^2 = 1 patch.
    assert cu_seqlens.tolist() == [0, 4, 5]
    assert cu_seqlens.dtype == torch.int32
    assert max_seq_len == 4


def test_compute_rope_args_rejects_empty():
    with pytest.raises(ValueError):
        compute_rope_args([], patch_size=14, to_cuda=False)


def test_build_image_for_insert_orders_patches_then_images():
    items = [
        (torch.zeros(3, 4, 4), IMAGE_ITEM_TYPE),
        (torch.zeros(3, 4, 4), IMAGE_ITEM_TYPE),
        (torch.zeros(3, 2, 2), PATCH_ITEM_TYPE),
    ]
    result = build_image_for_insert(items, patch_start_id=200, image_start_id=100, dtype=None, to_cuda=False)

    assert len(result) == 2
    # Patches are emitted first, then images.
    assert result[0].insert_start_token == 200
    assert tuple(result[0].images.shape) == (1, 3, 2, 2)
    assert result[1].insert_start_token == 100
    assert tuple(result[1].images.shape) == (2, 3, 4, 4)


def test_build_image_for_insert_applies_limits():
    items = [
        (torch.zeros(3, 4, 4), IMAGE_ITEM_TYPE),
        (torch.zeros(3, 4, 4), IMAGE_ITEM_TYPE),
        (torch.zeros(3, 4, 4), IMAGE_ITEM_TYPE),
    ]
    result = build_image_for_insert(
        items,
        patch_start_id=200,
        image_start_id=100,
        limit_images=1,
        dtype=None,
        to_cuda=False,
    )
    assert len(result) == 1
    assert tuple(result[0].images.shape) == (1, 3, 4, 4)


def test_build_image_for_insert_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_image_for_insert(
            [(torch.zeros(3, 4, 4), 5)],
            patch_start_id=200,
            image_start_id=100,
            dtype=None,
            to_cuda=False,
        )


def test_build_image_for_insert_attaches_rope_args():
    items = [(torch.zeros(3, 4, 4), IMAGE_ITEM_TYPE)]

    def fake_rope(tensors):
        return torch.tensor([0, 9], dtype=torch.int32), 9

    result = build_image_for_insert(
        items,
        patch_start_id=200,
        image_start_id=100,
        rope_args_fn=fake_rope,
        dtype=None,
        to_cuda=False,
    )
    assert result[0].rope_cu_seqlens.tolist() == [0, 9]
    assert result[0].rope_max_seq_len == 9
