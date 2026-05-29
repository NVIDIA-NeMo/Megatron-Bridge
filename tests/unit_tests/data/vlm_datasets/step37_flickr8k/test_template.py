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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.template import (
    MultimodalSFTSample,
    _expand_step37_image_placeholders,
    _identity_path,
)


pytestmark = pytest.mark.unit


def test_identity_path():
    assert _identity_path("/a/b.jpg") == "/a/b.jpg"


def test_expand_single_image_placeholder():
    out = _expand_step37_image_placeholders("<image>", image_token_count=2)
    assert out == "<im_start><im_patch><im_patch><im_end>"


def test_expand_keeps_surrounding_text():
    out = _expand_step37_image_placeholders("a <image> b", image_token_count=1)
    assert out == "a <im_start><im_patch><im_end> b"


def test_expand_multicrop_placeholders():
    out = _expand_step37_image_placeholders("<@image@> x <#image#>", image_token_count=2, patch_token_count=1)
    assert out == "<im_patch><im_patch> x <im_patch>"


def test_expand_noop_without_placeholder():
    assert _expand_step37_image_placeholders("plain text", image_token_count=2) == "plain text"


def test_sample_len_is_shifted_length():
    sample = MultimodalSFTSample(tokens=torch.zeros(5, dtype=torch.long))
    assert len(sample) == 4


def test_sample_len_floors_at_zero():
    assert len(MultimodalSFTSample(tokens=torch.zeros(1, dtype=torch.long))) == 0
    assert len(MultimodalSFTSample(tokens=torch.zeros(0, dtype=torch.long))) == 0
