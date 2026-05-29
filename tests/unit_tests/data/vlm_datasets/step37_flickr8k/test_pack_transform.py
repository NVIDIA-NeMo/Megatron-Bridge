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

from megatron.bridge.data.vlm_datasets.step37_flickr8k.pack_transform import (
    get_position_id_from_cu_seqlens,
    pack_samples,
)
from megatron.bridge.data.vlm_datasets.step37_flickr8k.template import MultimodalSFTSample


pytestmark = pytest.mark.unit


def _make_sample(tokens, image_paths=None):
    t = torch.tensor(tokens, dtype=torch.long)
    return MultimodalSFTSample(
        tokens=t,
        loss_mask=torch.ones_like(t, dtype=torch.float32),
        image_paths=image_paths or [],
    )


def test_position_id_resets_per_subseq():
    cu = torch.tensor([0, 4, 8], dtype=torch.int32)
    pos = get_position_id_from_cu_seqlens(cu)
    assert pos.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_position_id_single_subseq():
    pos = get_position_id_from_cu_seqlens(torch.tensor([0, 3]))
    assert pos.tolist() == [0, 1, 2]


def test_pack_samples_no_padding():
    s1 = _make_sample([1, 2, 3, 4, 5], image_paths=[("a.jpg", 0)])
    s2 = _make_sample([6, 7, 8, 9, 10], image_paths=[("b.jpg", 0)])

    packed = pack_samples([s1, s2], seqlen_divisible_by=4)

    # tokens / labels are the shift-by-one of each sub-seq, concatenated.
    assert packed["tokens"].tolist() == [1, 2, 3, 4, 6, 7, 8, 9]
    assert packed["labels"].tolist() == [2, 3, 4, 5, 7, 8, 9, 10]
    assert packed["loss_masks"].tolist() == [1.0] * 8
    assert packed["cu_seqlens"].tolist() == [0, 4, 8]
    assert packed["position_id"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert packed["image_paths"] == [("a.jpg", 0), ("b.jpg", 0)]
    assert int(packed["max_seq_len"]) == 4


def test_pack_samples_appends_padding_to_multiple():
    s1 = _make_sample([1, 2, 3, 4, 5])
    s2 = _make_sample([6, 7, 8, 9, 10])

    packed = pack_samples([s1, s2], seqlen_divisible_by=5)

    total = packed["tokens"].numel()
    assert total % 5 == 0
    # 4 + 4 NTP tokens -> pad sub-seq of length 2 appended (total 10).
    assert total == 10
    assert packed["cu_seqlens"].tolist() == [0, 4, 8, 10]
    # The padded tail is its own sub-seq with zeroed loss mask.
    assert packed["loss_masks"].tolist()[-2:] == [0.0, 0.0]


def test_pack_samples_does_not_mutate_caller_list_contents_order():
    s1 = _make_sample([1, 2, 3])
    pieces = [s1]
    pack_samples(pieces, seqlen_divisible_by=4)
    # pack_samples may append a padding piece, but the first stays put.
    assert pieces[0] is s1
