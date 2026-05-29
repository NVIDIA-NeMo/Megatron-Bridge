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

from megatron.bridge.models.stepfun.modelling_step37.image_insert_embedding import (
    ImageForInsert,
    ImageInsertEmbedding,
)


pytestmark = pytest.mark.unit

_CPU = torch.device("cpu")


# ─── _normalize_feature_list ────────────────────────────────────────────────


def test_normalize_unbinds_3d_tensor():
    feats = torch.zeros(2, 3, 4)
    out = ImageInsertEmbedding._normalize_feature_list(feats, device=_CPU, dtype=torch.float32)
    assert len(out) == 2
    assert all(tuple(f.shape) == (3, 4) for f in out)


def test_normalize_mixes_rank2_and_rank3_in_list():
    out = ImageInsertEmbedding._normalize_feature_list(
        [torch.zeros(3, 4), torch.zeros(2, 3, 4)], device=_CPU, dtype=torch.float32
    )
    # one rank-2 feature + two unbound from the rank-3 tensor.
    assert len(out) == 3


def test_normalize_rejects_bad_tensor_rank():
    with pytest.raises(ValueError):
        ImageInsertEmbedding._normalize_feature_list(torch.zeros(3, 4), device=_CPU, dtype=torch.float32)


def test_normalize_rejects_bad_list_rank():
    with pytest.raises(ValueError):
        ImageInsertEmbedding._normalize_feature_list([torch.zeros(1, 2, 3, 4)], device=_CPU, dtype=torch.float32)


# ─── insert_features ────────────────────────────────────────────────────────


def test_insert_features_writes_after_flag():
    # [S, B, H] sequence-first embeddings, B=1.
    embeddings = torch.zeros(5, 1, 2)
    input_ids = torch.tensor([[0, 7, 0, 0, 0]])  # flag=7 at seq idx 1
    feature = torch.ones(1, 1, 2)  # one feature row

    out = ImageInsertEmbedding.insert_features(embeddings, feature, input_ids, flag=7)

    # flag at idx1 -> +1 -> row 2 overwritten with the feature row.
    assert out[2, 0].tolist() == [1.0, 1.0]
    # everything else untouched.
    assert out[0, 0].tolist() == [0.0, 0.0]
    assert out[3, 0].tolist() == [0.0, 0.0]


def test_insert_features_no_flag_is_noop():
    embeddings = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)
    input_ids = torch.tensor([[0, 1, 2, 3, 4]])  # no flag==7
    out = ImageInsertEmbedding.insert_features(embeddings, torch.ones(1, 1, 2), input_ids, flag=7)
    assert torch.equal(out, embeddings)


def test_insert_features_truncates_on_mismatch():
    embeddings = torch.zeros(5, 1, 2)
    input_ids = torch.tensor([[7, 0, 0, 0, 0]])  # single flag location
    features = torch.ones(2, 1, 2)  # two features, one location

    out = ImageInsertEmbedding.insert_features(embeddings, features, input_ids, flag=7)
    # flag at idx0 -> +1 -> row 1 written; only first feature used.
    assert out[1, 0].tolist() == [1.0, 1.0]
    assert out[2, 0].tolist() == [0.0, 0.0]


# ─── forward ────────────────────────────────────────────────────────────────


def _make_module(encoder_output_dim=4, hidden_size=2):
    seq_len = {"S": 0}

    def stub_embedding(input_ids, position_ids=None):
        s = input_ids.shape[1]
        b = input_ids.shape[0]
        seq_len["S"] = s
        return torch.zeros(s, b, hidden_size)

    module = ImageInsertEmbedding(
        language_embedding=stub_embedding,
        encoder_output_dim=encoder_output_dim,
        hidden_size=hidden_size,
        projector_bias=False,
    )
    return module


def test_forward_without_images_returns_word_embeddings():
    module = _make_module()
    input_ids = torch.tensor([[1, 2, 3]])
    out = module.forward(input_ids, images=None)
    assert tuple(out.shape) == (3, 1, 2)
    assert torch.count_nonzero(out) == 0  # stub returns zeros, no insert


def test_forward_projects_and_inserts_image_features():
    module = _make_module(encoder_output_dim=4, hidden_size=2)
    # Make the projector deterministic: weight of all ones, no bias.
    with torch.no_grad():
        module.align_projector.weight.fill_(1.0)

    input_ids = torch.tensor([[0, 7, 0]])  # flag at idx1 -> insert at row 2
    images = [ImageForInsert(insert_start_token=7, image_features=torch.ones(1, 1, 4))]

    out = module.forward(input_ids, images=images)

    assert tuple(out.shape) == (3, 1, 2)
    # projected = ones[1,4] @ ones[4,2]^T = [4, 4]
    assert out[2, 0].tolist() == [4.0, 4.0]
    assert out[0, 0].tolist() == [0.0, 0.0]


def test_forward_requires_preencoded_features():
    module = _make_module()
    images = [ImageForInsert(insert_start_token=7, image_features=None)]
    with pytest.raises(ValueError):
        module.forward(torch.tensor([[0, 7, 0]]), images=images)
