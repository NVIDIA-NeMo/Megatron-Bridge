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
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as modeling
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLVisionConfig


@pytest.mark.parametrize("head_dim,axis_dim", [(8, 2), (80, 26)])
def test_hf_vision_rope_preserves_three_axes_and_frequency_ladder(head_dim, axis_dim):
    config = MiniMaxM3VLVisionConfig(
        hidden_size=head_dim * 2,
        num_attention_heads=2,
        intermediate_size=head_dim * 4,
        num_hidden_layers=1,
        spatial_merge_size=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    rotary = modeling.MiniMaxM3VLVisionModel(config).rotary_emb
    # Spatial merge order: width changes first, then height, then the video frame.
    positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    cosine, sine = rotary(torch.empty(8, head_dim), positions)

    bands = axis_dim // 2
    ladder = torch.logspace(0, -(bands - 1) / bands, bands, base=10000.0)
    angles = torch.cat([positions[:, axis, None] * ladder for axis in range(3)], dim=-1).repeat(1, 2)
    assert cosine.shape == sine.shape == (8, 3 * axis_dim)
    torch.testing.assert_close(cosine, angles.cos())
    torch.testing.assert_close(sine, angles.sin())
    for row in (1, 2, 4):
        assert not torch.equal(sine[0], sine[row])

    query = torch.randn(1, 8, 2, head_dim)
    key = torch.randn_like(query)
    rotated_query, rotated_key = modeling.apply_rotary_pos_emb_vision(query, key, cosine, sine)
    torch.testing.assert_close(rotated_query[..., 3 * axis_dim :], query[..., 3 * axis_dim :], rtol=0, atol=0)
    torch.testing.assert_close(rotated_key[..., 3 * axis_dim :], key[..., 3 * axis_dim :], rtol=0, atol=0)
    assert not torch.equal(rotated_query[..., : 3 * axis_dim], query[..., : 3 * axis_dim])
    # The inverse frequencies are non-persistent buffers, so checkpoints carry no RoPE tensors.
    assert rotary.state_dict() == {}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hf_vision_rope_rounds_its_frequency_buffers_on_dtype_casts(dtype):
    config = MiniMaxM3VLVisionConfig(
        hidden_size=160,
        num_attention_heads=2,
        intermediate_size=320,
        num_hidden_layers=1,
        spatial_merge_size=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    rotary = modeling.MiniMaxM3VLVisionModel(config).rotary_emb.to(dtype=dtype)
    positions = torch.tensor([[0, 0, 32], [1, 16, 0], [2, 4, 128]])

    actual = rotary(torch.empty(3, 80, dtype=dtype), positions)

    # The cast rounds the inverse-frequency buffers themselves, not only the outputs.
    axis_dim = 26
    frequencies = 1.0 / (10000.0 ** (torch.arange(0, axis_dim, 2).float() / axis_dim))
    angles = (positions.float()[..., None] * frequencies.to(dtype).float()).flatten(1).repeat(1, 2)
    for result, reference in zip(actual, (angles.cos().to(dtype), angles.sin().to(dtype))):
        assert result.dtype == dtype
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
