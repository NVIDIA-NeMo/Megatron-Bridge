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

import pickle

import pytest
import torch
import transformers
from transformers.models.minimax_m3_vl import modeling_minimax_m3_vl as modeling
from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLVisionConfig

from megatron.bridge.models.minimax_m3.transformers_compat import patch_minimax_m3_vision_rope


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
    grid = torch.tensor([[2, 2, 2]])
    # Spatial merge order: width changes first, then height, then the video frame.
    positions = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    if hasattr(modeling, "MiniMaxM3VLVisionRotaryEmbedding"):
        cosine, sine = rotary(torch.empty(8, head_dim), positions)
    else:
        cosine, sine = rotary(grid, device=torch.device("cpu"), dtype=torch.float32)

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
    assert rotary.state_dict() == {}
    # The registered replacement remains a normal serializable module.
    assert type(pickle.loads(pickle.dumps(rotary))) is type(rotary)


def test_vision_rope_correction_is_version_gated_and_idempotent(monkeypatch):
    active = transformers.__version__ == "5.17.0"
    symbol = (
        "MiniMaxM3VLVisionRotaryEmbedding"
        if hasattr(modeling, "MiniMaxM3VLVisionRotaryEmbedding")
        else "MiniMaxM3VL3DRotaryEmbedding"
    )
    original = getattr(modeling, symbol)
    assert patch_minimax_m3_vision_rope() is active
    assert patch_minimax_m3_vision_rope() is active
    assert getattr(modeling, symbol) is original
    assert bool(getattr(original, "_bridge_3d_rope", False)) is active
    for version in ("5.15.1", "5.16.1", "5.17.1", "5.18.0"):
        monkeypatch.setattr(transformers, "__version__", version)
        assert patch_minimax_m3_vision_rope() is False
        assert getattr(modeling, symbol) is original


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hf_vision_rope_keeps_fp32_frequencies_after_dtype_conversion(dtype):
    config = MiniMaxM3VLVisionConfig(
        hidden_size=160,
        num_attention_heads=2,
        intermediate_size=320,
        num_hidden_layers=1,
        spatial_merge_size=2,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    rotary = modeling.MiniMaxM3VLVisionModel(config).rotary_emb
    grid = torch.tensor([[2, 2, 64]])
    if hasattr(modeling, "MiniMaxM3VLVisionRotaryEmbedding"):
        positions = torch.tensor([[0, 0, 32], [1, 16, 0], [2, 4, 128]])
        inputs = torch.empty(3, 80, dtype=dtype)
        expected = rotary(inputs, positions)
        rotary.to(dtype=dtype)
        actual = rotary(inputs, positions)
    else:
        expected = rotary(grid, device=torch.device("cpu"), dtype=dtype)
        rotary.to(dtype=dtype)
        actual = rotary(grid, device=torch.device("cpu"), dtype=dtype)
    for result, reference in zip(actual, expected):
        assert result.dtype == dtype
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
