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

"""Padded equal-half attention preserves clean context, positions, and gradients."""

from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_mask

from megatron.bridge.diffusion.common.dllm import compute_block_mask
from megatron.bridge.diffusion.models.common import nemotron_labs_diffusion_attention as attention_module

from .test_nemotron_labs_diffusion_attention import _make_attention


pytestmark = pytest.mark.unit


def layout(device):
    width = 20
    prompts, responses = [1, 5, 9], [5, 8, 0]
    padding = torch.ones(3, 2 * width, dtype=torch.bool, device=device)
    oracle = torch.eye(2 * width, dtype=torch.bool, device=device).repeat(3, 1, 1)
    positions = torch.zeros(3, 2 * width, dtype=torch.long, device=device)
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        pad = (-prompt) % 4
        start = pad + prompt
        end = start + (response + 3) // 4 * 4
        padding[row, start:end] = False
        padding[row, width + pad : width + pad + prompt + response] = False
        pos = (torch.arange(width, device=device) - pad).clamp_min(0)
        positions[row] = pos.repeat(2)
        for block in range(start, end, 4):
            oracle[row, block : block + 4, block : block + 4] = True
            oracle[row, block : block + 4, width + pad : width + block] = True
        clean = width + pad
        length = prompt + response
        oracle[row, clean : clean + length, clean : clean + length] = torch.ones(
            length, length, dtype=torch.bool, device=device
        ).tril()
    return padding, positions, oracle[:, None]


def test_mask_excludes_padding_and_future_clean_tokens():
    padding, _, expected = layout("cpu")
    mask = compute_block_mask(4, 20, device="cpu", padding_mask=padding)
    actual = create_mask(mask.mask_mod, 3, 1, 40, 40, device="cpu")
    torch.testing.assert_close(actual, expected)
    assert actual.any(-1).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_output_and_gradients_match_logical_position_oracle(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required for compiled FlexAttention")
    layer = _make_attention(num_heads=4, num_kv_heads=2, head_dim=16, block_size=4, apply_llama4=True)
    layer.config.sequence_parallel = True
    layer.to(device)
    padding, positions, dense_mask = layout(device)
    torch.manual_seed(123)
    q, k, v = [torch.randn(40, 3, heads, 16, device=device, requires_grad=True) for heads in (4, 2, 2)]

    def dense_kernel(query, key, value, *, block_mask):
        mask = create_mask(block_mask.mask_mod, 3, 1, 40, 40, device=device)
        return F.scaled_dot_product_attention(
            query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1), attn_mask=mask
        )

    if device == "cpu":
        with patch.object(attention_module, "fused_flex_attention", dense_kernel):
            actual = layer(q, k, v, attention_mask=padding[:, None, None])
    else:
        actual = layer(q, k, v, attention_mask=padding[:, None, None])
    qr, kr, vr = [t.detach().clone().requires_grad_() for t in (q, k, v)]
    query, key, value = [t.permute(1, 2, 0, 3) for t in (qr, kr, vr)]
    cos, sin = layer.rope_embedding_module(query, positions)
    query, key = attention_module.apply_rotary_pos_emb(query, key, cos, sin)
    scale = 1 + layer.beta * torch.log(1 + torch.floor(positions / layer.max_position_embeddings))
    query = query * scale[:, None, :, None]
    expected = (
        F.scaled_dot_product_attention(
            query, key.repeat_interleave(2, 1), value.repeat_interleave(2, 1), attn_mask=dense_mask
        )
        .permute(2, 0, 1, 3)
        .reshape_as(actual)
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    cotangent = torch.randn_like(actual)
    cotangent = cotangent.masked_fill(padding.T[..., None], 0)
    actual.backward(cotangent)
    expected.backward(cotangent)
    for tensor, reference in zip((q, k, v), (qr, kr, vr)):
        torch.testing.assert_close(tensor.grad, reference.grad, atol=5e-5, rtol=5e-5)
        assert not tensor.grad[padding.T].any()

    # Changing excluded slots cannot affect any real query, even across layers.
    changed = [t.detach().masked_fill(padding.T[..., None, None], 100) for t in (q, k, v)]
    if device == "cpu":
        with patch.object(attention_module, "fused_flex_attention", dense_kernel):
            altered = layer(*changed, attention_mask=padding[:, None, None])
    else:
        altered = layer(*changed, attention_mask=padding[:, None, None])
    torch.testing.assert_close(altered[~padding.T], actual[~padding.T], atol=2e-5, rtol=2e-5)
