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

"""Unit tests for MiMoV2FlashTEDotProductAttention.attention_value_scale.

Verifies the ``forward`` override that multiplies the value tensor by
``attention_value_scale`` before calling the parent TE attention kernel.

Tests construct a real ``MiMoV2FlashTEDotProductAttention`` on GPU and
intercept the value tensor reaching the parent ``TEDotProductAttention.forward``
to assert the scaling is applied correctly.
"""

import pytest
import torch
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashTEDotProductAttention,
)


# Sequence / tensor dimensions used across tests.
_SEQ, _BATCH, _HEADS, _KV_HEADS, _HEAD_DIM = 8, 2, 4, 2, 64


def _make_config(attention_value_scale):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=_HEADS * _HEAD_DIM,
        num_attention_heads=_HEADS,
        num_query_groups=_KV_HEADS,
        kv_channels=_HEAD_DIM,
        use_cpu_initialization=True,
    )
    config.window_size = 128
    config.v_head_dim = _HEAD_DIM
    config.hybrid_attention_pattern = [1]
    config.attention_value_scale = attention_value_scale
    return config


def _make_attention(scale):
    """Construct a real MiMoV2FlashTEDotProductAttention on GPU."""
    config = _make_config(scale)
    return MiMoV2FlashTEDotProductAttention(
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
    ).cuda()


def _make_qkv(device="cuda", dtype=torch.bfloat16):
    """Return (query, key, value) tensors in sbhd format."""
    q = torch.randn(_SEQ, _BATCH, _HEADS, _HEAD_DIM, device=device, dtype=dtype)
    k = torch.randn(_SEQ, _BATCH, _KV_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    v = torch.randn(_SEQ, _BATCH, _KV_HEADS, _HEAD_DIM, device=device, dtype=dtype)
    return q, k, v


def _capture_parent_value(attn, q, k, v):
    """Run a forward pass and return the value tensor received by the parent.

    Temporarily replaces ``TEDotProductAttention.forward`` to record the
    value argument, then restores the original.
    """
    original_forward = TEDotProductAttention.forward
    captured = {}

    def _intercept(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
        captured["value"] = value.clone()
        return original_forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs)

    TEDotProductAttention.forward = _intercept
    try:
        attn(q, k, v, None, AttnMaskType.causal)
    finally:
        TEDotProductAttention.forward = original_forward

    return captured["value"]


@pytest.mark.run_only_on("GPU")
class TestAttentionValueScaleForward:
    """Regression coverage for the attention_value_scale forward path.

    The bug: ``_attention_value_scale`` was read from the HF config but
    silently dropped on the forward path, causing the attention output to
    diverge from the HF reference by a factor of ~1/scale.
    """

    def test_scale_applied_to_value(self):
        scale = 0.707
        attn = _make_attention(scale)
        q, k, v = _make_qkv()
        received_v = _capture_parent_value(attn, q, k, v)
        torch.testing.assert_close(received_v, v * scale)

    @pytest.mark.parametrize("scale", [0.5, 1.0, 1.5, 2.0])
    def test_scale_various_values(self, scale):
        attn = _make_attention(scale)
        q, k, v = _make_qkv()
        received_v = _capture_parent_value(attn, q, k, v)
        torch.testing.assert_close(received_v, v * scale)

    def test_none_scale_passes_value_unchanged(self):
        attn = _make_attention(None)
        q, k, v = _make_qkv()

        original_forward = TEDotProductAttention.forward
        captured = {}

        def _intercept(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
            captured["value"] = value
            return original_forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs)

        TEDotProductAttention.forward = _intercept
        try:
            attn(q, k, v, None, AttnMaskType.causal)
        finally:
            TEDotProductAttention.forward = original_forward

        assert captured["value"] is v

    def test_value_not_mutated_in_place(self):
        attn = _make_attention(0.5)
        q, k, v = _make_qkv()
        v_before = v.clone()
        attn(q, k, v, None, AttnMaskType.causal)
        torch.testing.assert_close(v, v_before)

    def test_query_and_key_unchanged(self):
        attn = _make_attention(0.707)
        q, k, v = _make_qkv()
        q_before, k_before = q.clone(), k.clone()

        original_forward = TEDotProductAttention.forward
        captured = {}

        def _intercept(self, query, key, value, attention_mask, attn_mask_type, **kwargs):
            captured["query"] = query
            captured["key"] = key
            return original_forward(self, query, key, value, attention_mask, attn_mask_type, **kwargs)

        TEDotProductAttention.forward = _intercept
        try:
            attn(q, k, v, None, AttnMaskType.causal)
        finally:
            TEDotProductAttention.forward = original_forward

        assert captured["query"] is q
        assert captured["key"] is k
        torch.testing.assert_close(q, q_before)
        torch.testing.assert_close(k, k_before)

    def test_output_shape(self):
        attn = _make_attention(0.707)
        q, k, v = _make_qkv()
        out = attn(q, k, v, None, AttnMaskType.causal)
        assert out.shape == (_SEQ, _BATCH, _HEADS * _HEAD_DIM)

    def test_different_scales_produce_different_outputs(self):
        attn = _make_attention(0.5)
        q, k, v = _make_qkv()
        out_half = attn(q, k, v, None, AttnMaskType.causal)

        attn._attention_value_scale = 2.0
        out_double = attn(q, k, v, None, AttnMaskType.causal)

        assert not torch.allclose(out_half, out_double, atol=1e-2)
