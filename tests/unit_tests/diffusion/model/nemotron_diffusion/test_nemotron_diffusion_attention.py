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

"""Unit tests for NemotronDiffusionAttention and its helper functions."""

import math
import types
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.models.common.nemotron_diffusion_attention import (
    Ministral3RotaryEmbedding,
    _get_llama_4_attn_scale,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------


def _make_config(
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    seq_len: int = 16,
    block_size: int = 4,
    apply_llama4: bool = True,
    apply_qk_scaling: bool = False,
) -> TransformerConfig:
    """Build a minimal TransformerConfig for NemotronDiffusionAttention."""
    hf_text_config = types.SimpleNamespace(
        max_position_embeddings=seq_len,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": seq_len,
        },
        num_attention_heads=num_heads,
        hidden_size=num_heads * head_dim,
    )
    cfg = TransformerConfig(
        num_layers=1,
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        kv_channels=head_dim,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        use_cpu_initialization=True,
    )
    cfg.seq_length = seq_len
    cfg.block_size = block_size
    cfg.apply_llama4_style_query_key_layer_scaling = apply_llama4
    cfg.hf_config = types.SimpleNamespace(text_config=hf_text_config)
    cfg.sequence_parallel = False
    cfg.apply_query_key_layer_scaling = apply_qk_scaling
    cfg.attention_dropout = 0.0
    return cfg


def _make_pg_collection() -> MagicMock:
    mock_tp_pg = MagicMock()
    mock_tp_pg.size.return_value = 1
    pg_collection = MagicMock()
    pg_collection.tp = mock_tp_pg
    return pg_collection


def _make_attention(
    num_heads: int = 4,
    num_kv_heads: int = 2,
    head_dim: int = 8,
    seq_len: int = 16,
    block_size: int = 4,
    layer_number: int = 1,
    apply_llama4: bool = True,
    apply_qk_scaling: bool = False,
):
    """Instantiate NemotronDiffusionAttention with compute_block_mask mocked."""
    cfg = _make_config(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len=seq_len,
        block_size=block_size,
        apply_llama4=apply_llama4,
        apply_qk_scaling=apply_qk_scaling,
    )
    from megatron.bridge.diffusion.models.common.nemotron_diffusion_attention import (
        NemotronDiffusionAttention,
    )

    with patch(
        "megatron.bridge.diffusion.models.common.nemotron_diffusion_attention.compute_block_mask",
        return_value=MagicMock(),
    ):
        return NemotronDiffusionAttention(
            cfg,
            layer_number,
            AttnMaskType.causal,
            "self",
            pg_collection=_make_pg_collection(),
        )


# ---------------------------------------------------------------------------
# TestRotateHalf
# ---------------------------------------------------------------------------


class TestRotateHalf:
    def test_shape_preserved(self):
        x = torch.randn(3, 4, 8)
        assert rotate_half(x).shape == x.shape

    def test_negation_of_second_half(self):
        x = torch.randn(2, 8)
        out = rotate_half(x)
        assert torch.allclose(out[..., : x.shape[-1] // 2], -x[..., x.shape[-1] // 2 :])

    def test_first_half_is_second_input_half(self):
        x = torch.randn(2, 8)
        out = rotate_half(x)
        assert torch.allclose(out[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2])

    def test_odd_last_dim_rounds_down(self):
        """With last dim=5, floor(5/2)=2: x1 has 2 elements, x2 has 3."""
        x = torch.randn(2, 5)
        out = rotate_half(x)
        assert out.shape == x.shape
        assert torch.allclose(out[..., :3], -x[..., 2:])
        assert torch.allclose(out[..., 3:], x[..., :2])


# ---------------------------------------------------------------------------
# TestApplyRotaryPosEmb
# ---------------------------------------------------------------------------


class TestApplyRotaryPosEmb:
    def test_output_shapes(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        cos = torch.randn(2, 8, 16)
        sin = torch.randn(2, 8, 16)
        q_e, k_e = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_e.shape == q.shape
        assert k_e.shape == k.shape

    def test_identity_with_zero_sin(self):
        """When cos=1 and sin=0, embedding should be identity."""
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        cos = torch.ones(2, 8, 16)
        sin = torch.zeros(2, 8, 16)
        q_e, k_e = apply_rotary_pos_emb(q, k, cos, sin)
        assert torch.allclose(q_e, q, atol=1e-6)
        assert torch.allclose(k_e, k, atol=1e-6)

    def test_unsqueeze_dim_zero(self):
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        cos = torch.ones(4, 8, 16)
        sin = torch.zeros(4, 8, 16)
        q_e, k_e = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)
        assert q_e.shape == q.shape
        assert torch.allclose(q_e, q, atol=1e-6)

    def test_q_and_k_transformed_independently(self):
        """Modifying k should not affect q_embed."""
        q = torch.randn(1, 4, 8, 16)
        k1 = torch.randn(1, 4, 8, 16)
        k2 = torch.randn(1, 4, 8, 16)
        cos = torch.randn(1, 8, 16)
        sin = torch.randn(1, 8, 16)
        q_e1, _ = apply_rotary_pos_emb(q, k1, cos, sin)
        q_e2, _ = apply_rotary_pos_emb(q, k2, cos, sin)
        assert torch.allclose(q_e1, q_e2)


# ---------------------------------------------------------------------------
# TestRepeatKv
# ---------------------------------------------------------------------------


class TestRepeatKv:
    def test_n_rep_1_returns_same_tensor(self):
        x = torch.randn(2, 4, 8, 16)
        out = repeat_kv(x, 1)
        assert out is x

    def test_output_shape_with_n_rep(self):
        batch, num_kv_heads, slen, head_dim = 2, 2, 8, 16
        x = torch.randn(batch, num_kv_heads, slen, head_dim)
        n_rep = 4
        out = repeat_kv(x, n_rep)
        assert out.shape == (batch, num_kv_heads * n_rep, slen, head_dim)

    def test_values_repeated(self):
        """Each KV head value should appear n_rep times consecutively."""
        batch, num_kv_heads, slen, head_dim = 1, 2, 4, 8
        x = torch.randn(batch, num_kv_heads, slen, head_dim)
        n_rep = 3
        out = repeat_kv(x, n_rep)
        for h in range(num_kv_heads):
            for r in range(n_rep):
                assert torch.allclose(out[:, h * n_rep + r, :, :], x[:, h, :, :])


# ---------------------------------------------------------------------------
# TestGetLlama4AttnScale
# ---------------------------------------------------------------------------


class TestGetLlama4AttnScale:
    def test_scale_at_position_zero_is_one(self):
        pos = torch.tensor([0.0])
        out = _get_llama_4_attn_scale(pos, beta=0.5, max_position_embeddings=8)
        assert torch.allclose(out, torch.ones_like(out), atol=1e-6)

    def test_scale_increases_with_position(self):
        pos_low = torch.tensor([4.0])
        pos_high = torch.tensor([16.0])
        out_low = _get_llama_4_attn_scale(pos_low, beta=0.1, max_position_embeddings=8)
        out_high = _get_llama_4_attn_scale(pos_high, beta=0.1, max_position_embeddings=8)
        assert out_high.item() > out_low.item()

    def test_output_shape(self):
        """Output should have an extra trailing dimension of size 1."""
        pos = torch.arange(4, dtype=torch.float)
        out = _get_llama_4_attn_scale(pos, beta=0.1, max_position_embeddings=8)
        assert out.shape[-1] == 1
        assert out.shape[:-1] == pos.shape


# ---------------------------------------------------------------------------
# TestMinistral3RotaryEmbedding
# ---------------------------------------------------------------------------


class TestMinistral3RotaryEmbedding:
    def _make_hf_config(self, seq_len: int = 16, num_heads: int = 4, head_dim: int = 8):
        return types.SimpleNamespace(
            max_position_embeddings=seq_len,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            num_attention_heads=num_heads,
            hidden_size=num_heads * head_dim,
        )

    def test_forward_output_shapes(self):
        hf_cfg = self._make_hf_config(seq_len=16, num_heads=4, head_dim=8)
        rope = Ministral3RotaryEmbedding(hf_cfg)
        x = torch.randn(2, 4, 16, 8)  # [b, np, sq, hn]
        position_ids = torch.arange(16).unsqueeze(0)  # [1, sq]
        cos, sin = rope(x, position_ids)
        assert cos.shape == (1, 16, 8)
        assert sin.shape == (1, 16, 8)

    def test_forward_cos_sin_dtype_matches_input(self):
        hf_cfg = self._make_hf_config()
        rope = Ministral3RotaryEmbedding(hf_cfg)
        x = torch.randn(1, 4, 16, 8).to(torch.float32)
        pos = torch.arange(16).unsqueeze(0)
        cos, sin = rope(x, pos)
        assert cos.dtype == x.dtype
        assert sin.dtype == x.dtype

    def test_default_rope_type_initializes(self):
        hf_cfg = self._make_hf_config()
        rope = Ministral3RotaryEmbedding(hf_cfg)
        assert rope.rope_type == "default"
        assert rope.inv_freq is not None


# ---------------------------------------------------------------------------
# TestNemotronDiffusionAttentionInit
# ---------------------------------------------------------------------------


class TestNemotronDiffusionAttentionInit:
    def test_softmax_scale_computed_correctly(self):
        head_dim = 8
        attn = _make_attention(head_dim=head_dim)
        assert abs(attn.softmax_scale - 1.0 / math.sqrt(head_dim)) < 1e-6

    def test_layer_scaling_divides_softmax_scale(self):
        head_dim = 8
        layer_number = 3
        attn = _make_attention(head_dim=head_dim, layer_number=layer_number, apply_qk_scaling=True)
        expected = (1.0 / math.sqrt(head_dim)) / layer_number
        assert abs(attn.softmax_scale - expected) < 1e-6

    def test_inference_mode_defaults(self):
        attn = _make_attention()
        assert attn._inference_mode is False
        assert attn._cache_enabled is False
        assert attn._kv_cache_k is None
        assert attn._kv_cache_v is None
        assert attn._kv_cache_seq_len == 0

    def test_set_inference_mode_true(self):
        attn = _make_attention()
        attn.set_inference_mode(True)
        assert attn._inference_mode is True

    def test_set_inference_mode_false_clears_cache(self):
        attn = _make_attention()
        attn.set_inference_mode(True)
        attn.set_inference_params(causal=True, cache_enabled=True)
        # Simulate a cached state
        attn._kv_cache_k = torch.randn(2, 2, 4, 8)
        attn._kv_cache_v = torch.randn(2, 2, 4, 8)
        attn._kv_cache_seq_len = 4

        attn.set_inference_mode(False)
        assert attn._kv_cache_k is None
        assert attn._kv_cache_v is None
        assert attn._kv_cache_seq_len == 0

    def test_set_inference_params(self):
        attn = _make_attention()
        attn.set_inference_params(causal=False, cache_enabled=True)
        assert attn._inference_causal is False
        assert attn._cache_enabled is True

    def test_clear_kv_cache(self):
        attn = _make_attention()
        attn._kv_cache_k = torch.randn(2, 2, 4, 8)
        attn._kv_cache_v = torch.randn(2, 2, 4, 8)
        attn._kv_cache_seq_len = 4
        attn.clear_kv_cache()
        assert attn._kv_cache_k is None
        assert attn._kv_cache_v is None
        assert attn._kv_cache_seq_len == 0


# ---------------------------------------------------------------------------
# TestNemotronDiffusionAttentionInferenceForward
# ---------------------------------------------------------------------------


class TestNemotronDiffusionAttentionInferenceForward:
    """Tests for NemotronDiffusionAttention._inference_forward via forward()."""

    NUM_HEADS = 4
    NUM_KV_HEADS = 2
    HEAD_DIM = 8
    SEQ_LEN = 16
    BATCH = 2

    def _attn(self, **kw):
        defaults = dict(
            num_heads=self.NUM_HEADS,
            num_kv_heads=self.NUM_KV_HEADS,
            head_dim=self.HEAD_DIM,
            seq_len=self.SEQ_LEN,
        )
        defaults.update(kw)
        attn = _make_attention(**defaults)
        attn.set_inference_mode(True)
        return attn

    def _qkv(self, sq: int = None):
        sq = sq or self.SEQ_LEN
        q = torch.randn(sq, self.BATCH, self.NUM_HEADS, self.HEAD_DIM)
        k = torch.randn(sq, self.BATCH, self.NUM_KV_HEADS, self.HEAD_DIM)
        v = torch.randn(sq, self.BATCH, self.NUM_KV_HEADS, self.HEAD_DIM)
        return q, k, v

    def test_inference_forward_output_shape(self):
        attn = self._attn()
        attn.set_inference_params(causal=True, cache_enabled=False)
        q, k, v = self._qkv()
        out = attn(q, k, v)
        hidden_size_per_partition = self.NUM_HEADS * self.HEAD_DIM
        assert out.shape == (self.SEQ_LEN, self.BATCH, hidden_size_per_partition)

    def test_inference_causal_prefill_no_cache(self):
        """Causal prefill with sq==sk uses SDPA built-in causal (no explicit mask)."""
        attn = self._attn()
        attn.set_inference_params(causal=True, cache_enabled=False)
        q, k, v = self._qkv()
        out = attn(q, k, v)
        assert out.shape[0] == self.SEQ_LEN

    def test_inference_bidirectional_no_mask(self):
        attn = self._attn()
        attn.set_inference_params(causal=False, cache_enabled=False)
        q, k, v = self._qkv()
        out = attn(q, k, v)
        assert out.shape == (self.SEQ_LEN, self.BATCH, self.NUM_HEADS * self.HEAD_DIM)

    def test_inference_cache_stores_kv(self):
        attn = self._attn()
        attn.set_inference_params(causal=True, cache_enabled=True)
        q, k, v = self._qkv()
        attn(q, k, v)
        assert attn._kv_cache_k is not None
        assert attn._kv_cache_v is not None
        assert attn._kv_cache_seq_len == self.SEQ_LEN

    def test_inference_cache_grows_on_decode(self):
        attn = self._attn()
        attn.set_inference_params(causal=True, cache_enabled=True)
        q, k, v = self._qkv()
        attn(q, k, v)
        assert attn._kv_cache_seq_len == self.SEQ_LEN

        q2, k2, v2 = self._qkv(sq=1)
        attn(q2, k2, v2)
        assert attn._kv_cache_seq_len == self.SEQ_LEN + 1

    def test_inference_causal_decode_with_cache(self):
        """Decode step (sq < sk) builds an explicit causal mask."""
        attn = self._attn()
        attn.set_inference_params(causal=True, cache_enabled=True)
        q, k, v = self._qkv()
        attn(q, k, v)

        q2, k2, v2 = self._qkv(sq=1)
        out2 = attn(q2, k2, v2)
        assert out2.shape == (1, self.BATCH, self.NUM_HEADS * self.HEAD_DIM)

    def test_clear_kv_cache_resets_state(self):
        attn = self._attn()
        attn.set_inference_params(causal=True, cache_enabled=True)
        q, k, v = self._qkv()
        attn(q, k, v)
        assert attn._kv_cache_k is not None

        attn.clear_kv_cache()
        assert attn._kv_cache_k is None
        assert attn._kv_cache_v is None
        assert attn._kv_cache_seq_len == 0
