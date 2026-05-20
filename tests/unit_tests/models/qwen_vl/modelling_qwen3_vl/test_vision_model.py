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

"""Unit tests for Qwen3-VL vision model (CUDA graph sequence padding and helpers)."""

import pytest
import torch
import torch.nn as nn

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import (
    Qwen3VLVisionModel,
    _maybe_pad_vision_sequence_for_cuda_graph,
    _vision_forward_packed_attention_setup,
)


def _minimal_vision_config(**kwargs):
    base = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        kv_channels=16,
        ffn_hidden_size=128,
        vocab_size=100,
        language_max_sequence_length=128,
        num_position_embeddings=64,
        spatial_merge_size=2,
        patch_size=16,
        out_hidden_size=64,
    )
    base.update(kwargs)
    return Qwen3VLTransformerConfig(**base)


def _vision_model_shell(config: Qwen3VLTransformerConfig) -> Qwen3VLVisionModel:
    """Build a :class:`Qwen3VLVisionModel` instance without running ``__init__`` (for helper tests)."""
    m = Qwen3VLVisionModel.__new__(Qwen3VLVisionModel)
    nn.Module.__init__(m)
    m.config = config
    return m


class TestMaybePadVisionSequenceForCudaGraph:
    """Tests for ``_maybe_pad_vision_sequence_for_cuda_graph`` (vision CUDA graph fixed length)."""

    def test_no_pad_when_seq_equals_max(self):
        seq_len, hidden, rot_d = 8, 32, 16
        hidden_states = torch.randn(seq_len, hidden)
        rotary_pos_emb = torch.randn(seq_len, 1, 1, rot_d)
        out_h, out_r, out_len = _maybe_pad_vision_sequence_for_cuda_graph(
            hidden_states, rotary_pos_emb, seq_len, max_seq_len=8
        )
        assert out_len == 8
        assert out_h.shape == (8, hidden)
        assert out_r.shape == (8, 1, 1, rot_d)
        assert torch.equal(out_h, hidden_states)
        assert torch.equal(out_r, rotary_pos_emb)

    def test_pad_to_max_appends_zeros(self):
        seq_len, max_seq_len, hidden, rot_d = 3, 8, 4, 6
        hidden_states = torch.arange(seq_len * hidden, dtype=torch.float32).reshape(seq_len, hidden)
        rotary_pos_emb = torch.ones(seq_len, 1, 1, rot_d)
        out_h, out_r, out_len = _maybe_pad_vision_sequence_for_cuda_graph(
            hidden_states, rotary_pos_emb, seq_len, max_seq_len=max_seq_len
        )
        assert out_len == max_seq_len
        assert out_h.shape == (max_seq_len, hidden)
        assert out_r.shape == (max_seq_len, 1, 1, rot_d)
        assert torch.equal(out_h[:seq_len], hidden_states)
        assert torch.equal(out_h[seq_len:], torch.zeros(max_seq_len - seq_len, hidden))
        assert torch.equal(out_r[:seq_len], rotary_pos_emb)
        assert torch.equal(out_r[seq_len:], torch.zeros(max_seq_len - seq_len, 1, 1, rot_d))

    def test_raises_when_seq_exceeds_max(self):
        seq_len, max_seq_len = 10, 4
        hidden_states = torch.randn(seq_len, 8)
        rotary_pos_emb = torch.randn(seq_len, 1, 1, 4)
        with pytest.raises(ValueError, match="exceeds max_vision_cuda_graph_seq_length"):
            _maybe_pad_vision_sequence_for_cuda_graph(hidden_states, rotary_pos_emb, seq_len, max_seq_len)


class TestVisionForwardPackedAttentionSetup:
    """Tests for ``_vision_forward_packed_attention_setup``."""

    def test_non_cuda_graph_path_calls_builder_and_no_mask(self):
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
        hidden = torch.zeros(5, 1, 8, dtype=torch.float32)
        sentinel = object()

        def _builder(_g):
            assert torch.equal(_g, grid)
            return sentinel

        packed, mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=False,
            hidden_states=hidden,
            original_seq_len=5,
            seq_len=5,
            grid_thw=grid,
            build_packed_seq_params=_builder,
        )
        assert packed is sentinel
        assert mask is None

    def test_cuda_graph_no_padding_no_mask(self):
        grid = torch.tensor([[1, 2, 2]], dtype=torch.long)
        hidden = torch.randn(8, 1, 16, dtype=torch.float32)
        packed, mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=True,
            hidden_states=hidden,
            original_seq_len=8,
            seq_len=8,
            grid_thw=grid,
            build_packed_seq_params=lambda _: pytest.fail("should not call build_packed_seq_params"),
        )
        assert packed is None
        assert mask is None

    def test_cuda_graph_with_padding_additive_mask(self):
        original_seq_len, seq_len, hidden = 2, 5, 4
        hidden_states = torch.randn(seq_len, 1, hidden, dtype=torch.float32)
        grid = torch.tensor([[1, 2, 2]], dtype=torch.long)

        packed, mask = _vision_forward_packed_attention_setup(
            use_cuda_graph_padding=True,
            hidden_states=hidden_states,
            original_seq_len=original_seq_len,
            seq_len=seq_len,
            grid_thw=grid,
            build_packed_seq_params=lambda _: pytest.fail("should not call build_packed_seq_params"),
        )
        assert packed is None
        assert mask is not None
        assert mask.shape == (1, 1, seq_len, seq_len)
        assert mask.dtype == hidden_states.dtype
        assert mask.device == hidden_states.device

        min_val = torch.finfo(hidden_states.dtype).min
        # Real token block: no additive masking
        assert torch.all(mask[:, :, :original_seq_len, :original_seq_len] == 0)
        # Padding keys: columns >= original_seq_len
        assert torch.all(mask[:, :, :original_seq_len, original_seq_len:] == min_val)
        # Padding queries: rows >= original_seq_len
        assert torch.all(mask[:, :, original_seq_len:, :] == min_val)


class TestQwen3VLVisionModelCudaGraphHelpers:
    """Tests for ``_get_max_vision_seq_length`` and ``_uses_vision_cuda_graph``."""

    def test_get_max_vision_seq_length_uses_config_override(self):
        cfg = _minimal_vision_config(max_vision_cuda_graph_seq_length=512)
        m = _vision_model_shell(cfg)
        assert m._get_max_vision_seq_length() == 512

    def test_get_max_vision_seq_length_default_from_num_position_embeddings(self):
        cfg = _minimal_vision_config(
            num_position_embeddings=64,
            spatial_merge_size=2,
            max_vision_cuda_graph_seq_length=None,
        )
        m = _vision_model_shell(cfg)
        assert m._get_max_vision_seq_length() == 64 // (2**2)

    def test_get_max_vision_seq_length_zero_override_falls_back_to_default(self):
        """``0`` is falsy in ``if max_vision_cuda_graph_seq_length``; use derived default."""
        cfg = _minimal_vision_config(
            num_position_embeddings=64,
            spatial_merge_size=2,
            max_vision_cuda_graph_seq_length=0,
        )
        m = _vision_model_shell(cfg)
        assert m._get_max_vision_seq_length() == 16

    def test_uses_vision_cuda_graph_true_in_training_with_te_impl(self):
        cfg = _minimal_vision_config(cuda_graph_impl="transformer_engine")
        m = _vision_model_shell(cfg)
        m.train()
        assert m._uses_vision_cuda_graph() is True

    def test_uses_vision_cuda_graph_false_in_eval(self):
        cfg = _minimal_vision_config(cuda_graph_impl="transformer_engine")
        m = _vision_model_shell(cfg)
        m.eval()
        assert m._uses_vision_cuda_graph() is False

    def test_uses_vision_cuda_graph_false_when_impl_not_te(self):
        cfg = _minimal_vision_config(cuda_graph_impl="none")
        m = _vision_model_shell(cfg)
        m.train()
        assert m._uses_vision_cuda_graph() is False
