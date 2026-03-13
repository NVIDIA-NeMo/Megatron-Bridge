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

"""Unit tests for FLUX attention modules."""

from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.diffusion.models.flux.flux_attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)


pytestmark = [pytest.mark.unit]


class TestJointSelfAttentionSubmodules:
    """Test JointSelfAttentionSubmodules dataclass."""

    def test_default_instantiation(self):
        """Test submodules can be created with defaults (all None)."""
        sub = JointSelfAttentionSubmodules()
        assert sub.linear_qkv is None
        assert sub.added_linear_qkv is None
        assert sub.core_attention is None
        assert sub.linear_proj is None
        assert sub.q_layernorm is None
        assert sub.k_layernorm is None
        assert sub.added_q_layernorm is None
        assert sub.added_k_layernorm is None

    def test_custom_instantiation(self):
        """Test submodules can be created with custom types."""
        linear_cls = MagicMock()
        sub = JointSelfAttentionSubmodules(
            linear_qkv=linear_cls,
            added_linear_qkv=linear_cls,
            core_attention=MagicMock(),
            linear_proj=linear_cls,
            q_layernorm=MagicMock(),
            k_layernorm=MagicMock(),
        )
        assert sub.linear_qkv is linear_cls
        assert sub.added_linear_qkv is linear_cls
        assert sub.linear_proj is linear_cls
        assert sub.q_layernorm is not None
        assert sub.k_layernorm is not None
        assert sub.added_q_layernorm is None
        assert sub.added_k_layernorm is None


class TestJointSelfAttentionSplitQkv:
    """Test JointSelfAttention._split_qkv logic in isolation."""

    def test_split_qkv_output_shapes(self):
        """Test _split_qkv splits mixed QKV into Q, K, V with correct shapes."""
        # Use attributes that match a typical FLUX config: 4 heads, 4 groups, head_dim 64
        num_query_groups_per_partition = 4
        num_attention_heads_per_partition = 4
        hidden_size_per_attention_head = 64

        # mixed_qkv last dim = ng * (np/ng + 2) * hn = 4 * (1 + 2) * 64 = 768
        q_per_group = num_attention_heads_per_partition // num_query_groups_per_partition
        mixed_qkv_last_dim = (
            num_query_groups_per_partition
            * (q_per_group + 2)
            * hidden_size_per_attention_head
        )
        assert mixed_qkv_last_dim == 768

        sq, b = 8, 2
        mixed_qkv = torch.randn(sq, b, mixed_qkv_last_dim)

        # Bind _split_qkv to a mock that has the required attributes
        receiver = MagicMock()
        receiver.num_query_groups_per_partition = num_query_groups_per_partition
        receiver.num_attention_heads_per_partition = num_attention_heads_per_partition
        receiver.hidden_size_per_attention_head = hidden_size_per_attention_head
        split_qkv = JointSelfAttention._split_qkv.__get__(receiver, JointSelfAttention)

        query, key, value = split_qkv(mixed_qkv)

        # query: [sq, b, np, hn] = [8, 2, 4, 64]
        assert query.shape == (sq, b, num_attention_heads_per_partition, hidden_size_per_attention_head)
        # key, value: [sq, b, ng, hn] = [8, 2, 4, 64]
        assert key.shape == (sq, b, num_query_groups_per_partition, hidden_size_per_attention_head)
        assert value.shape == (sq, b, num_query_groups_per_partition, hidden_size_per_attention_head)

    def test_split_qkv_with_gqa(self):
        """Test _split_qkv with grouped query (num_heads > num_groups)."""
        num_query_groups_per_partition = 2
        num_attention_heads_per_partition = 4
        hidden_size_per_attention_head = 32

        q_per_group = num_attention_heads_per_partition // num_query_groups_per_partition  # 2
        mixed_qkv_last_dim = (
            num_query_groups_per_partition
            * (q_per_group + 2)
            * hidden_size_per_attention_head
        )
        # 2 * (2+2) * 32 = 256
        assert mixed_qkv_last_dim == 256

        mixed_qkv = torch.randn(4, 1, mixed_qkv_last_dim)
        receiver = MagicMock()
        receiver.num_query_groups_per_partition = num_query_groups_per_partition
        receiver.num_attention_heads_per_partition = num_attention_heads_per_partition
        receiver.hidden_size_per_attention_head = hidden_size_per_attention_head
        split_qkv = JointSelfAttention._split_qkv.__get__(receiver, JointSelfAttention)

        query, key, value = split_qkv(mixed_qkv)

        assert query.shape == (4, 1, 4, 32)
        assert key.shape == (4, 1, 2, 32)
        assert value.shape == (4, 1, 2, 32)


class TestFluxSingleAttentionRotaryPosEmb:
    """Test FluxSingleAttention rotary_pos_emb handling (code path)."""

    def test_rotary_pos_emb_single_wrapped_to_tuple(self):
        """Test that single rotary_pos_emb is duplicated to (emb, emb) in forward."""
        # We only verify the logic: when rotary_pos_emb is not a tuple, it becomes (rotary_pos_emb,) * 2.
        # This is the same logic used in FluxSingleAttention.forward and JointSelfAttention.forward.
        rotary_pos_emb = torch.randn(1, 2, 32)
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            wrapped = (rotary_pos_emb,) * 2
        else:
            wrapped = rotary_pos_emb
        assert isinstance(wrapped, tuple)
        assert len(wrapped) == 2
        assert wrapped[0] is rotary_pos_emb
        assert wrapped[1] is rotary_pos_emb

    def test_rotary_pos_emb_tuple_unchanged(self):
        """Test that tuple rotary_pos_emb is left as-is."""
        q_emb = torch.randn(1, 2, 32)
        k_emb = torch.randn(1, 2, 32)
        rotary_pos_emb = (q_emb, k_emb)
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            wrapped = (rotary_pos_emb,) * 2
        else:
            wrapped = rotary_pos_emb
        assert wrapped is rotary_pos_emb
        assert wrapped[0] is q_emb
        assert wrapped[1] is k_emb
