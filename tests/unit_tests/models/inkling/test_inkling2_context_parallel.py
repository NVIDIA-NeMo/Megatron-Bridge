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

"""Unit tests for Inkling2 context parallelism, playing every CP rank in one process (CPU)."""

import pytest
import torch

from megatron.bridge.models.inkling.modeling_inkling2 import ops
from megatron.bridge.models.inkling.modeling_inkling2.attention import (
    _eager_relative_sink_attention,
    context_parallel_relative_sink_attention,
)
from megatron.bridge.models.inkling.modeling_inkling2.context_parallel import (
    context_parallel_layout,
    context_parallel_token_indices,
)


pytestmark = pytest.mark.unit
CPU = torch.device("cpu")
WIDTH = 4
# (cp_size, packed boundaries, total tokens, batch). The last case has one-token chunks, shorter than the
# convolution's reach, so a chunk's halo comes from more than one earlier chunk.
CASES = [
    (2, None, 32, 2),
    (4, None, 64, 1),
    (2, [0, 8, 32], 32, 1),
    (4, [0, 8, 32, 64, 64], 64, 1),
]


class _Group:
    """Stand-in for a CP process group; layouts read only its size and rank."""

    def __init__(self, size: int, rank: int) -> None:
        self._size, self._rank = size, rank

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank


class _SimulatedHalo:
    """One rank's halo exchange, played with every rank's rows at hand."""

    def __init__(self, layout, all_rows):
        self.layout, self.all_rows = layout, all_rows

    def extend(self, x):
        halos = [rows.float().index_select(0, self.layout.halo_rows) for rows in self.all_rows]
        return torch.cat((x, *halos), dim=0)

    def shift(self, extended, x, lag):
        return self.layout.shift(extended, x, lag)


def _layouts(cp_size, cu_seqlens, total, batch):
    return [
        context_parallel_layout(
            _Group(cp_size, rank),
            local_tokens=total // cp_size,
            batch=batch,
            device=CPU,
            width=WIDTH,
            cu_seqlens=cu_seqlens,
        )
        for rank in range(cp_size)
    ]


def _boundaries(cu):
    return None if cu is None else torch.tensor(cu)


class TestTokenIndices:
    def test_unpacked_sequence_follows_zigzag_chunks(self):
        indices = context_parallel_token_indices(16, 2, CPU)
        assert indices.tolist() == [[0, 1, 2, 3, 12, 13, 14, 15], [4, 5, 6, 7, 8, 9, 10, 11]]

    def test_packed_documents_split_one_by_one(self):
        indices = context_parallel_token_indices(24, 2, CPU, torch.tensor([0, 8, 24]))
        assert indices.tolist() == [
            [0, 1, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23],
            [2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 19],
        ]

    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_every_token_lands_on_exactly_one_rank(self, cp_size, cu, total, batch):
        indices = context_parallel_token_indices(total, cp_size, CPU, _boundaries(cu))
        assert sorted(indices.flatten().tolist()) == list(range(total))

    def test_rejects_documents_not_divisible_by_twice_cp(self):
        with pytest.raises(ValueError, match=r"divisible by 2 \* CP"):
            context_parallel_token_indices(24, 2, CPU, torch.tensor([0, 6, 24]))


class TestLayout:
    def test_no_layout_without_context_parallelism(self):
        assert context_parallel_layout(_Group(1, 0), local_tokens=8, batch=1, device=CPU, width=WIDTH) is None

    def test_packed_layout_requires_batch_one(self):
        with pytest.raises(ValueError, match="batch size 1"):
            context_parallel_layout(
                _Group(2, 0), local_tokens=8, batch=2, device=CPU, width=WIDTH, cu_seqlens=torch.tensor([0, 16])
            )

    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_gather_order_restores_the_global_sequence(self, cp_size, cu, total, batch):
        tokens = torch.arange(total)
        layouts = _layouts(cp_size, _boundaries(cu), total, batch)
        gathered = torch.cat([tokens.index_select(0, layout.local_index) for layout in layouts])
        for layout in layouts:
            assert torch.equal(gathered.index_select(0, layout.gather_order), tokens)

    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_every_rank_shares_the_same_halo_rows(self, cp_size, cu, total, batch):
        layouts = _layouts(cp_size, _boundaries(cu), total, batch)
        for layout in layouts[1:]:
            assert torch.equal(layout.halo_rows, layouts[0].halo_rows)

    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_local_positions_match_the_full_sequence(self, cp_size, cu, total, batch):
        boundaries = _boundaries(cu)
        full = ops.document_positions(total, batch, CPU, boundaries)
        for layout in _layouts(cp_size, boundaries, total, batch):
            assert torch.equal(layout.local_positions, full.index_select(0, layout.local_index))


class TestShortConvolutions:
    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_static_convolution_matches_the_full_sequence(self, cp_size, cu, total, batch):
        torch.manual_seed(0)
        boundaries = _boundaries(cu)
        x = torch.randn(total, batch, 6, dtype=torch.bfloat16)
        weight = torch.randn(6, WIDTH)
        full = ops.static_short_conv(x, weight, ops.document_positions(total, batch, CPU, boundaries))
        layouts = _layouts(cp_size, boundaries, total, batch)
        shards = [x.index_select(0, layout.local_index) for layout in layouts]
        for layout, shard in zip(layouts, shards):
            halo = _SimulatedHalo(layout, shards)
            out = ops.static_short_conv(shard, weight, layout.local_positions, halo=halo)
            torch.testing.assert_close(out, full.index_select(0, layout.local_index), rtol=0, atol=0)

    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_dynamic_convolution_matches_the_full_sequence(self, cp_size, cu, total, batch):
        torch.manual_seed(1)
        boundaries = _boundaries(cu)
        heads, head_dim = 2, 3
        x = torch.randn(total, batch, heads * head_dim, dtype=torch.bfloat16)
        static = torch.randn(WIDTH, heads * head_dim)
        dynamic = torch.randn(total, batch, heads, WIDTH)
        full = ops.dynamic_short_conv(x, static, dynamic, ops.document_positions(total, batch, CPU, boundaries))
        layouts = _layouts(cp_size, boundaries, total, batch)
        shards = [x.index_select(0, layout.local_index) for layout in layouts]
        for layout, shard in zip(layouts, shards):
            halo = _SimulatedHalo(layout, shards)
            taps = dynamic.index_select(0, layout.local_index)
            out = ops.dynamic_short_conv(shard, static, taps, layout.local_positions, halo=halo)
            torch.testing.assert_close(out, full.index_select(0, layout.local_index), rtol=0, atol=0)


class TestAttention:
    @pytest.mark.parametrize("window", [None, 3])
    @pytest.mark.parametrize("cp_size, cu, total, batch", CASES)
    def test_local_queries_match_the_full_sequence(self, window, cp_size, cu, total, batch):
        torch.manual_seed(2)
        boundaries = _boundaries(cu)
        heads, kv_heads, dim, extent = 4, 2, 8, 5
        query = torch.randn(total, batch, heads, dim)
        key = torch.randn(total, batch, kv_heads, dim)
        value = torch.randn(total, batch, kv_heads, dim)
        rel_logits = torch.randn(batch, heads, total, extent)
        sink = torch.randn(batch, heads, total)
        docs = ops.document_ids(total, CPU, boundaries)
        full = _eager_relative_sink_attention(query, key, value, rel_logits, sink, window=window, docs=docs)
        for layout in _layouts(cp_size, boundaries, total, batch):
            index = layout.local_index
            out = context_parallel_relative_sink_attention(
                query.index_select(0, index),
                key,
                value,
                rel_logits.index_select(2, index),
                sink.index_select(2, index),
                window=window,
                layout=layout,
            )
            torch.testing.assert_close(out, full.index_select(0, index), rtol=1e-6, atol=1e-6)

    def test_rejects_keys_that_do_not_cover_the_sequence(self):
        layout = _layouts(2, None, 16, 1)[0]
        query = torch.randn(8, 1, 2, 4)
        short_keys = torch.randn(8, 1, 2, 4)
        with pytest.raises(ValueError, match="Keys cover 8 tokens"):
            context_parallel_relative_sink_attention(
                query,
                short_keys,
                short_keys,
                torch.zeros(1, 2, 8, 3),
                torch.zeros(1, 2, 8),
                window=None,
                layout=layout,
            )
