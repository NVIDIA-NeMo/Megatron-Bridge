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

"""CPU unit tests for dense Gemma 4 context-parallel layout helpers."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.gemma.gemma4_cp_attention import (
    Gemma4DenseHybridCPAttention,
    _nonpacked_local_positions,
    _nonpacked_rank_major_positions,
    _packed_document_metadata,
)
from megatron.bridge.models.gemma.modeling_gemma4 import get_gemma4_layer_spec


def test_nonpacked_positions_match_mcore_zigzag_layout() -> None:
    rank0 = _nonpacked_local_positions(4, 2, 0, torch.device("cpu"))
    rank1 = _nonpacked_local_positions(4, 2, 1, torch.device("cpu"))

    torch.testing.assert_close(rank0, torch.tensor([0, 1, 6, 7]))
    torch.testing.assert_close(rank1, torch.tensor([2, 3, 4, 5]))


def test_nonpacked_positions_reject_odd_local_length() -> None:
    with pytest.raises(ValueError, match="even rank-local"):
        _nonpacked_local_positions(3, 2, 0, torch.device("cpu"))


def test_nonpacked_rank_major_positions_match_collective_layout() -> None:
    positions = _nonpacked_rank_major_positions(4, 2, torch.device("cpu"))

    torch.testing.assert_close(positions, torch.tensor([0, 1, 6, 7, 2, 3, 4, 5]))


def test_packed_rank_major_positions_produce_exact_document_mask() -> None:
    # Two unequal documents: real lengths 5 and 7, each physically padded to 8.
    packed_seq_params = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, 5, 12]),
        cu_seqlens_q_padded=torch.tensor([0, 8, 16]),
    )
    rank0_positions = torch.tensor([0, 1, 6, 7, 8, 9, 14, 15])
    rank1_positions = torch.tensor([2, 3, 4, 5, 10, 11, 12, 13])
    # This is the actual all-gather layout: every rank's complete local pack,
    # not one globally ordered block per document.
    rank_major_key_positions = torch.cat((rank0_positions, rank1_positions))

    query_document_ids, key_document_ids, valid_queries, valid_keys = _packed_document_metadata(
        packed_seq_params,
        rank0_positions,
        rank_major_key_positions,
    )
    mask_mod = Gemma4DenseHybridCPAttention._mask_mod(
        rank0_positions,
        rank_major_key_positions,
        query_document_ids,
        key_document_ids,
        valid_keys,
    )
    query_indices = torch.arange(rank0_positions.numel())[:, None]
    key_indices = torch.arange(rank_major_key_positions.numel())[None, :]
    actual = mask_mod(None, None, query_indices, key_indices)
    expected = (
        (rank_major_key_positions[None, :] <= rank0_positions[:, None])
        & (query_document_ids[:, None] == key_document_ids[None, :])
        & valid_keys[None, :]
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(valid_queries, torch.tensor([True, True, False, False, True, True, True, False]))
    # Document 1's valid keys are split across the two rank-major blocks. This
    # assertion fails if gathered offsets are mistaken for global positions.
    assert actual[6].nonzero().flatten().tolist() == [4, 5, 6, 12, 13, 14, 15]


def test_layer_spec_selects_hybrid_attention_only_for_cp() -> None:
    cp1_spec = get_gemma4_layer_spec(SimpleNamespace(context_parallel_size=1))
    cp2_spec = get_gemma4_layer_spec(SimpleNamespace(context_parallel_size=2))

    assert cp1_spec.submodules.self_attention.submodules.core_attention is not Gemma4DenseHybridCPAttention
    assert cp2_spec.submodules.self_attention.submodules.core_attention is Gemma4DenseHybridCPAttention


def test_head_dim_512_uses_memory_safe_flex_tiles() -> None:
    assert Gemma4DenseHybridCPAttention._kernel_options(256) is None
    options = Gemma4DenseHybridCPAttention._kernel_options(512)
    assert options["BLOCK_M"] == 32
    assert options["BLOCK_N"] == 32


@pytest.mark.parametrize(
    ("num_heads", "num_query_groups", "tp_size", "cp_size", "expected"),
    [
        (32, 16, 4, 2, "a2a"),
        (32, 16, 4, 4, "a2a"),
        (8, 2, 2, 2, "all_gather"),
        (8, 4, 4, 2, "all_gather"),
    ],
)
def test_sliding_cp_falls_back_when_heads_do_not_divide_cp(
    num_heads: int,
    num_query_groups: int,
    tp_size: int,
    cp_size: int,
    expected: str,
) -> None:
    config = SimpleNamespace(
        num_attention_heads=num_heads,
        num_query_groups=num_query_groups,
        tensor_model_parallel_size=tp_size,
        context_parallel_size=cp_size,
    )

    assert Gemma4DenseHybridCPAttention._sliding_cp_comm_type(config) == expected
