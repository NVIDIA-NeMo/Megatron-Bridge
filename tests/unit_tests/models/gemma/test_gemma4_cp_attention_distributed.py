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

"""Two-rank forward/backward parity for dense Gemma 4 global CP attention.

Run with:
uv run python -m torch.distributed.run --nproc_per_node=2 -m pytest \
    tests/unit_tests/models/gemma/test_gemma4_cp_attention_distributed.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gemma.gemma4_cp_attention import (
    Gemma4DenseHybridCPAttention,
    _nonpacked_local_positions,
    _packed_local_positions,
)


_CP_SIZE = 2
_RTOL = 2e-2
_ATOL = 2e-2


def _assert_gradient_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare BF16 gradients by direction and relative norm across kernels."""
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    statistics = torch.stack(
        (
            (actual_float - expected_float).square().sum(),
            actual_float.square().sum(),
            expected_float.square().sum(),
            (actual_float * expected_float).sum(),
        )
    )
    dist.all_reduce(statistics)
    relative_l2 = (statistics[0] / statistics[2].clamp_min(torch.finfo(torch.float32).eps)).sqrt()
    cosine = statistics[3] / (statistics[1] * statistics[2]).sqrt().clamp_min(torch.finfo(torch.float32).eps)
    assert relative_l2.item() < 1e-2
    assert cosine.item() > 0.9999


def _build_attention() -> Gemma4DenseHybridCPAttention:
    attention = object.__new__(Gemma4DenseHybridCPAttention)
    torch.nn.Module.__init__(attention)
    attention.config = object()
    attention.layer_number = 1
    attention.attn_mask_type = AttnMaskType.causal
    attention.attention_type = "self"
    attention.softmax_scale = 1.0
    attention.cp_group = dist.group.WORLD
    attention._sliding_attention = None
    attention._block_mask_cache = {}
    return attention


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    allowed: torch.Tensor,
    valid_queries: torch.Tensor | None = None,
) -> torch.Tensor:
    output = F.scaled_dot_product_attention(
        query.permute(1, 2, 0, 3),
        key.permute(1, 2, 0, 3),
        value.permute(1, 2, 0, 3),
        attn_mask=allowed[None, None],
        scale=1.0,
        enable_gqa=query.size(2) != key.size(2),
    )
    if valid_queries is not None:
        output = output.masked_fill(~valid_queries[None, None, :, None], 0)
    return output.permute(2, 0, 1, 3).contiguous()


def _assert_global_parity(
    packed_seq_params: PackedSeqParams | None,
    global_positions: torch.Tensor,
    allowed: torch.Tensor,
    valid_queries: torch.Tensor | None = None,
) -> None:
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dtype = torch.bfloat16
    global_seq_len = global_positions.numel()
    local_seq_len = global_seq_len // _CP_SIZE

    generator = torch.Generator(device=device).manual_seed(1234)
    full_query = torch.randn(global_seq_len, 1, 2, 512, generator=generator, device=device, dtype=dtype)
    full_key = torch.randn(global_seq_len, 1, 1, 512, generator=generator, device=device, dtype=dtype)
    full_value = torch.randn(global_seq_len, 1, 1, 512, generator=generator, device=device, dtype=dtype)

    if packed_seq_params is None:
        local_positions = _nonpacked_local_positions(local_seq_len, _CP_SIZE, rank, device)
    else:
        local_positions = _packed_local_positions(
            packed_seq_params,
            local_seq_len,
            _CP_SIZE,
            rank,
            device,
            dist.group.WORLD,
        )

    query = full_query.index_select(0, local_positions).detach().requires_grad_(True)
    key = full_key.index_select(0, local_positions).detach().requires_grad_(True)
    value = full_value.index_select(0, local_positions).detach().requires_grad_(True)
    if packed_seq_params is not None:
        query_input = query.squeeze(1)
        key_input = key.squeeze(1)
        value_input = value.squeeze(1)
    else:
        query_input, key_input, value_input = query, key, value

    actual = _build_attention()(query_input, key_input, value_input, None, packed_seq_params=packed_seq_params)
    if packed_seq_params is not None:
        actual = actual.unsqueeze(1)
    actual_loss = actual.float().square().sum()
    actual_loss.backward()

    reference_query = full_query.detach().requires_grad_(True)
    reference_key = full_key.detach().requires_grad_(True)
    reference_value = full_value.detach().requires_grad_(True)
    reference = _reference_attention(
        reference_query,
        reference_key,
        reference_value,
        allowed,
        valid_queries,
    )
    local_reference = reference.index_select(0, local_positions)
    reference_loss = local_reference.float().square().sum()
    reference_loss.backward()
    dist.all_reduce(reference_key.grad)
    dist.all_reduce(reference_value.grad)

    local_reference = local_reference.flatten(-2)
    torch.testing.assert_close(actual, local_reference, rtol=_RTOL, atol=_ATOL)
    _assert_gradient_close(query.grad, reference_query.grad.index_select(0, local_positions))
    _assert_gradient_close(key.grad, reference_key.grad.index_select(0, local_positions))
    _assert_gradient_close(value.grad, reference_value.grad.index_select(0, local_positions))


@pytest.mark.gpu
@pytest.mark.parametrize("packed", [False, True])
def test_gemma4_global_cp2_matches_reference_forward_backward(packed: bool) -> None:
    """CP2 global head-dim-512 attention must match a full-sequence reference."""
    if int(os.environ.get("WORLD_SIZE", "1")) != _CP_SIZE:
        pytest.skip("requires a two-rank torch.distributed launch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Keep the group alive across parametrized cases. Destroying and recreating
    # it between cases races when Triton compilation takes different time per rank.
    device = torch.device("cuda", local_rank)
    if not packed:
        positions = torch.arange(16, device=device)
        allowed = positions[None, :] <= positions[:, None]
        _assert_global_parity(None, positions, allowed)
        return

    # Unequal real lengths exercise multi-document rank-major gather. Each
    # document is physically padded to a multiple of 2 * CP.
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 5, 12], device=device, dtype=torch.int32),
        cu_seqlens_kv=torch.tensor([0, 5, 12], device=device, dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 8, 16], device=device, dtype=torch.int32),
        cu_seqlens_kv_padded=torch.tensor([0, 8, 16], device=device, dtype=torch.int32),
        max_seqlen_q=8,
        max_seqlen_kv=8,
    )
    positions = torch.arange(16, device=device)
    document_ids = torch.where(positions < 8, 0, 1)
    valid_tokens = torch.where(document_ids == 0, positions < 5, positions - 8 < 7)
    allowed = (
        (positions[None, :] <= positions[:, None])
        & (document_ids[None, :] == document_ids[:, None])
        & valid_tokens[None, :]
    )
    _assert_global_parity(packed_seq_params, positions, allowed, valid_tokens)
