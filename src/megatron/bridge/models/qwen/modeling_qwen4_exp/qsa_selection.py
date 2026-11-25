# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Document-local QSA scores without replicating block keys for every query."""

import math

import torch
from torch import Tensor


@torch.no_grad()
def select_document_blocks(
    q: Tensor,
    pooled_keys: Tensor,
    block_valid: Tensor,
    doc_ids: Tensor,
    positions: Tensor,
    *,
    compress_ratio: int,
    block_topk: int,
    workspace_bytes: int = 256 << 20,
    unpacked_sequence_length: int | None = None,
) -> tuple[Tensor, bool]:
    """Return document-relative block bitsets with the dense selection shortcut.

    Each document shares its block keys across query rows. This avoids building
    ``[queries, padded_blocks, head_dim]`` replicated key tensors. Query chunks
    bound the FP32 head-score tensor. Top-k retains the padded block width and
    torch's tie behavior used by the original Megatron implementation.

    ``unpacked_sequence_length`` is trusted layout metadata from the indexer:
    each batch row is one document, flattened in batch-major order with positions
    starting at zero. It avoids reading document boundaries back from the GPU.
    Leave it unset for packed or arbitrary document layouts.
    """
    tokens, heads, dim = q.shape
    blocks = pooled_keys.shape[1]
    nbytes = (blocks + 1 + 7) // 8
    bits = torch.zeros(tokens, nbytes, dtype=torch.int32, device=q.device)
    visible = ((positions.long() + 1) // compress_ratio).clamp_max(blocks)
    if unpacked_sequence_length is None:
        all_selected = bool((visible <= block_topk).all().item())
    else:
        if (
            unpacked_sequence_length <= 0
            or tokens % unpacked_sequence_length != 0
            or pooled_keys.shape[0] != tokens // unpacked_sequence_length
            or blocks != unpacked_sequence_length // compress_ratio
        ):
            raise ValueError("Unpacked QSA metadata does not match the query and pooled-key shapes")
        all_selected = blocks <= block_topk
    if blocks == 0:
        return bits.to(torch.uint8), True
    block_ids = torch.arange(blocks, device=q.device)
    if all_selected:
        mask = block_ids.unsqueeze(0) < visible.unsqueeze(1)
        mask &= block_valid[doc_ids.long()]
        weights = (torch.ones_like(block_ids) << (block_ids & 7)).unsqueeze(0)
        byte_ids = (block_ids >> 3).unsqueeze(0).expand(tokens, -1)
        bits.scatter_add_(1, byte_ids, (mask * weights).to(torch.int32))
        return bits.to(torch.uint8), True
    chunk = max(1, workspace_bytes // max(1, heads * blocks * 4))
    documents = range(pooled_keys.shape[0]) if unpacked_sequence_length is not None else doc_ids.unique().tolist()
    for doc_id in documents:
        if unpacked_sequence_length is None:
            rows = torch.nonzero(doc_ids == doc_id, as_tuple=True)[0]
        else:
            # Retain advanced indexing and its contiguous GEMM inputs, including
            # the existing chunk boundaries and top-k tie behavior.
            rows = torch.arange(
                doc_id * unpacked_sequence_length, (doc_id + 1) * unpacked_sequence_length, device=q.device
            )
        keys = pooled_keys[doc_id].float()
        for start in range(0, rows.numel(), chunk):
            selected_rows = rows[start : start + chunk]
            scores = torch.einsum("thd,nd->thn", q[selected_rows].float(), keys).relu_().sum(1) / math.sqrt(dim)
            valid = (block_ids.unsqueeze(0) < visible[selected_rows].unsqueeze(1)) & block_valid[doc_id]
            scores.masked_fill_(~valid, float("-inf"))
            values, selected = scores.topk(min(block_topk, blocks), dim=-1)
            keep = torch.isfinite(values)
            selected = selected.masked_fill(~keep, 0)
            bit_values = ((torch.ones_like(selected) << (selected & 7)) * keep).to(torch.int32)
            row_bits = torch.zeros(selected_rows.numel(), nbytes, dtype=torch.int32, device=q.device)
            row_bits.scatter_add_(1, selected >> 3, bit_values)
            bits[selected_rows] = row_bits
    return bits.to(torch.uint8), False
