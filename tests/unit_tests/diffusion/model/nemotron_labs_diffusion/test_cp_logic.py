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

"""Unit tests for the diffusion context-parallel helpers (CPU only).

Covers the pure-tensor logic the sbd_block_diff context-parallel path relies on,
with no GPU and no model:
  - ``compute_block_bias`` exactly encodes the sbd_block_diff mask (no fully-masked row).
  - ``zigzag_slice`` / ``local_zigzag_mask`` agree, and partition the sequence.
  - ``all_gather_seq_cp`` is an exact forward round trip and a correct autograd
    transpose in backward (validated over a CPU gloo process group).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.bridge.diffusion.common.cp_utils import (
    all_gather_seq_cp,
    local_zigzag_mask,
    zigzag_slice,
)
from megatron.bridge.diffusion.common.dllm import compute_block_bias


# ---------------------------------------------------------------------------
# Reference implementations (independent of the code under test)
# ---------------------------------------------------------------------------


def _ref_sbd_allowed(block_size: int, n: int) -> torch.Tensor:
    """Independent dense ``[2n, 2n]`` boolean of the sbd_block_diff mask."""
    q_len = 2 * n
    allowed = torch.zeros(q_len, q_len, dtype=torch.bool)
    for q in range(q_len):
        for kv in range(q_len):
            x0_q = q >= n
            x0_kv = kv >= n
            bq = (q - n) // block_size if x0_q else q // block_size
            bkv = (kv - n) // block_size if x0_kv else kv // block_size
            block_diagonal = (bq == bkv) and (not x0_kv) and (not x0_q)
            offset_block_causal = (bq > bkv) and x0_kv and (not x0_q)
            fully_causal = (q >= kv) and x0_kv and x0_q
            allowed[q, kv] = block_diagonal or offset_block_causal or fully_causal
    return allowed


def _ref_zigzag_indices(seq_len: int, cp_rank: int, cp_size: int) -> list[int]:
    """Global positions owned by ``cp_rank`` under the load-balanced zigzag layout."""
    cs = seq_len // (2 * cp_size)
    out: list[int] = []
    for idx in (cp_rank, 2 * cp_size - 1 - cp_rank):
        out.extend(range(idx * cs, (idx + 1) * cs))
    return out


# ---------------------------------------------------------------------------
# Single-process checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_size,n", [(32, 128), (4, 16), (8, 64)])
def test_compute_block_bias_matches_sbd_mask(block_size: int, n: int) -> None:
    bias = compute_block_bias(block_size, n, dtype=torch.float32, device="cpu")
    assert bias.shape == (1, 1, 2 * n, 2 * n)

    allowed_from_bias = bias[0, 0] == 0.0
    ref = _ref_sbd_allowed(block_size, n)
    assert torch.equal(allowed_from_bias, ref)
    # Disallowed entries are the most-negative value (-> ~0 weight after softmax).
    assert (bias[0, 0][~ref] == torch.finfo(torch.float32).min).all()
    # Every query row keeps at least one allowed key (no fully-masked row -> no NaN).
    assert (allowed_from_bias.sum(dim=-1) > 0).all()


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_zigzag_slice_matches_owned_mask(cp_size: int) -> None:
    seq_len = 2 * cp_size * 6  # divisible by 2*cp
    x = torch.arange(seq_len).view(1, seq_len, 1).float()
    covered: set[int] = set()
    for r in range(cp_size):
        local_positions = zigzag_slice(x, r, cp_size, seq_dim=1).view(-1).long().tolist()
        ref = _ref_zigzag_indices(seq_len, r, cp_size)
        assert local_positions == ref
        # owned mask marks exactly the positions zigzag_slice selects
        mask = local_zigzag_mask(seq_len, r, cp_size, device="cpu")
        assert mask.nonzero().view(-1).tolist() == sorted(ref)
        assert covered.isdisjoint(local_positions), "ranks overlap"
        covered.update(local_positions)
    assert covered == set(range(seq_len)), "ranks do not cover the sequence"


# ---------------------------------------------------------------------------
# Distributed (gloo, CPU) check for the all-gather autograd
# ---------------------------------------------------------------------------


def _gather_worker(rank: int, world_size: int, port: int, seq_len: int, hidden: int, out_q) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        group = dist.group.WORLD
        full = torch.arange(seq_len * hidden).float().view(1, seq_len, hidden)  # identical on all ranks

        local = zigzag_slice(full, rank, world_size, seq_dim=1).clone().requires_grad_(True)
        gathered = all_gather_seq_cp(local, group, seq_dim=1)
        fwd_ok = torch.equal(gathered.detach(), full)

        # Backward: grad of sum(gathered * w) wrt local must equal the zigzag slice of
        # w (the all-gather's transpose is scatter), proving the autograd is correct.
        w = torch.arange(seq_len * hidden).float().view(1, seq_len, hidden) + 1.0
        (gathered * w).sum().backward()
        bwd_ok = torch.allclose(local.grad, zigzag_slice(w, rank, world_size, seq_dim=1))

        out_q.put((rank, bool(fwd_ok), bool(bwd_ok)))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("world_size", [2, 4])
def test_all_gather_seq_cp_roundtrip(world_size: int) -> None:
    seq_len, hidden = 2 * world_size * 5, 3
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_gather_worker, args=(r, world_size, port, seq_len, hidden, q)) for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in range(world_size)]
    for p in procs:
        p.join(timeout=120)
    for rank, fwd_ok, bwd_ok in results:
        assert fwd_ok, f"cp={world_size} rank={rank}: forward reconstruction wrong"
        assert bwd_ok, f"cp={world_size} rank={rank}: backward grad wrong"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
