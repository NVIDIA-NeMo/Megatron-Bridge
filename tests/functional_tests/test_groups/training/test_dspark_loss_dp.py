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

"""Multi-GPU functional test for the DSpark loss data-parallel reduction.

The DSpark loss (:func:`megatron.bridge.training.post_training.dspark.loss.dspark_loss`)
divides each per-token numerator by an all-reduced global denominator and scales
the backward by ``world_size``, so that the data-parallel-averaged gradient equals
the single-process full-batch gradient. That invariant cannot be exercised on a
single process; this test runs two NCCL ranks and asserts the DP-averaged gradient
through a shared head matches the single-process reference over the concatenated
batch. It needs 2 GPUs and is intended for a GPU node / cluster runner (not the
CPU unit-test lane). Run it directly on a 2-GPU node: ``pytest <this file>``.
"""

import os
import socket

import pytest
import torch
import torch.multiprocessing as mp

from megatron.bridge.training.post_training.dspark.heads import VanillaMarkov
from megatron.bridge.training.post_training.dspark.loss import DSparkForwardOutput, dspark_loss


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _grad_over(head, sl, batch, dp_group):
    """Loss backward over the block-slice ``sl``; return the shared head's grad.

    Args:
        head: Shared ``VanillaMarkov`` whose ``markov_w2.weight`` grad is returned.
        sl: Slice into the block axis (dim 0) selecting this rank's blocks.
        batch: ``(base_logits, prev_ids, target_ids, aligned_target_logits,
            eval_mask, block_keep_mask)`` for the full batch; each is
            ``[total_blocks, num_blocks, block_size(, vocab)]``.
        dp_group: Data-parallel process group for the loss denominator reduction,
            or ``None`` for the single-process reference.

    Returns:
        ``markov_w2.weight.grad`` after one backward.
    """
    base, prev, target_ids, aligned, eval_mask, keep = batch
    head.zero_grad(set_to_none=True)
    draft = head.apply_block_logits(base[sl], token_ids=prev[sl], hidden_states=None)
    out = DSparkForwardOutput(
        draft_logits=draft,
        target_ids=target_ids[sl],
        eval_mask=eval_mask[sl],
        block_keep_mask=keep[sl],
        aligned_target_logits=aligned[sl],
    )
    loss, _ = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0, dp_group=dp_group)
    loss.backward()
    return head.markov_w2.weight.grad.detach().clone()


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world_size))
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        dev = torch.device("cuda", rank)
        blocks_per_rank, num_blocks, block_size, vocab, rank_dim = 3, 2, 4, 32, 5
        total = blocks_per_rank * world_size

        # Identical head and full batch on every rank (seeded on CPU, then moved), so
        # each rank can form the same single-process reference locally.
        torch.manual_seed(0)
        head = VanillaMarkov(vocab_size=vocab, markov_rank=rank_dim).to(dev)
        torch.manual_seed(1234)
        shape4 = (total, num_blocks, block_size, vocab)
        shape3 = (total, num_blocks, block_size)
        batch = (
            torch.randn(*shape4).to(dev),  # base logits
            torch.randint(0, vocab, shape3).to(dev),  # prev token ids
            torch.randint(0, vocab, shape3).to(dev),  # target ids
            torch.randn(*shape4).to(dev),  # aligned target logits
            torch.ones(*shape3, dtype=torch.bool).to(dev),  # eval mask
            torch.ones(total, num_blocks, dtype=torch.bool).to(dev),  # block keep mask
        )

        grad_ref = _grad_over(head, slice(0, total), batch, dp_group=None)

        sl = slice(rank * blocks_per_rank, (rank + 1) * blocks_per_rank)
        grad_local = _grad_over(head, sl, batch, dp_group=torch.distributed.group.WORLD)
        grad_dist = grad_local.clone()
        torch.distributed.all_reduce(grad_dist, op=torch.distributed.ReduceOp.SUM)
        grad_dist /= world_size

        rel = (grad_dist - grad_ref).abs().max() / grad_ref.abs().max().clamp_min(1e-6)
        assert rel < 1e-4, f"[rank {rank}] DP-averaged gradient mismatch rel={rel.item():.3e}"
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="DSpark loss DP-gradient test needs 2 GPUs")
def test_dspark_loss_dp_gradient_matches_single_process():
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
