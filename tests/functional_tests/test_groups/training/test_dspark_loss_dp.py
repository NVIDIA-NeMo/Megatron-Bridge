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

"""Multi-GPU functional test for the DSpark loss under the Megatron-Core contract.

:func:`megatron.bridge.training.post_training.dspark.loss.dspark_loss` returns the
unnormalized micro-batch numerator plus ``num_tokens``, leaving Megatron-Core to
divide once by the globally reduced token count
(``calculate_per_token_loss=True``). This test reproduces that division over real
NCCL collectives across **two ranks running two micro-batches each, with a
different supervised token count in every micro-batch**, and asserts:

* the gradient obtained by accumulating local numerators and scaling by the global
  token total matches the single-process full-batch reference; and
* the biased alternative (normalizing each micro-batch by its own token count, the
  ``calculate_per_token_loss=False`` behaviour) does **not** match, so the test
  discriminates between the two contracts rather than passing either way.

Uneven counts are the whole point: with equal counts every normalization agrees.
Launched by ``L0_Launch_training_dspark.sh`` under
``torch.distributed.run --nproc_per_node=2``, like every other 2-rank test in this
directory.
"""

import os

import pytest
import torch

from megatron.bridge.training.post_training.dspark.heads import VanillaMarkov
from megatron.bridge.training.post_training.dspark.loss import DSparkForwardOutput, dspark_loss
from tests.functional_tests.utils import initialize_distributed


# Samples per (rank, micro-batch) along the batch axis. Deliberately unequal so a
# per-micro-batch normalization diverges from the global-token one.
SAMPLES_PER_MICRO_BATCH = ((1, 3), (3, 1))
NUM_BLOCKS, BLOCK_SIZE, VOCAB, MARKOV_RANK = 2, 4, 32, 5


def _grad(head):
    """The shared head parameter whose gradient this test watches."""
    return head.markov_w2.weight.grad.detach().clone()


def _backward_numerator(head, samples, batch):
    """Backward one micro-batch's unnormalized numerator into ``head``'s grads.

    Args:
        head: Shared ``VanillaMarkov`` whose gradients accumulate.
        samples: Slice into the batch axis selecting this micro-batch's samples.
        batch: ``(base_logits, prev_ids, target_ids, aligned_target_logits,
            eval_mask, block_keep_mask)`` for the full batch.

    Returns:
        The micro-batch's ``num_tokens`` as a float.
    """
    base, prev, target_ids, aligned, eval_mask, keep = batch
    draft = head.apply_block_logits(base[samples], token_ids=prev[samples], hidden_states=None)
    out = DSparkForwardOutput(
        draft_logits=draft,
        target_ids=target_ids[samples],
        eval_mask=eval_mask[samples],
        block_keep_mask=keep[samples],
        aligned_target_logits=aligned[samples],
    )
    loss, num_tokens, _ = dspark_loss(out, ce_alpha=0.1, l1_alpha=0.9, confidence_alpha=0.0)
    loss.backward()
    return float(num_tokens.item())


def _accumulate(head, slices, batch):
    """Sum numerator grads over ``slices``; return ``(grad, per_micro_batch_tokens)``."""
    head.zero_grad(set_to_none=True)
    tokens = [_backward_numerator(head, s, batch) for s in slices]
    return _grad(head), tokens


def _per_micro_batch_normalized(head, slices, batch):
    """The biased reference: divide each micro-batch by its own token count."""
    grad = None
    for s in slices:
        head.zero_grad(set_to_none=True)
        tokens = _backward_numerator(head, s, batch)
        local = _grad(head) / max(tokens, 1.0)
        grad = local if grad is None else grad + local
    return grad / len(slices)


def _micro_batch_slices(rank):
    """The ``slice`` per micro-batch for ``rank``, over the shared batch axis."""
    start = sum(sum(counts) for counts in SAMPLES_PER_MICRO_BATCH[:rank])
    slices = []
    for count in SAMPLES_PER_MICRO_BATCH[rank]:
        slices.append(slice(start, start + count))
        start += count
    return slices


def _build_batch(total_samples, device):
    """Identical head and full batch on every rank, so each can form the reference."""
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=VOCAB, markov_rank=MARKOV_RANK).to(device)
    torch.manual_seed(1234)
    shape4 = (total_samples, NUM_BLOCKS, BLOCK_SIZE, VOCAB)
    shape3 = (total_samples, NUM_BLOCKS, BLOCK_SIZE)
    eval_mask = torch.ones(*shape3, dtype=torch.bool)
    # Ragged supervision inside the blocks too, so token counts differ for reasons
    # beyond how many samples each micro-batch carries.
    eval_mask[::2, :, BLOCK_SIZE - 1 :] = False
    batch = (
        torch.randn(*shape4).to(device),  # base logits
        torch.randint(0, VOCAB, shape3).to(device),  # prev token ids
        torch.randint(0, VOCAB, shape3).to(device),  # target ids
        torch.randn(*shape4).to(device),  # aligned target logits
        eval_mask.to(device),
        torch.ones(total_samples, NUM_BLOCKS, dtype=torch.bool).to(device),
    )
    return head, batch


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="DSpark loss DP-gradient test needs 2 GPUs")
def test_dspark_loss_global_token_gradient_matches_single_process():
    initialize_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    assert world_size == len(SAMPLES_PER_MICRO_BATCH), (
        f"this test is written for {len(SAMPLES_PER_MICRO_BATCH)} ranks, got {world_size}"
    )
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))

    total_samples = sum(sum(counts) for counts in SAMPLES_PER_MICRO_BATCH)
    head, batch = _build_batch(total_samples, device)

    # Single-process reference: one numerator over the whole batch, divided by the
    # whole batch's token count.
    grad_num_ref, (tokens_ref,) = _accumulate(head, [slice(0, total_samples)], batch)
    grad_ref = grad_num_ref / tokens_ref

    slices = _micro_batch_slices(rank)
    grad_local, tokens_per_micro_batch = _accumulate(head, slices, batch)
    assert len(set(tokens_per_micro_batch)) > 1, (
        f"[rank {rank}] micro-batch token counts {tokens_per_micro_batch} are equal; the two "
        "normalizations coincide and this test would prove nothing"
    )

    # Megatron-Core's per-token-loss path: sum numerators and token counts over the
    # DP group, then divide once.
    grad_global = grad_local.clone()
    torch.distributed.all_reduce(grad_global, op=torch.distributed.ReduceOp.SUM)
    tokens_global = torch.tensor([sum(tokens_per_micro_batch)], device=device)
    torch.distributed.all_reduce(tokens_global, op=torch.distributed.ReduceOp.SUM)
    assert tokens_global.item() == tokens_ref, (
        f"[rank {rank}] token accounting mismatch: {tokens_global.item()} != {tokens_ref}"
    )
    grad_global /= tokens_global.item()

    rel = (grad_global - grad_ref).abs().max() / grad_ref.abs().max().clamp_min(1e-12)
    assert rel < 1e-4, f"[rank {rank}] global-token gradient mismatch rel={rel.item():.3e}"

    # The biased contract must NOT match, otherwise this test proves nothing.
    grad_biased = _per_micro_batch_normalized(head, slices, batch)
    torch.distributed.all_reduce(grad_biased, op=torch.distributed.ReduceOp.SUM)
    grad_biased /= world_size
    rel_biased = (grad_biased - grad_ref).abs().max() / grad_ref.abs().max().clamp_min(1e-12)
    assert rel_biased > 1e-2, (
        f"[rank {rank}] per-micro-batch normalization was expected to be biased but matched "
        f"the global reference (rel={rel_biased.item():.3e}); the micro-batch token counts "
        "are probably no longer uneven."
    )
