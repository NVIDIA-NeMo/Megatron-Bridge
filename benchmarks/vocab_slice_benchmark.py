#!/usr/bin/env python3
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

"""Benchmark: output vocab slicing for SFT with answer_only_loss.

Demonstrates the speedup from slicing the output projection to only active
vocabulary tokens during supervised fine-tuning.

Creates a toy model with:
- Large vocabulary (151,936 tokens, same as Qwen 2.5)
- Small transformer (~5M params excluding embedding/projection)
- 200 active answer tokens (simulating targeted SFT, e.g. single-language)

Usage:
    python benchmarks/vocab_slice_benchmark.py
    python benchmarks/vocab_slice_benchmark.py --device cpu --iters 10
    python benchmarks/vocab_slice_benchmark.py --n_active 50 --seq_len 4096
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyGPTModel(nn.Module):
    """Minimal causal LM with shared input/output embeddings.

    Designed to exaggerate the output projection cost: large vocab with a small
    transformer backbone, similar to fine-tuning Qwen 2.5 on a narrow task.
    """

    def __init__(
        self,
        vocab_size: int = 151_936,
        hidden_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_size: int = 768,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Shared embedding (used for both input and output projection)
        self.embedding = nn.Embedding(vocab_size, hidden_size)

        # Lightweight transformer backbone
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=ffn_size,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute hidden states from input token IDs.

        Returns:
            Hidden states of shape [B, S, H].
        """
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        active_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute output logits using shared embedding weight.

        Args:
            hidden_states: [B, S, H] from the transformer.
            active_ids: If provided, slice embedding weight to these rows,
                producing [B, S, N] logits instead of [B, S, V].

        Returns:
            Logits tensor of shape [B, S, V] or [B, S, N].
        """
        weight = self.embedding.weight
        if active_ids is not None:
            weight = weight[active_ids]
        return F.linear(hidden_states, weight)


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    remap: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy loss with label remapping and masking.

    Args:
        logits: [B, S, V] or [B, S, N] output logits.
        labels: [B, S] original token IDs.
        loss_mask: [B, S] binary mask (1.0 for answer positions).
        remap: Optional [V] tensor mapping original IDs to sliced indices.

    Returns:
        Scalar loss (masked sum, not averaged).
    """
    if remap is not None:
        labels = remap[labels]

    B, S, V = logits.shape
    loss_per_token = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1), reduction="none").reshape(B, S)

    return (loss_per_token * loss_mask).sum()


def _time_fn(fn, warmup: int, iters: int, device: str) -> float:
    """Time a function with warmup, returning average seconds per call."""
    for _ in range(warmup):
        fn()

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()

    if device == "cuda":
        torch.cuda.synchronize()

    return (time.perf_counter() - t0) / iters


def benchmark(
    device: str = "cuda",
    warmup: int = 5,
    iters: int = 50,
    seq_len: int = 2048,
    batch_size: int = 2,
    vocab_size: int = 151_936,
    n_active: int = 200,
) -> None:
    """Run full benchmark comparing full vocab vs sliced vocab training steps.

    Note: This benchmark re-implements slicing logic with a standalone ToyGPTModel
    rather than importing the production ``vocab_slice`` module, so that it runs
    without the full Megatron-Bridge dependency chain.
    """
    print(f"{'=' * 60}")
    print("Output Vocab Slice Benchmark")
    print(f"{'=' * 60}")
    print(f"Vocab size:     {vocab_size:,}")
    print(f"Active tokens:  {n_active}")
    print(f"Reduction:      {vocab_size / n_active:.0f}x")
    print(f"Seq length:     {seq_len}")
    print(f"Batch size:     {batch_size}")
    print(f"Device:         {device}")
    print()

    # -- Model --
    model = ToyGPTModel(vocab_size=vocab_size, hidden_size=256, num_layers=4, num_heads=4, ffn_size=768).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    embed_params = model.embedding.weight.numel()
    print(f"Total params:       {total_params:>12,}")
    print(f"Embedding params:   {embed_params:>12,}")
    print(f"Transformer params: {total_params - embed_params:>12,}")
    print()

    # -- Synthetic data --
    active_ids = torch.arange(n_active, device=device)
    remap = torch.zeros(vocab_size, dtype=torch.long, device=device)
    remap[active_ids] = torch.arange(n_active, device=device)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = active_ids[torch.randint(0, n_active, (batch_size, seq_len), device=device)]
    loss_mask = torch.zeros(batch_size, seq_len, device=device)
    loss_mask[:, seq_len // 2 :] = 1.0  # answer_only_loss on second half

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ================================================================
    # Correctness: verify sliced logits match full logits for active tokens
    # ================================================================
    print("Correctness check...")
    model.eval()
    with torch.no_grad():
        hidden = model(input_ids)

        logits_full = model.compute_logits(hidden)
        loss_full = compute_loss(logits_full, labels, loss_mask)

        logits_sliced = model.compute_logits(hidden, active_ids)
        loss_sliced = compute_loss(logits_sliced, labels, loss_mask, remap=remap)

        # Sliced logits should exactly match the corresponding rows of full logits
        logits_full_subset = logits_full[:, :, active_ids]
        logit_diff = (logits_full_subset - logits_sliced).abs().max().item()

        print(f"  Max logit difference: {logit_diff:.2e}")
        print(f"  Full vocab loss:      {loss_full.item():.6f}")
        print(f"  Sliced vocab loss:    {loss_sliced.item():.6f}")
        assert logit_diff < 1e-5, f"Logits differ: {logit_diff}"
        print("  Logit check: PASSED (active token logits are identical)")

        # Cross-entropy losses differ because softmax denominator changes
        # (V vs N classes). This is the documented trade-off: fewer distractors
        # in the softmax. In practice, quality is preserved (see issue #2473).
        print("  Note: loss values differ because softmax normalization changes")
        print(f"        (summing over {n_active} vs {vocab_size:,} classes)")
    print()

    # ================================================================
    # Benchmark: full training step (forward + backward + optimizer)
    # ================================================================
    model.train()

    def step_full():
        optimizer.zero_grad()
        hidden = model(input_ids)
        logits = model.compute_logits(hidden)
        loss = compute_loss(logits, labels, loss_mask)
        loss.backward()
        optimizer.step()

    def step_sliced():
        optimizer.zero_grad()
        hidden = model(input_ids)
        logits = model.compute_logits(hidden, active_ids)
        loss = compute_loss(logits, labels, loss_mask, remap=remap)
        loss.backward()
        optimizer.step()

    print(f"Benchmarking full training step ({iters} iters, {warmup} warmup)...")
    t_full = _time_fn(step_full, warmup, iters, device)
    t_sliced = _time_fn(step_sliced, warmup, iters, device)

    print()
    print(f"{'=' * 60}")
    print("Results: Full Training Step")
    print(f"{'=' * 60}")
    print(f"  Full vocab:   {t_full * 1000:>8.1f} ms/step")
    print(f"  Sliced vocab: {t_sliced * 1000:>8.1f} ms/step")
    print(f"  Speedup:      {t_full / max(t_sliced, 1e-9):>8.2f}x")
    print()

    # ================================================================
    # Breakdown: output matmul only
    # ================================================================
    model.eval()
    with torch.no_grad():
        hidden = model(input_ids)

        def matmul_full():
            model.compute_logits(hidden)

        def matmul_sliced():
            model.compute_logits(hidden, active_ids)

        t_mat_full = _time_fn(matmul_full, warmup, iters, device)
        t_mat_sliced = _time_fn(matmul_sliced, warmup, iters, device)

    print("Breakdown: Output Matmul (forward only)")
    print(f"  Full vocab:   {t_mat_full * 1000:>8.2f} ms")
    print(f"  Sliced vocab: {t_mat_sliced * 1000:>8.2f} ms")
    print(f"  Speedup:      {t_mat_full / max(t_mat_sliced, 1e-9):>8.0f}x")
    print()

    # ================================================================
    # Breakdown: cross-entropy only
    # ================================================================
    with torch.no_grad():
        logits_full = model.compute_logits(hidden)
        logits_sliced = model.compute_logits(hidden, active_ids)

        def ce_full():
            compute_loss(logits_full, labels, loss_mask)

        def ce_sliced():
            compute_loss(logits_sliced, labels, loss_mask, remap=remap)

        t_ce_full = _time_fn(ce_full, warmup, iters, device)
        t_ce_sliced = _time_fn(ce_sliced, warmup, iters, device)

    print("Breakdown: Cross-Entropy Loss")
    print(f"  Full vocab:   {t_ce_full * 1000:>8.2f} ms")
    print(f"  Sliced vocab: {t_ce_sliced * 1000:>8.2f} ms")
    print(f"  Speedup:      {t_ce_full / max(t_ce_sliced, 1e-9):>8.0f}x")
    print()

    # ================================================================
    # Memory
    # ================================================================
    if device == "cuda":
        # Measure peak memory for one step
        torch.cuda.reset_peak_memory_stats()
        step_full()
        mem_full = torch.cuda.max_memory_allocated() / 1e6

        torch.cuda.reset_peak_memory_stats()
        step_sliced()
        mem_sliced = torch.cuda.max_memory_allocated() / 1e6

        print("Peak GPU Memory")
        print(f"  Full vocab:   {mem_full:>8.0f} MB")
        print(f"  Sliced vocab: {mem_sliced:>8.0f} MB")
        print(f"  Saved:        {mem_full - mem_sliced:>8.0f} MB ({(mem_full - mem_sliced) / max(mem_full, 1e-9) * 100:.0f}%)")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark output vocab slicing for SFT")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=151_936, help="Full vocabulary size")
    parser.add_argument("--n_active", type=int, default=200, help="Number of active answer tokens")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    benchmark(
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        n_active=args.n_active,
    )
