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

"""
Compare logits between TP=1 and TP=4 using real HuggingFace model weights
loaded via AutoBridge.

Launches with torchrun --nproc_per_node=4, runs a single forward pass at
two different TP sizes, and prints detailed numerical comparison of the
resulting logits.

Usage:
    # Default: Qwen3-1.7B, bfloat16
    uv run python -m torch.distributed.run --nproc_per_node=4 \\
        examples/tp_investigation/compare_tp_logits.py

    # Custom model and precision
    uv run python -m torch.distributed.run --nproc_per_node=4 \\
        examples/tp_investigation/compare_tp_logits.py \\
        --hf-model-id Qwen/Qwen3-4B --dtype float32

    # Custom TP sizes
    uv run python -m torch.distributed.run --nproc_per_node=4 \\
        examples/tp_investigation/compare_tp_logits.py \\
        --tp-baseline 1 --tp-compare 2
"""

import argparse
import gc

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from transformers import AutoTokenizer

from megatron.bridge import AutoBridge
from megatron.bridge.utils.common_utils import get_local_rank_preinit, print_rank_0


class SingleBatchIterator:
    """Iterator that yields a single batch of data for a forward pass.

    Required by get_forward_backward_func. Yields exactly one batch
    containing input tokens and position IDs, then raises StopIteration.
    """

    def __init__(self, input_ids: torch.Tensor, position_ids: torch.Tensor):
        self.batch = dict(
            tokens=input_ids,
            position_ids=position_ids,
        )
        self._yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.batch


def text_forward_step(data_iterator: SingleBatchIterator, model, **kwargs) -> torch.Tensor:
    """Forward step function for single-step inference.

    Required by get_forward_backward_func. Extracts a batch from the
    data iterator and runs the model forward pass.
    """
    batch = next(data_iterator)
    forward_args = {
        "input_ids": batch["tokens"],
        "position_ids": batch["position_ids"],
        "attention_mask": batch.get("attention_mask", None),
    }

    def loss_func(x, **kwargs):
        return x

    return model(**forward_args), loss_func


def run_forward_pass(
    hf_model_id: str,
    prompt: str,
    tp: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Load a model via AutoBridge and run a single forward pass.

    Args:
        hf_model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-1.7B")
        prompt: Input text to tokenize and feed to the model
        tp: Tensor parallelism size
        dtype: Model precision (torch.bfloat16 or torch.float32)

    Returns:
        Full logits tensor [1, seq_len, vocab_size] on rank 0 (None on other ranks
        when not on the pipeline last stage).
    """
    print_rank_0(f"\n{'=' * 60}")
    print_rank_0(f"Running forward pass with TP={tp}, dtype={dtype}")
    print_rank_0(f"{'=' * 60}")

    # Load model via AutoBridge
    print_rank_0(f"Loading model: {hf_model_id}")
    bridge = AutoBridge.from_hf_pretrained(hf_model_id, trust_remote_code=True)
    model_provider = bridge.to_megatron_provider(load_weights=True)

    # Configure parallelism
    model_provider.tensor_model_parallel_size = tp
    model_provider.pipeline_model_parallel_size = 1
    model_provider.pipeline_dtype = dtype

    # Finalize and initialize parallel state
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)

    # Build the distributed model
    model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    # TEMP FIX for inference failure when mtp_num_layers is not None
    for m in model:
        m.config.mtp_num_layers = None

    model = [m.cuda() for m in model]
    for m in model:
        m.eval()

    # Tokenize prompt
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    seq_len = input_ids.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand_as(input_ids)

    print_rank_0(f"Input tokens: {input_ids.shape} = {input_ids.tolist()}")

    # Single forward pass
    logits = None
    with torch.no_grad():
        iterator = SingleBatchIterator(input_ids, position_ids)
        fwd_bwd_function = get_forward_backward_func()

        output = fwd_bwd_function(
            forward_step_func=text_forward_step,
            data_iterator=iterator,
            model=model,
            num_microbatches=1,
            forward_only=True,
            seq_length=seq_len,
            micro_batch_size=1,
            collect_non_loss_data=True,
        )

        if isinstance(output, list) and len(output) > 0:
            output = output[0]

        # Gather TP-sharded logits on last pipeline stage
        if parallel_state.is_pipeline_last_stage():
            tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
            gathered_tensors = [torch.zeros_like(output) for _ in range(tp_world_size)]
            dist.all_gather(gathered_tensors, output, group=parallel_state.get_tensor_model_parallel_group())
            logits = torch.cat(gathered_tensors, dim=2)  # [batch, seq, vocab]
            print_rank_0(f"Output logits shape: {logits.shape}")

    # Clean up: destroy parallel state and free GPU memory
    parallel_state.destroy_model_parallel()
    del model, model_provider, bridge
    gc.collect()
    torch.cuda.empty_cache()

    # Return logits from rank 0 (which is on pipeline last stage with PP=1)
    if logits is not None:
        return logits.cpu()
    return None


def compare_logits(logits_a: torch.Tensor, logits_b: torch.Tensor, tp_a: int, tp_b: int) -> None:
    """Compare two logit tensors and print detailed numerical analysis.

    Args:
        logits_a: Baseline logits [1, seq_len, vocab_size]
        logits_b: Comparison logits [1, seq_len, vocab_size]
        tp_a: Baseline TP size
        tp_b: Comparison TP size
    """
    assert logits_a.shape == logits_b.shape, f"Shape mismatch: {logits_a.shape} vs {logits_b.shape}"

    # Work in float32 for comparison accuracy
    a = logits_a.float()
    b = logits_b.float()
    seq_len = a.shape[1]
    vocab_size = a.shape[2]

    diff = (a - b).squeeze(0)  # [seq_len, vocab_size]
    abs_diff = diff.abs()

    print(f"\n{'=' * 70}")
    print(f"LOGIT COMPARISON: TP={tp_a} vs TP={tp_b}")
    print(f"Shape: {logits_a.shape} (batch=1, seq_len={seq_len}, vocab={vocab_size})")
    print(f"{'=' * 70}")

    # Overall statistics
    print("\n--- Overall Statistics ---")
    print(f"  Max absolute difference:    {abs_diff.max().item():.6f}")
    print(f"  Mean absolute difference:   {abs_diff.mean().item():.6f}")
    print(f"  Median absolute difference: {abs_diff.median().item():.6f}")
    print(f"  Std of absolute difference: {abs_diff.std().item():.6f}")

    # Cosine similarity per position and overall
    print("\n--- Cosine Similarity ---")
    cos_sims = []
    for pos in range(seq_len):
        cos_sim = torch.cosine_similarity(a[0, pos], b[0, pos], dim=0).item()
        cos_sims.append(cos_sim)
    cos_sims_tensor = torch.tensor(cos_sims)
    print(f"  Overall (flattened):  {torch.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item():.8f}")
    print(f"  Per-position min:     {cos_sims_tensor.min().item():.8f}")
    print(f"  Per-position mean:    {cos_sims_tensor.mean().item():.8f}")
    print(f"  Per-position max:     {cos_sims_tensor.max().item():.8f}")

    # Threshold analysis
    print("\n--- Elements Exceeding Thresholds (logits) ---")
    total_elements = abs_diff.numel()
    thresholds = [1e-4, 1e-3, 1e-2, 1e-1]
    for thresh in thresholds:
        count = (abs_diff > thresh).sum().item()
        pct = 100.0 * count / total_elements
        print(f"  > {thresh:.0e}: {count:>12,} / {total_elements:,} ({pct:6.2f}%)")

    # ==================== Log-Probability Error ====================
    # Following the methodology:
    #   1. logits → log-probs via log_softmax (numerically stable)
    #   2. Chosen-token logprob diff: Δᵢ = ℓᴬ(tᵢ) - ℓᴮ(tᵢ)
    #   3. Summary stats: mean|Δ|, max|Δ|, signed mean (bias), std
    #   4. Multiplicative prob error: (1/n) Σ exp(|Δᵢ|)
    #   5. Full-distribution KL divergence

    log_probs_a = torch.nn.functional.log_softmax(a.squeeze(0), dim=-1)  # [seq_len, vocab]
    log_probs_b = torch.nn.functional.log_softmax(b.squeeze(0), dim=-1)
    probs_a = log_probs_a.exp()

    # Chosen token = argmax from system A (baseline) at each position
    top1_a = a.squeeze(0).argmax(dim=-1)  # [seq_len] — reused below

    # Δᵢ = ℓᴬ(tᵢ) - ℓᴮ(tᵢ) for the chosen token at each position
    chosen_lp_a = log_probs_a[torch.arange(seq_len), top1_a]  # [seq_len]
    chosen_lp_b = log_probs_b[torch.arange(seq_len), top1_a]
    delta = chosen_lp_a - chosen_lp_b  # [seq_len], signed

    print("\n--- Chosen-Token Log-Prob Error (Δᵢ = logp_A(tᵢ) - logp_B(tᵢ)) ---")
    print(f"  Chosen token: argmax of TP={tp_a} (baseline)")
    print(f"  mean |Δ|:     {delta.abs().mean().item():.8f}")
    print(f"  max  |Δ|:     {delta.abs().max().item():.8f}")
    print(f"  signed mean:  {delta.mean().item():+.8f}  (bias)")
    print(f"  std:          {delta.std().item():.8f}")

    # Multiplicative probability error: (1/n) Σ exp(|Δᵢ|)
    # Interpretation: on average, how many × off is the token probability
    mult_prob_err = delta.abs().exp().mean().item()
    print(f"  mult prob err: {mult_prob_err:.8f}x  (avg p_A/p_B ratio)")

    # Per-position KL divergence: KL(A || B) = Σ_v p_A(v) * (ℓ_A(v) - ℓ_B(v))
    probs_b = log_probs_b.exp()
    kl_ab = (probs_a * (log_probs_a - log_probs_b)).sum(dim=-1)  # [seq_len]
    kl_ba = (probs_b * (log_probs_b - log_probs_a)).sum(dim=-1)
    # Total variation distance: TV = 0.5 * Σ|p_A - p_B|
    tv_dist = 0.5 * (probs_a - probs_b).abs().sum(dim=-1)  # [seq_len]

    print("\n--- Full-Distribution Divergence (per position) ---")
    print(f"  KL(TP={tp_a} || TP={tp_b}):  mean={kl_ab.mean().item():.8f}  max={kl_ab.max().item():.8f}")
    print(f"  KL(TP={tp_b} || TP={tp_a}):  mean={kl_ba.mean().item():.8f}  max={kl_ba.max().item():.8f}")
    print(f"  Total variation dist:  mean={tv_dist.mean().item():.8f}  max={tv_dist.max().item():.8f}")

    # Per-position breakdown table
    print("\n--- Per-Position Log-Prob Detail ---")
    print(
        f"{'Pos':>4} | {'Token':>8} | {'logp_A':>10} | {'logp_B':>10} | {'delta':>11} | {'|delta|':>10} | {'p_A/p_B':>9} | {'KL(A||B)':>12} | {'TV':>10}"
    )
    print(
        f"{'-' * 4}-+-{'-' * 8}-+-{'-' * 10}-+-{'-' * 10}-+-{'-' * 11}-+-{'-' * 10}-+-{'-' * 9}-+-{'-' * 12}-+-{'-' * 10}"
    )
    for pos in range(seq_len):
        tok = top1_a[pos].item()
        lp_a = chosen_lp_a[pos].item()
        lp_b = chosen_lp_b[pos].item()
        d = delta[pos].item()
        ratio = torch.exp(torch.tensor(abs(d))).item()
        print(
            f"{pos:>4} | {tok:>8} | {lp_a:>10.6f} | {lp_b:>10.6f} | {d:>+11.8f} | {abs(d):>10.8f}"
            f" | {ratio:>9.6f} | {kl_ab[pos].item():>12.8f} | {tv_dist[pos].item():>10.8f}"
        )

    # Top-k agreement
    print("\n--- Top-k Token Agreement ---")
    top1_b = b.squeeze(0).argmax(dim=-1)
    top1_agree = (top1_a == top1_b).float().mean().item()
    print(f"  Top-1 agreement: {top1_agree * 100:.1f}% ({(top1_a == top1_b).sum().item()}/{seq_len} positions)")

    top5_a = a.squeeze(0).topk(5, dim=-1).indices  # [seq_len, 5]
    top5_b = b.squeeze(0).topk(5, dim=-1).indices
    top5_agree_count = 0
    for pos in range(seq_len):
        set_a = set(top5_a[pos].tolist())
        set_b = set(top5_b[pos].tolist())
        top5_agree_count += len(set_a & set_b)
    top5_agree_pct = 100.0 * top5_agree_count / (seq_len * 5)
    print(f"  Top-5 agreement: {top5_agree_pct:.1f}% ({top5_agree_count}/{seq_len * 5} token slots)")

    # Per-position breakdown
    print("\n--- Per-Position Breakdown ---")
    print(f"{'Pos':>4} | {'Max Abs Diff':>13} | {'Mean Abs Diff':>14} | {'Cosine Sim':>11} | {'Top-1 Match':>11}")
    print(f"{'-' * 4}-+-{'-' * 13}-+-{'-' * 14}-+-{'-' * 11}-+-{'-' * 11}")
    for pos in range(seq_len):
        pos_diff = abs_diff[pos]
        max_d = pos_diff.max().item()
        mean_d = pos_diff.mean().item()
        cos_s = cos_sims[pos]
        t1_match = "YES" if top1_a[pos] == top1_b[pos] else "NO"
        print(f"{pos:>4} | {max_d:>13.6f} | {mean_d:>14.8f} | {cos_s:>11.8f} | {t1_match:>11}")

    # Show top-1 token differences
    mismatches = (top1_a != top1_b).nonzero(as_tuple=True)[0]
    if len(mismatches) > 0:
        print(f"\n--- Top-1 Token Mismatches ({len(mismatches)} positions) ---")
        print(
            f"{'Pos':>4} | {'TP={} token':>12} | {'TP={} token':>12} | {'TP={} logit':>12} | {'TP={} logit':>12} | {'Diff':>10}".format(
                tp_a, tp_b, tp_a, tp_b
            )
        )
        print(f"{'-' * 4}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")
        for pos in mismatches[:20]:  # Show at most 20
            pos = pos.item()
            tok_a = top1_a[pos].item()
            tok_b = top1_b[pos].item()
            val_a = a[0, pos, tok_a].item()
            val_b = b[0, pos, tok_b].item()
            print(f"{pos:>4} | {tok_a:>12} | {tok_b:>12} | {val_a:>12.4f} | {val_b:>12.4f} | {val_a - val_b:>10.4f}")

    print(f"\n{'=' * 70}")


def main():
    """Parse args, run forward passes at two TP sizes, compare logits."""
    parser = argparse.ArgumentParser(
        description="Compare logits between different TP sizes using real HF model weights via AutoBridge"
    )
    parser.add_argument(
        "--hf-model-id", type=str, default="Qwen/Qwen3-1.7B", help="HuggingFace model ID (default: Qwen/Qwen3-1.7B)"
    )
    parser.add_argument(
        "--prompt", type=str, default="The capital of France is", help="Input text for the forward pass"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Model precision (default: bfloat16)",
    )
    parser.add_argument("--tp-baseline", type=int, default=1, help="Baseline tensor parallelism size (default: 1)")
    parser.add_argument("--tp-compare", type=int, default=4, help="Comparison tensor parallelism size (default: 4)")
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Initialize distributed before any dist.* calls
    if not dist.is_initialized():
        torch.cuda.set_device(get_local_rank_preinit())
        dist.init_process_group("nccl")

    rank = dist.get_rank()

    print_rank_0(f"Model: {args.hf_model_id}")
    print_rank_0(f"Prompt: '{args.prompt}'")
    print_rank_0(f"Dtype: {args.dtype}")
    print_rank_0(f"Comparing TP={args.tp_baseline} vs TP={args.tp_compare}")
    print_rank_0(f"World size: {dist.get_world_size()}")

    # Run baseline (e.g. TP=1)
    logits_baseline = run_forward_pass(args.hf_model_id, args.prompt, args.tp_baseline, dtype)

    # Run comparison (e.g. TP=4)
    logits_compare = run_forward_pass(args.hf_model_id, args.prompt, args.tp_compare, dtype)

    # Compare on rank 0
    if rank == 0 and logits_baseline is not None and logits_compare is not None:
        compare_logits(logits_baseline, logits_compare, args.tp_baseline, args.tp_compare)
    elif rank == 0:
        print("ERROR: Failed to collect logits on rank 0")

    # Clean up distributed
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
