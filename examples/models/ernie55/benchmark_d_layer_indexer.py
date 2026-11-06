#!/usr/bin/env python3
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

"""
Measure the indexer top-k cost of the ERNIE 5.5 D layer on the fused cuDNN path (one GPU).

Without packed metadata (SBHD batches), ERNIE's remote candidate bounds ``[doc_start, t - W]``
are not causal, so Megatron-Core's cuDNN DSA backend scores them with its generic
``UNFUSED_BOUNDS`` plan (fp32 per-head score chunks over the full ``T x T`` rectangle). This
script times Megatron-Core's top-k hooks with explicit bounds, for ERNIE bounds and for causal
document bounds, at several pack lengths. With a single document (``--num-docs 1``) the causal row is the
cost of Megatron-Core's fused causal scorer, so the gap between the two rows is the performance
cliff. That comparison needs Megatron-Core dev: only dev's hooks accept ``varlen_is_plain_causal``.
On Megatron-Core main the causal row also takes the generic chunked scorer, and the script says so.

Packed THD training rows take other paths, which this script does not time: at context_parallel_size 1 the D layer
selects its top-k with the reference implementation's cuDNN score call and calls only Megatron-Core's loss hook, and
at every context_parallel_size the hooks get packed-THD arguments that select ``PACKED_THD_GENERAL`` (see
``latent_mqa_core``).

Reported per pack length: milliseconds per call, the dense-rectangle FLOPs
``2 * T^2 * heads * head_dim`` and the resulting TFLOP/s, and the peak allocated memory.

Usage:

    uv run python examples/models/ernie55/benchmark_d_layer_indexer.py \
        --seq-lens 4096 8192 16384 32768 --num-docs 1

    # Include the sparse indexer loss (adds the FlashMLA teacher pass):
    uv run python examples/models/ernie55/benchmark_d_layer_indexer.py --with-loss

Requires CUDA (SM90+), cudnn-frontend with the DSA kernels and, for ``--with-loss``, FlashMLA.
The first fused call in a process is a warm-up; long packs (> 32768 rows) should also be
checked against the reference selector for the cuDNN indexer first-call contract.
"""

from __future__ import annotations

import argparse
import inspect
from types import SimpleNamespace

import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import dsa_kernels
from rich.console import Console
from rich.table import Table

from megatron.bridge.models.ernie.modeling_ernie55.ops import (
    ernie55_document_starts,
    ernie55_kernel_selection_ends,
    ernie55_selection_bounds,
)


console = Console()
_BLOCK_SIZE = 8192


def _packed_seq_params(seq_len: int, num_docs: int, device: torch.device) -> PackedSeqParams:
    """THD metadata for ``num_docs`` equal documents covering ``seq_len`` rows."""
    bounds = torch.linspace(0, seq_len, num_docs + 1, device=device).round().to(torch.int32)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=bounds,
        cu_seqlens_kv=bounds,
        max_seqlen_q=seq_len,
        max_seqlen_kv=seq_len,
    )


def _bounds(seq_len: int, num_docs: int, window: int, device: torch.device) -> dict[str, tuple[torch.Tensor, ...]]:
    """ERNIE kernel bounds and plain causal document bounds, both int32 ``[T]``."""
    doc_start = ernie55_document_starts(_packed_seq_params(seq_len, num_docs, device), seq_len, device)
    starts, ends = ernie55_selection_bounds(doc_start, window)
    causal_ends = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device)
    return {
        "ernie": (starts, ernie55_kernel_selection_ends(starts=starts, ends=ends)),
        "causal": (doc_start.to(torch.int32), causal_ends),
    }


def _time(fn, warmup: int, iters: int) -> tuple[float, float]:
    """Mean milliseconds per call and peak allocated GiB."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters, torch.cuda.max_memory_allocated() / 2**30


def _inputs(seq_len: int, args: argparse.Namespace, device: torch.device) -> tuple[tuple, dict]:
    """Random bf16 indexer inputs ``(q, k, w)`` and, with ``--with-loss``, the loss keyword arguments."""
    generator = torch.Generator(device=device).manual_seed(seq_len)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, device=device, generator=generator).to(torch.bfloat16)

    heads, dim = args.indexer_heads, args.indexer_head_dim
    indexer_inputs = (randn(seq_len, 1, heads, dim), randn(seq_len, 1, dim), randn(seq_len, 1, heads))
    loss_kwargs = {}
    if args.with_loss:
        loss_kwargs = dict(
            query=randn(seq_len, 1, args.attention_heads, 576),
            key=randn(seq_len, 1, 1, 576),
            # ERNIE's logit scale: 1 / sqrt(qk_head_dim + qk_pos_emb_head_dim) = 1 / sqrt(448 + 64).
            softmax_scale=512**-0.5,
            loss_coeff=0.01,
            pg_collection=SimpleNamespace(tp=None),
            query_valid_rows=None,
            calculate_per_token_loss=False,
        )
    return indexer_inputs, loss_kwargs


def main() -> None:
    """Parse arguments, run the hook for each pack length and bound type, print a table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--num-docs", type=int, default=1, help="Equal-length documents per THD pack.")
    parser.add_argument("--indexer-heads", type=int, default=64)
    parser.add_argument("--indexer-head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--attention-heads", type=int, default=64, help="Only used with --with-loss.")
    parser.add_argument("--with-loss", action="store_true", help="Time run_fused_qk_topk_with_loss instead.")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs a CUDA device.")
    device = torch.device("cuda", torch.cuda.current_device())
    config = SimpleNamespace(dsa_kernel_backend="cudnn", kv_lora_rank=512)
    hook = dsa_kernels.run_fused_qk_topk_with_loss if args.with_loss else dsa_kernels.run_fused_qk_topk
    supports_plain_causal = "varlen_is_plain_causal" in inspect.signature(hook).parameters
    if not supports_plain_causal:
        console.print(
            "[yellow]This Megatron-Core build has no varlen_is_plain_causal hook argument (main): the causal "
            "row also uses the generic chunked scorer. Run on Megatron-Core dev to measure the gap.[/yellow]"
        )

    table = Table(title=f"ERNIE 5.5 D-layer indexer ({hook.__name__}, {torch.cuda.get_device_name(device)})")
    for column in ("T", "bounds", "ms/call", "dense TFLOP", "TFLOP/s", "peak GiB"):
        table.add_column(column, justify="right")

    for seq_len in args.seq_lens:
        indexer_inputs, loss_kwargs = _inputs(seq_len, args, device)
        dense_tflop = 2.0 * seq_len * seq_len * args.indexer_heads * args.indexer_head_dim / 1e12
        for name, (starts, ends) in _bounds(seq_len, args.num_docs, args.window, device).items():
            plain_causal = name == "causal" and args.num_docs == 1 and supports_plain_causal
            extra = {"varlen_is_plain_causal": True} if plain_causal else {}

            def call(starts=starts, ends=ends, extra=extra, label=name, inputs=indexer_inputs, kwargs=loss_kwargs):
                result = hook(config, *inputs, args.topk, starts, ends, _BLOCK_SIZE, **kwargs, **extra)
                if result is None:
                    raise RuntimeError(f"{hook.__name__} declined the {label} bounds.")
                return result

            ms, peak = _time(call, args.warmup, args.iters)
            table.add_row(
                str(seq_len), name, f"{ms:.2f}", f"{dense_tflop:.2f}", f"{dense_tflop / (ms / 1e3):.1f}", f"{peak:.2f}"
            )
        del indexer_inputs, loss_kwargs
        torch.cuda.empty_cache()
    console.print(table)


if __name__ == "__main__":
    main()
