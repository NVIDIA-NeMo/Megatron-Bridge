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

"""Tiny WAN forward parity check for TP/SP vs non-parallel inference.

This script intentionally uses a tiny latent grid so both the TP=1/SP=off
baseline and TP>1/SP=on path fit on one GPU node.  The transformer block
parameters are zeroed so the test exercises WAN forward plumbing, TP modules,
and sequence-parallel scatter/gather without accepting nondeterministic
attention softmax/reduction differences.  The final comparison is exact:
``torch.equal`` with no tolerance.
"""

from __future__ import annotations

import argparse
import datetime
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider


@dataclass(frozen=True)
class TinyWanShape:
    """Tiny WAN latent and text dimensions used for the parity check."""

    frames: int = 2
    height: int = 2
    width: int = 2
    latent_channels: int = 2
    text_len: int = 4
    text_dim: int = 64

    @property
    def seq_length(self) -> int:
        return self.frames * self.height * self.width


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the parity check."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=4, help="Tensor/SP size for the parallel comparison.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--ffn-hidden-size", type=int, default=128)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def init_dist() -> tuple[int, int, int]:
    """Initialize torch distributed and return rank metadata."""

    if not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    return dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", "0"))


def reset_model_parallel(
    *,
    tp_size: int,
    sequence_parallel: bool,
    seed: int,
) -> None:
    """Reset Megatron model-parallel state for one comparison pass."""

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(seed)
    torch.manual_seed(seed)


def build_provider(
    *,
    shape: TinyWanShape,
    tp_size: int,
    sequence_parallel: bool,
    hidden_size: int,
    ffn_hidden_size: int,
    num_attention_heads: int,
) -> WanModelProvider:
    """Build a tiny WAN provider with the requested parallelism settings."""

    provider = WanModelProvider(
        num_layers=1,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        crossattn_emb_size=hidden_size,
        layernorm_epsilon=1e-6,
        normalization="RMSNorm",
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        qkv_format="sbhd",
        seq_length=shape.seq_length,
        in_channels=shape.latent_channels,
        out_channels=shape.latent_channels,
        patch_spatial=1,
        patch_temporal=1,
        freq_dim=24,
        text_len=shape.text_len,
        text_dim=shape.text_dim,
    )
    provider.tensor_model_parallel_size = tp_size
    provider.pipeline_model_parallel_size = 1
    provider.context_parallel_size = 1
    provider.sequence_parallel = sequence_parallel
    provider.pipeline_dtype = torch.bfloat16
    provider.kv_channels = hidden_size // num_attention_heads
    provider.num_query_groups = num_attention_heads
    provider.finalize()
    return provider


def zero_transformer_make_io_deterministic(model: torch.nn.Module, *, shape: TinyWanShape) -> None:
    """Zero transformer parameters and keep deterministic embedding/head projections."""

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        if getattr(model, "pre_process", False):
            weight = model.patch_embedding.weight
            for hidden_idx in range(weight.size(0)):
                channel_idx = hidden_idx % shape.latent_channels
                weight[hidden_idx, channel_idx, 0, 0, 0] = (hidden_idx + 1) / weight.size(0)

        if getattr(model, "post_process", False):
            head = model.head.head
            for channel_idx in range(shape.latent_channels):
                hidden_idx = channel_idx % head.weight.size(1)
                head.weight[channel_idx, hidden_idx] = 1.0


def make_inputs(shape: TinyWanShape, device: torch.device) -> dict[str, torch.Tensor | dict[str, PackedSeqParams]]:
    """Create deterministic tiny WAN inputs and packed sequence metadata."""

    seq = shape.seq_length
    x = torch.arange(seq * shape.latent_channels, dtype=torch.float32, device=device).reshape(
        seq, 1, shape.latent_channels
    )
    x = (x / 16.0).to(torch.bfloat16)
    timesteps = torch.tensor([0.25], dtype=torch.float32, device=device)
    context = torch.arange(shape.text_len * shape.text_dim, dtype=torch.float32, device=device).reshape(
        shape.text_len, 1, shape.text_dim
    )
    context = (context / 1024.0).to(torch.bfloat16)
    grid_sizes = torch.tensor([[shape.frames, shape.height, shape.width]], dtype=torch.int32, device=device)
    cu_q = torch.tensor([0, seq], dtype=torch.int32, device=device)
    cu_kv = torch.tensor([0, shape.text_len], dtype=torch.int32, device=device)
    packed_seq_params = {
        "self_attention": PackedSeqParams(
            cu_seqlens_q=cu_q,
            cu_seqlens_q_padded=cu_q,
            cu_seqlens_kv=cu_q,
            cu_seqlens_kv_padded=cu_q,
            qkv_format="sbhd",
        ),
        "cross_attention": PackedSeqParams(
            cu_seqlens_q=cu_q,
            cu_seqlens_q_padded=cu_q,
            cu_seqlens_kv=cu_kv,
            cu_seqlens_kv_padded=cu_kv,
            qkv_format="sbhd",
        ),
    }
    return {
        "x": x,
        "grid_sizes": grid_sizes,
        "t": timesteps,
        "context": context,
        "packed_seq_params": packed_seq_params,
    }


def forward_once(
    *,
    shape: TinyWanShape,
    tp_size: int,
    sequence_parallel: bool,
    hidden_size: int,
    ffn_hidden_size: int,
    num_attention_heads: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Run one WAN forward pass under the requested TP/SP settings."""

    reset_model_parallel(tp_size=tp_size, sequence_parallel=sequence_parallel, seed=seed)
    provider = build_provider(
        shape=shape,
        tp_size=tp_size,
        sequence_parallel=sequence_parallel,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
    )
    model = provider.provide().to(device=device, dtype=torch.bfloat16)
    model.eval()
    zero_transformer_make_io_deterministic(model, shape=shape)
    inputs = make_inputs(shape, device)
    with torch.no_grad():
        output = model(**inputs)
    del model
    torch.cuda.empty_cache()
    return output.detach().contiguous()


def main() -> None:
    """Run the exact TP/SP parity comparison."""

    args = parse_args()
    rank, world_size, local_rank = init_dist()
    if args.tp_size > world_size:
        raise ValueError(f"--tp-size={args.tp_size} requires world_size >= {args.tp_size}, got {world_size}")
    if args.num_attention_heads % args.tp_size != 0:
        raise ValueError("--num-attention-heads must be divisible by --tp-size")

    device = torch.device("cuda", local_rank)
    shape = TinyWanShape()

    baseline = forward_once(
        shape=shape,
        tp_size=1,
        sequence_parallel=False,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        seed=args.seed,
        device=device,
    )
    parallel = forward_once(
        shape=shape,
        tp_size=args.tp_size,
        sequence_parallel=True,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        seed=args.seed,
        device=device,
    )

    exact = torch.equal(baseline, parallel)
    max_abs = (baseline - parallel).abs().max()
    exact_int = torch.tensor(int(exact), dtype=torch.int32, device=device)
    dist.all_reduce(exact_int, op=dist.ReduceOp.MIN)
    max_abs_global = max_abs.detach().clone()
    dist.all_reduce(max_abs_global, op=dist.ReduceOp.MAX)

    if rank == 0:
        print(
            "WAN_TINY_SP_TP_PARITY "
            f"world_size={world_size} tp_size={args.tp_size} "
            "baseline_tp=1 baseline_sp=False parallel_sp=True "
            f"latent_shape=[1,{shape.frames},{shape.latent_channels},{shape.height},{shape.width}] "
            f"seq_length={shape.seq_length} output_shape={list(baseline.shape)} "
            f"strict_equal={bool(exact_int.item())} max_abs={max_abs_global.item():.8e}"
        )

    if not bool(exact_int.item()):
        raise AssertionError(
            "Tiny WAN TP/SP parity failed exact tensor comparison: "
            f"rank={rank} local_exact={exact} global_max_abs={max_abs_global.item():.8e}"
        )

    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
