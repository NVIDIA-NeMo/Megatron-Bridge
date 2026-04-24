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

"""Pure-PyTorch reference port of DeepSeek-V4-Flash `inference/model.py`.

Parameter layout — including names, shapes, and dtypes — is byte-identical to
the upstream inference script so that `state_dict()` keys match the HF
safetensors shards exactly. This is used as the "HF side" for weight-mapping
verification.

The forward path is simplified: custom CUDA kernels (`fp8_gemm`, `fp4_gemm`,
`sparse_attn`, `hc_split_sinkhorn`) are replaced with straightforward PyTorch
equivalents. Numerical parity with the real inference is NOT a goal here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DeepSeekV4ModelArgs:
    """Mirrors `ModelArgs` in inference/model.py. Field names match HF config keys."""

    max_batch_size: int = 4
    max_seq_len: int = 4096
    dtype: Literal["bf16", "fp8"] = "bf16"
    scale_fmt: Literal[None, "ue8m0"] = None
    expert_dtype: Literal[None, "fp4"] = None
    scale_dtype: Literal["fp32", "fp8"] = "fp32"
    vocab_size: int = 129280
    dim: int = 4096
    moe_inter_dim: int = 4096
    n_layers: int = 7
    n_hash_layers: int = 0
    n_mtp_layers: int = 1
    n_heads: int = 64
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # mla + grouped-O
    q_lora_rank: int = 1024
    head_dim: int = 512
    rope_head_dim: int = 64
    norm_eps: float = 1e-6
    o_groups: int = 8
    o_lora_rank: int = 1024
    window_size: int = 128
    compress_ratios: Tuple[int, ...] = field(default_factory=lambda: (0, 0, 4, 128, 4, 128, 4, 0))
    # yarn / rope
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # indexer
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    # hyper-connections
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6


class Linear(nn.Module):
    """Plain dense Linear. No quantization. Optionally carries a `scale` param so
    state_dict keys match the FP8 checkpoint (present for quantized layers in
    the real model; elided for BF16 layers like the Compressor)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        emit_scale: bool = True,
        block_size: int = 128,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if emit_scale:
            scale_rows = (out_features + block_size - 1) // block_size
            scale_cols = (in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(torch.empty(scale_rows, scale_cols, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - toy path
        return F.linear(x, self.weight.to(x.dtype), self.bias)


class RMSNorm(nn.Module):
    """Root-mean-square normalization; same as inference/model.py."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        return (self.weight * x * torch.rsqrt(var + self.eps)).to(dtype)


class Compressor(nn.Module):
    """Learned KV compressor (wkv + wgate + ape + norm). BF16 params, no `.scale`."""

    def __init__(self, args: DeepSeekV4ModelArgs, compress_ratio: int = 4, head_dim: int = 512) -> None:
        super().__init__()
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.overlap = compress_ratio == 4
        coff = 1 + self.overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32))
        self.wkv = Linear(args.dim, coff * head_dim, dtype=torch.float32, emit_scale=False)
        self.wgate = Linear(args.dim, coff * head_dim, dtype=torch.float32, emit_scale=False)
        self.norm = RMSNorm(head_dim, args.norm_eps)


class Indexer(nn.Module):
    """Sparse-attention top-k indexer. Only present when compress_ratio == 4."""

    def __init__(self, args: DeepSeekV4ModelArgs, compress_ratio: int = 4) -> None:
        super().__init__()
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        # wq_b is quantized (has .scale); weights_proj is bf16 (no .scale)
        self.wq_b = Linear(args.q_lora_rank, self.n_heads * self.head_dim, emit_scale=True)
        self.weights_proj = Linear(args.dim, self.n_heads, dtype=torch.bfloat16, emit_scale=False)
        self.compressor = Compressor(args, compress_ratio, self.head_dim)


class Attention(nn.Module):
    """Per-layer attention: low-rank Q (wq_a→q_norm→wq_b), single KV proj (wkv→kv_norm),
    grouped low-rank O (wo_a→wo_b). Optional Compressor + Indexer per-layer."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.n_groups = args.o_groups
        self.compress_ratio = args.compress_ratios[layer_id]

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        self.wq_a = Linear(args.dim, args.q_lora_rank, emit_scale=True)
        self.q_norm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.wq_b = Linear(args.q_lora_rank, self.n_heads * self.head_dim, emit_scale=True)
        self.wkv = Linear(args.dim, self.head_dim, emit_scale=True)
        self.kv_norm = RMSNorm(self.head_dim, args.norm_eps)
        # wo_a lives as [n_heads*head_dim/n_groups, n_groups*o_lora_rank] per inference impl
        self.wo_a = Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
            emit_scale=True,
        )
        self.wo_b = Linear(self.n_groups * args.o_lora_rank, args.dim, emit_scale=True)

        if self.compress_ratio:
            self.compressor = Compressor(args, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(args, self.compress_ratio)


class Gate(nn.Module):
    """Routing gate. Hash layers use a lookup `tid2eid`; score layers have a `bias`."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.hash = layer_id < args.n_hash_layers
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim, dtype=torch.bfloat16))
        if self.hash:
            self.tid2eid = nn.Parameter(
                torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32),
                requires_grad=False,
            )
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))
            self.register_parameter("tid2eid", None)


class Expert(nn.Module):
    """SwiGLU expert. w1=gate, w3=up, w2=down. FP4/FP8 weights in real ckpt."""

    def __init__(self, dim: int, inter_dim: int, *, emit_scale: bool = True) -> None:
        super().__init__()
        self.w1 = Linear(dim, inter_dim, emit_scale=emit_scale)
        self.w2 = Linear(inter_dim, dim, emit_scale=emit_scale)
        self.w3 = Linear(dim, inter_dim, emit_scale=emit_scale)


class MoE(nn.Module):
    """Mixture-of-experts: routing gate + per-rank experts + one shared expert."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.gate = Gate(layer_id, args)
        # Experts are FP4 in real ckpt; we use BF16 here but keep `.scale` so the
        # state_dict-key shape matches.
        self.experts = nn.ModuleList(
            [Expert(args.dim, args.moe_inter_dim, emit_scale=True) for _ in range(args.n_routed_experts)]
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, emit_scale=True)


class Block(nn.Module):
    """Transformer block with Hyper-Connections (HC) pre/post mixing params."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.attn = Attention(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.attn_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)
        hc_mult = args.hc_mult
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))


class MTPBlock(Block):
    """MTP head: adds e_proj/h_proj, enorm/hnorm/norm, and hc_head_* HC params."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__(layer_id, args)
        self.e_proj = Linear(args.dim, args.dim, emit_scale=True)
        self.h_proj = Linear(args.dim, args.dim, emit_scale=True)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.norm = RMSNorm(args.dim, args.norm_eps)
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))


class DeepSeekV4Transformer(nn.Module):
    """Top-level model. Mirrors inference `Transformer` class; parameter names in
    `state_dict()` match the HF safetensors keys byte-for-byte."""

    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed = nn.Parameter(torch.empty(args.vocab_size, args.dim, dtype=torch.bfloat16))
        # Rename trick: original code uses `self.embed = ParallelEmbedding(...)`; its
        # state_dict key is `embed.weight`. We expose the bare Parameter `embed` to
        # make the top-level module's state_dict key just `embed`. Downstream, the
        # actual HF key is `embed.weight`; we remap in the bridge.
        #
        # To keep parity with inference state_dict (which uses `embed.weight`), we
        # wrap the bare parameter in a tiny sub-module.
        del self.embed  # drop the bare param we just created
        self.embed = _EmbedModule(args.vocab_size, args.dim)

        self.layers = nn.ModuleList([Block(i, args) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.head = _HeadModule(args.vocab_size, args.dim)
        self.mtp = nn.ModuleList([MTPBlock(args.n_layers + i, args) for i in range(args.n_mtp_layers)])

        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))


class _EmbedModule(nn.Module):
    """Emits `embed.weight` to match the HF key."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))


class _HeadModule(nn.Module):
    """Emits `head.weight` to match the HF key. FP32 in ckpt."""

    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))
