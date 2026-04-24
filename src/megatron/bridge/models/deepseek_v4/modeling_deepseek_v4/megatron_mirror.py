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

"""Megatron-side mirror of DeepSeek-V4-Flash.

This is a pure-parameter reference used for verifying that the bridge's
`mapping_registry()` emits keys that a real Megatron V4 model can absorb.
Conventions that differ from `reference.py`:

* All parameters are BF16 — `.scale` sidecars are absent. The bridge's
  ``maybe_modify_loaded_hf_weight`` dequantizes FP8/FP4 to BF16 on load.
* MoE expert FCs are fused gate+up (`linear_fc1.weight [2·inter, hidden]`)
  and stored per-expert under indexed names `linear_fc1.weight0, weight1, …`
  so they line up with the bridge's wildcard `linear_fc1.weight*` pattern.
* Block HC uses mcore `HyperConnectionModule`-compatible layout
  (`mapping_proj.weight [n²+2n, n·C]`, `bias [n²+2n]`, `alpha [3]`).
"""

from __future__ import annotations

import torch
from torch import nn

from megatron.bridge.models.deepseek_v4.modeling_deepseek_v4.reference import (
    DeepSeekV4ModelArgs,
    RMSNorm,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _BF16Linear(nn.Module):
    """BF16 Linear weight holder. No `.scale` — BF16 everywhere after dequant."""

    def __init__(self, in_features: int, out_features: int, *, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))


class _IndexedBF16Linear(nn.Module):
    """Per-expert indexed Linears. State_dict keys are `weight0`, `weight1`, …

    Matches the bridge's wildcard pattern
    `decoder.layers.*.mlp.experts.linear_fc1.weight*`.
    """

    def __init__(self, num_experts: int, in_features: int, out_features: int) -> None:
        super().__init__()
        for i in range(num_experts):
            self.register_parameter(
                f"weight{i}",
                nn.Parameter(torch.empty(out_features, in_features, dtype=torch.bfloat16)),
            )


# ---------------------------------------------------------------------------
# HC — mcore-compatible block-level layout
# ---------------------------------------------------------------------------


class _MegatronHC(nn.Module):
    """mcore `HyperConnectionModule`-compatible layout.

    `alpha` is a single [3] tensor ( `(pre, post, res)` ). Forward code that
    uses real mcore `HyperConnectionModule` splits `alpha` at construction
    time: `alpha[0:1]`, `alpha[1:2]`, `alpha[2:3]`.
    """

    def __init__(self, n: int, hidden_size: int) -> None:
        super().__init__()
        mix = n * n + 2 * n
        self.mapping_proj = nn.Linear(n * hidden_size, mix, bias=False)
        self.alpha = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(mix, dtype=torch.float32))


# ---------------------------------------------------------------------------
# HCA — Head-grouped Compressed Attention (grouped low-rank O)
# ---------------------------------------------------------------------------


class HeadGroupedCompressedAttention(nn.Module):
    """V4 grouped low-rank O-projection."""

    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.linear_o_down_proj = _BF16Linear(
            args.n_heads * args.head_dim // args.o_groups,
            args.o_groups * args.o_lora_rank,
        )
        self.linear_o_up_proj = _BF16Linear(args.o_groups * args.o_lora_rank, args.dim)


# ---------------------------------------------------------------------------
# CSA — Compressed Sparse Attention
# ---------------------------------------------------------------------------


class _Compressor(nn.Module):
    """wkv / wgate / ape / norm. Already BF16 in the real checkpoint."""

    def __init__(self, args: DeepSeekV4ModelArgs, compress_ratio: int, head_dim: int) -> None:
        super().__init__()
        overlap = compress_ratio == 4
        coff = 1 + overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32))
        self.wkv = _BF16Linear(args.dim, coff * head_dim, dtype=torch.float32)
        self.wgate = _BF16Linear(args.dim, coff * head_dim, dtype=torch.float32)
        self.norm = RMSNorm(head_dim, args.norm_eps)


class _Indexer(nn.Module):
    """Top-k sparse-attention indexer (compress_ratio == 4 only)."""

    def __init__(self, args: DeepSeekV4ModelArgs, compress_ratio: int) -> None:
        super().__init__()
        self.linear_q_up_proj = _BF16Linear(args.q_lora_rank, args.index_n_heads * args.index_head_dim)
        self.weights_proj = _BF16Linear(args.dim, args.index_n_heads)
        self.compressor = _Compressor(args, compress_ratio, args.index_head_dim)


class CompressedSparseAttention(nn.Module):
    """V4 attention block: low-rank Q + single KV + HCA O + optional Compressor / Indexer."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        compress_ratio = args.compress_ratios[layer_id]

        self.attn_sink = nn.Parameter(torch.empty(args.n_heads, dtype=torch.float32))
        self.linear_q_down_proj = _BF16Linear(args.dim, args.q_lora_rank)
        self.q_layernorm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.linear_q_up_proj = _BF16Linear(args.q_lora_rank, args.n_heads * args.head_dim)
        self.linear_kv_proj = _BF16Linear(args.dim, args.head_dim)
        self.kv_layernorm = RMSNorm(args.head_dim, args.norm_eps)
        self.o_head_grouped = HeadGroupedCompressedAttention(args)

        if compress_ratio:
            self.compressor = _Compressor(args, compress_ratio, args.head_dim)
            if compress_ratio == 4:
                self.indexer = _Indexer(args, compress_ratio)


# ---------------------------------------------------------------------------
# MoE — Router (hash / score) + fused experts
# ---------------------------------------------------------------------------


class _Router(nn.Module):
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


class _RoutedExperts(nn.Module):
    """Per-expert indexed fused `linear_fc1` (gate + up) and `linear_fc2`."""

    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.linear_fc1 = _IndexedBF16Linear(args.n_routed_experts, args.dim, 2 * args.moe_inter_dim)
        self.linear_fc2 = _IndexedBF16Linear(args.n_routed_experts, args.moe_inter_dim, args.dim)


class _SharedExpert(nn.Module):
    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.linear_fc1 = _BF16Linear(args.dim, 2 * args.moe_inter_dim)
        self.linear_fc2 = _BF16Linear(args.moe_inter_dim, args.dim)


class _MoE(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.router = _Router(layer_id, args)
        self.experts = _RoutedExperts(args)
        self.shared_experts = _SharedExpert(args)


# ---------------------------------------------------------------------------
# Transformer / MTP / Root
# ---------------------------------------------------------------------------


class _TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.self_attention = CompressedSparseAttention(layer_id, args)
        self.mlp = _MoE(layer_id, args)
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.pre_mlp_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.hc_attn = _MegatronHC(args.hc_mult, args.dim)
        self.hc_ffn = _MegatronHC(args.hc_mult, args.dim)


class _MTPLayer(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.mtp_model_layer = _TransformerBlock(layer_id, args)
        self.e_proj = _BF16Linear(args.dim, args.dim)
        self.h_proj = _BF16Linear(args.dim, args.dim)
        self.enorm = RMSNorm(args.dim, args.norm_eps)
        self.hnorm = RMSNorm(args.dim, args.norm_eps)
        self.final_layernorm = RMSNorm(args.dim, args.norm_eps)
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))


class _WordEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))


class _Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.word_embeddings = _WordEmbeddings(vocab_size, dim)


class _OutputLayer(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))


class _Decoder(nn.Module):
    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TransformerBlock(i, args) for i in range(args.n_layers)])
        self.final_layernorm = RMSNorm(args.dim, args.norm_eps)
        hc_mult = args.hc_mult
        hc_dim = hc_mult * args.dim
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))


class _MTPStack(nn.Module):
    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_MTPLayer(args.n_layers + i, args) for i in range(args.n_mtp_layers)])


class DeepSeekV4MegatronModel(nn.Module):
    """Megatron-convention parameter container for V4 (BF16 only)."""

    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embedding = _Embedding(args.vocab_size, args.dim)
        self.decoder = _Decoder(args)
        self.output_layer = _OutputLayer(args.vocab_size, args.dim)
        self.mtp = _MTPStack(args)
