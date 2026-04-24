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

Parameter names follow Megatron conventions, with two V4-specific sub-modules
introduced:

* **CSA** — Compressed Sparse Attention. Full V4 attention block: low-rank Q
  (linear_q_down_proj / q_layernorm / linear_q_up_proj), single KV projection
  (linear_kv_proj / kv_layernorm), per-layer Compressor + Indexer (extending
  mcore's DSA pattern), and the grouped low-rank O delegated to HCA.
* **HCA** — Head-grouped Compressed Attention. The V4-specific grouped
  low-rank O projection (`linear_o_down_proj` of shape
  `[n_heads·head_dim/n_groups, n_groups·o_lora_rank]`, then `linear_o_up_proj`).
  Isolated so alternative O variants can be slotted in without touching CSA.

Block-level HC (hc_attn, hc_ffn) uses mcore's `HyperConnectionModule` parameter
layout (fused `mapping_proj.weight [n²+2n, n·C]` + `bias` + three scalar alphas)
so a real Megatron training run can drop in `mcore.HyperConnectionModule`
directly. The head-level HC (`hc_head_*`) stays as raw parameters: the inference
head uses a smaller (n-row) form that mcore's HC module does not cover.

This mirror is a pure `nn.Module` container — forward is deferred; it exists to
validate parameter-name, shape, and dtype parity via the bridge mapping.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from megatron.bridge.models.deepseek_v4.modeling_deepseek_v4.reference import (
    DeepSeekV4ModelArgs,
    RMSNorm,
)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _Linear(nn.Module):
    """Mirror `reference.Linear`: `.weight` + optional `.scale` (FP8 block)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dtype: Optional[torch.dtype] = None,
        emit_scale: bool = True,
        block_size: int = 128,
    ) -> None:
        super().__init__()
        dtype = dtype or torch.bfloat16
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
        if emit_scale:
            scale_rows = (out_features + block_size - 1) // block_size
            scale_cols = (in_features + block_size - 1) // block_size
            self.scale = nn.Parameter(torch.empty(scale_rows, scale_cols, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)


# ---------------------------------------------------------------------------
# HC — mcore-compatible block-level layout
# ---------------------------------------------------------------------------


class _MegatronHC(nn.Module):
    """Param-for-param compatible with `mcore.HyperConnectionModule`.

    * `mapping_proj.weight [n²+2n, n·C]` — the fused H_pre/H_post/H_res logits
      projection. Matches HF's `hc_attn_fn` / `hc_ffn_fn` when n = hc_mult.
    * `bias [n²+2n]`                     — static biases; matches HF's
      `hc_attn_base` / `hc_ffn_base`.
    * `alpha_pre / alpha_post / alpha_res [1]` — three learnable gates; together
      they correspond to HF's `hc_*_scale [3]` in a fixed order.
    """

    def __init__(self, n: int, hidden_size: int) -> None:
        super().__init__()
        mix = n * n + 2 * n
        self.mapping_proj = nn.Linear(n * hidden_size, mix, bias=False)
        self.alpha_pre = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(mix, dtype=torch.float32))

    def _reset(self) -> None:  # pragma: no cover - placeholder for real init
        self.mapping_proj.weight.data.zero_()
        self.bias.data.zero_()
        self.alpha_pre.data.zero_()
        self.alpha_post.data.zero_()
        self.alpha_res.data.zero_()


# ---------------------------------------------------------------------------
# HCA — Head-grouped Compressed Attention (grouped low-rank O)
# ---------------------------------------------------------------------------


class HeadGroupedCompressedAttention(nn.Module):
    """V4 grouped low-rank O-projection.

    Given `n_heads` heads split across `n_groups`, each group owns an
    `o_lora_rank`-sized latent dimension. Export shapes match the HF
    inference checkpoint verbatim:

    * `linear_o_down_proj.weight`  [n_groups · o_lora_rank, n_heads·head_dim / n_groups]
    * `linear_o_up_proj.weight`    [hidden_size, n_groups · o_lora_rank]

    Forward is intentionally omitted here — see notes at the module level.
    """

    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.n_groups = args.o_groups
        self.o_lora_rank = args.o_lora_rank
        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.linear_o_down_proj = _Linear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            dtype=torch.bfloat16,
            emit_scale=True,
        )
        self.linear_o_up_proj = _Linear(
            self.n_groups * self.o_lora_rank,
            args.dim,
            emit_scale=True,
        )


# ---------------------------------------------------------------------------
# CSA — Compressed Sparse Attention (V4 full attention block)
# ---------------------------------------------------------------------------


class _Compressor(nn.Module):
    """KV Compressor: wkv/wgate/ape/norm. BF16 params; no `.scale`."""

    def __init__(self, args: DeepSeekV4ModelArgs, compress_ratio: int, head_dim: int) -> None:
        super().__init__()
        overlap = compress_ratio == 4
        coff = 1 + overlap
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, dtype=torch.float32))
        self.wkv = _Linear(args.dim, coff * head_dim, dtype=torch.float32, emit_scale=False)
        self.wgate = _Linear(args.dim, coff * head_dim, dtype=torch.float32, emit_scale=False)
        self.norm = RMSNorm(head_dim, args.norm_eps)


class _Indexer(nn.Module):
    """Top-k sparse-attention indexer. Present when compress_ratio == 4."""

    def __init__(self, args: DeepSeekV4ModelArgs, compress_ratio: int) -> None:
        super().__init__()
        self.linear_q_up_proj = _Linear(args.q_lora_rank, args.index_n_heads * args.index_head_dim, emit_scale=True)
        self.weights_proj = _Linear(args.dim, args.index_n_heads, dtype=torch.bfloat16, emit_scale=False)
        self.compressor = _Compressor(args, compress_ratio, args.index_head_dim)


class CompressedSparseAttention(nn.Module):
    """V4 attention block: low-rank Q + single KV + HCA O + optional
    Compressor/Indexer. Extends mcore DSA (indexer/sparse_attn) with a learned
    Compressor over KV so sparse-attention can attend to compressed slots."""

    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.layer_id = layer_id
        compress_ratio = args.compress_ratios[layer_id]

        self.attn_sink = nn.Parameter(torch.empty(args.n_heads, dtype=torch.float32))
        self.linear_q_down_proj = _Linear(args.dim, args.q_lora_rank, emit_scale=True)
        self.q_layernorm = RMSNorm(args.q_lora_rank, args.norm_eps)
        self.linear_q_up_proj = _Linear(args.q_lora_rank, args.n_heads * args.head_dim, emit_scale=True)
        self.linear_kv_proj = _Linear(args.dim, args.head_dim, emit_scale=True)
        self.kv_layernorm = RMSNorm(args.head_dim, args.norm_eps)

        # HCA: grouped low-rank O-projection.
        self.o_head_grouped = HeadGroupedCompressedAttention(args)

        if compress_ratio:
            self.compressor = _Compressor(args, compress_ratio, args.head_dim)
            if compress_ratio == 4:
                self.indexer = _Indexer(args, compress_ratio)


# ---------------------------------------------------------------------------
# MoE — Router (hash / score) + Experts
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


class _Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, *, emit_scale: bool) -> None:
        super().__init__()
        self.linear_fc1_gate = _Linear(dim, inter_dim, emit_scale=emit_scale)
        self.linear_fc1_up = _Linear(dim, inter_dim, emit_scale=emit_scale)
        self.linear_fc2 = _Linear(inter_dim, dim, emit_scale=emit_scale)


class _MoE(nn.Module):
    def __init__(self, layer_id: int, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.router = _Router(layer_id, args)
        self.experts = nn.ModuleList(
            [_Expert(args.dim, args.moe_inter_dim, emit_scale=True) for _ in range(args.n_routed_experts)]
        )
        self.shared_experts = _Expert(args.dim, args.moe_inter_dim, emit_scale=True)


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
        self.e_proj = _Linear(args.dim, args.dim, emit_scale=True)
        self.h_proj = _Linear(args.dim, args.dim, emit_scale=True)
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
    """Megatron-style layout; `state_dict()` keys are the bridge's target set."""

    def __init__(self, args: DeepSeekV4ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embedding = _Embedding(args.vocab_size, args.dim)
        self.decoder = _Decoder(args)
        self.output_layer = _OutputLayer(args.vocab_size, args.dim)
        self.mtp = _MTPStack(args)
