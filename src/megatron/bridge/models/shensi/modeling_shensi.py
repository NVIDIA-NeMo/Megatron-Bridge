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

"""Shensi decoder components built on Megatron-Core transformer primitives."""

import copy
import functools
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.models.backends import get_backend_from_config
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.kda import KimiDeltaAttention, KimiDeltaAttentionSubmodules
from megatron.core.tensor_parallel.cross_entropy import vocab_parallel_cross_entropy
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossAutoScaler
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.training import print_rank_0
from torch import Tensor

from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig


# The MCore schedule does not thread input ids down to every layer; DeepEmbed and ROSA read them
# through this context variable.
SHENSI_INPUT_IDS: ContextVar[Optional[Tensor]] = ContextVar("shensi_input_ids", default=None)
# The ROSA auxiliary loss needs the shifted next-token labels at the injection layers.
SHENSI_LABELS: ContextVar[Optional[Tensor]] = ContextVar("shensi_labels", default=None)
# The MTP heads mirror the last decoder layer and therefore need that layer's completed
# attention-residual blocks. The decoder stack fills this sink in its post-process step; the MTP
# layer consumes it during the same forward. Without it the MTP attention-residual routing sees a
# single candidate and its q/k/scale parameters never receive a gradient.
SHENSI_FINAL_BLOCKS: ContextVar[Optional[list]] = ContextVar("shensi_final_blocks", default=None)


def publish_input_ids(input_ids: Optional[Tensor]):
    return SHENSI_INPUT_IDS.set(input_ids)


def reset_input_ids(token) -> None:
    SHENSI_INPUT_IDS.reset(token)


def publish_labels(labels: Optional[Tensor]):
    return SHENSI_LABELS.set(labels)


def reset_labels(token) -> None:
    SHENSI_LABELS.reset(token)


def publish_final_blocks():
    return SHENSI_FINAL_BLOCKS.set([])


def reset_final_blocks(token) -> None:
    SHENSI_FINAL_BLOCKS.reset(token)


def record_final_blocks(blocks: Tensor) -> None:
    sink = SHENSI_FINAL_BLOCKS.get()
    if sink is not None:
        sink.append(blocks)


def _current_final_blocks() -> Optional[Tensor]:
    sink = SHENSI_FINAL_BLOCKS.get()
    return sink[-1] if sink else None


def _current_input_ids(input_ids: Optional[Tensor]) -> Optional[Tensor]:
    return input_ids if input_ids is not None else SHENSI_INPUT_IDS.get()


def _tp_group():
    """The tensor-parallel group when the parallel state is initialized."""
    from megatron.core import parallel_state

    if parallel_state.is_initialized():
        return parallel_state.get_tensor_model_parallel_group()
    return None


# HF keeps these parameters in fp32 while the rest of the model trains in bf16.
FP32_KEEP_SUFFIXES: Tuple[str, ...] = (
    "A_log",
    "basis",
    "conv1d",
    "convs",
    "dt_bias",
    "hc_base",
    "hc_fn",
    "input_layernorm",
    "k_norm",
    "kv_norm",
    "norm",
    "o_norm",
    "post_attention_layernorm",
    "q_a_layernorm",
    "routed_expert_norm",
    "fc2_norm",
    "scale",
)


def apply_fp32_keep(model: torch.nn.Module, *, verbose: bool = False) -> List[str]:
    """Mark the checkpoint's fp32 parameters so the optimizer and casting wrappers keep them."""
    kept: List[str] = []
    for name, parameter in model.named_parameters():
        if any(name.endswith(suffix) or f".{suffix}." in name for suffix in FP32_KEEP_SUFFIXES):
            mark_keep_in_fp32(parameter)
            kept.append(name)
    if verbose:
        print_rank_0(f"[shensi][fp32keep] {len(kept)} parameters kept in fp32")
    return kept


def _mark_tp_replicated(module: nn.Module, *, reduction: str = "average") -> None:
    """Make a replicated module's parameter gradients consistent across the TP group."""
    for parameter in module.parameters():
        setattr(parameter, f"{reduction}_gradients_across_tp_domain", True)


def _duplicated_linear(
    config,
    input_size: int,
    output_size: int,
    *,
    bias: bool = False,
    init_method=None,
):
    """Replicated linear from the backend (Transformer Engine's ``parallel_mode="duplicated"``)."""
    return get_backend_from_config(config).linear()(
        input_size,
        output_size,
        config=config,
        init_method=init_method or config.init_method,
        bias=bias,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        parallel_mode="duplicated",
    )


def _column_linear(
    config,
    input_size: int,
    output_size: int,
    *,
    tp_group=None,
    bias: bool = False,
):
    backend = get_backend_from_config(config)
    return backend.column_parallel_linear()(
        input_size,
        output_size,
        config=config,
        init_method=config.init_method,
        bias=bias,
        gather_output=False,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )


def _row_linear(
    config,
    input_size: int,
    output_size: int,
    *,
    tp_group=None,
    bias: bool = False,
):
    backend = get_backend_from_config(config)
    return backend.row_parallel_linear()(
        input_size,
        output_size,
        config=config,
        init_method=config.init_method,
        bias=bias,
        input_is_parallel=True,
        skip_bias_add=False,
        is_expert=False,
        tp_group=tp_group,
    )


def _build_rms_norm(config, hidden_size: int, eps: Optional[float] = None):
    """The backend's native RMSNorm (``TENorm`` with Transformer Engine)."""
    eps = float(config.layernorm_epsilon if eps is None else eps)
    return get_backend_from_config(config).layer_norm(rms_norm=True)(config=config, hidden_size=hidden_size, eps=eps)


def block_mean_pooling(k: torch.Tensor, k_scale: Optional[torch.Tensor], k_block_size: int) -> torch.Tensor:
    """Average keys within non-overlapping blocks."""
    seq_len, head_dim = k.shape[-2:]
    num_blocks = (seq_len + k_block_size - 1) // k_block_size
    compute_dtype = k.dtype if k.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

    dequant = k.to(compute_dtype)
    if k_scale is not None:
        dequant = dequant * k_scale.to(compute_dtype).unsqueeze(-1)

    counts = torch.full((*k.shape[:-2], num_blocks), k_block_size, device=k.device, dtype=torch.float32)
    padded_len = num_blocks * k_block_size
    if padded_len != seq_len:
        dequant = F.pad(dequant, (0, 0, 0, padded_len - seq_len))
        counts[..., -1] = seq_len - (num_blocks - 1) * k_block_size

    blocked = dequant.reshape(*dequant.shape[:-2], num_blocks, k_block_size, head_dim).sum(dim=-2, dtype=torch.float32)
    return blocked / counts.unsqueeze(-1)


def pool_mqa(
    q: torch.Tensor,
    blocked_k: torch.Tensor,
    blocked_k_scale: Optional[torch.Tensor],
    weights: torch.Tensor,
) -> torch.Tensor:
    """Score query heads against pooled key blocks."""
    q = q.transpose(-3, -2)
    k = blocked_k.to(q.dtype)
    if blocked_k_scale is not None:
        k = k * blocked_k_scale.to(q.dtype).unsqueeze(-1)

    scores = F.relu(torch.matmul(q, k.transpose(-1, -2).unsqueeze(-3)))
    weights = weights.transpose(-1, -2).float().unsqueeze(-1)
    return (scores.float() * weights).sum(dim=-3)


def clean_and_maintain_logits(
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Keep the first and last visible blocks selectable and mask the rest."""
    num_blocks = logits.shape[-1]
    positions = torch.arange(num_blocks, device=logits.device)

    first_block = cu_seqlen_ks.clamp(0, num_blocks - 1).long().unsqueeze(-1)
    last_block = (cu_seqlen_ke - 1).clamp(0, num_blocks - 1).long().unsqueeze(-1)
    logits.scatter_(-1, first_block, float("inf"))
    logits.scatter_(-1, last_block, float("inf"))

    logits.masked_fill_(
        (positions < cu_seqlen_ks.unsqueeze(-1)) | (positions >= cu_seqlen_ke.unsqueeze(-1)),
        float("-inf"),
    )
    return logits


def block_sparse_mqa(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    topk_block_indices: torch.Tensor,
    k_block_size: int,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    """Score all tokens belonging to the selected key blocks."""
    num_kv = k.shape[-2]
    block_starts = topk_block_indices.long() * k_block_size
    pos_in_block = torch.arange(k_block_size, device=q.device)
    k_abs = block_starts.unsqueeze(-1) + pos_in_block
    k_safe = k_abs.clamp(0, num_kv - 1)

    k = k.to(q.dtype)
    if k_scale is not None:
        k = k * k_scale.to(q.dtype).unsqueeze(-1)
    if k.dim() == 2:
        gathered_k = k[k_safe]
    else:
        batch_index = torch.arange(k.shape[0], device=k.device).view(-1, 1, 1, 1)
        gathered_k = k[batch_index, k_safe]

    scores = F.relu(torch.matmul(q, gathered_k.flatten(-3, -2).transpose(-1, -2)))
    weights = weights.float().unsqueeze(-1)
    logits = (scores.float() * weights).sum(dim=-2)

    in_range = (
        (k_abs >= cu_seqlen_ks.unsqueeze(-1).unsqueeze(-1))
        & (k_abs < cu_seqlen_ke.unsqueeze(-1).unsqueeze(-1))
        & (k_abs < num_kv)
    )
    return logits.masked_fill(~in_range.flatten(-2), float("-inf"))


def hisa_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: Optional[torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    *,
    k_block_size: int,
    block_topk: int,
    topk_tokens: int,
) -> torch.Tensor:
    """Select token indices with hierarchical block-sparse attention."""
    blocked_k = block_mean_pooling(k, k_scale, k_block_size)
    cu_seqlen_blocked_ks = cu_seqlen_ks // k_block_size
    cu_seqlen_blocked_ke = (cu_seqlen_ke + k_block_size - 1) // k_block_size

    block_k_score = pool_mqa(q, blocked_k, None, weights)
    clean_and_maintain_logits(block_k_score, cu_seqlen_blocked_ks, cu_seqlen_blocked_ke)

    block_topk = min(block_topk, block_k_score.shape[-1])
    topk_block_indices = block_k_score.bfloat16().topk(block_topk, dim=-1, sorted=False).indices

    block_sparse_logits = block_sparse_mqa(
        q,
        k,
        k_scale,
        topk_block_indices,
        k_block_size,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    topk_tokens = min(topk_tokens, block_sparse_logits.shape[-1])
    relevant_scores, relevant_topk_indices = block_sparse_logits.topk(topk_tokens, dim=-1)
    absolute_topk_block_indices = topk_block_indices.gather(-1, relevant_topk_indices // k_block_size)
    topk_indices = absolute_topk_block_indices * k_block_size + (relevant_topk_indices % k_block_size)
    topk_indices = topk_indices.to(torch.int32)

    topk_indices -= cu_seqlen_ks.unsqueeze(-1).to(torch.int32)
    mask_lo = topk_indices >= 0
    mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks).unsqueeze(-1).to(torch.int32) < 0
    in_range = mask_lo & mask_hi
    topk_scores = relevant_scores.masked_fill(~in_range, float("-inf"))
    return topk_indices.masked_fill(~in_range, -1), topk_scores


def include_self_token(
    topk_indices: torch.Tensor,
    self_positions: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Ensure every valid query keeps its own token in the selected top-k set."""
    self_positions = self_positions.unsqueeze(-1)
    has_self = topk_indices.eq(self_positions).any(dim=-1, keepdim=True)
    replace = valid.unsqueeze(-1) & ~has_self

    last_token = topk_indices[..., -1:]
    last_token = torch.where(replace, self_positions, last_token)

    return torch.cat((topk_indices[..., :-1], last_token), dim=-1)


# Adapted from megatron/core/transformer/experimental_attention_variant/dsa.py (DSAIndexer)
# with the Shensi index field names, the HISA block-then-token selection and the
# reference's low-rank cached keys + shared token table (DSAIndexer pools with learned
# gates; HISA mean-pools).
@dataclass
class ShensiIndexerSubmodules:
    """Modules of the HISA indexer (mirrors Megatron-Core's ``DSAIndexerSubmodules``)."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_wk: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None
    linear_key_down: Union[ModuleSpec, type] = None
    linear_key_up: Union[ModuleSpec, type] = None


# Adapted from megatron/core/transformer/experimental_attention_variant/dsa.py (DSAIndexer)
# with the reference's index projections (low-rank cached keys plus a shared token table instead
# of RoPE and the learned kpool gates) and the HISA block-then-token selection.
class ShensiIndexer(MegatronModule):
    """HISA lightning indexer: block selection, then token refinement, over low-rank cached keys.

    ``forward_before_topk`` projects the queries/keys and produces the selection inputs;
    ``forward`` adds the HISA selection and returns the indices plus the scores that train the
    indexer through its KL auxiliary loss (the native DSAIndexer split).
    """

    def __init__(
        self,
        config: ShensiTransformerConfig,
        submodules: ShensiIndexerSubmodules,
        layer_number: int,
    ) -> None:
        super().__init__(config=config)
        self.layer_idx = layer_number - 1
        self.hidden_size = int(config.hidden_size)
        self.index_n_heads = int(config.index_n_heads)
        self.index_head_dim = int(config.index_head_dim)
        self.index_topk = int(config.index_topk)
        self.q_lora_rank = int(config.q_lora_rank)
        self.softmax_scale = self.index_head_dim**-0.5
        self.topk_blocks = int(config.index_topk_blocks)
        self.block_size = int(config.index_block_size)
        self.latent_rank = int(config.index_latent_rank)

        indexer_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wq_b = build_module(
            submodules.linear_wq_b,
            self.q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            **indexer_kwargs,
        )
        self.linear_wk = build_module(submodules.linear_wk, self.hidden_size, self.index_head_dim, **indexer_kwargs)
        k_norm_config = copy.copy(config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(submodules.k_norm, config=k_norm_config, hidden_size=self.index_head_dim, eps=1e-6)
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj, self.hidden_size, self.index_n_heads, **indexer_kwargs
        )
        self.linear_key_down = build_module(
            submodules.linear_key_down, self.index_head_dim, self.latent_rank, **indexer_kwargs
        )
        self.linear_key_up = build_module(
            submodules.linear_key_up, self.latent_rank, self.index_head_dim, **indexer_kwargs
        )
        self.token_emb = (
            nn.Embedding(int(config.actual_vocab_size or config.vocab_size or 0), self.index_head_dim)
            if self.layer_idx == list(config.indexer_types).index("full")
            else None
        )
        # The indexer projections are replicated across tensor-parallel ranks, so their gradients
        # are averaged during final gradient synchronization (native DSAIndexer convention).
        for parameter in self.parameters():
            setattr(parameter, "average_gradients_across_tp_domain", True)

    def forward_before_topk(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ):
        """Project queries/keys and score weights; return the top-k selection inputs."""
        batch_size, seq_len = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_len, -1, self.index_head_dim)

        q = self.linear_wq_b(q_resid)[0].view(hidden_shape) * self.softmax_scale
        k_input = self.linear_wk(hidden_states)[0]
        k = self.k_norm(k_input.to(self.k_norm.weight.dtype)).to(k_input.dtype)
        k = k.view(hidden_shape).squeeze(2)

        if attention_mask is None:
            attention_mask = hidden_states.new_ones((batch_size, seq_len), dtype=torch.bool)
        attention_mask = attention_mask.bool()
        valid_channel = attention_mask.to(k.dtype)[..., None]

        keys = self.linear_key_up(self.linear_key_down(k)[0])[0]
        if self.token_emb is not None and token_ids is not None and token_ids.shape[1] >= keys.shape[1]:
            keys = keys * self.token_emb(token_ids[:, -keys.shape[1] :]).to(keys.dtype)

        weights = self.linear_weights_proj(hidden_states)[0].float() * (self.index_n_heads**-0.5)

        valid_keys = valid_channel[..., 0].bool()
        any_valid = valid_keys.any(dim=-1)
        first_valid = torch.where(
            any_valid,
            valid_keys.to(torch.int64).argmax(dim=-1),
            torch.zeros(batch_size, dtype=torch.int64, device=hidden_states.device),
        )
        last_valid = torch.where(
            any_valid,
            seq_len - 1 - valid_keys.flip(-1).to(torch.int64).argmax(dim=-1),
            torch.full_like(first_valid, -1),
        )
        q_positions = torch.arange(seq_len, device=hidden_states.device)
        query_end = torch.minimum(q_positions.unsqueeze(0) + 1, last_valid.unsqueeze(-1) + 1)
        query_start = first_valid.unsqueeze(-1).expand(-1, seq_len)
        query_start = torch.where(attention_mask, query_start, torch.zeros_like(query_start))
        query_end = torch.where(attention_mask, query_end, torch.zeros_like(query_end))
        return q, keys, weights, query_start, query_end, valid_keys, attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return absolute top-k key positions and the indexer's scores for them.

        The scores carry gradients: the auxiliary loss that trains the indexer is attached by the
        attention core (KL against the attention distribution over the same candidates). The top-k
        selection itself stays non-differentiable.
        """
        q, keys, weights, query_start, query_end, valid_keys, attention_mask = self.forward_before_topk(
            hidden_states, q_resid, attention_mask, token_ids
        )
        batch_size, seq_len = hidden_states.shape[:2]

        topk_indices, topk_scores = hisa_indexer(
            q,
            keys,
            None,
            weights,
            query_start,
            query_end,
            k_block_size=self.block_size,
            block_topk=self.topk_blocks,
            topk_tokens=self.index_topk,
        )
        keep = topk_indices >= 0
        topk_indices = torch.where(keep, topk_indices + query_start.unsqueeze(-1).to(torch.int32), topk_indices)
        selected_valid = (
            valid_keys.unsqueeze(1).expand(-1, seq_len, -1).gather(2, topk_indices.clamp(0, keys.shape[1] - 1))
        )
        drop = (topk_indices >= 0) & ~selected_valid
        topk_indices = topk_indices.masked_fill(drop, -1)
        topk_scores = topk_scores.masked_fill(drop, float("-inf"))

        self_positions = torch.arange(seq_len, device=hidden_states.device).to(torch.int32)
        self_positions = self_positions.unsqueeze(0).expand(batch_size, -1)
        self_valid = attention_mask & valid_keys.gather(1, self_positions.clamp(0, seq_len - 1).long())
        topk_indices = include_self_token(topk_indices, self_positions, self_valid)
        # The self token is forced into the selection; score it at the row maximum so the indexer
        # distribution covers it (the reference implementation is inference-only and skips this).
        row_max = topk_scores.max(dim=-1, keepdim=True).values
        topk_scores = torch.cat(
            [topk_scores[..., :-1], torch.where(self_valid, row_max, topk_scores[..., -1:])], dim=-1
        )

        topk_indices = F.pad(topk_indices, (0, self.index_topk - topk_indices.shape[-1]), value=-1)
        topk_scores = F.pad(topk_scores, (0, self.index_topk - topk_scores.shape[-1]), value=float("-inf"))
        topk_indices = topk_indices[..., : self.index_topk]
        topk_scores = topk_scores[..., : self.index_topk]
        invalid = topk_indices < 0
        topk_indices = topk_indices.masked_fill(~attention_mask[..., None], -1)
        topk_scores = topk_scores.masked_fill(~attention_mask[..., None] | invalid, float("-inf"))
        return topk_indices.to(torch.int32), topk_scores


def build_topk_mask(topk_indices: torch.Tensor, kv_length: int) -> torch.Tensor:
    """Boolean ``[b, s, kv]`` mask of the selected top-k key positions."""
    valid = topk_indices >= 0
    indices = topk_indices.clamp(0, max(kv_length - 1, 0))
    mask = torch.zeros(*topk_indices.shape[:2], kv_length, dtype=torch.bool, device=topk_indices.device)
    mask.scatter_(2, indices, valid)
    empty = ~mask.any(dim=-1, keepdim=True)
    if empty.any():
        mask = mask | (empty & (torch.arange(kv_length, device=mask.device).view(1, 1, -1) == 0))
    return mask


def build_causal_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    seq_len: int,
    kv_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean ``[b, s, kv]`` causal mask, optionally intersected with a 2D padding mask."""
    causal = torch.ones(seq_len, kv_length, dtype=torch.bool, device=device).tril()
    mask = causal.unsqueeze(0).expand(batch_size, -1, -1)
    if attention_mask is not None:
        valid = attention_mask.bool()
        if valid.dim() == 2 and valid.shape[0] != batch_size and valid.shape[1] == batch_size:
            valid = valid.transpose(0, 1)
        if valid.dim() == 2 and valid.shape == (batch_size, kv_length):
            mask = mask & valid.unsqueeze(1)
    empty = ~mask.any(dim=-1, keepdim=True)
    if empty.any():
        mask = mask | (empty & (torch.arange(kv_length, device=device).view(1, 1, -1) == 0))
    return mask


class ShensiIndexedAttention(MegatronModule):
    """DSA-style sparse attention: low-rank KV latent, HISA top-k, XSA and output gate."""

    def __init__(
        self,
        config: ShensiTransformerConfig,
        layer_number: int,
        pg_collection: Optional[ProcessGroupCollection] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config)
        self.is_mtp_layer = bool(kwargs.pop("is_mtp_layer", False))
        del kwargs
        self.layer_idx = int(config.num_layers) - 1 if self.is_mtp_layer else int(layer_number) - 1
        self.layer_number = self.layer_idx + 1
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        self.pg_collection = pg_collection
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size() if self.tp_group is not None else 1
        self.sequence_parallel = bool(config.sequence_parallel)

        self.num_heads = int(config.num_attention_heads)
        if self.num_heads % self.tp_size:
            raise ValueError(f"Shensi attention heads {self.num_heads} must be divisible by TP {self.tp_size}")
        self.local_num_heads = self.num_heads // self.tp_size
        self.q_lora_rank = int(config.q_lora_rank)
        self.qk_head_dim = int(config.qk_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.head_dim = int(config.v_head_dim)
        self.scaling = self.qk_head_dim**-0.5

        self.linear_config = copy.copy(config)
        # Projections operate on the local token chunk; the stream path owns SP.
        self.linear_config.sequence_parallel = False
        hidden = int(config.hidden_size)

        self.q_a_proj = _duplicated_linear(
            self.linear_config,
            hidden,
            self.q_lora_rank,
            bias=bool(config.add_bias_linear),
        )
        self.q_a_layernorm = _build_rms_norm(config, self.q_lora_rank)
        self.q_b_proj = _column_linear(
            self.linear_config,
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            tp_group=self.tp_group,
        )
        self.gate_proj = _column_linear(
            self.linear_config,
            self.q_lora_rank,
            self.num_heads * self.v_head_dim,
            tp_group=self.tp_group,
        )
        self.kv_latent_proj = _duplicated_linear(self.linear_config, hidden, int(config.deepemb_kv_rank))
        self.kv_up_proj = _duplicated_linear(self.linear_config, int(config.deepemb_kv_rank), self.head_dim)
        self.kv_norm = _build_rms_norm(config, self.head_dim)
        self.o_proj = _row_linear(
            self.linear_config,
            self.num_heads * self.v_head_dim,
            hidden,
            tp_group=self.tp_group,
        )

        indexer_types = list(config.indexer_types)
        self.last_topk_indices = None
        self.skip_topk = indexer_types[self.layer_idx] == "shared"
        self.next_skip_topk = (
            (not self.is_mtp_layer)
            and (not self.skip_topk)
            and indexer_types[min(self.layer_idx + 1, len(indexer_types) - 1)] == "shared"
        )
        self.indexer = (
            None
            if self.skip_topk
            else build_module(_indexer_spec(config), config=config, layer_number=self.layer_idx + 1)
        )
        # A shared-indexer MTP head has no preceding full indexer, so it uses dense causal attention.
        self.mtp_dense_attention = self.is_mtp_layer and self.skip_topk
        # The indexer's top-k output is non-differentiable; the indexer trains through a KL
        # auxiliary loss against the attention distribution (DeepSeek-V3 lightning-indexer style).
        self.indexer_loss_coeff = float(getattr(config, "indexer_loss_coeff", 0.0) or 0.0)

        first_write = list(config.layer_types).index("indexed_attention")
        vocab = int(config.actual_vocab_size or config.vocab_size or 0)
        self.kv_emb_down = nn.Embedding(vocab, int(config.kv_emb_rank)) if self.layer_idx == first_write else None
        self.kv_emb_up = (
            _duplicated_linear(self.linear_config, int(config.kv_emb_rank), self.head_dim)
            if self.layer_idx == first_write
            else None
        )
        for module in (self.q_a_layernorm, self.kv_norm, self.kv_emb_down, self.kv_emb_up):
            if module is not None:
                _mark_tp_replicated(module)

    def _indexer_loss_active(self) -> bool:
        """Whether the indexer KL term runs in this pass (it is a training-only objective)."""
        return self.training and torch.is_grad_enabled() and self.indexer_loss_coeff > 0

    def _indexer_attention_target(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        topk_indices: torch.Tensor,
        chunk: int = 512,
    ) -> torch.Tensor:
        """Head-averaged attention distribution over the selected candidates, ``[b, s, k]``.

        Computed from the gathered candidates in query chunks: softmax over the selected logits
        equals the full softmax gathered and renormalized, without materializing the
        ``[b, heads, s, kv]`` matrix the sparse selection exists to avoid.
        """
        batch, _, seq_len, head_dim = query.shape
        n_selected = topk_indices.shape[-1]
        valid = topk_indices >= 0
        indices = topk_indices.clamp_min(0).long()
        target = query.new_zeros(batch, seq_len, n_selected, dtype=torch.float32)
        with torch.no_grad():
            kv_flat = kv.squeeze(1)  # [b, kv, d]; the KV latent carries a single head
            for start in range(0, seq_len, chunk):
                stop = min(start + chunk, seq_len)
                gather_index = indices[:, start:stop].unsqueeze(-1).expand(-1, -1, -1, head_dim)
                # take_along_dim indexes the KV axis (dim 2); gather(dim=1) would index the chunk.
                selected = torch.take_along_dim(
                    kv_flat.unsqueeze(1).expand(-1, stop - start, -1, -1), gather_index, dim=2
                )
                logits = torch.einsum("bhcd,bckd->bhck", query[:, :, start:stop], selected) * self.scaling
                logits = logits.float().masked_fill(~valid[:, start:stop].unsqueeze(1), float("-inf"))
                target[:, start:stop] = torch.softmax(logits, dim=-1).mean(dim=1)
        return target

    def _attach_indexer_loss(
        self,
        activation: torch.Tensor,
        target: torch.Tensor,
        topk_indices: torch.Tensor,
        indexer_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Attach the indexer KL loss through Megatron-Core's auxiliary-loss autoscaler.

        Target: the head-averaged attention distribution over the selected candidates (detached).
        Prediction: the indexer's score distribution over the same candidates. The top-k selection
        itself stays non-differentiable — only the scores carry the signal.
        """
        if topk_indices is None or not self._indexer_loss_active():
            return activation
        valid = topk_indices >= 0  # [b, s, k]
        if not bool(valid.any()):
            return activation
        log_probs = torch.log_softmax(indexer_scores.float().masked_fill(~valid, float("-inf")), dim=-1)
        kl = (target * (torch.log(target.clamp_min(1e-9)) - log_probs)).sum(dim=-1)  # [b, s]
        loss = kl[valid.any(dim=-1)].mean() * self.indexer_loss_coeff
        return DSAIndexerLossAutoScaler.apply(activation, loss)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, None]:
        del kwargs
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(hidden_states, group=self.tp_group)
        output = self._forward(hidden_states, attention_mask, prev_topk_indices, input_ids)
        if self.sequence_parallel:
            output = scatter_to_sequence_parallel_region(output, group=self.tp_group)
        return output, None

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prev_topk_indices: Optional[torch.Tensor],
        input_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[1], hidden_states.shape[0]
        x = hidden_states.transpose(0, 1).contiguous()
        input_ids = _current_input_ids(input_ids)

        q_residual = self.q_a_layernorm(self.q_a_proj(x)[0])
        query = self.q_b_proj(q_residual)[0]
        query = query.view(batch_size, seq_len, self.local_num_heads, self.qk_head_dim).transpose(1, 2)
        gate = self.gate_proj(q_residual)[0].view(batch_size, seq_len, self.local_num_heads, self.v_head_dim)

        kv = self.kv_norm(self.kv_up_proj(self.kv_latent_proj(x)[0])[0])
        ids = input_ids
        if ids is not None and self.kv_emb_down is not None:
            token_emb = self.kv_emb_up(self.kv_emb_down(ids))[0]
            kv = kv * token_emb.to(kv.dtype)
        kv = kv.unsqueeze(1)

        indexer_scores = None
        if self.mtp_dense_attention:
            topk_indices = None
            mask = build_causal_mask(attention_mask, batch_size, seq_len, kv.shape[2], hidden_states.device)
        elif self.skip_topk:
            if prev_topk_indices is None:
                raise ValueError("Shared indexed-attention layers require top-k indices from a full indexer layer.")
            topk_indices = prev_topk_indices
        else:
            # The indexer only needs a graph for its KL auxiliary loss; the top-k indices are
            # integers either way, so an inactive loss runs the indexer without autograd.
            need_grad = self.training and torch.is_grad_enabled() and self.indexer_loss_coeff > 0
            with torch.set_grad_enabled(need_grad):
                topk_indices, indexer_scores = self.indexer(x, q_residual, attention_mask, ids)

        if not self.mtp_dense_attention:
            mask = build_topk_mask(topk_indices, kv.shape[2])
        # Scaled dot-product attention keeps the [b, h, s, k] matrix inside the kernel: the
        # manual path materialized it (and an fp32 softmax copy of it) on every layer. The KV
        # latent has a single head, so it is broadcast to the query heads (a view; the fused
        # kernels reject a head count of one).
        kv_heads = kv.expand(-1, query.shape[1], -1, -1)
        attention_output = F.scaled_dot_product_attention(
            query, kv_heads, kv_heads, attn_mask=mask.unsqueeze(1), scale=self.scaling
        ).transpose(1, 2)

        # The indexer trains against the attention distribution over the same candidates
        # (DeepSeek-V3 lightning-indexer style): a KL auxiliary loss attached through the
        # auxiliary-loss autoscaler, so the forward output is unchanged.
        if indexer_scores is not None and self._indexer_loss_active():
            target = self._indexer_attention_target(query, kv, topk_indices)
            attention_output = self._attach_indexer_loss(attention_output, target, topk_indices, indexer_scores)

        # XSA removes the component along the query's own value.
        positions = torch.arange(seq_len, device=hidden_states.device)
        self_values = kv[:, 0].index_select(1, positions).unsqueeze(2).to(attention_output.dtype)
        denom = self_values.square().sum(dim=-1, keepdim=True) + 1e-6
        attention_output = (
            attention_output - (attention_output * self_values).sum(dim=-1, keepdim=True) / denom * self_values
        )
        attention_output = attention_output * torch.sigmoid(gate.to(attention_output.dtype))
        attention_output = attention_output.reshape(batch_size, seq_len, -1).contiguous()
        self.last_topk_indices = topk_indices if self.next_skip_topk else None
        return self.o_proj(attention_output)[0].transpose(0, 1)


class ShensiMLP(MLP):
    """Megatron-Core's gated MLP plus the token shift, clamps and DeepEmbed scaling.

    Megatron-Core's ``MLP`` already fuses gate/up into one column-parallel ``linear_fc1`` and its
    non-fused gated path clamps exactly like the reference (``gate.clamp(max=limit)``,
    ``up.clamp(min=-limit, max=limit)``); this subclass supplies the reference's ReLU² activation,
    the previous-token shift on the input and the token-conditioned DeepEmbed scale on the hidden
    activation.
    """

    def __init__(
        self,
        config: ShensiTransformerConfig,
        submodules: MLPSubmodules,
        layer_number: int,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        name: Optional[str] = None,
    ) -> None:
        self.layer_idx = int(layer_number) - 1
        self.hidden_size = int(config.hidden_size)
        self.ffn_hidden_size = int(config.moe_ffn_hidden_size)
        self.limit = float(config.activation_func_clamp_value)
        # Token-wise MLP: the sequence-parallel split is handled by the surrounding stream path.
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        super().__init__(
            config=self.linear_config,
            submodules=submodules,
            ffn_hidden_size=int(config.moe_ffn_hidden_size),
            tp_group=tp_group,
            name=name,
        )
        self.token_shift = nn.Parameter(torch.zeros(self.hidden_size))
        self.deepemb_rank = int(config.deepemb_rank)
        self.deepemb = (
            nn.Embedding(int(config.actual_vocab_size or config.vocab_size or 0), self.deepemb_rank**2)
            if self.layer_idx == list(config.mlp_layer_types).index("dense")
            else None
        )
        self.deepemb_a_proj = _duplicated_linear(self.linear_config, self.hidden_size, self.deepemb_rank)
        self.deepemb_b_proj = _column_linear(
            self.linear_config,
            self.deepemb_rank,
            self.ffn_hidden_size,
            tp_group=self.tp_group,
            bias=True,
        )
        if self.deepemb is not None:
            _mark_tp_replicated(self.deepemb)
        self.token_shift.average_gradients_across_tp_domain = True

    def _deepemb_scale(self, x: torch.Tensor, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        input_ids = _current_input_ids(input_ids)
        deepemb_states = self.deepemb_a_proj(x)[0]
        if self.deepemb is not None and input_ids is not None:
            # DeepEmbed consumes this rank's sequence-parallel token chunk.
            raw_ids = input_ids
            if self.config.sequence_parallel and raw_ids.shape[1] != x.shape[0]:
                tp_group = _tp_group()
                if tp_group is not None and tp_group.size() > 1:
                    rank, local = tp_group.rank(), x.shape[0]
                    raw_ids = raw_ids[:, rank * local : (rank + 1) * local]
            ids = raw_ids.transpose(0, 1).contiguous()
            matrices = self.deepemb(ids).view(*ids.shape, self.deepemb_rank, self.deepemb_rank)
            # HF runs this einsum in the model dtype; keep the same precision.
            deepemb_states = torch.einsum("...r,...rs->...s", deepemb_states, matrices)
        return self.deepemb_b_proj(deepemb_states)[0]

    def forward(self, hidden_states: torch.Tensor, input_ids: Optional[torch.Tensor] = None, **kwargs: Any):
        del kwargs
        x = hidden_states
        shifted = torch.cat([x[:1], x[:-1]], dim=0)
        x = x + (shifted - x) * self.token_shift.to(x.dtype)

        intermediate, bias = self.linear_fc1(x)
        if bias is not None:
            intermediate = intermediate + bias
        gate, up = torch.chunk(intermediate, 2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        hidden = F.relu(gate).square() * up * self._deepemb_scale(x, input_ids)
        return self.linear_fc2(hidden)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            self.token_shift.zero_()
            if self.deepemb is not None:
                eye = torch.eye(self.deepemb_rank, device=self.deepemb.weight.device, dtype=self.deepemb.weight.dtype)
                self.deepemb.weight.copy_(eye.reshape(1, -1).expand_as(self.deepemb.weight))
            self.deepemb_b_proj.weight.zero_()
            if self.deepemb_b_proj.bias is not None:
                self.deepemb_b_proj.bias.fill_(1.0)


class ShensiMoELayer(MoELayer):
    """Latent MoE with Megatron-Core's latent norm/up projection, plus the ERC loss."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.config.moe_latent_size is None:
            raise ValueError("Shensi MoE layers require moe_latent_size")
        self.erc_loss_coef = float(getattr(self.config, "erc_loss_coef", 0.0) or 0.0)
        self.erc_loss_alpha = float(getattr(self.config, "erc_loss_alpha", 0.5))
        self.erc_num_groups = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = super().forward(*args, **kwargs)
        if not self._should_compute_erc():
            return output
        return attach_erc_loss(self, output)

    def _should_compute_erc(self) -> bool:
        return self.erc_num_groups > 0 and self.erc_loss_coef != 0.0 and self.training and torch.is_grad_enabled()

    def _erc_token_scale(self, hidden: torch.Tensor) -> torch.Tensor:
        """Per-token loss scale used by Megatron-Core's MoE auxiliary losses."""
        count = torch.full(
            (),
            hidden.shape[0] * hidden.shape[1],
            dtype=torch.int64,
            device=hidden.device,
        )
        group = getattr(self.router, "tp_cp_group", None)
        if group is not None and group.size() > 1:
            torch.distributed.all_reduce(count, group=group)
        return count.to(hidden.dtype)

    def compute_erc_loss(self) -> torch.Tensor:
        """Expert-router coupling loss for the pool this layer owns."""
        router_weight, down_proj_weight, gate_up = erc_gather_expert_weights(self)
        loss = erc_loss_func(
            router_weight,
            down_proj_weight,
            gate_up,
            alpha=self.erc_loss_alpha,
        )
        return loss * (self.erc_loss_coef / float(self.erc_num_groups))

    def erc_weights(self):
        """Router, latent down-projection and routed-expert gate/up weights for the ERC loss."""
        linear_fc1 = getattr(self.experts, "linear_fc1", None)
        if linear_fc1 is None:
            gate_up = torch.stack(
                [expert.linear_fc1.weight for expert in self.experts.local_experts],
                dim=0,
            )
        else:
            gate_up = getattr(linear_fc1, "weight", None)
            if gate_up is None:
                gate_up = torch.stack(
                    [getattr(linear_fc1, f"weight{i}") for i in range(self.num_local_experts)],
                    dim=0,
                )
        return self.router.weight, self.fc1_latent_proj.weight, gate_up

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        norm = getattr(self, "fc2_norm", None)
        if norm is not None:
            with torch.no_grad():
                norm.weight.fill_(1.0)


def attach_erc_loss(layer, output):
    """Attach the ERC gradient through Megatron-Core's auxiliary-loss autoscaler."""
    hidden = output[0] if isinstance(output, tuple) else output
    erc = layer.compute_erc_loss()
    if getattr(layer.config, "calculate_per_token_loss", False):
        erc = erc * layer._erc_token_scale(hidden)
    hidden = MoEAuxLossAutoScaler.apply(hidden, erc)
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class ShensiRoutedAdapter(nn.Module):
    """Low-rank adapter attached to one routed hyper-connection stream."""

    def __init__(self, hidden_size: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        _mark_tp_replicated(self)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(hidden_states))


# Adapted from megatron/core/transformer/hyper_connection.py (HyperConnectionModule)
# with the reference parameterization: a softmax `pre` collapse, per-stream `post` write
# weights, optional routed streams (`route` + low-rank adapters + per-stream decay) and the
# depthwise-conv stream features (mHC uses a sinkhorn-routed mapping instead).
# Adapted from megatron/core/transformer/hyper_connection.py (HyperConnectionModule) with the
# reference parameterization: a softmax `pre` collapse, per-stream `post` write weights, optional
# routed streams (`route` + low-rank adapters + per-stream decay) and the depthwise-conv stream
# features (mHC instead routes a sinkhorn-projected mapping). The method decomposition follows
# the native module: `aggregate` (H_pre @ x), `apply_h_post` (H_post^T @ F(x)) and
# `apply_h_res` (H_res^T @ x + F(x)).
class ShensiHyperConnection(MegatronModule):
    """Hyper-connection streams: ``x_{l+1} = H_res^T @ x_l + H_post^T @ F(H_pre @ x_l)``.

    One module wraps a sublayer: `aggregate` collapses the stream bank into the sublayer input,
    and `forward` writes the sublayer output back into the active streams while merging the
    residual on the anchor streams.
    """

    def __init__(self, config: ShensiTransformerConfig, with_conv: bool) -> None:
        super().__init__(config=config)
        self.num_streams = int(config.hc_mult)
        self.active_streams = int(config.hc_active_streams)
        self.fixed_streams = int(config.hc_fixed_streams)
        self.routed_streams = self.num_streams - self.fixed_streams
        self.hidden_size = int(config.hidden_size)

        flat_size = self.num_streams * self.hidden_size
        self.pre = nn.Linear(flat_size, self.num_streams, bias=False)
        self.scale = nn.Parameter(torch.ones(2))

        use_conditional = self.active_streams < self.num_streams and self.routed_streams > 0
        self.route = nn.Linear(flat_size, self.routed_streams, bias=False) if use_conditional else None
        self.adapters = (
            nn.ModuleList(
                [
                    ShensiRoutedAdapter(self.hidden_size, int(config.hc_adapter_rank))
                    for _ in range(self.routed_streams)
                ]
            )
            if use_conditional
            else None
        )
        self.decay_logit = nn.Parameter(self._default_decay_logits(torch.empty(0).device)) if use_conditional else None

        kernels = list(config.hc_conv_kernels) if with_conv else []
        self.post = nn.Linear(flat_size, self.active_streams * (1 + len(kernels)), bias=True)
        self.convs = (
            nn.ModuleList(
                [
                    nn.Conv1d(self.hidden_size, self.hidden_size, kernel, groups=self.hidden_size, padding=kernel - 1)
                    for kernel in kernels
                ]
            )
            if kernels
            else None
        )
        _mark_tp_replicated(self)
        mark_keep_in_fp32(self.scale)
        if self.decay_logit is not None:
            mark_keep_in_fp32(self.decay_logit)

    def _default_decay_logits(self, device=None) -> torch.Tensor:
        if self.routed_streams == 0:
            return torch.full((self.num_streams,), -10.0, device=device)
        decays = torch.linspace(0.999, 0.9, self.routed_streams, device=device)
        routed = torch.log((1.0 - decays) / decays)
        return torch.cat([torch.full((self.fixed_streams,), -10.0, device=device), routed])

    def aggregate(self, streams: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Collapse the streams and choose the active write targets (``H_pre @ x``)."""
        seq_len, batch = streams.shape[0], streams.shape[1]
        flat = streams.flatten(2).float()
        pre = torch.softmax(self.pre(flat.to(self.pre.weight.dtype)).float() * self.scale[0], dim=-1).to(streams.dtype)
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2)

        if self.route is not None:
            scores = self.route(flat.to(self.route.weight.dtype))
            top = scores.topk(self.active_streams - self.fixed_streams, dim=-1).indices
            fixed = torch.arange(self.fixed_streams, device=streams.device).expand(seq_len, batch, -1)
            indices = torch.cat([fixed, top + self.fixed_streams], dim=-1)
            route_weights = torch.cat(
                [torch.ones_like(fixed, dtype=scores.dtype), scores.gather(-1, top).sigmoid()], dim=-1
            )
        else:
            indices = torch.arange(self.active_streams, device=streams.device).expand(seq_len, batch, -1)
            route_weights = torch.ones(indices.shape, device=streams.device, dtype=streams.dtype)

        active = streams.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, self.hidden_size))
        state = {
            "streams": streams,
            "indices": indices,
            "route_weights": route_weights.to(streams.dtype),
            "active": active,
            "flat": flat,
        }
        return collapsed, state

    def apply_h_post(self, state: dict, output: torch.Tensor) -> torch.Tensor:
        """Project the sublayer output onto the active streams (``H_post^T @ F(x)``)."""
        active = state["active"]
        seq_len, batch = active.shape[0], active.shape[1]
        features = [output]
        if self.convs is not None:
            tp_group = _tp_group() if self.config.sequence_parallel else None
            sp = tp_group is not None and tp_group.size() > 1
            # The depthwise causal conv needs the full sequence; scatter the local chunk back.
            conv_source = gather_from_sequence_parallel_region(output, group=tp_group) if sp else output
            for conv in self.convs:
                conv_input = conv_source.permute(1, 2, 0).to(conv.weight.dtype)
                conv_output = conv(conv_input)[..., : conv_source.shape[0]].permute(2, 0, 1).to(output.dtype)
                if sp:
                    conv_output = scatter_to_sequence_parallel_region(conv_output, group=tp_group)
                features.append(conv_output)

        post = torch.sigmoid(self.post(state["flat"].to(self.post.weight.dtype)) * self.scale[1]) * 2.0
        post = post.view(seq_len, batch, self.active_streams, len(features))
        addition = sum(post[..., i].unsqueeze(-1) * feature.unsqueeze(2) for i, feature in enumerate(features))

        if self.adapters is not None:
            adapter_update = torch.zeros_like(addition)
            for slot in range(self.fixed_streams, self.active_streams):
                selected = state["indices"][:, :, slot] - self.fixed_streams
                for expert_idx, adapter in enumerate(self.adapters):
                    mask = selected == expert_idx
                    if mask.any():
                        update = adapter(output[mask])
                        scale = state["route_weights"][..., slot][mask].unsqueeze(-1)
                        adapter_update[..., slot, :][mask] = update * scale.to(update.dtype)
            addition = addition + adapter_update
        return addition

    def apply_h_res(self, state: dict, addition: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Merge the decayed streams and the residual anchor into the written streams."""
        active = state["active"]
        if self.decay_logit is not None:
            decay = 1.0 - torch.sigmoid(self.decay_logit.to(active.dtype))
            if self.fixed_streams > 0:
                decay = torch.cat(
                    [
                        torch.ones(self.fixed_streams, device=active.device, dtype=active.dtype),
                        decay[self.fixed_streams :],
                    ]
                )
            mixed = active * decay[state["indices"]].unsqueeze(-1)
        else:
            mixed = active

        if self.fixed_streams > 0:
            residual_mask = torch.zeros(self.active_streams, device=active.device, dtype=active.dtype)
            residual_mask[: min(self.fixed_streams, self.active_streams)] = 1.0
            updated = mixed + addition + residual_mask.view(1, 1, -1, 1) * residual.unsqueeze(2)
            residual_streams = state["streams"]
        else:
            updated = mixed + addition
            residual_streams = state["streams"] + residual.unsqueeze(2)
        updated = updated.to(residual_streams.dtype)
        return residual_streams.scatter(2, state["indices"].unsqueeze(-1).expand_as(updated), updated)

    def forward(self, state: dict, output: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Write a sublayer output back into the active streams."""
        return self.apply_h_res(state, self.apply_h_post(state, output), residual)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            self.pre.weight.zero_()
            self.post.weight.zero_()
            if self.post.bias is not None:
                self.post.bias.zero_()
            self.scale.fill_(1.0)
            if self.route is not None:
                self.route.weight.normal_(0.0, std)
            if self.adapters is not None:
                for adapter in self.adapters:
                    adapter.down.weight.normal_(0.0, std)
                    adapter.up.weight.zero_()
                self.decay_logit.normal_(-3.0, 0.5)
            if self.convs is not None:
                for conv in self.convs:
                    conv.weight.zero_()


class ShensiHyperHead(MegatronModule):
    """Contract the residual streams into one hidden state for the output head."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.eps = float(config.hc_eps)
        streams = int(config.hc_mult)
        hidden = int(config.hidden_size)
        self.input_norm = torch.nn.RMSNorm(
            streams * hidden, eps=float(config.layernorm_epsilon), elementwise_affine=False
        )
        self.hc_fn = nn.Parameter(torch.empty(streams, streams * hidden))
        self.hc_base = nn.Parameter(torch.zeros(streams))
        self.hc_scale = nn.Parameter(torch.ones(1))
        _mark_tp_replicated(self)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = self.input_norm(streams.flatten(2).float())
        pre = torch.sigmoid(flat @ self.hc_fn.float().transpose(0, 1) * self.hc_scale.float() + self.hc_base.float())
        pre = pre + self.eps
        return ((pre.unsqueeze(-1) * streams).sum(dim=2)).to(streams.dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            self.hc_fn.normal_(0.0, std)
            self.hc_base.zero_()
            self.hc_scale.fill_(1.0)


# Adapted from megatron/core/transformer/residual_connection.py (ResidualConnection)
# as a concrete low-rank depth-routing residual; Megatron-Core ships only the interface and
# the wide/mHC implementations.
class ShensiAttentionResidual(MegatronModule):
    """Low-rank depth routing plus channel-wise decay/erase/write and a ReZero gate."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.hidden_size = int(config.hidden_size)
        self.rank = int(config.attn_res_rank)
        self.norm = torch.nn.RMSNorm(self.hidden_size, eps=float(config.layernorm_epsilon), elementwise_affine=False)
        self.q_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.gate_down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.gate_up = nn.Linear(self.rank, 3 * self.hidden_size, bias=True)
        self.logit_scale = nn.Parameter(torch.empty(1))
        self.residual_gate = nn.Parameter(torch.empty(1))
        _mark_tp_replicated(self)

    def forward(
        self,
        prefix: torch.Tensor,
        delta: Optional[torch.Tensor],
        blocks: torch.Tensor,
        num_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the running residual and read completed block deltas into the branch."""
        is_2d = prefix.dim() == 2
        if is_2d:
            prefix = prefix.unsqueeze(1)
            delta = delta.unsqueeze(1) if delta is not None else None
            blocks = blocks.unsqueeze(1)
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        plain = prefix + delta if delta is not None else prefix
        state = self.norm(plain)

        gates = self.gate_up(
            self.gate_down(state.to(self.gate_down.weight.dtype)).to(self.gate_up.weight.dtype)
        ).float()
        decay, erase, write = gates.split(self.hidden_size, dim=-1)
        decay = 1.0 + torch.tanh(decay)
        write = 1.0 + torch.tanh(write)
        erase = torch.tanh(erase)
        updated = decay * prefix + write * (delta if delta is not None else 0.0)
        if delta is not None:
            direction = F.normalize(delta, dim=-1)
            updated = updated - erase * direction * torch.einsum("...d,...d->...", direction, updated).unsqueeze(-1)

        routed = torch.zeros_like(updated)
        if num_blocks > 0:
            blk = blocks[..., :num_blocks, :].float()
            query = self.q_proj(state.to(self.q_proj.weight.dtype)).float()
            keys = F.normalize(self.k_proj(self.norm(blk).to(self.k_proj.weight.dtype)).float(), dim=-1)
            logits = torch.einsum("sbr,sbnr->sbn", query, keys) * self.logit_scale.float()
            weights = torch.softmax(logits, dim=-1)
            routed = torch.einsum("sbn,sbnd->sbd", weights, blk)

        branch = updated + routed
        plain_rms = plain.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        branch_rms = branch.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        output_scale = (plain_rms / branch_rms).detach().clamp(max=10.0)
        gate = torch.tanh(self.residual_gate.float())
        output = plain + gate * branch * output_scale
        updated = plain + gate * updated * output_scale
        if is_2d:
            output = output.squeeze(1)
            updated = updated.squeeze(1)
        return output.to(out_dtype), updated.to(out_dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            self.q_proj.weight.normal_(0.0, std)
            self.k_proj.weight.normal_(0.0, std)
            self.gate_down.weight.normal_(0.0, std)
            self.gate_up.weight.normal_(0.0, std)
            self.gate_up.bias.zero_()
            self.logit_scale.fill_(1.0)
            self.residual_gate.fill_(0.1)


class ShensiAttnResBank(MegatronModule):
    """Low-rank bank of completed attention-residual block summaries."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.hidden_size = int(config.hidden_size)
        self.rank = int(config.attn_res_block_summary_rank)
        self.basis = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        _mark_tp_replicated(self)

    def initialize(self, inputs_embeds: torch.Tensor, num_blocks: int) -> torch.Tensor:
        return inputs_embeds.new_zeros(*inputs_embeds.shape[:2], num_blocks, self.rank)

    def write(self, blocks: torch.Tensor, block_idx: int, summary: torch.Tensor) -> torch.Tensor:
        summary = self.encode(summary.unsqueeze(-2))
        return blocks.index_copy(
            -2,
            torch.tensor([block_idx - 1], device=summary.device, dtype=torch.long),
            summary,
        )

    def read(
        self, blocks: torch.Tensor, inputs_embeds: torch.Tensor, num_blocks: Optional[int] = None
    ) -> torch.Tensor:
        if num_blocks is not None:
            blocks = blocks[..., :num_blocks, :]
        return torch.cat([inputs_embeds.unsqueeze(-2), self.decode(blocks)], dim=-2)

    def encode(self, blocks: torch.Tensor) -> torch.Tensor:
        return torch.einsum("sbnh,rh->sbnr", blocks.float(), self.basis.float()).to(blocks.dtype)

    def decode(self, blocks: torch.Tensor) -> torch.Tensor:
        return torch.einsum("sbnr,rh->sbnh", blocks.float(), self.basis.float()).to(blocks.dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            eye = torch.eye(max(self.hidden_size, self.rank), device=self.basis.device, dtype=self.basis.dtype)
            self.basis.copy_(eye[: self.rank, : self.hidden_size])


class ShensiStreamExpansion(MegatronModule):
    """Create the hyper-connection streams from the token embeddings."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        hidden = int(config.hidden_size)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel, groups=hidden, padding=kernel - 1) for kernel in config.hc_conv_kernels]
        )
        _mark_tp_replicated(self)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        streams = [embeds]
        conv_dtype = self.convs[0].weight.dtype
        tp_group = _tp_group() if self.config.sequence_parallel else None
        sp = tp_group is not None and tp_group.size() > 1
        # The depthwise conv reads the full sequence, then returns to the local SP chunk.
        conv_source = gather_from_sequence_parallel_region(embeds, group=tp_group) if sp else embeds
        conv_input = conv_source.permute(1, 2, 0).to(conv_dtype)
        for conv in self.convs:
            conv_output = conv(conv_input)[..., : conv_source.shape[0]].permute(2, 0, 1).to(embeds.dtype)
            if sp:
                conv_output = scatter_to_sequence_parallel_region(conv_output, group=tp_group)
            streams.append(conv_output)
        return torch.stack(streams, dim=2)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            for conv in self.convs:
                conv.weight.zero_()


class ShensiStreamCompression(MegatronModule):
    """Compress the streams to a low-rank basis between layers."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.hidden_size = int(config.hidden_size)
        self.rank = int(config.hc_stream_rank)
        self.basis = nn.Parameter(torch.empty(self.rank, self.hidden_size))
        _mark_tp_replicated(self)

    def encode(self, streams: torch.Tensor) -> torch.Tensor:
        return torch.einsum("sbkh,rh->sbkr", streams.float(), self.basis.float()).to(streams.dtype)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.einsum("sbkr,rh->sbkh", latents.float(), self.basis.float()).to(latents.dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            eye = torch.eye(max(self.hidden_size, self.rank), device=self.basis.device, dtype=self.basis.dtype)
            self.basis.copy_(eye[: self.rank, : self.hidden_size])


# The reference builds the ROSA suffix automaton token by token. Training always scores a full
# sequence, so the automaton lives in module functions instead of the reference's incremental
# state object (its incremental ``update`` API exists only for generation).
def _rosa_new_state() -> dict:
    return {
        "transitions": [{}],
        "suffix_links": [-1],
        "lengths": [0],
        "last_positions": [-1],
        "tokens": [],
        "last_state": 0,
        "state_count": 1,
        "position": 0,
    }


def _rosa_grow_state(state: dict, index: int) -> None:
    transitions = state["transitions"]
    while len(transitions) <= index:
        transitions.append({})
        state["suffix_links"].append(-1)
        state["lengths"].append(0)
        state["last_positions"].append(-1)


def _rosa_update_state(state: dict, token: int, eos_token_id: Optional[int] = None) -> int:
    transitions = state["transitions"]
    suffix_links = state["suffix_links"]
    lengths = state["lengths"]
    last_positions = state["last_positions"]
    tokens = state["tokens"]

    tokens.append(token)
    current_state = state["state_count"]
    state["state_count"] = current_state + 1
    _rosa_grow_state(state, current_state)
    lengths[current_state] = lengths[state["last_state"]] + 1
    index = state["last_state"]

    while index != -1:
        next_state = transitions[index].get(token)
        if next_state is not None:
            break
        transitions[index][token] = current_state
        index = suffix_links[index]

    if index == -1:
        suffix_links[current_state] = 0
    else:
        candidate = next_state
        if lengths[index] + 1 == lengths[candidate]:
            suffix_links[current_state] = candidate
        else:
            clone_state = state["state_count"]
            state["state_count"] = clone_state + 1
            _rosa_grow_state(state, clone_state)
            transitions[clone_state] = transitions[candidate].copy()
            lengths[clone_state] = lengths[index] + 1
            suffix_links[clone_state] = suffix_links[candidate]
            last_positions[clone_state] = last_positions[candidate]
            while index != -1 and transitions[index].get(token) == candidate:
                transitions[index][token] = clone_state
                index = suffix_links[index]
            suffix_links[candidate] = clone_state
            suffix_links[current_state] = clone_state

    index = state["last_state"] = current_state
    prediction = -1
    while index != -1:
        if lengths[index] > 0 and last_positions[index] >= 0:
            prediction = tokens[last_positions[index] + 1]
            break
        index = suffix_links[index]

    index = state["last_state"]
    position = state["position"]
    while index != -1 and last_positions[index] < position:
        last_positions[index] = position
        index = suffix_links[index]

    state["position"] = position + 1
    if eos_token_id is not None and token == eos_token_id:
        state.clear()
        state.update(_rosa_new_state())
    return prediction


@torch.no_grad()
def rosa(
    token_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    eos_token_id: Optional[int] = None,
) -> torch.LongTensor:
    """Compute ROSA next-token predictions for every position."""
    is_1d = token_ids.dim() == 1
    if is_1d:
        token_ids = token_ids.unsqueeze(0)
    if attention_mask is not None:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.shape != token_ids.shape:
            raise ValueError(
                f"`attention_mask` shape {tuple(attention_mask.shape)} does not match "
                f"`token_ids` shape {tuple(token_ids.shape)}."
            )
        attention_mask = attention_mask.tolist()

    predictions = []
    for batch_idx, sequence in enumerate(token_ids.tolist()):
        state = _rosa_new_state()
        batch_predictions = []
        for position, token in enumerate(sequence):
            if attention_mask is not None and not attention_mask[batch_idx][position]:
                batch_predictions.append(-1)
                continue
            batch_predictions.append(_rosa_update_state(state, token, eos_token_id))
        predictions.append(batch_predictions)

    if not predictions:
        return token_ids.new_full(token_ids.shape, -1)
    outputs = torch.tensor(predictions, dtype=torch.long, device=token_ids.device)
    return outputs.squeeze(0) if is_1d else outputs


class _SharedROSAEmbedding(VocabParallelEmbedding):
    """Vocab-parallel lookup over an already-registered shared token table.

    The shared weight (the input embedding or the output layer) is injected without
    re-registration so ``named_parameters`` keeps its canonical global name. The lookup
    all-reduces instead of reduce-scattering: the ROSA layer consumes local id chunks.
    """

    def __init__(self, weight: torch.nn.Parameter, config: ShensiTransformerConfig) -> None:
        tp_group = _tp_group()
        tp_size = tp_group.size() if tp_group is not None else 1
        super().__init__(
            int(weight.shape[0]) * tp_size,
            int(weight.shape[1]),
            init_method=config.init_method,
            reduce_scatter_embeddings=False,
            config=config,
        )
        object.__setattr__(self, "weight", weight)
        self._parameters.pop("weight", None)


class ShensiROSALayer(MegatronModule):
    """Inject ROSA predictions into every hyper-connection stream."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.embed_dim = int(config.rosa_embed_dim or config.hidden_size)
        self.hidden_size = int(config.hidden_size)
        self.streams = int(config.hc_mult)
        self.vocab_size = int(config.actual_vocab_size or config.vocab_size or 0)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.streams, bias=True)
        self.rosa_loss_coef = float(getattr(config, "rosa_loss_coef", 0.0) or 0.0)
        self.num_rosa_layers = max(len(getattr(config, "rosa_layer_ids", None) or []), 1)
        _mark_tp_replicated(self)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Megatron-Core's vocab-parallel embedding masks out-of-partition ids; -1 is clamped
        # here and zeroed by the validity mask.
        valid = token_ids >= 0
        embeddings = self.embedding(token_ids.clamp_min(0))
        return embeddings * valid.unsqueeze(-1).to(embeddings.dtype)

    def _rosa_token_loss(self, rosa_features: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Cross-entropy of the injected ROSA feature against the shifted next-token labels.

        ``labels`` is passed explicitly (captured by activation checkpointing) rather than read
        from the context variable, which is already reset when checkpoint recompute runs.
        """
        if (
            labels is None
            or self.rosa_loss_coef == 0.0
            or self.embed_dim != self.hidden_size
            or not self.training
            or not torch.is_grad_enabled()
        ):
            return None
        target = labels
        if target.shape[0] == rosa_features.shape[1]:
            pass
        elif target.shape[1] == rosa_features.shape[1]:
            target = target.transpose(0, 1)
        else:
            return None
        tp_group = _tp_group()
        tp_size = tp_group.size() if tp_group is not None else 1
        if target.shape[1] != rosa_features.shape[0]:
            if tp_size <= 1:
                return None
            rank, local = tp_group.rank(), rosa_features.shape[0]
            target = target[:, rank * local : (rank + 1) * local]

        weight = self.embedding.weight
        logits = F.linear(rosa_features.float(), weight.float())
        target_t = target.transpose(0, 1).contiguous()
        if weight.shape[0] != self.vocab_size and tp_size > 1:
            per_token = vocab_parallel_cross_entropy(logits, target_t, tp_group=tp_group)
        else:
            per_token = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_t.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(logits.shape[0], logits.shape[1])
        mask = target_t != -100
        denom = mask.sum().clamp(min=1)
        aux = per_token[mask].sum().float() / denom.float()
        aux = aux * (self.rosa_loss_coef / float(self.num_rosa_layers))
        if getattr(self.config, "calculate_per_token_loss", False):
            aux = aux * torch.full(
                (),
                rosa_features.shape[0] * rosa_features.shape[1],
                dtype=aux.dtype,
                device=aux.device,
            )
        return aux

    def forward(
        self, streams: torch.Tensor, rosa_token_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        ids = rosa_token_ids
        if ids.shape[0] != streams.shape[0]:
            ids = ids.transpose(0, 1)
        rosa_features = self.value_proj(self.embed(ids).to(self.value_proj.weight.dtype))
        gate = torch.sigmoid(self.gate_proj(streams.mean(dim=2))).unsqueeze(-1)
        out = gate * rosa_features.unsqueeze(2)
        aux = self._rosa_token_loss(out.mean(dim=2), labels)
        if aux is not None:
            out = MoEAuxLossAutoScaler.apply(out, aux)
        return out

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            self.gate_proj.weight.zero_()
            self.gate_proj.bias.fill_(-4.0)


@dataclass
class ShensiTransformerLayerSubmodules(TransformerLayerSubmodules):
    """Submodules of a Shensi decoder layer."""

    self_attention_attn_res: Union[ModuleSpec, type] = IdentityOp
    mlp_attn_res: Union[ModuleSpec, type] = IdentityOp


def shensi_layer_plan(config, layer_number: int, is_mtp_layer: bool) -> Tuple[int, bool, bool, int]:
    """Return ``(layer_idx, is_write, is_dense, block_idx)`` for a global layer number."""
    plan_idx = int(config.num_layers) - 1 if is_mtp_layer else int(layer_number) - 1
    is_write = list(config.layer_types)[plan_idx] == "indexed_attention"
    is_dense = list(config.mlp_layer_types)[plan_idx] == "dense"
    writes_before = sum(
        1 for block_type in list(config.attn_res_block_layer_types)[:plan_idx] if block_type == "block_write_layer"
    )
    block_idx = writes_before if is_write else writes_before - 1
    return plan_idx, is_write, is_dense, block_idx


class ShensiTransformerLayer(TransformerLayer):
    """Decoder layer implementing the Shensi stream / attention-residual / ROSA flow."""

    def __init__(
        self,
        config,
        submodules: ShensiTransformerLayerSubmodules,
        layer_number: int = 1,
        is_mtp_layer: bool = False,
        **kwargs: Any,
    ) -> None:
        if not isinstance(config, ShensiTransformerConfig):
            config = ShensiTransformerConfig.from_config(config)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            is_mtp_layer=is_mtp_layer,
            **kwargs,
        )
        self.is_mtp_layer = bool(is_mtp_layer)
        config = self.config
        self.layer_idx, self.is_write, self.is_dense, self.block_idx = shensi_layer_plan(
            config, int(self.layer_number), self.is_mtp_layer
        )
        self.attn_hc = ShensiHyperConnection(config, with_conv=False)
        self.ffn_hc = ShensiHyperConnection(config, with_conv=True)
        self.self_attention_attn_res = ShensiAttentionResidual(config)
        self.mlp_attn_res = ShensiAttentionResidual(config)
        self.rosa = (
            ShensiROSALayer(config)
            if (not self.is_mtp_layer and self.layer_idx in set(config.rosa_layer_ids or []))
            else None
        )
        for norm_name in ("input_layernorm", "pre_mlp_layernorm"):
            norm = getattr(self, norm_name, None)
            if norm is None or not hasattr(norm, "weight"):
                raise RuntimeError(
                    f"ShensiTransformerLayer needs {norm_name} to expose .weight for the "
                    "attention-residual output norm."
                )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        partial_block: Optional[torch.Tensor] = None,
        blocks: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        rosa_token_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        del kwargs
        if hidden_states.dim() == 3:
            streams = hidden_states.unsqueeze(2).expand(-1, -1, int(self.config.hc_mult), -1).contiguous()
        else:
            streams = hidden_states
        seq_len, batch_size = streams.shape[0], streams.shape[1]
        if partial_block is None:
            partial_block = streams.new_zeros(seq_len, batch_size, int(self.config.hidden_size))
        if blocks is None:
            if self.is_mtp_layer:
                blocks = _current_final_blocks()
            if blocks is None:
                blocks = streams.new_zeros(seq_len, batch_size, 0, int(self.config.hidden_size))

        if self.is_write:
            if self.block_idx > 0:
                blocks = blocks.index_copy(
                    -2,
                    torch.tensor([self.block_idx], device=blocks.device, dtype=torch.long),
                    partial_block.unsqueeze(-2),
                )
            partial_block = torch.zeros_like(partial_block)
        partial_blocks = torch.cat([blocks[..., : 1 + self.block_idx, :], partial_block.unsqueeze(-2)], dim=-2)

        if self.rosa is not None and rosa_token_ids is not None:
            streams = streams + self.rosa(streams, rosa_token_ids, labels)

        collapsed, state = self.attn_hc.aggregate(streams)
        topk_indices = prev_topk_indices
        attn_input = self.input_layernorm(collapsed)
        if self.is_write:
            attn_output, _ = self.self_attention(
                attn_input,
                attention_mask=attention_mask,
                prev_topk_indices=prev_topk_indices,
                input_ids=input_ids,
            )
            topk_indices = getattr(self.self_attention, "last_topk_indices", None)
        else:
            attn_output, _ = self.self_attention(attn_input, attention_mask=attention_mask)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        attn_residual, _ = self.self_attention_attn_res(
            partial_block,
            attn_output,
            partial_blocks,
            num_blocks=1 + self.block_idx + 1,
        )
        attn_residual = attn_residual - (partial_block + attn_output)
        streams = self.attn_hc(state, attn_output, attn_residual)
        partial_block = partial_block + attn_output

        collapsed, state = self.ffn_hc.aggregate(streams)
        mlp_input = self.pre_mlp_layernorm(collapsed)
        if self.is_dense:
            mlp_output = self.mlp(mlp_input, input_ids=input_ids)
        else:
            mlp_output = self.mlp(mlp_input)
        if isinstance(mlp_output, tuple):
            mlp_output = mlp_output[0]
        partial_blocks = torch.cat([blocks[..., : 1 + self.block_idx, :], partial_block.unsqueeze(-2)], dim=-2)
        mlp_residual, _ = self.mlp_attn_res(
            partial_block,
            mlp_output,
            partial_blocks,
            num_blocks=1 + self.block_idx + 1,
        )
        mlp_residual = mlp_residual - (partial_block + mlp_output)
        streams = self.ffn_hc(state, mlp_output, mlp_residual)
        partial_block = partial_block + mlp_output

        if hidden_states.dim() == 3:
            streams = streams.mean(dim=2)
        if self.is_mtp_layer:
            return streams, None
        return streams, partial_block, blocks, topk_indices


def _norm_spec(config) -> ModuleSpec:
    return ModuleSpec(module=get_backend_from_config(config).layer_norm(rms_norm=True))


def _indexer_spec(config) -> ModuleSpec:
    """HISA indexer spec: replicated projections from the backend, backend LayerNorm."""
    backend = get_backend_from_config(config)
    duplicated = backend.linear()
    return ModuleSpec(
        module=ShensiIndexer,
        submodules=ShensiIndexerSubmodules(
            linear_wq_b=duplicated,
            linear_wk=duplicated,
            k_norm=backend.layer_norm(),
            linear_weights_proj=duplicated,
            linear_key_down=duplicated,
            linear_key_up=duplicated,
        ),
    )


def _kda_spec(config) -> ModuleSpec:
    backend = get_backend_from_config(config)
    duplicated = backend.linear()
    return ModuleSpec(
        module=KimiDeltaAttention,
        submodules=KimiDeltaAttentionSubmodules(
            in_proj=backend.column_parallel_linear(),
            beta_proj=backend.column_parallel_linear(),
            f_a_proj=duplicated,
            f_b_proj=backend.column_parallel_linear(),
            g_a_proj=duplicated,
            g_b_proj=backend.column_parallel_linear(),
            out_norm=_norm_spec(config),
            out_proj=backend.row_parallel_linear(),
        ),
    )


def _build_dense_mlp(
    config,
    pg_collection=None,
    is_mtp_layer: bool = False,
    name: Optional[str] = None,
    layer_number: int = 1,
    **kwargs: Any,
) -> ShensiMLP:
    del is_mtp_layer, kwargs
    backend = get_backend_from_config(config)
    tp_group = getattr(pg_collection, "tp", None) if pg_collection is not None else None
    return ShensiMLP(
        config,
        submodules=MLPSubmodules(
            linear_fc1=backend.column_parallel_linear(),
            linear_fc2=backend.row_parallel_linear(),
        ),
        layer_number=layer_number,
        tp_group=tp_group,
        name=name,
    )


def _build_moe_mlp(config, pg_collection=None, is_mtp_layer: bool = False, name: Optional[str] = None, **kwargs: Any):
    del kwargs
    backend = get_backend_from_config(config)
    experts = backend.grouped_mlp_modules(bool(config.moe_grouped_gemm))
    router = backend.moe_router()
    submodule_kwargs = {"router": router} if router is not None else {}
    submodules = MoESubmodules(experts=experts, **submodule_kwargs)
    return ShensiMoELayer(
        config=config,
        submodules=submodules,
        pg_collection=pg_collection,
        is_mtp_layer=is_mtp_layer,
        name=name,
    )


def get_shensi_layer_spec(
    use_te: bool,
    config,
    layer_number: int = 1,
    is_mtp_layer: bool = False,
) -> ModuleSpec:
    """Module spec of one Shensi decoder layer."""
    del use_te
    layer_idx = min(max(int(layer_number) - 1, 0), int(config.num_layers) - 1)
    if is_mtp_layer:
        layer_idx = int(config.num_layers) - 1
    attn_type = list(config.layer_types)[layer_idx]
    mlp_type = list(config.mlp_layer_types)[layer_idx]

    self_attention = (
        ModuleSpec(module=ShensiIndexedAttention, params={"attn_mask_type": AttnMaskType.causal})
        if attn_type == "indexed_attention"
        else _kda_spec(config)
    )
    mlp = (
        functools.partial(_build_dense_mlp, layer_number=layer_number)
        if mlp_type == "dense"
        else functools.partial(_build_moe_mlp)
    )
    return ModuleSpec(
        module=ShensiTransformerLayer,
        submodules=ShensiTransformerLayerSubmodules(
            input_layernorm=_norm_spec(config),
            self_attention=self_attention,
            pre_mlp_layernorm=_norm_spec(config),
            mlp=mlp,
        ),
    )


def get_shensi_decoder_block_spec(
    config,
    use_transformer_engine: bool = True,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
    **kwargs: Any,
) -> TransformerBlockSubmodules:
    """Decoder block spec for the local layers of this pipeline stage."""
    del kwargs
    layer_specs = [
        get_shensi_layer_spec(use_te=use_transformer_engine, config=config, layer_number=layer_idx + 1)
        for layer_idx in range(int(config.num_layers))
    ]
    local_layer_specs = [
        layer_specs[layer_id] for layer_id in local_layer_ids(config, vp_stage=vp_stage, pp_rank=pp_rank)
    ]
    return TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=None)


def get_shensi_mtp_layer_spec(config, use_transformer_engine: bool = True) -> ModuleSpec:
    return get_shensi_layer_spec(
        use_te=use_transformer_engine,
        config=config,
        layer_number=int(config.num_layers),
        is_mtp_layer=True,
    )


def get_shensi_mtp_block_spec(
    config,
    *,
    use_transformer_engine: bool = True,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

    layer_spec = get_shensi_mtp_layer_spec(config=config, use_transformer_engine=use_transformer_engine)
    return get_gpt_mtp_block_spec(
        config,
        TransformerBlockSubmodules(layer_specs=[layer_spec], layer_norm=None),
        use_transformer_engine=use_transformer_engine,
        vp_stage=vp_stage,
        pp_rank=pp_rank,
    )


def local_layer_ids(config, *, vp_stage=None, pp_rank=None) -> List[int]:
    """Layer ids built by this pipeline / virtual-pipeline stage."""
    layout = getattr(config, "pipeline_model_parallel_layout", None)
    if layout is not None:
        return list(layout.get_layer_id_list(layer_type=LayerType.decoder, vp_stage=vp_stage, pp_rank=pp_rank))
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return list(range(offset, offset + get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)))


@dataclass
class ShensiStreamState:
    """State carried across pipeline stages: embeddings, streams, residual and ROSA ids."""

    inputs_embeds: Tensor
    streams: Tensor
    prefix_sum: Tensor
    residual_encoded: Tensor
    rosa_ids: Optional[Tensor] = None
    topk_indices: Optional[Tensor] = None


def shensi_topk_payload_rows(index_topk: int, hidden_size: int) -> int:
    """Rows per token that carry the HISA top-k indices through a pipeline payload."""
    values = 1 + 4 * int(index_topk)
    return max(1, (values + int(hidden_size) - 1) // int(hidden_size))


def shensi_payload_rows(num_streams: int, num_blocks: int, topk_rows: int = 0) -> int:
    """Extra rows per token the stream state appends to a stage payload."""
    return 1 + int(num_streams) + 1 + int(num_blocks) + 1 + int(topk_rows)


def _encode_topk_rows(
    topk_indices: Optional[Tensor],
    *,
    seq_len: int,
    batch: int,
    hidden: int,
    topk_rows: int,
    index_topk: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Optional[Tensor]:
    """Pack ``[b, s, k]`` int32 indices into ``topk_rows`` payload rows per token."""
    if topk_rows <= 0:
        return None
    values_per_token = topk_rows * hidden
    if topk_indices is None:
        return torch.zeros(seq_len * topk_rows, batch, hidden, device=device, dtype=dtype)
    ids = topk_indices.to(torch.int32).contiguous()
    k = int(ids.shape[-1])
    if 1 + 4 * k > values_per_token:
        raise ValueError(f"topk payload rows ({topk_rows}) cannot hold {k} indices for hidden size {hidden}")
    bytes_ = ids.view(torch.uint8).reshape(batch, seq_len, 4 * k).to(dtype)
    flag = torch.ones(batch, seq_len, 1, device=ids.device, dtype=dtype)
    values = torch.cat([flag, bytes_], dim=-1)
    values = torch.nn.functional.pad(values, (0, values_per_token - values.shape[-1]))
    values = values.transpose(0, 1)
    rows = values.view(seq_len, batch, topk_rows, hidden).permute(0, 2, 1, 3)
    return rows.reshape(seq_len * topk_rows, batch, hidden)


def _decode_topk_rows(
    rows: Tensor,
    *,
    seq_len: int,
    batch: int,
    hidden: int,
    topk_rows: int,
    index_topk: int,
) -> Optional[Tensor]:
    """Inverse of :func:`_encode_topk_rows`."""
    if topk_rows <= 0:
        return None
    values = rows.view(seq_len, topk_rows, batch, hidden).permute(2, 0, 1, 3)
    values = values.reshape(batch, seq_len, topk_rows * hidden)
    if not bool(values[..., 0].to(torch.int64).any()):
        return None
    nbytes = 4 * int(index_topk)
    raw = values[..., 1 : 1 + nbytes].to(torch.uint8).contiguous()
    return raw.view(torch.int32).reshape(batch, seq_len, int(index_topk))


def pack_shensi_payload(
    stage_output: Tensor,
    state: ShensiStreamState,
    *,
    compression: ShensiStreamCompression,
    bank: ShensiAttnResBank,
    topk_rows: int = 0,
    index_topk: int = 0,
) -> Tensor:
    """Append the stream state to a stage output as extra sequence rows."""
    seq_len, batch, hidden = stage_output.shape
    num_streams = state.streams.shape[2]
    num_blocks = state.residual_encoded.shape[2]
    streams = compression.decode(state.streams).permute(0, 2, 1, 3).reshape(seq_len * num_streams, batch, hidden)
    blocks = bank.decode(state.residual_encoded).permute(0, 2, 1, 3).reshape(seq_len * num_blocks, batch, hidden)
    if state.rosa_ids is None:
        rosa_row = stage_output.new_full((seq_len, batch, hidden), -1)
    else:
        rosa_row = state.rosa_ids.transpose(0, 1).to(stage_output.dtype).unsqueeze(-1).expand(-1, -1, hidden)
    rows = [state.inputs_embeds, streams, state.prefix_sum, blocks, rosa_row]
    topk_rows_tensor = _encode_topk_rows(
        state.topk_indices,
        seq_len=seq_len,
        batch=batch,
        hidden=hidden,
        topk_rows=topk_rows,
        index_topk=index_topk,
        dtype=stage_output.dtype,
        device=stage_output.device,
    )
    if topk_rows_tensor is not None:
        rows.append(topk_rows_tensor.to(stage_output.device))
    return torch.cat([stage_output, torch.cat(rows, dim=0).to(stage_output.dtype)], dim=0)


def unpack_shensi_payload(
    payload: Tensor,
    *,
    compression: ShensiStreamCompression,
    bank: ShensiAttnResBank,
    num_streams: int,
    num_blocks: int,
    topk_rows: int = 0,
    index_topk: int = 0,
) -> Tuple[Tensor, ShensiStreamState]:
    """Split a stage payload into activations and the stream state."""
    rows_per_token = 1 + shensi_payload_rows(num_streams, num_blocks, topk_rows)
    total = int(payload.shape[0])
    if total % rows_per_token:
        raise ValueError(
            f"stage payload with {total} rows does not carry a Shensi stream state "
            f"(expected a multiple of {rows_per_token} rows)"
        )
    seq_len = total // rows_per_token
    batch, hidden = int(payload.shape[1]), int(payload.shape[2])
    activations = payload[:seq_len]
    offset = seq_len
    inputs_embeds = payload[offset : offset + seq_len]
    offset += seq_len
    streams = payload[offset : offset + seq_len * num_streams].reshape(seq_len, num_streams, batch, hidden)
    streams = streams.permute(0, 2, 1, 3)
    streams = compression.encode(streams)
    offset += seq_len * num_streams
    prefix_sum = payload[offset : offset + seq_len]
    offset += seq_len
    blocks = payload[offset : offset + seq_len * num_blocks].reshape(seq_len, num_blocks, batch, hidden)
    blocks = blocks.permute(0, 2, 1, 3)
    residual_encoded = bank.encode(blocks)
    offset += seq_len * num_blocks
    rosa_row = payload[offset : offset + seq_len]
    rosa_values = rosa_row[..., 0].transpose(0, 1).long()
    rosa_ids = rosa_values if bool((rosa_values >= 0).any()) else None
    offset += seq_len
    topk_indices = None
    if topk_rows:
        topk_block = payload[offset : offset + seq_len * topk_rows]
        topk_indices = _decode_topk_rows(
            topk_block,
            seq_len=seq_len,
            batch=batch,
            hidden=hidden,
            topk_rows=topk_rows,
            index_topk=index_topk,
        )
        offset += seq_len * topk_rows
    return activations, ShensiStreamState(
        inputs_embeds=inputs_embeds,
        streams=streams,
        prefix_sum=prefix_sum,
        residual_encoded=residual_encoded,
        rosa_ids=rosa_ids,
        topk_indices=topk_indices,
    )


def erc_loss_func(
    router_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    gate_up_proj: torch.Tensor,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Expert-router coupling loss (computed in fp32; cdist has no bf16 CUDA kernel)."""
    r = router_weight.float()
    down_proj_weight = down_proj_weight.float()
    gate_up_proj = gate_up_proj.float()
    norm_r = torch.norm(r, dim=1)
    distances = torch.cdist(r, r, p=2)
    distances = distances.masked_fill(torch.eye(r.size(0), dtype=torch.bool, device=distances.device), float("inf"))
    min_dist, _ = torch.min(distances, dim=1)
    eps = min_dist / 2 / norm_r

    low = (1 - eps).unsqueeze(1)
    high = (1 + eps).unsqueeze(1)
    noise = torch.rand_like(r)
    r_tilde = (low + noise * (high - low)) * r

    proxy = F.linear(r_tilde, down_proj_weight)
    coupling = torch.norm(torch.einsum("jDd,id->ijD", gate_up_proj, proxy), dim=-1)
    row_diff = torch.clamp(coupling - alpha * torch.diag(coupling).unsqueeze(1), min=0.0)
    col_diff = torch.clamp(coupling - alpha * torch.diag(coupling).unsqueeze(0), min=0.0)
    mask = torch.ones_like(coupling) - torch.eye(coupling.size(0), device=coupling.device)
    return ((row_diff + col_diff) * mask).mean()


class ShensiModel(HybridModel):
    """Shensi model: stream loop and attention-residual state on MCore's HybridModel."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config
        if not isinstance(cfg, ShensiTransformerConfig):
            raise TypeError(
                f"ShensiModel needs a ShensiTransformerConfig, got {type(cfg).__name__}; "
                "build it through ShensiModelProvider / build_shensi_transformer_config."
            )
        hidden = int(cfg.hidden_size)
        self.norm = _build_rms_norm(cfg, hidden)
        self.hc_head = ShensiHyperHead(cfg)
        self.stream_expansion = ShensiStreamExpansion(cfg)
        self.stream_compression = ShensiStreamCompression(cfg)
        self.bank_attn_res = ShensiAttnResBank(cfg)
        self.output_attn_res = ShensiAttentionResidual(cfg)
        self._init_shensi_only_weights()
        self.tie_mlp()
        apply_fp32_keep(self, verbose=True)
        bind = getattr(self.decoder, "set_stream_components", None)
        if bind is None:
            raise RuntimeError("ShensiModel needs a ShensiDecoderStack with set_stream_components")
        bind(
            hc_head=self.hc_head,
            stream_expansion=self.stream_expansion,
            stream_compression=self.stream_compression,
            bank=self.bank_attn_res,
            output_attn_res=self.output_attn_res,
            norm=self.norm,
        )

    def forward(self, input_ids, position_ids=None, attention_mask=None, *args: Any, **kwargs: Any):
        packed = kwargs.get("packed_seq_params")
        if packed is not None:
            raise NotImplementedError(
                "Shensi does not support packed sequences (THD layout) yet: the HISA indexer, the "
                "attention-residual bank and the ROSA automaton all rank over one contiguous "
                "sequence per sample. Pass a 2D padding mask instead, or keep context "
                "parallelism disabled."
            )
        if kwargs.get("cp_batch") is not None:
            raise NotImplementedError(
                "Shensi does not support context parallelism: its attention-residual bank and HISA "
                "indexer operate on the full sequence."
            )
        if position_ids is None:
            seq_len = input_ids.shape[1]
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(
                "Shensi supports only a 2D padding mask; the indexer derives the causal structure "
                f"from its top-k indices, got attention_mask shape {tuple(attention_mask.shape)}"
            )
        if attention_mask is not None:
            valid = attention_mask.bool()
            after_pad = (~valid).cummax(dim=-1).values
            if bool((valid & after_pad).any()):
                raise ValueError(
                    "Shensi attention masks must be right-padded; the input is not right-padded "
                    "(left padding is not supported)."
                )
        token = publish_input_ids(input_ids)
        label_token = publish_labels(kwargs.get("labels"))
        blocks_token = publish_final_blocks()
        try:
            return super().forward(input_ids, position_ids, attention_mask, *args, **kwargs)
        finally:
            reset_final_blocks(blocks_token)
            reset_labels(label_token)
            reset_input_ids(token)

    def _init_shensi_only_weights(self) -> None:
        std = float(self.config.init_method_std)
        for module in self.modules():
            init = getattr(module, "init_shensi_weights", None)
            if callable(init):
                init(std=std, hidden_size=int(self.config.hidden_size))
        for layer in self.decoder.layers:
            attn = getattr(layer, "self_attention", None)
            if attn is None:
                continue
            if getattr(attn, "kv_emb_down", None) is not None:
                with torch.no_grad():
                    attn.kv_emb_down.weight.fill_(1.0)
                    attn.kv_emb_up.weight.fill_(1.0 / max(int(self.config.kv_emb_rank), 1))
            indexer = getattr(attn, "indexer", None)
            if indexer is not None:
                with torch.no_grad():
                    rank = int(self.config.index_latent_rank)
                    head = int(self.config.index_head_dim)
                    eye = torch.eye(max(rank, head), device=indexer.linear_key_down.weight.device)
                    indexer.linear_key_down.weight.copy_(eye[:rank, :head])
                    indexer.linear_key_up.weight.copy_(eye[:head, :rank])
                    if indexer.token_emb is not None:
                        indexer.token_emb.weight.fill_(1.0)

    def tie_mlp(self) -> None:
        """Share routers/experts per attention-residual block and the token tables."""
        cfg = self.config
        layers = list(self.decoder.layers)
        write_layers = [
            idx for idx, block_type in enumerate(cfg.attn_res_block_layer_types) if block_type == "block_write_layer"
        ]
        leaders: Dict[int, Any] = {}
        for layer in layers:
            layer_idx = int(getattr(layer, "layer_idx", layer.layer_number - 1))
            if getattr(layer, "is_dense", True):
                continue
            block_id = max(w for w in write_layers if w <= layer_idx)
            leader = leaders.setdefault(block_id, layer.mlp)
            if leader is not layer.mlp:
                layer.mlp.router = leader.router
                layer.mlp.experts = leader.experts
        num_erc_groups = len(
            {
                max(w for w in write_layers if w <= layer_idx)
                for layer_idx, layer_type in enumerate(cfg.mlp_layer_types)
                if layer_type != "dense"
            }
        )
        for block_id, leader in leaders.items():
            leader.erc_num_groups = num_erc_groups
        self.moe_group_owners = leaders

        dense_layers = [layer for layer in layers if getattr(layer, "is_dense", False)]
        if len(dense_layers) > 1 and dense_layers[0].mlp.deepemb is not None:
            # HF ties the deep-embedding group by prefix (`layers.N.mlp.deepemb` expands to the
            # table and both low-rank projections), so every dense layer shares one copy and the
            # checkpoint stores it under the first dense layer only.
            first_mlp = dense_layers[0].mlp
            for layer in dense_layers[1:]:
                layer.mlp.deepemb = first_mlp.deepemb
                layer.mlp.deepemb_a_proj = first_mlp.deepemb_a_proj
                layer.mlp.deepemb_b_proj = first_mlp.deepemb_b_proj

        indexed_layers = [
            layer
            for layer in layers
            if getattr(layer, "is_write", False) and hasattr(layer.self_attention, "kv_emb_down")
        ]
        if len(indexed_layers) > 1 and indexed_layers[0].self_attention.kv_emb_down is not None:
            first = indexed_layers[0].self_attention
            for layer in indexed_layers[1:]:
                layer.self_attention.kv_emb_down = first.kv_emb_down
                layer.self_attention.kv_emb_up = first.kv_emb_up

        full_indexers = [
            layer.self_attention
            for layer in indexed_layers
            if getattr(layer.self_attention, "indexer", None) is not None
            and layer.self_attention.indexer.token_emb is not None
        ]
        if len(full_indexers) > 1:
            first_emb = full_indexers[0].indexer.token_emb
            for attn in full_indexers[1:]:
                attn.indexer.token_emb = first_emb

        rosa_layers = [layer for layer in layers if getattr(layer, "rosa", None) is not None]
        if rosa_layers and int(cfg.rosa_embed_dim or cfg.hidden_size) == int(cfg.hidden_size):
            # HF ties ROSA to the token table. Reuse Megatron-Core's own shared weight: the input
            # embedding on the first stage, the output layer on the last stage.
            shared_weight = self.shared_embedding_or_output_weight()
            if shared_weight is None:
                raise ValueError(
                    "ROSA is tied to the token embedding, but this pipeline stage has neither the "
                    "input embedding nor the shared output layer. Put ROSA layers on the first or "
                    "last pipeline stage, or set rosa_embed_dim != hidden_size."
                )
            tied_embedding = _SharedROSAEmbedding(shared_weight, cfg)
            for layer in rosa_layers:
                layer.rosa.embedding = tied_embedding
        elif len(rosa_layers) > 1:
            first = rosa_layers[0].rosa.embedding
            for layer in rosa_layers[1:]:
                layer.rosa.embedding = first


def _resolve_parallel_group(kind: str, group):
    size, rank = 1, 0
    try:
        from megatron.core import parallel_state as _ps

        if not _ps.is_initialized():
            return size, rank, group
        if kind == "expert_model_parallel":
            size = int(_ps.get_expert_model_parallel_world_size())
            rank = int(_ps.get_expert_model_parallel_rank())
            if group is None and size > 1:
                group = _ps.get_expert_model_parallel_group()
        elif kind == "expert_tensor_parallel":
            size = int(_ps.get_expert_tensor_parallel_world_size())
            rank = int(_ps.get_expert_tensor_parallel_rank())
            if group is None and size > 1:
                group = _ps.get_expert_tensor_parallel_group()
    except Exception:  # noqa: BLE001
        return 1, 0, group
    return size, rank, group


def _group_size(group) -> int:
    import torch.distributed as dist

    return int(group.size()) if hasattr(group, "size") else int(dist.get_world_size(group))


def _gather_along_dim(t: torch.Tensor, group, dim: int) -> torch.Tensor:
    world = _group_size(group)
    if world <= 1:
        return t
    if dim == 0 and t.is_cuda:
        try:
            from megatron.core.tensor_parallel.mappings import _gather_along_first_dim

            return _gather_along_first_dim(t, group)
        except Exception:  # noqa: BLE001
            pass
    if dim != 0:
        order = (dim,) + tuple(i for i in range(t.dim()) if i != dim)
        inv = tuple(sorted(range(t.dim()), key=lambda i: order[i]))
        t_perm = t.permute(order).contiguous()
    else:
        order, inv, t_perm = None, None, t.contiguous()
    out_shape = list(t_perm.shape)
    out_shape[0] = out_shape[0] * world
    out = torch.empty(out_shape, dtype=t_perm.dtype, device=t_perm.device)
    torch.distributed.all_gather_into_tensor(out, t_perm, group=group)
    return out.permute(inv) if order is not None else out


def _split_along_dim(t: torch.Tensor, group, dim: int, rank: Optional[int] = None) -> torch.Tensor:
    import torch.distributed as dist

    world = _group_size(group)
    if world <= 1:
        return t
    rank = dist.get_rank(group) if rank is None else rank
    n = t.shape[dim] // world
    return t.narrow(dim, rank * n, n).contiguous()


# Copied from megatron/core/tensor_parallel/mappings.py (_GatherFromModelParallelRegion) with
# the gather made dim-generic and a selectable gradient reduction: a "slice" backward hands each
# rank its own shard (exported expert-parallel tensors hold disjoint experts), while
# "reduce_scatter_mean" averages over replicas.
class _GatherFromExpertParallelRegion(torch.autograd.Function):
    """All-gather a tensor along ``dim`` with a configurable gradient reduction."""

    @staticmethod
    def symbolic(graph, input_, group, dim: int, grad_reduce: str):
        """Symbolic function for tracing."""
        return _gather_along_dim(input_, group, dim)

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group, dim: int, grad_reduce: str):
        """Forward function."""
        if grad_reduce not in ("slice", "reduce_scatter_mean"):
            raise NotImplementedError(f"unknown grad_reduce={grad_reduce!r} (slice / reduce_scatter_mean)")
        ctx.group, ctx.dim, ctx.grad_reduce = group, dim, grad_reduce
        return _gather_along_dim(tensor, group, dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward function."""
        import torch.distributed as dist

        if ctx.grad_reduce == "slice":
            return _split_along_dim(grad_output, ctx.group, ctx.dim), None, None, None
        world = _group_size(ctx.group)
        dim = ctx.dim
        order = None
        g = grad_output
        if dim != 0:
            order = (dim,) + tuple(i for i in range(g.dim()) if i != dim)
            inv = tuple(sorted(range(g.dim()), key=lambda i: order[i]))
            g = g.permute(order).contiguous()
        out_shape = list(g.shape)
        out_shape[0] = out_shape[0] // world
        out = torch.empty(out_shape, dtype=g.dtype, device=g.device)
        try:
            dist.reduce_scatter_tensor(out, g.contiguous(), group=ctx.group)
        except (RuntimeError, ValueError) as exc:  # noqa: BLE001
            print_rank_0(
                f"[shensi][erc] [warn] this backend has no reduce_scatter_tensor ({exc!r}); "
                "falling back to all_reduce + narrow (identical numerics, more traffic)"
            )
            full = g.contiguous().clone()
            dist.all_reduce(full, group=ctx.group)
            out = full.narrow(0, 0, out_shape[0]).clone()
        out = out / world
        return (out.permute(inv) if order is not None else out), None, None, None


def erc_gather_expert_weights(
    mlp,
    ep_group=None,
    *,
    grad_reduce: str = "slice",
    local_view: bool = False,
    verbose: bool = False,
):
    """Return the router and expert weights used by the ERC loss.

    Args:
        mlp: Routed-expert layer that owns the weights.
        ep_group: Expert-parallel group; resolved from the parallel state when omitted.
        grad_reduce: ``"slice"`` keeps each rank's local gradient, ``"reduce_scatter_mean"``
            averages it over the group.
        local_view: Score only this rank's experts (debugging on multi-EP runs).
        verbose: Log the resolved view on rank 0.
    """
    ep_size, ep_rank, ep_group = _resolve_parallel_group("expert_model_parallel", ep_group)
    etp_size, etp_rank, etp_group = _resolve_parallel_group("expert_tensor_parallel", None)
    grad_reduce = (grad_reduce or "slice").strip().lower()
    if local_view:
        router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
        n_local = int(getattr(mlp, "num_local_experts", 0) or 0)
        if ep_size > 1 and n_local > 0:
            router_weight = router_weight[ep_rank * n_local : (ep_rank + 1) * n_local]
        if verbose:
            print_rank_0(
                "[shensi][erc] erc_gather_expert_weights: local_view=True -> scoring "
                f"the {n_local} routed experts this rank holds"
            )
        return router_weight, down_proj_weight, gate_up_local
    router_weight, down_proj_weight, gate_up_local = mlp.erc_weights()
    if ep_size <= 1 and etp_size <= 1:
        return router_weight, down_proj_weight, gate_up_local
    if not hasattr(mlp, "fc1_latent_proj"):
        raise NotImplementedError(
            "erc_gather_expert_weights: the ERC term needs the down/up projections of the routed "
            "experts (exposed as fc1_latent_proj on the moe_latent_size path), but this MoE "
            "layer has no such module; configure moe_latent_size = routed_expert_hidden_size."
        )
    if ep_size > 1 and ep_group is None:
        raise RuntimeError(
            f"[shensi][erc] expert_model_parallel_size={ep_size} > 1 but no expert-parallel group "
            "is available (the parallel state is not initialized): the ERC term couples the router "
            "with the experts of the whole group, and a per-rank view scores a different "
            "objective. Build the model after initialize_model_parallel, or pass local_view=True to "
            "score the local experts on purpose."
        )
    if etp_size > 1 and etp_group is None:
        raise RuntimeError(
            f"[shensi][erc] expert_tensor_parallel_size={etp_size} > 1 but no expert-tensor-parallel "
            "group is available: the ERC norms span the full intermediate dimension."
        )
    gate_up = gate_up_local
    if ep_size > 1:
        gate_up = _GatherFromExpertParallelRegion.apply(gate_up, ep_group, 0, grad_reduce)
    if etp_size > 1:
        gate_up = _GatherFromExpertParallelRegion.apply(gate_up, etp_group, 1, grad_reduce)
    if verbose:
        print_rank_0(
            f"[shensi][erc] global view: ep={ep_size} (rank {ep_rank}, "
            f"{getattr(mlp, 'num_local_experts', '?')} local experts), etp={etp_size} "
            f"(rank {etp_rank}), gate_up_proj={tuple(gate_up.shape)} grad_reduce={grad_reduce}"
        )
    return router_weight, down_proj_weight, gate_up
