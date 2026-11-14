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
from megatron.core.activations import squared_relu
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
from megatron.core.transformer.experimental_attention_variant.dsa_indexer_loss import indexer_kl_per_row
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
from torch import Tensor

from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig
from megatron.bridge.utils.common_utils import print_rank_0


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


@dataclass
class ShensiIndexerSubmodules:
    """Modules of the HISA indexer (mirrors Megatron-Core's ``DSAIndexerSubmodules``)."""

    linear_wq_b: Union[ModuleSpec, type] = None
    linear_key: Union[ModuleSpec, type] = None
    linear_key_up: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    linear_weights_proj: Union[ModuleSpec, type] = None


# Adapted from transformers/models/shensi (ShensiIndexer) with Megatron-Core's replicated
# projection convention; training additionally keeps the selected logits so the indexer can be
# trained through its KL auxiliary loss (the reference is inference-only).
class ShensiIndexer(MegatronModule):
    """HISA indexer: block selection, then token refinement, over a low-rank key latent."""

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
        self.index_topk_tokens = int(config.index_topk_tokens)
        self.index_k_size = int(config.index_k_size)
        self.index_block_size = int(config.index_block_size)
        self.index_block_topk = int(config.index_block_topk)
        self.q_lora_rank = int(config.q_lora_rank)
        self.softmax_scale = self.index_head_dim**-0.5

        indexer_kwargs = dict(
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_wq_b = build_module(
            submodules.linear_wq_b, self.q_lora_rank, self.index_n_heads * self.index_head_dim, **indexer_kwargs
        )
        self.linear_key = build_module(submodules.linear_key, self.hidden_size, self.index_k_size, **indexer_kwargs)
        self.linear_key_up = build_module(
            submodules.linear_key_up, self.index_k_size, self.index_head_dim, **indexer_kwargs
        )
        self.linear_weights_proj = build_module(
            submodules.linear_weights_proj, self.hidden_size, self.index_n_heads, **indexer_kwargs
        )
        k_norm_config = copy.copy(config)
        k_norm_config.normalization = "LayerNorm"
        self.k_norm = build_module(submodules.k_norm, config=k_norm_config, hidden_size=self.index_head_dim, eps=1e-6)
        self.linear_key_emb = (
            nn.Embedding(int(config.actual_vocab_size or config.vocab_size or 0), self.index_head_dim)
            if self.layer_idx == list(config.indexer_types).index("full")
            else None
        )
        for parameter in self.parameters():
            setattr(parameter, "average_gradients_across_tp_domain", True)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            if self.linear_key_emb is not None:
                self.linear_key_emb.weight.fill_(1.0)

    def forward_before_topk(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None,
    ):
        """Project queries/keys and score weights; return the top-k selection inputs."""
        batch_size, seq_len = hidden_states.shape[:2]
        q = self.linear_wq_b(q_resid)[0].view(batch_size, seq_len, -1, self.index_head_dim) * self.softmax_scale
        keys = self.linear_key_up(self.linear_key(hidden_states)[0])[0]
        if self.linear_key_emb is not None and token_ids is not None and token_ids.shape[1] >= keys.shape[1]:
            keys = keys * self.linear_key_emb(token_ids[:, -keys.shape[1] :]).to(keys.dtype)
        keys = self.k_norm(keys.to(self.k_norm.weight.dtype)).to(keys.dtype)

        if attention_mask is None:
            attention_mask = hidden_states.new_ones((batch_size, seq_len), dtype=torch.bool)
        attention_mask = attention_mask.bool()
        valid_keys = attention_mask
        weights = self.linear_weights_proj(hidden_states)[0].float() * (self.index_n_heads**-0.5)

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
        return q, keys, weights, query_start, query_end, attention_mask

    def _select(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        weights: torch.Tensor,
        query_start: torch.Tensor,
        query_end: torch.Tensor,
    ):
        batch_size, seq_len = query_start.shape
        blocked_seq_len = keys.shape[-2]
        num_blocks = (blocked_seq_len + self.index_block_size - 1) // self.index_block_size
        counts = torch.full(
            (*keys.shape[:-2], num_blocks), self.index_block_size, device=keys.device, dtype=torch.float32
        )
        padded_len = num_blocks * self.index_block_size
        blocked_k = keys
        if padded_len != blocked_seq_len:
            blocked_k = F.pad(blocked_k, (0, 0, 0, padded_len - blocked_seq_len))
            counts[..., -1] = blocked_seq_len - (num_blocks - 1) * self.index_block_size
        blocked_k = blocked_k.reshape(*blocked_k.shape[:-2], num_blocks, self.index_block_size, -1)
        blocked_k = blocked_k.sum(dim=-2, dtype=torch.float32) / counts.unsqueeze(-1)
        blocked_k = blocked_k.to(q.dtype)

        block_scores = torch.matmul(q.transpose(-3, -2), blocked_k.transpose(-1, -2).unsqueeze(-3)).relu_()
        block_k_score = torch.einsum("bhsp,bhs->bsp", block_scores.float(), weights.transpose(-1, -2))

        cu_lo = query_start // self.index_block_size
        cu_hi = (query_end + self.index_block_size - 1) // self.index_block_size
        block_positions = torch.arange(num_blocks, device=block_k_score.device)
        first_block = cu_lo.clamp(0, num_blocks - 1).long().unsqueeze(-1)
        last_block = (cu_hi - 1).clamp(0, num_blocks - 1).long().unsqueeze(-1)
        block_k_score.scatter_(-1, first_block, float("inf"))
        block_k_score.scatter_(-1, last_block, float("inf"))
        block_k_score.masked_fill_(
            (block_positions < cu_lo.unsqueeze(-1)) | (block_positions >= cu_hi.unsqueeze(-1)), float("-inf")
        )
        block_topk_eff = min(self.index_block_topk, block_k_score.shape[-1])
        topk_block_indices = block_k_score.bfloat16().topk(block_topk_eff, dim=-1, sorted=False).indices

        num_kv = keys.shape[-2]
        pos_in_block = torch.arange(self.index_block_size, device=q.device)
        batch_offsets = torch.arange(batch_size, device=keys.device).view(-1, 1, 1, 1) * num_kv
        keys_flat = keys.reshape(batch_size * num_kv, self.index_head_dim)
        token_weights = weights.to(q.dtype)

        block_chunk = 2 if block_topk_eff >= 32 and batch_size * seq_len <= 8192 else 1
        block_sparse_logits = q.new_empty((batch_size, seq_len, block_topk_eff * self.index_block_size), dtype=q.dtype)
        for block_start in range(0, block_topk_eff, block_chunk):
            block_end = min(block_start + block_chunk, block_topk_eff)
            block_ids = topk_block_indices[..., block_start:block_end]
            k_abs = block_ids.unsqueeze(-1) * self.index_block_size + pos_in_block
            flat_indices = (k_abs.clamp(0, num_kv - 1) + batch_offsets).reshape(-1)
            gathered_k = keys_flat[flat_indices].reshape(
                batch_size, seq_len, (block_end - block_start) * self.index_block_size, self.index_head_dim
            )
            token_scores = torch.matmul(q, gathered_k.transpose(-1, -2)).relu_()
            logits = torch.einsum("bsh,bshk->bsk", token_weights, token_scores)
            in_range = (
                (k_abs >= query_start.unsqueeze(-1).unsqueeze(-1))
                & (k_abs < query_end.unsqueeze(-1).unsqueeze(-1))
                & (k_abs < num_kv)
            )
            logits.masked_fill_(~in_range.flatten(-2), float("-inf"))
            block_sparse_logits[..., block_start * self.index_block_size : block_end * self.index_block_size] = logits

        topk_tokens_eff = min(self.index_topk_tokens, block_sparse_logits.shape[-1])
        topk_logits, relevant_topk_indices = block_sparse_logits.topk(topk_tokens_eff, dim=-1, sorted=False)
        absolute_blocks = topk_block_indices.gather(-1, relevant_topk_indices // self.index_block_size)
        topk_indices = absolute_blocks * self.index_block_size + (relevant_topk_indices % self.index_block_size)
        topk_indices = topk_indices.to(torch.int32) - query_start.unsqueeze(-1).to(torch.int32)
        in_range = (topk_indices >= 0) & (topk_indices - (query_end - query_start).unsqueeze(-1).to(torch.int32) < 0)
        topk_indices = topk_indices.masked_fill(~in_range, -1)
        topk_logits = topk_logits.masked_fill(~in_range, float("-inf"))
        topk_indices = torch.where(
            topk_indices >= 0, topk_indices + query_start.unsqueeze(-1).to(torch.int32), topk_indices
        )
        return topk_indices, topk_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the selected positions and their logits (the logits train the indexer)."""
        q, keys, weights, query_start, query_end, valid = self.forward_before_topk(
            hidden_states, q_resid, attention_mask, token_ids
        )
        batch_size, seq_len = hidden_states.shape[:2]
        kv_len = keys.shape[1]
        topk_indices, topk_logits = self._select(q, keys, weights, query_start, query_end)
        selected_valid = valid.unsqueeze(1).expand(-1, seq_len, -1).gather(2, topk_indices.clamp(0, kv_len - 1))
        drop = (topk_indices >= 0) & ~selected_valid
        topk_indices = topk_indices.masked_fill(drop, -1)
        topk_logits = topk_logits.masked_fill(drop, float("-inf"))

        # The self token is forced into the selection by replacing the last slot (reference
        # semantics; the last slot is the weakest candidate).
        self_positions = torch.arange(seq_len, device=hidden_states.device).to(torch.int32)
        self_positions = self_positions.unsqueeze(0).expand(batch_size, -1).unsqueeze(-1)
        has_self = topk_indices.eq(self_positions).any(dim=-1, keepdim=True)
        self_valid = valid & (self_positions.squeeze(-1) < kv_len)
        replace = self_valid.unsqueeze(-1) & ~has_self
        last_token = torch.where(replace, self_positions, topk_indices[..., -1:])
        topk_indices = torch.cat((topk_indices[..., :-1], last_token), dim=-1)

        output_width = self.index_topk_tokens
        pad = output_width - topk_indices.shape[-1]
        topk_indices = F.pad(topk_indices, (0, pad), value=-1)
        topk_logits = F.pad(topk_logits, (0, pad), value=float("-inf"))
        topk_indices = topk_indices[..., :output_width].masked_fill(~valid[..., None], -1)
        topk_logits = topk_logits[..., :output_width].masked_fill(~valid[..., None], float("-inf"))
        return topk_indices.to(torch.int32), topk_logits


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
    """Gated sparse attention: low-rank KV latent, HISA top-k, self-position exclusion."""

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
        self.tp_group = pg_collection.tp
        self.tp_size = self.tp_group.size() if self.tp_group is not None else 1
        self.sequence_parallel = bool(config.sequence_parallel)

        self.num_heads = int(config.num_attention_heads)
        self.local_num_heads = self.num_heads // self.tp_size
        self.qk_head_dim = int(config.qk_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.head_dim = self.qk_head_dim
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.scaling = self.qk_head_dim**-0.5
        self.indexer_loss_coeff = float(getattr(config, "indexer_loss_coeff", 0.0) or 0.0)

        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        self.q_a_proj = _duplicated_linear(self.linear_config, int(config.hidden_size), self.q_lora_rank)
        self.q_a_layernorm = _build_rms_norm(config, self.q_lora_rank)
        self.q_b_proj = _column_linear(
            self.linear_config,
            self.q_lora_rank,
            self.num_heads * (self.qk_head_dim + self.v_head_dim),
            tp_group=self.tp_group,
            bias=bool(config.add_bias_linear),
        )
        self.kv = _duplicated_linear(self.linear_config, int(config.hidden_size), self.kv_lora_rank)
        self.kv_up = _duplicated_linear(self.linear_config, self.kv_lora_rank, self.head_dim)
        first_indexed = list(config.layer_types).index("indexed_attention")
        self.kv_emb = (
            nn.Embedding(int(config.actual_vocab_size or config.vocab_size or 0), self.kv_lora_rank)
            if self.layer_idx == first_indexed
            else None
        )
        self.kv_norm = _build_rms_norm(config, self.head_dim)
        self.o_proj = _row_linear(
            self.linear_config,
            self.num_heads * self.v_head_dim,
            int(config.hidden_size),
            tp_group=self.tp_group,
            bias=bool(config.add_bias_linear),
        )
        self.skip_topk = list(config.indexer_types)[self.layer_idx] == "shared"
        self.indexer = (
            None
            if self.skip_topk
            else build_module(_indexer_spec(config), config=config, layer_number=self.layer_idx + 1)
        )
        # The MTP head has no predecessor to take shared indices from: fall back to dense causal.
        self.mtp_dense_attention = self.is_mtp_layer and self.skip_topk
        indexer_types = list(config.indexer_types)
        self.next_skip_topk = (
            not self.skip_topk and indexer_types[min(self.layer_idx + 1, len(indexer_types) - 1)] == "shared"
        )
        for module in (self.q_a_layernorm, self.kv_norm, self.kv_emb):
            if module is not None:
                _mark_tp_replicated(module)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            if self.kv_emb is not None:
                self.kv_emb.weight.zero_()

    def _indexer_loss_active(self) -> bool:
        """Whether the indexer KL term runs in this pass (it is a training-only objective)."""
        return self.training and torch.is_grad_enabled() and self.indexer_loss_coeff > 0

    def _indexer_attention_target(
        self, query: torch.Tensor, kv: torch.Tensor, topk_indices: torch.Tensor, chunk: int = 512
    ) -> torch.Tensor:
        """Head-averaged attention distribution over the selected candidates, ``[b, s, k]``."""
        batch, _, seq_len, head_dim = query.shape
        n_selected = topk_indices.shape[-1]
        valid = topk_indices >= 0
        indices = topk_indices.clamp_min(0).long()
        target = query.new_zeros(batch, seq_len, n_selected, dtype=torch.float32)
        with torch.no_grad():
            kv_flat = kv.squeeze(1)
            for start in range(0, seq_len, chunk):
                stop = min(start + chunk, seq_len)
                gather_index = indices[:, start:stop].unsqueeze(-1).expand(-1, -1, -1, head_dim)
                selected = torch.take_along_dim(
                    kv_flat.unsqueeze(1).expand(-1, stop - start, -1, -1), gather_index, dim=2
                )
                logits = torch.einsum("bhcd,bckd->bhck", query[:, :, start:stop], selected) * self.scaling
                logits = logits.float().masked_fill(~valid[:, start:stop].unsqueeze(1), float("-inf"))
                target[:, start:stop] = torch.softmax(logits, dim=-1).mean(dim=1)
        return target

    def _attach_indexer_loss(
        self, activation: torch.Tensor, target: torch.Tensor, topk_indices: torch.Tensor, indexer_scores: torch.Tensor
    ) -> torch.Tensor:
        """Attach the indexer KL loss through Megatron-Core's indexer autoscaler."""
        if topk_indices is None or not self._indexer_loss_active():
            return activation
        valid = topk_indices >= 0
        if not bool(valid.any()):
            return activation
        log_probs = torch.log_softmax(indexer_scores.float().masked_fill(~valid, float("-inf")), dim=-1)
        kl = indexer_kl_per_row(target, log_probs)
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
        qkv = self.q_b_proj(q_residual)[0].view(
            batch_size, seq_len, self.local_num_heads, self.qk_head_dim + self.v_head_dim
        )
        query, gate = torch.chunk(qkv, 2, dim=-1)
        query = query.transpose(1, 2)
        gate = gate.reshape(batch_size, seq_len, -1)

        kv_latents = self.kv(x)[0]
        if self.kv_emb is not None and input_ids is not None and input_ids.shape[1] >= kv_latents.shape[1]:
            kv_latents = kv_latents + self.kv_emb(input_ids[:, -kv_latents.shape[1] :]).to(kv_latents.dtype)
        kv = self.kv_norm(self.kv_up(kv_latents)[0]).unsqueeze(1)

        indexer_scores = None
        if self.mtp_dense_attention:
            topk_indices = None
            mask = build_causal_mask(attention_mask, batch_size, seq_len, kv.shape[2], hidden_states.device)
        elif self.skip_topk:
            if prev_topk_indices is None:
                raise ValueError("Shared indexed-attention layers require top-k indices from a full indexer layer.")
            topk_indices = prev_topk_indices
        else:
            need_grad = self._indexer_loss_active()
            with torch.set_grad_enabled(need_grad):
                topk_indices, indexer_scores = self.indexer(x, q_residual, attention_mask, input_ids)

        if not self.mtp_dense_attention:
            mask = build_topk_mask(topk_indices, kv.shape[2])
        kv_heads = kv.expand(-1, query.shape[1], -1, -1)
        attention_output = F.scaled_dot_product_attention(
            query, kv_heads, kv_heads, attn_mask=mask.unsqueeze(1), scale=self.scaling
        ).transpose(1, 2)

        if indexer_scores is not None and self._indexer_loss_active():
            target = self._indexer_attention_target(query, kv, topk_indices)
            attention_output = self._attach_indexer_loss(attention_output, target, topk_indices, indexer_scores)

        # Self-position exclusion: remove the component along the query's own (normalized) value.
        positions = torch.arange(seq_len, device=hidden_states.device)
        self_values = F.normalize(kv[:, 0].index_select(1, positions).unsqueeze(2), dim=-1)
        attention_output = attention_output - (attention_output * self_values).sum(dim=-1, keepdim=True) * self_values
        attention_output = attention_output.reshape(batch_size, seq_len, -1).contiguous()
        attention_output = attention_output * torch.sigmoid(gate.to(attention_output.dtype))
        # A shared layer must keep forwarding the indices it reused, otherwise a chain of two
        # shared layers loses them and the second one has nothing to reuse.
        self.last_topk_indices = topk_indices if (self.next_skip_topk or self.skip_topk) else None
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
        # Token-wise MLP: the sequence-parallel split is handled by the surrounding stream path.
        self.linear_config = copy.copy(config)
        self.linear_config.sequence_parallel = False
        # Megatron-Core's gated path applies the clamp and the activation; the dense FFN is gated
        # squared-ReLU (routed experts keep the global activation, SiLU in the reference).
        self.linear_config.activation_func = squared_relu
        self.linear_config.use_te_activation_func = False
        self.linear_config.bias_activation_fusion = False
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
        limit = self.config.activation_func_clamp_value
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        # The DeepEmbed scale is a per-channel vector, not Megatron-Core's scalar
        # `per_token_scale`, so the activation and the scale stay here.
        hidden = self.activation_func(gate) * up * self._deepemb_scale(x, input_ids)
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


class ShensiHyperConnection(MegatronModule):
    """Hyper-connection streams: one projection predicts the pre, route and post coefficients.

    ``x_{l+1} = H_res^T @ x_l + H_post^T @ F(H_pre @ x_l)`` with the reference parameterization:
    a single ``fn`` matrix plus per-group ``base``/``scale`` produce the coefficient groups.
    """

    def __init__(self, config: ShensiTransformerConfig, is_mlp: bool) -> None:
        super().__init__(config=config)
        self.hc_mult = int(config.hc_mult)
        self.hc_active_streams = int(config.hc_active_streams)
        self.hc_fixed_streams = int(config.hc_fixed_streams)
        self.hidden_size = int(config.hidden_size)
        self.num_active_routed_streams = self.hc_active_streams - self.hc_fixed_streams
        self.num_routed_streams = self.hc_mult - self.hc_fixed_streams
        kernels = list(config.hc_conv_kernels) if is_mlp else []
        self.num_features = 1 + len(kernels)
        self.num_route_coefficients = self.num_routed_streams if self.hc_active_streams < self.hc_mult else 0
        self.num_post_coefficients = self.hc_active_streams * self.num_features
        num_coefficients = self.hc_mult + self.num_route_coefficients + self.num_post_coefficients
        self.rms_norm_eps = float(config.layernorm_epsilon)
        # Training-only: the reference has no balancing objective, but per-token top-k stream
        # routing starves an adapter whenever its stream is never selected.
        self.router_aux_loss_coef = float(getattr(config, "hc_router_aux_loss_coef", 0.0) or 0.0)

        self.fn = nn.Parameter(torch.zeros(num_coefficients, self.hc_mult * self.hidden_size))
        self.base = nn.Parameter(torch.zeros(num_coefficients))
        self.scale = nn.Parameter(torch.ones(3))
        self.adapters = (
            nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.hidden_size, int(config.hc_adapter_rank), bias=False),
                        nn.Linear(int(config.hc_adapter_rank), self.hidden_size, bias=False),
                    )
                    for _ in range(self.num_routed_streams)
                ]
            )
            if self.num_route_coefficients > 0
            else None
        )
        self.decay_logit = (
            nn.Parameter(torch.empty(self.num_routed_streams)) if self.num_route_coefficients > 0 else None
        )
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
        mark_keep_in_fp32(self.base)
        mark_keep_in_fp32(self.scale)

    def forward(self, streams: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """Collapse the streams and prepare the write-back state."""
        seq_len, batch = streams.shape[0], streams.shape[1]
        hidden_size = streams.shape[-1]
        flat = F.rms_norm(streams.flatten(2).float(), (self.hc_mult * hidden_size,), None, self.rms_norm_eps)
        weights = F.linear(flat, self.fn.float())
        split_sizes = [self.hc_mult, self.num_route_coefficients, self.num_post_coefficients]
        pre_w, route_w, post_w = weights.split(split_sizes, dim=-1)
        pre_b, route_b, post_b = self.base.split(split_sizes, dim=-1)
        pre_scale, route_scale, post_scale = self.scale.unbind(0)
        pre_w = torch.addcmul(pre_b, pre_w, pre_scale)
        route_w = torch.addcmul(route_b, route_w, route_scale)
        post_w = torch.addcmul(post_b, post_w, post_scale)

        pre = torch.softmax(pre_w, dim=-1).to(streams.dtype)
        collapsed = torch.matmul(pre.unsqueeze(-2), streams).squeeze(-2)

        if self.num_route_coefficients > 0:
            scores, top = route_w.topk(self.num_active_routed_streams, dim=-1)
            if self.router_aux_loss_coef > 0.0 and self.training and torch.is_grad_enabled():
                collapsed = self._attach_router_balance_loss(collapsed, route_w, top)
            fixed = torch.arange(self.hc_fixed_streams, device=streams.device).expand(seq_len, batch, -1)
            indices = torch.cat([fixed, top + self.hc_fixed_streams], dim=-1)
            route_weights = torch.cat(
                [
                    torch.ones(self.hc_fixed_streams, device=streams.device, dtype=scores.dtype).expand(
                        seq_len, batch, -1
                    ),
                    scores.sigmoid(),
                ],
                dim=-1,
            )
        else:
            indices = torch.arange(self.hc_active_streams, device=streams.device).expand(seq_len, batch, -1)
            route_weights = torch.ones(
                seq_len, batch, self.hc_active_streams, device=streams.device, dtype=streams.dtype
            )

        active = streams.gather(2, indices.unsqueeze(-1).expand(-1, -1, -1, hidden_size))
        if self.decay_logit is not None:
            decay = torch.sigmoid(-self.decay_logit.to(active.dtype))
            if self.hc_fixed_streams > 0:
                decay = torch.cat([torch.ones(self.hc_fixed_streams, device=active.device, dtype=active.dtype), decay])
            mixed = active * decay[indices].unsqueeze(-1)
        else:
            mixed = active

        post = torch.sigmoid(post_w) * 2.0
        post = post.view(seq_len, batch, self.hc_active_streams, self.num_features).to(streams.dtype)
        return collapsed, {
            "residual": streams,
            "mixed": mixed,
            "post": post,
            "indices": indices,
            "route_weights": route_weights.to(streams.dtype),
        }

    def _attach_router_balance_loss(
        self, activation: torch.Tensor, route_logits: torch.Tensor, top: torch.Tensor
    ) -> torch.Tensor:
        """Switch-style load balancing over the routed streams, attached through the aux scaler.

        ``load`` is the (non-differentiable) fraction of tokens each routed stream won; ``prob``
        is the mean sigmoid gate over all routed streams, normalized across them. Minimizing their
        dot product pushes the gates of over-subscribed streams down, which is what makes every
        adapter trainable.
        """
        routed = self.num_routed_streams
        load = torch.zeros(routed, device=top.device, dtype=torch.float32)
        load.index_add_(0, top.reshape(-1), torch.ones(top.numel(), device=top.device, dtype=torch.float32))
        load = load / float(top.numel())
        prob = route_logits.sigmoid().float().mean(dim=(0, 1))
        prob = prob / prob.sum().clamp_min(1e-9)
        loss = (load.detach() * prob).sum() * routed * self.router_aux_loss_coef
        return MoEAuxLossAutoScaler.apply(activation, loss)

    def write_back(self, output: torch.Tensor, state: dict, residual: torch.Tensor) -> torch.Tensor:
        """Write the sublayer output (and its causal features) back into the active streams."""
        if self.convs is None:
            addition = state["post"] * output.unsqueeze(2)
        else:
            # The depthwise conv reads the full sequence, then returns to the local SP chunk.
            tp_group = _tp_group() if self.config.sequence_parallel else None
            sp = tp_group is not None and tp_group.size() > 1
            conv_source = gather_from_sequence_parallel_region(output, group=tp_group) if sp else output
            conv_input = conv_source.permute(1, 2, 0).to(
                self.convs[0].weight.dtype, memory_format=torch.contiguous_format
            )
            branches = [output]
            for conv in self.convs:
                conv_output = conv(conv_input)[..., : conv_source.shape[0]].permute(2, 0, 1).to(output.dtype)
                branches.append(
                    scatter_to_sequence_parallel_region(conv_output, group=tp_group) if sp else conv_output
                )
            addition = torch.matmul(state["post"], torch.stack(branches, dim=2))

        if self.adapters is not None:
            for slot in range(self.hc_fixed_streams, self.hc_active_streams):
                selected = state["indices"][:, :, slot] - self.hc_fixed_streams
                slot_update = None
                for expert_idx, adapter in enumerate(self.adapters):
                    candidate = adapter(output)
                    slot_update = (
                        candidate
                        if slot_update is None
                        else torch.where((selected == expert_idx).unsqueeze(-1), candidate, slot_update)
                    )
                addition[:, :, slot, :] += slot_update * state["route_weights"][..., slot].unsqueeze(-1)

        if self.hc_fixed_streams > 0:
            updated = state["mixed"] + addition
            updated[:, :, : self.hc_fixed_streams, :] += residual.unsqueeze(2)
            residual_streams = state["residual"]
        else:
            updated = state["mixed"] + addition
            residual_streams = state["residual"] + residual.unsqueeze(2)
        updated = updated.to(residual_streams.dtype)
        return residual_streams.scatter(2, state["indices"].unsqueeze(-1).expand_as(updated), updated)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            # Only the route rows start random; the pre/post rows stay zero so the initial
            # mixing is uniform (reference initialization).
            self.fn.zero_()
            if self.num_route_coefficients > 0:
                self.fn[self.hc_mult : self.hc_mult + self.num_route_coefficients].normal_(0.0, std)
            self.base.zero_()
            self.scale.fill_(1.0)
            if self.adapters is not None:
                for adapter in self.adapters:
                    adapter[0].weight.normal_(0.0, std)
                    adapter[1].weight.zero_()
                self.decay_logit.normal_(-3.0, 0.5)
            if self.convs is not None:
                for conv in self.convs:
                    conv.weight.zero_()


class ShensiHyperHead(MegatronModule):
    """Contract the residual streams into one hidden state for the output head."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.streams = int(config.hc_mult)
        self.hidden_size = int(config.hidden_size)
        self.eps = float(config.hc_eps)
        self.eps_norm = float(config.layernorm_epsilon)
        self.hc_fn = nn.Parameter(torch.zeros(self.streams, self.streams * self.hidden_size))
        self.hc_base = nn.Parameter(torch.zeros(self.streams))
        self.hc_scale = nn.Parameter(torch.ones(1))
        _mark_tp_replicated(self)
        mark_keep_in_fp32(self.hc_fn)
        mark_keep_in_fp32(self.hc_base)
        mark_keep_in_fp32(self.hc_scale)

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        flat = F.rms_norm(streams.flatten(2).float(), (self.streams * self.hidden_size,), None, self.eps_norm)
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            self.hc_fn.normal_(0.0, std)
            self.hc_base.zero_()
            self.hc_scale.fill_(1.0)


class ShensiAttentionResidual(MegatronModule):
    """Channel gates, low-rank depth routing and a ReZero gate over attention-residual blocks."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.hidden_size = int(config.hidden_size)
        self.rank = int(config.attn_res_rank)
        self.eps = float(config.layernorm_epsilon)
        self.g_proj = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.q_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.scale = nn.Parameter(torch.tensor([1.0, 0.1]))
        _mark_tp_replicated(self)
        mark_keep_in_fp32(self.scale)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (self.hidden_size,), None, self.eps)

    def forward(
        self,
        prefix: torch.Tensor,
        delta: Optional[torch.Tensor],
        blocks: torch.Tensor,
        num_blocks: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update the running residual and read the completed block summaries into the branch."""
        out_dtype = prefix.dtype
        prefix = prefix.float()
        delta = delta.float() if delta is not None else None
        plain = prefix + delta if delta is not None else prefix
        state = self._norm(plain)
        state_proj = state.to(self.g_proj.weight.dtype)
        logit_scale, residual_gate = self.scale.float().unbind(0)

        gates = self.g_proj(state_proj).float()
        decay, erase, write = gates.split(self.hidden_size, dim=-1)
        decay = 1.0 + torch.tanh(decay)
        if delta is None:
            updated = decay * prefix
        else:
            write = 1.0 + torch.tanh(write)
            erase = torch.tanh(erase)
            updated = torch.addcmul(decay * prefix, write, delta)
            direction = F.normalize(delta, dim=-1)
            projection = torch.einsum("...d,...d->...", direction, updated)[..., None]
            updated = torch.addcmul(updated, erase * direction, projection, value=-1.0)

        branch = updated
        if num_blocks > 0:
            blk = blocks[..., :num_blocks, :].float()
            query = self.q_proj(state_proj).float()
            keys = F.normalize(self.k_proj(self._norm(blk).to(self.k_proj.weight.dtype)).float(), dim=-1)
            logits = torch.einsum("...r,...nr->...n", query, keys) * logit_scale
            weights = torch.softmax(logits, dim=-1)
            branch = updated + torch.einsum("...n,...nd->...d", weights, blk)

        plain_rms = (plain.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        branch_rms = (branch.square().mean(dim=-1, keepdim=True) + self.eps).sqrt()
        output_scale = (plain_rms / branch_rms).detach().clamp(max=10.0)
        residual_scale = torch.tanh(residual_gate)
        updated = torch.addcmul(plain, residual_scale * updated, output_scale)
        output = updated if num_blocks == 0 else torch.addcmul(plain, residual_scale * branch, output_scale)
        return output.to(out_dtype), updated.to(out_dtype)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            self.g_proj.weight.normal_(0.0, std)
            self.g_proj.bias.zero_()
            self.q_proj.weight.normal_(0.0, std)
            self.k_proj.weight.normal_(0.0, std)
            self.scale.copy_(torch.tensor([1.0, 0.1], device=self.scale.device, dtype=self.scale.dtype))


class ShensiAttnResBank(MegatronModule):
    """Persistent low-rank residuals: block summaries and the inter-layer stream latents.

    Both bases stay in fp32 at rest; encode/decode cast them to the activation dtype so the GEMMs
    run with tensor cores. The exact embedding head and its token bias stay outside the compressed
    bank, matching the reference.
    """

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.rank = int(config.attn_res_block_summary_rank)
        self.stream_rank = int(config.hc_stream_rank)
        self.hidden_size = int(config.hidden_size)
        self.basis = nn.Parameter(torch.zeros(self.rank, self.hidden_size))
        self.stream_basis = nn.Parameter(torch.zeros(self.stream_rank, self.hidden_size))
        self.token_down = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.token_up = nn.Linear(self.rank, self.hidden_size, bias=False)
        _mark_tp_replicated(self)
        mark_keep_in_fp32(self.basis)
        mark_keep_in_fp32(self.stream_basis)

    def initialize(self, inputs_embeds: torch.Tensor, num_blocks: int) -> torch.Tensor:
        self._token_bias = self.token_up(self.token_down(inputs_embeds))
        head = inputs_embeds.unsqueeze(-2)
        bias = self._token_bias.unsqueeze(-2).expand(*self._token_bias.shape[:-1], num_blocks, self.hidden_size)
        return torch.cat([head, bias], dim=-2)

    def finalize(self, blocks: torch.Tensor, prefix_sum: torch.Tensor, num_blocks: int) -> torch.Tensor:
        """Encode the completed rows and write the last partial block; return the decoded bank."""
        encoded = self.encode(blocks[..., 1:, :] - self._token_bias.unsqueeze(-2))
        summary = self.encode((prefix_sum - self._token_bias).unsqueeze(-2))
        encoded = encoded.index_copy(
            -2, torch.tensor([num_blocks - 1], device=prefix_sum.device, dtype=torch.long), summary
        )
        decoded = self.decode(encoded) + self._token_bias.unsqueeze(-2)
        return torch.cat([blocks[..., :1, :], decoded], dim=-2)

    def encode(self, blocks: torch.Tensor) -> torch.Tensor:
        return F.linear(blocks, self.basis.to(blocks.dtype))

    def decode(self, blocks: torch.Tensor) -> torch.Tensor:
        return F.linear(blocks, self.basis.to(blocks.dtype).t())

    def encode_streams(self, streams: torch.Tensor) -> torch.Tensor:
        return F.linear(streams, self.stream_basis.to(streams.dtype))

    def decode_streams(self, latents: torch.Tensor) -> torch.Tensor:
        return F.linear(latents, self.stream_basis.to(latents.dtype).t())

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            eye = torch.eye(max(self.hidden_size, self.rank), device=self.basis.device, dtype=self.basis.dtype)
            self.basis.copy_(eye[: self.rank, : self.hidden_size])
            eye = torch.eye(
                max(self.hidden_size, self.stream_rank), device=self.stream_basis.device, dtype=self.stream_basis.dtype
            )
            self.stream_basis.copy_(eye[: self.stream_rank, : self.hidden_size])
            self.token_down.weight.normal_(0.0, 0.02)
            self.token_up.weight.zero_()


class ShensiStreamEmbeddings(MegatronModule):
    """Embed a token sequence into the hyper-connection streams.

    Stream 0 is the shared token embedding; each auxiliary stream owns an independent token table
    and adds a depthwise causal convolution for local temporal context.
    """

    def __init__(self, config: ShensiTransformerConfig, vocab_size: Optional[int] = None) -> None:
        super().__init__(config=config)
        self.num_streams = int(config.hc_mult)
        self.hidden_size = int(config.hidden_size)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.hidden_size, self.hidden_size, kernel, groups=self.hidden_size, padding=kernel - 1)
                for kernel in config.hc_conv_kernels
            ]
        )
        # The table follows Megatron-Core's vocabulary layout: padded to the same size as the token
        # embedding, and sharded across TP when there is more than one rank. The bridge crops it
        # back to the reference vocabulary on export.
        vocab = int(vocab_size or getattr(config, "actual_vocab_size", None) or getattr(config, "vocab_size", 0) or 0)
        tp_group = _tp_group()
        tp_size = tp_group.size() if tp_group is not None else 1
        self.sharded_embedding = bool(self.num_streams > 1 and tp_size > 1)
        self.reduce_scatter = bool(getattr(config, "sequence_parallel", False))
        if self.num_streams <= 1:
            self.token_embeddings = None
        elif self.sharded_embedding:
            self.token_embeddings = VocabParallelEmbedding(
                vocab,
                (self.num_streams - 1) * self.hidden_size,
                init_method=getattr(config, "init_method", None) or torch.nn.init.normal_,
                reduce_scatter_embeddings=self.reduce_scatter,
                config=config,
                tp_group=tp_group,
            )
        else:
            self.token_embeddings = nn.Embedding(vocab, (self.num_streams - 1) * self.hidden_size)
        # The convolutions stay replicated; the embedding shards its own gradients across TP.
        _mark_tp_replicated(self.convs)

    def forward(self, inputs_embeds: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the streams for ``[s, b, h]`` embeddings (Megatron's sequence-first layout)."""
        seq_len = inputs_embeds.shape[0]
        conv_dtype = self.convs[0].weight.dtype if self.convs else inputs_embeds.dtype
        tp_group = _tp_group() if self.config.sequence_parallel else None
        sp = tp_group is not None and tp_group.size() > 1
        convs = []
        if self.convs:
            # The depthwise conv reads the full sequence, then returns to the local SP chunk.
            conv_source = gather_from_sequence_parallel_region(inputs_embeds, group=tp_group) if sp else inputs_embeds
            conv_input = conv_source.permute(1, 2, 0).to(conv_dtype)
            for conv in self.convs:
                conv_output = conv(conv_input)[..., : conv_source.shape[0]].permute(2, 0, 1).to(inputs_embeds.dtype)
                convs.append(scatter_to_sequence_parallel_region(conv_output, group=tp_group) if sp else conv_output)

        streams = [inputs_embeds]
        if self.token_embeddings is None:
            streams.extend(convs)
        elif input_ids is None:
            streams.extend(inputs_embeds + conv for conv in convs)
        else:
            ids = input_ids
            if not self.sharded_embedding and sp and ids.shape[1] != seq_len:
                rank = tp_group.rank()
                ids = ids[:, rank * seq_len : (rank + 1) * seq_len]
            tokens = self.token_embeddings(ids.to(self.token_embeddings.weight.device))
            if self.sharded_embedding and self.reduce_scatter:
                # Vocab-parallel lookup returns the local sequence chunk.
                tokens = tokens.view(seq_len, ids.shape[0], self.num_streams - 1, self.hidden_size)
            else:
                tokens = tokens.view(ids.shape[0], seq_len, self.num_streams - 1, self.hidden_size).transpose(0, 1)
            for index, conv in enumerate(convs):
                streams.append(tokens[:, :, index, :].to(inputs_embeds.dtype) + conv)
        return torch.stack(streams, dim=2)

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del std, hidden_size
        with torch.no_grad():
            for conv in self.convs:
                conv.weight.zero_()
                if conv.bias is not None:
                    conv.bias.zero_()


_ROSA_CE_TOKEN_CHUNK = 1024


class ShensiROSAEmbedding(MegatronModule):
    """Inject ROSA suffix-automaton predictions into every hyper-connection stream."""

    def __init__(self, config: ShensiTransformerConfig) -> None:
        super().__init__(config=config)
        self.embed_dim = int(config.rosa_embed_dim or config.hidden_size)
        self.hidden_size = int(config.hidden_size)
        self.streams = int(config.hc_mult)
        self.vocab_size = int(config.actual_vocab_size or config.vocab_size or 0)
        self.embed_tokens = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_proj = nn.Linear(self.embed_dim, self.hidden_size, bias=False)
        self.stream_gate = nn.Linear(self.hidden_size, self.streams, bias=True)
        self.rosa_loss_coef = float(getattr(config, "rosa_loss_coef", 0.0) or 0.0)
        self.num_rosa_layers = max(len(getattr(config, "rosa_layer_ids", None) or []), 1)
        _mark_tp_replicated(self)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Megatron-Core's vocab-parallel embedding masks out-of-partition ids; -1 is clamped
        # here and zeroed by the validity mask.
        valid = token_ids >= 0
        embeddings = self.embed_tokens(token_ids.clamp_min(0))
        return embeddings * valid.unsqueeze(-1).to(embeddings.dtype)

    def _rosa_token_loss(self, rosa_features: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Cross-entropy of the injected ROSA feature against the shifted next-token labels."""
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

        weight = self.embed_tokens.weight
        target_t = target.transpose(0, 1).contiguous()
        # Stream the vocabulary projection in token chunks: the fp32 logits of a full batch would
        # be [tokens, vocab] (1.3 GB at the reference geometry for a 2.5k-token microbatch).
        features_2d = rosa_features.reshape(-1, rosa_features.shape[-1])
        target_1d = target_t.reshape(-1)
        denom = (target_1d != -100).sum().clamp(min=1)
        total = features_2d.new_zeros((), dtype=torch.float32)
        for start in range(0, features_2d.shape[0], _ROSA_CE_TOKEN_CHUNK):
            stop = min(start + _ROSA_CE_TOKEN_CHUNK, features_2d.shape[0])
            logits = F.linear(features_2d[start:stop], weight).float()
            chunk_target = target_1d[start:stop]
            if weight.shape[0] != self.vocab_size and tp_size > 1:
                per_token = vocab_parallel_cross_entropy(
                    logits.unsqueeze(1), chunk_target.unsqueeze(1), tp_group=tp_group
                ).reshape(-1)
            else:
                per_token = F.cross_entropy(logits, chunk_target, ignore_index=-100, reduction="none")
            total = total + per_token[chunk_target != -100].sum().float()
        aux = total / denom.float()
        aux = aux * (self.rosa_loss_coef / float(self.num_rosa_layers))
        if getattr(self.config, "calculate_per_token_loss", False):
            aux = aux * torch.full(
                (), rosa_features.shape[0] * rosa_features.shape[1], dtype=aux.dtype, device=aux.device
            )
        return aux

    def forward(
        self, streams: torch.Tensor, rosa_token_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # ROSA ids are batch-first; the stream path is sequence-first.
        features = self.embed_proj(self.embed(rosa_token_ids).to(self.embed_proj.weight.dtype)).transpose(0, 1)
        gate = torch.sigmoid(self.stream_gate(streams.mean(dim=2))).unsqueeze(-1)
        out = gate * features.unsqueeze(2)
        aux = self._rosa_token_loss(out.mean(dim=2), labels)
        if aux is not None:
            out = MoEAuxLossAutoScaler.apply(out, aux)
        return out

    def init_shensi_weights(self, std: float = 0.02, hidden_size: Optional[int] = None) -> None:
        del hidden_size
        with torch.no_grad():
            self.embed_tokens.weight.normal_(0.0, std)
            self.embed_proj.weight.normal_(0.0, std)
            self.stream_gate.weight.zero_()
            self.stream_gate.bias.fill_(-4.0)


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


def _compiled_hyper_connection_call(module, streams):
    return module(streams)


def _compiled_attention_residual_call(module, prefix, delta, blocks, num_blocks, block_idx):
    visible = torch.cat([blocks[..., : 1 + block_idx, :], prefix.unsqueeze(-2)], dim=-2)
    return module(prefix, delta, visible, num_blocks)


# One compiled function serves every layer instance: the graph is keyed on this callable, not on
# the module, so the first layer pays the compile cost and the rest reuse it.
_compiled_hyper_connection = torch.compile(_compiled_hyper_connection_call)
_compiled_attention_residual = torch.compile(_compiled_attention_residual_call)


class ShensiTransformerLayer(TransformerLayer):
    """Decoder layer implementing the Shensi stream / attention-residual / ROSA flow."""

    def __init__(
        self,
        config,
        submodules: TransformerLayerSubmodules,
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
        # These two bridge-owned modules keep large fp32 tensors for their backward. Recomputing
        # them is cheaper than reading those tensors back, so do it whenever the stack-level
        # recompute is off (with `full`/`selective` the whole iteration is already checkpointed).
        self.recompute_short_modules = str(getattr(config, "recompute_granularity", "none")) == "none"
        self.compile_short_modules = bool(getattr(config, "shensi_compile_modules", False))
        self.attn_hc = ShensiHyperConnection(config, is_mlp=False)
        self.ffn_hc = ShensiHyperConnection(config, is_mlp=True)
        self.self_attention_res = ShensiAttentionResidual(config)
        self.mlp_res = ShensiAttentionResidual(config)
        self.rosa = (
            ShensiROSAEmbedding(config)
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

        if self.rosa is not None and rosa_token_ids is not None:
            streams = streams + self.rosa(streams, rosa_token_ids, labels)

        collapsed, state = self._hyper_connection(self.attn_hc, streams)
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
        attn_residual, _ = self._attention_residual(
            self.self_attention_res, partial_block, attn_output, blocks, 1 + self.block_idx + 1
        )
        attn_residual = attn_residual - (partial_block + attn_output)
        streams = self.attn_hc.write_back(attn_output, state, attn_residual)
        partial_block = partial_block + attn_output

        collapsed, state = self._hyper_connection(self.ffn_hc, streams)
        mlp_input = self.pre_mlp_layernorm(collapsed)
        if self.is_dense:
            mlp_output = self.mlp(mlp_input, input_ids=input_ids)
        else:
            mlp_output = self.mlp(mlp_input)
        if isinstance(mlp_output, tuple):
            mlp_output = mlp_output[0]
        mlp_residual, _ = self._attention_residual(
            self.mlp_res, partial_block, mlp_output, blocks, 1 + self.block_idx + 1
        )
        mlp_residual = mlp_residual - (partial_block + mlp_output)
        streams = self.ffn_hc.write_back(mlp_output, state, mlp_residual)
        partial_block = partial_block + mlp_output

        if hidden_states.dim() == 3:
            streams = streams.mean(dim=2)
        if self.is_mtp_layer:
            return streams, None
        return streams, partial_block, blocks, topk_indices

    def _hyper_connection(self, module, streams):
        """Read the streams through one hyper-connection, compiling or recomputing it if enabled."""
        if self.compile_short_modules:
            if self.recompute_short_modules and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(
                    _compiled_hyper_connection, module, streams, use_reentrant=False
                )
            return _compiled_hyper_connection(module, streams)
        if self.recompute_short_modules and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(module, streams, use_reentrant=False)
        return module(streams)

    def _attention_residual(self, module, prefix, delta, blocks, num_blocks):
        """Run one attention-residual update over the visible block rows.

        The visible view (committed rows + the in-progress prefix) is built inside the
        recompute region, so the concatenated tensor is never retained for backward.
        """
        block_idx = self.block_idx
        if self.compile_short_modules:
            if self.recompute_short_modules and self.training and torch.is_grad_enabled():
                return torch.utils.checkpoint.checkpoint(
                    _compiled_attention_residual,
                    module,
                    prefix,
                    delta,
                    blocks,
                    num_blocks,
                    block_idx,
                    use_reentrant=False,
                )
            return _compiled_attention_residual(module, prefix, delta, blocks, num_blocks, block_idx)

        def run(prefix, delta, blocks):
            visible = torch.cat([blocks[..., : 1 + block_idx, :], prefix.unsqueeze(-2)], dim=-2)
            return module(prefix, delta, visible, num_blocks)

        if self.recompute_short_modules and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(run, prefix, delta, blocks, use_reentrant=False)
        return run(prefix, delta, blocks)


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
            linear_key=duplicated,
            linear_key_up=duplicated,
            k_norm=backend.layer_norm(),
            linear_weights_proj=duplicated,
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
        submodules=TransformerLayerSubmodules(
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
    """State carried across pipeline stages: embeddings, streams, blocks and ROSA ids."""

    inputs_embeds: Tensor
    streams: Tensor
    prefix_sum: Tensor
    blocks: Tensor
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
    topk_rows: int = 0,
    index_topk: int = 0,
) -> Tensor:
    """Append the stream state to a stage output as extra sequence rows."""
    seq_len, batch, hidden = stage_output.shape
    num_streams = state.streams.shape[2]
    num_blocks = state.blocks.shape[2]
    streams = state.streams.permute(0, 2, 1, 3).reshape(seq_len * num_streams, batch, hidden)
    blocks = state.blocks.permute(0, 2, 1, 3).reshape(seq_len * num_blocks, batch, hidden)
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
    offset += seq_len * num_streams
    prefix_sum = payload[offset : offset + seq_len]
    offset += seq_len
    blocks = payload[offset : offset + seq_len * num_blocks].reshape(seq_len, num_blocks, batch, hidden)
    blocks = blocks.permute(0, 2, 1, 3)
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
        blocks=blocks,
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
        # Boundary modules follow Megatron-Core's pipeline convention: the stage that consumes a
        # module owns it, so other stages do not carry its parameters.
        self.norm = _build_rms_norm(cfg, hidden) if self.post_process else None
        self.hc_head = ShensiHyperHead(cfg) if self.post_process else None
        self.stream_embeddings = ShensiStreamEmbeddings(cfg, vocab_size=self.vocab_size) if self.pre_process else None
        self.attn_bank = ShensiAttnResBank(cfg)
        self.output_attn_res = ShensiAttentionResidual(cfg) if self.post_process else None
        self._init_shensi_only_weights()
        self.tie_mlp()
        apply_fp32_keep(self, verbose=True)
        bind = getattr(self.decoder, "set_stream_components", None)
        if bind is None:
            raise RuntimeError("ShensiModel needs a ShensiDecoderStack with set_stream_components")
        bind(
            hc_head=self.hc_head,
            stream_embeddings=self.stream_embeddings,
            bank=self.attn_bank,
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
        # The auxiliary stream tables start as cyclic copies of the shared token table. Both use
        # Megatron-Core's vocabulary layout, so the local shards line up without a gather.
        streams = getattr(self, "stream_embeddings", None)
        if streams is not None and streams.token_embeddings is not None:
            source = self.shared_embedding_or_output_weight()
            if source is not None:
                with torch.no_grad():
                    table = streams.token_embeddings.weight
                    rows = min(source.shape[0], table.shape[0])
                    per_stream = int(self.config.hidden_size)
                    for index in range(streams.num_streams - 1):
                        start = index * per_stream
                        table[:rows, start : start + per_stream].copy_(source[:rows].to(table.dtype))

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
            # The reference ties the whole deep-embedding group (table and both projections) and
            # the checkpoint stores it under the first dense layer only.
            first_mlp = dense_layers[0].mlp
            for layer in dense_layers[1:]:
                layer.mlp.deepemb = first_mlp.deepemb
                layer.mlp.deepemb_a_proj = first_mlp.deepemb_a_proj
                layer.mlp.deepemb_b_proj = first_mlp.deepemb_b_proj

        indexed_layers = [
            layer for layer in layers if getattr(layer, "is_write", False) and hasattr(layer.self_attention, "kv_emb")
        ]
        if len(indexed_layers) > 1 and indexed_layers[0].self_attention.kv_emb is not None:
            first = indexed_layers[0].self_attention
            for layer in indexed_layers[1:]:
                layer.self_attention.kv_emb = first.kv_emb

        full_indexers = [
            layer.self_attention
            for layer in indexed_layers
            if getattr(layer.self_attention, "indexer", None) is not None
            and layer.self_attention.indexer.linear_key_emb is not None
        ]
        if len(full_indexers) > 1:
            first_emb = full_indexers[0].indexer.linear_key_emb
            for attn in full_indexers[1:]:
                attn.indexer.linear_key_emb = first_emb

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
                layer.rosa.embed_tokens = tied_embedding
        elif len(rosa_layers) > 1:
            first = rosa_layers[0].rosa.embed_tokens
            for layer in rosa_layers[1:]:
                layer.rosa.embed_tokens = first


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
