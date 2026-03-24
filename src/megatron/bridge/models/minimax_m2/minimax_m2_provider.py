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

"""MiniMax-M2 custom layer spec with full-dimension QK normalization.

MiniMax-M2 applies RMSNorm to the entire Q/K projection (weight shape =
num_heads * head_dim) before splitting into heads. Megatron's built-in
QK norm applies per-head (weight shape = head_dim). This module bridges
the gap by applying full-partition-dimension RMSNorm inside the standard
SelfAttention flow.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from megatron.core.transformer import ModuleSpec, TransformerConfig
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint


class _FullDimRMSNorm(nn.Module):
    """RMSNorm applied across all attention heads (full Q/K dimension).

    Standard per-head QK norm normalizes over ``head_dim`` independently per head.
    This module normalizes over the *full* ``num_heads * head_dim`` dimension,
    matching HuggingFace models that use ``nn.RMSNorm(num_heads * head_dim)`` on
    the full Q/K vector before reshaping into heads.

    With TP > 1 each rank holds only ``num_heads_per_partition`` heads, so the
    sum-of-squares is all-reduced across the TP group before computing the RMS.
    This keeps the normalization denominator identical to the single-GPU case.
    """

    def __init__(self, local_dim: int, global_dim: int, tp_group_getter, eps: float = 1e-6):
        super().__init__()
        self.local_dim = local_dim
        self.global_dim = global_dim
        self._tp_group_getter = tp_group_getter
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(local_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [sq, b, num_heads_per_partition, head_dim]
        orig_shape = x.shape
        x = x.reshape(*orig_shape[:2], -1)  # [sq, b, local_dim]
        dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        # sum-of-squares over the local partition
        local_ss = x_fp32.pow(2).sum(-1, keepdim=True)  # [sq, b, 1]

        # all-reduce across TP ranks so every rank sees the global sum-of-squares
        tp_group = self._tp_group_getter()
        if tp_group is not None and dist.get_world_size(tp_group) > 1:
            dist.all_reduce(local_ss, op=dist.ReduceOp.SUM, group=tp_group)

        variance = local_ss / self.global_dim
        x = x_fp32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x.to(dtype)).reshape(orig_shape)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict] = None,
    ) -> Dict[str, "ShardedTensor"]:  # noqa: F821
        """Weight is TP-sharded along axis 0 (same as ColumnParallelLinear)."""
        state_dict = self.state_dict(prefix="", keep_vars=True)
        tp_group = self._tp_group_getter()
        if metadata is None:
            from megatron.core.transformer.utils import ensure_metadata_has_dp_cp_group

            metadata = ensure_metadata_has_dp_cp_group(metadata)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0},
            sharded_offsets,
            tp_group=tp_group,
            dp_cp_group=metadata["dp_cp_group"],
        )


def _get_tp_group():
    """Lazy accessor for the TP process group (not available at module init time)."""
    from megatron.core.parallel_state import get_tensor_model_parallel_group

    return get_tensor_model_parallel_group(check_initialized=False)


class FullDimQNorm:
    """Factory callable that creates a full-dimension RMSNorm for Q heads.

    Passed as ``q_layernorm`` in the layer spec. The ``SelfAttention`` constructor
    calls ``submodules.q_layernorm(hidden_size=head_dim, config=..., eps=...)``;
    this factory ignores the per-head ``hidden_size`` and computes the correct
    full partition dimension from ``config``.
    """

    def __new__(cls, hidden_size: int, config: TransformerConfig, eps: float = 1e-6):
        tp = config.tensor_model_parallel_size
        num_heads = config.num_attention_heads
        local_dim = (num_heads // tp) * hidden_size
        global_dim = num_heads * hidden_size
        return _FullDimRMSNorm(local_dim, global_dim, _get_tp_group, eps)


class FullDimKNorm:
    """Factory callable that creates a full-dimension RMSNorm for K heads.

    Same as ``FullDimQNorm`` but uses ``num_query_groups`` (GQA key-value heads)
    instead of ``num_attention_heads``.
    """

    def __new__(cls, hidden_size: int, config: TransformerConfig, eps: float = 1e-6):
        tp = config.tensor_model_parallel_size
        num_kv_heads = config.num_query_groups or config.num_attention_heads
        local_dim = (num_kv_heads // tp) * hidden_size
        global_dim = num_kv_heads * hidden_size
        return _FullDimRMSNorm(local_dim, global_dim, _get_tp_group, eps)


def minimax_m2_layer_spec(config: "GPTModelProvider") -> ModuleSpec:  # noqa: F821
    """Build a TE layer spec for MiniMax-M2 with full-dimension QK norm.

    Starts from the standard TE MoE spec (which handles grouped-gemm experts,
    router, etc.) and replaces the per-head ``TENorm`` Q/K layernorm with
    ``FullDimQNorm`` / ``FullDimKNorm``.
    """
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )

    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        qk_layernorm=True,
    )
    attn_sub = spec.submodules.self_attention.submodules
    attn_sub.q_layernorm = FullDimQNorm
    attn_sub.k_layernorm = FullDimKNorm
    return spec
