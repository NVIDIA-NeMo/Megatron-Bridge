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

"""Kimi Model Providers.

This module provides backward-compatible aliases for Kimi model providers.
The base MLAModelProvider is now the recommended way to create MLA-based models.

Migration:
    Old: from megatron.bridge.models.kimi.kimi_provider import KimiK2Provider
    New: from megatron.bridge.models.mla_provider import MLAModelProvider
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Union

import torch

from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.bridge.utils.common_utils import get_rank_safe


def _warn_deprecated(old_cls: str) -> None:
    if get_rank_safe() == 0:
        warnings.warn(
            f"{old_cls} is deprecated and will be removed in a future release. "
            f"Use MLAModelProvider with MEGATRON_DEFAULTS in the bridge instead.",
            DeprecationWarning,
            stacklevel=3,
        )


@dataclass
class KimiK2Provider(MLAModelProvider):
    """Deprecated: Use MLAModelProvider with appropriate MEGATRON_DEFAULTS."""

    num_layers: int = 61
    hidden_size: int = 7168
    ffn_hidden_size: int = 18432
    num_moe_experts: int = 384
    moe_ffn_hidden_size: int = 2048
    moe_shared_expert_intermediate_size: int = 2048
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] + [1] * 60)
    num_attention_heads: int = 64
    kv_channels: int = 64
    max_position_embeddings: int = 4096
    seq_length: int = 4096
    rotary_base: float = 50000.0
    make_vocab_size_divisible_by: int = 1280
    mtp_num_layers: Optional[int] = None
    mtp_loss_scaling_factor: Optional[float] = None
    moe_router_topk: int = 8
    moe_router_num_groups: int = 1
    moe_router_group_topk: int = 1
    moe_router_topk_scaling_factor: float = 2.827
    moe_aux_loss_coeff: float = 1e-3
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_head_dim: int = 128
    qk_pos_emb_head_dim: int = 64
    v_head_dim: int = 128
    rotary_scaling_factor: float = 32
    beta_fast: float = 1.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    init_method_std: float = 0.006
    layernorm_epsilon: float = 1e-6
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vocab_size: int = 163840

    def __post_init__(self) -> None:
        _warn_deprecated("KimiK2Provider")
        super().__post_init__()
