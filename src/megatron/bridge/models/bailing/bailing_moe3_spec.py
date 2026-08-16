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

"""Megatron-Core module spec used by the Ling 3.0 Bridge."""

import copy
from typing import Any

from megatron.core.extensions.transformer_engine import TEColumnParallelLinear, TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import MLASelfAttention


class BailingMoe3DirectQMLASelfAttention(MLASelfAttention):
    """MLA adapter for Flash's direct-Q plus KV-only RMSNorm contract.

    The current MCore resolver treats ``qk_layernorm=True`` as a request to
    fuse a Q norm when ``q_lora_rank`` is ``None``.  Ling 3.0 Flash instead has
    a plain ``q_proj`` and a standalone ``kv_a_layernorm``.  Keep this small
    compatibility adapter in Bridge so the MCore implementation remains
    untouched; it only changes module selection, while the inherited forward,
    backward, and sharding implementation remains MCore's.
    """

    def _resolve_qk_norm_config(self, submodules) -> dict[str, Any]:
        """Select plain direct-Q and standalone KV normalization modules."""
        return {
            "linear_q_proj": submodules.linear_q_proj,
            "linear_q_up_proj": IdentityOp,
            "linear_kv_up_proj": submodules.linear_kv_up_proj,
            "q_layernorm": IdentityOp,
            "kv_layernorm": submodules.kv_layernorm,
        }


def bailing_moe3_hybrid_stack_spec(config: object | None = None) -> ModuleSpec:
    """Return the Transformer Engine Hybrid spec required by Ling 3.0.

    The official generic Hybrid spec intentionally leaves MLA output gating and
    low-rank Q/KV normalization disabled.  Both public variants store the KV
    layernorm vector and a head-wise output gate.  Tiny additionally stores the
    low-rank Q layernorm, while Flash uses direct Q and must not create that
    parameter.

    Args:
        config: Optional provider.  Its ``q_lora_rank`` selects the low-rank Q
            spec used by Tiny versus the direct-Q spec used by Flash.  Omitting
            it preserves the Tiny spec for legacy serialized DCP configs.

    Returns:
        A fresh, serializable module spec for the Ling 3.0 HybridModel.
    """
    spec = copy.deepcopy(hybrid_stack_spec)
    use_low_rank_q = True if config is None else getattr(config, "q_lora_rank", None) is not None
    mla_attention_spec = spec.submodules.mla_layer.submodules.self_attention
    mla_attention = spec.submodules.mla_layer.submodules.self_attention.submodules
    if not use_low_rank_q:
        mla_attention_spec.module = BailingMoe3DirectQMLASelfAttention
    mla_attention.q_layernorm = TENorm if use_low_rank_q else IdentityOp
    mla_attention.kv_layernorm = TENorm
    mla_attention.linear_gate = TEColumnParallelLinear
    return spec


def bailing_moe3_tiny_hybrid_stack_spec() -> ModuleSpec:
    """Return the Ling 3.0 Tiny low-rank-MLA Hybrid spec."""
    return bailing_moe3_hybrid_stack_spec()


def bailing_moe3_flash_hybrid_stack_spec(config: object | None = None) -> ModuleSpec:
    """Return the Ling 3.0 Flash direct-Q MLA Hybrid spec."""
    return bailing_moe3_hybrid_stack_spec(config if config is not None else object())
