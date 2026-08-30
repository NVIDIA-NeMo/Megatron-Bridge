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

from __future__ import annotations

import copy

from megatron.core.extensions.transformer_engine import TENorm
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.transformer import ModuleSpec


def bailing_moe3_hybrid_stack_spec(config: object) -> ModuleSpec:
    """Return the Transformer Engine Hybrid spec required by Ling 3.0.

    Both public variants store the MLA KV layernorm vector. Tiny additionally
    stores the low-rank Q layernorm, while Flash uses direct Q and keeps the
    default identity builder. All projection builders retain MCore defaults.

    Args:
        config: Provider whose ``q_lora_rank`` selects the low-rank Q spec used
            by Tiny versus the direct-Q spec used by Flash.

    Returns:
        A fresh, serializable module spec for the Ling 3.0 HybridModel.
    """
    spec = copy.deepcopy(hybrid_stack_spec)
    mla_attention = spec.submodules.mla_layer.submodules.self_attention.submodules
    if getattr(config, "q_lora_rank", None) is not None:
        mla_attention.q_layernorm = TENorm
    mla_attention.kv_layernorm = TENorm

    return spec
