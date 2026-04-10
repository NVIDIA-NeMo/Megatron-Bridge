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

"""
Provider for ERNIE 4.5 text-only MoE model.

Maps HuggingFace Ernie4_5_MoEConfig to Megatron-Core TransformerConfig
and provides model instantiation logic for the standard single-pool MoE
architecture (64 experts, top-6 routing, shared experts).
"""

from dataclasses import dataclass
from typing import Callable

import torch.nn.functional as F

from megatron.bridge.models.gpt_provider import GPTModelProvider


def _ernie45_decoder_block_spec(config: "Ernie45ModelProvider", vp_stage: int | None = None):
    """Create a decoder block spec that respects ``moe_layer_freq``.

    The default ``GPTModelProvider.transformer_layer_spec`` calls
    ``get_gpt_layer_with_transformer_engine_spec`` which returns a single
    MoE layer spec applied uniformly to ALL layers, ignoring
    ``moe_layer_freq``.

    ERNIE 4.5 has mixed dense/MoE layers (layer 0 is dense, layers 1-N
    are MoE).  This function uses ``get_gpt_decoder_block_spec`` which
    calls ``get_gpt_decoder_layer_specs`` — the code path that parses
    ``config.moe_layer_freq`` and creates per-layer specs (dense for
    pattern=0, MoE for pattern=1).
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    return get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )


@dataclass
class Ernie45ModelProvider(GPTModelProvider):
    """
    Model provider for ERNIE 4.5 text-only MoE.

    This provider extends GPTModelProvider with ERNIE 4.5-specific
    architectural defaults (normalization, activation, RoPE).

    MoE-specific runtime settings (router, dispatcher, expert bias) are
    configured in ``Ernie45Bridge.provider_bridge()`` following the
    convention used by other bridges in the codebase.
    """

    # Architecture
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    hidden_dropout: float = 0.0

    # RoPE
    position_embedding_type: str = "rope"
    rotary_base: float = 500000.0
    rotary_interleaved: bool = True

    # MoE load-balancing type (architectural, not runtime)
    moe_router_load_balancing_type: str = "aux_loss"

    # Use decoder block spec that respects moe_layer_freq for mixed
    # dense/MoE layer patterns (layer 0 dense, layers 1-N MoE).
    transformer_layer_spec: Callable = _ernie45_decoder_block_spec
