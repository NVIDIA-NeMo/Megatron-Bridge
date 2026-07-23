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

"""Step-3.5-Flash Model Provider for Megatron-Core.

Step-3.5-Flash uses a hybrid attention pattern: full-attention layers
(num_attention_heads=64) interleave with sliding-attention layers
(num_attention_heads=96). The HF config carries the per-layer attention type
in ``layer_types`` and the sliding-layer shape overrides in
``attention_other_setting``.

This provider surfaces ``layer_types`` (per-layer attention type) as a
dataclass field and ``attention_other_setting`` as the enable-flag for the
sliding-attention path. The actual sliding-layer shape values are forwarded
through the ``sliding_attention_setting`` field populated by
``Step35Bridge.provider_bridge``. The custom ``Step35DecoderLayer`` reads
all three at construction time to decide, on a per-layer basis, whether to
use the global config or the sliding-attention overrides when building its
sub-modules.
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import get_pg_rank

from megatron.bridge.models.gpt_provider import GPTModelProvider


class Step35DecoderLayer(TransformerLayer):
    """Hybrid full/sliding attention decoder layer for Step-3.5-Flash.

    Resolves a global 0-indexed ``layer_idx`` (MTP layers are offset past
    the main decoder; otherwise ``layer_number + pp_offset - 1``, or
    ``layer_number - 1`` when ``add_layer_offset=False``) and uses it to
    perform three per-layer config lookups before delegating to
    ``TransformerLayer.__init__``:

    1. **RoPE percentage** — ``rotary_percents[layer_idx]`` overrides
       ``config.rotary_percent`` (Step-3.5 alternates 0.5 / 1.0). Out of
       range → reset to ``1.0`` (the sliding-layer default), so MTP /
       unconfigured layers don't inherit the previous layer's value.
    2. **Attention type** — when ``layer_types[layer_idx] ==
       "sliding_attention"`` and ``attention_other_setting`` is truthy, the
       config is deep-copied and ``window_size`` / ``num_attention_heads`` /
       ``num_query_groups`` / ``kv_channels`` are overridden from
       ``sliding_attention_setting`` (already in Megatron-facing names; the
       HF→mcore renaming happens in ``Step35Bridge.provider_bridge``).
       ``rotary_percent`` is **not** touched here.
    3. **SwiGLU clamp** — ``swiglu_limits[layer_idx]`` /
       ``swiglu_limits_shared[layer_idx]`` overwrite
       ``activation_func_clamp_value`` / ``activation_func_clamp_value_shared``.
       Out of range → skipped, keeping the global value.

    All lookups are bounds-checked rather than raising. The spec-builder
    must keep these lists (and ``rotary_base_per_layer``) indexed by the
    global 0-indexed layer id.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: Optional[int] = None,
        name: str | None = None,
    ):
        pp_rank = get_pg_rank(pg_collection.pp)
        if is_mtp_layer:
            layer_idx = layer_number + config.num_layers + get_transformer_layer_offset(config, vp_stage, pp_rank) - 1
        elif add_layer_offset:
            layer_idx = layer_number + get_transformer_layer_offset(config, vp_stage, pp_rank) - 1
        else:
            layer_idx = layer_number - 1

        rotary_percents = getattr(config, "rotary_percents", None) or []
        if 0 <= layer_idx < len(rotary_percents):
            config.rotary_percent = rotary_percents[layer_idx]
        else:
            config.rotary_percent = 1.0

        layer_types = getattr(config, "layer_types", None) or []
        is_sliding = (
            layer_types is not None
            and 0 <= layer_idx < len(layer_types)
            and layer_types[layer_idx] == "sliding_attention"
        )
        if is_sliding:
            if getattr(config, "sliding_attention_setting", None):
                config = copy.deepcopy(config)
                config.window_size = config.sliding_attention_setting["window_size"]
                config.num_attention_heads = config.sliding_attention_setting["num_attention_heads"]
                config.num_query_groups = config.sliding_attention_setting["num_query_groups"]
                config.kv_channels = config.sliding_attention_setting["kv_channels"]
        else:
            config.window_size = None

        swiglu_limits = getattr(config, "swiglu_limits", None) or []
        swiglu_limits_shared = getattr(config, "swiglu_limits_shared", None) or []
        if 0 <= layer_idx < len(swiglu_limits):
            v = swiglu_limits[layer_idx]
            config.activation_func_clamp_value = None if (v is None or float(v) == 0.0) else float(v)
        # use separate swiglu limit for shared expert with MCore
        if 0 <= layer_idx < len(swiglu_limits_shared):
            v = swiglu_limits_shared[layer_idx]
            config.activation_func_clamp_value_shared_expert = None if (v is None or float(v) == 0.0) else float(v)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
            is_mtp_layer=is_mtp_layer,
            add_layer_offset=add_layer_offset,
            pp_layer_offset=pp_layer_offset,
            name=name,
        )


class Step35SharedExpertMLP(SharedExpertMLP):
    """Shared-expert MLP for Step-3.5 honoring a per-shared-expert SwiGLU clamp.

    ``SharedExpertMLP.__init__`` private-deepcopies its config so the shared
    expert can mutate ``ffn_hidden_size`` without affecting the routed experts.
    Step-3.5 sets a separate per-layer ``activation_func_clamp_value_shared_expert``
    field on the config in ``Step35DecoderLayer.__init__`` (with documented
    fallback to ``activation_func_clamp_value`` when it is ``None``). This
    subclass surfaces that field to ``MLP.forward`` — which only reads
    ``self.config.activation_func_clamp_value`` for SwiGLU clamping — by
    swapping the value on the private config for the duration of the forward
    pass.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward function"""
        # ``MLP.forward`` (called via ``super().forward``) reads
        # ``self.config.activation_func_clamp_value``. To honor
        # ``activation_func_clamp_value_shared_expert`` for the shared expert (and
        # the documented fallback to ``activation_func_clamp_value`` when it is
        # None), temporarily override the field on this instance's config (which
        # is a private deepcopy and is not shared with routed experts) and
        # restore it after ``super().forward`` returns.
        shared_clamp = getattr(self.config, "activation_func_clamp_value_shared_expert", None)
        if shared_clamp is not None:
            original_clamp = self.config.activation_func_clamp_value
            self.config.activation_func_clamp_value = shared_clamp
            try:
                output = super().forward(hidden_states)
            finally:
                self.config.activation_func_clamp_value = original_clamp
        else:
            output = super().forward(hidden_states)
        return output


@dataclass
class Step35ModelProvider(GPTModelProvider):
    """Model provider for Step-3.5-Flash.

    Adds Step3.5-specific fields on top of ``GPTModelProvider``:

    * ``layer_types``: 0-indexed list of attention types (e.g.
      ``"full_attention"`` / ``"sliding_attention"``). The provider may carry
      main decoder entries plus MTP entries because ``Step35DecoderLayer``
      indexes MTP layers after ``config.num_layers``.
    * ``attention_other_setting``: HF dict that enables and describes the
      sliding-attention override.
    * ``sliding_attention_setting``: normalized Megatron-facing shape overrides
      derived from ``attention_other_setting``.
    * ``head_wise_attn_gate``: whether to map HF's per-head ``g_proj`` gate
      through native per-head gate rows when MCore supports them. Older MCore
      versions use the full-head ``attention_output_gate`` layout as a fallback.

    These fields are populated from the HF config inside
    ``Step35Bridge.provider_bridge``.
    """

    layer_types: list[str] | None = None
    attention_other_setting: dict[str, Any] | None = None
    sliding_attention_setting: dict[str, Any] | None = None
    rotary_base_per_layer: list[float] | None = None
    head_wise_attn_gate: Optional[bool] = False
