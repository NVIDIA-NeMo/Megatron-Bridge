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

"""Step-3.5-Flash Model Provider for Megatron-Core.

Step-3.5-Flash uses a hybrid attention pattern: full-attention layers
(num_attention_heads=64) interleave with sliding-attention layers
(num_attention_heads=96). The HF config carries the per-layer attention type
in ``layer_types`` and the sliding-layer shape overrides in
``attention_other_setting``.

This provider surfaces ``layer_types`` (per-layer attention type) as a
dataclass field and ``attention_other_setting`` as the enable-flag for the
sliding-attention path. The actual sliding-layer shape values are forwarded
through the dynamic ``sliding_attention_setting`` attribute populated by
``Step35Bridge.provider_bridge``. The custom ``Step35DecoderLayer`` reads
all three at construction time to decide, on a per-layer basis, whether to
use the global config or the sliding-attention overrides when building its
sub-modules.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional

from megatron.core.process_groups_config import ProcessGroupCollection
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

    On construction the layer resolves a global 0-indexed ``layer_idx``:

    * For MTP layers or when ``add_layer_offset=False``,
      ``layer_idx = layer_number - 1`` (MTP layers live in their own 1-indexed
      namespace).
    * Otherwise ``layer_idx = layer_number + get_transformer_layer_offset(
      config, vp_stage, pp_rank) - 1``, so PP>1 still maps correctly.

    It then looks up ``config.layer_types[layer_idx]``. If the entry is
    ``"sliding_attention"`` (and ``config.attention_other_setting`` is set as
    the enable flag), the config is deep-copied and the shape-related fields
    are overridden from ``config.sliding_attention_setting`` before delegating
    to ``TransformerLayer.__init__``. The overridden config is what every
    downstream sub-module (``self_attention``, ``linear_qkv`` -- with the
    per-head ``g_proj`` rows fused into its tail -- and ``linear_proj``) ends
    up reading, so each layer is sized correctly without changing
    Megatron-LM core.

    Fields read from ``config.sliding_attention_setting`` (HF key on the left,
    ``TransformerConfig`` attribute on the right):

    * ``rotary_percent``        -> ``rotary_percent``
    * ``num_attention_heads``   -> ``num_attention_heads``
    * ``num_query_groups``      -> ``num_query_groups``
    * ``head_dim``              -> ``kv_channels``

    Notes
    -----
    * The spec-builder must keep ``layer_types`` indexed by the global
      0-indexed layer id (same constraint as ``rotary_base_per_layer``).
    * Layers whose resolved ``layer_idx`` falls outside ``layer_types`` (e.g.
      MTP layers, which are not enumerated in ``layer_types``) fall through
      to the global config.
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
    ):
        pp_rank = get_pg_rank(pg_collection.pp)
        if is_mtp_layer:
          layer_idx = layer_number + config.num_layers + get_transformer_layer_offset(config, vp_stage, pp_rank) - 1
        elif add_layer_offset:
            layer_idx = layer_number + get_transformer_layer_offset(config, vp_stage, pp_rank) - 1
        else:
            layer_idx = layer_number - 1
        layer_types = getattr(config, "layer_types", None) or []

        is_sliding = (
            layer_types is not None
            and 0 <= layer_idx < len(layer_types)
            and layer_types[layer_idx] == "sliding_attention"
        )
        if is_sliding:
            config = copy.deepcopy(config)
            # Override the Q/KV shape fields on the deep-copied config so the
            # sub-modules built by super().__init__ see sliding-layer shapes.
            # Source dict is ``config.sliding_attention_setting``; HF -> mcore
            # attribute mapping: num_attention_groups -> num_query_groups,
            # head_dim -> kv_channels.
            config.rotary_percent = config.sliding_attention_setting["rotary_percent"]
            config.num_attention_heads = config.sliding_attention_setting["num_attention_heads"]
            config.num_query_groups = config.sliding_attention_setting["num_query_groups"]
            config.kv_channels = config.sliding_attention_setting["head_dim"]

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
        )


@dataclass
class Step35ModelProvider(GPTModelProvider):
    """Model provider for Step-3.5-Flash.

    Adds two Step3.5-specific fields on top of ``GPTModelProvider``:

    * ``layer_types``: 0-indexed list of attention types (e.g.
      ``"full_attention"`` / ``"sliding_attention"``), one entry per main
      decoder layer. Read by ``Step35DecoderLayer`` to decide whether the
      current layer is a sliding-attention layer.
    * ``head_wise_attn_gate``: whether to use head-wise attention gate.
      ``Step35DecoderLayer`` only uses it as a truthy enable-flag for the
      sliding-attention override path; the actual shape values that get
      applied to sliding layers are taken from
      ``provider.sliding_attention_setting`` (set as a dynamic attribute by
      ``Step35Bridge.provider_bridge`` and read via ``getattr`` on the
      ``TransformerConfig`` at layer-construction time).

    Both fields are populated from the HF config inside
    ``Step35Bridge.provider_bridge``.
    """

    layer_types: Optional[List[str]] = None
    head_wise_attn_gate: Optional[bool] = False
