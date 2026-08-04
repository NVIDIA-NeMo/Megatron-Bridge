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

"""Provider-neutral Step3.5 decoder primitives and layer-spec factory."""

import copy
from functools import partial
from typing import Any

import torch
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.utils import get_pg_rank


class Step35DecoderLayer(TransformerLayer):
    """Build a Step decoder layer using explicit per-layer family settings."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        vp_stage: int | None = None,
        is_mtp_layer: bool = False,
        add_layer_offset: bool = True,
        pp_layer_offset: int | None = None,
        name: str | None = None,
        layer_types: list[str] | None = None,
        rotary_percents: list[float] | None = None,
        sliding_attention_setting: dict[str, Any] | None = None,
        swiglu_limits: list[float | None] | None = None,
        swiglu_limits_shared: list[float | None] | None = None,
    ) -> None:
        config = copy.deepcopy(config)
        pp_rank = get_pg_rank(pg_collection.pp)
        if is_mtp_layer:
            layer_idx = layer_number + config.num_layers + get_transformer_layer_offset(config, vp_stage, pp_rank) - 1
        elif add_layer_offset:
            layer_idx = layer_number + get_transformer_layer_offset(config, vp_stage, pp_rank) - 1
        else:
            layer_idx = layer_number - 1

        rotary_percents = rotary_percents or []
        config.rotary_percent = rotary_percents[layer_idx] if 0 <= layer_idx < len(rotary_percents) else 1.0
        layer_types = layer_types or []
        is_sliding = 0 <= layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention"
        if is_sliding and sliding_attention_setting:
            config.window_size = sliding_attention_setting["window_size"]
            config.num_attention_heads = sliding_attention_setting["num_attention_heads"]
            config.num_query_groups = sliding_attention_setting["num_query_groups"]
            config.kv_channels = sliding_attention_setting["kv_channels"]
        elif not is_sliding:
            config.window_size = None

        swiglu_limits = swiglu_limits or []
        if 0 <= layer_idx < len(swiglu_limits):
            value = swiglu_limits[layer_idx]
            config.activation_func_clamp_value = None if value is None or float(value) == 0.0 else float(value)
        shared_clamp = None
        swiglu_limits_shared = swiglu_limits_shared or []
        if 0 <= layer_idx < len(swiglu_limits_shared):
            value = swiglu_limits_shared[layer_idx]
            shared_clamp = None if value is None or float(value) == 0.0 else float(value)

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
        shared_experts = getattr(getattr(self, "mlp", None), "shared_experts", None)
        if shared_experts is not None:
            shared_experts.activation_func_clamp_value_shared_expert = shared_clamp


class Step35SharedExpertMLP(SharedExpertMLP):
    """Apply the Step-specific shared-expert SwiGLU clamp."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the shared expert with its independent clamp."""
        shared_clamp = getattr(self, "activation_func_clamp_value_shared_expert", None)
        if shared_clamp is None:
            return super().forward(hidden_states)
        original_clamp = self.config.activation_func_clamp_value
        self.config.activation_func_clamp_value = shared_clamp
        try:
            return super().forward(hidden_states)
        finally:
            self.config.activation_func_clamp_value = original_clamp


class _MTPDenseLayerSpecsList(list):
    def __init__(self, data, dense_mtp_spec):
        super().__init__(data)
        self._dense_mtp_spec = dense_mtp_spec

    def __getitem__(self, index):
        if isinstance(index, int) and index < 0:
            return self._dense_mtp_spec
        return super().__getitem__(index)


def build_step35_layer_spec(config, vp_stage: int | None = None):
    """Build the Step3.5/3.7 decoder spec from outer family settings."""
    transformer = config.transformer
    block = get_gpt_decoder_block_spec(
        transformer,
        use_transformer_engine=True,
        normalization="RMSNorm",
        vp_stage=vp_stage,
    )
    explicit = {
        "layer_types": config.layer_types,
        "rotary_percents": config.rotary_percents,
        "sliding_attention_setting": config.sliding_attention_setting,
        "swiglu_limits": config.swiglu_limits,
        "swiglu_limits_shared": config.swiglu_limits_shared,
    }
    for spec in block.layer_specs:
        spec.module = Step35DecoderLayer
        spec.params = dict(getattr(spec, "params", None) or {}, **explicit)
        mlp_submodules = getattr(spec.submodules.mlp, "submodules", None)
        shared = getattr(mlp_submodules, "shared_experts", None)
        if shared is not None:
            mlp_submodules.shared_experts = partial(Step35SharedExpertMLP, **shared.keywords)
    dense = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=transformer.qk_layernorm,
    )
    dense.module = Step35DecoderLayer
    dense.params = dict(getattr(dense, "params", None) or {}, **explicit)
    block.layer_specs = _MTPDenseLayerSpecsList(block.layer_specs, dense)
    return block


__all__ = ["Step35DecoderLayer", "Step35SharedExpertMLP", "build_step35_layer_spec"]
