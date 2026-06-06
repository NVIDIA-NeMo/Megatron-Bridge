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

"""GPT layer-spec helper for Bridge-local Dual Chunk Attention."""

import copy

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gpt.dca_attention import DualChunkAttention, DualChunkSelfAttention


def _transformer_config(config):
    return getattr(config, "transformer", config)


def _dca_value(config, name: str, default: int) -> int:
    if hasattr(config, name):
        return getattr(config, name)
    return getattr(_transformer_config(config), name, default)


def get_dca_gpt_layer_spec(
    config,
    vp_stage: int | None = None,
    *,
    use_transformer_engine: bool = True,
) -> ModuleSpec:
    """Build a GPT decoder block spec with DCA self-attention modules."""
    transformer_config = _transformer_config(config)
    dca_chunk_size = _dca_value(config, "dca_chunk_size", 8192)
    dca_local_size = _dca_value(config, "dca_local_size", 1024)

    spec = copy.deepcopy(
        get_gpt_decoder_block_spec(
            transformer_config,
            use_transformer_engine=use_transformer_engine,
            normalization=getattr(transformer_config, "normalization", None),
            qk_l2_norm=getattr(transformer_config, "qk_l2_norm", False),
            vp_stage=vp_stage,
        )
    )

    for layer_spec in spec.layer_specs:
        self_attention_spec = layer_spec.submodules.self_attention
        self_attention_spec.module = DualChunkSelfAttention
        self_attention_spec.params = dict(getattr(self_attention_spec, "params", {}) or {})
        self_attention_spec.params.update(
            {
                "attn_mask_type": AttnMaskType.causal,
                "dca_chunk_size": dca_chunk_size,
                "dca_local_size": dca_local_size,
            }
        )
        self_attention_spec.submodules.core_attention = ModuleSpec(
            module=DualChunkAttention,
            params={"dca_chunk_size": dca_chunk_size, "dca_local_size": dca_local_size},
        )

    return spec
