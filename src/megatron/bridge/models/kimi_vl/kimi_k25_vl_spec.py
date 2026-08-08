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

"""MCore layer specification for Kimi K2.5 VL legacy-prefix inference."""

import copy
from functools import partial
from typing import Any

import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.moe.moe_layer import MoELayer


class KimiK25VLChunkedMoELayer(MoELayer):
    """Bound expert activation lifetime during full-prefix eval forwards."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        intermediate_tensors: Any = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run legacy-prefix inference as sequential sequence chunks."""
        num_chunks = min(getattr(self.config, "legacy_prefix_moe_chunks", 1), hidden_states.shape[0])
        should_chunk = (
            num_chunks > 1
            and not self.training
            and intermediate_tensors is None
            and padding_mask is None
            and self.config.transformer_impl != "inference_optimized"
        )
        if not should_chunk:
            return super().forward(hidden_states, intermediate_tensors, padding_mask)

        outputs = []
        for chunk in hidden_states.chunk(num_chunks, dim=0):
            output, bias = super(KimiK25VLChunkedMoELayer, self).forward(chunk)
            assert bias is None, "MCore MoE token dispatchers do not support output bias."
            outputs.append(output)
        return torch.cat(outputs, dim=0), None


def _replace_moe_builder(builder: Any) -> Any:
    """Replace only the stock MCore MoE builder, preserving its submodules."""
    if not isinstance(builder, partial) or builder.func is not MoELayer:
        return builder
    return partial(KimiK25VLChunkedMoELayer, *builder.args, **(builder.keywords or {}))


def build_kimi_k25_vl_spec(
    config: Any,
    vp_stage: int | None = None,
    *,
    use_transformer_engine: bool,
) -> Any:
    """Build Kimi K2.5 layers with bounded legacy-prefix MoE forwards."""
    block_spec = get_gpt_decoder_block_spec(
        config,
        use_transformer_engine=use_transformer_engine,
        vp_stage=vp_stage,
    )
    layer_specs = []
    for layer_spec in block_spec.layer_specs:
        layer_spec = copy.deepcopy(layer_spec)
        layer_spec.submodules.mlp = _replace_moe_builder(layer_spec.submodules.mlp)
        layer_specs.append(layer_spec)
    block_spec.layer_specs = layer_specs
    return block_spec
