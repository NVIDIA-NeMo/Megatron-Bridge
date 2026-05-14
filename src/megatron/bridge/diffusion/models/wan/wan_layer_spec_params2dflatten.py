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

# pylint: disable=C0115,C0116,C0301

import copy
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules

from megatron.bridge.diffusion.models.wan.wan_layer_spec import (
    WanAdaLN,
    WanLayerWithAdaLN,
    get_wan_block_with_transformer_engine_spec as _get_spec_original,
)


class WanAdaLNParams2DFlatten(WanAdaLN):
    """WanAdaLN with modulation optionally stored as 2D [6, hidden_size] for MUON compatibility.

    When "modulation" is in config.muon_non_2d_params_mode, replaces the original [1, 6, hidden_size]
    parameter with [6, hidden_size]. The forward pass is unchanged: [6, hidden_size] broadcasts
    with timestep_emb [B, 6, hidden_size] identically to [1, 6, hidden_size].
    """

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        if "modulation" in config.muon_non_2d_params_mode:
            # Replace [1, 6, hidden_size] with [6, hidden_size] — MUON requires ndim == 2.
            # Broadcasting: [6, hidden_size] + [B, 6, hidden_size] → [B, 6, hidden_size], same result.
            self.modulation = nn.Parameter(torch.randn(6, config.hidden_size) / config.hidden_size**0.5)
            setattr(self.modulation, "sequence_parallel", config.sequence_parallel)


class WanLayerWithAdaLNParams2DFlatten(WanLayerWithAdaLN):
    """WanLayerWithAdaLN using WanAdaLNParams2DFlatten instead of WanAdaLN."""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ):
        super().__init__(config, submodules, layer_number, hidden_dropout, pg_collection, vp_stage)
        # Replace the standard WanAdaLN created by super().__init__() with the 2D-flatten version.
        self.adaLN = WanAdaLNParams2DFlatten(config=self.config)
        # Re-mark the new adaLN's params for TP gradient averaging.
        self._mark_trainable_params_for_tp_grad_avg([self.adaLN])


def get_wan_block_with_transformer_engine_spec(qkv_format: str = "thd"):  # noqa: D103
    spec = copy.deepcopy(_get_spec_original(qkv_format=qkv_format))
    spec.module = WanLayerWithAdaLNParams2DFlatten
    return spec
