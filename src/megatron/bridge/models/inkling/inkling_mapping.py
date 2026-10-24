# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Published Inkling checkpoint layouts, using native Bridge TP/EP conversion."""

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import (
    FusedGatedExpertMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    RowParallelMapping,
)
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Restore checkpoint rows [gate0, up0, gate1, up1, ...]."""
    return torch.stack((gate, up), dim=-2).flatten(-3, -2)


class InklingGatedMapping(MegatronParamMapping[torch.Tensor]):
    """Map an interleaved checkpoint FC1 to native TP-partitioned SwiGLU."""

    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, hf_param)
        self._gated = GatedMLPMapping(megatron_param, gate=hf_param + ".gate", up=hf_param + ".up")

    def local_hf_param_specs(self, global_param_name=None) -> tuple:
        return ()  # The native rows require interleaving before transfer.

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        return self._gated.hf_to_megatron({"gate": hf_weights[0::2], "up": hf_weights[1::2]}, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
        converted = self._gated.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}
        return {
            self.hf_param: interleave_gate_up(converted[self.hf_param + ".gate"], converted[self.hf_param + ".up"])
        }


class InklingExpertGatedMapping(FusedGatedExpertMapping):
    """Select one routed expert and preserve its published interleaved rows."""

    def local_hf_param_specs(self, global_param_name=None) -> tuple:
        return ()

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        expert = hf_weights[extract_expert_number_from_param(self.megatron_param)]
        # Slice before concatenation: never copy the full routed-expert stack.
        return super().hf_to_megatron(torch.cat((expert[0::2], expert[1::2])), megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
        converted = super().megatron_to_hf(megatron_weights, megatron_module)
        return {name: interleave_gate_up(*value.chunk(2, dim=-2)) for name, value in converted.items()}


class InklingSharedExpertMapping(MegatronParamMapping[torch.Tensor]):
    """Map a TP-sharded shared MLP to one slice of a replicated expert stack."""

    is_grouped_export = True

    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True
        mapping = InklingGatedMapping if ".linear_fc1." in megatron_param else RowParallelMapping
        self._mapping = None if "*" in megatron_param else mapping(megatron_param, hf_param)

    @property
    def group_key(self) -> str:
        return self.hf_param

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        expert = hf_weights[extract_expert_number_from_param(self.megatron_param)]
        return self._mapping.hf_to_megatron(expert, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module) -> dict[str, torch.Tensor]:
        return self._mapping.megatron_to_hf(megatron_weights, megatron_module)
