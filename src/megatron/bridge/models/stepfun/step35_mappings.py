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

"""Provider-neutral stacked-expert mappings shared by Step3.5 and Step3.7."""

from typing import Dict

import torch

from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping


class StackedExpertAutoMapping(AutoMapping):
    """Map an MCore per-expert tensor to one slice of an HF stacked tensor."""

    is_grouped_export = True

    def _expert_idx(self) -> int:
        return int(self.megatron_param.rsplit("weight", 1)[-1])

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        """Select this expert before applying the standard mapping."""
        return super().hf_to_megatron(hf_weights[self._expert_idx()], megatron_module)


class StackedExpertGatedMLPMapping(GatedMLPMapping):
    """Map stacked HF gate/up expert tensors to one fused MCore expert."""

    is_grouped_export = True

    def _expert_idx(self) -> int:
        return int(self.megatron_param.rsplit("weight", 1)[-1])

    def hf_to_megatron(self, hf_weights: Dict[str, torch.Tensor], megatron_module) -> torch.Tensor:
        """Select this expert from both gate and up tensors."""
        expert_idx = self._expert_idx()
        return super().hf_to_megatron(
            {"gate": hf_weights["gate"][expert_idx], "up": hf_weights["up"][expert_idx]},
            megatron_module,
        )


__all__ = ["StackedExpertAutoMapping", "StackedExpertGatedMLPMapping"]
