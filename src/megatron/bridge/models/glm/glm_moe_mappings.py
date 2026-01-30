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

"""GLM MoE mapping helpers for fused expert weights in transformers 5.0+."""

from typing import Dict

import torch

from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


def _select_expert_weight(hf_weights: torch.Tensor, expert_idx: int) -> torch.Tensor:
    if hf_weights.ndim >= 3:
        return hf_weights[expert_idx]
    return hf_weights


def _align_weight_to_shape(weight: torch.Tensor, target_shape: torch.Size, name: str) -> torch.Tensor:
    if tuple(weight.shape) == tuple(target_shape):
        return weight
    if weight.ndim == 2 and tuple(weight.t().shape) == tuple(target_shape):
        return weight.t().contiguous()
    raise ValueError(f"Unexpected {name} shape {tuple(weight.shape)}; expected {tuple(target_shape)}.")


class _LooseGatedMLPMapping(GatedMLPMapping):
    def _validate_patterns(self, *args, **kwargs):
        # Allow mismatched wildcard counts for fused expert mappings.
        pass


class GLMExpertGateUpProjMapping(AutoMapping):
    """Mapping for fused expert gate+up projection weights."""

    def __init__(self, megatron_param: str, hf_param: str, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims)
        self._gated_mapping = _LooseGatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = _select_expert_weight(hf_weights, global_expert_number)

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        target_shape = target_param.shape
        gate_target_shape = (target_shape[0] // 2, target_shape[1])

        if target_shape[0] % 2 != 0:
            raise ValueError(f"Expected even fused dim for {self.megatron_param}, got {target_shape}.")

        if expert_weight.ndim == 3 and expert_weight.shape[0] == 2:
            gate = _align_weight_to_shape(expert_weight[0], gate_target_shape, "gate")
            up = _align_weight_to_shape(expert_weight[1], gate_target_shape, "up")
        else:
            expert_weight = _align_weight_to_shape(expert_weight, target_shape, "gate_up")
            gate, up = torch.chunk(expert_weight, 2, dim=0)

        return self._gated_mapping.hf_to_megatron({"gate": gate, "up": up}, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor, megatron_module: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        converted = self._gated_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not converted:
            return {}

        fused: Dict[str, torch.Tensor] = {}
        for name, tensor in converted.items():
            if not name.endswith(".gate"):
                continue
            base_name = name[: -len(".gate")]
            up_tensor = converted.get(f"{base_name}.up")
            if up_tensor is None:
                continue
            concat_dim = 0 if tensor.ndim == 2 else 1
            fused[base_name] = torch.cat([tensor, up_tensor], dim=concat_dim)
        return fused

    def _validate_patterns(self, *args, **kwargs):
        # Allow number of wildcards to mismatch in this mapping.
        pass


class GLMExpertDownProjMapping(AutoMapping):
    """Mapping for fused expert down projection weights."""

    def __init__(self, megatron_param: str, hf_param: str, permute_dims=None):
        super().__init__(megatron_param, hf_param, permute_dims)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: torch.nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        expert_weight = _select_expert_weight(hf_weights, global_expert_number)

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        expert_weight = _align_weight_to_shape(expert_weight, target_param.shape, "down_proj")
        return super().hf_to_megatron(expert_weight, megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # Allow number of wildcards to mismatch in this mapping.
        pass
