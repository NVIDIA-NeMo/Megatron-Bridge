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

"""Kimi K3 tensor-layout mappings for native Hybrid layers."""

import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)


class KimiK3QKVMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Pack equal-width KDA Q/K/V projections or convolutions within each TP shard."""

    def __init__(self, megatron_param: str, *, q: str, k: str, v: str):
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def resolve(self, captures: tuple[str, ...]) -> "KimiK3QKVMapping":
        """Resolve layer or expert captures before distributed conversion."""
        megatron, hf = self._resolve_names(captures)
        return type(self)(megatron, **hf)

    def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module: nn.Module) -> torch.Tensor:
        """Transpose component-major HF tensors into rank-major fused shards."""
        packed = None
        if self.tp_rank == 0:
            parts = [hf_weights[key] for key in ("q", "k", "v")]
            if not all(part.shape == parts[0].shape for part in parts) or parts[0].shape[0] % self.tp_size:
                raise ValueError("Kimi K3 Q/K/V tensors must have equal shapes and rows divisible by TP.")
            rows = parts[0].shape[0] // self.tp_size
            packed = torch.stack(parts).reshape(3, self.tp_size, rows, *parts[0].shape[1:])
            packed = packed.transpose(0, 1).reshape(-1, *parts[0].shape[1:]).contiguous()
        return self._tp_mapping.hf_to_megatron(packed, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None) -> dict:
        """Gather rank-major shards and restore separate HF tensors."""
        gathered = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not gathered:
            return {}
        packed = gathered[self.megatron_param]
        if packed.shape[0] % (3 * self.tp_size):
            raise ValueError("Kimi K3 fused Q/K/V rows must be divisible by three times TP.")
        rows = packed.shape[0] // (3 * self.tp_size)
        parts = packed.reshape(self.tp_size, 3, rows, *packed.shape[1:]).transpose(0, 1)
        parts = parts.reshape(3, -1, *packed.shape[1:])
        return {self.hf_param[key]: part.contiguous() for key, part in zip(("q", "k", "v"), parts, strict=True)}


class KimiK3AttnResQueryMapping(ReplicatedMapping):
    """Translate the HF one-row score projection to a native pseudo-query vector."""

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        """Remove only the score projection's singleton output dimension."""
        if hf_weights.ndim != 2 or hf_weights.shape[0] != 1:
            raise ValueError("Kimi K3 attention-residual score projections must have shape [1, hidden_size].")
        config = self._get_config(megatron_module)
        if not config.attn_res_query_as_matrix:
            hf_weights = hf_weights.squeeze(0)
        return super().hf_to_megatron(hf_weights, megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor | None, megatron_module: nn.Module | None) -> dict:
        """Restore the singleton projection output dimension."""
        return {
            name: value.unsqueeze(0) if value.ndim == 1 else value
            for name, value in super().megatron_to_hf(megatron_weights, megatron_module).items()
        }

    def local_hf_param_specs(self, global_param_name: str | None = None) -> tuple:
        """Use the shape-aware conversion path for local exports too."""
        return ()
