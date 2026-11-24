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

"""Ling 3.0 KDA parameter layout mappings shared by Tiny and Flash."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.distributed._tensor import DTensor

from megatron.bridge.models.conversion.param_mapping import (
    MegatronParamMapping,
)
from megatron.bridge.models.conversion.utils import get_module_and_param_from_name, remove_non_pickleables


class _BailingMoe3KDAInProjMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Map Ling 3.0's five separate KDA projections to fused ``in_proj``.

    Ling 3.0 stores Q, K, V, F, and G as separate HF tensors.  MCore's direct KDA
    projection is ordered as Q, K, V, G, Gate, where HF F is the MCore G
    section and HF G is the MCore Gate section.
    """

    def __init__(
        self,
        megatron_param: str,
        q: str,
        k: str,
        v: str,
        f: str,
        g: str,
    ) -> None:
        super().__init__(
            megatron_param,
            {"q": q, "k": k, "v": v, "f": f, "g": g},
        )

    @staticmethod
    def _section_sizes(config: Any) -> tuple[int, int, int, int, int]:
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads
        return qk_dim, qk_dim, v_dim, qk_dim, v_dim

    def hf_to_megatron(
        self,
        hf_weights: dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Shard every KDA section independently, then form the local target layout."""
        sections = [hf_weights[name] for name in ("q", "k", "v", "f", "g")] if self.tp_rank == 0 else None
        if self.tp_size == 1:
            return torch.cat(sections, dim=0)

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        if isinstance(target_param, DTensor):
            output_shape = target_param.orig_param.shape
        else:
            output_shape = target_param.shape

        splits = None
        if sections is not None:
            section_splits = []
            for section in sections:
                if section.shape[0] % self.tp_size != 0:
                    raise ValueError(
                        f"Cannot evenly split KDA section shape {section.shape} across TP={self.tp_size}: "
                        f"{self.megatron_param=}"
                    )
                section_splits.append(torch.chunk(section, self.tp_size, dim=0))
            splits = [
                torch.cat([section_splits[section_index][rank] for section_index in range(5)], dim=0)
                for rank in range(self.tp_size)
            ]

        return self.scatter_to_tp_ranks(
            splits,
            output_shape,
            target_param.dtype,
            target_param.device,
        )

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        """Gather each local KDA section and restore the five HF projection tensors."""
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}

        megatron_weights = self.maybe_dequantize(megatron_weights)
        config = self._broadcast_config(megatron_module)
        global_sizes = self._section_sizes(config)
        if any(size % self.tp_size != 0 for size in global_sizes):
            raise ValueError(f"KDA sections are not divisible by TP={self.tp_size}: {global_sizes=}")
        local_sizes = tuple(size // self.tp_size for size in global_sizes)
        local_weights = self.gather_from_tp_ranks(megatron_weights)
        local_sections = [torch.split(weight, local_sizes, dim=0) for weight in local_weights]
        sections = [torch.cat([rank_sections[index] for rank_sections in local_sections], dim=0) for index in range(5)]
        q, k, v, f, g = sections
        return {
            self.hf_param["q"]: q,
            self.hf_param["k"]: k,
            self.hf_param["v"]: v,
            self.hf_param["f"]: f,
            self.hf_param["g"]: g,
        }

    def _broadcast_config(self, megatron_module: nn.Module | None) -> Any:
        if megatron_module is None:
            return self.broadcast_obj_from_pp_rank(None, cache_key="bailing_moe3_config")
        config = self._get_config(megatron_module)
        config = remove_non_pickleables(config, max_depth=3)
        return self.broadcast_obj_from_pp_rank(config, cache_key="bailing_moe3_config")

    def resolve(self, captures: tuple[str, ...]) -> _BailingMoe3KDAInProjMapping:
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
            resolved_hf_param["f"],
            resolved_hf_param["g"],
        )


class _BailingMoe3KDAConv1dMapping(MegatronParamMapping[dict[str, torch.Tensor]]):
    """Map Ling 3.0's separate KDA Q/K/V convolution weights to ``conv1d``."""

    def __init__(self, megatron_param: str, q: str, k: str, v: str) -> None:
        super().__init__(megatron_param, {"q": q, "k": k, "v": v})

    @staticmethod
    def _section_sizes(config: Any) -> tuple[int, int, int]:
        qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
        v_dim = config.linear_value_head_dim * config.linear_num_value_heads
        return qk_dim, qk_dim, v_dim

    def hf_to_megatron(
        self,
        hf_weights: dict[str, torch.Tensor],
        megatron_module: nn.Module,
    ) -> torch.Tensor:
        """Shard Q/K/V convolution channels independently, then concatenate locally."""
        sections = [hf_weights[name] for name in ("q", "k", "v")] if self.tp_rank == 0 else None
        if self.tp_size == 1:
            return torch.cat(sections, dim=0)

        normalized_param = self._normalize_expert_param_name(self.megatron_param)
        _, target_param = get_module_and_param_from_name(megatron_module, normalized_param)
        if isinstance(target_param, DTensor):
            output_shape = target_param.orig_param.shape
        else:
            output_shape = target_param.shape

        splits = None
        if sections is not None:
            section_splits = []
            for section in sections:
                if section.shape[0] % self.tp_size != 0:
                    raise ValueError(
                        f"Cannot evenly split KDA convolution section shape {section.shape} across "
                        f"TP={self.tp_size}: {self.megatron_param=}"
                    )
                section_splits.append(torch.chunk(section, self.tp_size, dim=0))
            splits = [
                torch.cat([section_splits[section_index][rank] for section_index in range(3)], dim=0)
                for rank in range(self.tp_size)
            ]

        return self.scatter_to_tp_ranks(
            splits,
            output_shape,
            target_param.dtype,
            target_param.device,
        )

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: nn.Module | None,
    ) -> dict[str, torch.Tensor]:
        """Gather each convolution section and restore the three HF tensors."""
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}

        megatron_weights = self.maybe_dequantize(megatron_weights)
        config = self._broadcast_config(megatron_module)
        global_sizes = self._section_sizes(config)
        if any(size % self.tp_size != 0 for size in global_sizes):
            raise ValueError(f"KDA convolution sections are not divisible by TP={self.tp_size}: {global_sizes=}")
        local_sizes = tuple(size // self.tp_size for size in global_sizes)
        local_weights = self.gather_from_tp_ranks(megatron_weights)
        local_sections = [torch.split(weight, local_sizes, dim=0) for weight in local_weights]
        q, k, v = [torch.cat([rank_sections[index] for rank_sections in local_sections], dim=0) for index in range(3)]
        return {
            self.hf_param["q"]: q,
            self.hf_param["k"]: k,
            self.hf_param["v"]: v,
        }

    def _broadcast_config(self, megatron_module: nn.Module | None) -> Any:
        if megatron_module is None:
            return self.broadcast_obj_from_pp_rank(None, cache_key="bailing_moe3_config")
        config = self._get_config(megatron_module)
        config = remove_non_pickleables(config, max_depth=3)
        return self.broadcast_obj_from_pp_rank(config, cache_key="bailing_moe3_config")

    def resolve(self, captures: tuple[str, ...]) -> _BailingMoe3KDAConv1dMapping:
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["q"],
            resolved_hf_param["k"],
            resolved_hf_param["v"],
        )
