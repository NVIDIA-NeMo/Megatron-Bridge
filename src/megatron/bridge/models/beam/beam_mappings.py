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

"""Beam full-width attention-gate packing.

Each query group contains ``[Q, gate, K, V]``. Q and gate each have
``heads_per_group * head_dim`` rows; K and V each have ``head_dim`` rows.
TP/PP communication uses the inherited AutoMapping delegate.
"""

from typing import Tuple

import torch

from megatron.bridge.models.conversion.param_mapping import QKVGMapping
from megatron.bridge.models.conversion.utils import remove_non_pickleables


def _geometry(config) -> Tuple[int, int, int, int]:
    """Return ``(head_num, num_query_groups, heads_per_group, head_size)``."""
    head_num = config.num_attention_heads
    num_query_groups = config.num_query_groups or head_num
    heads_per_group = head_num // num_query_groups
    head_size = config.kv_channels or (config.hidden_size // head_num)
    return head_num, num_query_groups, heads_per_group, head_size


def merge_qkv_gate_weights(config, q, k, v, g) -> torch.Tensor:
    """Interleave Q, K, V and a full-width gate into Megatron's ``linear_qkv``."""
    head_num, num_query_groups, heads_per_group, head_size = _geometry(config)
    hidden_size = q.shape[-1]

    if g.shape[0] != head_num * head_size:
        raise ValueError(
            f"Beam gate_proj should emit num_heads*head_dim={head_num * head_size} rows, got {tuple(g.shape)}."
        )

    width = heads_per_group * head_size
    blocks = [
        weight.reshape(num_query_groups, rows, hidden_size)
        for weight, rows in zip((q, g, k, v), (width, width, head_size, head_size))
    ]
    return torch.cat(blocks, dim=1).reshape(-1, hidden_size)


def split_qkv_gate_weights(config, qkv: torch.Tensor):
    """Inverse of :func:`merge_qkv_gate_weights`; returns ``(q, k, v, g)``."""
    _, groups, heads_per_group, head_size = _geometry(config)
    width = heads_per_group * head_size
    rows = qkv.reshape(groups, -1, qkv.shape[-1])
    q, g, k, v = torch.split(rows, [width, width, head_size, head_size], dim=1)
    return tuple(weight.reshape(-1, qkv.shape[-1]) for weight in (q, k, v, g))


class BeamQKVGateMapping(QKVGMapping):
    """Fuse Q/K/V and a full-width attention gate for Beam."""

    def hf_to_megatron(self, hf_weights: dict[str, torch.Tensor], megatron_module: torch.nn.Module) -> torch.Tensor:
        """Pack the full-width gate with Q/K/V and distribute the fused weight."""
        merged = None
        if self.tp_rank == 0:
            if hf_weights["q"].ndim == 1:
                raise NotImplementedError("BeamQKVGateMapping handles weights only.")
            config = self._get_config(megatron_module)
            merged = merge_qkv_gate_weights(config, hf_weights["q"], hf_weights["k"], hf_weights["v"], hf_weights["g"])
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(
        self, megatron_weights: torch.Tensor | None, megatron_module: torch.nn.Module | None
    ) -> dict[str, torch.Tensor]:
        """Gather the fused weight and split Beam's four attention projections."""
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)

        config = None
        if megatron_module is not None:
            config = remove_non_pickleables(self._get_config(megatron_module), max_depth=3)
        config = self.broadcast_obj_from_pp_rank(config, "beam_qkvg_config")

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}
        packed = next(iter(packed_dict.values()))
        if packed.ndim == 1:
            raise NotImplementedError("BeamQKVGateMapping handles weights only.")
        q, k, v, g = split_qkv_gate_weights(config, packed)
        return {
            self.hf_param["q"]: q,
            self.hf_param["k"]: k,
            self.hf_param["v"]: v,
            self.hf_param["g"]: g,
        }
