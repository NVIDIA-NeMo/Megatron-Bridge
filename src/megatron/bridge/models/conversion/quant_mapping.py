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

import re

import torch

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping, ReplicatedMapping


class AmaxMapping(ReplicatedMapping):
    """Amax mapping for quantization."""

    def __init__(self, megatron_param: str, hf_param: str | dict[str, str]):
        """Initialize the Amax mapping."""
        super().__init__(megatron_param, hf_param)
        self.allow_hf_name_mismatch = True


class AmaxFanoutMapping(AmaxMapping):
    """Replicated amax mapping that fans out one Megatron amax to multiple HF targets.

    Used for QKV and gate/up where the amax values are shared but need to be
    written/read under multiple HF parameter names.
    """

    def __init__(self, megatron_param: str, hf_params: list[str]):
        assert hf_params, "hf_params must be non-empty"
        self.hf_targets = hf_params
        # Use the first target as the canonical HF name for HF->Megatron loading
        super().__init__(megatron_param, hf_params[0])

    def megatron_to_hf(self, megatron_weights, megatron_module):
        base = super().megatron_to_hf(megatron_weights, megatron_module)
        if not base:
            return {}
        weight = next(iter(base.values()))
        return {t: weight for t in self.hf_targets}

    def resolve(self, captures: tuple[str, ...]):
        """Resolve wildcards for both megatron_param and all HF targets."""
        resolved_megatron_param = self.megatron_param
        capture_index = 0
        # Resolve ** then * in megatron_param
        while "**" in resolved_megatron_param and capture_index < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("**", captures[capture_index], 1)
            capture_index += 1
        while "*" in resolved_megatron_param and capture_index < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("*", captures[capture_index], 1)
            capture_index += 1

        # Resolve HF targets separately with a fresh capture index
        resolved_hf_targets = []
        for target in self.hf_targets:
            t = target
            ci = 0
            while "**" in t and ci < len(captures):
                t = t.replace("**", captures[ci], 1)
                ci += 1
            while "*" in t and ci < len(captures):
                t = t.replace("*", captures[ci], 1)
                ci += 1
            resolved_hf_targets.append(t)

        new_mapping = type(self)(resolved_megatron_param, resolved_hf_targets)
        new_mapping.allow_hf_name_mismatch = self.allow_hf_name_mismatch
        return new_mapping


class MoeAmaxFanoutMapping(AmaxMapping):
    """Shared MoE amax mapping that fans out to per-expert HF quantizers.

    Megatron grouped-MoE layers use one quantizer for each rank's local expert
    block, while HF names carry an expert wildcard. This mapping gathers those
    per-EP-rank amax values and expands the HF expert wildcard during export.
    """

    _EXPERT_WILDCARD_RE = re.compile(r"(\.experts\.)(\*)(\.)")

    def __init__(
        self,
        megatron_param: str,
        hf_patterns: list[str],
        num_experts: int | None = None,
    ) -> None:
        assert hf_patterns, "hf_patterns must be non-empty"
        self.hf_patterns = hf_patterns
        self.num_experts = num_experts
        super().__init__(megatron_param, {})

    def _validate_patterns(self) -> None:
        """Allow one extra HF wildcard for the expert index."""
        return

    @property
    def is_expert(self) -> bool:
        """Use normal TP handling; EP fanout is handled explicitly here."""
        return False

    def hf_to_megatron(self, hf_weights, megatron_module):
        """Grouped-MoE amax fanout is export-only."""
        return None

    def _get_num_experts(self, megatron_module: object | None) -> int | None:
        if self.num_experts is not None:
            return self.num_experts

        config = getattr(megatron_module, "config", None)
        for source in (config, megatron_module):
            if source is None:
                continue
            for attr in ("num_moe_experts", "num_experts", "n_routed_experts"):
                value = getattr(source, attr, None)
                if value is not None:
                    return value
        return None

    @classmethod
    def _resolve_pattern(
        cls,
        pattern: str,
        captures: tuple[str, ...],
        max_captures: int,
    ) -> str:
        resolved = pattern
        capture_index = 0
        while "**" in resolved and capture_index < len(captures) and capture_index < max_captures:
            resolved = resolved.replace("**", captures[capture_index], 1)
            capture_index += 1
        while "*" in resolved and capture_index < len(captures) and capture_index < max_captures:
            resolved = resolved.replace("*", captures[capture_index], 1)
            capture_index += 1
        return resolved

    def _get_num_experts_for_rank(self, megatron_module: object | None) -> int | None:
        if self.num_experts is not None:
            return self.num_experts

        if megatron_module is None:
            return self.broadcast_obj_from_pp_rank(None, cache_key=f"{self.megatron_param}:num_experts")

        num_experts = self._get_num_experts(megatron_module)
        return self.broadcast_obj_from_pp_rank(num_experts, cache_key=f"{self.megatron_param}:num_experts")

    def _gather_amax_by_ep_rank(self, weight: torch.Tensor) -> list[torch.Tensor]:
        if self.ep_size == 1:
            return [weight]

        gathered_weights = [torch.empty_like(weight) for _ in range(self.ep_size)]
        torch.distributed.all_gather(gathered_weights, weight, group=self.ep_group)
        return gathered_weights

    def megatron_to_hf(
        self,
        megatron_weights: torch.Tensor | None,
        megatron_module: object | None,
    ) -> dict[str, torch.Tensor]:
        base = super().megatron_to_hf(megatron_weights, megatron_module)
        if not base:
            return {}
        weight = next(iter(base.values()))

        num_experts = self._get_num_experts_for_rank(megatron_module)
        if num_experts is None or num_experts <= 0:
            raise RuntimeError(
                f"Could not determine num_experts for {self.megatron_param}. "
                "Expected megatron_module.config.num_moe_experts or num_experts."
            )
        if num_experts % self.ep_size != 0:
            raise RuntimeError(
                f"num_experts ({num_experts}) must be divisible by EP size ({self.ep_size}) for {self.megatron_param}."
            )

        result = {}
        weights_by_ep_rank = self._gather_amax_by_ep_rank(weight)
        experts_per_rank = num_experts // self.ep_size
        for pattern in self.hf_patterns:
            if self._EXPERT_WILDCARD_RE.search(pattern):
                for ep_rank, ep_weight in enumerate(weights_by_ep_rank):
                    expert_start = ep_rank * experts_per_rank
                    expert_stop = expert_start + experts_per_rank
                    for expert_idx in range(expert_start, expert_stop):
                        hf_name = self._EXPERT_WILDCARD_RE.sub(
                            rf"\g<1>{expert_idx}\g<3>",
                            pattern,
                            count=1,
                        )
                        result[hf_name] = ep_weight
            else:
                result[pattern] = weight
        return result

    def resolve(self, captures: tuple[str, ...]) -> "MoeAmaxFanoutMapping":
        """Resolve layer wildcards while preserving the HF expert wildcard."""
        megatron_wildcards = self._count_wildcard_groups(self.megatron_param)
        resolved_megatron_param = self._resolve_pattern(
            self.megatron_param,
            captures,
            megatron_wildcards,
        )
        resolved_hf_patterns = [
            self._resolve_pattern(pattern, captures, megatron_wildcards) for pattern in self.hf_patterns
        ]
        new_mapping = type(self)(
            resolved_megatron_param,
            resolved_hf_patterns,
            self.num_experts,
        )
        new_mapping.allow_hf_name_mismatch = self.allow_hf_name_mismatch
        return new_mapping


def _convert_hf_weight_names(hf_param: str | dict[str, str], mapped_name: str) -> list[str]:
    if isinstance(hf_param, dict):
        return [
            value.removesuffix(".weight") + mapped_name for value in hf_param.values() if value.endswith(".weight")
        ]
    if isinstance(hf_param, str) and hf_param.endswith(".weight"):
        return [hf_param.removesuffix(".weight") + mapped_name]
    return []


def convert_to_amax_map(
    mappings: list[MegatronParamMapping], mapped_name: str = ".weight_quantizer._amax"
) -> list[MegatronParamMapping]:
    """Convert weight mappings to amax mappings for quantization.

    This function converts parameter mappings for weights to their corresponding
    amax (absolute maximum) parameter mappings used in quantization. For example:
    - "layer.weight" -> "layer.weight_quantizer._amax"

    Args:
        mappings: List of MegatronParamMapping objects for weight parameters

    Returns:
        List of new MegatronParamMapping objects for amax parameters

    Note:
        Mappings ending in '.weight' become regular amax mappings. MoE expert
        mappings ending in '.weight*' become fanout mappings because Megatron
        stores a shared expert amax while HF stores per-expert amax names.
    """
    extended_mapping = []

    for mapping in mappings:
        if mapping.megatron_param.endswith(".weight*"):
            new_megatron_param = mapping.megatron_param[: -len(".weight*")] + mapped_name
            hf_patterns = _convert_hf_weight_names(mapping.hf_param, mapped_name)

            if hf_patterns:
                extended_mapping.append(
                    MoeAmaxFanoutMapping(
                        megatron_param=new_megatron_param,
                        hf_patterns=hf_patterns,
                    )
                )
            continue

        if not mapping.megatron_param.endswith(".weight"):
            continue

        new_megatron_param = mapping.megatron_param.removesuffix(".weight") + mapped_name

        if isinstance(mapping.hf_param, dict):
            # For dict-based hf_param (e.g., QKVMapping, GatedMLPMapping)
            new_hf_param = {
                key: (value.removesuffix(".weight") + mapped_name if value.endswith(".weight") else value)
                for key, value in mapping.hf_param.items()
            }
        elif isinstance(mapping.hf_param, str):
            if mapping.hf_param.endswith(".weight"):
                new_hf_param = mapping.hf_param.removesuffix(".weight") + mapped_name
            else:
                continue
        else:
            print(f"Unknown hf_param type: {type(mapping.hf_param)}")
            continue

        # Amax tensors are small scalars and should not be TP-sharded. Always map
        # them as replicated parameters to avoid any TP chunking logic.
        # For dict-based mappings (e.g., QKV or gate/up), emit one fan-out mapping
        # so each of q/k/v (or gate/up) receives the same amax in Megatron->HF.
        if isinstance(new_hf_param, dict):
            if not new_hf_param:
                continue
            new_mapping = AmaxFanoutMapping(
                megatron_param=new_megatron_param,
                hf_params=list(new_hf_param.values()),
            )
            extended_mapping.append(new_mapping)
        else:
            new_mapping = AmaxMapping(
                megatron_param=new_megatron_param,
                hf_param=new_hf_param,
            )
            extended_mapping.append(new_mapping)

    return extended_mapping
