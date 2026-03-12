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

from megatron.bridge.models.conversion.param_mapping import (
    MegatronParamMapping,
    ReplicatedMapping,
)


class AmaxMapping(ReplicatedMapping):
    """Amax mapping for quantization."""

    def __init__(self, megatron_param: str, hf_param: str):
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


def convert_to_amax_map(
    mappings: list[MegatronParamMapping], mapped_name=".weight_quantizer._amax"
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
        Only mappings with parameter names ending in '.weight' are converted.
        Other mappings are ignored.
    """
    extended_mapping = []

    for mapping in mappings:
        if not mapping.megatron_param.endswith(".weight"):
            continue

        new_megatron_param = mapping.megatron_param.replace(".weight", mapped_name)

        if isinstance(mapping.hf_param, dict):
            # For dict-based hf_param (e.g., QKVMapping, GatedMLPMapping)
            new_hf_param = {
                key: (value.replace(".weight", mapped_name) if value.endswith(".weight") else value)
                for key, value in mapping.hf_param.items()
            }
        elif isinstance(mapping.hf_param, str):
            if mapping.hf_param.endswith(".weight"):
                new_hf_param = mapping.hf_param.replace(".weight", mapped_name)
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
