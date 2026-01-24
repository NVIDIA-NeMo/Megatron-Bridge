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

"""Quantization mapping utilities for Megatron Bridge.

This module provides mapping classes and utilities for handling quantization
parameters (Currently, only amax values) during Megatron -> HuggingFace conversions.
"""

import re
from typing import Optional

from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping, ReplicatedMapping


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

    def megatron_to_hf(
        self, megatron_weights, megatron_module
    ):
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
    """Amax mapping for MoE layers where Megatron has shared quantizer across experts.

    In MoE layers:
    - Megatron: One quantizer shared across all experts (no expert index in param name)
      e.g., decoder.layers.*.mlp.experts.linear_fc1.weight_quantizer._amax
    - HF: Per-expert quantizers (with expert index wildcard)
      e.g., model.layers.*.mlp.experts.*.gate_proj.weight_quantizer._amax

    This mapping handles the asymmetry where Megatron has 1 wildcard (layer) but
    HF has 2 wildcards (layer + expert). During megatron_to_hf, the single Megatron
    amax is fanned out to all expert indices.

    Expert Parallelism (EP) Note:
        Unlike regular MoE weight mappings which need to gather weights from different
        EP ranks (each rank owns different experts), the shared quantizer case is simpler:
        - ALL EP ranks have the SAME quantizer value (it's shared across experts)
        - No EP communication (all_gather) is needed
        - We simply replicate the same value to all expert quantizer names
        This is why we inherit from AmaxMapping -> ReplicatedMapping, which already
        handles the "same value everywhere" pattern correctly.

    Example:
        Megatron: decoder.layers.0.mlp.experts.linear_fc1.weight_quantizer._amax
        HF: model.layers.0.mlp.experts.0.gate_proj.weight_quantizer._amax
            model.layers.0.mlp.experts.1.gate_proj.weight_quantizer._amax
            ... (for all num_experts)
    """

    # Regex to find the expert index placeholder in HF patterns
    # Matches patterns like "model.layers.0.mlp.experts.*.gate_proj..."
    # where we need to enumerate the expert wildcard *
    _EXPERT_PATTERN = re.compile(r"(.*\.experts\.)(\*)(\..+)")

    def __init__(self, megatron_param: str, hf_patterns: list[str], num_experts: Optional[int] = None):
        """
        Args:
            megatron_param: Megatron pattern with layer wildcard only
            hf_patterns: HF patterns with layer AND expert wildcards
            num_experts: Number of experts (if None, will be inferred at runtime)
        """
        self.hf_patterns = hf_patterns
        self.num_experts = num_experts
        # Use placeholder for parent init
        super().__init__(megatron_param, hf_patterns[0] if hf_patterns else "")
    
    def _validate_patterns(self):
        """Skip wildcard validation - we intentionally have asymmetric wildcards.
        
        MoeAmaxFanoutMapping allows:
        - Megatron: 1 wildcard (layer index)
        - HF: 2 wildcards (layer index + expert index)
        
        The expert wildcard is expanded at runtime in megatron_to_hf() when we
        know the actual number of experts.
        """
        pass  # Intentionally skip validation

    @property
    def is_expert(self) -> bool:
        """Shared quantizers are NOT expert-parallel - all EP ranks have the same value.
        
        Unlike regular MoE weight parameters which are distributed across EP ranks
        (each rank owns different experts), the shared quantizer has the same value
        on all EP ranks. No EP gathering is needed - we simply replicate the value
        to all expert quantizer names in megatron_to_hf().
        """
        return False

    def megatron_to_hf(self, megatron_weights, megatron_module):
        """Fan out the single amax to all expert HF params."""
        base = super().megatron_to_hf(megatron_weights, megatron_module)
        if not base:
            return {}
        weight = next(iter(base.values()))

        # Determine number of experts
        num_experts = self.num_experts
        if num_experts is None:
            # The model_config (TransformerConfig) is attached to modules during task building
            # in model_bridge.py, so megatron_module.config should have num_moe_experts
            if hasattr(megatron_module, 'config') and hasattr(megatron_module.config, 'num_moe_experts'):
                num_experts = megatron_module.config.num_moe_experts
            elif hasattr(megatron_module, 'config') and hasattr(megatron_module.config, 'num_experts'):
                num_experts = megatron_module.config.num_experts
            else:
                num_experts = self._infer_num_experts(megatron_module)

        if num_experts is None or num_experts <= 0:
            # Can't determine num_experts - this shouldn't happen if model_config was attached
            raise RuntimeError(
                f"MoeAmaxFanoutMapping: Could not determine num_experts for {self.megatron_param}. "
                f"The model config may not have been attached to the module. "
                f"Consider setting num_experts explicitly when creating the mapping."
            )

        # Fan out to all experts for all HF patterns
        result = {}
        for pattern in self.hf_patterns:
            match = self._EXPERT_PATTERN.match(pattern)
            if match:
                prefix, _, suffix = match.groups()
                for expert_idx in range(num_experts):
                    hf_name = f"{prefix}{expert_idx}{suffix}"
                    result[hf_name] = weight
            else:
                result[pattern] = weight

        return result

    def _infer_num_experts(self, megatron_module):
        """Try to infer number of experts from module structure."""
        module = megatron_module
        for _ in range(10):  # Limit depth
            if hasattr(module, 'config'):
                if hasattr(module.config, 'num_moe_experts'):
                    return module.config.num_moe_experts
                if hasattr(module.config, 'num_experts'):
                    return module.config.num_experts
            if hasattr(module, 'num_experts'):
                return module.num_experts
            if not hasattr(module, 'parent') and not hasattr(module, '_parent'):
                break
            module = getattr(module, 'parent', None) or getattr(module, '_parent', None)
            if module is None:
                break
        return None

    def resolve(self, captures: tuple[str, ...]):
        """Resolve layer wildcard only (Megatron side), keep expert wildcard for HF."""
        resolved_megatron_param = self.megatron_param
        capture_index = 0

        # Resolve wildcards in megatron_param (typically just layer number)
        while "**" in resolved_megatron_param and capture_index < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("**", captures[capture_index], 1)
            capture_index += 1
        while "*" in resolved_megatron_param and capture_index < len(captures):
            resolved_megatron_param = resolved_megatron_param.replace("*", captures[capture_index], 1)
            capture_index += 1

        # For HF patterns, resolve only the FIRST wildcard (layer), keep expert wildcard
        resolved_hf_patterns = []
        for pattern in self.hf_patterns:
            if captures:
                # Replace only the first * with layer number, keep others
                resolved = pattern.replace("*", captures[0], 1)
                resolved_hf_patterns.append(resolved)
            else:
                resolved_hf_patterns.append(pattern)

        new_mapping = type(self)(resolved_megatron_param, resolved_hf_patterns, self.num_experts)
        new_mapping.allow_hf_name_mismatch = self.allow_hf_name_mismatch
        return new_mapping


def convert_to_amax_map(mappings: list[MegatronParamMapping], mapped_name='.weight_quantizer._amax') -> list[MegatronParamMapping]:
    """Convert weight mappings to amax mappings for quantization.
    
    This function converts parameter mappings for weights to their corresponding 
    amax (absolute maximum) parameter mappings used in quantization. For example:
    - "layer.weight" -> "layer.weight_quantizer._amax"
    - MoE: "layer.experts.linear_fc1.weight*" -> "layer.experts.linear_fc1.weight_quantizer._amax"
           (single Megatron quantizer fans out to all HF expert quantizers)
    
    Args:
        mappings: List of MegatronParamMapping objects for weight parameters
        mapped_name: The quantizer suffix to append (default: '.weight_quantizer._amax')
        
    Returns:
        List of new MegatronParamMapping objects for amax parameters
        
    Note:
        - Regular mappings ending in '.weight' are converted to amax mappings
        - MoE mappings ending in '.weight*' (with expert index suffix) get special 
          handling where the single shared Megatron quantizer maps to all expert 
          quantizers on the HF side
    """
    extended_mapping = []
    
    for mapping in mappings:
        megatron_param = mapping.megatron_param
        
        # Check for MoE pattern: ends with .weight* (expert-indexed weights)
        # Example: decoder.layers.*.mlp.experts.linear_fc1.weight*
        is_moe_expert_weight = megatron_param.endswith('.weight*')
        
        if is_moe_expert_weight:
            # MoE case: Megatron has single quantizer shared across experts
            # Strip the trailing * to get the base quantizer name
            # "decoder.layers.*.mlp.experts.linear_fc1.weight*" 
            # -> "decoder.layers.*.mlp.experts.linear_fc1" + mapped_name
            base_param = megatron_param[:-len('.weight*')]
            new_megatron_param = base_param + mapped_name
            
            if isinstance(mapping.hf_param, dict):
                # GatedMLPMapping case - collect all HF patterns (keep expert wildcard)
                hf_patterns = []
                for key, value in mapping.hf_param.items():
                    if '.weight' in value:
                        # Replace .weight with quantizer suffix, keep expert *
                        new_hf = value.replace('.weight', mapped_name)
                        hf_patterns.append(new_hf)
                
                if hf_patterns:
                    new_mapping = MoeAmaxFanoutMapping(
                        megatron_param=new_megatron_param,
                        hf_patterns=hf_patterns,
                    )
                    extended_mapping.append(new_mapping)
            elif isinstance(mapping.hf_param, str):
                if '.weight' in mapping.hf_param:
                    new_hf = mapping.hf_param.replace('.weight', mapped_name)
                    new_mapping = MoeAmaxFanoutMapping(
                        megatron_param=new_megatron_param,
                        hf_patterns=[new_hf],
                    )
                    extended_mapping.append(new_mapping)
            continue
        
        # Regular case: ends with .weight (non-MoE)
        if not megatron_param.endswith('.weight'):
            continue
        
        new_megatron_param = megatron_param.replace('.weight', mapped_name)
        
        if isinstance(mapping.hf_param, dict):
            # For dict-based hf_param (e.g., QKVMapping, GatedMLPMapping)
            new_hf_param = {
                key: value.replace('.weight', mapped_name) if value.endswith('.weight') else value
                for key, value in mapping.hf_param.items()
            }
        elif isinstance(mapping.hf_param, str):
            if mapping.hf_param.endswith('.weight'):
                new_hf_param = mapping.hf_param.replace('.weight', mapped_name)
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
