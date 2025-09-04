# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Check if this is DoRA based on use_dora flag
if config.get("use_dora", False):
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

from typing import List, Optional, Union

import torch
from peft import LoraConfig

from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
)
from megatron.bridge.peft.conversion.param_mapping import (
    AdapterAutoMapping,
    AdapterGatedMLPMapping,
    AdapterQKVMapping,
)
from megatron.bridge.peft.conversion.peft_bridge import MegatronPEFTBridge
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters
from megatron.bridge.peft.lora.dora import DoRA
from megatron.bridge.peft.lora.lora import LoRA
from megatron.bridge.peft.lora.canonical_lora import CanonicalLoRA


@MegatronPEFTBridge.register_bridge(source=LoraConfig, target=LoRA)  # Handles both LoRA and DoRA
class LoRABridge(MegatronPEFTBridge):
    """
    Unified Megatron Bridge for LoRA, DoRA, and Canonical LoRA adapters.
    
    This bridge automatically detects the adapter type based on configuration:
    - DoRA: 'use_dora' flag set to True
    - Canonical LoRA: Target modules use individual projections (q_proj, k_proj, etc.)
    - Fused LoRA: Default case with fused projections

    As a user you would not use this bridge directly, but through `AutoPEFTBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> from megatron.bridge.peft import AutoPEFTBridge
        >>> base_bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B")
        >>>
        >>> # Works with all LoRA variants
        >>> lora_bridge = AutoPEFTBridge.from_hf_pretrained("username/llama-lora-adapters")
        >>> dora_bridge = AutoPEFTBridge.from_hf_pretrained("username/llama-dora-adapters")
        >>> canonical_bridge = AutoPEFTBridge.from_hf_pretrained("username/llama-canonical-lora")
        >>>
        >>> lora_model = lora_bridge.to_megatron_model(base_bridge)
        >>> dora_model = dora_bridge.to_megatron_model(base_bridge)
        >>> canonical_model = canonical_bridge.to_megatron_model(base_bridge)
    """

    def peft_bridge(self, adapters: PreTrainedAdapters) -> Union[LoRA, DoRA, CanonicalLoRA]:
        """Convert HF adapter config to Megatron LoRA, DoRA, or Canonical LoRA transform.
        
        Automatically detects adapter type based on configuration:
        - DoRA: 'use_dora' flag set to True
        - Canonical LoRA: Target modules use individual projections
        - Fused LoRA: Default case
        """
        config = adapters.config
        hf_target_modules = getattr(config, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        
        # Detect canonical LoRA by checking if target modules use individual projections
        canonical_indicators = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}
        is_canonical = any(target in canonical_indicators for target in hf_target_modules)
        
        print(f"📋 Adapter Analysis:")
        print(f"   • Target modules: {hf_target_modules}")
        print(f"   • Detected as: {'DoRA' if getattr(config, 'use_dora', False) else 'Canonical LoRA' if is_canonical else 'Fused LoRA'}")
        print(f"   • Config attributes: r={config.r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
        
        if config.get("use_dora", False):
            # DoRA: LoRA + magnitude vectors
            megatron_target_modules = self._hf_to_megatron_target_modules(hf_target_modules)
            print(f"   • Converted to Megatron targets: {megatron_target_modules}")
            return DoRA(
                target_modules=megatron_target_modules,
                dim=config.r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                dropout_position="pre",
                lora_A_init_method="xavier",
                lora_B_init_method="zero",
                # lora_dtype=self._parse_dtype(config.get("lora_dtype"),
            )
        elif is_canonical:
            # Canonical LoRA: Individual projections
            canonical_target_modules = self._hf_to_canonical_target_modules(hf_target_modules)
            print(f"   • Converted to canonical targets: {canonical_target_modules}")
            self._canonical_mode = True  # Track for mapping logic
            return CanonicalLoRA(
                target_modules=canonical_target_modules,
                dim=config.r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                lora_A_init_method="xavier",
                lora_B_init_method="zero",
                # lora_dtype=self._parse_dtype(config.lora_dtype),
            )
        else:
            # Fused LoRA: Default case
            megatron_target_modules = self._hf_to_megatron_target_modules(hf_target_modules)
            print(f"   • Converted to Megatron targets: {megatron_target_modules}")
            self._canonical_mode = False  # Track for mapping logic
            return LoRA(
                target_modules=megatron_target_modules,
                dim=config.r,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                dropout_position="pre",
                lora_A_init_method="xavier",
                lora_B_init_method="zero",
                lora_dtype=self._parse_dtype(config.get("lora_dtype")),
            )

    def create_peft_mapping(
        self,
        base_mapping: MegatronParamMapping,
        adapter_megatron_param: str
    ) -> MegatronParamMapping:
        """Create LoRA/DoRA/Canonical adapter mapping from base mapping.
        
        Uses different mapping strategies based on canonical vs fused mode:
        - Canonical: Uses only AutoMapping (no fusion)
        - Fused: Uses specialized QKV/GatedMLP mappings
        """
        adapter_type = self._get_adapter_type_from_param(adapter_megatron_param)
        hf_suffix = self._get_hf_suffix_for_adapter_type(adapter_type)
        
        # Different mapping strategy for canonical vs fused
        if getattr(self, '_canonical_mode', False):
            # Canonical LoRA: Always use AutoMapping (no fusion)
            match base_mapping:
                case QKVMapping() | GatedMLPMapping():
                    # For canonical LoRA, construct proper HF param with wildcards preserved
                    if isinstance(base_mapping.hf_param, dict):
                        # QKV or GatedMLP mapping - use first projection as template
                        template_hf = next(iter(base_mapping.hf_param.values()))
                    else:
                        # Auto mapping - use the hf_param directly
                        template_hf = base_mapping.hf_param
                    
                    # Replace .weight with adapter suffix to preserve wildcards
                    canonical_hf_param = template_hf.replace('.weight', hf_suffix)
                    
                    return AdapterAutoMapping.from_base_mapping(
                        AutoMapping(
                            hf_param=canonical_hf_param,
                            megatron_param=adapter_megatron_param
                        ),
                        adapter_megatron_param,
                        hf_suffix
                    )
                case AutoMapping():
                    return AdapterAutoMapping.from_base_mapping(
                        base_mapping, adapter_megatron_param, hf_suffix
                    )
                case _:
                    raise self._unsupported_mapping_error(base_mapping)
        else:
            # Fused LoRA/DoRA: Use specialized mappings
            match base_mapping:
                case QKVMapping():
                    return AdapterQKVMapping.from_base_mapping(
                        base_mapping, adapter_megatron_param, hf_suffix
                    )
                case GatedMLPMapping():
                    return AdapterGatedMLPMapping.from_base_mapping(
                        base_mapping, adapter_megatron_param, hf_suffix
                    )
                case AutoMapping():
                    return AdapterAutoMapping.from_base_mapping(
                        base_mapping, adapter_megatron_param, hf_suffix
                    )
                case _:
                    raise self._unsupported_mapping_error(base_mapping)
    
    def _get_adapter_type_from_param(self, adapter_param: str) -> str:
        """Determine LoRA/DoRA adapter parameter type from parameter name."""
        match adapter_param:
            case _ if ".linear_in." in adapter_param:
                return "A_matrix"
            case _ if ".linear_out." in adapter_param:
                return "B_matrix"
            case _ if ".weight_magnitude" in adapter_param:
                return "magnitude"
            case _:
                return "unknown"
    
    def _get_hf_suffix_for_adapter_type(self, adapter_type: str) -> str:
        """Get HuggingFace suffix for LoRA/DoRA adapter parameter type."""
        match adapter_type:
            case "A_matrix":
                return ".lora_A.weight"
            case "B_matrix":
                return ".lora_B.weight"
            case "magnitude":
                return ".lora_magnitude_vector"
            case _:
                return ".weight"

    def _hf_to_megatron_target_modules(self, hf_targets: List[str]) -> List[str]:
        """Convert HF target module names to Megatron target module names."""
        # Dictionary maps HF projection names -> Megatron fused module names
        hf_to_megatron_map = {
            "q_proj": "linear_qkv",
            "k_proj": "linear_qkv",
            "v_proj": "linear_qkv",
            "o_proj": "linear_proj",
            "gate_proj": "linear_fc1",
            "up_proj": "linear_fc1",
            "down_proj": "linear_fc2",
        }

        megatron_targets = set()
        for hf_target in hf_targets:
            if hf_target in hf_to_megatron_map:
                megatron_targets.add(hf_to_megatron_map[hf_target])
            else:
                # Pass through unknown targets (might already be Megatron names)
                megatron_targets.add(hf_target)

        return list(megatron_targets)
    
    def _hf_to_canonical_target_modules(self, hf_targets: List[str]) -> List[str]:
        """Convert HF target module names to canonical Megatron target module names.
        
        For canonical LoRA, convert to individual projection names.
        """
        hf_to_canonical_map = {
            "q_proj": "linear_q",
            "k_proj": "linear_k",
            "v_proj": "linear_v",
            "o_proj": "linear_proj",
            "gate_proj": "linear_fc1_gate",
            "up_proj": "linear_fc1_up",
            "down_proj": "linear_fc2",
        }
        
        canonical_targets = []
        for hf_target in hf_targets:
            if hf_target in hf_to_canonical_map:
                canonical_targets.append(hf_to_canonical_map[hf_target])
            else:
                # Pass through unknown targets (might already be canonical)
                canonical_targets.append(hf_target)
        
        return canonical_targets
