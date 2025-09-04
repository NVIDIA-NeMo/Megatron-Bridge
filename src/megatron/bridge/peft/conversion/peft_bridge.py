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

"""PEFT bridge base class for HuggingFace PEFT integration.

This module provides the abstract base class for bridging between HuggingFace PEFT
configurations and Megatron PEFT implementations, following the same battle-tested
patterns as the model bridge infrastructure.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union, TYPE_CHECKING

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_pg_size
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask, HFWeightTuple
from megatron.bridge.models.conversion.param_mapping import QKVMapping, GatedMLPMapping, AutoMapping, MegatronParamMapping
from megatron.bridge.models.conversion.utils import extract_sort_key, get_module_and_param_from_name
from megatron.bridge.models.decorators.dispatch import dispatch
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.conversion.param_mapping import AdapterAutoMapping, AdapterQKVMapping, AdapterGatedMLPMapping
from megatron.bridge.peft.conversion.pretrained_adapters import PreTrainedAdapters
from megatron.bridge.utils.common_utils import unwrap_model

if TYPE_CHECKING:
    from megatron.bridge.models.conversion.auto_bridge import AutoBridge

logger = logging.getLogger(__name__)

# Registry for bridge implementations
_BRIDGE_REGISTRY: Dict[Type, Type['MegatronPEFTBridge']] = {}


class MegatronPEFTBridge(ABC):
    """Abstract base class for PEFT adapter bridges.

    This class follows the same battle-tested patterns as MegatronModelBridge,
    providing robust conversion between HuggingFace PEFT adapters and Megatron
    PEFT implementations with full distributed training support.
    """

    def __init__(self, base_bridge: Optional['AutoBridge'] = None) -> None:
        """Initialize PEFT bridge with optional base model bridge.

        Args:
            base_bridge: Optional base model bridge for deriving mappings
        """
        self.base_bridge = base_bridge

    @classmethod
    def register_bridge(cls, *, source: Type, target: Type[PEFT]):
        """Decorator for registering PEFT bridge implementations."""
        def decorator(bridge_class: Type['MegatronPEFTBridge']) -> Type['MegatronPEFTBridge']:
            register_bridge_implementation(
                source=source,
                target=target,
                bridge_class=bridge_class
            )
            return bridge_class
        return decorator

    @abstractmethod
    def peft_bridge(self, adapters: PreTrainedAdapters) -> PEFT:
        """Convert HF adapter config to Megatron PEFT transform."""
        pass
    
    @abstractmethod
    def create_peft_mapping(
        self,
        base_mapping: MegatronParamMapping,
        adapter_megatron_param: str
    ) -> MegatronParamMapping:
        """Create adapter mapping from base mapping.
        
        Each bridge implements this method to handle the conversion from
        base model parameter mappings to adapter parameter mappings.
        Bridge determines the appropriate HF parameter suffix based on
        the adapter parameter name and PEFT type.
        
        Args:
            base_mapping: Base model parameter mapping to adapt
            adapter_megatron_param: Megatron adapter parameter name
            
        Returns:
            Appropriate adapter mapping instance
        """
        pass

    def mapping_registry(self, adapters: PreTrainedAdapters) -> MegatronMappingRegistry:
        """Universal algorithm that infers adapter mappings automatically.
        
        This method differs from the model bridge pattern where mappings are manually defined.
        Instead, it algorithmically derives adapter mappings by combining:
        - Base model mappings (from self.base_mapping_registry)
        - PEFT target modules and parameter patterns (from self.peft)
        """
        adapter_mappings = []
        
        # Get PEFT instance and base mappings
        peft = self.peft_bridge(adapters)
        base_mapping_registry = self.base_bridge._model_bridge.mapping_registry()
        
        # Set temporary PEFT context for the mapping conversion
        self._current_peft = peft
        
        try:
            # Iterate through base model parameter mappings
            for base_mapping in base_mapping_registry.get_all_mappings():
                if peft.affects_module(base_mapping.megatron_param):
                    # Convert base mapping to adapter mappings using PEFT knowledge
                    converted = self._convert_base_to_adapter_mapping(base_mapping)
                    adapter_mappings.extend(converted)
            
            if not adapter_mappings:
                raise ValueError(
                    f"No compatible mappings found for {peft.__class__.__name__} "
                    f"target modules {peft.target_modules}. Ensure the base model "
                    f"has parameter mappings for these module types."
                )
        finally:
            # Clean up temporary PEFT context
            delattr(self, '_current_peft')
        
        return MegatronMappingRegistry(*adapter_mappings)
    
    
    def _unsupported_mapping_error(self, base_mapping: MegatronParamMapping) -> ValueError:
        """Generate expressive error for unsupported mapping types."""
        bridge_name = self.__class__.__name__
        mapping_type = type(base_mapping).__name__
        
        return ValueError(
            f"\n✗ {bridge_name} does not support {mapping_type}\n\n"
            f"The {bridge_name} implementation does not include adapter mapping "
            f"logic for {mapping_type} parameters.\n\n"
            f"To add support:\n"
            f"1. Add {mapping_type} case to {bridge_name}.create_peft_mapping()\n"
            f"2. Create appropriate Adapter{mapping_type.replace('Mapping', 'Mapping')} instance\n\n"
            f"Example:\n"
            f"  elif isinstance(base_mapping, {mapping_type}):\n"
            f"      return Adapter{mapping_type}.from_base_mapping(\n"
            f"          base_mapping, adapter_megatron_param, hf_suffix\n"
            f"      )"
        )
    
    def _parse_dtype(self, dtype_str: Optional[str]) -> Optional[torch.dtype]:
        """Parse dtype string to torch.dtype using elegant getattr approach."""
        if dtype_str is None:
            return None
        
        # Use getattr to dynamically get the dtype from torch module
        try:
            return getattr(torch, dtype_str.lower())
        except AttributeError:
            # Fallback for common aliases or invalid types
            return None
    
    def _convert_base_to_adapter_mapping(self, base_mapping: MegatronParamMapping) -> List[MegatronParamMapping]:
        """Convert base mapping to adapter mappings using PEFT and HF-specific logic."""
        # Get Megatron adapter parameters from PEFT
        base_megatron = base_mapping.megatron_param
        adapter_megatron_params = self._current_peft.get_megatron_adapter_params(base_megatron)
        
        adapter_mappings = []
        
        # Let bridge handle the parameter-to-mapping conversion
        for adapter_megatron_param in adapter_megatron_params:
            # Bridge determines adapter type and creates mapping directly
            adapter_mapping = self.create_peft_mapping(
                base_mapping, adapter_megatron_param
            )
            adapter_mappings.append(adapter_mapping)
        return adapter_mappings
    

    def _megatron_global_param_names_all_pp_ranks(
        self, megatron_model: Union[MegatronModule, List[MegatronModule]]
    ) -> List[str]:
        """Get all PEFT parameter names across all pipeline parallel ranks.

        This follows the same pattern as the model bridge but filters for adapter parameters.
        """
        # Cache the result after first call
        if hasattr(self, "_cached_adapter_param_names"):
            return self._cached_adapter_param_names

        # Compute the result
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        model_config = unwrap_model(megatron_model)[0].config
        global_param_names = []

        # Ensure megatron_model is a list for consistent handling
        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]

        for vp_stage, model in enumerate(models_list):
            for local_param_name, _ in model.named_parameters():
                # Only include adapter parameters
                if ".adapter." not in local_param_name:
                    continue

                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = self._megatron_local_name_to_global(
                    models_list, model_config, local_param_name, vp_stage
                )
                global_param_names.append(global_param_name)

        # Guard distributed operations for single-process runs
        if torch.distributed.is_initialized():
            gathered_global_param_names = [None] * get_pg_size(pp_group)
            torch.distributed.all_gather_object(gathered_global_param_names, global_param_names, group=pp_group)
            # Flatten, sort and remove duplicates - order matters for distributed coordination
            flattened_names = list(set(sum(gathered_global_param_names, [])))
        else:
            flattened_names = global_param_names
        gathered_global_param_names = sorted(flattened_names, key=extract_sort_key)

        # Cache the result
        self._cached_adapter_param_names = gathered_global_param_names
        return self._cached_adapter_param_names

    def _with_progress_tracking(self, tasks, description: str, show_progress: bool = True):
        """Helper method to wrap an iterable with progress tracking."""
        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        bridge_name = self.__class__.__name__

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[bridge]}"),
            disable=not (is_main_rank and show_progress),
        ) as progress:
            task_id = progress.add_task(description, total=len(tasks), bridge=bridge_name)

            for task in tasks:
                yield task
                progress.update(task_id, advance=1)

    def build_conversion_tasks(
        self,
        adapters: PreTrainedAdapters,
        megatron_model: List[MegatronModule],
        base_bridge: Optional['AutoBridge'] = None
    ) -> List[Optional[WeightConversionTask]]:
        """Build conversion tasks following the battle-tested model bridge pattern.

        This method follows the exact same algorithm as MegatronModelBridge.build_conversion_tasks
        but filters for adapter parameters only.
        """
        # Ensure adapters has the required state structure
        if not (hasattr(adapters, "state") and hasattr(adapters.state, "source")):
            raise ValueError("adapters.state.source is required for weight ordering")

        hf_keys: Iterable[str] = adapters.state.source.get_all_keys()
        model_config = unwrap_model(megatron_model)[0].config
        mapping_registry = self.mapping_registry(adapters)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()

        # Get all adapter parameter names across PP ranks
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(megatron_model)
        global_names_index_dict = {name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)}

        tasks = [None] * len(sorted_global_param_names_all_pp_ranks)

        for vp_stage, model in enumerate(megatron_model):
            for local_name, _ in model.named_parameters():
                # Skip non-adapter parameters
                if ".adapter." not in local_name:
                    continue

                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = self._megatron_local_name_to_global(megatron_model, model_config, local_name, vp_stage)

                if global_name not in global_names_index_dict:
                    continue

                global_name_idx = global_names_index_dict[global_name]
                mapping = mapping_registry.megatron_to_hf_lookup(global_name)

                if not mapping:
                    logger.debug(f"No mapping found for adapter parameter: {global_name}")
                    continue

                # Ensure HF weights exist
                if isinstance(mapping.hf_param, str):
                    if mapping.hf_param not in hf_keys:
                        logger.warning(f"Can't find {mapping.hf_param} in HF adapter weights")
                        continue
                else:
                    # Handle Dict[str, str] mappings
                    missing_params = [hf_param for hf_param in mapping.hf_param.values() if hf_param not in hf_keys]
                    if missing_params:
                        logger.warning(f"Can't find HF adapter parameters: {missing_params}")
                        continue

                # Get the local module and weight using battle-tested utility
                local_module, local_weights = get_module_and_param_from_name(megatron_model, local_name, vp_stage)

                tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

        # Fill remaining slots for PP communication (same pattern as model bridge)
        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            mapping = mapping_registry.megatron_to_hf_lookup(global_name)
            if tasks[idx] is None:
                tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                )

        return tasks

    def load_adapters_hf_to_megatron(
        self,
        adapters: PreTrainedAdapters,
        megatron_model: Union[MegatronModule, List[MegatronModule]],
        base_bridge: Optional['AutoBridge'] = None
    ) -> None:
        """Load HF adapter weights into Megatron model following model bridge patterns."""
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        hf_to_megatron_tasks = self.build_conversion_tasks(adapters, megatron_model, base_bridge)
        hf_state_dict: Mapping[str, torch.Tensor] = adapters.state

        description = f"Loading adapters from {adapters.model_name_or_path}"
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            # Skip if module not on current rank
            if task.megatron_module is None:
                continue

            # Fetch source tensor(s) from HF state dict
            if isinstance(task.mapping.hf_param, str):
                if task.mapping.hf_param not in hf_state_dict:
                    continue
                hf_weights = hf_state_dict[task.mapping.hf_param]
            else:
                # Handle Dict[str, str] mappings
                hf_weights = {}
                for k, v in task.mapping.hf_param.items():
                    if v in hf_state_dict:
                        hf_weights[k] = hf_state_dict[v]
                if not hf_weights:
                    continue

            # Delegate conversion & distribution to the mapping
            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

            # Copy into Megatron param if this rank received a shard
            if converted_weights is not None:
                assert task.param_weight is not None, "param_weight required for loading"

                # Check shape compatibility
                if converted_weights.shape != task.param_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for adapter param {task.mapping.megatron_param}:\n"
                        f"  Expected shape: {task.param_weight.shape}\n"
                        f"  Got shape: {converted_weights.shape}\n"
                        f"  Mapping type: {type(task.mapping).__name__}\n"
                        f"  HF mapping: {task.mapping.hf_param}"
                    )
                task.param_weight.data.copy_(converted_weights)

    def stream_adapters_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModule, List[MegatronModule]],
        adapters: PreTrainedAdapters,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        """Stream adapter weights from Megatron to HF format following model bridge patterns."""
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(adapters, megatron_model)

        for task in self._with_progress_tracking(conversion_tasks, "Converting adapters to HuggingFace", show_progress):
            if task.param_weight is None:
                continue

            converted_weights_dict = task.mapping.megatron_to_hf(task.param_weight, task.megatron_module)

            # All ranks get the full tensor
            for hf_name, tensor in converted_weights_dict.items():
                final_tensor = tensor.cpu() if cpu else tensor
                yield HFWeightTuple(hf_name, final_tensor)

    def _megatron_local_name_to_global(
        self,
        models: Union[MegatronModule, List[MegatronModule]],
        config,
        param_name: str,
        vp_stage: Optional[int] = None,
    ) -> str:
        """Convert local parameter names to global names for adapter parameters.

        This is the same logic as model_bridge but focused on adapter parameters.
        """
        # Import here to avoid circular imports
        from megatron.bridge.models.conversion.model_bridge import _megatron_local_name_to_global
        return _megatron_local_name_to_global(models, config, param_name, vp_stage)

    def _unwrap_name(self, name: str) -> str:
        """Unwrap parameter name from DDP wrappers."""
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")

        while name.startswith("module."):
            name = name[len("module."):]
        return name



def register_bridge_implementation(
    *,
    source: Type,
    target: Type[PEFT],
    bridge_class: Type[MegatronPEFTBridge]
) -> None:
    """Register a PEFT bridge implementation with the dispatch system.
    
    Args:
        source: PEFT config class (e.g., LoraConfig)
        target: Megatron PEFT class (e.g., LoRA)
        bridge_class: MegatronPEFTBridge implementation class
    """
    _BRIDGE_REGISTRY[source] = bridge_class
    bridge_class_name = bridge_class.__name__
    
    @get_peft_bridge.impl(source)
    def _get_peft_bridge_impl(_) -> "MegatronPEFTBridge":
        bridge = bridge_class()
        return bridge
    
    @stream_adapters_megatron_to_hf.impl((source, target))
    def _adapters_to_hf_registered_impl(
        _,
        megatron_model: Union[MegatronModule, List[MegatronModule]],
        adapters: PreTrainedAdapters,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()
        return bridge.stream_adapters_megatron_to_hf(
            megatron_model, adapters, cpu=cpu, show_progress=show_progress, conversion_tasks=conversion_tasks
        )
    
    # Set meaningful names for debugging
    _get_peft_bridge_impl.__name__ = f"_peft_bridge_with_{bridge_class_name}"
    _adapters_to_hf_registered_impl.__name__ = f"_adapters_to_hf_with_{bridge_class_name}"


def list_registered_bridges() -> Dict[Type, Type[MegatronPEFTBridge]]:
    """List all registered bridge implementations."""
    return _BRIDGE_REGISTRY.copy()


# Core dispatch functions
@dispatch
def get_peft_bridge(peft_config_class) -> "MegatronPEFTBridge":
    """Get the appropriate PEFT bridge for a given PEFT configuration class."""
    ...


@dispatch
def stream_adapters_megatron_to_hf(
    dispatch_instance,
    megatron_model: Union[MegatronModule, List[MegatronModule]],
    adapters: PreTrainedAdapters,
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: Optional[List[WeightConversionTask]] = None,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron adapter weights to HuggingFace format."""
    ...
