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

import abc
import contextlib
import fnmatch
import itertools
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Type,
    TypeVar,
    Union,
)

import torch
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import (
    get_pg_size,
    unwrap_model,
)
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from transformers.modeling_utils import PreTrainedModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    MegatronParamMapping,
    ReplicatedMapping,
    RowParallelMapping,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.utils import (
    extract_sort_key,
    get_module_and_param_from_name,
    persistent_buffers,
)
from megatron.bridge.models.decorators.dispatch import dispatch
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.peft.canonical_lora import ModuleDict
from megatron.bridge.peft.lora import LoRAMerge
from megatron.bridge.peft.utils import get_adapter_attributes_from_linear
from megatron.bridge.utils.common_utils import print_rank_0


logger = logging.getLogger(__name__)

MappingT = TypeVar("MappingT", bound=MegatronParamMapping)
HFPreTrained = TypeVar("HFPreTrained")
ModelProviderTarget = TypeVar("ModelProviderTarget", bound=ModelProviderMixin)
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
_BridgeImplClass = TypeVar("_BridgeImplClass", bound="MegatronModelBridge")

ADAPTER_NAME_MAP = {
    # Map HF weight suffixes (keys) to CanonicalLoRA adapter keys (values)
    ".q_proj.weight": "adapter_q",
    ".k_proj.weight": "adapter_k",
    ".v_proj.weight": "adapter_v",
    ".gate_proj.weight": "adapter_gate",
    ".up_proj.weight": "adapter_up",
}


class MegatronWeightTuple(NamedTuple):
    """Tuple representing a Megatron model weight with its metadata."""

    param_name: str
    weight: torch.Tensor
    vp_stage: int


class HFWeightTuple(NamedTuple):
    """Tuple representing a HuggingFace model weight with its metadata."""

    param_name: str
    weight: torch.Tensor


@dataclass(frozen=True)
class WeightConversionTask(Generic[MappingT]):
    """A unified task for converting weights between HuggingFace and Megatron formats.

    This class combines both HF->Megatron and Megatron->HF conversion tasks since they
    have different method names (hf_to_megatron vs megatron_to_hf) and can coexist safely.

    The task encapsulates all information needed for weight conversion in either direction,
    with different fields being relevant depending on the conversion type.

    Attributes:
        param_name (str): *unwrapped, local* parameter name (no ``module.`` prefixes).
        global_param_name (str): *unwrapped, global* parameter name (no ``module.`` prefixes).
        mapping (MappingT): Concrete :pyclass:`MegatronParamMapping` instance responsible
            for weight transformation and distribution.

        pp_rank (Optional[int]): Pipeline-parallel rank that owns the parameter (required for saves).
        vp_stage (Optional[int]): Virtual-pipeline stage index (required for loads).
        megatron_module (Optional[torch.nn.Module]): Reference to the Megatron model or
            sub-module that owns the parameter (required for loads).
        param_weight (Optional[torch.Tensor]): The actual parameter tensor that will
            receive the converted weight (required for loads).

    """

    param_name: str
    global_param_name: str
    mapping: MappingT
    pp_rank: Optional[int] = None
    vp_stage: Optional[int] = None
    megatron_module: Optional[torch.nn.Module] = None
    param_weight: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class AdapterWeightConversionTask:
    """Task describing an adapter's LoRA weights for conversion or merging.

    The task reuses :class:`WeightConversionTask` to gather the adapter's
    linear_in/linear_out weights (if they are tensor-parallel) and carries the
    adapter metadata required by the merge step.
    """

    global_base_prefix: str
    adapter_key: Optional[str]  # For canonical LoRA only
    alpha: int
    dim: int
    linear_in_task: WeightConversionTask
    linear_out_task: WeightConversionTask


@dataclass(frozen=True)
class AdapterWeight:
    """Materialized adapter weights ready for merge."""

    global_base_prefix: str
    adapter_key: Optional[str]  # For canonical LoRA only
    alpha: int
    dim: int
    linear_in_weight: MegatronWeightTuple
    linear_out_weight: MegatronWeightTuple


def _megatron_local_name_to_global(
    models: MegatronModule | List[MegatronModule],
    config: TransformerConfig,
    param_name: str,
    vp_stage: Optional[int] = None,
) -> str:
    """Adjust layer number and expert number from local to global numbering."""
    # PP
    pp_group = parallel_state.get_pipeline_model_parallel_group()
    if "layers." in param_name and get_pg_size(pp_group) > 1:
        match = re.match(r"^(.+?\.layers\.\d+)", param_name)
        assert match is not None
        layer_prefix = match.group(1)
        _, layer_module = get_module_and_param_from_name(models=models, param_name=layer_prefix, vp_stage=vp_stage)

        local_layer_number = int(param_name.split("layers.")[1].split(".")[0])
        if isinstance(layer_module, MegatronModule):
            global_layer_number = layer_module.layer_number - 1
            param_name = param_name.replace(
                f"layers.{local_layer_number}.",
                f"layers.{global_layer_number}.",
            )

    # EP
    ep_group = parallel_state.get_expert_model_parallel_group()
    # For now adapters are not sharded across EP ranks
    if ".mlp.experts.linear_fc" in param_name and get_pg_size(ep_group) > 1 and not ".adapter." in param_name:
        num_experts = config.num_moe_experts
        num_experts_per_rank = num_experts // ep_group.size()

        def _update_expert_number(param_name: str, param_type: str) -> str:
            """Update expert number from local to global for weight or bias parameters."""
            local_expert_number = int(param_name.split(f".{param_type}")[-1])
            global_expert_number = num_experts_per_rank * ep_group.rank() + local_expert_number
            return param_name.replace(
                f".{param_type}{local_expert_number}",
                f".{param_type}{global_expert_number}",
            )

        # Handle weight and bias parameters
        if ".weight" in param_name:
            param_name = _update_expert_number(param_name, "weight")
        elif ".bias" in param_name:
            param_name = _update_expert_number(param_name, "bias")
    return param_name


class MegatronModelBridge(Generic[HFPreTrained, ModelProviderTarget, MegatronModel]):
    """
    High-level orchestrator for HuggingFace ↔ Megatron model conversions.

    This abstract base class provides the framework for converting models between
    HuggingFace and Megatron formats. It acts as an orchestrator that coordinates
    the conversion process without directly handling the complex details of
    tensor parallelism or weight transformations.

    The bridge pattern separates concerns:
    - MegatronModelBridge: Orchestrates the overall conversion process
    - MegatronMappingRegistry: Manages parameter name mappings
    - MegatronParamMapping: Handles actual weight transformations and distribution

    Key responsibilities:
    1. Build conversion tasks that map each parameter to its appropriate bridge
    2. Execute tasks with proper error handling and progress tracking
    3. Provide utilities for configuration translation
    4. Handle virtual pipeline parallelism (VP) complexities

    To implement a bridge for a new model architecture:

    1. Create a subclass decorated with @MegatronModelBridge.register_bridge:

        .. code-block:: python

            @MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
            class MegatronCausalLlamaBridge(MegatronModelBridge):
                pass

    2. Implement provider_bridge to create Megatron configurations:

        .. code-block:: python

            def provider_bridge(self, hf_pretrained) -> LlamaModelProvider:
                return LlamaModelProvider(
                    num_layers=hf_pretrained.config.num_hidden_layers,
                    hidden_size=hf_pretrained.config.hidden_size,
                    ...
                )

    3. Implement mapping_registry to define weight mappings:

        .. code-block:: python

            def mapping_registry(self) -> MegatronMappingRegistry:
                return MegatronMappingRegistry(
                    AutoMapping(
                        megatron_param="embedding.word_embeddings.weight",
                        hf_param="model.embed_tokens.weight"
                    ),
                    ...
                )

    Example:
        .. code-block:: python

            # The bridge is typically not instantiated directly
            # Instead, use AutoBridge or AutoBridge which handle this
            bridge = AutoBridge.from_hf_pretrained("meta-llama/Meta-Llama-3-8B")
            provider = bridge.to_megatron_provider()

    Note:
        This class uses generic type parameters to ensure type safety:
        - HFPreTrained: The HuggingFace model type
        - ModelProviderTarget: The Megatron model provider type
        - MegatronModel: The Megatron model type
    """

    @abc.abstractmethod
    def provider_bridge(self, hf_pretrained: HFPreTrained) -> ModelProviderTarget:
        """Create a Megatron model provider from HuggingFace configuration.

        This abstract method must be implemented by subclasses to translate
        HuggingFace model configurations into Megatron model provider instances.
        The provider contains all necessary configuration for creating Megatron models.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or configuration
                containing the source model's architecture details.

        Returns:
            ModelProviderTarget: A configured model provider instance (e.g.,
                GPTModelProvider, LlamaModelProvider) ready to create Megatron
                models.

        Example:
            .. code-block:: python

                def provider_bridge(self, hf_pretrained):
                    return LlamaModelProvider(
                        num_layers=hf_pretrained.config.num_hidden_layers,
                        hidden_size=hf_pretrained.config.hidden_size,
                        num_attention_heads=hf_pretrained.config.num_attention_heads,
                        ffn_hidden_size=hf_pretrained.config.intermediate_size,
                        # ... other configuration mappings
                    )
        """
        raise NotImplementedError("Subclass must implement bridge method")

    @abc.abstractmethod
    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define weight mappings between HuggingFace and Megatron formats.

        This abstract method must be implemented by subclasses to specify how
        parameters map between the two formats. The returned MegatronMappingRegistry
        contains all param mappings needed for the model architecture.

        Returns:
            MegatronMappingRegistry: MegatronMappingRegistry containing all weight
                mapping definitions.

        Example:
            .. code-block:: python

                def mapping_registry(self):
                    return MegatronMappingRegistry(
                        AutoMapping(
                            megatron_param="embedding.word_embeddings.weight",
                            hf_param="model.embed_tokens.weight"
                        ),
                        QKVMapping(
                            megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                            q="model.layers.*.self_attn.q_proj.weight",
                            k="model.layers.*.self_attn.k_proj.weight",
                            v="model.layers.*.self_attn.v_proj.weight"
                        ),
                        # ... more param mappings
                    )
        """
        raise NotImplementedError("Subclass must implement mapping_registry method")

    def _get_adapter_wrap_module(
        self, local_base_prefix: str, megatron_model: Union[MegatronModel, List[MegatronModel]], vp_stage: int
    ) -> tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        """Locate the adapter wrapper and its underlying module.

        Args:
            local_base_prefix: Module prefix without the ``.adapter`` suffix (e.g. ``decoder.layers.0.mlp.linear_fc1``).
            megatron_model: Single model or list of models indexed by virtual pipeline stage.
            vp_stage: Virtual pipeline stage corresponding to the provided prefix.

        Returns:
            A tuple ``(adapter, to_wrap)`` where ``adapter`` is the LoRA wrapper (or ``None`` if absent)
            and ``to_wrap`` is the base linear module being wrapped.
        """
        lora_module, _ = get_module_and_param_from_name(megatron_model, local_base_prefix, vp_stage)
        adapter = getattr(lora_module, "adapter", None)
        if adapter is None:
            # For CanonicalLoRA module
            lora_module, _ = get_module_and_param_from_name(megatron_model, local_base_prefix + ".to_wrap", vp_stage)
        return getattr(lora_module, "adapter", None), getattr(lora_module, "to_wrap", None)

    def _megatron_global_param_names_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[str]:
        """Get all parameter names across all pipeline parallel ranks."""
        # Cache the result after first call
        if hasattr(self, "_cached_param_names"):
            return self._cached_param_names

        # Compute the result
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        model_config = unwrap_model(megatron_model)[0].config
        global_param_names = []

        # Ensure megatron_model is a list for consistent handling
        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]

        for vp_stage, model in enumerate(models_list):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_param_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    models_list, model_config, local_param_name, vp_stage
                )
                if self._is_adapter_param_name(global_param_name):
                    continue
                global_param_names.append(global_param_name)

        gathered_global_param_names = [None] * pp_group.size()
        torch.distributed.all_gather_object(gathered_global_param_names, global_param_names, group=pp_group)

        # flatten the list, sort it and remove duplicates
        # the order matters here, casually re-order will cause a hang.
        # e.g. decoder.layers.0.mlp.experts.linear_fc1.weight100
        flattened_names = list(set(sum(gathered_global_param_names, [])))

        # the order cannot be changed, this sync for all ranks for conversion
        # change this might cause a hang
        gathered_global_param_names = sorted(flattened_names, key=extract_sort_key)

        # Cache the result
        self._cached_param_names = gathered_global_param_names

        return self._cached_param_names

    def _megatron_global_adapters_info_all_pp_ranks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[tuple[str, str, bool, bool, int, int, int, int]]:
        """Get all adapters' information tuple:
         (global_base_name, local_base_prefix, input_is_parallel, base_linear_is_parallel, alpha, dim, pp_rank, vp_stage)
        across all pipeline parallel ranks."""
        # Cache the result after first call
        if hasattr(self, "_cached_param_objects_adapter"):
            return self._cached_param_objects_adapter

        # Compute the result
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        model_config = unwrap_model(megatron_model)[0].config
        global_param_objects = []

        # Ensure megatron_model is a list for consistent handling
        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]

        for vp_stage, model in enumerate(models_list):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_param_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_param_name:
                    continue
                local_param_name = self._unwrap_name(local_param_name)
                global_param_name = _megatron_local_name_to_global(
                    models_list, model_config, local_param_name, vp_stage
                )
                is_adapter_param = self._is_adapter_param_name(global_param_name)
                # only collect linear_in.weight for deduplication
                if not is_adapter_param or not global_param_name.endswith(".linear_in.weight"):
                    continue
                local_base_prefix = local_param_name.partition(".adapter.")[0]
                global_base_name = global_param_name[: -len(".linear_in.weight")]
                adapter, to_wrap = self._get_adapter_wrap_module(local_base_prefix, models_list, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter_name = local_param_name.removeprefix(local_base_prefix + ".adapter.").split(".")[0]
                    adapter = adapter[adapter_name]
                input_is_parallel, _, _, _, _, base_linear_is_parallel = get_adapter_attributes_from_linear(to_wrap)
                global_param_objects.append(
                    (
                        global_base_name,
                        local_base_prefix,
                        input_is_parallel,
                        base_linear_is_parallel,
                        adapter.alpha,
                        adapter.dim,
                        pp_rank,
                        vp_stage,
                    )
                )

        gathered_global_param_objects = [None] * pp_group.size()
        torch.distributed.all_gather_object(gathered_global_param_objects, global_param_objects, group=pp_group)

        # flatten the list, sort it and remove duplicates
        # the order matters here, casually re-order will cause a hang.
        flattened_names = list(set(sum(gathered_global_param_objects, [])))

        # the order cannot be changed, this sync for all ranks for conversion
        # change this might cause a hang
        gathered_global_param_objects = sorted(flattened_names, key=lambda x: extract_sort_key(x[0]))

        self._cached_param_objects_adapter = gathered_global_param_objects

        return gathered_global_param_objects

    def _construct_adapters_names(self, prefix: str, adapter_key: Optional[str]) -> tuple[str, str]:
        """Build linear_in/linear_out parameter names for an adapter.

        Args:
            prefix: Base module prefix without any adapter suffix (global or local, depending on caller).
            adapter_key: Optional adapter identifier used by CanonicalLoRA (e.g. ``adapter_q``). ``None`` for
                standard single-adapter LoRA modules.

        Returns:
            Tuple ``(linear_in_name, linear_out_name)`` containing the parameter names for the adapter's
            input and output projection weights.
        """
        linear_in_name, linear_out_name = prefix + ".adapter", prefix + ".adapter"
        if adapter_key is not None:
            linear_in_name += f".{adapter_key}"
            linear_out_name += f".{adapter_key}"
        linear_in_name += ".linear_in.weight"
        linear_out_name += ".linear_out.weight"
        return linear_in_name, linear_out_name

    def build_adapter_conversion_tasks(
        self, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> Dict[str, List[AdapterWeightConversionTask]]:
        """Construct adapter merge tasks keyed by their base parameter.

        The returned dict is keyed by the *global* LoRA-wrapped parameter name
        (e.g., ``decoder.layers.0.mlp.linear_fc1.to_wrap.weight``). Each value
        contains the adapter tasks (canonical or regular) that should be
        merged into that base weight.
        """

        models_list = megatron_model if isinstance(megatron_model, list) else [megatron_model]

        adapters_info = self._megatron_global_adapters_info_all_pp_ranks(models_list)
        tasks_by_base: Dict[str, List[AdapterWeightConversionTask]] = defaultdict(list)

        for (
            global_base_name,
            local_base_prefix,
            input_is_parallel,
            base_linear_is_parallel,
            alpha,
            dim,
            pp_rank,
            vp_stage,
        ) in adapters_info:
            # global_base_name example: decoder.layers.0.mlp.linear_fc1.adapter.adapter_q
            global_base_prefix, _, adapter_suffix = global_base_name.partition(".adapter")

            adapter_key = None
            if adapter_suffix:
                key_token = adapter_suffix.split(".")[-1]
                if key_token.startswith("adapter_"):
                    adapter_key = key_token

            global_linear_in_name, global_linear_out_name = self._construct_adapters_names(
                global_base_prefix, adapter_key
            )
            # In case the adapter doesn't exist locally, we use the global names
            local_linear_in_name, local_linear_out_name = global_linear_in_name, global_linear_out_name
            linear_in_module, linear_in_weight = None, None
            linear_out_module, linear_out_weight = None, None
            if parallel_state.get_pipeline_model_parallel_rank() == pp_rank:
                adapter, _ = self._get_adapter_wrap_module(local_base_prefix, models_list, vp_stage)
                if isinstance(adapter, ModuleDict):
                    adapter = adapter[adapter_key]
                linear_in_module, linear_in_weight = adapter.linear_in, adapter.linear_in.weight
                linear_out_module, linear_out_weight = adapter.linear_out, adapter.linear_out.weight
                local_linear_in_name, local_linear_out_name = self._construct_adapters_names(
                    local_base_prefix, adapter_key
                )

            # Pick mapping strategies based on base layer parallelism
            if base_linear_is_parallel:
                linear_in_mapping_cls = RowParallelMapping if input_is_parallel else ColumnParallelMapping
                linear_out_mapping_cls = ColumnParallelMapping
            else:
                linear_in_mapping_cls = ReplicatedMapping
                linear_out_mapping_cls = ReplicatedMapping

            linear_in_task = WeightConversionTask(
                param_name=local_linear_in_name,
                global_param_name=global_linear_in_name,
                # TODO: use some actual HF param name mapping
                mapping=linear_in_mapping_cls(local_linear_in_name, local_linear_out_name),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_in_module,
                param_weight=linear_in_weight,
            )

            linear_out_task = WeightConversionTask(
                param_name=local_linear_out_name,
                global_param_name=global_linear_out_name,
                # TODO: use some actual HF param name mapping
                mapping=linear_out_mapping_cls(local_linear_out_name, local_linear_out_name),
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                megatron_module=linear_out_module,
                param_weight=linear_out_weight,
            )

            tasks_by_base[global_base_prefix].append(
                AdapterWeightConversionTask(
                    global_base_prefix=global_base_prefix,
                    adapter_key=adapter_key,
                    alpha=alpha,
                    dim=dim,
                    linear_in_task=linear_in_task,
                    linear_out_task=linear_out_task,
                )
            )

        return tasks_by_base

    def materialize_adapter_weights(self, adapter_tasks: List[AdapterWeightConversionTask]) -> List[AdapterWeight]:
        """Run adapter merge tasks to gather full adapter weights."""

        materialized: List[AdapterWeight] = []
        for adapter_task in adapter_tasks:
            mapping = adapter_task.linear_in_task.mapping
            linear_in_dict = mapping.megatron_to_hf(
                adapter_task.linear_in_task.param_weight, adapter_task.linear_in_task.megatron_module
            )
            linear_in_tensor = next(iter(linear_in_dict.values()))

            mapping = adapter_task.linear_out_task.mapping
            linear_out_dict = mapping.megatron_to_hf(
                adapter_task.linear_out_task.param_weight, adapter_task.linear_out_task.megatron_module
            )
            linear_out_tensor = next(iter(linear_out_dict.values()))

            materialized.append(
                AdapterWeight(
                    global_base_prefix=adapter_task.global_base_prefix,
                    adapter_key=adapter_task.adapter_key,
                    alpha=adapter_task.alpha,
                    dim=adapter_task.dim,
                    linear_in_weight=MegatronWeightTuple(
                        adapter_task.linear_in_task.param_name,
                        linear_in_tensor,
                        adapter_task.linear_in_task.vp_stage,
                    ),
                    linear_out_weight=MegatronWeightTuple(
                        adapter_task.linear_out_task.param_name,
                        linear_out_tensor,
                        adapter_task.linear_out_task.vp_stage,
                    ),
                )
            )

        return materialized

    def _with_progress_tracking(self, tasks, description: str, show_progress: bool = True):
        """Helper method to wrap an iterable with progress tracking.

        Args:
            tasks: Iterable of tasks to process
            description: Description for the progress bar
            show_progress: Whether to show progress (defaults to True)

        Yields:
            Items from the tasks iterable while updating progress
        """
        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        bridge_name = self.__class__.__name__

        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                TextColumn("({task.completed}/{task.total})"),
                TextColumn("{task.fields[bridge]}"),
                disable=not is_main_rank,
            ) as progress:
                task_id = progress.add_task(description, total=len(tasks), bridge=bridge_name)

                for task in tasks:
                    yield task
                    progress.update(task_id, advance=1)
        else:
            # not using disable above because we notice it will dump some empty progress bar,
            # even when disable is set to True
            for task in tasks:
                yield task

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Load weights from HuggingFace state dict.
        This function can be overridden by subclasses to preprocess the HF weights before conversion, such as renaming
        certain parameters to avoid mapping conflicts, or dequantize the weights.

        Note that loading is done lazily before this function is called, so the weights are actually loaded in
        this function when hf_state_dict.__getitem__ is called.

        Args:
            hf_param: The parameter name or dictionary of parameter names to load.
            hf_state_dict: The HuggingFace state dictionary.

        Returns:
            The loaded weights.
        """
        if isinstance(hf_param, str):
            hf_weights = hf_state_dict[hf_param]
        else:
            hf_weights = {k: hf_state_dict[v] for k, v in hf_param.items()}
        return hf_weights

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: Dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Modify the converted weights after conversion. By default, no modification is done.
        This function can be overridden by subclasses to postprocess the converted weights, such as merging the
        weights of multiple experts or quantizing the weights.

        Args:
            task: The WeightConversionTask object.
            converted_weights_dict: The converted weights dictionary.
            hf_state_dict: The HuggingFace state dict accessor for expected-key checks.

        Returns:
            The modified weights dictionary.
        """
        return converted_weights_dict

    def load_weights_hf_to_megatron(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        allowed_mismatched_params: Optional[List[str]] = None,
    ) -> List[MegatronModel]:
        """Load HuggingFace weights into Megatron models.

        This method orchestrates the complete weight loading process from HuggingFace
        format to Megatron's distributed format. It builds a conversion task and
        executes it with proper progress tracking and error handling.

        The actual weight transformations and distribution are delegated to the
        appropriate MegatronParamMapping instances based on the state mappings.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing the
                weights to load.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).
            allowed_mismatched_params (Optional[List[str]]): List of parameter names or patterns
                to allow mismatch (skip instead of raise error).

        Returns:
            List[MegatronModel]: The input megatron_model as a list with loaded weights.

        Process:
        1. Build a task mapping each Megatron parameter to its source
        2. For each parameter in the task:
            - Fetch source weights from HuggingFace state
            - Apply format transformation via the param mapping
            - Distribute to appropriate TP/PP ranks
            - Copy into the Megatron parameter

        Example:
            .. code-block:: python

                hf_model = PreTrainedCausalLM.from_pretrained("gpt2")
                megatron_model = create_megatron_model()  # Single model or list
                bridge.load_weights_hf_to_megatron(hf_model, megatron_model)

        Note:
            Progress is shown only on rank 0 to avoid cluttered output in
            distributed environments.

        Raises:
            ValueError: If hf_pretrained doesn't have state attribute or if weight shapes don't match.
            AttributeError: If required HF weights are missing.
        """
        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # [ModelOpt]: Hide extra parameters registered in Distillation mode
        with contextlib.ExitStack() as stack:
            if hasattr(megatron_model[0], "hide_teacher_model"):
                stack.enter_context(megatron_model[0].hide_teacher_model())
            if hasattr(megatron_model[0], "hide_loss_modules"):
                stack.enter_context(megatron_model[0].hide_loss_modules())

            hf_to_megatron_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)
        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        description = f"Loading from {hf_pretrained.model_name_or_path}"
        for task in self._with_progress_tracking(hf_to_megatron_tasks, description):
            # None means megatron module not on current rank, skip if this task is not going to happen
            if task.megatron_module is None:
                continue
            # 1) Fetch source tensor(s) from HF state dict
            hf_weights = self.maybe_modify_loaded_hf_weight(task.mapping.hf_param, hf_state_dict)

            # 2) Delegate conversion & distribution to the bridge
            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)

            # 3) Copy into Megatron param if this rank received a shard
            if converted_weights is not None:
                # Assert that param_weight is not None for HF->Megatron tasks
                assert task.param_weight is not None, "param_weight is required for HF->Megatron conversion"

                # Check shape compatibility before copying
                if converted_weights.shape != task.param_weight.shape:
                    # Check whitelist
                    is_whitelisted = False
                    if allowed_mismatched_params:
                        for pattern in allowed_mismatched_params:
                            if fnmatch.fnmatch(task.mapping.megatron_param, pattern) or fnmatch.fnmatch(
                                task.param_name, pattern
                            ):
                                is_whitelisted = True
                                break

                    if is_whitelisted:
                        print_rank_0(
                            f"WARNING: Shape mismatch for megatron param {task.mapping.megatron_param} allowed by whitelist. Skipping."
                        )
                        continue

                    raise ValueError(
                        f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                        f"  Expected shape: {task.param_weight.shape}\n"
                        f"  Got shape: {converted_weights.shape}\n"
                        f"  Bridge type: {type(task.mapping).__name__}\n"
                        f"  HF mapping: {task.mapping.hf_param}"
                    )
                task.param_weight.data.copy_(converted_weights)

        self._broadcast_shared_embeddings(megatron_model)
        return megatron_model

    def stream_weights_hf_to_megatron(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[MegatronWeightTuple]:
        """Generator variant of load_weights_hf_to_megatron for streaming weight conversion.

        This method provides a memory-efficient way to convert weights by yielding
        them one at a time instead of loading all at once. Useful for processing
        very large models or when implementing custom weight handling logic.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing
                the weights.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances to extract configuration from.
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.

        Yields:
            MegatronWeightTuple: Named tuples containing:
                - vp_stage: Index of the model in megatron_model list
                - param_name: Name of the parameter
                - weight: Transformed weight tensor for this rank

        Example:
            .. code-block:: python

                # Process weights one by one
                for weight_tuple in bridge.stream_weights_hf_to_megatron(hf_model, megatron_model):
                    print(f"Processing {weight_tuple.param_name}: {weight_tuple.weight.shape}")
                    # Custom processing logic here

                # Or use pre-built conversion tasks
                tasks = bridge.build_conversion_tasks(hf_model, megatron_model)
                for weight_tuple in bridge.stream_weights_hf_to_megatron(hf_model, megatron_model, tasks):
                    print(f"Processing {weight_tuple.param_name}: {weight_tuple.weight.shape}")

        Note:
            Only yields weights that belong to the current rank after TP/PP distribution.

        Raises:
            ValueError: If input parameters are invalid.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        for task in conversion_tasks:
            # None means megatron module not on current rank, skip if this task is not going to happen
            if task.megatron_module is None:
                continue
            hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state
            if isinstance(task.mapping.hf_param, str):
                hf_weights = hf_state_dict[task.mapping.hf_param]
            else:
                hf_weights = {k: hf_state_dict[v] for k, v in task.mapping.hf_param.items()}

            converted_weights = task.mapping.hf_to_megatron(hf_weights, task.megatron_module)
            if converted_weights is not None:
                # Assert that vp_stage is not None for HF->Megatron tasks
                yield MegatronWeightTuple(task.param_name, converted_weights, task.vp_stage)

    def stream_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        """Export Megatron weights to HuggingFace format.

        This method orchestrates the conversion of weights from Megatron's distributed
        format back to HuggingFace format. It handles gathering from tensor parallel
        ranks, broadcasting across pipeline parallel ranks, and format conversions.
        All ranks receive the full tensors.

        The export order is determined automatically:
        - First tries safetensors order (if key_to_filename_map is available)
        - Falls back to HuggingFace state dict order

        Args:
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).
            hf_pretrained (HFPreTrained): HuggingFace model/config for metadata
                and mapping info.
            cpu (bool, optional): Whether to move tensors to CPU before yielding.
                Defaults to True.
            show_progress (bool, optional): Display progress bar during export.
                Defaults to True.
            conversion_tasks (Optional[List[WeightConversionTask]]): Pre-built conversion tasks.
                If not provided, tasks will be built automatically from the models.

        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor) in HF format.

        Example:
            .. code-block:: python

                # Export weights
                for name, weight in bridge.stream_weights_megatron_to_hf(megatron_model, hf_config):
                    print(f"Exported {name}: {weight.shape}")

                # Or use pre-built conversion tasks
                tasks = bridge.build_conversion_tasks(hf_config, megatron_model)
                for name, weight in bridge.stream_weights_megatron_to_hf(
                    megatron_model, hf_config, conversion_tasks=tasks
                ):
                    print(f"Exported {name}: {weight.shape}")

        Raises:
            ValueError: If input parameters are invalid.

        Note:
            All ranks yield the full tensors after gathering from distributed format.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        # Use provided conversion tasks or build them
        if conversion_tasks is None:
            conversion_tasks = self.build_conversion_tasks(hf_pretrained, megatron_model)

        # Collect adapter conversion tasks
        adapter_tasks_by_base = self.build_adapter_conversion_tasks(megatron_model)

        megatron_to_hf_tasks = conversion_tasks
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config, unwrapped_model)

        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        for task in self._with_progress_tracking(megatron_to_hf_tasks, "Converting to HuggingFace", show_progress):
            converted_weights_dict = task.mapping.megatron_to_hf(task.param_weight, task.megatron_module)
            converted_weights_dict = self.maybe_modify_converted_hf_weight(
                task,
                converted_weights_dict,
                hf_state_dict,
            )  # dict will be none except for one expert;
            # All ranks get the full tensor

            adapter_tasks = None
            if "to_wrap.weight" in task.global_param_name:
                task_global_base_prefix, _, _ = task.global_param_name.partition(".to_wrap.weight")
                adapter_tasks = adapter_tasks_by_base.get(task_global_base_prefix)
            if adapter_tasks:
                adapter_weights = self.materialize_adapter_weights(adapter_tasks)
                # Merge LoRA adapter weights back into the base tensor for HF export
                converted_weights_dict = self._merge_lora_adapter_weights(
                    megatron_model,
                    converted_weights_dict,
                    adapter_weights,
                )

            for hf_name, tensor in converted_weights_dict.items():
                final_tensor = tensor.cpu() if cpu else tensor

                # Handle tied embeddings case
                # TODO(yuya): fix this hard coded naming
                if embeddings_are_tied and hf_name == "model.embed_tokens.weight":
                    # Yield the embedding weight
                    yield HFWeightTuple(hf_name, final_tensor)

                    # Also yield as lm_head.weight if it's expected
                    if hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source"):
                        expected_keys = hf_pretrained.state.source.get_all_keys()
                        if "lm_head.weight" in expected_keys:
                            yield HFWeightTuple("lm_head.weight", final_tensor.clone().detach())
                elif embeddings_are_tied and hf_name == "lm_head.weight":
                    # This should not happen when embeddings are tied - assert error
                    raise ValueError(
                        "Encountered lm_head.weight when embeddings are tied. This indicates a mapping error."
                    )
                else:
                    # Regular case - yield the tensor normally
                    yield HFWeightTuple(hf_name, final_tensor)

    def _merge_lora_adapter_weights(
        self,
        megatron_model: List[MegatronModel],
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge LoRA adapter weights back into the base tensor for HF export."""

        # CanonicalLoRA case when adapter_keys are provided via adapter_weights
        if len(adapter_weights) > 1 and all(
            w.adapter_key in ADAPTER_NAME_MAP.values() for w in adapter_weights if w.adapter_key
        ):
            return self._merge_canonical_adapter_from_weights(converted_weights_dict, adapter_weights)

        assert len(adapter_weights) == 1, "Expected a single adapter weight for standard LoRA merging"

        adapter_weight = adapter_weights[0]
        alpha, dim = adapter_weight.alpha, adapter_weight.dim
        linear_in_weight, linear_out_weight = (
            adapter_weight.linear_in_weight.weight,
            adapter_weight.linear_out_weight.weight,
        )

        # Check if this is a fused layer that gets split into multiple projections
        # For fused FC1: splits into gate_proj and up_proj (2 parts)
        # For fused QKV: splits into q_proj, k_proj, v_proj (3 parts, interleaved)
        base_weight_shape = next(iter(converted_weights_dict.values())).shape
        weight_names = converted_weights_dict.keys()
        is_fused_fc1 = (
            len(weight_names) % 2 == 0
            and all("gate_proj" in name or "up_proj" in name for name in weight_names)
            and linear_out_weight.shape[0] == 2 * base_weight_shape[0]
        )
        is_fused_qkv = len(weight_names) == 3 and all(
            "q_proj" in name or "k_proj" in name or "v_proj" in name for name in weight_names
        )

        # For QKV, split using the same interleaving logic as the base weight
        if is_fused_qkv:
            # Use the same interleaving pattern as split_qkv_weights
            q_out, k_out, v_out = split_qkv_weights(megatron_model[0].config, linear_out_weight)
            qkv_linear_out_weights = {
                "q_proj": q_out,
                "k_proj": k_out,
                "v_proj": v_out,
            }
        else:
            qkv_linear_out_weights = None

        # All ranks get the gathered weights, so we can merge on all ranks
        for hf_name, base_weight in list(converted_weights_dict.items()):
            # For fused layers, split linear_out_weight based on which projection we're merging
            current_linear_out_weight = linear_out_weight
            if is_fused_fc1:
                split_size = linear_out_weight.shape[0] // 2
                if "gate_proj" in hf_name:
                    # FC1: first half for gate_proj
                    current_linear_out_weight = linear_out_weight[:split_size, :]
                elif "up_proj" in hf_name:
                    # FC1: second half for up_proj
                    current_linear_out_weight = linear_out_weight[split_size:, :]
                else:
                    raise ValueError(f"Unknown weight name: {hf_name}")
            elif is_fused_qkv and qkv_linear_out_weights is not None:
                # QKV: Use properly split weights based on interleaving pattern
                if "q_proj" in hf_name:
                    current_linear_out_weight = qkv_linear_out_weights["q_proj"]
                elif "k_proj" in hf_name:
                    current_linear_out_weight = qkv_linear_out_weights["k_proj"]
                elif "v_proj" in hf_name:
                    current_linear_out_weight = qkv_linear_out_weights["v_proj"]
                else:
                    raise ValueError(f"Unknown weight name: {hf_name}")

            # Merge LoRA weights for each converted weight in the dict
            merged_weight = self._merge_single_adapter_weight(
                base_weight, alpha, dim, linear_in_weight, current_linear_out_weight
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict

    def _merge_single_adapter_weight(
        self,
        base_weight: torch.Tensor,
        alpha: int,
        dim: int,
        linear_in_weight: torch.Tensor,
        linear_out_weight: torch.Tensor,
    ) -> torch.Tensor:
        """Merge a single adapter's weights with base weight.

        Args:
            base_weight: Base weight tensor to merge with
            alpha: Alpha value for the adapter
            dim: Dimension of the adapter
            linear_in_weight: Gathered linear_in weight
            linear_out_weight: linear_out weight

        Returns:
            Merged weight tensor
        """
        merger = LoRAMerge()
        base_device = base_weight.device
        return merger.merge(
            base_weight,
            linear_out_weight.to(base_device),
            linear_in_weight.to(base_device),
            alpha,
            dim,
        )

    def _merge_canonical_adapter_from_weights(
        self,
        converted_weights_dict: Dict[str, torch.Tensor],
        adapter_weights: List[AdapterWeight],
    ) -> Dict[str, torch.Tensor]:
        """Merge CanonicalLoRA adapters using pre-materialized adapter weights."""
        adapter_lookup = {aw.adapter_key: aw for aw in adapter_weights}

        for hf_name, base_weight in converted_weights_dict.items():
            target_adapter = None
            for suffix, adapter_key in ADAPTER_NAME_MAP.items():
                if hf_name.endswith(suffix):
                    target_adapter = adapter_lookup.get(adapter_key)
                    break

            if target_adapter is None:
                raise ValueError(f"Adapter name mapping not found for {hf_name}")

            merged_weight = self._merge_single_adapter_weight(
                base_weight,
                target_adapter.alpha,
                target_adapter.dim,
                target_adapter.linear_in_weight.weight,
                target_adapter.linear_out_weight.weight,
            )
            converted_weights_dict[hf_name] = merged_weight

        return converted_weights_dict

    def dtype_from_hf(self, config, default=None):
        """Extract torch dtype from a HuggingFace config.

        This utility method handles the conversion of dtype specifications in
        HuggingFace configs to PyTorch dtype objects. Supports both direct
        torch.dtype objects and string representations.

        Args:
            config: HuggingFace configuration object with a torch_dtype attribute.
            default (Any, optional): Default value to return if torch_dtype is
                not str or torch.dtype. Defaults to None.

        Returns:
            torch.dtype: The corresponding PyTorch dtype.

        Raises:
            AssertionError: If config doesn't have torch_dtype attribute.
            ValueError: If torch_dtype is neither a string nor torch.dtype.

        Example:
            .. code-block:: python

                dtype = bridge.dtype_from_hf(hf_config)
                print(dtype)  # torch.float16
        """
        assert hasattr(config, "torch_dtype"), "Expected config to have attr `torch_dtype`"
        torch_dtype = config.torch_dtype
        if isinstance(torch_dtype, torch.dtype):
            return torch_dtype
        elif isinstance(torch_dtype, str):
            return self.dtype_from_str(torch_dtype)
        elif default is not None:
            return default

        raise ValueError("torch_dtype is not of type str/torch.dtype")

    def dtype_from_str(self, dtype: str) -> torch.dtype:
        """Convert a string precision identifier to equivalent torch dtype.

        This utility method handles various string representations of PyTorch
        data types, including common abbreviations and mixed precision formats.

        Args:
            dtype (str): String representation of dtype (e.g., "float16", "fp16",
                "bf16-mixed").

        Returns:
            torch.dtype: Corresponding PyTorch dtype (defaults to float32 if unknown).

        Supported formats:
            - float16/fp16/16/16-mixed → torch.float16
            - bfloat16/bf16-mixed → torch.bfloat16
            - Others → torch.float32 (default)

        Example:
            .. code-block:: python

                dtype = bridge.dtype_from_str("fp16")
                print(dtype)  # torch.float16

                dtype = bridge.dtype_from_str("bf16-mixed")
                print(dtype)  # torch.bfloat16
        """
        assert isinstance(dtype, str)
        if dtype in ["float16", "fp16", "16", "16-mixed"]:
            return torch.float16
        elif dtype in ["bfloat16", "bf16-mixed"]:
            return torch.bfloat16
        else:
            return torch.float32

    def make_vocab_size_divisible_by(self, vocab_size: int) -> int:
        """Calculate an appropriate divisor for vocabulary size padding.

        Megatron requires vocabulary sizes to be divisible by certain values for
        efficient tensor parallelism. This method finds the largest power of 2
        (up to 128) that evenly divides the vocabulary size.

        Args:
            vocab_size (int): Original vocabulary size from the model.

        Returns:
            int: Largest power of 2 (≤ 128) that divides vocab_size.

        Example:
            .. code-block:: python

                # For vocab_size=50257 (GPT-2)
                divisor = bridge.make_vocab_size_divisible_by(50257)
                print(divisor)  # 1 (50257 is prime)

                # For vocab_size=32000 (Llama)
                divisor = bridge.make_vocab_size_divisible_by(32000)
                print(divisor)  # 128

        Note:
            The returned value is used by Megatron to potentially pad the
            vocabulary to ensure efficient parallelization.
        """
        base = 128
        while vocab_size % base != 0:
            base //= 2
        return base

    def _get_provider_from_model(self, model: MegatronModule) -> ModelProviderTarget:
        """Extract provider/config from model."""
        model = unwrap_model(model)
        return model.config

    def _share_embeddings_and_output_weights(
        self, model_config: TransformerConfig, model: Optional[MegatronModule]
    ) -> bool:
        """Fallback-aware accessor for shared embedding setting."""
        fallback = getattr(model, "share_embeddings_and_output_weights", False) if model else False
        return getattr(model_config, "share_embeddings_and_output_weights", fallback)

    def _unwrap_name(self, name: str) -> str:
        """Unwrap name from DDP or other wrappers.

        Args:
            name: Parameter name that may have 'module.' prefixes

        Returns:
            Unwrapped parameter name with 'module.' prefixes removed

        Example:
            'module.module.decoder.weight' -> 'decoder.weight'
        """
        if not isinstance(name, str):
            raise ValueError(f"name must be a string, got {type(name)}")

        while name.startswith("module."):
            name = name[len("module.") :]
        return name

    def _broadcast_shared_embeddings(self, megatron_model: Union[MegatronModel, List[MegatronModel]]) -> None:
        """Broadcast shared embeddings and output weights across embedding group.

        When embeddings and output weights are shared and pipeline parallelism is enabled,
        this method ensures all ranks in the embedding group have the same weights by
        broadcasting from rank 0.

        Args:
            megatron_model: Megatron model instance or list of model instances.
        """
        unwrapped_model = unwrap_model(megatron_model)[0]
        # hack for vlm to work properly
        if hasattr(unwrapped_model, "language_model") and unwrapped_model.language_model is not None:
            unwrapped_model = unwrapped_model.language_model
        model_config = unwrapped_model.config
        share_embeddings = self._share_embeddings_and_output_weights(model_config, unwrapped_model)

        # TODO(yuya): Fix for VPP, the vp stage needs to be passed in for stage checks
        if (share_embeddings and model_config.pipeline_model_parallel_size > 1) and (
            parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage()
        ):
            # Broadcast embeddings and output weights from rank 0 to embedding group
            embd_group = parallel_state.get_embedding_group()
            embd_group_ranks = torch.distributed.get_process_group_ranks(embd_group)
            if embd_group is not None and torch.distributed.get_rank() in embd_group_ranks:
                # Get embeddings and output weights from rank 0
                if hasattr(unwrapped_model, "embedding") and hasattr(unwrapped_model.embedding, "word_embeddings"):
                    embd_weights = unwrapped_model.embedding.word_embeddings.weight.data
                else:
                    assert hasattr(unwrapped_model, "output_layer"), "Output layer not found"
                    embd_weights = torch.empty_like(unwrapped_model.output_layer.weight.data)
                torch.distributed.broadcast(embd_weights, src=embd_group_ranks[0], group=embd_group)
                if hasattr(unwrapped_model, "output_layer"):
                    unwrapped_model.output_layer.weight.data.copy_(embd_weights)

    def _get_lora_unwrapped_name(self, megatron_param: str) -> str:
        """Remove .to_wrap from LoRA parameter names."""
        return megatron_param.replace(".to_wrap.", ".")

    def _is_adapter_param_name(self, param_name: str) -> bool:
        """Return True if the parameter only belongs to a PEFT adapter."""

        return ".adapter." in param_name

    def build_conversion_tasks(
        self,
        hf_pretrained: HFPreTrained,
        megatron_model: List[MegatronModel],
    ) -> List[None | WeightConversionTask]:
        """Construct the conversion tasks between HF and megatron.

        The algorithm walks over every parameter of every destination model,
        asks the :class:`MegatronMappingRegistry` whether it has a mapping for that
        parameter, and – if the corresponding HF weights actually exist – yields
        an :class:`_HFLoadTask` describing exactly how that parameter will be
        populated.
        """

        # Ensure hf_pretrained has the required state structure
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()

        mapping_registry = self.mapping_registry()
        unwrapped_model = unwrap_model(megatron_model)[0]
        model_config = unwrapped_model.config
        embeddings_are_tied = self._share_embeddings_and_output_weights(model_config, unwrapped_model)
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        sorted_global_param_names_all_pp_ranks = self._megatron_global_param_names_all_pp_ranks(megatron_model)

        # Filter out output_layer related parameters if embeddings are tied
        if embeddings_are_tied:
            sorted_global_param_names_all_pp_ranks = [
                name for name in sorted_global_param_names_all_pp_ranks if "output_layer" not in name
            ]

        global_names_index_dict = {name: idx for idx, name in enumerate(sorted_global_param_names_all_pp_ranks)}

        tasks = [None] * len(sorted_global_param_names_all_pp_ranks)
        for vp_stage, model in enumerate(megatron_model):
            # persistent buffers are part of the model's state_dict, but not the named_parameters, so we must include them here separately
            for local_name, _ in itertools.chain(model.named_parameters(), persistent_buffers(model)):
                if "_extra_state" in local_name or self._is_adapter_param_name(local_name):
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _megatron_local_name_to_global(megatron_model, model_config, local_name, vp_stage)
                # if name removed due to some reason, continue. e.g. embeddings_are_tied
                if global_name not in global_names_index_dict:
                    print_rank_0(f"WARNING: {global_name} not in global_names_index_dict")
                    continue
                global_name_idx = global_names_index_dict[global_name]
                mapping = mapping_registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(global_name))

                if not mapping:
                    logger.warning(f"WARNING: No mapping found for megatron_param: {global_name}")
                    continue

                # ensure hf weights exist
                if not mapping.allow_hf_name_mismatch:
                    if isinstance(mapping.hf_param, str):
                        if mapping.hf_param not in hf_keys:
                            logger.warning(f"WARNING: Can't find {mapping.hf_param} in hf_keys")
                            continue
                    else:
                        missing_params = [
                            hf_param for hf_param in mapping.hf_param.values() if hf_param not in hf_keys
                        ]
                        if missing_params:
                            logger.warning(
                                f"WARNING: Can't find the following HF parameters in hf_keys: {missing_params}"
                            )
                            continue

                local_module, local_weights = get_module_and_param_from_name(megatron_model, local_name, vp_stage)
                if local_module is not None and not hasattr(local_module, "config"):
                    # If module is not a MegatronModule (e.g. torch.nn.Conv1d or a module list) we need
                    # to get the config from the model
                    setattr(local_module, "config", model_config)

                tasks[global_name_idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    global_param_name=global_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

        # Fill the remaining ones for pp communications
        for idx, global_name in enumerate(sorted_global_param_names_all_pp_ranks):
            if tasks[idx] is None:
                mapping = mapping_registry.megatron_to_hf_lookup(self._get_lora_unwrapped_name(global_name))
                # Skip tasks with no mapping found
                if mapping is None:
                    continue
                # This is an exception here we pass in global name
                # we are not using global_name to extract module and weights
                # only use it for param mapping auto dispatch checks
                tasks[idx] = WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=None,
                    param_name=global_name,
                    global_param_name=global_name,
                    megatron_module=None,
                    param_weight=None,
                    mapping=mapping,
                )

        return tasks

    @classmethod
    def register_bridge(
        cls, *, source: Type[PreTrainedModel] | str, target: Type[MegatronModel]
    ) -> Callable[[_BridgeImplClass], _BridgeImplClass]:
        """Class decorator for registering bridge implementations.

        This decorator registers a MegatronModelBridge subclass with the dispatch
        system, enabling automatic routing of conversions based on the source
        HuggingFace model type and target Megatron model type.

        Args:
            source (Type[PreTrainedModel] | str): HuggingFace PreTrainedModel class
                (e.g., LlamaForCausalLM) or the class name as a string. Using a
                string allows registering bridges for architectures that are only
                available via auto_map.
            target (Type[MegatronModel]): Megatron model class (e.g., GPTModel).

        Returns:
            Callable[[_BridgeImplClass], _BridgeImplClass]: Decorator function
                that registers the bridge implementation.

        Example:
            .. code-block:: python

                @MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
                class MegatronCausalLlamaBridge(MegatronModelBridge):
                    def provider_bridge(self, hf_pretrained):
                        # Implementation
                        pass

                    def mapping_registry(self):
                        # Implementation
                        pass

            String-based registration is also supported:

            .. code-block:: python

                @MegatronModelBridge.register_bridge(source="DeepseekV3ForCausalLM", target=GPTModel)
                class MegatronDeepseekV3Bridge(MegatronModelBridge):
                    ...

        Note:
            The decorated class is registered with multiple dispatchers to handle
            different conversion scenarios. The registration is automatic when the
            class is defined.
        """

        return create_bridge_decorator(source=source, target=target)


def is_tensor_parallel(param) -> bool:
    """Check if a parameter is tensor parallel distributed."""
    return hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel


# Core dispatch functions
@dispatch
def get_model_bridge(hf_architecture) -> "MegatronModelBridge":
    """Get the appropriate model bridge for a given HuggingFace architecture."""
    ...


@dispatch
def stream_weights_megatron_to_hf(
    dispatch_instance: MegatronModel,
    megatron_model: Union[MegatronModel, List[MegatronModel]],
    hf_pretrained: HFPreTrained,
    cpu: bool = True,
    show_progress: bool = True,
    conversion_tasks: Optional[List[WeightConversionTask]] = None,
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format."""
    ...


def register_bridge_implementation(
    *,
    source: Type["PreTrainedModel"] | str,
    target: Type["MegatronModule"],
    bridge_class: Type["MegatronModelBridge"],
) -> None:
    """Register a bridge implementation with the dispatch system.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string.
            Using a string allows registering bridges for architectures that are
            available only via auto_map.
        target: Megatron model class (e.g., GPTModel)
        bridge_class: MegatronModelBridge implementation class
    """
    bridge_class_name = bridge_class.__name__

    @get_model_bridge.impl(source)
    def _get_model_bridge_impl(_) -> "MegatronModelBridge":
        bridge = bridge_class()
        return bridge

    @stream_weights_megatron_to_hf.impl((source, target))
    def _megatron_to_hf_registered_impl(
        _,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks: Optional[List[WeightConversionTask]] = None,
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()

        # allow bridge to access model config
        bridge.hf_config = hf_pretrained.config

        return bridge.stream_weights_megatron_to_hf(
            megatron_model, hf_pretrained, cpu=cpu, show_progress=show_progress, conversion_tasks=conversion_tasks
        )

    # Set meaningful names for debugging
    _get_model_bridge_impl.__name__ = f"_bridge_with_{bridge_class_name}"
    _megatron_to_hf_registered_impl.__name__ = f"_megatron_to_hf_with_{bridge_class_name}"


def create_bridge_decorator(
    *, source: Type["PreTrainedModel"] | str, target: Type["MegatronModule"]
) -> Callable[[Type["MegatronModelBridge"]], Type["MegatronModelBridge"]]:
    """Create a decorator for registering bridge implementations.

    Args:
        source: HuggingFace PreTrainedModel class or the class name as a string
            (useful for auto_map architectures)
        target: Megatron model class

    Returns:
        Decorator function that registers the bridge implementation
    """

    def decorator(bridge_class: Type["MegatronModelBridge"]) -> Type["MegatronModelBridge"]:
        register_bridge_implementation(source=source, target=target, bridge_class=bridge_class)
        return bridge_class

    return decorator
