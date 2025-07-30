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
import logging
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
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
from megatron.core import parallel_state as mpu
from megatron.core.transformer.module import MegatronModule
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn
from transformers.modeling_utils import PreTrainedModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
from megatron.bridge.models.conversion.utils import get_transformer_layer_offset
from megatron.bridge.models.decorators.dispatch import dispatch
from megatron.bridge.models.model_provider import ModelProviderProtocol
from megatron.bridge.utils.common_utils import unwrap_model


logger = logging.getLogger(__name__)

MappingT = TypeVar("MappingT", bound=MegatronParamMapping)
HFPreTrained = TypeVar("HFPreTrained")
ModelProviderTarget = TypeVar("ModelProviderTarget", bound=ModelProviderProtocol)
MegatronModel = TypeVar("MegatronModel", bound=MegatronModule)
_BridgeImplClass = TypeVar("_BridgeImplClass", bound="MegatronModelBridge")


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
        param_name (str): Fully-qualified, *unwrapped* parameter name (no ``module.`` prefixes).
        mapping (MappingT): Concrete :pyclass:`MegatronParamMapping` instance responsible
            for weight transformation and distribution.

        # Fields for HF->Megatron loading:
        vp_stage (Optional[int]): Virtual-pipeline stage index (required for loads).
        megatron_module (Optional[torch.nn.Module]): Reference to the Megatron model or
            sub-module that owns the parameter (required for loads).
        param_weight (Optional[torch.Tensor]): The actual parameter tensor that will
            receive the converted weight (required for loads).

        # Fields for Megatron->HF saving:
        pp_rank (Optional[int]): Pipeline-parallel rank that owns the parameter (required for saves).
    """

    param_name: str
    mapping: MappingT
    pp_rank: Optional[int] = None
    vp_stage: Optional[int] = None
    megatron_module: Optional[torch.nn.Module] = None
    param_weight: Optional[torch.Tensor] = None

    def hf_to_megatron(
        self,
        hf_weights: Union[torch.Tensor, Mapping[str, torch.Tensor]],
        megatron_module: torch.nn.Module,
    ) -> torch.Tensor:
        """Convert HuggingFace weights to Megatron format.

        This method delegates the actual conversion to the underlying mapping's
        hf_to_megatron method, which handles format transformation, tensor parallel
        distribution, and pipeline parallel communication.

        Args:
            hf_weights: HuggingFace weights to convert (single tensor or dict of tensors).
            megatron_module: Megatron module that owns the target parameter.

        Returns:
            Converted weight tensor ready for copying into Megatron parameter.

        Raises:
            ValueError: If required fields for HF->Megatron conversion are missing.
        """

        return self.mapping.hf_to_megatron(hf_weights, megatron_module, self.param_name)

    def megatron_to_hf(
        self,
        megatron_weights: Optional[torch.Tensor],
        megatron_module: Optional[torch.nn.Module],
    ) -> Dict[str, torch.Tensor]:
        """Convert Megatron weights to HuggingFace format.

        This method delegates the actual conversion to the underlying mapping's
        megatron_to_hf method, which handles tensor parallel gathering, pipeline
        parallel broadcasting, and format transformation.

        Args:
            megatron_weights: Megatron weight tensor to convert (may be None if not owned by this rank).
            megatron_module: Megatron module that owns the parameter (may be None).

        Returns:
            Dictionary mapping HuggingFace parameter names to converted tensors.

        Raises:
            ValueError: If required fields for Megatron->HF conversion are missing.
        """

        return self.mapping.megatron_to_hf(megatron_weights, megatron_module, self.param_name)


def _adjust_layer_number_to_global(name: str, layer_offset: int) -> str:
    """Adjust layer number from local to global numbering."""
    if "layers." not in name:
        return name

    local_layer_number = int(name.split("layers.")[1].split(".")[0])
    global_layer_number = local_layer_number + layer_offset
    name = name.replace(
        f"layers.{local_layer_number}.",
        f"layers.{global_layer_number}.",
    )

    return name


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
    1. Build conversion plans that map each parameter to its appropriate bridge
    2. Execute plans with proper error handling and progress tracking
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
            bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3-8B")
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

    def load_weights_hf_to_megatron(
        self, hf_pretrained: HFPreTrained, megatron_model: Union[MegatronModel, List[MegatronModel]]
    ) -> List[MegatronModel]:
        """Load HuggingFace weights into Megatron models.

        This method orchestrates the complete weight loading process from HuggingFace
        format to Megatron's distributed format. It builds a conversion plan and
        executes it with proper progress tracking and error handling.

        The actual weight transformations and distribution are delegated to the
        appropriate MegatronParamMapping instances based on the state mappings.

        Args:
            hf_pretrained (HFPreTrained): HuggingFace model or state source containing the
                weights to load.
            megatron_model (Union[MegatronModel, List[MegatronModel]]): Megatron model instance
                or list of model instances (one per virtual pipeline stage).

        Returns:
            List[MegatronModel]: The input megatron_model as a list with loaded weights.

        Process:
        1. Build a plan mapping each Megatron parameter to its source
        2. For each parameter in the plan:
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

        hf_to_megatron_plans = list(self._build_plan_hf_to_megatron(hf_pretrained, megatron_model))

        hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}

        is_main_rank = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        bridge_name = self.__class__.__name__

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TextColumn("({task.completed}/{task.total})"),
            TextColumn("{task.fields[bridge]}"),
            disable=not is_main_rank,
        ) as progress:
            task_id = progress.add_task(
                f"Loading from {hf_pretrained.model_name_or_path}", total=len(hf_to_megatron_plans), bridge=bridge_name
            )

            for task in hf_to_megatron_plans:
                # 1) Fetch source tensor(s) from HF state dict
                if isinstance(task.mapping.hf_param, str):
                    hf_weights = hf_state_dict[task.mapping.hf_param]
                else:
                    hf_weights = {k: hf_state_dict[v] for k, v in task.mapping.hf_param.items()}

                # 2) Delegate conversion & distribution to the bridge
                local_weights = task.hf_to_megatron(hf_weights, task.megatron_module)

                # 3) Copy into Megatron param if this rank received a shard
                if local_weights is not None:
                    # Assert that param_weight is not None for HF->Megatron tasks
                    assert task.param_weight is not None, "param_weight is required for HF->Megatron conversion"

                    # Check shape compatibility before copying
                    if local_weights.shape != task.param_weight.shape:
                        raise ValueError(
                            f"Shape mismatch for megatron param {task.mapping.megatron_param}:\n"
                            f"  Expected shape: {task.param_weight.shape}\n"
                            f"  Got shape: {local_weights.shape}\n"
                            f"  Bridge type: {type(task.mapping).__name__}\n"
                            f"  HF mapping: {task.mapping.hf_param}"
                        )
                    task.param_weight.data.copy_(local_weights)

                progress.update(task_id, advance=1)

        self._broadcast_shared_embeddings(megatron_model)
        return megatron_model

    def stream_weights_hf_to_megatron(
        self, hf_pretrained: HFPreTrained, megatron_model: Union[MegatronModel, List[MegatronModel]]
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

        Note:
            Only yields weights that belong to the current rank after TP/PP distribution.

        Raises:
            ValueError: If input parameters are invalid.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        for task in self._build_plan_hf_to_megatron(hf_pretrained, megatron_model):
            hf_state_dict: Mapping[str, torch.Tensor] = hf_pretrained.state
            if isinstance(task.mapping.hf_param, str):
                hf_weights = hf_state_dict[task.mapping.hf_param]
            else:
                hf_weights = {k: hf_state_dict[v] for k, v in task.mapping.hf_param.items()}

            local_weights = task.hf_to_megatron(hf_weights, task.megatron_module)
            if local_weights is not None:
                # Assert that vp_stage is not None for HF->Megatron tasks
                yield MegatronWeightTuple(task.param_name, local_weights, task.vp_stage)

    def stream_weights_megatron_to_hf(
        self,
        megatron_model: Union[MegatronModel, List[MegatronModel]],
        hf_pretrained: HFPreTrained,
        cpu: bool = True,
        show_progress: bool = True,
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

        Yields:
            HFWeightTuple: Named tuples of (param_name, weight_tensor) in HF format.

        Example:
            .. code-block:: python

                # Export weights
                for name, weight in bridge.stream_weights_megatron_to_hf(megatron_model, hf_config):
                    print(f"Exported {name}: {weight.shape}")

        Raises:
            ValueError: If input parameters are invalid.

        Note:
            All ranks yield the full tensors after gathering from distributed format.
        """

        if not isinstance(megatron_model, list):
            megatron_model = [megatron_model]

        megatron_to_hf_plans = list(self._build_plan_megatron_to_hf(megatron_model, hf_pretrained))

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
            task_id = progress.add_task(
                "Converting to HuggingFace", total=len(megatron_to_hf_plans), bridge=bridge_name
            )

            for task in megatron_to_hf_plans:
                # Owns param? fetch weight & module; otherwise None (bridge will broadcast)
                local_weights = None
                local_module = None
                if task.pp_rank == mpu.get_pipeline_model_parallel_rank():
                    local_module, local_weights = self._get_param_and_module_from_vp(
                        megatron_model, task.vp_stage, task.param_name
                    )

                kv_pairs = task.megatron_to_hf(local_weights, local_module)

                # All ranks get the full tensor
                for name, tensor in kv_pairs.items():
                    yield HFWeightTuple(name, tensor.cpu() if cpu else tensor)

                progress.update(task_id, advance=1)

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
        model_config = unwrapped_model.config
        if model_config.share_embeddings_and_output_weights and model_config.pipeline_model_parallel_size > 1:
            # Broadcast embeddings and output weights from rank 0 to embedding group
            embd_group = mpu.get_embedding_group()
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

    def _collect_all_params_info(self, models: List[MegatronModule]) -> List[Tuple[int, Optional[int], str]]:
        """Collect all parameter names across PP/VP stages.

        Args:
            models: List of Megatron model instances

        Returns:
            List of tuples (pp_rank, vp_stage, param_name) for all parameters
            across all pipeline parallel ranks.

        Note:
            This method uses all_gather to collect parameter information from
            all pipeline parallel ranks to build a complete view of the model.
        """

        pp_rank = mpu.get_pipeline_model_parallel_rank()

        # Collect parameter names from this pipeline rank's model parameters
        local_param_infos = []
        for vp_stage, model in enumerate(models):
            for local_param_name, _ in model.named_parameters():
                local_param_infos.append((pp_rank, vp_stage if len(models) > 1 else None, local_param_name))

        # All-gather across PP ranks
        gathered_param_infos = [None] * mpu.get_pipeline_model_parallel_world_size()
        torch.distributed.all_gather_object(
            gathered_param_infos, local_param_infos, group=mpu.get_pipeline_model_parallel_group()
        )

        # Flatten
        all_param_infos = sum(gathered_param_infos, [])
        return all_param_infos

    def _get_param_and_module_from_vp(
        self, models: List[MegatronModule], vp_stage: Optional[int], param_name: str
    ) -> Tuple[torch.nn.Module, torch.Tensor]:
        """
        Get parameter from specific VP stage, ensuring that parameter
        attributes are preserved.

        Args:
            models: List of Megatron model instances
            vp_stage: Virtual pipeline stage index (None for single stage)
            param_name: Dot-separated parameter name

        Returns:
            Tuple of (module, parameter) where module owns the parameter

        Raises:
            ValueError: If vp_stage is out of range or parameter doesn't exist
        """

        if vp_stage is None:
            model = models[0]
        else:
            if vp_stage >= len(models):
                raise ValueError(f"VP stage {vp_stage} out of range (max: {len(models) - 1})")
            model = models[vp_stage]

        param = unwrap_model(model)
        module = param
        splitted_name = param_name.split(".")

        try:
            for i, part in enumerate(splitted_name):
                param = getattr(param, part)
                if i < len(splitted_name) - 1:
                    module = getattr(module, part)
        except AttributeError as e:
            raise ValueError(f"Parameter '{param_name}' not found in model at VP stage {vp_stage}") from e

        return module, param

    def _build_plan_hf_to_megatron(
        self, hf_pretrained: HFPreTrained, megatron_model: List[MegatronModel]
    ) -> Iterable[WeightConversionTask]:
        """Construct the *HF ➜ Megatron* load plan.

        The algorithm walks over every parameter of every destination model,
        asks the :class:`MegatronMappingRegistry` whether it has a mapping for that
        parameter, and – if the corresponding HF weights actually exist – yields
        an :class:`_HFLoadTask` describing exactly how that parameter will be
        populated.
        """

        mapping_registry = self.mapping_registry()
        hf_state_dict = hf_pretrained.state if hasattr(hf_pretrained, "state") else {}
        model_config = unwrap_model(megatron_model)[0].config
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        for vp_stage, model in enumerate(megatron_model):
            layer_offset = get_transformer_layer_offset(model_config, pipeline_rank=pp_rank, vp_stage=vp_stage)
            for local_name, _ in model.named_parameters():
                if "_extra_state" in local_name:
                    continue

                local_name = self._unwrap_name(local_name)
                global_name = _adjust_layer_number_to_global(local_name, layer_offset)
                mapping = mapping_registry.megatron_to_hf_lookup(global_name)

                if not mapping:
                    logger.warning(f"WARNING: No megatron to hf mapping found for {global_name}")
                    continue

                # ensure hf weights exist
                if isinstance(mapping.hf_param, str):
                    if mapping.hf_param not in hf_state_dict:
                        logger.warning(f"WARNING: Can't find {mapping.hf_param} in hf_state_dict")
                        continue
                else:
                    missing_params = [
                        hf_param for hf_param in mapping.hf_param.values() if hf_param not in hf_state_dict
                    ]
                    if missing_params:
                        logger.warning(
                            f"WARNING: Can't find the following HF parameters in hf_state_dict: {missing_params}"
                        )
                        continue

                local_module, local_weights = self._get_param_and_module_from_vp(megatron_model, vp_stage, local_name)

                yield WeightConversionTask(
                    pp_rank=pp_rank,
                    vp_stage=vp_stage,
                    param_name=local_name,
                    megatron_module=local_module,
                    param_weight=local_weights,
                    mapping=mapping,
                )

    def _build_plan_megatron_to_hf(
        self,
        megatron_model: List[MegatronModel],
        hf_pretrained: HFPreTrained,
    ) -> Iterable[WeightConversionTask]:
        """Construct the *Megatron ➜ HF* save plan.

        Uses safetensors ordering if available (when key_to_filename_map exists),
        otherwise falls back to HuggingFace state dict ordering.

        Args:
            megatron_model (List[MegatronModel]): List of local Megatron
                *pipeline* replicas (length ≥ 1, length > 1 only when
                virtual-pipeline-parallelism (VP) is enabled).
            hf_pretrained (HFPreTrained): HF model whose *state* object provides
                ordering information.

        Returns:
            Iterable[WeightConversionTask]: The save plan.
        """

        # Ensure hf_pretrained has the required state structure
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            raise ValueError("hf_pretrained.state.source is required for weight ordering")

        # Try safetensors order first, fallback to hf order
        if hasattr(hf_pretrained.state.source, "key_to_filename_map"):
            # Use safetensors ordering (grouped by file, then by key)
            key_to_filename: Mapping[str, str] = hf_pretrained.state.source.key_to_filename_map
            filename_to_keys = defaultdict(list)
            for key, filename in key_to_filename.items():
                filename_to_keys[filename].append(key)

            hf_keys = (key for fname in sorted(filename_to_keys.keys()) for key in filename_to_keys[fname])
        else:
            # Fallback to hf order
            hf_keys: Iterable[str] = hf_pretrained.state.source.get_all_keys()

        model_config = unwrap_model(megatron_model)[0].config
        mapping_registry = self.mapping_registry()
        emitted = set()

        param_locations = defaultdict(list)
        for pp_rank, vp_stage, local_name in self._collect_all_params_info(megatron_model):
            layer_offset = get_transformer_layer_offset(model_config, pipeline_rank=pp_rank, vp_stage=vp_stage)
            local_name = self._unwrap_name(local_name)
            global_name = _adjust_layer_number_to_global(local_name, layer_offset)
            param_locations[global_name].append((pp_rank, vp_stage, local_name))

        for hf_key in hf_keys:
            mapping = mapping_registry.hf_to_megatron_lookup(hf_key)
            if not mapping:
                logger.warning(f"WARNING: No hf to megatron mapping found for {hf_key}")
                continue

            global_name = mapping.megatron_param if hasattr(mapping, "megatron_param") else None
            if not global_name or global_name in emitted:
                continue

            if global_name not in param_locations:
                # HF layer is not in current PP rank of Megatron model
                param_locations[global_name].append((None, None, global_name))

            emitted.add(global_name)
            pp_rank, vp_stage, local_name = sorted(param_locations[global_name])[0]
            yield WeightConversionTask(
                pp_rank=pp_rank,
                vp_stage=vp_stage,
                param_name=local_name,
                mapping=mapping,
            )

    @classmethod
    def register_bridge(
        cls, *, source: Type[PreTrainedModel], target: Type[MegatronModel]
    ) -> Callable[[_BridgeImplClass], _BridgeImplClass]:
        """Class decorator for registering bridge implementations.

        This decorator registers a MegatronModelBridge subclass with the dispatch
        system, enabling automatic routing of conversions based on the source
        HuggingFace model type and target Megatron model type.

        Args:
            source (Type[PreTrainedModel]): HuggingFace PreTrainedModel class
                (e.g., LlamaForCausalLM).
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
) -> Iterable[HFWeightTuple]:
    """Bridge Megatron model state to HuggingFace format."""
    ...


def register_bridge_implementation(
    *,
    source: Type["PreTrainedModel"],
    target: Type["MegatronModule"],
    bridge_class: Type["MegatronModelBridge"],
) -> None:
    """Register a bridge implementation with the dispatch system.

    Args:
        source: HuggingFace PreTrainedModel class (e.g., LlamaForCausalLM)
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
    ) -> Iterable[HFWeightTuple]:
        bridge = bridge_class()
        return bridge.stream_weights_megatron_to_hf(
            megatron_model, hf_pretrained, cpu=cpu, show_progress=show_progress
        )

    # Set meaningful names for debugging
    _get_model_bridge_impl.__name__ = f"_bridge_with_{bridge_class_name}"
    _megatron_to_hf_registered_impl.__name__ = f"_megatron_to_hf_with_{bridge_class_name}"


def create_bridge_decorator(
    *, source: Type["PreTrainedModel"], target: Type["MegatronModule"]
) -> Callable[[Type["MegatronModelBridge"]], Type["MegatronModelBridge"]]:
    """Create a decorator for registering bridge implementations.

    Args:
        source: HuggingFace PreTrainedModel class
        target: Megatron model class

    Returns:
        Decorator function that registers the bridge implementation
    """

    def decorator(bridge_class: Type["MegatronModelBridge"]) -> Type["MegatronModelBridge"]:
        register_bridge_implementation(source=source, target=target, bridge_class=bridge_class)
        return bridge_class

    return decorator
