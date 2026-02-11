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
import importlib
from dataclasses import dataclass, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar

import torch
from megatron.core import tensor_parallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module

from megatron.bridge.models.model_provider import _ddp_wrap, _print_num_params
from megatron.bridge.models.transformer_config import TransformerConfig


try:
    from megatron.core.fp8_utils import correct_amax_history_if_needed
except ImportError:
    correct_amax_history_if_needed = None


class Serializable(Protocol):
    """Protocol for serializable configurations."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with _target_ for class identification."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Deserialize from dictionary using _target_ to identify class."""
        ...


@dataclass
class ModelConfig(abc.ABC, Serializable):
    """Abstract base class for model build configurations.

    Each model type (GPT, T5, Mamba, etc.) has its own build config subclass.
    The build config contains:
    1. Builder path (serializable string) to link to the correct builder
    2. Model-specific parameters not in TransformerConfig
    3. HuggingFace metadata for checkpoint conversion

    Each subclass must define `builder` as a ClassVar string pointing to
    the appropriate ModelBuilder subclass path.
    """

    # === Builder Metadata (Serializable) ===
    builder: ClassVar[str]
    """Class variable with full path to builder class (e.g.,
    'megatron.bridge.builders.GPTModelBuilder').
    """

    # === ModelOpt ===
    restore_modelopt_state: bool = False
    """Restore ModelOpt quantization/sparsity state."""

    # === HuggingFace Metadata ===
    hf_model_id: str | None = None
    """HuggingFace model identifier."""

    generation_config: Any | None = None
    """Generation configuration."""

    def get_builder_cls(self) -> type:
        """Get the appropriate builder instance for this config.
        Dynamically imports and instantiates the builder from the string path.
        """
        module_path, class_name = self.builder.rsplit(".", 1)
        module = importlib.import_module(module_path)
        builder_cls = getattr(module, class_name)
        return builder_cls

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for saving.

        Includes:
        - _target_: Full class path for deserialization
        - _builder_: Full builder class path (serialized from ClassVar)
        - All dataclass fields, including nested dataclasses
        """

        def _to_dict(config):
            result = {
                "_target_": f"{config.__class__.__module__}.{config.__class__.__qualname__}",
            }
            for f in dataclass_fields(config):
                value = getattr(config, f.name)
                # Skip non-serializable fields
                if callable(value) or f.name.startswith("_"):
                    continue

                if is_dataclass(value):
                    result[f.name] = _to_dict(value)  # recurse on nested dataclasses
                else:
                    result[f.name] = value

            return result

        result = _to_dict(self)
        result["_builder_"] = self.builder  # Serialize the builder path
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize config from dictionary.

        Uses _target_ to determine the correct class to instantiate.
        The builder is restored from _builder_ or from the class's ClassVar.

        Args:
            data: Dictionary with _target_ and config fields

        Returns:
            Instance of the appropriate ModelBuildConfig subclass
        """

        def _from_dict(subdata):
            target = subdata.get("_target_")
            if target is None:
                raise ValueError("Cannot deserialize: missing '_target_' field")

            # Import the class from the target path
            module_path, class_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            config_cls = getattr(module, class_name)

            # Filter to valid fields for this class
            valid_fields = {f.name for f in dataclass_fields(config_cls)}
            filtered_data = {k: v for k, v in subdata.items() if k in valid_fields and not k.startswith("_")}

            # recurse on serialized nested dataclasses
            subconfigs = {}
            for k, v in filtered_data.items():
                if isinstance(v, dict):
                    subconfigs[k] = _from_dict(v)
            filtered_data.update(subconfigs)

            return config_cls(**filtered_data)

        result = _from_dict(data)
        result.builder = data["_builder_"]

        return result


ModelT = TypeVar("ModelT", bound=MegatronModule)
BuildConfigT = TypeVar("BuildConfigT", bound=ModelConfig)


class ModelBuilder(abc.ABC, Generic[ModelT, BuildConfigT]):
    """Abstract base class for model builders.

    A builder takes configuration(s) and produces model instances.

    Each builder subclass should:
    1. Implement build_model() for the specific model type
    2. Be linked to its corresponding ModelBuildConfig via the builder string

    Builders are factory objects, therefore any state saved in __init__ should not be modified
    and only used to build the model.

    Type Parameters:
        ModelT: The type of model this builder produces (e.g., MCoreGPTModel)
        BuildConfigT: The type of build config this builder accepts (e.g., GPTModelBuildConfig)
    """

    def __init__(self, model_config: ModelConfig):
        self._model_config = model_config

    @abc.abstractmethod
    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> ModelT:
        """Build a model from the provided configurations.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer
            post_process: Include output layer
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model
        """
        ...

    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = True,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[ModelT]:
        """Build model stages and wrap for distributed training.

        This default implementation only intends to support decoder-only models.

        Handles virtual pipeline parallelism, DDP wrapping, and
        mixed precision configuration.
        """
        if wrap_with_ddp and not ddp_config:
            raise ValueError("ddp_config is required when wrap_with_ddp is True")

        transformer_config: TransformerConfig | None = getattr(self._model_config, "transformer", None)

        def find_model_attr(attr_name):
            """Look for an attribute in both self._model_config and transformer_config."""
            attr_value = getattr(self._model_config, attr_name, None)
            if attr_value is None:
                getattr(transformer_config, attr_name, None)
            return attr_value

        # TODO (@maanug): handle pre/post wrap hooks

        # Get VP size from model config if available
        vp_size = find_model_attr("virtual_pipeline_model_parallel_size")
        init_model_with_meta_device = find_model_attr("init_model_with_meta_device") or False
        if init_model_with_meta_device:
            with torch.device("meta"):
                model_list = self._build_model_stages(pg_collection, vp_size)
        else:
            model_list = self._build_model_stages(pg_collection, vp_size)

        # Set tensor model parallel attributes if not set.
        # Only parameters that are already tensor model parallel have these
        # attributes set for them. We should make sure the default attributes
        # are set for all params so the optimizer can use them.
        for model_module in model_list:
            for param in model_module.parameters():
                tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

        _print_num_params(model_list, pg_collection=pg_collection)

        # GPU allocation.
        # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
        # in the fully_shard function of FSDP2 instead.
        use_cpu_initialization = find_model_attr("use_cpu_initialization") or False
        if not use_torch_fsdp2 and not use_cpu_initialization and not init_model_with_meta_device:
            for model_module in model_list:
                model_module.cuda(torch.cuda.current_device())

        fp16 = find_model_attr("fp16") or False
        bf16 = find_model_attr("bf16") or False
        if (fp16 or bf16) and mixed_precision_wrapper is not None:
            model_list = [mixed_precision_wrapper(transformer_config, model_module) for model_module in model_list]

            # Maintain expert bias in float32 wrapped in Float16Module
            for model_module in model_list:
                for submodule in model_module.modules():
                    if hasattr(submodule, "_maintain_float32_expert_bias"):
                        submodule._maintain_float32_expert_bias()

        # Materialize tensors on meta device (GPU allocation) if not using FSDP2 and not using Megatron FSDP.
        if init_model_with_meta_device and not use_torch_fsdp2 and not use_megatron_fsdp:
            model_list = [
                to_empty_if_meta_device(model_module, device=torch.device("cuda")) for model_module in model_list
            ]

        if correct_amax_history_if_needed is not None:
            correct_amax_history_if_needed(model_list)

        if wrap_with_ddp:
            model_list = _ddp_wrap(
                model_list,
                data_parallel_random_init,
                ddp_config,
                overlap_param_gather_with_optimizer_step,
                use_megatron_fsdp=use_megatron_fsdp,
                use_torch_fsdp2=use_torch_fsdp2,
                pg_collection=pg_collection,
            )

        return model_list

    def _build_model_stages(
        self,
        pg_collection: ProcessGroupCollection,
        vp_size: int | None,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[ModelT]:
        """Build virtual pipeline stages if using virtual pipeline parallelism."""
        from megatron.core.pipeline_parallel.utils import (
            is_pp_first_stage,
            is_pp_last_stage,
            is_vp_first_stage,
            is_vp_last_stage,
        )

        pp_group = pg_collection.pp
        if pp_group.size() > 1 and vp_size is not None:
            # Create multiple model stages for virtual pipeline
            model_list = []
            for i in range(vp_size):
                pre_process = is_vp_first_stage(vp_stage=i, vp_size=vp_size) and is_pp_first_stage(pp_group)
                post_process = is_vp_last_stage(vp_stage=i, vp_size=vp_size) and is_pp_last_stage(pp_group)
                model = self.build_model(
                    pg_collection,
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=i,
                )
                model.model_type = model_type
                model_list.append(model)
        else:
            # Single stage, no VP
            pre_process = is_pp_first_stage(pp_group)
            post_process = is_pp_last_stage(pp_group)
            model = self.build_model(pg_collection, pre_process=pre_process, post_process=post_process)
            model.model_type = model_type
            model_list = [model]

        return model_list


@dataclass
class ModelProvider(Generic[ModelT]):
    """General provider that takes model config + build config and builds models.

    This is the main entry point for model construction. It automatically
    selects the correct builder based on the model_config's builder attribute.

    Example:
        >>> provider = ModelProvider(model_cfg)
        >>> model = provider.provide(pg_collection)
        >>>
        >>> # Or for distributed training with DDP
        >>> models = provider.provide_distributed(pg_collection, wrap_with_ddp=True)
    """

    model_config: ModelConfig

    def provide(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> ModelT:
        """Build and return a model.

        Automatically selects the correct builder based on model_config.builder.

        Args:
            pg_collection: Process groups for distributed training
            pre_process: Include embedding layer (default: based on PP stage)
            post_process: Include output layer (default: based on PP stage)
            vp_stage: Virtual pipeline stage

        Returns:
            The constructed model
        """
        builder_cls = self.model_config.get_builder_cls()
        return builder_cls(self.model_config).build_model(
            pg_collection,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

    def provide_distributed(
        self,
        pg_collection: ProcessGroupCollection,
        wrap_with_ddp: bool = True,
        fp16: bool = False,
        bf16: bool = False,
    ) -> list[ModelT]:
        """Build models wrapped for distributed training.

        Handles virtual pipeline parallelism, DDP wrapping, and
        mixed precision configuration.

        Args:
            pg_collection: Process groups for distributed training
            wrap_with_ddp: Whether to wrap with DDP
            fp16: Use FP16 mixed precision
            bf16: Use BF16 mixed precision

        Returns:
            List of models (multiple for virtual pipeline parallelism)
        """
        builder_cls = self.model_config.get_builder_cls()
        return builder_cls(self.model_config).build_distributed_models(
            pg_collection,
            wrap_with_ddp=wrap_with_ddp,
            fp16=fp16,
            bf16=bf16,
        )


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().

    Officially, torch suggests to_empty() for meta device materialization. Under the hood,
    torch.empty_like() is applied to all parameters or buffers (see _apply). This may
    accidently overwrite buffers with precomputed values during construction. Given the
    goal is to only materialize those tensors on meta device, this function checks the
    device first and only move the tensor to the destination if it is not on meta device.

    Args:
        module: The target module to apply this transformation.
        device: The desired device of the parameters
            and buffers in this module.
        recurse: Whether parameters and buffers of submodules should
            be recursively moved to the specified device.
    """

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            return torch.empty_like(tensor, device=device)
        else:
            return tensor.to(device)

    return module._apply(lambda t: _empty_like_if_meta(t, device=device), recurse=recurse)
