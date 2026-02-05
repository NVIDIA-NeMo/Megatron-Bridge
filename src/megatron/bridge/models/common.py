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
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import torch
from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule


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
class ModelBuildConfig(abc.ABC, Serializable):
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

    def get_builder(self) -> "ModelBuilder":
        """Get the appropriate builder instance for this config.
        Dynamically imports and instantiates the builder from the string path.
        """
        module_path, class_name = self.builder.rsplit(".", 1)
        module = importlib.import_module(module_path)
        builder_cls = getattr(module, class_name)
        return builder_cls()

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to dictionary for saving.

        Includes:
        - _target_: Full class path for deserialization
        - _builder_: Full builder class path (serialized from ClassVar)
        - All dataclass fields
        """
        result = {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}",
            "_builder_": self.builder,  # Serialize the builder path
        }
        for f in dataclass_fields(self):
            value = getattr(self, f.name)
            # Skip non-serializable fields
            if callable(value) or f.name.startswith("_"):
                continue
            result[f.name] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelBuildConfig":
        """Deserialize config from dictionary.

        Uses _target_ to determine the correct class to instantiate.
        The builder is restored from _builder_ or from the class's ClassVar.

        Args:
            data: Dictionary with _target_ and config fields

        Returns:
            Instance of the appropriate ModelBuildConfig subclass
        """
        target = data.get("_target_")
        if target is None:
            raise ValueError("Cannot deserialize: missing '_target_' field")

        # Import the class from the target path
        module_path, class_name = target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        config_cls = getattr(module, class_name)

        # Filter to valid fields for this class
        valid_fields = {f.name for f in dataclass_fields(config_cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields and not k.startswith("_")}

        return config_cls(**filtered_data)


ModelT = TypeVar("ModelT", bound=MegatronModule)
BuildConfigT = TypeVar("BuildConfigT", bound=ModelBuildConfig)


class ModelBuilder(abc.ABC, Generic[ModelT, BuildConfigT]):
    """Abstract base class for model builders.

    A builder takes configuration(s) and produces model instances.

    Each builder subclass should:
    1. Implement build_model() for the specific model type
    2. Be linked to its corresponding ModelBuildConfig via the builder string

    Type Parameters:
        ModelT: The type of model this builder produces (e.g., MCoreGPTModel)
        BuildConfigT: The type of build config this builder accepts (e.g., GPTModelBuildConfig)
    """

    def __init__(self, model_config, build_config: BuildConfigT):
        self.model_config = model_config
        self.build_config = build_config

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
    builder: ModelBuilder[ModelT, BuildConfigT],
    pg_collection: ProcessGroupCollection,
    wrap_with_ddp: bool = True,
    use_torch_fsdp2: bool = False,
    use_cpu_initialization: bool = False,
    init_model_with_meta_device: bool = False,
    fp16: bool = False,
    bf16: bool = False,
) -> list[ModelT]:
    """Build models wrapped for distributed training.

    Handles virtual pipeline parallelism, DDP wrapping, and
    mixed precision configuration.
    """
    # Get VP size from model config if available
    vp_size = getattr(builder.model_config, "virtual_pipeline_model_parallel_size", None)
    pp_group = pg_collection.pp

    if pp_group.size() > 1 and vp_size is not None:
        # Create multiple models for virtual pipeline
        from megatron.core.pipeline_parallel.utils import (
            is_pp_first_stage,
            is_pp_last_stage,
            is_vp_first_stage,
            is_vp_last_stage,
        )

        models = []
        for i in range(vp_size):
            pre_process = is_vp_first_stage(vp_stage=i, vp_size=vp_size) and is_pp_first_stage(pp_group)
            post_process = is_vp_last_stage(vp_stage=i, vp_size=vp_size) and is_pp_last_stage(pp_group)
            model = builder.build_model(
                pg_collection,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=i,
            )
            models.append(model)
    else:
        # Single model
        model = builder.build_model(pg_collection)
        models = [model]

    for model_module in models:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    if not (use_torch_fsdp2 and use_cpu_initialization) and not init_model_with_meta_device:
        for model_module in models:
            model_module.cuda(torch.cuda.current_device())

    # TODO: handle Float16Module wrapper

    # Apply DDP wrapping if requested
    if wrap_with_ddp:
        models = _wrap_with_ddp(models, pg_collection, fp16, bf16)

    return models


def _wrap_with_ddp(
    models: list[ModelT],
    pg_collection: ProcessGroupCollection,
    fp16: bool,
    bf16: bool,
) -> list[ModelT]:
    """Wrap models with DDP for distributed training."""
    # TODO: impl
    ...
    return models
