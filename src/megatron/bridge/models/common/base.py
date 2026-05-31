# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
from dataclasses import dataclass, field, is_dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Callable, ClassVar, Generic, Protocol, TypeVar, runtime_checkable

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.module import Float16Module


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable configurations."""

    def as_dict(self) -> dict[str, Any]:
        """Serialize to dictionary with target metadata."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Serializable":
        """Deserialize from dictionary with target metadata."""
        ...


@dataclass
class ModelConfig:
    """Base class for model configurations."""

    builder: ClassVar[str]
    restore_modelopt_state: bool = False
    extra_checkpoint_metadata: dict[str, Any] | None = None
    pre_wrap_hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]] = field(default_factory=list)
    post_wrap_hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]] = field(default_factory=list)

    def get_builder_cls(self) -> type:
        """Get the builder class for this model config."""
        module_path, class_name = self.builder.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def as_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary."""

        def _as_dict(config):
            result = {"_target_": f"{config.__class__.__module__}.{config.__class__.__qualname__}"}
            for f in dataclass_fields(config):
                value = getattr(config, f.name)
                if callable(value) or f.name.startswith("_") or f.name in ("pre_wrap_hooks", "post_wrap_hooks"):
                    continue

                if is_dataclass(value):
                    result[f.name] = _as_dict(value)
                else:
                    result[f.name] = value

            return result

        result = _as_dict(self)
        result["_builder_"] = self.builder
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        """Deserialize a config from a dictionary produced by :meth:`as_dict`."""

        def _from_dict(subdata):
            target = subdata.get("_target_")
            if target is None:
                raise ValueError("Cannot deserialize: missing '_target_' field")

            module_path, class_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            config_cls = getattr(module, class_name)

            valid_fields = {f.name for f in dataclass_fields(config_cls)}
            filtered_data = {k: v for k, v in subdata.items() if k in valid_fields and not k.startswith("_")}

            subconfigs = {}
            for key, value in filtered_data.items():
                if isinstance(value, dict) and "_target_" in value:
                    subconfigs[key] = _from_dict(value)
            filtered_data.update(subconfigs)

            return config_cls(**filtered_data)

        result = _from_dict(data)
        result.builder = data["_builder_"]
        return result


ModelT = TypeVar("ModelT", bound=MegatronModule)
BuildConfigT = TypeVar("BuildConfigT", bound=ModelConfig)


class ModelBuilder(abc.ABC, Generic[ModelT, BuildConfigT]):
    """Abstract base class for model builders."""

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
        """Build a single model stage."""
        ...

    @abc.abstractmethod
    def build_distributed_models(
        self,
        pg_collection: ProcessGroupCollection,
        ddp_config: DistributedDataParallelConfig | None = None,
        overlap_param_gather_with_optimizer_step: bool = False,
        use_megatron_fsdp: bool = False,
        use_torch_fsdp2: bool = False,
        wrap_with_ddp: bool = True,
        data_parallel_random_init: bool = False,
        mixed_precision_wrapper: Callable[[Any, MegatronModule], MegatronModule] | None = Float16Module,
        model_type: ModelType = ModelType.encoder_or_decoder,
    ) -> list[ModelT]:
        """Build and wrap distributed model stages."""
        ...


def compose_hooks(
    hooks: list[Callable[[list[MegatronModule]], list[MegatronModule]]],
) -> Callable[[list[MegatronModule]], list[MegatronModule]]:
    """Compose pre/post-wrap hooks into a single function."""

    def composed_hook(model: list[MegatronModule]) -> list[MegatronModule]:
        for hook in hooks:
            model = hook(model)
        return model

    return composed_hook
