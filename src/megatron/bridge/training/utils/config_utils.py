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

import logging
from dataclasses import fields as dataclass_fields
from dataclasses import is_dataclass
from typing import Any, Mapping

from megatron.core.quantization.quant_config import GlobMatcher, Matcher, RecipeConfig
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.training.config.container import ConfigContainerBase as _MCoreConfigContainerBase
from megatron.training.config.utils import (
    _get_init_false_fields,  # noqa: F401
    _resolve_target_class,  # noqa: F401
)
from megatron.training.config.utils import (
    sanitize_dataclass_config as _sanitize_dataclass_config,  # noqa: F401
)
from megatron.training.models.base import Serializable
from transformers import PreTrainedConfig


logger = logging.getLogger(__name__)


def _materialize_gtp_weight_shards(value: Any, _memo: dict[int, Any] | None = None) -> Any:
    """Preserve runtime GTP sizes in constructor fields before sanitizing configs."""
    if not isinstance(value, (dict, list)):
        return value
    if _memo is None:
        _memo = {}
    if id(value) in _memo:
        return _memo[id(value)]
    if isinstance(value, list):
        result_list = []
        _memo[id(value)] = result_list
        result_list.extend(_materialize_gtp_weight_shards(item, _memo) for item in value)
        return result_list
    result = {}
    _memo[id(value)] = result
    result.update({key: _materialize_gtp_weight_shards(item, _memo) for key, item in value.items()})
    for shard_name, parallel_name, remat_name in (
        ("tensor_parallel_num_weight_shards", "tensor_model_parallel_size", "gtp_weight_remat_size"),
        (
            "expert_tensor_parallel_num_weight_shards",
            "expert_tensor_parallel_size",
            "expert_gtp_weight_remat_size",
        ),
    ):
        remat_size = result.get(remat_name)
        if result.get(shard_name) is not None or not isinstance(remat_size, int) or remat_size <= 1:
            continue
        tp_size = result.get("tensor_model_parallel_size", 1)
        parallel_size = result.get(parallel_name) or tp_size
        if isinstance(parallel_size, int) and parallel_size > 0:
            result[shard_name] = parallel_size * remat_size
    return result


class _ConfigContainerBase(_MCoreConfigContainerBase):
    """Bridge config container that lets composite HF configs construct their children.

    Nested ``PreTrainedConfig`` values are emitted as plain mappings because their
    parent config owns child construction. Other dataclasses retain MCore's recursive
    ``_target_`` serialization.

    Megatron Core serializes container sections with module-level helpers that do not
    dispatch to subclass hooks, so Bridge owns the traversal: every section and nested
    value goes through :meth:`_convert_value_to_dict`.
    """

    def to_dict(self, **section_overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Convert the container to a dictionary through Bridge's value conversion.

        Args:
            **section_overrides: Map a section name to field values that replace the
                section's own values in the result, as in Megatron Core.

        Returns:
            Dictionary representation of this config with an importable ``_target_``.
        """
        sections = {f.name for f in dataclass_fields(self) if not f.name.startswith("_")}
        unknown = set(section_overrides) - sections
        if unknown:
            raise ValueError(f"{self.__class__.__qualname__} has no section named {sorted(unknown)}")
        result: dict[str, Any] = {"_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}"}
        for f in dataclass_fields(self):
            if not f.name.startswith("_"):
                result[f.name] = self._convert_value_to_dict(getattr(self, f.name))
        for name, overrides in section_overrides.items():
            result[name] = self._overlay_section(name, getattr(self, name), overrides)
        return result

    @classmethod
    def _overlay_section(cls, name: str, section: Any, overrides: Mapping[str, Any]) -> dict[str, Any]:
        """Return the serialized ``section`` with ``overrides`` in place of its own field values."""
        if not isinstance(overrides, Mapping):
            raise TypeError(
                f"overrides of section {name!r} must map field names to values, got {type(overrides).__name__}"
            )
        if not (is_dataclass(section) and not isinstance(section, type)):
            raise TypeError(f"section {name!r} is not a dataclass; it takes no overrides")
        unknown = set(overrides) - {item.name for item in dataclass_fields(section)}
        if unknown:
            raise ValueError(f"section {name!r} has no field named {sorted(unknown)}")
        return {
            **cls._convert_value_to_dict(section),
            **{field_name: cls._convert_value_to_dict(value) for field_name, value in overrides.items()},
        }

    @classmethod
    def _convert_value_to_dict(cls, value: Any) -> Any:
        if isinstance(value, PipelineParallelLayerLayout):
            return cls._convert_value_to_dict(value.input_data)
        if isinstance(value, RecipeConfig) and not hasattr(value, "to_cfg_dict"):
            if type(value) is not RecipeConfig:
                recipe_type = f"{type(value).__module__}.{type(value).__qualname__}"
                raise TypeError(
                    f"Unsupported quantization recipe type: {recipe_type}. "
                    "RecipeConfig subclasses must implement to_cfg_dict()."
                )
            return cls._convert_recipe_config_to_dict(value)
        if isinstance(value, PreTrainedConfig) and not hasattr(value, "to_cfg_dict"):
            return cls._convert_pretrained_config_to_dict(value, include_target=True)
        return _materialize_gtp_weight_shards(cls._convert_generic_value_to_dict(value))

    @classmethod
    def _convert_generic_value_to_dict(cls, value: Any) -> Any:
        """Recursively convert containers, dataclasses and collections.

        Nested values go back through :meth:`_convert_value_to_dict` so Bridge-specific
        conversions apply at any depth.
        """
        if isinstance(value, type):
            return value
        if isinstance(value, _MCoreConfigContainerBase):
            return value.to_dict()
        if isinstance(value, Serializable):
            return value.as_dict()
        if hasattr(value, "to_cfg_dict"):
            return value.to_cfg_dict()
        if is_dataclass(value):
            result: dict[str, Any] = {"_target_": f"{value.__class__.__module__}.{value.__class__.__qualname__}"}
            for field in dataclass_fields(value):
                if not field.name.startswith("_"):
                    result[field.name] = cls._convert_value_to_dict(getattr(value, field.name))
            return result
        if isinstance(value, (list, tuple)):
            return [cls._convert_value_to_dict(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._convert_value_to_dict(item) for key, item in value.items()}
        return value

    @classmethod
    def _convert_recipe_config_to_dict(cls, value: RecipeConfig) -> dict[str, Any]:
        """Convert an MCore quantization recipe to an instantiable mapping."""
        serialized_matchers: list[dict[str, Any]] = []
        for matcher in value.matchers:
            serialized_matchers.append(cls._convert_recipe_matcher_to_dict(matcher))

        return {
            "_target_": f"{RecipeConfig.__module__}.{RecipeConfig.__qualname__}",
            "matchers": serialized_matchers,
            "config_dict": cls._convert_value_to_dict(value.configs),
        }

    @classmethod
    def _convert_recipe_matcher_to_dict(cls, matcher: Matcher) -> dict[str, Any]:
        """Convert a quantization recipe matcher to an instantiable mapping."""
        if type(matcher) is GlobMatcher:
            return {
                "_target_": f"{GlobMatcher.__module__}.{GlobMatcher.__qualname__}",
                "pattern": matcher.pattern,
                "config_key": matcher.config_key,
            }

        serialized_matcher = cls._convert_generic_value_to_dict(matcher)
        if not isinstance(serialized_matcher, dict) or "_target_" not in serialized_matcher:
            matcher_type = f"{type(matcher).__module__}.{type(matcher).__qualname__}"
            raise TypeError(
                f"Unsupported quantization recipe matcher type: {matcher_type}. "
                "Custom matchers must be dataclasses or implement to_cfg_dict()."
            )
        return serialized_matcher

    @classmethod
    def _convert_pretrained_config_to_dict(
        cls,
        value: PreTrainedConfig,
        *,
        include_target: bool,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if include_target:
            result["_target_"] = f"{value.__class__.__module__}.{value.__class__.__qualname__}"

        if include_target and is_dataclass(value):
            field_names = {field.name for field in dataclass_fields(value) if not field.name.startswith("_")}
            raw_values = vars(value)
            config_items = [
                (
                    field.name,
                    raw_values[field.name] if field.name in raw_values else object.__getattribute__(value, field.name),
                )
                for field in dataclass_fields(value)
                if not field.name.startswith("_")
            ]
            # Runtime normalization may add self-aliases that are not constructor state.
            config_items.extend(
                (key, item)
                for key, item in vars(value).items()
                if not key.startswith("_") and key not in field_names and item is not value
            )
        else:
            config_items = ((key, item) for key, item in value.to_dict().items() if not key.startswith("_"))

        for key, item in config_items:
            result[key] = cls._convert_pretrained_config_value_to_dict(item)
        return result

    @classmethod
    def _convert_pretrained_config_value_to_dict(cls, value: Any) -> Any:
        if isinstance(value, PreTrainedConfig):
            return cls._convert_pretrained_config_to_dict(value, include_target=False)
        if isinstance(value, (list, tuple)):
            return [cls._convert_pretrained_config_value_to_dict(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._convert_pretrained_config_value_to_dict(item) for key, item in value.items()}
        return cls._convert_value_to_dict(value)


def create_ddp_config(
    wrap_with_ddp: bool = True,
    use_distributed_optimizer: bool = True,
    use_megatron_fsdp: bool = False,
    overrides: Mapping[str, object] | None = None,
    finalize: bool = True,
) -> object | None:
    """Create a finalized Bridge DDP config for external model construction."""
    if not wrap_with_ddp:
        return None

    from megatron.bridge.training.config import DistributedDataParallelConfig

    ddp_config = {
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if use_megatron_fsdp:
        ddp_config.update(
            {
                "use_distributed_optimizer": True,
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    ddp_config.update(overrides or {})

    config = DistributedDataParallelConfig(**ddp_config)
    if finalize:
        config.finalize()
    return config
