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
    sanitize_dataclass_config as _sanitize_dataclass_config,
)
from transformers import PreTrainedConfig


logger = logging.getLogger(__name__)


class _ConfigContainerBase(_MCoreConfigContainerBase):
    """Bridge config container that lets composite HF configs construct their children.

    Nested ``PreTrainedConfig`` values are emitted as plain mappings because their
    parent config owns child construction. Other dataclasses retain MCore's recursive
    ``_target_`` serialization.
    """

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
        return super()._convert_value_to_dict(value)

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

        serialized_matcher = super()._convert_value_to_dict(matcher)
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
            config_items = (
                (field.name, getattr(value, field.name))
                for field in dataclass_fields(value)
                if not field.name.startswith("_")
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


def apply_run_config_backward_compat(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply backward compatibility transformations to run config.

    This function handles dataclass config fields that should not be passed to
    the constructor when loading older checkpoints. It automatically detects
    init=False fields by inspecting the target class.

    The entire config is sanitized recursively to handle init=False fields in any part of the configuration hierarchy.

    Args:
        config_dict: The full run configuration dictionary.

    Returns:
        The config dictionary with backward compatibility fixes applied.
    """
    return _sanitize_dataclass_config(config_dict)
