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

import enum
import functools
import logging
from typing import Callable

from megatron.bridge.training.config import ConfigContainer


try:
    import nemo_run as run

    HAS_NEMO_RUN = True
except ImportError:
    run = None
    HAS_NEMO_RUN = False

logger: logging.Logger = logging.getLogger(__name__)


def get_partial_fn(
    target_fn: Callable[[ConfigContainer, Callable], None], config: ConfigContainer, forward_step_func: Callable
) -> "run.Partial":
    """
    Creates a run.Partial object for the given target function with configuration and forward step function.

    This is a convenience function that combines the preparation of the config for NeMo Run
    and the creation of the run.Partial object for any target function that expects a ConfigContainer
    and forward_step_func.

    Args:
        target_fn: The target function to be wrapped in run.Partial (e.g., pretrain).
        config: The ConfigContainer dataclass instance.
        forward_step_func: The forward step function to use for training.

    Returns:
        A run.Partial object ready to be executed with NeMo Run.

    Raises:
        ImportError: If nemo_run is not installed.
    """
    _check_nemo_run_available()

    # Prepare the config container for NeMo Run
    prepared_config = prepare_config_for_nemo_run(config)

    # Create and return the run.Partial object for the target function
    return run.Partial(target_fn, config=prepared_config, forward_step_func=forward_step_func)


def prepare_config_for_nemo_run(config: ConfigContainer) -> ConfigContainer:
    """
    Prepares a pure ConfigContainer instance for use with NeMo Run by patching
    fields that are not directly serializable by nemo_run, such as functools.partial objects
    and enum values that cause YAML serialization issues.
    Args:
        config: The ConfigContainer dataclass instance.
    Returns:
        The patched ConfigContainer instance compatible for execution with nemo_run.

    Raises:
        ImportError: If nemo_run is not installed.
    """
    _check_nemo_run_available()

    model_cfg = config.model
    patched_fields = []

    # Handle functools.partial objects
    if hasattr(model_cfg, "init_method") and isinstance(model_cfg.init_method, functools.partial):
        original_partial = model_cfg.init_method
        model_cfg.init_method = run.Partial(original_partial.func, *original_partial.args, **original_partial.keywords)
        patched_fields.append("model_config.init_method")

    if hasattr(model_cfg, "output_layer_init_method") and isinstance(
        model_cfg.output_layer_init_method, functools.partial
    ):
        original_partial = model_cfg.output_layer_init_method
        model_cfg.output_layer_init_method = run.Partial(
            original_partial.func, *original_partial.args, **original_partial.keywords
        )
        patched_fields.append("model_config.output_layer_init_method")

    # Check for other potential functools.partial objects in the model config
    for field_name in ("bias_init_method", "weight_init_method"):
        if hasattr(model_cfg, field_name):
            field_value = getattr(model_cfg, field_name)
            if isinstance(field_value, functools.partial):
                original_partial = field_value
                setattr(
                    model_cfg,
                    field_name,
                    run.Partial(original_partial.func, *original_partial.args, **original_partial.keywords),
                )
                patched_fields.append(f"model_config.{field_name}")

    # Fix YAML serialization issues with enum objects
    try:
        _fix_yaml_serialization_issues(config)
    except Exception as e:
        logger.warning(f"Failed to fix some YAML serialization issues: {e}")

    if patched_fields:
        logger.debug(f"Wrapped the following fields with run.Partial: {', '.join(patched_fields)}")

    return config


def _fix_yaml_serialization_issues(config: ConfigContainer) -> None:
    """Fix common YAML serialization issues in the configuration.
    This function handles specific known issues like enum objects that can't be
    serialized to YAML. It modifies the config in-place to use serializable
    representations while preserving functionality.
    Args:
        config: The ConfigContainer to fix
    """
    model_cfg = config.model
    fixed_fields = []

    # Handle specific enum fields that cause YAML serialization issues
    if hasattr(model_cfg, "attn_backend") and isinstance(model_cfg.attn_backend, enum.Enum):
        original_value = model_cfg.attn_backend
        model_cfg.attn_backend = original_value.value
        fixed_fields.append(f"model_config.attn_backend ({original_value} -> {original_value.value})")

    # Handle any other enum fields in model config
    for attr_name in ("attention_backend", "backend", "attn_type"):
        if hasattr(model_cfg, attr_name):
            attr_value = getattr(model_cfg, attr_name)
            if isinstance(attr_value, enum.Enum):
                setattr(model_cfg, attr_name, attr_value.value)
                fixed_fields.append(f"model_config.{attr_name} ({attr_value} -> {attr_value.value})")

    # Handle enum fields in other config sections
    for config_section_name in ("train", "optimizer", "scheduler"):
        if hasattr(config, config_section_name):
            config_section = getattr(config, config_section_name)
            if config_section is not None:
                for attr_name in dir(config_section):
                    if not attr_name.startswith("_"):
                        try:
                            attr_value = getattr(config_section, attr_name)
                            if isinstance(attr_value, enum.Enum):
                                setattr(config_section, attr_name, attr_value.value)
                                fixed_fields.append(
                                    f"{config_section_name}.{attr_name} ({attr_value} -> {attr_value.value})"
                                )
                        except (AttributeError, TypeError):
                            continue

    if fixed_fields:
        logger.debug(f"Fixed YAML serialization for enum fields: {', '.join(fixed_fields)}")


def _check_nemo_run_available() -> None:
    """Check if nemo_run is available and raise helpful error if not."""
    if not HAS_NEMO_RUN:
        raise ImportError(
            "nemo_run is required for recipe functionality. Install it with: pip install megatron-hub[recipes]"
        )
