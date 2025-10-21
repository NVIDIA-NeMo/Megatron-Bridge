#!/usr/bin/env python3
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

"""Utilities for working with OmegaConf and dataclass configurations."""

import dataclasses
import functools
import inspect
import logging
from typing import Any, TypeVar

import torch
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)

DataclassInstance = TypeVar("DataclassInstance")

# Sentinel object to distinguish between "exclude this field" and "field is legitimately None"
_EXCLUDE_FIELD = object()


def create_omegaconf_dict_config(config_container: Any) -> tuple[DictConfig, dict[str, Any]]:
    """Create OmegaConf while tracking excluded fields for later restoration.

    This function combines the conversion to OmegaConf with tracking of excluded
    callable fields, allowing them to be restored after override processing.

    Args:
        config_container: The dataclass instance to convert

    Returns:
        Tuple of (OmegaConf DictConfig, excluded fields dictionary)

    Raises:
        ValueError: If the conversion fails
    """
    logger.debug("Starting safe OmegaConf conversion with callable preservation...")

    # Track all callable fields that will be excluded
    excluded_callables = _track_excluded_fields(config_container, "root")
    logger.debug(f"Found {len(excluded_callables)} callable fields to preserve")

    # Convert to OmegaConf (excluding callables)
    base_dict = _dataclass_to_omegaconf_dict(config_container, "root")

    if base_dict is _EXCLUDE_FIELD:
        raise ValueError("Root configuration object was excluded (likely a callable)")

    # Verify no callables remain
    if not _verify_no_callables(base_dict, "root"):
        raise ValueError("Callable objects found in converted dictionary")

    # Create OmegaConf
    omega_conf = OmegaConf.create(base_dict)

    return omega_conf, excluded_callables


def apply_overrides(
    config_obj: DataclassInstance, overrides_dict: dict[str, Any], excluded_fields: dict[str, Any]
) -> None:
    """Apply overrides while preserving excluded callable fields.

    This function first applies the overrides using the standard recursive approach,
    then restores the callable fields that were excluded during OmegaConf conversion.

    Args:
        config_obj: The dataclass instance to modify
        overrides_dict: Dictionary of override values to apply
        excluded_fields: Dictionary of excluded callable fields to restore
    """
    # Apply normal overrides
    _apply_overrides(config_obj, overrides_dict)

    # Restore excluded fields
    _restore_excluded_fields(config_obj, excluded_fields)

    logger.debug("Configuration updated with overrides and excluded fields preserved")


def parse_hydra_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
    """Parse and apply Hydra overrides to an OmegaConf config.

    This function uses Hydra's override parser to support advanced override syntax
    including additions (+), deletions (~), and complex nested operations.

    Args:
        cfg: OmegaConf config to apply overrides to
        overrides: List of Hydra override strings

    Returns:
        Updated config with overrides applied

    Raises:
        OverridesError: If there's an error parsing or applying overrides
    """
    try:
        OmegaConf.set_struct(cfg, True)
        parser = OverridesParser.create()
        parsed = parser.parse_overrides(overrides=overrides)
        ConfigLoaderImpl._apply_overrides_to_config(overrides=parsed, cfg=cfg)
        return cfg
    except Exception as e:
        raise OverridesError(f"Failed to parse Hydra overrides: {str(e)}") from e


class OverridesError(Exception):
    """Custom exception for Hydra override parsing errors."""

    pass


def _is_omegaconf_problematic(val: Any) -> bool:
    """Check if a value is a callable that OmegaConf cannot handle.

    OmegaConf cannot serialize function objects, methods, or partial functions.
    This function identifies such problematic callables while allowing class types.

    Args:
        val: The value to check

    Returns:
        True if the value is a problematic callable, False otherwise
    """
    if not callable(val):
        return False

    # Allow classes/types
    if isinstance(val, type):
        return False

    # Block function objects, methods, partial functions, etc.
    return (
        hasattr(val, "__call__")
        and not isinstance(val, type)
        and (hasattr(val, "__module__") or hasattr(val, "__qualname__") or isinstance(val, functools.partial))
    )


def _dataclass_to_omegaconf_dict(val_to_convert: Any, path: str = "") -> Any:
    """Recursively convert a dataclass instance to a dictionary suitable for OmegaConf.create.

    This function completely excludes problematic callable objects to prevent OmegaConf errors.
    It handles dataclasses, lists, tuples, dictionaries, and primitive types, while converting
    torch.dtype objects to strings for serialization.

    Args:
        val_to_convert: The value to convert
        path: Current path for debugging (e.g., "model_config.activation_func")

    Returns:
        Converted value suitable for OmegaConf, or _EXCLUDE_FIELD for excluded callables
    """
    current_path = path

    # Handle Hugging Face GenerationConfig / PretrainedConfig by converting to a callable dict
    # compatible with our YAML representer logic
    try:
        from transformers import GenerationConfig, PretrainedConfig  # type: ignore

        if isinstance(val_to_convert, (GenerationConfig, PretrainedConfig)):
            cfg_class = val_to_convert.__class__
            target = f"{inspect.getmodule(cfg_class).__name__}.{cfg_class.__qualname__}.from_dict"
            logger.debug(f"Converting {cfg_class.__qualname__} at {current_path} to callable dict")
            return {
                "_target_": target,
                "_call_": True,
                "config_dict": val_to_convert.to_dict(),
            }
    except ModuleNotFoundError:
        # transformers is optional; if unavailable, fall through to other handlers
        pass

    # Explicitly handle torch.dtype - convert to string
    if isinstance(val_to_convert, torch.dtype):
        logger.debug(f"Converting torch.dtype at {current_path}: {val_to_convert}")
        return str(val_to_convert)

    # Handle callables - exclude them completely
    if _is_omegaconf_problematic(val_to_convert):
        logger.debug(f"Excluding callable at {current_path}: {type(val_to_convert)} - {val_to_convert}")
        return _EXCLUDE_FIELD

    # Handle dataclasses
    elif dataclasses.is_dataclass(val_to_convert) and not isinstance(val_to_convert, type):
        res = {}
        for field in dataclasses.fields(val_to_convert):
            field_name = field.name
            field_path = f"{current_path}.{field_name}" if current_path else field_name

            try:
                field_value = getattr(val_to_convert, field_name)
                converted_value = _dataclass_to_omegaconf_dict(field_value, field_path)

                # Only exclude fields marked with sentinel (not legitimate None values)
                if converted_value is not _EXCLUDE_FIELD:
                    res[field_name] = converted_value
                else:
                    logger.debug(f"Excluded field {field_path}")

            except (AttributeError, TypeError) as e:
                # Only catch specific exceptions from field access
                logger.warning(f"Error processing field {field_path}: {e}")
                continue

        return res

    # Handle lists
    elif isinstance(val_to_convert, list):
        result = []
        for i, item in enumerate(val_to_convert):
            item_path = f"{current_path}[{i}]"
            converted_item = _dataclass_to_omegaconf_dict(item, item_path)

            # Only exclude items marked with sentinel (not legitimate None values)
            if converted_item is not _EXCLUDE_FIELD:
                result.append(converted_item)

        return result

    # Handle tuples
    elif isinstance(val_to_convert, tuple):
        converted_items = []
        for i, item in enumerate(val_to_convert):
            item_path = f"{current_path}[{i}]"
            converted_item = _dataclass_to_omegaconf_dict(item, item_path)

            # Only exclude items marked with sentinel (not legitimate None values)
            if converted_item is not _EXCLUDE_FIELD:
                converted_items.append(converted_item)

        return tuple(converted_items)

    # Handle dictionaries
    elif isinstance(val_to_convert, dict):
        result = {}
        for key, value in val_to_convert.items():
            key_path = f"{current_path}.{key}" if current_path else str(key)
            converted_value = _dataclass_to_omegaconf_dict(value, key_path)

            # Only exclude values marked with sentinel (not legitimate None values)
            if converted_value is not _EXCLUDE_FIELD:
                result[key] = converted_value

        return result

    # Return primitive types as-is (including legitimate None values)
    else:
        return val_to_convert


def _track_excluded_fields(obj: Any, path: str = "") -> dict[str, Any]:
    """Track all excluded callable fields and their original values.

    This function recursively traverses a dataclass structure and builds a mapping
    of field paths to their original callable values that will be excluded during
    OmegaConf conversion.

    Args:
        obj: The object to analyze for callable fields
        path: Current path prefix for building field paths

    Returns:
        Dictionary mapping field paths to their original callable values
    """
    excluded_fields = {}

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            field_name = field.name
            field_path = f"{path}.{field_name}" if path else field_name
            field_value = getattr(obj, field_name)

            if _is_omegaconf_problematic(field_value):
                excluded_fields[field_path] = field_value
                logger.debug(f"Tracking excluded callable: {field_path}")
            elif dataclasses.is_dataclass(field_value):
                nested_excluded = _track_excluded_fields(field_value, field_path)
                excluded_fields.update(nested_excluded)
            elif isinstance(field_value, dict):
                for key, value in field_value.items():
                    if _is_omegaconf_problematic(value):
                        excluded_fields[f"{field_path}.{key}"] = value

    return excluded_fields


def _restore_excluded_fields(config_obj: Any, excluded_fields: dict[str, Any]) -> None:
    """Restore excluded callable fields to their original values.

    After applying overrides from OmegaConf, this function restores the callable
    fields that were excluded during the conversion process.

    Args:
        config_obj: The configuration object to restore fields on
        excluded_fields: Dictionary mapping field paths to their original values
    """
    for field_path, original_value in excluded_fields.items():
        try:
            # Navigate to the parent object and field name
            path_parts = field_path.split(".")
            if path_parts[0] == "root":
                path_parts = path_parts[1:]  # Remove "root" prefix

            current_obj = config_obj

            # Navigate to the parent object
            for part in path_parts[:-1]:
                current_obj = getattr(current_obj, part)

            field_name = path_parts[-1]

            # Restore the original callable
            setattr(current_obj, field_name, original_value)
            logger.debug(f"Restored callable field: {field_path}")

        except (AttributeError, TypeError) as e:
            logger.warning(f"Failed to restore callable field {field_path}: {e}")


def _verify_no_callables(obj: Any, path: str = "") -> bool:
    """Recursively verify that no callable objects remain in the converted structure.

    This function is used for validation to ensure that all problematic callables
    have been successfully excluded from a data structure before OmegaConf conversion.

    Args:
        obj: The object to verify
        path: Current path for error reporting

    Returns:
        True if no problematic callables are found, False otherwise
    """
    if _is_omegaconf_problematic(obj):
        logger.error(f"Found problematic callable at {path}: {obj}")
        return False

    elif isinstance(obj, dict):
        for key, value in obj.items():
            key_path = f"{path}.{key}" if path else str(key)
            if not _verify_no_callables(value, key_path):
                return False

    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            item_path = f"{path}[{i}]"
            if not _verify_no_callables(item, item_path):
                return False

    return True


def _apply_overrides(config_obj: DataclassInstance, overrides_dict: dict[str, Any]) -> None:
    """Recursively apply overrides from a Python dictionary to a dataclass instance.

    This function traverses nested dataclass structures and applies override values
    from a dictionary. It handles type conversions for special cases like torch.dtype.
    It also handles dictionaries with _target_ fields by instantiating them properly.

    Args:
        config_obj: The dataclass instance to modify
        overrides_dict: Dictionary of override values to apply
    """
    if not dataclasses.is_dataclass(config_obj):
        logger.debug(f"Skipping apply_overrides for non-dataclass config_obj: {type(config_obj)}")
        return

    for key, value in overrides_dict.items():
        if not hasattr(config_obj, key):
            logger.warning(
                f"Key '{key}' in overrides not found in config object {type(config_obj).__name__}. Skipping."
            )
            continue

        current_attr = getattr(config_obj, key)

        # Handle dictionaries with _target_ fields
        if isinstance(value, dict) and "_target_" in value:
            try:
                from megatron.bridge.utils.instantiate_utils import instantiate

                instantiated_obj = instantiate(value)
                setattr(config_obj, key, instantiated_obj)
                logger.debug(f"Successfully instantiated {key} from _target_: {value['_target_']}")
                continue
            except Exception as e:
                logger.warning(f"Failed to instantiate {key} from _target_: {e}")

        # Handle nested dataclass structures
        if dataclasses.is_dataclass(current_attr) and isinstance(value, dict):
            _apply_overrides(current_attr, value)
        else:
            try:
                # Handle special case conversions if needed
                final_value = value

                # If the original was a torch.dtype and value is a string, convert back
                if isinstance(current_attr, torch.dtype) and isinstance(value, str):
                    try:
                        final_value = getattr(torch, value.split(".")[-1])
                    except AttributeError:
                        logger.warning(f"Could not convert string '{value}' back to torch.dtype")
                        final_value = value

                setattr(config_obj, key, final_value)
                logger.debug(f"Set {type(config_obj).__name__}.{key} = {final_value}")

            except Exception as e:
                logger.warning(
                    f"Could not set attribute {type(config_obj).__name__}.{key} to value '{value}'. Error: {e}"
                )
