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
import logging
from typing import Any, Dict, Tuple, TypeVar

import torch
from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

DataclassInstance = TypeVar('DataclassInstance')


class OverridesError(Exception):
    """Custom exception for Hydra override parsing errors."""

    pass


def is_problematic_callable(val: Any) -> bool:
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
        hasattr(val, '__call__')
        and not isinstance(val, type)
        and (hasattr(val, '__module__') or hasattr(val, '__qualname__') or isinstance(val, functools.partial))
    )


def dataclass_to_dict_for_omegaconf(val_to_convert: Any, path: str = "") -> Any:
    """Recursively convert a dataclass instance to a dictionary suitable for OmegaConf.create.

    This function completely excludes problematic callable objects to prevent OmegaConf errors.
    It handles dataclasses, lists, tuples, dictionaries, and primitive types, while converting
    torch.dtype objects to strings for serialization.

    Args:
        val_to_convert: The value to convert
        path: Current path for debugging (e.g., "model_config.activation_func")

    Returns:
        Converted value suitable for OmegaConf, or None for excluded callables
    """
    current_path = path

    # Handle torch.dtype - convert to string
    if isinstance(val_to_convert, torch.dtype):
        logger.debug(f"Converting torch.dtype at {current_path}: {val_to_convert}")
        return str(val_to_convert)

    # Handle problematic callables - EXCLUDE them completely
    elif is_problematic_callable(val_to_convert):
        logger.info(f"Excluding callable at {current_path}: {type(val_to_convert)} - {val_to_convert}")
        return None  # Signal to exclude this field

    # Handle dataclasses
    elif dataclasses.is_dataclass(val_to_convert) and not isinstance(val_to_convert, type):
        res = {}
        for field in dataclasses.fields(val_to_convert):
            field_name = field.name
            field_path = f"{current_path}.{field_name}" if current_path else field_name

            try:
                field_value = getattr(val_to_convert, field_name)
                converted_value = dataclass_to_dict_for_omegaconf(field_value, field_path)

                # Only include non-None values (excludes callables)
                if converted_value is not None:
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
            converted_item = dataclass_to_dict_for_omegaconf(item, item_path)

            # Only include non-None values
            if converted_item is not None:
                result.append(converted_item)

        return result

    # Handle tuples
    elif isinstance(val_to_convert, tuple):
        converted_items = []
        for i, item in enumerate(val_to_convert):
            item_path = f"{current_path}[{i}]"
            converted_item = dataclass_to_dict_for_omegaconf(item, item_path)

            # Only include non-None values
            if converted_item is not None:
                converted_items.append(converted_item)

        return tuple(converted_items)

    # Handle dictionaries
    elif isinstance(val_to_convert, dict):
        result = {}
        for key, value in val_to_convert.items():
            key_path = f"{current_path}.{key}" if current_path else str(key)
            converted_value = dataclass_to_dict_for_omegaconf(value, key_path)

            # Only include non-None values
            if converted_value is not None:
                result[key] = converted_value

        return result

    # Return primitive types as-is
    else:
        return val_to_convert


def _track_excluded_fields(obj: Any, path: str = "") -> Dict[str, Any]:
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

            if is_problematic_callable(field_value):
                excluded_fields[field_path] = field_value
                logger.debug(f"Tracking excluded callable: {field_path}")
            elif dataclasses.is_dataclass(field_value):
                nested_excluded = _track_excluded_fields(field_value, field_path)
                excluded_fields.update(nested_excluded)
            elif isinstance(field_value, dict):
                for key, value in field_value.items():
                    if is_problematic_callable(value):
                        excluded_fields[f"{field_path}.{key}"] = value

    return excluded_fields


def _restore_excluded_fields(config_obj: Any, excluded_fields: Dict[str, Any]) -> None:
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
            path_parts = field_path.split('.')
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
    if is_problematic_callable(obj):
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


def safe_create_omegaconf(config_container: Any) -> DictConfig:
    """Safely create OmegaConf from dataclass, with comprehensive error handling.

    This function converts a dataclass to OmegaConf while excluding problematic
    callable fields. It provides detailed error reporting and debugging information
    if the conversion fails.

    Args:
        config_container: The dataclass instance to convert

    Returns:
        OmegaConf DictConfig created from the input dataclass

    Raises:
        ValueError: If the conversion fails or problematic callables are found
    """
    logger.info("Starting safe OmegaConf conversion...")

    # Step 1: Convert dataclass to dictionary
    logger.info("Converting dataclass to dictionary (excluding callables)...")
    base_dict = dataclass_to_dict_for_omegaconf(config_container, "root")

    if base_dict is None:
        raise ValueError("Root configuration object was excluded (likely a callable)")

    # Step 2: Verify no callables remain
    logger.info("Verifying no problematic callables remain...")
    if not _verify_no_callables(base_dict, "root"):
        raise ValueError("Callable objects found in converted dictionary")

    # Step 3: Create OmegaConf
    logger.info("Creating OmegaConf from clean dictionary...")
    try:
        omega_conf = OmegaConf.create(base_dict)
        logger.info("✓ Successfully created OmegaConf configuration")
        return omega_conf

    except Exception as e:
        logger.error(f"Failed to create OmegaConf even after cleaning: {e}")

        # Additional debugging: try to identify the exact problematic field
        if hasattr(e, 'full_key'):
            logger.error(f"Problematic key: {e.full_key}")

        # Try creating smaller pieces to isolate the issue
        logger.info("Attempting to isolate problematic section...")
        if isinstance(base_dict, dict):
            for key, value in base_dict.items():
                try:
                    test_dict = {key: value}
                    test_omega = OmegaConf.create(test_dict)
                    logger.info(f"  ✓ Section '{key}' is OK")
                except Exception as section_error:
                    logger.error(f"  ✗ Section '{key}' failed: {section_error}")
                    if hasattr(section_error, 'full_key'):
                        logger.error(f"    Full key: {section_error.full_key}")

        raise


def safe_create_omegaconf_with_preservation(config_container: Any) -> Tuple[DictConfig, Dict[str, Any]]:
    """Create OmegaConf while tracking excluded callables for later restoration.

    This function combines the conversion to OmegaConf with tracking of excluded
    callable fields, allowing them to be restored after override processing.

    Args:
        config_container: The dataclass instance to convert

    Returns:
        Tuple of (OmegaConf DictConfig, excluded callables dictionary)

    Raises:
        ValueError: If the conversion fails
    """
    logger.info("Starting safe OmegaConf conversion with callable preservation...")

    # Step 1: Track all callable fields that will be excluded
    excluded_callables = _track_excluded_fields(config_container, "root")
    logger.info(f"Found {len(excluded_callables)} callable fields to preserve")

    # Step 2: Convert to OmegaConf (excluding callables)
    base_dict = dataclass_to_dict_for_omegaconf(config_container, "root")

    if base_dict is None:
        raise ValueError("Root configuration object was excluded (likely a callable)")

    # Step 3: Verify no callables remain
    if not _verify_no_callables(base_dict, "root"):
        raise ValueError("Callable objects found in converted dictionary")

    # Step 4: Create OmegaConf
    omega_conf = OmegaConf.create(base_dict)

    return omega_conf, excluded_callables


def apply_overrides_recursively(config_obj: DataclassInstance, overrides_dict: Dict[str, Any]) -> None:
    """Recursively apply overrides from a Python dictionary to a dataclass instance.

    This function traverses nested dataclass structures and applies override values
    from a dictionary. It handles type conversions for special cases like torch.dtype.

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

        current_attr: Any = getattr(config_obj, key)

        if isinstance(value, dict) and dataclasses.is_dataclass(current_attr) and not isinstance(current_attr, type):
            apply_overrides_recursively(current_attr, value)
        else:
            try:
                # Handle special case conversions if needed
                final_value = value

                # If the original was a torch.dtype and value is a string, convert back
                if isinstance(current_attr, torch.dtype) and isinstance(value, str):
                    try:
                        final_value = getattr(torch, value.split('.')[-1])
                    except AttributeError:
                        logger.warning(f"Could not convert string '{value}' back to torch.dtype")
                        final_value = value

                setattr(config_obj, key, final_value)
                logger.debug(f"Set {type(config_obj).__name__}.{key} = {final_value}")

            except Exception as e:
                logger.warning(
                    f"Could not set attribute {type(config_obj).__name__}.{key} to value '{value}'. Error: {e}"
                )


def apply_overrides_with_preservation(
    config_obj: DataclassInstance, overrides_dict: Dict[str, Any], excluded_callables: Dict[str, Any]
) -> None:
    """Apply overrides while preserving excluded callable fields.

    This function first applies the overrides using the standard recursive approach,
    then restores the callable fields that were excluded during OmegaConf conversion.

    Args:
        config_obj: The dataclass instance to modify
        overrides_dict: Dictionary of override values to apply
        excluded_callables: Dictionary of excluded callable fields to restore
    """
    # Step 1: Apply normal overrides
    apply_overrides_recursively(config_obj, overrides_dict)

    # Step 2: Restore excluded callable fields
    _restore_excluded_fields(config_obj, excluded_callables)

    logger.info("Configuration updated with overrides and callable fields preserved")


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
