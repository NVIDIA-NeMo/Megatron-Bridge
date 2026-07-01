# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Lazy, model-specific compatibility checks for Hugging Face Transformers."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

from packaging.version import InvalidVersion, Version


class TransformersVersionError(RuntimeError):
    """Raised when a selected model is incompatible with installed Transformers.

    Args:
        model_name: User-facing model or architecture name.
        installed_version: Installed Transformers version.
        required_version: Minimum Transformers version required by the model.
        missing_symbols: Required Transformers symbols that are unavailable.
        action: Operation that triggered the compatibility check.
    """

    def __init__(
        self,
        model_name: str,
        installed_version: Version,
        required_version: Version,
        *,
        missing_symbols: tuple[str, ...] = (),
        action: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.installed_version = installed_version
        self.required_version = required_version
        self.missing_symbols = missing_symbols
        self.action = action

        operation = action or "use this model"
        details = [
            f"Cannot {operation} for {model_name} with Transformers {installed_version}.",
            f"{model_name} requires Transformers>={required_version}.",
        ]
        if missing_symbols:
            details.append(f"Missing required symbol(s): {', '.join(missing_symbols)}.")
        details.append(
            f"Install or upgrade to a compatible Transformers version (>={required_version}) before retrying."
        )
        super().__init__(" ".join(details))


def get_transformers_version() -> Version:
    """Return the installed Transformers version as a PEP 440 version."""
    try:
        installed_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("Transformers is not installed.") from error

    try:
        return Version(installed_version)
    except InvalidVersion as error:
        raise RuntimeError(f"Installed Transformers has an invalid version: {installed_version!r}.") from error


def _parse_required_version(version: str) -> Version:
    try:
        return Version(version)
    except InvalidVersion as error:
        raise ValueError(f"Invalid minimum Transformers version: {version!r}.") from error


def is_transformers_min_version(version: str) -> bool:
    """Return whether installed Transformers satisfies ``version``."""
    return get_transformers_version() >= _parse_required_version(version)


def _module_not_found_targets(module_name: str, error: ModuleNotFoundError) -> bool:
    missing_name = error.name
    return missing_name is not None and (missing_name == module_name or module_name.startswith(f"{missing_name}."))


def _has_transformers_symbol(symbol_path: str) -> bool:
    """Resolve a dotted module/attribute path without hiding unrelated import failures."""
    if symbol_path != "transformers" and not symbol_path.startswith("transformers."):
        raise ValueError(f"Transformers symbol paths must start with 'transformers': {symbol_path!r}.")

    parts = symbol_path.split(".")
    module_name = None
    attribute_parts: list[str] = []
    for split_index in range(len(parts), 0, -1):
        candidate = ".".join(parts[:split_index])
        try:
            spec = importlib.util.find_spec(candidate)
        except ModuleNotFoundError as error:
            if not _module_not_found_targets(candidate, error):
                raise
            continue
        except (AttributeError, ValueError):
            continue
        if spec is not None:
            module_name = candidate
            attribute_parts = parts[split_index:]
            break

    if module_name is None:
        return False

    try:
        resolved: object = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if _module_not_found_targets(module_name, error):
            return False
        raise

    for attribute_name in attribute_parts:
        try:
            resolved = getattr(resolved, attribute_name)
        except AttributeError:
            return False
    return True


def require_transformers_version(
    model_name: str,
    min_version: str,
    *,
    symbols: tuple[str, ...] = (),
    action: str | None = None,
) -> None:
    """Require a Transformers version and optional symbols for one model.

    Symbol imports are attempted only after the minimum version is satisfied, so
    selecting a model on an older installation fails without importing its
    version-sensitive Transformers modules.

    Args:
        model_name: User-facing model or architecture name.
        min_version: Minimum compatible Transformers version.
        symbols: Dotted Transformers module or attribute paths required by the model.
        action: Operation that triggered the compatibility check.

    Raises:
        TransformersVersionError: If the version is too old or a required symbol is missing.
        ValueError: If ``min_version`` or a symbol path is invalid.
        ImportError: If resolving a symbol fails because of an unrelated dependency error.
    """
    installed_version = get_transformers_version()
    required_version = _parse_required_version(min_version)
    if installed_version < required_version:
        raise TransformersVersionError(
            model_name,
            installed_version,
            required_version,
            action=action,
        )

    missing_symbols = tuple(symbol for symbol in symbols if not _has_transformers_symbol(symbol))
    if missing_symbols:
        raise TransformersVersionError(
            model_name,
            installed_version,
            required_version,
            missing_symbols=missing_symbols,
            action=action,
        )
