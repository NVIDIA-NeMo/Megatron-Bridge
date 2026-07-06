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

"""Helpers for recipe unit tests."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator
from types import ModuleType

import pytest


def _iter_recipe_owner_modules(module: ModuleType) -> Iterator[ModuleType]:
    """Yield modules that define recipe functions exported by a compatibility module."""
    exported_names = getattr(module, "__all__", ())
    for name in exported_names:
        value = getattr(module, name, None)
        if callable(value) and hasattr(value, "__module__"):
            yield importlib.import_module(value.__module__)


def patch_recipe_module_global(
    monkeypatch: pytest.MonkeyPatch,
    module_or_recipe_func: ModuleType | Callable[..., object],
    name: str,
    value: object,
) -> None:
    """Patch a global on the module that owns a recipe function.

    Compatibility recipe modules re-export canonical H100 recipe functions with
    direct import aliases. Tests should patch the canonical function owner rather
    than the compatibility module, because patching the compatibility module no
    longer affects the aliased function body.
    """
    if callable(module_or_recipe_func) and hasattr(module_or_recipe_func, "__module__"):
        owner = importlib.import_module(module_or_recipe_func.__module__)
        monkeypatch.setattr(owner, name, value)
        return

    module = module_or_recipe_func
    patched_ids: set[int] = set()
    if hasattr(module, name):
        monkeypatch.setattr(module, name, value)
        patched_ids.add(id(module))
    for owner in _iter_recipe_owner_modules(module):
        if hasattr(owner, name) and id(owner) not in patched_ids:
            monkeypatch.setattr(owner, name, value)
            patched_ids.add(id(owner))
