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

"""Library recipes with explicit performance benchmark overrides.

Existing benchmark names select a library recipe and model-specific overrides.
Launchers read the literal selector tables without importing this package.
"""

from __future__ import annotations

import ast
import importlib
from collections.abc import Callable
from functools import lru_cache, partial
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from megatron.bridge.training.config import ConfigContainer


@lru_cache(maxsize=None)
def benchmark_names(family: str | None = None) -> tuple[str, ...]:
    """List benchmark selectors without importing model or CUDA dependencies."""
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) if family is None else [root / f"{family}.py"]
    names: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "LIBRARY_RECIPES" for target in node.targets
            ):
                assert isinstance(node.value, ast.Dict)
                for key in node.value.keys:
                    assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                    names.append(key.value)
    return tuple(sorted(names))


@lru_cache(maxsize=None)
def benchmark_family(name: str) -> str:
    """Resolve a benchmark selector to its model-specific override module."""
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if name in benchmark_names(path.stem):
            return path.stem
    raise ValueError(f"Unknown performance benchmark overrides: {name}")


def build_perf_benchmark_config(name: str, *, num_gpus: int | None = None) -> ConfigContainer:
    """Construct a library config and apply the selected benchmark overrides.

    Args:
        name: Existing exact benchmark selector, including hardware and precision.
        num_gpus: Actual allocation for the allocation-aware Nemotron Ultra variants.
            Other variants retain their fixed allocation and reject this argument.

    Returns:
        A fresh ConfigContainer with the original benchmark's effective settings.
    """
    family = importlib.import_module(f"{__name__}.{benchmark_family(name)}")
    kwargs = {}
    if num_gpus is not None:
        if name not in family.ALLOCATION_AWARE:
            raise TypeError(f"{name} does not accept num_gpus")
        kwargs["num_gpus"] = num_gpus
    from megatron.bridge.training.config import ConfigContainer

    cfg = family.LIBRARY_RECIPES[name]()
    assert isinstance(cfg, ConfigContainer)
    family.apply_overrides(cfg, name, **kwargs)
    return cfg


def get_perf_benchmark_recipe(name: str) -> Callable[..., ConfigContainer] | None:
    """Resolve an existing benchmark name to its library-plus-overrides constructor."""
    if name not in benchmark_names():
        return None
    return partial(build_perf_benchmark_config, name)
