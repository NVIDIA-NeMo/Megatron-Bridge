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

from typing import Any

from megatron.core.transformer.enums import CudaGraphScope


try:
    from megatron.core.transformer.enums import CudaGraphModule
except ImportError:
    CudaGraphModule = None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        if not value or value == "full":
            return []
        return value.split(",")
    if isinstance(value, list):
        return value
    return [value]


def _member_name(value: Any) -> str:
    if isinstance(value, str):
        return value.rsplit(".", 1)[-1]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value).rsplit(".", 1)[-1]


def _member_name_list(value: Any) -> list[str]:
    return [_member_name(item) for item in _as_list(value)]


def _member_names(value: Any) -> set[str]:
    return set(_member_name_list(value))


def _supports_cuda_graph_modules(config: Any) -> bool:
    return CudaGraphModule is not None and hasattr(config, "cuda_graph_modules")


def _module_value(name: str):
    if CudaGraphModule is not None:
        return CudaGraphModule[name]
    return CudaGraphScope[name]


def cuda_graph_module_names(config: Any) -> list[str]:
    """Return configured per-layer CUDA graph module names."""

    if getattr(config, "cuda_graph_modules", None) is not None:
        return _member_name_list(getattr(config, "cuda_graph_modules"))
    names = _member_name_list(getattr(config, "cuda_graph_scope", None))
    return [name for name in names if name not in ("full_iteration", "full_iteration_inference")]


def set_cuda_graph_modules(config: Any, modules: Any) -> None:
    """Set per-layer CUDA graph modules using the current MCore API when available."""

    module_names = _member_name_list(modules)
    if _supports_cuda_graph_modules(config):
        config.cuda_graph_modules = [_module_value(name) for name in module_names]
        if hasattr(config, "cuda_graph_scope"):
            config.cuda_graph_scope = None
    else:
        config.cuda_graph_scope = [CudaGraphScope[name] for name in module_names]


def clear_cuda_graph_modules(config: Any) -> None:
    """Clear per-layer CUDA graph modules using the active MCore API."""

    set_cuda_graph_modules(config, [])


def set_full_iteration_cuda_graph(config: Any) -> None:
    """Enable full-iteration CUDA graph capture using the current MCore API."""

    if _supports_cuda_graph_modules(config):
        config.cuda_graph_impl = "full_iteration"
        config.cuda_graph_modules = []
        if hasattr(config, "cuda_graph_scope"):
            config.cuda_graph_scope = None
    else:
        config.cuda_graph_impl = "local"
        config.cuda_graph_scope = [CudaGraphScope.full_iteration]


def has_cuda_graph_module(config: Any, module: Any) -> bool:
    """Return whether a per-layer CUDA graph module is enabled.

    Supports both the current MCore ``cuda_graph_modules`` API and the deprecated
    ``cuda_graph_scope`` values still present in older Bridge configs.
    """

    module_name = _member_name(module)
    module_names = _member_names(getattr(config, "cuda_graph_modules", None))
    legacy_scope_names = _member_names(getattr(config, "cuda_graph_scope", None))
    return module_name in module_names or module_name in legacy_scope_names


def is_full_iteration_cuda_graph(config: Any) -> bool:
    """Return whether config enables full-iteration CUDA graph capture."""

    cuda_graph_impl = getattr(config, "cuda_graph_impl", "none")
    if cuda_graph_impl == "full_iteration":
        return True
    if cuda_graph_impl != "local":
        return False
    return "full_iteration" in _member_names(getattr(config, "cuda_graph_scope", None))


def uses_local_cuda_graph_manager(config: Any) -> bool:
    """Return whether Bridge should create a local MCore CudaGraphManager."""

    return getattr(config, "cuda_graph_impl", "none") == "local" and not is_full_iteration_cuda_graph(config)
