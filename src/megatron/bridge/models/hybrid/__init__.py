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

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider


_LAZY_EXPORTS = {
    "HybridModelBuilder": "megatron.bridge.models.hybrid.hybrid_builder",
    "HybridModelConfig": "megatron.bridge.models.hybrid.hybrid_builder",
    "get_default_hybrid_stack_spec": "megatron.bridge.models.hybrid.hybrid_builder",
    "modelopt_hybrid_stack_spec": "megatron.bridge.models.hybrid.hybrid_builder",
    "transformer_engine_hybrid_stack_spec": "megatron.bridge.models.hybrid.hybrid_builder",
}

__all__ = [
    "HybridModelBuilder",
    "HybridModelConfig",
    "HybridModelProvider",
    "get_default_hybrid_stack_spec",
    "modelopt_hybrid_stack_spec",
    "transformer_engine_hybrid_stack_spec",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        module = importlib.import_module(_LAZY_EXPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
