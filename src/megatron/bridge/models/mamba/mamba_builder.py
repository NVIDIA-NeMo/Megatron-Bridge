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

from dataclasses import dataclass
from typing import ClassVar

from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.hybrid.hybrid_builder import (
    HybridModelBuilder,
    HybridModelConfig,
    get_default_hybrid_stack_spec,
    modelopt_hybrid_stack_spec,
    transformer_engine_hybrid_stack_spec,
)


@dataclass(kw_only=True)
class MambaModelConfig(HybridModelConfig):
    """Backward-compatible wrapper around :class:`HybridModelConfig`."""

    builder: ClassVar[str] = "megatron.bridge.models.mamba.MambaModelBuilder"


class MambaModelBuilder(HybridModelBuilder):
    """Backward-compatible wrapper around :class:`HybridModelBuilder`."""


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Backward-compatible alias for ``transformer_engine_hybrid_stack_spec``."""
    return transformer_engine_hybrid_stack_spec()


def modelopt_mamba_stack_spec(config: "MambaModelConfig | None" = None) -> ModuleSpec:
    """Backward-compatible alias for ``modelopt_hybrid_stack_spec``."""
    return modelopt_hybrid_stack_spec(config)


def get_default_mamba_stack_spec(config: "MambaModelConfig") -> ModuleSpec:
    """Backward-compatible alias for ``get_default_hybrid_stack_spec``."""
    return get_default_hybrid_stack_spec(config)
