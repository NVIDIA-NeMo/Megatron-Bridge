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

from dataclasses import dataclass, fields
from typing import Any, Callable

from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.hybrid.hybrid_provider import (
    HybridModelProvider,
    get_default_hybrid_stack_spec,
    modelopt_hybrid_stack_spec,
    transformer_engine_hybrid_stack_spec,
)


@dataclass
class MambaModelProvider(HybridModelProvider):
    """Backward-compatible wrapper around :class:`HybridModelProvider`."""

    mamba_stack_spec: ModuleSpec | Callable[[], ModuleSpec] | Callable[["MambaModelProvider"], ModuleSpec] | None = (
        None
    )

    def __post_init__(self) -> None:
        """Normalize the deprecated Mamba stack-spec field to the Hybrid field."""
        if self.hybrid_stack_spec is not None and self.mamba_stack_spec is not None:
            raise ValueError(
                "Cannot specify both hybrid_stack_spec and mamba_stack_spec. "
                "mamba_stack_spec has been deprecated; use hybrid_stack_spec instead."
            )
        if self.mamba_stack_spec is not None:
            self.hybrid_stack_spec = self.mamba_stack_spec
            self.mamba_stack_spec = None

    def to_cfg_dict(self) -> dict[str, Any]:
        """Serialize without the deprecated ``mamba_stack_spec`` field."""
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase

        result = {"_target_": f"{self.__class__.__module__}.{self.__class__.__qualname__}"}
        for field in fields(self):
            if field.name.startswith("_") or field.name == "mamba_stack_spec":
                continue
            result[field.name] = _ConfigContainerBase._convert_value_to_dict(getattr(self, field.name))
        return result


def modelopt_mamba_stack_spec(config: "MambaModelProvider | None" = None) -> ModuleSpec:
    """Backward-compatible alias for ``modelopt_hybrid_stack_spec``."""
    return modelopt_hybrid_stack_spec(config)


def transformer_engine_mamba_stack_spec() -> ModuleSpec:
    """Backward-compatible alias for ``transformer_engine_hybrid_stack_spec``."""
    return transformer_engine_hybrid_stack_spec()


def get_default_mamba_stack_spec(config: "MambaModelProvider") -> ModuleSpec:
    """Backward-compatible alias for ``get_default_hybrid_stack_spec``."""
    return get_default_hybrid_stack_spec(config)
