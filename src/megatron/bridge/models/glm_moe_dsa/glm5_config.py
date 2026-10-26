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

"""Builder-backed model config for the GLM-5 DSA family."""

from dataclasses import dataclass
from typing import ClassVar

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.glm_moe_dsa.glm5_hybrid import glm_hybrid_stack_spec
from megatron.bridge.models.glm_moe_dsa.glm5_provider import finalize_glm_hybrid_pattern
from megatron.bridge.models.hybrid import BridgeHybridModelBuilder
from megatron.bridge.models.hybrid.model_config import BridgeHybridMLAModelConfig


@dataclass(kw_only=True)
class GLM5ModelConfig(BridgeHybridMLAModelConfig):
    """GLM-5 as DSA, dense and MoE Hybrid layers with MLA, the builder counterpart of ``GLM5ModelProvider``.

    ``finalize()`` places DSA-aligned pipeline boundaries in ``hybrid_layer_pattern`` and derives the
    MTP pattern and MoE layer schedule, as the provider does, before the Hybrid finalization.
    """

    builder: ClassVar[str] = "megatron.bridge.models.glm_moe_dsa.glm5_config.GLM5ModelBuilder"

    def finalize(self) -> None:
        """Validate the physical layout and preserve DSA sharing within each pipeline stage."""
        finalize_glm_hybrid_pattern(self)
        super().finalize()


class GLM5ModelBuilder(BridgeHybridModelBuilder):
    """Build GLM-5 stages from GLM's Hybrid stack spec, as ``GLM5ModelProvider`` does.

    The spec isolates DSA index-sharing state per forward and replays a whole stage on
    recomputation. Like the provider, the builder uses it regardless of ``hybrid_stack_spec``.
    """

    def _resolve_hybrid_stack_spec(self) -> ModuleSpec:
        """Return GLM's Hybrid stack spec."""
        return glm_hybrid_stack_spec(self._model_config)


__all__ = ["GLM5ModelBuilder", "GLM5ModelConfig"]
