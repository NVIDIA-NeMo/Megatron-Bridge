# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Deprecated provider entry point using the canonical native Kimi stack."""

from dataclasses import dataclass, field
from typing import Callable

from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.kimi.kimi_k3_hybrid_config import kimi_k3_hybrid_stack_spec, kimi_k3_pipeline_pattern
from megatron.bridge.models.kimi.kimi_k3_provider import KimiK3ModelProvider


@dataclass
class KimiK3HybridModelProvider(HybridModelProvider, KimiK3ModelProvider):
    """Build native Kimi K3 with the same stack specification as the model config."""

    hybrid_stack_spec: ModuleSpec | Callable = field(default_factory=lambda: kimi_k3_hybrid_stack_spec)

    pipeline_block_counts: tuple[int, ...] | None = None

    def finalize(self) -> None:
        """Use the canonical complete-block pipeline partition."""
        self.hybrid_layer_pattern = kimi_k3_pipeline_pattern(
            self.hybrid_layer_pattern,
            pipeline_size=self.pipeline_model_parallel_size,
            blocks_per_stage=self.pipeline_block_counts,
        )
        super().finalize()
