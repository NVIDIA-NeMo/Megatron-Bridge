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

"""Provider for DeepSeek-V4 expressed as a Megatron-Core ``HybridModel``.

DeepSeek-V4 is a Multi-Latent-Attention (MLA) model, so it needs every MLA
configuration field (``q_lora_rank``, ``output_projection_groups``,
``v_head_dim``, ``rope_type`` / YaRN parameters, …). Those live on
:class:`MLAModelProvider`.
It is also a *hybrid* model: each logical DeepSeek-V4 block is expressed as two
Megatron hybrid layers — an attention-only layer (``W``/``C``/``H`` symbol) and
a MoE-only layer (``E`` symbol) — driven by ``hybrid_layer_pattern`` and built
by MCore's standard hybrid stack. The hybrid ``provide()``/``finalize()`` logic
lives on :class:`HybridModelProvider`.

This provider combines both. :class:`HybridModelProvider` is listed first so its
``provide()`` (which instantiates :class:`~megatron.core.models.hybrid.hybrid_model.HybridModel`)
and ``finalize()`` (which derives ``num_layers`` from ``hybrid_layer_pattern``)
win over the GPT-model versions inherited via :class:`MLAModelProvider`, while
``issubclass(DeepSeekV4HybridModelProvider, MLAModelProvider)`` stays ``True`` so
:meth:`MegatronModelBridge.provider_bridge` keeps mapping the HF config through
the direct MLA field names.
"""

from dataclasses import dataclass

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.bridge.models.mla_provider import MLAModelProvider


@dataclass
class DeepSeekV4HybridModelProvider(HybridModelProvider, MLAModelProvider):
    """MLA-capable :class:`HybridModelProvider` for DeepSeek-V4.

    All configuration is supplied by :class:`DeepSeekV4Bridge.provider_bridge`;
    this class combines the MLA config and hybrid builder and keeps native
    pipeline segments aligned with public runner topology overrides.
    """

    def _pipeline_model_parallel_layout_builder(
        self, pipeline_model_parallel_size: int, virtual_pipeline_model_parallel_size: int | None
    ) -> list[list[str]] | None:
        """Keep the native layer pattern aligned with public runner PP/VPP overrides."""
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout

        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        set_deepseek_v4_pipeline_model_parallel_layout(self)
        return self.pipeline_model_parallel_layout
