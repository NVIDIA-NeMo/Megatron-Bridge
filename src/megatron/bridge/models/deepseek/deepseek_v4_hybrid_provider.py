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

from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.enums import LayerType

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
        """Rebuild this provider's pattern and layout after runner PP/VPP overrides.

        This hook mutates ``hybrid_layer_pattern`` as well as the PP/VP fields
        and ``pipeline_model_parallel_layout``. Existing pipe separators,
        including a recipe's uneven stage allocation, are replaced by an even
        split of logical attention/MoE pairs; the MTP suffix is preserved.
        The runner skips this hook when the CLI explicitly supplies a layout,
        so that caller must also supply matching native pattern segments;
        :meth:`finalize` rejects segments that do not match the topology.
        """
        from megatron.bridge.models.deepseek.deepseek_v4_bridge import set_deepseek_v4_pipeline_model_parallel_layout

        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
        set_deepseek_v4_pipeline_model_parallel_layout(self)
        return self.pipeline_model_parallel_layout

    def finalize(self) -> None:
        """Finalize the hybrid provider and reject stale native pipeline segments."""
        super().finalize()
        self._validate_native_pipeline_segments()

    def _validate_native_pipeline_segments(self) -> None:
        """Require one ``|`` segment per PP/VP stage, matching any explicit layout.

        MCore picks segment ``vp_stage * PP + pp_rank`` and only checks that the
        segment count is divisible by PP, so segments left over from an earlier
        topology can build only part of the decoder without an error: a PP4
        pattern used at PP1 builds only its first segment. A pattern without
        ``|`` keeps MCore's even split by PP.
        """
        main_pattern = (self.hybrid_layer_pattern or "").partition(Symbols.MTP_SEPARATOR)[0]
        segments = main_pattern.split(Symbols.PIPE)
        if len(segments) == 1:
            return

        pp_size = self.pipeline_model_parallel_size or 1
        vp_size = self.virtual_pipeline_model_parallel_size or 1
        if len(segments) != pp_size * vp_size:
            raise ValueError(
                f"DSv4 hybrid_layer_pattern has {len(segments)} pipeline segments, but "
                f"pipeline_model_parallel_size={pp_size} and virtual_pipeline_model_parallel_size="
                f"{self.virtual_pipeline_model_parallel_size} need {pp_size * vp_size}. Call "
                "set_deepseek_v4_pipeline_model_parallel_layout() after changing PP or VP, or supply "
                "a hybrid_layer_pattern with one '|' segment per stage."
            )

        # MCore's finalize has already converted any layout to PipelineParallelLayerLayout.
        layout = self.pipeline_model_parallel_layout
        if layout is None:
            return
        decoder_counts = [
            layout.layout[pp_rank][vp_rank].count(LayerType.decoder)
            for vp_rank in range(layout.virtual_pipeline_model_parallel_size)
            for pp_rank in range(layout.pipeline_model_parallel_size)
        ]
        segment_lengths = [len(segment) for segment in segments]
        if decoder_counts != segment_lengths:
            raise ValueError(
                f"pipeline_model_parallel_layout places {decoder_counts} decoder layers per stage, but the "
                f"DSv4 hybrid_layer_pattern segments hold {segment_lengths} layers. Supply both from "
                "set_deepseek_v4_pipeline_model_parallel_layout()."
            )
