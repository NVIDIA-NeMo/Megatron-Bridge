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

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.beam.beam_config import BeamTransformerConfig, beam_pipeline_layer_pattern
from megatron.bridge.models.beam.modeling_beam import BeamOutputLayer, beam_hybrid_stack_spec
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider


@dataclass
class BeamModelProvider(HybridModelProvider, BeamTransformerConfig):
    """Legacy provider compatibility using the same layers and validation as the Beam builder.

    The flattened dataclass is serializable by the provider checkpoint path.
    New integrations should use BeamModelConfig and BeamModelBuilder.
    """

    def finalize(self) -> None:
        """Split pipeline stages at Beam-layer boundaries and validate the architecture."""
        if self.virtual_pipeline_model_parallel_size is not None:
            raise ValueError("Beam does not support virtual pipeline parallelism.")
        if self.hybrid_layer_pattern and (
            self.num_layers_in_first_pipeline_stage is None and self.num_layers_in_last_pipeline_stage is None
        ):
            self.hybrid_layer_pattern = beam_pipeline_layer_pattern(
                self.hybrid_layer_pattern, self.pipeline_model_parallel_size
            )
        super().finalize()

    def _resolve_hybrid_stack_spec(self) -> ModuleSpec:
        # Generic Hybrid specs omit Beam's sandwich norms, SSMax and fp32 router.
        if self.hybrid_stack_spec is not None:
            raise ValueError("BeamModelProvider supplies its own hybrid_stack_spec; custom specs are unsupported.")
        return beam_hybrid_stack_spec(self)

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build the Hybrid model with Beam layers and the final-logit soft cap."""
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
        if model.post_process and self.final_logit_softcapping:
            extend_instance(model.output_layer, BeamOutputLayer)
        return model
