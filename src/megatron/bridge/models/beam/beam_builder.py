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

"""Model builder for Beam: the Hybrid builder with the Beam stack spec and final-logit soft-capping."""

from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.beam.beam_config import BeamModelConfig
from megatron.bridge.models.beam.modeling_beam import BeamOutputLayer, beam_hybrid_stack_spec
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.hybrid.hybrid_builder import HybridModelBuilder


class BeamModelBuilder(HybridModelBuilder):
    """Build ``HybridModel`` stages from the Beam stack spec; soft-cap the output layer on the last one."""

    def __init__(self, model_config: BeamModelConfig) -> None:
        if not isinstance(model_config, BeamModelConfig):
            raise TypeError(f"Expected BeamModelConfig, got {type(model_config).__name__}.")
        super().__init__(model_config)

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> HybridModel:
        """Build one stage; the stage that owns ``output_layer`` gets the soft cap."""
        config = self._model_config
        restore_spec = config.hybrid_stack_spec is None
        if restore_spec:
            config.hybrid_stack_spec = beam_hybrid_stack_spec(config.transformer)
        try:
            model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        finally:
            if restore_spec:
                config.hybrid_stack_spec = None
        if model.post_process and config.transformer.final_logit_softcapping:
            extend_instance(model.output_layer, BeamOutputLayer)
        return model
