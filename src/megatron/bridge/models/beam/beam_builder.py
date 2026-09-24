# SPDX-License-Identifier: Apache-2.0
"""Model builder for Beam: the GPT builder with the Beam layer spec and final-logit soft-capping."""

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.beam.beam_config import BeamModelConfig
from megatron.bridge.models.beam.modeling_beam import BeamOutputLayer, beam_decoder_block_spec
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder


def beam_layer_spec(model_config: BeamModelConfig, vp_stage: int | None = None):
    """The Beam decoder block spec, with the arguments of Megatron-Core's default GPT spec."""
    transformer = model_config.transformer
    return beam_decoder_block_spec(
        transformer,
        use_transformer_engine=HAVE_TE and transformer.transformer_impl == "transformer_engine",
        normalization=transformer.normalization,
        qk_l2_norm=transformer.qk_l2_norm,
        vp_stage=vp_stage,
    )


class BeamModelBuilder(GPTModelBuilder):
    """Build ``GPTModel`` stages from the Beam layer spec; soft-cap the output layer on the last one."""

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
    ) -> GPTModel:
        """Build one stage; the stage that owns ``output_layer`` gets the soft cap."""
        config = self._model_config
        restore_spec = config.transformer_layer_spec is None
        if restore_spec:
            config.transformer_layer_spec = beam_layer_spec
        try:
            model = super().build_model(pg_collection, pre_process, post_process, vp_stage)
        finally:
            if restore_spec:
                config.transformer_layer_spec = None
        if model.post_process and config.transformer.final_logit_softcapping:
            extend_instance(model.output_layer, BeamOutputLayer)
        return model
