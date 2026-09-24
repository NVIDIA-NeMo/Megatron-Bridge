# SPDX-License-Identifier: Apache-2.0
"""Beam model provider: the Beam layer spec and fp32 final-logit soft-capping."""

from dataclasses import dataclass
from functools import partial

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.pipeline_parallel.utils import is_pp_last_stage, is_vp_last_stage

from megatron.bridge.models.beam.beam_config import BeamArchitectureConfig
from megatron.bridge.models.beam.modeling_beam import BeamOutputLayer, beam_decoder_block_spec
from megatron.bridge.models.gemma.modules import extend_instance
from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class BeamModelProvider(BeamArchitectureConfig, GPTModelProvider):
    """``GPTModelProvider`` with the Beam architecture fields and layer spec.

    ``final_logit_softcapping`` mirrors ``BeamConfig.final_logit_softcapping``.
    ``None`` or ``0`` disables the cap.
    """

    transformer_layer_spec: object = partial(beam_decoder_block_spec, use_transformer_engine=HAVE_TE)

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """Build the GPT model and, on the last stage, soft-cap ``output_layer``."""
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if (
            self.final_logit_softcapping
            and is_vp_last_stage(vp_stage=vp_stage, vp_size=self.virtual_pipeline_model_parallel_size)
            and is_pp_last_stage(self._pg_collection.pp)
        ):
            extend_instance(model.output_layer, BeamOutputLayer)

        return model
