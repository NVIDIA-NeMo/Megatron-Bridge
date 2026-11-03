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

"""Model builder for Qwen3.5 and Qwen3.6 vision-language models."""

from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.gpt.gpt_builder import GPTModelBuilder, mtp_block_spec
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig
from megatron.bridge.models.qwen_vl.qwen35_vl_provider import (
    _patch_standard_attention_specs,
    _qwen35_build_language_block_spec,
)


class Qwen35VLModelBuilder(GPTModelBuilder):
    """Combine the Qwen3.5 hybrid language decoder with the Qwen3-VL vision encoder."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Qwen3VLModel:
        """Build one Qwen3.5-VL pipeline stage.

        Args:
            pg_collection: Process groups used for distributed construction.
            pre_process: Whether this stage owns the embeddings and the vision encoder.
            post_process: Whether this stage owns the language-model output layer.
            vp_stage: Optional virtual pipeline stage index.

        Returns:
            A Qwen3-VL model stage built from the Qwen3.5 language decoder.

        Raises:
            TypeError: If the builder was given another model config.
            ValueError: If ``vision_config`` is unset or ``transformer_layer_spec`` is set.
        """
        model_config = self._model_config
        if not isinstance(model_config, Qwen35VLModelConfig):
            raise TypeError(f"Expected Qwen35VLModelConfig, got {type(model_config).__name__}.")
        if model_config.vision_config is None:
            raise ValueError("Qwen3.5-VL vision_config must be configured before model construction.")
        if model_config.transformer_layer_spec is not None:
            raise ValueError(
                "Qwen3.5-VL builds its hybrid decoder spec from experimental_attention_variant; "
                "leave transformer_layer_spec unset."
            )

        vp_size = model_config.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pg_collection.pp)
        if post_process is None:
            post_process = is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pg_collection.pp)

        pp_rank = pg_collection.pp.rank()
        language_spec = _qwen35_build_language_block_spec(model_config.transformer, vp_stage, pp_rank=pp_rank)
        mtp_spec = mtp_block_spec(model_config, language_spec, vp_stage=vp_stage, pp_rank=pp_rank)
        _patch_standard_attention_specs(mtp_spec, Qwen3VLSelfAttention)

        vision_config = model_config.vision_config
        vision_config.torch_dtype = model_config.params_dtype
        model = Qwen3VLModel(
            language_transformer_config=model_config.transformer,
            language_transformer_layer_spec=language_spec,
            vision_transformer_config=vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            mtp_block_spec=mtp_spec,
            vp_stage=vp_stage,
            model_config=model_config,
        )
        if vp_stage is not None:
            model.vp_stage = vp_stage
        if model.vision_model is not None:
            # get_vision_model_config() leaves the vision encoder in fp32. The provider path relies on
            # Float16Module to cast it with the rest of the model; conversion builds without that wrapper.
            model.vision_model.to(dtype=model_config.params_dtype)

        if (
            model_config.freeze_language_model
            or model_config.freeze_vision_model
            or model_config.freeze_vision_projection
        ):
            model.freeze(
                freeze_language_model=model_config.freeze_language_model,
                freeze_vision_model=model_config.freeze_vision_model,
                freeze_vision_projection=model_config.freeze_vision_projection,
            )
        return model


__all__ = ["Qwen35VLModelBuilder"]
