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

"""Serializable Nemotron VL config and LLaVA builder."""

import copy
from dataclasses import dataclass
from typing import ClassVar

from megatron.core.activations import fast_gelu
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import get_submodules
from megatron.training.models.hybrid import HybridModelBuilder

from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel
from megatron.bridge.models.nemotronh.model_config import NemotronHModelConfig


@dataclass(kw_only=True)
class NemotronVLModelConfig(NemotronHModelConfig):
    """Pure-data Nemotron VL build configuration."""

    builder: ClassVar[str] = "megatron.bridge.models.nemotron_vl.model_config.NemotronVLModelBuilder"
    image_token_index: int = 131072
    tokenizer_type: str = "nemotron-h-5p5-reasoning"
    vision_proj_ffn_hidden_size: int = 20480
    dynamic_resolution: bool = False
    use_vision_backbone_fp8_arch: bool = True
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    scatter_embedding_sequence_parallel: bool = False


class NemotronVLModelBuilder(HybridModelBuilder):
    """Assemble Nemotron's hybrid language model and RADIO LLaVA stack."""

    @staticmethod
    def _build_vision_config(language_cfg):
        """Create the RADIO vision config from the language config."""
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
        vision_cfg.mtp_num_layers = None
        vision_cfg.num_layers = 32
        vision_cfg.num_attention_heads = 16
        vision_cfg.add_bias_linear = True
        vision_cfg.add_qkv_bias = True
        vision_cfg.hidden_size = 1280
        vision_cfg.ffn_hidden_size = 5120
        vision_cfg.gated_linear_unit = False
        vision_cfg.activation_func = fast_gelu
        vision_cfg.kv_channels = 80
        vision_cfg.num_query_groups = 16
        vision_cfg.layernorm_zero_centered_gamma = False
        vision_cfg.apply_query_key_layer_scaling = False
        vision_cfg.attention_softmax_in_fp32 = True
        vision_cfg.normalization = "LayerNorm"
        vision_cfg.qk_layernorm = False
        vision_cfg.layernorm_epsilon = 1e-6
        return vision_cfg

    @staticmethod
    def _build_projection_config(language_cfg, hidden_size: int):
        """Create a non-parallel multimodal projection config."""
        projection_cfg = copy.deepcopy(language_cfg)
        projection_cfg.sequence_parallel = False
        projection_cfg.context_parallel_size = 1
        projection_cfg.tp_comm_overlap = False
        projection_cfg.recompute_granularity = None
        projection_cfg.recompute_method = None
        projection_cfg.recompute_num_layers = None
        projection_cfg.ffn_hidden_size = hidden_size
        projection_cfg.bias_activation_fusion = False
        return projection_cfg

    @staticmethod
    def _projection_spec():
        """Clone the hybrid language MLP submodules for a projector."""
        language_submodules = get_submodules(hybrid_stack_spec)
        mlp_layer_submodules = get_submodules(language_submodules.mlp_layer)
        return copy.deepcopy(get_submodules(mlp_layer_submodules.mlp))

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> NemotronVLModel:
        """Build one Nemotron VL pipeline stage."""
        config = self._model_config
        language_cfg = config.transformer
        vision_cfg = self._build_vision_config(language_cfg)
        projection_cfg = self._build_projection_config(language_cfg, config.vision_proj_ffn_hidden_size)
        projection_spec = self._projection_spec()
        add_encoder = pre_process if pre_process is not None else True

        llava_model = LLaVAModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=hybrid_stack_spec,
            language_vocab_size=config.vocab_size,
            language_max_sequence_length=config.seq_length,
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=get_vit_layer_with_transformer_engine_spec(),
            drop_vision_class_token=True,
            vision_projection_config=projection_cfg,
            vision_projection_layer_spec=projection_spec,
            vision_projection_type="mlp",
            parallel_output=config.parallel_output,
            share_embeddings_and_output_weights=config.share_embeddings_and_output_weights,
            language_position_embedding_type=config.position_embedding_type,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            add_encoder=add_encoder,
            add_decoder=True,
            img_h=512,
            img_w=512,
            patch_dim=16,
            hybrid_layer_pattern=config.hybrid_layer_pattern,
            image_token_index=config.image_token_index,
            pixel_shuffle=True,
            max_num_tiles=12,
            tokenizer_type=config.tokenizer_type,
            use_vision_backbone_fp8_arch=config.use_vision_backbone_fp8_arch,
            dynamic_resolution=config.dynamic_resolution,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        model = NemotronVLModel(llava_model=llava_model, config=language_cfg)
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        return model


__all__ = ["NemotronVLModelBuilder", "NemotronVLModelConfig"]
