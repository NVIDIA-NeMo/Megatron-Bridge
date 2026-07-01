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

"""Serializable Nemotron Omni config and independent multimodal builder."""

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar

import torch
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ProcessGroupCollection

from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder
from megatron.bridge.models.nemotron_vl.model_config import NemotronVLModelBuilder, NemotronVLModelConfig


@dataclass(kw_only=True)
class NemotronOmniModelConfig(NemotronVLModelConfig):
    """Pure VL and sound assembly inputs for Nemotron Omni."""

    builder: ClassVar[str] = "megatron.bridge.models.nemotron_omni.model_config.NemotronOmniModelBuilder"
    language_model_type: str = "nemotron6-moe"
    tokenizer_type: str = "nemotron6-moe"
    img_start_token_id: int = 21
    img_end_token_id: int = 22
    vision_class_token_len: int = 10
    has_sound: bool = False
    sound_model_type: str = "parakeet"
    sound_hidden_size: int = 1024
    sound_projection_hidden_size: int = 4096
    sound_context_token_id: int = 0
    sound_config: dict[str, object] = field(default_factory=dict)
    freeze_sound_encoder: bool = False
    freeze_sound_projection: bool = False
    temporal_patch_dim: int = 1
    separate_video_embedder: bool = False
    temporal_ckpt_compat: bool = False
    radio_force_eval_mode: bool = True
    radio_force_cpe_eval_mode: bool = True
    radio_interpolate_only_cpe: bool = True
    radio_cpe_aspect_ratio_select: bool = False
    radio_disable_cpe: bool = False


class NemotronOmniModelBuilder(NemotronVLModelBuilder):
    """Assemble vision, language, video, and sound modules before wrapping."""

    def build_model(
        self,
        pg_collection: ProcessGroupCollection,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> NemotronOmniModel:
        """Build one Nemotron Omni pipeline stage."""
        config = self._model_config
        assert isinstance(config, NemotronOmniModelConfig)
        language_cfg = copy.copy(config.transformer)
        # LLaVAModel dispatches between GPTModel and HybridModel using this
        # runtime family marker. Keep it out of the serialized MCore config.
        language_cfg.language_model_type = config.language_model_type
        vision_cfg = self._build_vision_config(language_cfg)
        vision_cfg.vision_model_type = config.vision_model_type
        vision_cfg.pipeline_model_parallel_size = 1
        vision_cfg.class_token_len = config.vision_class_token_len
        projection_cfg = self._build_projection_config(language_cfg, config.vision_proj_ffn_hidden_size)
        projection_cfg.pipeline_model_parallel_size = 1
        projection_cfg.activation_func = torch.nn.functional.relu
        projection_spec = self._projection_spec()
        add_encoder = pre_process if pre_process is not None else True

        sound_model = None
        sound_projection = None
        if config.has_sound and add_encoder:
            sound = config.sound_config
            sound_namespace = SimpleNamespace(
                hidden_size=sound["hidden_size"],
                num_hidden_layers=sound["num_hidden_layers"],
                num_attention_heads=sound["num_attention_heads"],
                intermediate_size=sound["intermediate_size"],
                num_mel_bins=sound["num_mel_bins"],
                subsampling_factor=sound["subsampling_factor"],
                conv_kernel_size=sound.get("conv_kernel_size", 9),
                use_bias=sound.get("convolution_bias", False),
                sound_model_type=config.sound_model_type,
                sound_pad_to_clip_duration=False,
                sound_batch_split=1,
            )
            sound_model = BridgeSoundEncoder(sound_namespace)
            sound_projection_cfg = self._build_projection_config(language_cfg, config.sound_projection_hidden_size)
            sound_projection_cfg.pipeline_model_parallel_size = 1
            sound_projection = MultimodalProjector(
                config=sound_projection_cfg,
                submodules=self._projection_spec(),
                projector_type="mlp",
                input_size=config.sound_hidden_size,
                pg_collection=pg_collection,
            )

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
            sound_model=sound_model,
            sound_projection=sound_projection,
            sound_token_index=config.sound_context_token_id,
            temporal_patch_dim=config.temporal_patch_dim,
            separate_video_embedder=config.separate_video_embedder,
            temporal_ckpt_compat=config.temporal_ckpt_compat,
            radio_force_eval_mode=config.radio_force_eval_mode,
            radio_force_cpe_eval_mode=config.radio_force_cpe_eval_mode,
            radio_interpolate_only_cpe=config.radio_interpolate_only_cpe,
            radio_cpe_aspect_ratio_select=config.radio_cpe_aspect_ratio_select,
            radio_disable_cpe=config.radio_disable_cpe,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        llava_model.img_start_token_id = config.img_start_token_id
        llava_model.img_end_token_id = config.img_end_token_id
        model = NemotronOmniModel(llava_model=llava_model, config=language_cfg)
        if config.freeze_language_model or config.freeze_vision_model or config.freeze_vision_projection:
            model.freeze(
                freeze_language_model=config.freeze_language_model,
                freeze_vision_model=config.freeze_vision_model,
                freeze_vision_projection=config.freeze_vision_projection,
            )
        if config.freeze_sound_encoder or config.freeze_sound_projection:
            model.freeze(
                freeze_sound_model=config.freeze_sound_encoder,
                freeze_sound_projection=config.freeze_sound_projection,
            )
        return model


__all__ = ["NemotronOmniModelBuilder", "NemotronOmniModelConfig"]
