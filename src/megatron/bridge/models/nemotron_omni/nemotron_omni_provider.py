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

import copy
from abc import ABC
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

from megatron.core import parallel_state
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder
from megatron.bridge.models.nemotron_vl.nemotron_vl_provider import (
    NemotronNano3Bv3VLModelProvider,
    NemotronNano12Bv2VLModelProvider,
    NemotronVLModelProvider,
)


@dataclass
class NemotronOmniModelProvider(NemotronVLModelProvider, ABC):
    """Base provider for Nemotron Omni (VL + sound) models.

    Extends NemotronVLModelProvider with sound-specific fields. When has_sound
    is False, behaves identically to the VL provider (backward compatible).
    """

    has_sound: bool = False
    sound_model_type: str = "parakeet"
    sound_hidden_size: int = 1024
    sound_projection_hidden_size: int = 4096
    sound_context_token_id: int = 0
    sound_config: Optional[dict] = None
    freeze_sound_encoder: bool = False
    freeze_sound_projection: bool = False

    def _build_sound_projection_config(self, language_cfg):
        """Build sound projection config (mirrors _build_vision_projection_config)."""
        sound_proj_cfg = copy.deepcopy(language_cfg)
        sound_proj_cfg.sequence_parallel = False
        sound_proj_cfg.context_parallel_size = 1
        sound_proj_cfg.tp_comm_overlap = False
        sound_proj_cfg.recompute_granularity = None
        sound_proj_cfg.recompute_method = None
        sound_proj_cfg.recompute_num_layers = None
        sound_proj_cfg.ffn_hidden_size = self.sound_projection_hidden_size
        sound_proj_cfg.bias_activation_fusion = False
        return sound_proj_cfg

    def _build_sound_encoder(self):
        """Build BridgeSoundEncoder from sound_config dict."""
        sc = self.sound_config
        config = SimpleNamespace(
            hidden_size=sc["hidden_size"],
            num_hidden_layers=sc["num_hidden_layers"],
            num_attention_heads=sc["num_attention_heads"],
            intermediate_size=sc["intermediate_size"],
            num_mel_bins=sc["num_mel_bins"],
            subsampling_factor=sc["subsampling_factor"],
            conv_kernel_size=sc.get("conv_kernel_size", 9),
            use_bias=sc.get("convolution_bias", False),
            sound_model_type=self.sound_model_type,
            sound_pad_to_clip_duration=True,
            sound_batch_split=1,
        )
        return BridgeSoundEncoder(config)

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Assemble NemotronOmniModel wrapping a LLaVAModel with optional sound support.

        Duplicates the VL provide() logic because LLaVAModel requires sound kwargs
        at construction time -- they can't be added after. This is intentional to
        maintain zero changes to nemotron_vl/.
        """
        language_cfg = copy.deepcopy(self)

        vision_cfg = self._build_vision_config(language_cfg)
        vision_proj_cfg = self._build_vision_projection_config(language_cfg)

        language_spec = mamba_stack_spec
        vision_spec = get_vit_layer_with_transformer_engine_spec()
        vision_proj_spec = copy.deepcopy(language_spec.submodules.mlp_layer.submodules.mlp.submodules)

        add_encoder_flag = (
            parallel_state.is_pipeline_first_stage()
            if self.pipeline_model_parallel_size > 1
            else True
        )
        add_decoder_flag = True

        # Build sound components (only on PP first stage, only when sound present)
        sound_model = None
        sound_projection = None
        sound_token_index = self.sound_context_token_id

        if self.has_sound and add_encoder_flag:
            sound_model = self._build_sound_encoder()

            sound_proj_cfg = self._build_sound_projection_config(language_cfg)
            sound_proj_spec = copy.deepcopy(language_spec.submodules.mlp_layer.submodules.mlp.submodules)
            sound_projection = MultimodalProjector(
                config=sound_proj_cfg,
                submodules=sound_proj_spec,
                projector_type="mlp",
                input_size=self.sound_hidden_size,
            )

        llava_model = LLaVAModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=language_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=vision_spec,
            drop_vision_class_token=True,
            vision_projection_config=vision_proj_cfg,
            vision_projection_layer_spec=vision_proj_spec,
            vision_projection_type="mlp",
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            language_position_embedding_type=self.position_embedding_type,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            add_encoder=add_encoder_flag,
            add_decoder=add_decoder_flag,
            img_h=512,
            img_w=512,
            patch_dim=16,
            hybrid_attention_ratio=0.0,
            hybrid_mlp_ratio=0.0,
            hybrid_override_pattern=self.hybrid_override_pattern,
            image_token_index=self.image_token_index,
            pixel_shuffle=True,
            dynamic_resolution=self.dynamic_resolution,
            max_num_tiles=12,
            tokenizer_type=self.tokenizer_type,
            use_vision_backbone_fp8_arch=self.use_vision_backbone_fp8_arch,
            radio_force_eval_mode=self.radio_force_eval_mode,
            radio_force_cpe_eval_mode=self.radio_force_cpe_eval_mode,
            radio_interpolate_only_cpe=self.radio_interpolate_only_cpe,
            radio_cpe_aspect_ratio_select=self.radio_cpe_aspect_ratio_select,
            radio_disable_cpe=self.radio_disable_cpe,
            video_temporal_patch_size=self.video_temporal_patch_size,
            separate_video_embedder=self.separate_video_embedder,
            sound_model=sound_model,
            sound_projection=sound_projection,
            sound_token_index=sound_token_index,
        )

        model = NemotronOmniModel(llava_model=llava_model)

        llava_model.img_start_token_id = self.img_start_token_id
        llava_model.img_end_token_id = self.img_end_token_id

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        if self.freeze_sound_encoder or self.freeze_sound_projection:
            model.freeze(
                freeze_sound_model=self.freeze_sound_encoder,
                freeze_sound_projection=self.freeze_sound_projection,
            )

        return model


@dataclass
class NemotronNano3Bv3OmniModelProvider(NemotronOmniModelProvider, NemotronNano3Bv3VLModelProvider):
    """Omni provider for Nemotron Nano Next 3B v3 (MoE)."""

    # Explicit overrides required: MRO puts NemotronOmniModelProvider (which
    # inherits base defaults from NemotronVLModelProvider) before
    # NemotronNano3Bv3VLModelProvider.  Without these, the base defaults
    # (e.g. image_token_index=0) silently win over the correct values.
    language_model_type: str = "nemotron6-moe"
    image_token_index: int = 18
    img_start_token_id: int = 19
    img_end_token_id: int = 20
    tokenizer_type: str = "nemotron6-moe"
    dynamic_resolution: bool = True


@dataclass
class NemotronNano12Bv2OmniModelProvider(NemotronOmniModelProvider, NemotronNano12Bv2VLModelProvider):
    """Omni provider for Nemotron Nano 12B v2 (dense)."""

    language_model_type: str = "nemotron5-hybrid-12b"
    image_token_index: int = 131072
    img_start_token_id: int = 131073
    img_end_token_id: int = 131074
    tokenizer_type: str = "nemotron-h-5p5-reasoning"
