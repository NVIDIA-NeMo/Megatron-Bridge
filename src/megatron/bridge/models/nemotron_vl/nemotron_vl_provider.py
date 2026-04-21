# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Optional

import torch
from megatron.core import parallel_state
from megatron.core.activations import fast_gelu
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

from megatron.bridge.models.nemotronh.nemotron_h_provider import (
    NemotronNanoModelProvider12Bv2,
    NemotronNanoModelProviderNext3Bv3,
)


@dataclass
class NemotronVLModelProvider(ABC):
    """Base configuration provider for Nemotron Vision-Language models.

    This base class contains common logic for all Nemotron VL variants.
    Subclasses should set model-specific parameters as class fields.
    """

    # For VL models we do *not* scatter embeddings across the sequence
    # parallel region because we need to splice vision embeddings later.
    scatter_embedding_sequence_parallel: bool = False
    attention_softmax_in_fp32: bool = True

    cuda_graph_impl: str = "none"

    vision_model_type: str = "radio"
    language_model_type: str = ""  # Set by subclasses
    generation_config: Optional[Any] = None

    # Freeze knobs useful for transfer-learning scenarios
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # Model-specific parameters (set by subclasses)
    vision_proj_ffn_hidden_size: int = 0  # Vision projection hidden size
    image_token_index: int = 0  # <image> token ID
    img_start_token_id: int = 0  # <img> wrapper token ID
    img_end_token_id: int = 0  # </img> wrapper token ID
    tokenizer_type: str = ""  # Tokenizer type
    use_vision_backbone_fp8_arch: bool = False  # Use FP8 architecture for vision backbone
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16

    # RADIO CPE configuration (defaults are backward compatible)
    radio_force_eval_mode: bool = False
    radio_force_cpe_eval_mode: bool = False
    radio_interpolate_only_cpe: bool = False
    radio_cpe_aspect_ratio_select: bool = False
    radio_disable_cpe: bool = False

    # Dynamic resolution for variable image sizes (set True for nano-v3-vl)
    dynamic_resolution: bool = False

    # Temporal compression: group T consecutive video frames into tubelets
    video_temporal_patch_size: int = 1
    separate_video_embedder: bool = False

    def _build_vision_config(self, language_cfg):
        """Build vision transformer config with RADIO model defaults."""
        vision_cfg = copy.deepcopy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False
        vision_cfg.recompute_granularity = None
        vision_cfg.recompute_method = None
        vision_cfg.recompute_num_layers = None
        # Overrides for vision_model_type = "radio"
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

    def _build_vision_projection_config(self, language_cfg):
        """Build vision projection config."""
        vision_proj_cfg = copy.deepcopy(language_cfg)
        vision_proj_cfg.sequence_parallel = False
        vision_proj_cfg.context_parallel_size = 1
        vision_proj_cfg.tp_comm_overlap = False
        vision_proj_cfg.recompute_granularity = None
        vision_proj_cfg.recompute_method = None
        vision_proj_cfg.recompute_num_layers = None
        vision_proj_cfg.ffn_hidden_size = self.vision_proj_ffn_hidden_size
        vision_proj_cfg.bias_activation_fusion = False
        return vision_proj_cfg

    def provide(self, pre_process=None, post_process=None, vp_stage=None):  # noqa: D401
        """Assemble a full :class:`~megatron.core.models.multimodal.llava_model.LLaVAModel` and wrap it.

        This is a *very* trimmed-down version of the assembly code used in
        `pretrain_vlm.py` – it relies only on parameters already stored in the
        provider so that it works in any script (no Megatron-training CLI
        required).
        """

        # ------------------------------------------------------------------
        # Build configs and layer specs
        # ------------------------------------------------------------------

        # Language config is basically *self*, but we make a shallow copy
        # so tweaks do not leak back.
        language_cfg = copy.deepcopy(self)

        # Vision transformer config – start from language_cfg but ensure SP/CP disabled
        vision_cfg = self._build_vision_config(language_cfg)

        # Vision-projection config/spec: a tiny two-layer MLP; for now just reuse
        # the MLP sub-modules from the language layer spec if available.
        vision_proj_cfg = self._build_vision_projection_config(language_cfg)

        language_spec = mamba_stack_spec
        vision_spec = get_vit_layer_with_transformer_engine_spec()
        vision_proj_spec = copy.deepcopy(language_spec.submodules.mlp_layer.submodules.mlp.submodules)

        # ------------------------------------------------------------------
        # Instantiate LLaVA
        # ------------------------------------------------------------------
        # For pipeline parallelism, the vision encoder should only be on the first stage
        add_encoder_flag = parallel_state.is_pipeline_first_stage() if self.pipeline_model_parallel_size > 1 else True
        add_decoder_flag = True
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
        )

        from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel

        model = NemotronVLModel(llava_model=llava_model)

        # Store wrapper token IDs for use in token collapsing (NeMo-RL)
        llava_model.img_start_token_id = self.img_start_token_id
        llava_model.img_end_token_id = self.img_end_token_id

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    # Alias that NemotronVLModel relies on to create the LM component
    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        """Provide the language model component only."""
        # Call the parent class's provide method (not this VL provide)
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class NemotronNano3Bv3VLModelProvider(NemotronVLModelProvider, NemotronNanoModelProviderNext3Bv3):
    """Configuration provider for Nemotron Nano Next 3B v3 VL model."""

    language_model_type: str = "nemotron6-moe"
    vision_proj_ffn_hidden_size: int = 20480
    image_token_index: int = 20  # <image>
    img_start_token_id: int = 21  # <img>
    img_end_token_id: int = 22  # </img>
    tokenizer_type: str = "nemotron6-moe"
    use_vision_backbone_fp8_arch: bool = False
    dynamic_resolution: bool = True

    def _build_vision_config(self, language_cfg):
        # RADIOv4/radio-so400m
        vision_cfg = super()._build_vision_config(language_cfg)
        vision_cfg.class_token_len = 10
        return vision_cfg


@dataclass
class NemotronNano12Bv2VLModelProvider(NemotronVLModelProvider, NemotronNanoModelProvider12Bv2):
    """Configuration provider for Nemotron Nano 12B v2 VL model."""

    language_model_type: str = "nemotron5-hybrid-12b"
    vision_proj_ffn_hidden_size: int = 20480
    image_token_index: int = 131072  # <image>
    img_start_token_id: int = 131073  # <img>
    img_end_token_id: int = 131074  # </img>
    tokenizer_type: str = "nemotron-h-5p5-reasoning"
    use_vision_backbone_fp8_arch: bool = True
