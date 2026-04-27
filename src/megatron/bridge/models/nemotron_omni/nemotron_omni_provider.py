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
from typing import Callable, Optional

import torch
from megatron.core import parallel_state
from megatron.core.activations import fast_gelu, squared_relu
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.nemotron_omni.modeling_nemotron_omni import NemotronOmniModel
from megatron.bridge.models.nemotron_omni.nemotron_omni_sound import BridgeSoundEncoder


@dataclass
class NemotronVLModelProvider(MambaModelProvider, ABC):
    """Abstract base provider for Nemotron VL model variants.

    Provides common VL fields, RADIO ViT-H vision config building methods, and
    vision projection config building methods shared by dense and MoE variants.
    Concrete subclasses set LLM-specific defaults (hidden_size, hybrid pattern,
    etc.) and may override ``provide()`` for variant-specific assembly.
    """

    # NemotronH base defaults
    mamba_num_groups: int = 8
    num_query_groups: int = 8
    make_vocab_size_divisible_by: int = 128
    activation_func: Callable = squared_relu
    masked_softmax_fusion: bool = True
    apply_query_key_layer_scaling: bool = False
    persist_layer_norm: bool = True
    first_last_layers_bf16: bool = True
    is_hybrid_model: bool = True

    # MoE defaults (shared across Nemotron VL variants)
    moe_aux_loss_coeff: float = 0.0001
    moe_router_score_function: str = "sigmoid"
    moe_router_enable_expert_bias: bool = True
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_router_dtype: str = "fp32"
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True
    moe_shared_expert_overlap: bool = True

    # VL common overrides
    scatter_embedding_sequence_parallel: bool = False
    attention_softmax_in_fp32: bool = True

    vision_model_type: str = "radio"
    language_model_type: str = ""

    # Token IDs (overridden by concrete subclasses)
    image_token_index: int = 0
    img_start_token_id: int = 0
    img_end_token_id: int = 0
    tokenizer_type: str = ""

    # Vision backbone control
    dynamic_resolution: bool = False
    use_vision_backbone_fp8_arch: bool = True
    radio_force_eval_mode: bool = True
    radio_force_cpe_eval_mode: bool = True
    radio_interpolate_only_cpe: bool = True
    radio_cpe_aspect_ratio_select: bool = False
    radio_disable_cpe: bool = False
    vision_proj_ffn_hidden_size: int = 20480
    vision_class_token_len: Optional[int] = None

    # Freeze control
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def _build_vision_config(self, language_cfg):
        """Build RADIO ViT-H vision encoder config from a language config copy."""
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
        if self.vision_class_token_len is not None:
            vision_cfg.class_token_len = self.vision_class_token_len
        return vision_cfg

    def _build_vision_projection_config(self, language_cfg):
        """Build vision projection MLP config from a language config copy."""
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

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class NemotronOmniModelProvider(NemotronVLModelProvider):
    """Provider for Nemotron Omni (VL + sound) models.

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

    # Temporal video embedder
    temporal_patch_dim: int = 1
    separate_video_embedder: bool = False
    temporal_ckpt_compat: bool = False  # formerly allow_checkpoint_without_temporal_compression

    def _build_vision_config(self, language_cfg):
        """Pin vision encoder to PP=1 (Omni training uses PP>1 on the LLM).

        The dense Nemotron-VL recipe runs with PP=1 everywhere, so the base
        VL provider doesn't pin this; for Omni we always co-locate the
        vision encoder with the first PP stage.
        """
        vision_cfg = super()._build_vision_config(language_cfg)
        vision_cfg.pipeline_model_parallel_size = 1
        return vision_cfg

    def _build_vision_projection_config(self, language_cfg):
        """Build vision projection MLP config, overriding activation to ReLU.

        The HF Nemotron-Omni model uses plain ReLU in its vision projection
        MLP (mlp1), not the squared_relu used by the language model. Also
        pin to PP=1 (see :meth:`_build_vision_config`).
        """
        vision_proj_cfg = super()._build_vision_projection_config(language_cfg)
        vision_proj_cfg.activation_func = torch.nn.functional.relu
        vision_proj_cfg.pipeline_model_parallel_size = 1
        return vision_proj_cfg

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
        sound_proj_cfg.pipeline_model_parallel_size = 1
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
            sound_pad_to_clip_duration=False,
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

        add_encoder_flag = parallel_state.is_pipeline_first_stage() if self.pipeline_model_parallel_size > 1 else True
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
            hybrid_layer_pattern=self.hybrid_layer_pattern,
            image_token_index=self.image_token_index,
            pixel_shuffle=True,
            max_num_tiles=12,
            tokenizer_type=self.tokenizer_type,
            use_vision_backbone_fp8_arch=self.use_vision_backbone_fp8_arch,
            dynamic_resolution=self.dynamic_resolution,
            radio_force_eval_mode=self.radio_force_eval_mode,
            radio_force_cpe_eval_mode=self.radio_force_cpe_eval_mode,
            radio_interpolate_only_cpe=self.radio_interpolate_only_cpe,
            radio_cpe_aspect_ratio_select=self.radio_cpe_aspect_ratio_select,
            radio_disable_cpe=self.radio_disable_cpe,
            sound_model=sound_model,
            sound_projection=sound_projection,
            sound_token_index=sound_token_index,
            temporal_patch_dim=self.temporal_patch_dim,
            separate_video_embedder=self.separate_video_embedder,
            temporal_ckpt_compat=self.temporal_ckpt_compat,
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
