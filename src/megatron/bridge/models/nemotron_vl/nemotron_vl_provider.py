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
from dataclasses import dataclass

from megatron.core.activations import fast_gelu
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.multimodal.llava_model import LLaVAModel
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec

from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronNano12Bv2Provider


@dataclass
class NemotronNano12Bv2VLModelProvider(NemotronNano12Bv2Provider):
    """Configuration provider for Nemotron-VL models."""

    # ------------------------------------------------------------------
    # Language configuration – inherit sensible defaults from NemotronNano12Bv2Provider
    # ------------------------------------------------------------------

    # For VL models we do *not* scatter embeddings across the sequence
    # parallel region because we need to splice vision embeddings later.
    scatter_embedding_sequence_parallel: bool = False
    attention_softmax_in_fp32: bool = True

    vision_model_type: str = "radio"
    language_model_type: str = "nemotron5-hybrid-12b"

    # Freeze knobs useful for transfer-learning scenarios
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    # ------------------------------------------------------------------
    # Provider API
    # ------------------------------------------------------------------

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

        # Language config is basically *self* (GPTModelProvider), but we make a
        # shallow copy so tweaks do not leak back.
        language_cfg = copy.deepcopy(self)

        # Vision transformer config – start from language_cfg but ensure SP/CP disabled
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

        # Vision-projection config/spec: a tiny two-layer MLP; for now just reuse
        # the MLP sub-modules from the language layer spec if available.
        vision_proj_cfg = copy.deepcopy(language_cfg)
        vision_proj_cfg.sequence_parallel = False
        vision_proj_cfg.context_parallel_size = 1
        vision_proj_cfg.tp_comm_overlap = False
        vision_proj_cfg.recompute_granularity = None
        vision_proj_cfg.recompute_method = None
        vision_proj_cfg.recompute_num_layers = None
        # Overrides for language_model_type = "nemotron5-hybrid-12b"
        vision_proj_cfg.ffn_hidden_size = 20480
        vision_proj_cfg.bias_activation_fusion = False

        language_spec = mamba_stack_spec
        vision_spec = get_vit_layer_with_transformer_engine_spec()
        vision_proj_spec = copy.deepcopy(language_spec.submodules.mlp_layer.submodules.mlp.submodules)

        # ------------------------------------------------------------------
        # Instantiate LLaVA
        # ------------------------------------------------------------------
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
            add_encoder=True,
            add_decoder=True,
            img_h=512,
            img_w=512,
            patch_dim=16,
            hybrid_attention_ratio=0.0,
            hybrid_mlp_ratio=0.0,
            hybrid_override_pattern=self.hybrid_override_pattern,
            image_token_index=131072,
            pixel_shuffle=True,
            max_num_tiles=12,
            tokenizer_type="nemotron-h-5p5-reasoning",
            use_vision_backbone_fp8_arch=True,  # Note: this is true in mlm code
        )

        from megatron.bridge.models.nemotron_vl.modeling_nemotron_vl import NemotronVLModel

        model = NemotronVLModel(llava_model=llava_model)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    # Alias that NemotronVLModel relies on to create the LM component
    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None):
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
