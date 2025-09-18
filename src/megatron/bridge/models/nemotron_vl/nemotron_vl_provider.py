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

from dataclasses import dataclass, field
from typing import List

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class NemotronVLModelProvider(GPTModelProvider):
    """Configuration provider for Nemotron-VL models."""

    # ------------------------------------------------------------------
    # Language configuration – inherit sensible defaults from GPTProvider
    # ------------------------------------------------------------------
    gated_linear_unit: bool = False
    # For VL models we do *not* scatter embeddings across the sequence
    # parallel region because we need to splice vision embeddings later.
    scatter_embedding_sequence_parallel: bool = False

    # vision_config: _DummyVisionConfig = field(default_factory=_DummyVisionConfig)

    # Special tokens (values here follow Qwen-VL for convenience)
    # bos_token_id: int = 151643
    # eos_token_id: int = 151645
    # vision_start_token_id: int = 151652
    # vision_end_token_id: int = 151653
    # vision_token_id: int = 151654
    # image_token_id: int = 151655
    # video_token_id: int = 151656

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
        import copy
        from megatron.core.transformer.enums import AttnMaskType
        from megatron.core.transformer.spec_utils import import_module
        from megatron.core.models.multimodal.llava_model import LLaVAModel
        from megatron.core.models.vision.vit_layer_specs import (
            get_vit_layer_with_local_spec,
        )

        # Language config is basically *self* (GPTModelProvider), but we make a
        # shallow copy so tweaks do not leak back.
        language_cfg = copy.copy(self)

        # Determine language layer spec using the same helper logic as GPTProvider
        if callable(language_cfg.transformer_layer_spec):
            language_spec = language_cfg.transformer_layer_spec(language_cfg)
        else:
            language_spec = language_cfg.transformer_layer_spec

        # Vision transformer config – start from language_cfg but ensure SP/CP disabled
        vision_cfg = copy.copy(language_cfg)
        vision_cfg.sequence_parallel = False
        vision_cfg.context_parallel_size = 1
        vision_cfg.tp_comm_overlap = False

        # Use simple local ViT spec (no TE dependency)
        vision_spec = get_vit_layer_with_local_spec()

        # Vision-projection config/spec: a tiny two-layer MLP; for now just reuse
        # the MLP sub-modules from the language layer spec if available.
        vision_proj_cfg = copy.copy(language_cfg)
        vision_proj_cfg.sequence_parallel = False
        vision_proj_cfg.context_parallel_size = 1

        if hasattr(language_spec.submodules, "mlp") and hasattr(language_spec.submodules.mlp, "submodules"):
            vision_proj_spec = copy.deepcopy(language_spec.submodules.mlp.submodules)
        else:
            vision_proj_spec = vision_spec.submodules  # fallback

        # ------------------------------------------------------------------
        # Instantiate LLaVA
        # ------------------------------------------------------------------
        # model = LLaVAModel(
        #     language_transformer_config=language_config,
        #     language_transformer_layer_spec=language_transformer_layer_spec,
        #     language_vocab_size=args.padded_vocab_size,
        #     language_max_sequence_length=args.decoder_seq_length,
        #     vision_transformer_config=vision_config,
        #     vision_transformer_layer_spec=vision_transformer_layer_spec,
        #     drop_vision_class_token=args.disable_vision_class_token,
        #     vision_projection_config=vision_projection_config,
        #     vision_projection_layer_spec=vision_projection_layer_spec,
        #     vision_projection_type="mlp",
        #     allow_missing_vision_projection_checkpoint=args.allow_missing_vision_projection_checkpoint,
        #     parallel_output=parallel_output,
        #     share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        #     language_position_embedding_type=args.position_embedding_type,
        #     language_rotary_percent=args.rotary_percent,
        #     pre_process=pre_process,
        #     post_process=post_process,
        #     add_encoder=add_encoder,
        #     add_decoder=add_decoder,
        #     img_h=args.img_h,
        #     img_w=args.img_w,
        #     patch_dim=args.patch_dim,
        #     language_rotary_base=args.rotary_base,
        #     language_rope_scaling=args.use_rope_scaling,
        #     hybrid_attention_ratio=args.hybrid_attention_ratio,
        #     hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        #     hybrid_override_pattern=args.hybrid_override_pattern,
        #     fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        #     image_token_index=image_token_index,
        #     pixel_shuffle=args.pixel_shuffle,
        #     tile_tags=tile_tags,
        #     dynamic_resolution=args.dynamic_resolution,
        #     max_num_tiles=args.max_num_tiles,
        #     tokenizer_type=args.tokenizer_prompt_format,
        #     use_vision_backbone_fp8_arch=args.use_vision_backbone_fp8_arch,
        #     image_break_token=tokenizer.convert_tokens_to_ids(args.image_break_token) if args.image_break_token is not None else None,
        #     conv_merging=args.conv_merging,
        #     allow_missing_conv_merge_checkpoint=args.allow_missing_conv_merge_checkpoint,
        #     efficient_video_sampling_variant=args.efficient_video_sampling_variant,
        #     sound_model=sound_model,
        #     sound_projection=sound_projection,
        #     sound_token_index=sound_token_index,
        # )



        llava_model = LLaVAModel(
            language_transformer_config=language_cfg,
            language_transformer_layer_spec=language_spec,
            language_vocab_size=self.vocab_size,
            language_max_sequence_length=self.seq_length,
            vision_transformer_config=vision_cfg,
            vision_transformer_layer_spec=vision_spec,
            drop_vision_class_token=False,
            vision_projection_config=vision_proj_cfg,
            vision_projection_layer_spec=vision_proj_spec,
            vision_projection_type="mlp",
            parallel_output=self.parallel_output,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            language_position_embedding_type=self.position_embedding_type,
            language_rotary_percent=self.rotary_percent,
            pre_process=pre_process if pre_process is not None else True,
            post_process=post_process if post_process is not None else True,
            add_encoder=True,
            add_decoder=True,
            img_h=512,
            img_w=512,
            patch_dim=16,
        )

        from .modeling_nemotron_vl import NemotronVLModel

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
