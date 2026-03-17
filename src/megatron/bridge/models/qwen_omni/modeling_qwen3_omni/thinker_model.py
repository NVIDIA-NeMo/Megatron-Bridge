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

import torch
from megatron.core import InferenceParams, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeThinkerConfig as Qwen3OmniMoeThinkerConfigHF,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as Qwen3OmniMoeVisionEncoderHF,
)

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import (
    Qwen3OmniTransformerConfig,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


def _build_text_only_mrope_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Create text-only multimodal rope ids shaped [3, batch, seq]."""
    batch_size, seq_len = input_ids.shape
    base = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    base = base.unsqueeze(0).expand(batch_size, -1)
    return torch.stack([base, base, base], dim=0)


class Qwen3OmniThinkerModel(MegatronModule):
    """Stage-1 Qwen3-Omni thinker model.

    This stage intentionally supports the language backbone first. Vision/audio
    runtime support is added in later milestones.
    """

    def __init__(
        self,
        language_transformer_config: Qwen3OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        thinker_transformer_config: Qwen3OmniMoeThinkerConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.visual = None
        self.audio_model = None
        self.language_model = None

        self.pg_collection = pg_collection
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.audio_token_id = language_transformer_config.audio_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id

        if self.pre_process:
            self.visual = Qwen3OmniMoeVisionEncoderHF._from_config(thinker_transformer_config.vision_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

        self.language_model = Qwen3VLGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

        # Keep the HF configs attached for later-stage vision/audio integration.
        self.thinker_transformer_config = thinker_transformer_config

    def shared_embedding_or_output_weight(self):
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3Omni"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_audio_model: bool = False,
    ):
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.visual is not None:
            modules.append(self.visual)
        if freeze_audio_model and self.audio_model is not None:
            modules.append(self.audio_model)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _get_placeholder_mask(
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor | None = None,
        video_features: torch.FloatTensor | None = None,
        image_token_id: int = 151655,
        video_token_id: int = 151656,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id

        expanded_image_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        expanded_video_mask = video_mask.unsqueeze(-1).expand_as(inputs_embeds)

        if image_features is not None and inputs_embeds[expanded_image_mask].numel() != image_features.numel():
            raise ValueError(
                f"Image features and image tokens do not match: tokens={image_mask.sum()}, "
                f"features={image_features.shape[0]}"
            )
        if video_features is not None and inputs_embeds[expanded_video_mask].numel() != video_features.numel():
            raise ValueError(
                f"Video features and video tokens do not match: tokens={video_mask.sum()}, "
                f"features={video_features.shape[0]}"
            )

        return image_mask, video_mask

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        target_dtype = getattr(self.visual, "dtype", pixel_values.dtype)
        image_embeds, image_embeds_multiscale = self.visual(pixel_values.to(dtype=target_dtype), grid_thw=image_grid_thw)
        return image_embeds, list(image_embeds_multiscale)

    def get_video_features(
        self,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        target_dtype = getattr(self.visual, "dtype", pixel_values_videos.dtype)
        video_embeds, video_embeds_multiscale = self.visual(
            pixel_values_videos.to(dtype=target_dtype), grid_thw=video_grid_thw
        )
        return video_embeds, list(video_embeds_multiscale)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("Stage 1 Qwen3-Omni does not support inference.")
        if packed_seq_params is not None:
            raise NotImplementedError("Stage 1 Qwen3-Omni does not support packed sequences.")
        if any(value is not None for value in (input_features, feature_attention_mask, audio_feature_lengths)):
            raise NotImplementedError("Audio runtime support is added in later stages.")
        if self.config.sequence_parallel and any(value is not None for value in (pixel_values, pixel_values_videos)):
            raise NotImplementedError("Stage 2 vision runtime does not yet support sequence parallel.")

        if position_ids is None:
            if pixel_values is None and pixel_values_videos is None:
                position_ids = _build_text_only_mrope_position_ids(input_ids)
            else:
                position_ids, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )

        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            visual_pos_masks = None
            deepstack_visual_embeds = None

            if pixel_values is not None or pixel_values_videos is not None:
                inputs_embeds_bsh = combined_embeddings.transpose(0, 1).contiguous()

                image_embeds = None
                image_embeds_multiscale = None
                if pixel_values is not None:
                    if image_grid_thw is None:
                        raise ValueError("image_grid_thw is required when pixel_values is provided")
                    image_embeds, image_embeds_multiscale = self.get_image_features(pixel_values, image_grid_thw)
                    image_embeds = image_embeds.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype)
                    image_embeds_multiscale = [embed.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype) for embed in image_embeds_multiscale]

                video_embeds = None
                video_embeds_multiscale = None
                if pixel_values_videos is not None:
                    if video_grid_thw is None:
                        raise ValueError("video_grid_thw is required when pixel_values_videos is provided")
                    video_embeds, video_embeds_multiscale = self.get_video_features(pixel_values_videos, video_grid_thw)
                    video_embeds = video_embeds.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype)
                    video_embeds_multiscale = [embed.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype) for embed in video_embeds_multiscale]

                image_mask, video_mask = self._get_placeholder_mask(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds_bsh,
                    image_features=image_embeds,
                    video_features=video_embeds,
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                )

                if image_embeds is not None:
                    inputs_embeds_bsh.masked_scatter_(image_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), image_embeds)
                    visual_pos_masks = image_mask
                    deepstack_visual_embeds = image_embeds_multiscale

                if video_embeds is not None:
                    inputs_embeds_bsh.masked_scatter_(video_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), video_embeds)
                    if deepstack_visual_embeds is None:
                        visual_pos_masks = video_mask
                        deepstack_visual_embeds = video_embeds_multiscale
                    else:
                        assert visual_pos_masks is not None
                        visual_pos_masks = visual_pos_masks | video_mask
                        joint_embeds: list[torch.Tensor] = []
                        image_mask_joint = image_mask[visual_pos_masks]
                        video_mask_joint = video_mask[visual_pos_masks]
                        for image_embed, video_embed in zip(deepstack_visual_embeds, video_embeds_multiscale):
                            embed_joint = image_embed.new_zeros((int(visual_pos_masks.sum().item()), image_embed.shape[-1]))
                            embed_joint[image_mask_joint] = image_embed
                            embed_joint[video_mask_joint] = video_embed
                            joint_embeds.append(embed_joint)
                        deepstack_visual_embeds = joint_embeds

                combined_embeddings = inputs_embeds_bsh.transpose(0, 1).contiguous()
            else:
                visual_pos_masks = None
                deepstack_visual_embeds = None

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_visual_embeds = None

        return self.language_model(
            input_ids=None if combined_embeddings is not None else input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            loss_mask=loss_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
