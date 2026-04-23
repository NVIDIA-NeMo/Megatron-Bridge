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
    Qwen3OmniMoeAudioEncoder as Qwen3OmniMoeAudioEncoderHF,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoder as Qwen3OmniMoeVisionEncoderHF,
)

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import get_rope_index
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import (
    Qwen3OmniTransformerConfig,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    split_data_cp_rank,
    split_deepstack_embs,
)
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


def _deep_getattr(module: torch.nn.Module, attr_path: str) -> torch.nn.Module:
    value = module
    for attr in attr_path.split("."):
        value = getattr(value, attr)
    return value


def _patch_get_input_embeddings(module: torch.nn.Module, attr_path: str) -> None:
    """Match ms-swift's tower patching for gradient-checkpoint input hooks."""

    def _get_input_embeddings(self) -> torch.nn.Module:
        return _deep_getattr(self, attr_path)

    module.get_input_embeddings = _get_input_embeddings.__get__(module, module.__class__)


def _build_text_only_mrope_position_ids(input_ids: torch.Tensor) -> torch.Tensor:
    """Create text-only multimodal rope ids shaped [3, batch, seq]."""
    batch_size, seq_len = input_ids.shape
    base = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    base = base.unsqueeze(0).expand(batch_size, -1)
    return torch.stack([base, base, base], dim=0)


def _configure_multimodal_attn_impl(config: object, attn_impl: str | None) -> None:
    """Apply a requested attention implementation to HF multimodal configs."""
    if not attn_impl or attn_impl == "auto":
        return

    setattr(config, "_attn_implementation", attn_impl)
    setattr(config, "_attn_implementation_internal", attn_impl)
    setattr(config, "attn_implementation", attn_impl)

    use_flash_attn = attn_impl in {"flash_attn", "flash_attention_2"}
    setattr(config, "use_flash_attn", use_flash_attn)
    setattr(config, "_use_flash_attention_2", use_flash_attn)
    setattr(config, "_flash_attn_2_enabled", use_flash_attn)


def _enable_multimodal_gradient_checkpointing(module: torch.nn.Module) -> None:
    """Best-effort enable gradient checkpointing for HF multimodal towers."""
    if hasattr(module, "gradient_checkpointing_enable"):
        try:
            module.gradient_checkpointing_enable()
        except TypeError:
            module.gradient_checkpointing_enable({})
        except NotImplementedError:
            pass

    if hasattr(module, "enable_input_require_grads"):
        try:
            module.enable_input_require_grads()
        except AttributeError:
            pass
        except NotImplementedError:
            pass


def _trim_feature_sequence(
    features: torch.Tensor | None,
    multiscale_features: list[torch.Tensor] | None,
    expected_tokens: int,
    feature_name: str,
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    if features is None:
        return None, multiscale_features

    produced_tokens = int(features.shape[0])
    if expected_tokens > produced_tokens:
        raise ValueError(
            f"{feature_name} placeholders exceed produced features: expected={expected_tokens}, produced={produced_tokens}"
        )
    if expected_tokens == produced_tokens:
        return features, multiscale_features

    trimmed_features = features[:expected_tokens]
    trimmed_multiscale = None
    if multiscale_features is not None:
        trimmed_multiscale = [feature[:expected_tokens] for feature in multiscale_features]
    return trimmed_features, trimmed_multiscale


class Qwen3OmniThinkerModel(MegatronModule):
    """Qwen3-Omni thinker model.

    The current implementation supports multimodal thinker-side forward paths
    for text, vision, and audio inputs.
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
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds
        self.seconds_per_chunk = language_transformer_config.seconds_per_chunk
        self.thinker_transformer_config = thinker_transformer_config

        if self.pre_process:
            multimodal_attn_impl = getattr(language_transformer_config, "multimodal_attn_impl", "auto")
            _configure_multimodal_attn_impl(thinker_transformer_config.vision_config, multimodal_attn_impl)
            _configure_multimodal_attn_impl(thinker_transformer_config.audio_config, multimodal_attn_impl)

            self.visual = Qwen3OmniMoeVisionEncoderHF._from_config(thinker_transformer_config.vision_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)
            self.audio_model = Qwen3OmniMoeAudioEncoderHF._from_config(thinker_transformer_config.audio_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.audio_model)
            _patch_get_input_embeddings(self.visual, "patch_embed")
            _patch_get_input_embeddings(self.audio_model, "conv_out")

            _configure_multimodal_attn_impl(
                getattr(self.visual, "config", self.thinker_transformer_config.vision_config), multimodal_attn_impl
            )
            _configure_multimodal_attn_impl(
                getattr(self.audio_model, "config", self.thinker_transformer_config.audio_config), multimodal_attn_impl
            )
            if getattr(language_transformer_config, "vit_gradient_checkpointing", False):
                _enable_multimodal_gradient_checkpointing(self.visual)
                _enable_multimodal_gradient_checkpointing(self.audio_model)

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
        image_embeds, image_embeds_multiscale = self.visual(
            pixel_values.to(dtype=target_dtype), grid_thw=image_grid_thw
        )
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

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
        expected_audio_token_counts: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

        if audio_feature_lengths is None:
            raise ValueError("Either feature_attention_mask or audio_feature_lengths must be provided")

        target_dtype = getattr(self.audio_model, "dtype", input_features.dtype)
        audio_outputs = self.audio_model(
            input_features.to(dtype=target_dtype),
            feature_lens=audio_feature_lengths,
        )
        audio_embeds = audio_outputs.last_hidden_state

        if expected_audio_token_counts is None:
            return audio_embeds

        _, produced_output_lengths = self.audio_model._get_feat_extract_output_lengths(audio_feature_lengths)
        expected_audio_token_counts = expected_audio_token_counts.to(produced_output_lengths.device)

        trimmed_embeds = []
        start = 0
        for produced_len, expected_len in zip(
            produced_output_lengths.tolist(), expected_audio_token_counts.tolist(), strict=True
        ):
            end = start + int(produced_len)
            if expected_len > produced_len:
                raise ValueError(
                    f"Audio token placeholders exceed produced audio features: expected={expected_len}, produced={produced_len}"
                )
            trimmed_embeds.append(audio_embeds[start : start + int(expected_len)])
            start = end

        return (
            torch.cat(trimmed_embeds, dim=0) if trimmed_embeds else audio_embeds.new_zeros((0, audio_embeds.shape[-1]))
        )

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
        video_second_per_grid: torch.Tensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("Qwen3-Omni Megatron inference is not implemented yet.")
        if packed_seq_params is not None:
            raise NotImplementedError("Qwen3-Omni packed sequence support is not implemented yet.")

        cp_size = self.pg_collection.cp.size() if self.pg_collection is not None else 1
        cp_rank = self.pg_collection.cp.rank() if self.pg_collection is not None else 0
        tp_size = self.pg_collection.tp.size() if self.pg_collection is not None else 1
        tp_rank = self.pg_collection.tp.rank() if self.pg_collection is not None else 0

        if audio_feature_lengths is None and feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        if position_ids is None:
            has_multimodal_inputs = any(
                value is not None
                for value in (pixel_values, pixel_values_videos, input_features, audio_feature_lengths)
            )
            if not has_multimodal_inputs:
                position_ids = _build_text_only_mrope_position_ids(input_ids)
            else:
                position_ids, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.audio_token_id,
                    self.vision_start_token_id,
                    self.audio_start_token_id,
                    self.position_id_per_seconds,
                    input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grids=video_second_per_grid,
                    attention_mask=attention_mask,
                    audio_seqlens=audio_feature_lengths,
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
                    image_embeds_multiscale = [
                        embed.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype)
                        for embed in image_embeds_multiscale
                    ]
                    image_embeds, image_embeds_multiscale = _trim_feature_sequence(
                        image_embeds,
                        image_embeds_multiscale,
                        expected_tokens=int((input_ids == self.image_token_id).sum().item()),
                        feature_name="Image features",
                    )

                video_embeds = None
                video_embeds_multiscale = None
                if pixel_values_videos is not None:
                    if video_grid_thw is None:
                        raise ValueError("video_grid_thw is required when pixel_values_videos is provided")
                    video_embeds, video_embeds_multiscale = self.get_video_features(
                        pixel_values_videos, video_grid_thw
                    )
                    video_embeds = video_embeds.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype)
                    video_embeds_multiscale = [
                        embed.to(inputs_embeds_bsh.device, inputs_embeds_bsh.dtype)
                        for embed in video_embeds_multiscale
                    ]
                    video_embeds, video_embeds_multiscale = _trim_feature_sequence(
                        video_embeds,
                        video_embeds_multiscale,
                        expected_tokens=int((input_ids == self.video_token_id).sum().item()),
                        feature_name="Video features",
                    )

                image_mask, video_mask = self._get_placeholder_mask(
                    input_ids=input_ids,
                    inputs_embeds=inputs_embeds_bsh,
                    image_features=image_embeds,
                    video_features=video_embeds,
                    image_token_id=self.image_token_id,
                    video_token_id=self.video_token_id,
                )

                if image_embeds is not None:
                    inputs_embeds_bsh.masked_scatter_(
                        image_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), image_embeds
                    )
                    visual_pos_masks = image_mask
                    deepstack_visual_embeds = image_embeds_multiscale

                if video_embeds is not None:
                    inputs_embeds_bsh.masked_scatter_(
                        video_mask.unsqueeze(-1).expand_as(inputs_embeds_bsh), video_embeds
                    )
                    if deepstack_visual_embeds is None:
                        visual_pos_masks = video_mask
                        deepstack_visual_embeds = video_embeds_multiscale
                    else:
                        assert visual_pos_masks is not None
                        visual_pos_masks = visual_pos_masks | video_mask
                        joint_embeds: list[torch.Tensor] = []
                        image_mask_joint = image_mask[visual_pos_masks]
                        video_mask_joint = video_mask[visual_pos_masks]
                        for image_embed, video_embed in zip(
                            deepstack_visual_embeds, video_embeds_multiscale, strict=True
                        ):
                            embed_joint = image_embed.new_zeros(
                                (int(visual_pos_masks.sum().item()), image_embed.shape[-1])
                            )
                            embed_joint[image_mask_joint] = image_embed
                            embed_joint[video_mask_joint] = video_embed
                            joint_embeds.append(embed_joint)
                        deepstack_visual_embeds = joint_embeds

                combined_embeddings = inputs_embeds_bsh.transpose(0, 1).contiguous()
            else:
                visual_pos_masks = None
                deepstack_visual_embeds = None

            if input_features is not None:
                expected_audio_token_counts = (input_ids == self.audio_token_id).sum(dim=1)
                audio_embeds = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                    expected_audio_token_counts=expected_audio_token_counts,
                )
                audio_embeds = audio_embeds.to(combined_embeddings.device, combined_embeddings.dtype)
                combined_embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                audio_mask = input_ids == self.audio_token_id
                if combined_embeddings_bsh[audio_mask].numel() != audio_embeds.numel():
                    raise ValueError(
                        f"Audio features and audio tokens do not match: tokens={audio_mask.sum()}, "
                        f"features={audio_embeds.shape[0]}"
                    )
                combined_embeddings_bsh.masked_scatter_(
                    audio_mask.unsqueeze(-1).expand_as(combined_embeddings_bsh),
                    audio_embeds,
                )
                combined_embeddings = combined_embeddings_bsh.transpose(0, 1).contiguous()

            if combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)

            sp_pad_len = 0
            if self.config.sequence_parallel:
                seq_len = combined_embeddings.shape[0]
                sp_pad_len = (tp_size - seq_len % tp_size) % tp_size
                if sp_pad_len > 0:
                    combined_embeddings = torch.nn.functional.pad(combined_embeddings, (0, 0, 0, 0, 0, sp_pad_len))
                    if visual_pos_masks is not None:
                        visual_pos_masks = torch.nn.functional.pad(visual_pos_masks, (0, sp_pad_len), value=False)
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
            visual_pos_masks = None
            deepstack_visual_embeds = None
            sp_pad_len = 0

        if sp_pad_len > 0 and position_ids is not None:
            position_ids = torch.nn.functional.pad(position_ids, (0, sp_pad_len), mode="replicate")

        if self.config.sequence_parallel or cp_size > 1:
            visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                sequence_parallel=self.config.sequence_parallel,
            )

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
