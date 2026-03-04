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
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniThinkerConfig as Qwen2_5OmniThinkerConfigHF,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAudioEncoder as Qwen2_5OmniAudioEncoderHF,
)
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniVisionEncoder as Qwen2_5OmniVisionEncoderHF,
)

from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.rope import get_rope_index
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.transformer_config import Qwen25OmniTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    split_data_cp_rank,
)
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class Qwen25OmniThinkerModel(MegatronModule):
    """Qwen2.5 Omni Thinker Model.

    Key differences from Qwen3OmniMoeThinkerModel:
    - Uses HF vision encoder (Qwen2_5OmniVisionEncoder) directly, not Megatron-native
    - Uses HF audio encoder (Qwen2_5OmniAudioEncoder) directly
    - No deepstack visual embeddings
    - Vision embeddings inserted only at input level
    - Dense LLM (Qwen2 architecture), not MoE
    """

    def __init__(
        self,
        language_transformer_config: Qwen25OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        thinker_transformer_config: Qwen2_5OmniThinkerConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        language_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.visual = None
        self.audio_model = None
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.audio_token_id = language_transformer_config.audio_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds
        self.seconds_per_chunk = language_transformer_config.seconds_per_chunk

        self.square_merge_size = thinker_transformer_config.vision_config.spatial_merge_size**2

        self.share_embeddings_and_output_weights = False
        self.pg_collection = pg_collection
        self.cp_group = pg_collection.cp
        self.tp_group = pg_collection.tp
        self.pp_group = pg_collection.pp
        assert hasattr(self.pg_collection, "embd"), (
            "pg_collection must have a embd. In previous version, it used default "
            "`parallel_state.default_embedding_ranks` to create the process group."
            "If you are using the default process group, please use"
            "`parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        self.embd_group = pg_collection.embd
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        if self.pre_process:
            # Use HF vision encoder directly (ReplicatedMapping in bridge)
            self.visual = Qwen2_5OmniVisionEncoderHF._from_config(thinker_transformer_config.vision_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.visual)

            # Use HF audio encoder directly (ReplicatedMapping in bridge)
            self.audio_model = Qwen2_5OmniAudioEncoderHF._from_config(thinker_transformer_config.audio_config)
            hook_hf_module_setattr_for_tp_grad_sync(self.audio_model)

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
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen25Omni"

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
        """Freeze model modules.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_audio_model (bool): Freeze the audio model module.
        """
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

    def get_audio_features(
        self,
        input_features: torch.FloatTensor,
        feature_attention_mask: torch.LongTensor | None = None,
        audio_feature_lengths: torch.LongTensor | None = None,
    ):
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
            input_features = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)
        else:
            audio_feature_lengths = None

        feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        audio_feat_lengths, audio_output_lengths = self.audio_model._get_feat_extract_output_lengths(feature_lens)

        audio_outputs = self.audio_model(
            input_features,
            feature_lens=feature_lens,
            aftercnn_lens=audio_feat_lengths,
        )

        return audio_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features=None,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        cp_img_num: list[int] = None,
        images_padded: list[bool] = None,
        use_audio_in_video=None,
        video_second_per_grid=None,
        **kwargs,
    ) -> torch.Tensor:
        assert inference_params is None, "not support inference"
        assert packed_seq_params is None, "not support packed_seq_params"

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()

        if self.pre_process:
            # Run HF vision encoder to get vision embeddings (no deepstack)
            vision_embeds = None
            vision_mask = None
            if pixel_values is not None or pixel_values_videos is not None:
                # Build vision mask from input_ids
                image_mask = input_ids == self.image_token_id
                video_mask = input_ids == self.video_token_id
                vision_mask = image_mask | video_mask

                # Process images through vision encoder
                if pixel_values is not None and image_grid_thw is not None:
                    image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                else:
                    image_embeds = None

                # Process videos through vision encoder
                if pixel_values_videos is not None and video_grid_thw is not None:
                    video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                else:
                    video_embeds = None

                # Combine image and video embeddings
                if image_embeds is not None and video_embeds is not None:
                    vision_embeds = torch.cat([image_embeds, video_embeds], dim=0)
                elif image_embeds is not None:
                    vision_embeds = image_embeds
                elif video_embeds is not None:
                    vision_embeds = video_embeds

            # Extract audio features
            audio_embeds = None
            if input_features is not None:
                audio_embeds = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_mask = input_ids == self.audio_token_id

            # Get text embeddings from language model
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()  # [text_seq_len, b, h_language]

            # Replace vision/audio token positions with vision_embeds/audio_embeds
            if vision_embeds is not None or audio_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if vision_embeds is not None:
                    combined_embeddings[vision_mask] = vision_embeds
                if audio_embeds is not None:
                    combined_embeddings[audio_mask] = audio_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)

            # Track SP padding amount for position_ids alignment
            sp_pad_len = 0
            if self.config.sequence_parallel:
                tp_size = self.pg_collection.tp.size()
                seq_len = combined_embeddings.shape[0]
                sp_pad_len = (tp_size - seq_len % tp_size) % tp_size
                if sp_pad_len > 0:
                    combined_embeddings = torch.nn.functional.pad(combined_embeddings, (0, 0, 0, 0, 0, sp_pad_len))
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None
            sp_pad_len = 0

        # Compute audio feature lengths for rope computation
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        # Compute position IDs via get_rope_index if not provided
        if position_ids is None:
            position_ids, _ = get_rope_index(
                self.config.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.audio_token_id,
                self.vision_start_token_id,
                self.audio_start_token_id,
                self.position_id_per_seconds,
                self.seconds_per_chunk,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_feature_lengths,
                second_per_grids=video_second_per_grid,
            )

        # Pad position_ids to match SP-padded embeddings so rotary_pos_emb
        # has the same sequence length as the all-gathered query/key tensors.
        if sp_pad_len > 0 and position_ids is not None:
            # position_ids shape: [3, batch, seq_len] → pad last dim
            position_ids = torch.nn.functional.pad(position_ids, (0, sp_pad_len), mode="replicate")

        # No deepstack for Qwen2.5 Omni
        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            visual_pos_masks=None,
            deepstack_visual_embeds=None,
            **(extra_block_kwargs or {}),
        )

        return output
