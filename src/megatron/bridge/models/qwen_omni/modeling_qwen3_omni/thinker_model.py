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
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.models.vision.vit_layer_specs import get_vit_layer_with_transformer_engine_spec
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

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.rope import get_rope_index
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import Qwen3OmniTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import get_vision_model_config
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    AllGatherVisionEmbeddings,
    PatchMergerSubmodules,
    collapse_thw,
    get_vision_cp_data,
    qwen3vl_cp_split,
    reorganize_inputs,
    split_data_cp_rank,
    split_deepstack_embs,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class Qwen3OmniMoeThinkerModel(MegatronModule):
    """Qwen3 Omni Moe Thinker Model."""

    def __init__(
        self,
        language_transformer_config: Qwen3OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config: Qwen3OmniMoeThinkerConfigHF,
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
        self.vision_model = None
        self.audio_model = None
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.audio_token_id = language_transformer_config.audio_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id
        self.position_id_per_seconds = language_transformer_config.position_id_per_seconds

        self.square_merge_size = vision_transformer_config.vision_config.spatial_merge_size**2

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        # process groups
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
            if language_transformer_config.use_hf_vision_model:
                raise ValueError("use_hf_vision_model is not supported for Qwen3VLModel for now")
            # use megatron vision model
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )

            vision_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention
            megatron_vision_transformer_config = get_vision_model_config(
                vision_transformer_config.vision_config, megatron_config=language_transformer_config
            )
            megatron_vision_transformer_config.pipeline_model_parallel_size = 1
            megatron_vision_transformer_config.first_pipeline_num_layers = None

            self.vision_model = Qwen3VLVisionModel(
                megatron_vision_transformer_config,
                vision_transformer_layer_spec,
                vision_patch_merger_spec,
                pre_process=True,
                post_process=True,
            )

            # Initialize audio model with random weights from config
            self.audio_model = Qwen3OmniMoeAudioEncoderHF._from_config(vision_transformer_config.audio_config)
            # Ensure HF audio tower params are marked for TP grad sync and future assignments are hooked.
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
        assert len(vision_transformer_config.vision_config.deepstack_visual_indexes) < len(
            self.language_model.decoder.layers
        ), (
            "the deepstack_visual_embeds should on the first pp-stage",
            f"got {len(vision_transformer_config.vision_config.deepstack_visual_indexes)} deepstack_visual_indexes, "
            f" {len(self.language_model.decoder.layers)} language model layers",
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3VL"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vision_projection: bool = False,
        freeze_audio_model: bool = False,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection modules.
            freeze_audio_model (bool): Freeze the audio model module.
        """
        modules = []

        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)

        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.decoder.deepstack_merger_list)
            modules.append(self.vision_model.merger)

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

        # feature_lens = audio_feature_lengths if audio_feature_lengths is not None else feature_attention_mask.sum(-1)
        feature_lens = audio_feature_lengths
        audio_outputs = self.audio_model(
            input_features,
            feature_lens=feature_lens,
        )

        return audio_outputs.last_hidden_state

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features=None,
        position_ids: torch.Tensor = None,  # can set at dataset
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
        # cat set at dataset
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

        vision_grid_thw = None
        vision_data = None
        vision_mask = None
        deepstack_feature_lists = None

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()

        if self.pre_process:
            vision_data, vision_grid_thw, vision_mask = reorganize_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_input_mask=image_input_mask,
                video_input_mask=video_input_mask,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                square_merge_size=self.square_merge_size,
            )

            vision_embeds = None
            if vision_grid_thw is not None and vision_grid_thw.shape[0] > 0:
                if cp_size > 1:
                    if cp_img_num is None:
                        assert images_padded is None
                        vision_data, vision_grid_thw, cp_img_num, images_padded = qwen3vl_cp_split(
                            cp_size,
                            vision_data,
                            vision_grid_thw,
                        )
                    vision_data, vision_grid_thw, seqlen_on_cp_ranks = get_vision_cp_data(
                        vision_data,
                        vision_grid_thw,
                        self.square_merge_size,
                        cp_img_num,
                        images_padded,
                        cp_rank,
                        cp_size,
                    )
                    vision_grid_thw = collapse_thw(vision_grid_thw)

                if vision_data.shape[0] > 0:
                    vision_embeds, deepstack_feature_lists = self.vision_model(
                        hidden_states=vision_data,
                        grid_thw=vision_grid_thw,
                    )
                else:
                    vision_embeds = torch.zeros(
                        (0, self.language_model.config.hidden_size),
                        device=vision_data.device,
                        dtype=torch.bfloat16,
                    )
                    deepstack_feature_lists = []
                    for _ in self.vision_model.config.deepstack_visual_indexes:
                        deepstack_feature_lists.append(
                            torch.zeros(
                                (0, self.language_model.config.hidden_size),
                                device=vision_data.device,
                                dtype=torch.bfloat16,
                            )
                        )

                if cp_size > 1:
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds,
                        seqlen_on_cp_ranks,
                        self.pg_collection.cp,
                    )
                    for i in range(len(deepstack_feature_lists)):
                        deepstack_feature_lists[i] = AllGatherVisionEmbeddings.apply(
                            deepstack_feature_lists[i],
                            seqlen_on_cp_ranks,
                            self.pg_collection.cp,
                        )

            audio_embeds = None
            if input_features is not None:
                audio_embeds = self.get_audio_features(
                    input_features,
                    feature_attention_mask=feature_attention_mask,
                    audio_feature_lengths=audio_feature_lengths,
                )
                audio_mask = input_ids == self.audio_token_id

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None or audio_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if vision_embeds is not None:
                    combined_embeddings[vision_mask] = vision_embeds
                if audio_embeds is not None:
                    combined_embeddings[audio_mask] = audio_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        else:
            audio_feature_lengths = None

        if position_ids is None:
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
                attention_mask=attention_mask,
                use_audio_in_video=use_audio_in_video,
                audio_seqlens=audio_feature_lengths,
                second_per_grids=video_second_per_grid,
            )

        visual_pos_masks = vision_mask
        deepstack_visual_embeds = deepstack_feature_lists
        if self.config.sequence_parallel or cp_size > 1:
            visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=self.pg_collection.tp.size(),
                tp_rank=self.pg_collection.tp.rank(),
                cp_size=cp_size,
                cp_rank=self.pg_collection.cp.rank(),
                sequence_parallel=self.config.sequence_parallel,
            )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        return output
