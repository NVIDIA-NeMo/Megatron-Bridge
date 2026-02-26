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
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as Qwen3VLConfigHF

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.rope import get_rope_index
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import (
    Qwen3VLTransformerConfig,
    get_vision_model_config,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    AllGatherVisionEmbeddings,
    PatchMergerSubmodules,
    collapse_thw,
    get_vision_cp_data,
    preprocess_packed_seqs,
    qwen3vl_cp_split,
    reorganize_inputs,
    split_data_cp_rank,
    split_deepstack_embs,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel


class Qwen3VLModel(MegatronModule):
    """Qwen3VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
        vision_transformer_config (Qwen3VLConfigHF): HF config for the vision model.
        parallel_output (bool): Do not gather the outputs, keep them split across tensor parallel ranks. This
            is typically True for training and False for inference.
        language_rotary_percent (float): Percent of rotary dimension to use for rotary position embeddings
            in the language model. Defaults to 1.0.
        pre_process (bool): Include the embedding layer in the gpt decoder (used with pipeline parallelism).
            Defaults to True.
        post_process (bool): Include an output layer and a layernorm in the gpt decoder (used with pipeline
            parallelism). Defaults to True.
        add_encoder (bool): Construct the encoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the encoder
            will live on only a subset of the pipeline stages (specifically, only the first stage).
        add_decoder (bool): Construct the decoder module (used with pipeline parallelism). Defaults to True.
            When we use pipelining, the decoder
            will live on only a subset of the pipeline stages (specifically, every stage after the first one).
    """

    def __init__(
        self,
        language_transformer_config: Qwen3VLTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config: Qwen3VLConfigHF,
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
        self.language_model = None
        self.image_token_id = language_transformer_config.image_token_id
        self.video_token_id = language_transformer_config.video_token_id
        self.vision_start_token_id = language_transformer_config.vision_start_token_id

        self.square_merge_size = vision_transformer_config.spatial_merge_size**2

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
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )

            vision_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention
            megatron_vision_transformer_config = get_vision_model_config(
                vision_transformer_config, megatron_config=language_transformer_config
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
        if pre_process:
            assert len(vision_transformer_config.deepstack_visual_indexes) <= len(
                self.language_model.decoder.layers
            ), (
                "the deepstack_visual_embeds should on the first pp-stage of language model",
                f"got {len(vision_transformer_config.deepstack_visual_indexes)} deepstack_visual_indexes, "
                f" {len(self.language_model.decoder.layers)} language model layers",
            )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

        if self.pg_collection.cp.size() > 1:
            assert self.config.calculate_per_token_loss, (
                "Qwen3-VL model only supports context parallelism with calculate_per_token_loss enabled"
            )

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
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.decoder.deepstack_merger_list)
            modules.append(self.vision_model.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        if freeze_vision_model and not freeze_vision_projection:
            if self.vision_model is not None:
                for param in self.vision_model.decoder.deepstack_merger_list.parameters():
                    param.requires_grad = True
                for param in self.vision_model.merger.parameters():
                    param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor,
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
        # can set at dataset
        image_input_mask: torch.Tensor = None,
        video_input_mask: torch.Tensor = None,
        cp_img_num: list[int] = None,
        images_padded: list[bool] = None,
        inference_context: object | None = None,
        runtime_gather_output: bool | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward function of the Qwen3VL model.
        # there is a workaround for supporting sequence packing with context parallelism
        # cp split with sequence packing will make model lose vision token information, so we need to keep
        # the original input_ids and pack them after vision embedding is calculated,
        # cooporate with verl's models/mcore/model_forward.py
        # pack the combined_embeddings to thd here, we check if packed_seq_params is None to determine if we need to pack the combined_embeddings to thd
        # this function needs the position_ids and attention_mask in BSHD format, no matter use packed_seq or not

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
                combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """
        del inference_context, runtime_gather_output  # Unused, kept for API compatibility
        assert inference_params is None, "not support inference"

        vision_grid_thw = None
        vision_data = None
        vision_mask = None
        deepstack_feature_lists = None

        # position ids is computed within the model
        position_ids = None

        torch.cuda.nvtx.range_push("Qwen3VLModel.forward.pre_process")

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()

        if self.pre_process:
            # can reorganize_inputs at dataset
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
                if cp_size > 1 and self.config.vision_dp_when_cp:
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
                if cp_size > 1 and self.config.vision_dp_when_cp:
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds,
                        seqlen_on_cp_ranks,
                        cp_group=self.pg_collection.cp,
                    )
                    for i in range(len(deepstack_feature_lists)):
                        deepstack_feature_lists[i] = AllGatherVisionEmbeddings.apply(
                            deepstack_feature_lists[i],
                            seqlen_on_cp_ranks,
                            cp_group=self.pg_collection.cp,
                        )

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[vision_mask] = vision_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if combined_embeddings is not None and cp_size > 1 and packed_seq_params is None:
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)
            if packed_seq_params is not None:
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.int32, device=input_ids.device)
                input_ids_thd, _ = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True, pg_collection=self.pg_collection
                )
                _, _, vision_mask_thd = reorganize_inputs(
                    input_ids=input_ids_thd,
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

                if deepstack_feature_lists is not None:
                    tmp_embeddings = torch.zeros_like(combined_embeddings.transpose(0, 1))
                    new_deepstack_feature_lists = []
                    for deepstack_visual_embed in deepstack_feature_lists:
                        tmp_embeddings[vision_mask] = deepstack_visual_embed
                        tmp_embeddings_thd = preprocess_packed_seqs(
                            tmp_embeddings.contiguous(),
                            attention_mask,
                            pre_process=True,
                            pg_collection=self.pg_collection,
                        )[0]
                        new_deepstack_feature_lists.append(tmp_embeddings_thd[vision_mask_thd].contiguous())

                    deepstack_feature_lists = new_deepstack_feature_lists

                vision_mask = vision_mask_thd
                combined_embeddings_thd = (
                    preprocess_packed_seqs(
                        combined_embeddings.transpose(0, 1).contiguous(),
                        attention_mask,
                        pre_process=True,
                        pg_collection=self.pg_collection,
                    )[0]
                    .transpose(0, 1)
                    .contiguous()
                )
                combined_embeddings = combined_embeddings_thd

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        else:
            combined_embeddings = None

        visual_pos_masks = vision_mask
        deepstack_visual_embeds = deepstack_feature_lists
        if self.config.sequence_parallel or cp_size > 1:
            if packed_seq_params is None:  # BSHD
                visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    tp_size=self.pg_collection.tp.size(),
                    tp_rank=self.pg_collection.tp.rank(),
                    cp_size=cp_size,
                    cp_rank=self.pg_collection.cp.rank(),
                    sequence_parallel=self.config.sequence_parallel,
                )
            elif self.config.sequence_parallel:  # THD and SP
                visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                    visual_pos_masks,
                    deepstack_visual_embeds,
                    tp_size=self.pg_collection.tp.size(),
                    tp_rank=self.pg_collection.tp.rank(),
                    cp_size=1,
                    cp_rank=0,
                    sequence_parallel=self.config.sequence_parallel,
                )

        if position_ids is None:
            # BSHD
            position_ids, _ = get_rope_index(
                self.config.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )  #  [3*b*s]
            if packed_seq_params is not None:
                # convert position_ids to THD format
                position_ids = (
                    preprocess_packed_seqs(
                        position_ids.permute(1, 2, 0),
                        attention_mask,
                        pre_process=True,
                        pg_collection=self.pg_collection,
                    )[0]
                    .permute(2, 0, 1)
                    .contiguous()
                )
                attention_mask = None
                self.language_model.rotary_pos_emb.is_thd_format = True

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Qwen3VLModel.forward.language_model")

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,  # Added for THD training compatibility
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
            **kwargs,
        )
        torch.cuda.nvtx.range_pop()

        return output
