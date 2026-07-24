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

from typing import Optional

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
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config

from megatron.bridge.models.exaone.exaone45.modelling_exaone45.rope import get_rope_index
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.text_model import Exaone45GPTModel
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.transformer_config import (
    Exaone45TransformerConfig,
    get_vision_model_config,
)
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.utils import (
    AllGatherVisionEmbeddings,
    Exaone45_cp_split,
    PatchMergerSubmodules,
    collapse_thw,
    get_vision_cp_data,
    reorganize_inputs,
    split_data_cp_rank,
)
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.vision_model import Exaone45VisionModel


class Exaone45Model(MegatronModule):
    """Exaone45 multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
        vision_transformer_config: HF config for the vision model.
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
        language_transformer_config: Exaone45TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        vision_transformer_config: Exaone4_5_Config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

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
        self.position_embedding_type = language_transformer_config.position_embedding_type
        if self.position_embedding_type not in ("rope", "mrope"):
            raise ValueError(
                "Exaone45Model only supports position_embedding_type='rope' or 'mrope', "
                f"got {self.position_embedding_type!r}"
            )

        self.spatial_merge_size = vision_transformer_config.spatial_merge_size
        self.square_merge_size = vision_transformer_config.spatial_merge_size**2

        # process groups
        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()

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

        self.vp_stage = vp_stage
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        if self.pre_process:
            vision_transformer_layer_spec = get_vit_layer_with_transformer_engine_spec()
            vision_patch_merger_spec = PatchMergerSubmodules(
                patch_norm=TENorm,
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            )
            megatron_vision_transformer_config = get_vision_model_config(
                vision_transformer_config, megatron_config=language_transformer_config
            )
            megatron_vision_transformer_config.context_parallel_size = 1
            megatron_vision_transformer_config.pipeline_model_parallel_size = 1
            megatron_vision_transformer_config.first_pipeline_num_layers = None
            megatron_vision_transformer_config.add_qkv_bias = True

            self.vision_model = Exaone45VisionModel(
                megatron_vision_transformer_config,
                vision_transformer_layer_spec,
                vision_patch_merger_spec,
                pre_process=True,
                post_process=True,
                pg_collection=pg_collection,
            )

        if language_transformer_config.mtp_num_layers is not None and language_transformer_config.mtp_num_layers >= 1:
            from megatron.bridge.models.exaone.exaone45.exaone45_provider import exaone_45_mtp_block_spec

            mtp_block_spec = exaone_45_mtp_block_spec(language_transformer_config, vp_stage=vp_stage)
            self.use_mtp_postprocess = True
        else:
            mtp_block_spec = None
            self.use_mtp_postprocess = False

        self.language_model = Exaone45GPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.hf_text_config.max_position_embeddings,
            parallel_output=parallel_output,
            position_embedding_type=self.position_embedding_type,
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            rope_scaling=language_transformer_config.rope_scaling,
            rope_scaling_factor=language_transformer_config.rope_scaling_factor,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
            pg_collection=pg_collection,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

        self._expose_language_model_for_cuda_graph_helper()

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        """Expose LM fields on the VLM root for the CUDA graph helper when enabled."""
        llm_cuda_graph_enabled = (
            self.language_model is not None
            and getattr(self.language_model.config, "cuda_graph_impl", "none") != "none"
        )
        if not llm_cuda_graph_enabled:
            return
        assert not self.language_model.config.variable_seq_lengths, (
            "EXAONE 4.5 with CUDA graph requires fixed sequence lengths (variable_seq_lengths=False). "
            "Disable variable-length / packed pipelines or turn off CUDA graph."
        )
        self.position_embedding_type = self.language_model.position_embedding_type
        self.rotary_pos_emb = self.language_model.rotary_pos_emb
        self.__dict__["decoder"] = self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    @property
    def decoder(self):
        """Expose language model decoder for mcore inference compatibility."""
        return getattr(self.language_model, "decoder", None)

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Exaone45"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
        freeze_mtp_model: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False for the module's parameters.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module.
            freeze_vision_projection (bool): Freeze the vision projection module.
            freeze_mtp_model (bool): Freeze the multi-token prediction module.
        """
        modules = []
        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)
        if freeze_vision_model and self.vision_model is not None:
            modules.append(self.vision_model)
        if freeze_vision_projection and self.vision_model is not None:
            modules.append(self.vision_model.merger)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

        if freeze_vision_model and not freeze_vision_projection:
            if self.vision_model is not None:
                for param in self.vision_model.merger.parameters():
                    param.requires_grad = True

        if self.language_model is not None:
            for name, param in self.language_model.named_parameters():
                if "mtp" not in name:
                    continue
                if freeze_language_model and not freeze_mtp_model:
                    param.requires_grad = True
                elif freeze_mtp_model:
                    param.requires_grad = False

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
        """Forward function of the Exaone45 model.
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
        del inference_context  # Unused, kept for API compatibility
        assert inference_params is None, "not support inference"

        vision_grid_thw = None
        vision_data = None
        vision_mask = None

        # position ids is computed within the model
        torch.cuda.nvtx.range_push("Exaone45Model.forward.pre_process")

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
                if cp_size > 1:
                    vision_data, vision_grid_thw, cp_img_num, images_padded = Exaone45_cp_split(
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
                    vision_embeds = self.vision_model(
                        hidden_states=vision_data,
                        grid_thw=vision_grid_thw,
                    )

                else:
                    vision_embeds = torch.zeros(
                        (0, self.language_model.config.hidden_size),
                        device=vision_data.device,
                        dtype=torch.bfloat16,
                    )
                if cp_size > 1:
                    vision_embeds = AllGatherVisionEmbeddings.apply(
                        vision_embeds,
                        seqlen_on_cp_ranks,
                    )

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if vision_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[vision_mask] = vision_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if combined_embeddings is not None and cp_size > 1:
                # combined_embeddings shape: {seq_length, micro_batch_size, hidden_size}
                combined_embeddings = split_data_cp_rank(combined_embeddings, cp_size, 0, cp_rank)

                if attention_mask is not None and cp_size > 1:
                    attention_mask = split_data_cp_rank(attention_mask, cp_size, 2, cp_rank)

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    combined_embeddings, group=self.pg_collection.tp
                )
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        if loss_mask is not None and cp_size > 1:
            # loss_mask shape: {micro_batch_size, seq_length}
            loss_mask = split_data_cp_rank(loss_mask, cp_size, 1, cp_rank, packed_seq_params)

        if labels is not None and cp_size > 1:
            # labels shape: {micro_batch_size, seq_length}
            labels = split_data_cp_rank(labels, cp_size, 1, cp_rank, packed_seq_params)

        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push("Exaone45Model.forward.language_model")

        if self.position_embedding_type == "mrope" and (position_ids is None or position_ids.ndim == 2):
            position_ids, _ = get_rope_index(
                spatial_merge_size=self.spatial_merge_size,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
        else:
            position_ids = None

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,  # Added for THD training compatibility
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            runtime_gather_output=runtime_gather_output,
            **(extra_block_kwargs or {}),
            **kwargs,
        )
        torch.cuda.nvtx.range_pop()

        if loss_mask is None:
            return output
        else:
            return output, loss_mask
