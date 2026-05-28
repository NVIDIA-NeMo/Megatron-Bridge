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
from megatron.core.pipeline_parallel.utils import is_pp_last_stage
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
    get_dist_train_vision_dp_data,
    get_vision_cp_data,
    pack_dist_train_vision_module_output,
    preprocess_packed_seqs,
    qwen3vl_cp_split,
    reorganize_inputs,
    split_data_cp_rank,
    split_deepstack_embs,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.vision_model import Qwen3VLVisionModel

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


def _compact_thd_cp_index(
    packed_seq_params: PackedSeqParams,
    total_tokens: int,
    cp_size: int,
    cp_rank: int,
) -> torch.Tensor | None:
    """Return the CP-local token indices for an already-packed THD stream."""
    if cp_size <= 1:
        return None
    if tex is None:
        raise RuntimeError("QWEN3VL_THD_COMPACT_PACKING with CP>1 requires transformer_engine_torch")
    return tex.thd_get_partitioned_indices(
        packed_seq_params.cu_seqlens_q_padded,
        total_tokens,
        cp_size,
        cp_rank,
    )


def _pack_position_ids_to_compact_thd(
    position_ids: torch.Tensor,
    packed_seq_params: PackedSeqParams,
) -> torch.Tensor:
    """Pack BSHD MRoPE position IDs into the same compact THD layout as input_ids.

    Qwen3-VL's ``get_rope_index`` understands per-sample vision placeholder
    order best in BSHD form. Compact THD computes those BSHD position IDs first,
    then copies each real sample span into the physical padded THD segment.
    Alignment padding keeps the default zero position IDs and is masked later.
    """
    real_lengths = packed_seq_params.cu_seqlens_q[1:] - packed_seq_params.cu_seqlens_q[:-1]
    cu_seqlens_padded = packed_seq_params.cu_seqlens_q_padded
    total_padded_len = int(cu_seqlens_padded[-1].item())
    packed_position_ids = torch.zeros(
        position_ids.size(0),
        1,
        total_padded_len,
        dtype=position_ids.dtype,
        device=position_ids.device,
    )
    for batch_idx in range(real_lengths.numel()):
        real_len = int(real_lengths[batch_idx].item())
        start = int(cu_seqlens_padded[batch_idx].item())
        packed_position_ids[:, 0, start : start + real_len] = position_ids[:, batch_idx, :real_len]
    return packed_position_ids


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
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if hasattr(language_transformer_layer_spec, "submodules"):
            language_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention

        self.vision_transformer_config = vision_transformer_config
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
        self.vp_stage = None
        self.vp_size = self.config.virtual_pipeline_model_parallel_size

        if hasattr(self.config, "dist_train") and getattr(self.config.dist_train, "use_dist_train", False) is True:
            self.use_dist_train = True
            self.vision_to_llm_dp_ratio = self.config.dist_train.vision_to_llm_dp_ratio
            self.vision_embeds = None
            self.deepstack_feature_lists = None
            assert not (self.add_encoder and self.add_decoder) and (self.add_encoder or self.add_decoder), (
                "add_encoder and add_decoder should not be both True or both False "
                f"if use_dist_train is True, got {self.add_encoder} and {self.add_decoder}"
            )
            assert self.pg_collection.cp.size() == 1, (
                "currently, dist train does not support context parallelism for encoder."
            )
        else:
            self.use_dist_train = False

        if self.pre_process and self.add_encoder:
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
                pg_collection=pg_collection,
            )
        if self.add_decoder:
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
                mtp_block_spec=mtp_block_spec,
                vp_stage=vp_stage,
                pg_collection=pg_collection,
            )
            if pre_process:
                deepstack_indexes = getattr(vision_transformer_config, "deepstack_visual_indexes", [])
                assert len(deepstack_indexes) <= len(self.language_model.decoder.layers), (
                    "the deepstack_visual_embeds should on the first pp-stage of language model",
                    f"got {len(deepstack_indexes)} deepstack_visual_indexes, "
                    f" {len(self.language_model.decoder.layers)} language model layers",
                )

            self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

        if self.pg_collection.cp.size() > 1:
            assert self.config.calculate_per_token_loss, (
                "Qwen3-VL model only supports context parallelism with calculate_per_token_loss enabled"
            )

        self._expose_language_model_for_cuda_graph_helper()

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        """Expose LM fields on the VLM root for the CUDA graph helper when cuda_graph_impl is enabled.

        The CUDA graph helper expects ``position_embedding_type``, ``rotary_pos_emb``, and ``decoder`` on
        the model, but in Qwen3-VL these live on ``language_model``. Assigning ``decoder`` here shadows
        the :meth:`decoder` property for this instance only when graphs are used.
        """
        llm_cuda_graph_enabled = (
            self.language_model is not None
            and getattr(self.language_model.config, "cuda_graph_impl", "none") != "none"
        )
        if not llm_cuda_graph_enabled:
            return
        assert not self.language_model.config.variable_seq_lengths, (
            "Qwen3-VL MoE with CUDA graph requires fixed sequence lengths (variable_seq_lengths=False). "
            "Disable variable-length / packed pipelines (e.g. in-batch packing with PP, or other modes "
            "that set variable_seq_lengths) or turn off CUDA graph."
        )
        self.position_embedding_type = self.language_model.position_embedding_type
        self.rotary_pos_emb = self.language_model.rotary_pos_emb
        self.decoder = self.language_model.decoder

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        if self.add_decoder:
            return self.language_model.shared_embedding_or_output_weight()
        return None

    @property
    def decoder(self):
        """Expose language model decoder for mcore inference compatibility.

        mcore's MambaInferenceStateConfig.from_model() calls get_attr_wrapped_model(model, "decoder"),
        which only traverses .module wrappers. VLM models store the decoder under language_model.decoder,
        so we expose it here to allow the Mamba check to run and correctly return None.
        """
        return getattr(self.language_model, "decoder", None)

    def set_dist_train_input_tensors(self, input_tensor) -> None:
        """Set input tensor for the model for dist train.

        Args:
            input_tensor (list): Input tensor.
        """
        assert isinstance(input_tensor, list), f"Input tensor must be a list, but got {type(input_tensor)}"
        assert len(input_tensor) == 1, f"Input tensor must be a list of length 1, but got {len(input_tensor)}"
        assert isinstance(input_tensor[0], dict), (
            f"Input tensor[0] must be a dictionary, but got {type(input_tensor[0])}"
        )
        input_dict = input_tensor[0]

        if "vision_module" in input_dict:
            vision_module_output_tensor = input_dict["vision_module"]
            assert vision_module_output_tensor.dim() == 3, (
                f"vision_module must be 3D [b, s, h], got shape {tuple(vision_module_output_tensor.shape)}"
            )
            # bridge communicator send and receive tensors in 3D shape, [batch, seq, hidden].
            # Qwen3VL model needs vision embeds in 2D [batch*seq, hidden].
            # So we merge leading dims, e.g. [b, s, h] -> [b*s, h].
            d0, d1, d2 = vision_module_output_tensor.shape
            vision_module_output_tensor = vision_module_output_tensor.reshape(d0 * d1, d2)
            num_chunks = len(self.vision_transformer_config.deepstack_visual_indexes) + 1
            chunks = torch.chunk(vision_module_output_tensor, chunks=num_chunks, dim=0)
            self.vision_embeds = chunks[-1]
            self.deepstack_feature_lists = chunks[:-1]
        if "language_module" in input_dict:
            self.language_model.set_input_tensor(input_dict["language_module"])

    def set_input_tensor(self, input_tensor) -> None:
        """Set input tensor for the model.

        Args:
            input_tensor (list): Input tensor.
        """
        if self.use_dist_train:
            self.set_dist_train_input_tensors(input_tensor)
            return
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
        mm_token_type_ids: torch.Tensor = None,
        moe_padding_mask: torch.Tensor = None,
        qwen3vl_thd_compact_packing: bool = False,
        qwen3vl_compact_input_ids_bshd: torch.Tensor = None,
        qwen3vl_compact_attention_mask_bshd: torch.Tensor = None,
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
            mm_token_type_ids (torch.Tensor): Token type IDs from transformers >= 5.3.0 processors.
                Not used by Qwen3VL (which computes its own rope positions).

        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """
        del inference_context, mm_token_type_ids  # Unused, kept for API compatibility
        assert inference_params is None, "not support inference"

        vision_grid_thw = None
        vision_data = None
        vision_mask = None
        vision_embeds = None
        deepstack_feature_lists = None

        # position ids is computed within the model
        position_ids = None

        torch.cuda.nvtx.range_push("Qwen3VLModel.forward.pre_process")

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()

        # input_ids to pass to the language model for MTP (Multi-Token Prediction).
        # MTP's _get_embeddings rolls input_ids to generate future-token embeddings,
        # so it must be a real tensor. For packed sequences we use the THD-format
        # input_ids_thd (updated below); for regular sequences we use input_ids as-is.
        lm_input_ids = input_ids
        # In compact THD mode qwen3_vl_step has already changed input_ids from
        # BSHD [B, S] to THD [1, T]. Cache the full physical length before any
        # CP-local index_select so we can avoid double-splitting labels/masks.
        compact_total_tokens = input_ids.size(1) if qwen3vl_thd_compact_packing else None
        compact_cp_index = None
        if qwen3vl_thd_compact_packing:
            assert packed_seq_params is not None, "QWEN3VL_THD_COMPACT_PACKING requires packed_seq_params"
            compact_cp_index = _compact_thd_cp_index(
                packed_seq_params,
                compact_total_tokens,
                cp_size,
                cp_rank,
            )

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
                    if self.use_dist_train:
                        if self.vision_model is not None:
                            vision_data, vision_grid_thw = get_dist_train_vision_dp_data(
                                vision_data,
                                vision_grid_thw,
                                num_chunks=self.vision_to_llm_dp_ratio,
                                dp_rank=self.pg_collection.dp.rank(),
                            )
                            vision_embeds, deepstack_feature_lists = self.vision_model(
                                hidden_states=vision_data,
                                grid_thw=vision_grid_thw,
                            )
                            output_vision_module = pack_dist_train_vision_module_output(
                                vision_embeds,
                                deepstack_feature_lists,
                            )
                            torch.cuda.nvtx.range_pop()
                            return output_vision_module
                        else:
                            vision_embeds = self.vision_embeds
                            deepstack_feature_lists = self.deepstack_feature_lists
                    else:
                        vision_embeds, deepstack_feature_lists = self.vision_model(
                            hidden_states=vision_data,
                            grid_thw=vision_grid_thw,
                        )
                else:
                    vision_embeds = torch.zeros(
                        (0, self.config.hidden_size),
                        device=vision_data.device,
                        dtype=torch.bfloat16,
                    )
                    deepstack_feature_lists = []
                    for _ in self.vision_transformer_config.deepstack_visual_indexes:
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
            if packed_seq_params is not None and not qwen3vl_thd_compact_packing:
                # Legacy Qwen3-VL THD path: model.forward receives padded BSHD
                # input_ids so it can do vision/text combine first, then this
                # helper converts BSHD -> CP-local THD.
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
                input_ids_thd, _ = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True, pg_collection=self.pg_collection
                )
                lm_input_ids = input_ids_thd
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

            if self.config.sequence_parallel and not qwen3vl_thd_compact_packing:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        else:
            combined_embeddings = None
            # On non-pre_process PP stages (e.g. the last stage where MTP runs),
            # convert lm_input_ids to THD format so it matches position_ids.
            if packed_seq_params is not None and not qwen3vl_thd_compact_packing:
                # Same legacy conversion is needed on later PP stages because MTP
                # reads input_ids even though embeddings were produced upstream.
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=input_ids.device)
                lm_input_ids, _ = preprocess_packed_seqs(
                    input_ids, attention_mask, pre_process=True, pg_collection=self.pg_collection
                )

        if qwen3vl_thd_compact_packing:
            # Compact path: input_ids/combined_embeddings are already global THD.
            # Keep the old invariant "combine before CP split", but use TE's THD
            # partition index instead of preprocess_packed_seqs' BSHD conversion.
            if compact_cp_index is not None:
                if self.pre_process and combined_embeddings is not None:
                    full_vision_mask = vision_mask
                    local_vision_mask = (
                        full_vision_mask.index_select(1, compact_cp_index) if full_vision_mask is not None else None
                    )
                    if deepstack_feature_lists is not None and full_vision_mask is not None:
                        # Deepstack features are produced only at visual token
                        # positions. Re-materialize them in full THD layout, CP
                        # slice, then gather back the local visual positions.
                        full_embeddings_bsh = combined_embeddings.transpose(0, 1).contiguous()
                        new_deepstack_feature_lists = []
                        for deepstack_visual_embed in deepstack_feature_lists:
                            tmp_embeddings = torch.zeros_like(full_embeddings_bsh)
                            tmp_embeddings[full_vision_mask] = deepstack_visual_embed
                            tmp_embeddings = tmp_embeddings.index_select(1, compact_cp_index)
                            new_deepstack_feature_lists.append(tmp_embeddings[local_vision_mask].contiguous())
                        deepstack_feature_lists = new_deepstack_feature_lists
                    combined_embeddings = combined_embeddings.index_select(0, compact_cp_index)
                    vision_mask = local_vision_mask

                # input_ids is passed to language_model for MTP/future-token
                # embedding paths; it must match the CP-local decoder_input.
                input_ids = input_ids.index_select(1, compact_cp_index)
                lm_input_ids = input_ids
                if labels is not None and labels.dim() >= 2 and labels.size(1) == compact_total_tokens:
                    labels = labels.index_select(1, compact_cp_index)
                if loss_mask is not None and loss_mask.dim() >= 2 and loss_mask.size(1) == compact_total_tokens:
                    loss_mask = loss_mask.index_select(1, compact_cp_index)
                if (
                    moe_padding_mask is not None
                    and moe_padding_mask.dim() >= 2
                    and moe_padding_mask.size(1) == compact_total_tokens
                ):
                    moe_padding_mask = moe_padding_mask.index_select(1, compact_cp_index)

            if self.pre_process and self.config.sequence_parallel:
                # Legacy path scatters after BSHD->THD conversion. Compact path
                # skips that block, so SP scatter has to happen here instead.
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
            if self.config.sequence_parallel and moe_padding_mask is not None:
                tp_size = self.pg_collection.tp.size()
                if tp_size > 1:
                    tp_rank = self.pg_collection.tp.rank()
                    assert moe_padding_mask.size(1) % tp_size == 0, (
                        "compact THD MoE padding mask must be divisible by TP size under sequence parallelism"
                    )
                    seq_len_per_tp = moe_padding_mask.size(1) // tp_size
                    start = tp_rank * seq_len_per_tp
                    moe_padding_mask = moe_padding_mask[:, start : start + seq_len_per_tp].contiguous()

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

        if qwen3vl_thd_compact_packing:
            if position_ids is None:
                assert qwen3vl_compact_input_ids_bshd is not None, "compact THD packing requires original BSHD input_ids"
                # MRoPE must see per-sample BSHD order, otherwise [1, T] would
                # look like one long sample and image/video grids would be
                # consumed with the wrong boundaries.
                position_ids, _ = get_rope_index(
                    self.config.spatial_merge_size,
                    self.image_token_id,
                    self.video_token_id,
                    self.vision_start_token_id,
                    qwen3vl_compact_input_ids_bshd,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=qwen3vl_compact_attention_mask_bshd,
                )
                position_ids = _pack_position_ids_to_compact_thd(position_ids, packed_seq_params)
            if compact_cp_index is not None:
                position_ids = position_ids.index_select(2, compact_cp_index)
            # THD attention receives boundaries via packed_seq_params. A dense
            # mask would describe BSHD layout and is intentionally disabled.
            attention_mask = None
            self.language_model.rotary_pos_emb.is_thd_format = True
        elif position_ids is None:
            # BSHD
            # Megatron uses 4D bool masks ([B|1,1,S,S], True=masked); HF uses 2D keep masks ([B,S], 1=keep)
            # For simplicity, we set hf_attention_mask to None.
            hf_attention_mask = None
            position_ids, _ = get_rope_index(
                self.config.spatial_merge_size,
                self.image_token_id,
                self.video_token_id,
                self.vision_start_token_id,
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=hf_attention_mask,
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
            input_ids=lm_input_ids,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            loss_mask=loss_mask,  # Added for THD training compatibility
            padding_mask=moe_padding_mask,
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            runtime_gather_output=runtime_gather_output,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
            **kwargs,
        )
        torch.cuda.nvtx.range_pop()
        if self.use_dist_train:
            if not is_pp_last_stage(self.pg_collection.pp):
                return {"language_module": output}
        return output
