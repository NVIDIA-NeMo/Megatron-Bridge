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
from megatron.core import InferenceParams, mpu, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig as Qwen3VLConfigHF
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel as Qwen3VLVisionModelHF

from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_config import Qwen3VLTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import get_rope_index, split_deepstack_embs
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from typing import Optional, Dict
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.core import parallel_state

class Qwen3VLModel(MegatronModule):
    """Qwen3VL multi-modal model.

    Args:
        language_transformer_config (TransformerConfig): Transformer config for the language model.
        language_transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers of the
        vision_transformer_config (TransformerConfig): Transformer config for the vision model, copy from HF config.
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
        modality_decoupled_parallel: bool = False,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
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

        # This attribute is needed to check if an all-reduce is required
        # on the word embeddings inside `finalize_model_grads._allreduce_word_embedding_grads`.
        self.share_embeddings_and_output_weights = False
        self.modality_decoupled_parallel = modality_decoupled_parallel
        self.execute_mode = None
        self.batch = None
        self.vision_embeds = None
        self.deepstack_feature_lists = None

        if self.pre_process or self.modality_decoupled_parallel:
            # Initialize vision model with random weights from config
            self.vision_model = Qwen3VLVisionModelHF._from_config(vision_transformer_config)
            # Ensure HF visual tower params are marked for TP grad sync and future assignments are hooked.
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            # Move to device if available
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")

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
        )
        assert len(vision_transformer_config.deepstack_visual_indexes) < len(self.language_model.decoder.layers), (
            "the deepstack_visual_embeds should on the first pp-stage",
            f"got {len(vision_transformer_config.deepstack_visual_indexes)} deepstack_visual_indexes, "
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

    def set_execute_mode(self, execute_mode: dict[str, bool]):
        self.execute_mode = execute_mode

    def get_execute_mode(self):
        return self.execute_mode

    def set_batch(self, batch):
        self.batch = batch

    def get_batch(self):
        return self.batch

    def set_vision_model_output(self, vision_model_output: tuple[torch.Tensor, torch.Tensor]):
        self.vision_embeds = vision_model_output["visual_embeds"]
        self.deepstack_feature_lists = vision_model_output["deepstack_visual_embeds"]

    def freeze(
        self,
        freeze_language_model: bool,
        freeze_vision_model: bool,
        freeze_vision_projection: bool,
    ):
        """Freeze model modules.

        Make specific modules non-trainable by setting requires_grad to False.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_vision_model (bool): Freeze the vision model module (patch_embed, blocks, pos_embed).
            freeze_vision_projection (bool): Freeze the vision projection modules (merger and deepstack_merger_list).
        """
        modules = []

        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)

        if freeze_vision_model and self.vision_model is not None:
            # Freeze vision encoder components (patch_embed, blocks, pos_embed, rotary_pos_emb)
            if hasattr(self.vision_model, "patch_embed"):
                modules.append(self.vision_model.patch_embed)
            if hasattr(self.vision_model, "blocks"):
                modules.append(self.vision_model.blocks)
            if hasattr(self.vision_model, "pos_embed"):
                modules.append(self.vision_model.pos_embed)
            if hasattr(self.vision_model, "rotary_pos_emb"):
                modules.append(self.vision_model.rotary_pos_emb)

        if freeze_vision_projection and self.vision_model is not None:
            # Freeze vision projection components (merger and deepstack_merger_list)
            if hasattr(self.vision_model, "merger"):
                modules.append(self.vision_model.merger)
            if hasattr(self.vision_model, "deepstack_merger_list"):
                modules.append(self.vision_model.deepstack_merger_list)

        for module in modules:
            for param in module.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,  # can set at dataset
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        # cat set at dataset
        image_input_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward function of the Qwen3VL model.

        Args:
            image_data (torch.Tensor): input image of shape [total_thw_size, n_features].
            input_ids (torch.Tensor): input text ids [batch, text_seq_len].
            position_ids (torch.Tensor): input text position ids [batch, text_seq_len].
            attention_mask (torch.Tensor): attention mask for the language model [batch, 1, combined_seq_len,
                combined_seq_len].
            labels (torch.Tensor): Optional target text labels [batch, combined_seq_len].
            inference_params (InferenceParams): Inference-time parameters including KV cache.

            video_start_index:
                0 -- all video
                len(video_seq) -- all image
                others -- mixture
            *_input_mask: should not be None in the first PP stage
        Returns:
            output (torch.Tensor): Loss of shape [b, s] if labels are provided, otherwise logits of shape
                [b, s, vocab_size].
        """
        assert pixel_values_videos is None and video_grid_thw is None, "not support video now"
        assert inference_params is None, "not support inference"
        if self.execute_mode["execute_vision_model_forward"]:
            if image_grid_thw is not None and image_grid_thw.shape[0] > 0:
                image_embeds, deepstack_image_embeds = self.vision_model(
                    hidden_states=pixel_values,
                    grid_thw=image_grid_thw,
                )
            # We do not support video now, so we only use image_embes as visual_embes.
            visual_embeds = image_embeds
            deepstack_visual_embeds = deepstack_image_embeds
            if not self.execute_mode["execute_language_model_forward"]:
                output_dict = {
                    "visual_embeds": visual_embeds,
                    "deepstack_visual_embeds": deepstack_visual_embeds,
                }
                return output_dict
        else:
            visual_embeds = self.vision_embeds
            deepstack_visual_embeds = self.deepstack_feature_lists

        image_mask = None
        visual_pos_masks = None
        # position ids is computed within the model
        position_ids = None
        if self.pre_process:
            visual_embeds = self.vision_embeds
            deepstack_visual_embeds = self.deepstack_feature_lists
            if image_grid_thw is not None:
                image_mask = image_input_mask
                if image_mask is None:
                    image_mask = (input_ids == self.image_token_id).contiguous()

            image_embeds = visual_embeds
            deepstack_image_embeds = deepstack_visual_embeds
            visual_pos_masks = image_mask

            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,  # NOTE: disable
            ).clone()  # [text_seq_len, b, h_language]

            if image_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                combined_embeddings[image_mask] = image_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()
        else:
            combined_embeddings = None

        if position_ids is None:
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

        if self.config.sequence_parallel:
            visual_pos_masks, deepstack_visual_embeds = split_deepstack_embs(
                visual_pos_masks,
                deepstack_visual_embeds,
                tp_size=mpu.get_tensor_model_parallel_world_size(),
                tp_rank=mpu.get_tensor_model_parallel_rank(),
            )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,  # None in encoder
            attention_mask=attention_mask,  # None in encoder
            decoder_input=combined_embeddings,  # only not None in the first decoder PP stage
            labels=labels,  # only not None in the last decoder PP stage
            inference_params=inference_params,  # currently always None
            packed_seq_params=packed_seq_params,  # currently always None
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **(extra_block_kwargs or {}),
        )

        return output

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.
        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.
        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.
        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        if self.modality_decoupled_parallel:
            # TODO(shifang): get the correct pp_id, tp_id, dp_id from the vision_model.
            rank = torch.distributed.get_rank()
            if rank == 0:
                pp_id = 0
                tp_id = 0
                dp_id = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            else:
                pp_id = 1
                tp_id = 0
                dp_id = parallel_state.get_data_parallel_rank(with_context_parallel=True)
            replica_id = (
                pp_id,
                tp_id,
                dp_id,
            )
            for name, param in self.named_parameters():
                if name.startswith("vision_model"):
                    assert name in sharded_state_dict, f"name {name} not in sharded_state_dict"
                    del sharded_state_dict[name]
                    sharded_state_dict[name] = make_tp_sharded_tensor_for_checkpoint(
                        tensor=param,
                        key=name,
                        replica_id=replica_id,
                        allow_shape_mismatch=True,
                    )

        return sharded_state_dict