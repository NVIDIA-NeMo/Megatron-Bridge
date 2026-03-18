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
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import (
    Qwen3ASRThinkerConfig as Qwen3ASRThinkerConfigHF,
)
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    Qwen3ASRAudioEncoder as Qwen3ASRAudioEncoderHF,
)

from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.rope import get_rope_index
from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.transformer_config import Qwen3ASRTransformerConfig
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import Qwen3VLGPTModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
    split_data_cp_rank,
)
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync


class Qwen3ASRThinkerModel(MegatronModule):
    """Qwen3-ASR Thinker Model.

    Audio-only model that combines an HF audio encoder with a Qwen3-based language model.
    Follows the Qwen2.5-Omni thinker pattern but simplified for ASR (no vision/video).
    """

    def __init__(
        self,
        language_transformer_config: Qwen3ASRTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        thinker_transformer_config: Qwen3ASRThinkerConfigHF,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        language_transformer_layer_spec.submodules.self_attention.module = Qwen3VLSelfAttention

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder

        self.encoder_hidden_state = None
        self.audio_model = None
        self.language_model = None
        self.audio_token_id = language_transformer_config.audio_token_id
        self.audio_start_token_id = language_transformer_config.audio_start_token_id

        self.share_embeddings_and_output_weights = False
        if pg_collection is None:
            raise ValueError(
                "pg_collection is required for Qwen3ASRThinkerModel. "
                "Use ProcessGroupCollection.use_mpu_process_groups() to get the default collection."
            )
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
            # Use HF audio encoder directly (ReplicatedMapping in bridge)
            self.audio_model = Qwen3ASRAudioEncoderHF._from_config(thinker_transformer_config.audio_config)
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
        assert len(input_tensor) == 1, "input_tensor should only be length 1 for Qwen3ASR"

        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_audio_model: bool = False,
    ):
        """Freeze model modules.

        Args:
            freeze_language_model (bool): Freeze the language model module.
            freeze_audio_model (bool): Freeze the audio model module.
        """
        modules = []

        if freeze_language_model and self.language_model is not None:
            modules.append(self.language_model)

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
        """Extract audio features using the HF audio encoder.

        Follows the HF Qwen3ASRThinkerForConditionalGeneration.get_audio_features pattern:
        processes each audio individually for precision.
        """
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)

        if audio_feature_lengths is None:
            raise ValueError("Either feature_attention_mask or audio_feature_lengths must be provided")

        feature_lens = audio_feature_lengths

        # Cast input features to audio model dtype (e.g. bf16) to match conv2d weights
        audio_dtype = next(self.audio_model.parameters()).dtype
        input_features = input_features.to(dtype=audio_dtype)

        # Process each audio individually (same as HF implementation for precision)
        audio_features = []
        for input_feature, feature_len in zip(input_features, feature_lens, strict=True):
            # Convert 0-dim tensor to Python int for slicing
            feature_len_int = feature_len.item() if feature_len.dim() == 0 else int(feature_len)
            audio_output = self.audio_model(
                input_feature[:, :feature_len_int],
                feature_lens=torch.tensor([feature_len_int], device=input_feature.device, dtype=torch.long),
            )
            audio_features.append(audio_output.last_hidden_state)
        audio_features = torch.cat(audio_features, dim=0)

        return audio_features

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        loss_mask: torch.Tensor | None = None,
        inference_params: InferenceParams | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        extra_block_kwargs: dict | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        audio_feature_lengths: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if inference_params is not None:
            raise NotImplementedError("inference is not supported")
        if packed_seq_params is not None:
            raise NotImplementedError("packed_seq_params is not supported")

        cp_rank = self.pg_collection.cp.rank()
        cp_size = self.pg_collection.cp.size()

        if self.pre_process:
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

            # Replace audio token positions with audio embeddings
            if audio_embeds is not None:
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
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

        # Compute position IDs via get_rope_index if not provided
        if position_ids is None:
            position_ids, _ = get_rope_index(
                input_ids,
                attention_mask=attention_mask,
            )

        # Split position_ids for CP just like combined_embeddings
        # position_ids shape: [3, batch, seq_len] -> split on last dim
        if position_ids is not None and cp_size > 1 and packed_seq_params is None:
            position_ids = split_data_cp_rank(position_ids, cp_size, 2, cp_rank)

        # Pad position_ids to match SP-padded embeddings so rotary_pos_emb
        # has the same sequence length as the all-gathered query/key tensors.
        if sp_pad_len > 0 and position_ids is not None:
            # position_ids shape: [3, batch, seq_len] -> pad last dim
            position_ids = torch.nn.functional.pad(position_ids, (0, sp_pad_len), mode="replicate")

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
