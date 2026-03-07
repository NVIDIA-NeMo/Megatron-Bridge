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
from megatron.core import InferenceParams
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
)

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import Qwen3OmniMoeThinkerModel
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import Qwen3OmniTransformerConfig


class Qwen3OmniMoeModel(MegatronModule):
    """Qwen3 Omni Moe Model."""

    def __init__(
        self,
        language_transformer_config: Qwen3OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        thinker_transformer_config: Qwen3OmniMoeThinkerConfig,
        talker_transformer_config: Qwen3OmniMoeTalkerConfig | None = None,
        code2wav_transformer_config: Qwen3OmniMoeCode2WavConfig | None = None,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.thinker = Qwen3OmniMoeThinkerModel(
            language_transformer_config,
            language_transformer_layer_spec,
            thinker_transformer_config,
            parallel_output,
            pre_process,
            post_process,
            add_encoder,
            add_decoder,
            pg_collection,
        )

    def shared_embedding_or_output_weight(self):
        """This is a convenience method to surface the language model's word embeddings, which is
        necessary for `finalize_model_grads._allreduce_word_embedding_grads`."""
        return self.thinker.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        return self.thinker.set_input_tensor(input_tensor)

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
            freeze_vision_projection (bool): Freeze the vision projection modules (merger and deepstack_merger_list).
            freeze_audio_model (bool): Freeze the audio model module.
        """
        return self.thinker.freeze(
            freeze_language_model,
            freeze_vision_model,
            freeze_vision_projection,
            freeze_audio_model,
        )

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
        return self.thinker(
            input_ids=input_ids,
            input_features=input_features,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_input_mask=image_input_mask,
            video_input_mask=video_input_mask,
            feature_attention_mask=feature_attention_mask,
            audio_feature_lengths=audio_feature_lengths,
            cp_img_num=cp_img_num,
            images_padded=images_padded,
            use_audio_in_video=use_audio_in_video,
            video_second_per_grid=video_second_per_grid,
            **kwargs,
        )
