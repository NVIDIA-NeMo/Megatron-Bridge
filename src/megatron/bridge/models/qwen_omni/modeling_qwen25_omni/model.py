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
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import (
    Qwen2_5OmniTalkerConfig,
    Qwen2_5OmniThinkerConfig,
    Qwen2_5OmniToken2WavConfig,
)

from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.thinker_model import Qwen25OmniThinkerModel
from megatron.bridge.models.qwen_omni.modeling_qwen25_omni.transformer_config import Qwen25OmniTransformerConfig


class Qwen25OmniModel(MegatronModule):
    """Qwen2.5 Omni Model.

    Top-level wrapper that delegates to Qwen25OmniThinkerModel.
    Same pattern as Qwen3OmniMoeModel but simpler (no deepstack, dense LLM).
    """

    def __init__(
        self,
        language_transformer_config: Qwen25OmniTransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        thinker_transformer_config: Qwen2_5OmniThinkerConfig,
        talker_transformer_config: Qwen2_5OmniTalkerConfig | None = None,
        token2wav_transformer_config: Qwen2_5OmniToken2WavConfig | None = None,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.thinker = Qwen25OmniThinkerModel(
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
        return self.thinker.set_input_tensor(input_tensor)

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
        return self.thinker.freeze(
            freeze_language_model,
            freeze_vision_model,
            freeze_audio_model,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_features=None,
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
        image_input_mask: torch.Tensor | None = None,
        video_input_mask: torch.Tensor | None = None,
        feature_attention_mask=None,
        audio_feature_lengths=None,
        cp_img_num: list[int] | None = None,
        images_padded: list[bool] | None = None,
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
