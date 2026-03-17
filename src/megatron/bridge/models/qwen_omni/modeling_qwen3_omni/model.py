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

from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.thinker_model import Qwen3OmniThinkerModel
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.transformer_config import (
    Qwen3OmniTransformerConfig,
)


class Qwen3OmniModel(MegatronModule):
    """Qwen3-Omni model wrapper."""

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
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.thinker = Qwen3OmniThinkerModel(
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

        self.talker_transformer_config = talker_transformer_config
        self.code2wav_transformer_config = code2wav_transformer_config

    def shared_embedding_or_output_weight(self):
        return self.thinker.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        return self.thinker.set_input_tensor(input_tensor)

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_audio_model: bool = False,
    ):
        return self.thinker.freeze(
            freeze_language_model=freeze_language_model,
            freeze_vision_model=freeze_vision_model,
            freeze_audio_model=freeze_audio_model,
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
        **kwargs,
    ) -> torch.Tensor:
        return self.thinker(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            **kwargs,
        )
