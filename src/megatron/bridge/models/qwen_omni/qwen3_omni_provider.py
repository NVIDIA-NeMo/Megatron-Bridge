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

from dataclasses import dataclass, field
from typing import Callable

import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeThinkerConfig,
)

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.qwen_omni.modeling_qwen3_omni.model import Qwen3OmniModel


@dataclass
class Qwen3OmniModelProvider(GPTModelProvider):
    """Provider for Qwen3-Omni.

    The current implementation focuses on thinker-side multimodal training and
    checkpoint conversion paths.
    """

    thinker_config: Qwen3OmniMoeThinkerConfig = field(default_factory=lambda: Qwen3OmniMoeThinkerConfig())
    talker_config: Qwen3OmniMoeTalkerConfig | None = None
    code2wav_config: Qwen3OmniMoeCode2WavConfig | None = None

    pretrained_model_name: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    hidden_dropout: float = 0.0
    qk_layernorm: bool = True
    moe_grouped_gemm: bool = True
    moe_router_load_balancing_type: str = "aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_pre_softmax: bool = False
    moe_token_dispatcher_type: str = "alltoall"
    moe_permute_fusion: bool = True

    image_token_id: int = 151655
    video_token_id: int = 151656
    audio_token_id: int = 151646
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    audio_start_token_id: int = 151647
    audio_end_token_id: int = 151648
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    language_max_sequence_length: int = 32768
    position_embedding_type: str = "mrope"
    position_id_per_seconds: int = 25
    seconds_per_chunk: int = 2
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    scatter_embedding_sequence_parallel: bool = False

    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_audio_model: bool = False
    vit_gradient_checkpointing: bool = False
    multimodal_attn_impl: str = "auto"

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        pp_group = self._pg_collection.pp if self._pg_collection is not None else None
        vp_size = self.virtual_pipeline_model_parallel_size
        if pre_process is None:
            pre_process = (
                is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_first_stage(pp_group)
                if pp_group is not None
                else True
            )
        if post_process is None:
            post_process = (
                is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size) and is_pp_last_stage(pp_group)
                if pp_group is not None
                else True
            )

        language_transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.num_moe_experts,
            moe_grouped_gemm=self.moe_grouped_gemm,
            qk_layernorm=self.qk_layernorm,
            fp8=False,
        )

        model = Qwen3OmniModel(
            language_transformer_config=self,
            language_transformer_layer_spec=language_transformer_layer_spec,
            thinker_transformer_config=self.thinker_config,
            talker_transformer_config=self.talker_config,
            code2wav_transformer_config=self.code2wav_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
        )

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_audio_model:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_audio_model=self.freeze_audio_model,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
