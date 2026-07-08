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

"""EXAONE 4.5 VL model provider configuration for Megatron-Core."""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlockSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from transformers.models.exaone4_5.configuration_exaone4_5 import Exaone4_5_Config, Exaone4_5_VisionConfig

from megatron.bridge.models.exaone.exaone4.exaone4_provider import exaone4_layer_spec
from megatron.bridge.models.exaone.exaone45.modelling_exaone45.model import Exaone45Model
from megatron.bridge.models.gpt_provider import GPTModelProvider


def exaone_45_transformer_layer_spec(config: "Exaone45ModelProvider") -> ModuleSpec:
    """Create an EXAONE 4.5 layer spec backed by the EXAONE Post-LN layer pattern."""
    return exaone4_layer_spec(config)


def exaone_45_mtp_block_spec(
    config: "Exaone45ModelProvider", vp_stage: int | None = None
) -> MultiTokenPredictionBlockSubmodules | None:
    """Create an MTP block spec that preserves the EXAONE transformer layer."""
    if not config.mtp_num_layers:
        return None

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

    layer_spec = exaone_45_transformer_layer_spec(config)
    block_spec = TransformerBlockSubmodules(layer_specs=[layer_spec])
    return get_gpt_mtp_block_spec(config, block_spec, use_transformer_engine=True, vp_stage=vp_stage)


@dataclass
class Exaone45ModelProvider(GPTModelProvider):
    """Model provider for EXAONE 4.5 VL models.

    Note: num_query_groups in parent class corresponds to num_key_value_heads in HF config.
    """

    transformer_layer_spec: ModuleSpec | Callable[["Exaone45ModelProvider"], ModuleSpec] = (
        exaone_45_transformer_layer_spec
    )

    vision_config: Exaone4_5_VisionConfig = field(default_factory=lambda: Exaone4_5_VisionConfig())
    hf_text_config: Exaone4_5_Config | None = None

    image_token_id: int = 67
    video_token_id: int = 68
    vision_token_id: int = 67
    vision_start_token_id: int = 73
    vision_end_token_id: int = 74
    bos_token_id: int = 1
    eos_token_id: int = 53
    spatial_merge_size: int = 2

    rotary_base: float = 1000000.0
    rope_scaling: bool = False
    rope_scaling_factor: int = 1

    position_embedding_type: str = "rope"

    # Override to disable scattering embeddings for vision insertion
    scatter_embedding_sequence_parallel: bool = False

    # Freeze options for fine-tuning scenarios
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False
    freeze_mtp_model: bool = False
    # EXAONE 4.5 LLM architecture
    qk_layernorm: bool = True
    activation_func: Callable = F.silu
    normalization: str = "RMSNorm"
    add_bias_linear: bool = False
    add_qkv_bias: bool = False

    # Attention softmax FP32
    attention_softmax_in_fp32: bool = True
    # Dropout
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    # fusions
    apply_rope_fusion: bool = False
    bias_activation_fusion: bool = False
    bias_dropout_fusion: bool = False
    masked_softmax_fusion: bool = True
    gradient_accumulation_fusion: bool = False

    def provide(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> Exaone45Model:
        """Provide an EXAONE 4.5 VL model instance with vision and language components."""
        language_transformer_config = self
        hf_vision_config = self.vision_config

        language_transformer_layer_spec = self.transformer_layer_spec(self)

        model = Exaone45Model(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            vision_transformer_config=hf_vision_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
            vp_stage=vp_stage,
        )

        # Apply freeze options if any are enabled for fine-tuning
        if (
            self.freeze_language_model
            or self.freeze_vision_model
            or self.freeze_vision_projection
            or self.freeze_mtp_model
        ):
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
                freeze_mtp_model=self.freeze_mtp_model,
            )

        return model

    def provide_language_model(
        self,
        pre_process: bool | None = None,
        post_process: bool | None = None,
        vp_stage: int | None = None,
    ) -> MCoreGPTModel:
        """Provide just the language model component without vision."""
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
