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
from functools import partial
from typing import Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import ModuleSpec
from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig

from megatron.bridge.models.gpt_provider import GPTModelProvider

from .modeling_glm_45v import GLM45VModel


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


@dataclass
class GLM45VModelProvider(GPTModelProvider):
    """
    Base model provider for GLM 4.5 Vision-Language (VL) Models.
    Combines GLM 4.5 Air 106B language config with VL-specific settings.
    """

    # GLM MoE base config (from former GLMMoEModelProvider + GLM45AirModelProvider106B)
    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = partial(
        get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE
    )
    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    add_qkv_bias: bool = True
    seq_length: int = 131072
    init_method_std: int = 0.02
    hidden_dropout: float = 0.0
    vocab_size: int = 151552
    share_embeddings_and_output_weights: Optional[bool] = False
    layernorm_epsilon: float = 1e-5
    autocast_dtype: torch.dtype = torch.bfloat16
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True

    # Attention
    num_query_groups: int = 8
    num_attention_heads: int = 96
    attention_dropout: float = 0.0
    kv_channels: int = 128

    # RoPE
    position_embedding_type: str = "mrope"
    rotary_base: float = 1000000.0
    rotary_percent: float = 0.5
    mrope_section: List[int] = field(default_factory=lambda: [8, 12, 12])

    # MoE specific parameters
    moe_router_topk: int = 8
    moe_shared_expert_overlap: bool = True
    moe_token_dispatcher_type: str = "alltoall"
    moe_router_load_balancing_type: str = "seq_aux_loss"
    moe_aux_loss_coeff: float = 1e-3
    moe_router_pre_softmax: bool = False
    moe_grouped_gemm: bool = True
    moe_router_score_function: str = "sigmoid"
    moe_permute_fusion: bool = True
    moe_router_dtype: str = "fp32"
    moe_router_enable_expert_bias: bool = True
    moe_router_bias_update_rate: float = 0

    # Optimization
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True

    # MTP
    mtp_num_layers: Optional[int] = 1
    mtp_loss_scaling_factor: Optional[float] = 0.3

    # GLM 4.5 Air 106B specifics
    num_layers: int = 46
    num_moe_experts: int = 128
    hidden_size: int = 4096
    ffn_hidden_size: int = 10944
    moe_layer_freq: Union[int, List[int]] = field(default_factory=lambda: [0] * 1 + [1] * 45)
    moe_ffn_hidden_size: int = 1408
    moe_shared_expert_intermediate_size: int = 1408
    qk_layernorm: bool = False
    moe_router_topk_scaling_factor: float = 1.0

    # VL configuration
    scatter_embedding_sequence_parallel: bool = False

    # Vision configuration
    vision_config: Glm4vVisionConfig = field(default_factory=Glm4vVisionConfig)
    return_dict: bool = True

    # Token IDs
    eos_token_id: int = 151329
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
    image_token_id: int = 151363
    video_token_id: int = 151364

    # Freeze options
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> GLM45VModel:
        model = GLM45VModel(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        return GPTModelProvider.provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
