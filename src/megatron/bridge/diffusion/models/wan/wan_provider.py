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


import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule

from megatron.bridge.diffusion.models.wan.wan_model import WanModel
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.models.transformer_config import TransformerConfig


logger = logging.getLogger(__name__)


@dataclass
class WanModelProvider(TransformerConfig, ModelProviderMixin[VisionModule]):  # noqa: D101
    crossattn_emb_size: int = 1536  # cross attention emebedding size after linear projection
    add_bias_linear: bool = True
    gated_linear_unit: bool = False

    num_layers: int = 30
    hidden_size: int = 1536
    ffn_hidden_size: int = 8960
    num_attention_heads: int = 12
    layernorm_epsilon: float = 1e-6
    normalization: str = "RMSNorm"
    layernorm_zero_centered_gamma: bool = False
    layernorm_across_heads: bool = True
    add_qkv_bias: bool = True
    rotary_interleaved: bool = True
    activation_func: Callable = F.gelu
    hidden_dropout: float = 0
    attention_dropout: float = 0
    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    bf16: bool = False
    params_dtype: torch.dtype = torch.float32
    qkv_format: str = "thd"  # "sbhd". NOTE: if we use context parallelism, we need to use "thd"
    apply_rope_fusion: bool = True
    bias_activation_fusion: bool = True
    # these attributes are unused for images/videos, we just set because bridge training requires for LLMs
    seq_length: int = 1024
    share_embeddings_and_output_weights: bool = False
    vocab_size: int = 25256 * 8
    make_vocab_size_divisible_by: int = 128

    # images/videos attributes
    in_channels: int = 16
    out_channels: int = 16
    patch_spatial: int = 2
    patch_temporal: int = 1
    freq_dim: int = 256
    text_len: int = 512
    text_dim: int = 4096

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> WanModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = WanModel

        return model(
            self,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
        )
