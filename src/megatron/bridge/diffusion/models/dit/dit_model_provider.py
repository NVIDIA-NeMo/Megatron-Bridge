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

import torch
from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.core import parallel_state
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.diffusion.common.utils.dynamic_import import dynamic_import
from megatron.bridge.diffusion.models.dit.dit_model import DiTCrossAttentionModel


logger = logging.getLogger(__name__)


@dataclass
class DiTModelProvider(TransformerConfig, ModelProviderMixin[VisionModule]):
    """
    Config for DiT-XL model
    """

    crossattn_emb_size: int = 1024
    add_bias_linear: bool = False
    gated_linear_unit: bool = False

    num_layers: int = 12
    hidden_size: int = 384
    max_img_h: int = 80
    max_img_w: int = 80
    max_frames: int = 34
    patch_spatial: int = 2
    patch_temporal: int = 1
    num_attention_heads: int = 6
    layernorm_epsilon = 1e-6
    normalization = "RMSNorm"
    add_bias_linear: bool = False
    qk_layernorm_per_head: bool = True
    layernorm_zero_centered_gamma: bool = False

    fp16_lm_cross_entropy: bool = False
    parallel_output: bool = True
    share_embeddings_and_output_weights: bool = True

    # max_position_embeddings: int = 5400
    hidden_dropout: float = 0
    attention_dropout: float = 0

    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    vae_module: str = "megatron.bridge.diffusion.common.tokenizers.cosmos.cosmos1.causal_video_tokenizer.CausalVideoTokenizer"
    vae_name: str = "Cosmos-0.1-Tokenizer-CV4x8x8"
    vae_cache_folder: str = None
    sigma_data: float = 0.5
    in_channels: int = 16
    layernorm_across_heads: bool = False

    # Validation parameters
    val_generation_guidance: float = 7.0
    val_generation_num_steps: int = 35

    replicated_t_embedder = True
    qkv_format: str = "thd"

    seq_length: int = 2048
    vocab_size: int = None
    make_vocab_size_divisible_by: int = 128

    loss_add_logvar: bool = True

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> DiTCrossAttentionModel:
        vp_size = self.virtual_pipeline_model_parallel_size
        if vp_size:
            p_size = self.pipeline_model_parallel_size
            assert (self.num_layers // p_size) % vp_size == 0, (
                "Make sure the number of model chunks is the same across all pipeline stages."
            )

        model = DiTCrossAttentionModel

        return model(
            self,
            fp16_lm_cross_entropy=self.fp16_lm_cross_entropy,
            parallel_output=self.parallel_output,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            max_img_h=self.max_img_h,
            max_img_w=self.max_img_w,
            max_frames=self.max_frames,
            patch_spatial=self.patch_spatial,
        )

    def configure_vae(self):
        return dynamic_import(self.vae_module).from_pretrained(self.vae_name, cache_dir=self.vae_cache_folder)


@dataclass
class DiTBModelProvider(DiTModelProvider):
    """DiT-B"""

    num_layers: int = 12
    hidden_size: int = 768
    num_attention_heads: int = 12


@dataclass
class DiTLModelProvider(DiTModelProvider):
    """DiT-L"""

    num_layers: int = 24
    hidden_size: int = 1024
    num_attention_heads: int = 16


@dataclass
class DiTXLModelProvider(DiTModelProvider):
    """DiT-XL"""

    num_layers: int = 28
    hidden_size: int = 1152
    num_attention_heads: int = 16
