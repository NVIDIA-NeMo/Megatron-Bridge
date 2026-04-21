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

from dataclasses import dataclass
from typing import Callable

import torch
from megatron.core import parallel_state
from megatron.core.activations import fast_gelu
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer.enums import AttnBackend

from megatron.bridge.models.gpt_provider import GPTModelProvider


@dataclass
class GemmaModelProvider(GPTModelProvider):
    """Configuration class for Gemma models."""

    # configs that are common across model sizes
    normalization: str = "RMSNorm"
    activation_func: Callable = fast_gelu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    seq_length: int = 8192
    kv_channels: int = 256
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = True
    # Note: different behavior compared to NeMo 1.0
    # NeMo 1.0 does not set layernorm_zero_centered_gamma and instead adds 1 in the HF -> NeMo conversion script
    # The present implementation is more in line with the official implementation
    layernorm_zero_centered_gamma: bool = True
    # Disable cuDNN attention since TE 1.8 does not support head dim > 128
    attention_backend: AttnBackend = AttnBackend.flash

    # Gemma defaults from HuggingFace
    layernorm_epsilon: float = 1e-06
    vocab_size: int = 256000
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "MCoreGPTModel":
        """Configure and instantiate a Megatron Core Gemma model.

        Extends the base configuration with Gemma-specific embedding scaling.

        Args:
            pre_process: Whether to include pre-processing in the model
            post_process: Whether to include post-processing in the model
            vp_stage: Virtual pipeline stage
            tokenizer: Tokenizer used with the model

        Returns:
            MCoreGPTModel: Configured Megatron Core GPT model instance
        """
        model = super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

        # Apply Embedding Scaling for Gemma: sqrt(hidden_size)
        if parallel_state.is_pipeline_first_stage(
            ignore_virtual=False,
            vp_stage=vp_stage,
        ):
            from megatron.bridge.models.gemma.modules import EmbeddingScalingMixin, extend_instance

            extend_instance(model.embedding, EmbeddingScalingMixin)

        return model


@dataclass
class GemmaModelProvider2B(GemmaModelProvider):
    """Configuration for a 2B parameter Gemma model.

    Specific configuration for the 2B Gemma model with 18 layers,
    2048 hidden size, and 8 attention heads.
    """

    num_layers: int = 18
    hidden_size: int = 2048
    num_attention_heads: int = 8
    num_query_groups: int = 1
    ffn_hidden_size: int = 16384


@dataclass
class GemmaModelProvider7B(GemmaModelProvider):
    """Configuration for a 7B parameter Gemma model.

    Specific configuration for the 7B Gemma model with 28 layers,
    3072 hidden size, and 16 attention heads.
    """

    num_layers: int = 28
    hidden_size: int = 3072
    num_attention_heads: int = 16
    num_query_groups: int = 16
    ffn_hidden_size: int = 24576


@dataclass
class CodeGemmaModelProvider2B(GemmaModelProvider2B):
    """Configuration for a 2B parameter Code Gemma model.

    Extends GemmaModelProvider with specific settings for code generation.
    Thism model has an identical configuration to GemmaModelProvider2B.
    """


@dataclass
class CodeGemmaModelProvider7B(GemmaModelProvider7B):
    """Configuration for a 7B parameter Code Gemma model.

    Extends GemmaModelProvider with specific settings for code generation.
    This model has an identical configuration to GemmaModelProvider7B.
    """
