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

"""
Ministral 3 Model Provider configurations for Megatron-Core.

This module provides configuration classes for Ministral 3 models (3B, 8B, 14B variants),
compatible with HuggingFace's Ministral-3 model configurations.

Reference: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512

Ministral 3 Key Features:
- Vision-language capabilities with separate language model and vision encoder
- Large context window (up to 256k tokens)
- Available in Base, Instruct, and Reasoning variants
- Edge-optimized for deployment on various hardware
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel as MCoreGPTModel

from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


if TYPE_CHECKING:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    from megatron.bridge.models.ministral3.modeling_ministral3 import Ministral3Model


logger = logging.getLogger(__name__)


# =============================================================================
# Ministral 3 Vision-Language Model Providers
# =============================================================================


@dataclass
class Ministral3ModelProvider(MistralModelProvider):
    """
    Base model provider for Ministral 3 Vision-Language Models.

    Ministral 3 is a family of edge-optimized vision-language models combining
    a language model with a vision encoder for multimodal capabilities.

    Reference:
    - https://huggingface.co/mistralai/Ministral-3-3B-Base-2512
    - https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
    - https://huggingface.co/mistralai/Ministral-3-14B-Base-2512
    """

    normalization: str = "RMSNorm"
    activation_func: Callable = F.silu
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    num_attention_heads: int = 32
    num_query_groups: int = 8
    kv_channels: int = 128

    seq_length: int = 32768  # Default, can be extended to 256k
    position_embedding_type: str = "yarn"
    rotary_base: int = 1000000
    yarn_rotary_scaling_factor: float = 16.0
    yarn_original_max_position_embeddings: int = 16384
    yarn_beta_fast: float = 32.0
    yarn_beta_slow: float = 1.0
    yarn_correction_range_round_to_int: bool = False
    yarn_mscale: Optional[float] = 1.0
    yarn_mscale_all_dim: Optional[float] = 1.0  # todo llama_4_scaling_beta

    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    init_method_std: float = 0.02
    layernorm_epsilon: float = 1e-5
    params_dtype: torch.dtype = torch.bfloat16
    bf16: bool = True

    # VL models shouldn't scatter embeddings across sequence parallel regions
    # because vision embeddings are inserted into language embeddings
    scatter_embedding_sequence_parallel: bool = False

    hf_config: Optional["Mistral3Config"] = None

    # Vision-specific token IDs (defaults, actual values come from HF config)
    image_token_id: int = 10

    # Freeze options for fine-tuning
    freeze_language_model: bool = False
    freeze_vision_model: bool = False
    freeze_vision_projection: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None) -> "Ministral3Model":
        """
        Provide a Ministral3Model instance with vision and language components.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            Ministral3Model instance with HF vision encoder and Megatron language model
        """
        from megatron.bridge.models.ministral3.modeling_ministral3 import Ministral3Model

        model = Ministral3Model(
            config=self,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        # Apply freeze options if any are enabled for fine-tuning
        if self.freeze_language_model or self.freeze_vision_model or self.freeze_vision_projection:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_vision_model=self.freeze_vision_model,
                freeze_vision_projection=self.freeze_vision_projection,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> MCoreGPTModel:
        """
        Provide just the language model component without vision.

        Args:
            pre_process: Whether this is the first stage in pipeline parallelism
            post_process: Whether this is the last stage in pipeline parallelism
            vp_stage: Virtual pipeline stage number

        Returns:
            MCoreGPTModel instance (language model only)
        """
        # Use parent class to create standard language model
        return super().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)


@dataclass
class Ministral3ModelProvider3B(Ministral3ModelProvider):
    """
    Config for Ministral 3 3B Vision-Language Model.

    Reference: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512

    Model specs:
    - 3.4B Language Model + 0.4B Vision Encoder
    """

    hidden_size: int = 3072
    ffn_hidden_size: int = 9216
    num_layers: int = 26
    share_embeddings_and_output_weights: bool = True


@dataclass
class Ministral3ModelProvider8B(Ministral3ModelProvider):
    """
    Config for Ministral 3 8B Vision-Language Model.

    Reference: https://huggingface.co/mistralai/Ministral-3-8B-Base-2512

    Model specs:
    - 8.4B Language Model + 0.4B Vision Encoder
    """

    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_layers: int = 34


@dataclass
class Ministral3ModelProvider14B(Ministral3ModelProvider):
    """
    Config for Ministral 3 14B Vision-Language Model.

    Reference: https://huggingface.co/mistralai/Ministral-3-14B-Base-2512

    Model specs:
    - 13.5B Language Model + 0.4B Vision Encoder
    """

    hidden_size: int = 5120
    ffn_hidden_size: int = 16384
    num_layers: int = 40
    rotary_base: int = 1000000000.0
