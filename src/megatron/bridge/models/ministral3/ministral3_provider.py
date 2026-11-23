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
Ministral 3 Model Provider configuration for Megatron-Core.

This module provides a provider class for Ministral 3 models, compatible with
HuggingFace's Ministral-3 model configurations.

Reference: https://huggingface.co/mistralai/Ministral-3-3B-Base-2512

Ministral 3 Key Features:
- Vision-language capabilities with separate language model and vision encoder
- Large context window (up to 256k tokens)
- Available in Base, Instruct, and Reasoning variants
- Edge-optimized for deployment on various hardware
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn.functional as F
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.transformer import ModuleSpec

from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


if TYPE_CHECKING:
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    from megatron.bridge.models.ministral3.modeling_ministral3 import Ministral3Model


logger = logging.getLogger(__name__)
from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import TEDotProductAttention as MCoreTEDotProductAttention
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType

from megatron.bridge.models.gpt_provider import GPTModelProvider, default_layer_spec
from megatron.bridge.training.utils.packed_seq_utils import (
    get_packed_seq_cp_partition_indices,
    get_packed_seq_q_cu_seqlens,
)


# =============================================================================
# Ministral 3 Vision-Language Model Providers
# =============================================================================


def ministral_layer_spec(config: "GPTModelProvider") -> ModuleSpec:
    """Layer spec for Ministral 3 models."""
    layer_spec = default_layer_spec(config)
    layer_spec.submodules.self_attention.submodules.core_attention = MinistralTEDotProductAttention
    return layer_spec


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

    transformer_layer_spec: Union[ModuleSpec, Callable[["GPTModelProvider"], ModuleSpec]] = ministral_layer_spec
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


def get_llama_4_scaling_positions(
    num_local_tokens: int,
    *,
    packed_seq_params: PackedSeqParams | None,
    cp_group: torch.distributed.ProcessGroup | None,
    device: torch.device,
) -> torch.Tensor:
    """Return the position of each local query token within its own sequence.

    Core attention sees only this rank's tokens: packed (THD) input restarts positions
    at every packed sequence, and context parallelism keeps zigzag chunks ``r`` and
    ``2 * cp_size - 1 - r`` of each sequence. The Llama 4 attention scale must use the
    token's real position, as Hugging Face computes it from ``position_ids``.

    Args:
        num_local_tokens: Number of query tokens on this rank (``query.shape[0]``).
        packed_seq_params: Packed sequence metadata, or None for unpacked input.
        cp_group: Context-parallel group the input was sliced with, or None without CP.
        device: Device of the returned positions.

    Returns:
        Long tensor of shape ``[num_local_tokens]`` with the in-sequence positions.
    """
    cp_size = cp_group.size() if cp_group is not None else 1
    cp_rank = cp_group.rank() if cp_size > 1 else 0

    if packed_seq_params is not None and packed_seq_params.qkv_format == "thd":
        # Global indices of the local tokens in the full packed stream, sliced the same way as the inputs.
        stream_indices = get_packed_seq_cp_partition_indices(
            packed_seq_params,
            total_tokens=num_local_tokens * cp_size,
            cp_size=cp_size,
            cp_rank=cp_rank,
            device=device,
            cp_group=cp_group,
        )
        # Physical (padded) offsets, as used for RoPE in THD format.
        _, cu_seqlens = get_packed_seq_q_cu_seqlens(packed_seq_params)
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
        sequence_index = torch.searchsorted(cu_seqlens, stream_indices, right=True) - 1
        return stream_indices - cu_seqlens[sequence_index]

    positions = torch.arange(num_local_tokens, device=device)
    if cp_size > 1:
        # Zigzag CP split of one sequence: this rank holds chunks cp_rank and 2 * cp_size - 1 - cp_rank.
        chunk = num_local_tokens // 2
        positions = torch.where(
            positions < chunk,
            positions + cp_rank * chunk,
            positions - chunk + (2 * cp_size - 1 - cp_rank) * chunk,
        )
    return positions


class MinistralTEDotProductAttention(MCoreTEDotProductAttention):
    """
    Implementation of the TEDotProductAttention mechanism for Ministral (Mistral) 3 models with Llama 4-style attention scaling.

    This class extends MCoreTEDotProductAttention by introducing the Llama 4 attention scaling factor,
    which is essential for robust long-context training. During the forward pass, a position-dependent scaling
    (1 + beta * log(1 + floor(positions / max_position_embeddings))) is applied to the query vectors.
    This approach, introduced in Llama 4, helps maintain stability and performance as context length increases,
    enabling effective training and inference on extended sequences (e.g., up to 256k tokens).

    **Key difference from MCoreTEDotProductAttention:**
    - Applies the Llama 4 scaling factor to the queries prior to standard attention computation for improved
      long-context capability.
    """

    def __init__(
        self,
        config,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type=attention_type,
            attention_dropout=attention_dropout,
            **kwargs,
        )

        # Initialize Llama 4 attention scaling parameters
        if self.config.hf_config is not None:
            self.beta = self.config.hf_config.text_config.rope_parameters["llama_4_scaling_beta"]
            self.max_position_embeddings = self.config.hf_config.text_config.rope_parameters[
                "original_max_position_embeddings"
            ]
        else:
            self.beta = 0  # No effect
            self.max_position_embeddings = self.config.seq_length

        # CP group the model slices its inputs with, needed to recover the local tokens' positions.
        self.positions_cp_group = None
        if self.config.context_parallel_size > 1:
            pg_collection = kwargs.get("pg_collection")
            self.positions_cp_group = (
                pg_collection.cp
                if pg_collection is not None
                else parallel_state.get_context_parallel_group(check_initialized=False)
            )

    @staticmethod
    def _get_llama_4_attn_scale(
        positions_ids: torch.Tensor, beta: float, max_position_embeddings: int, query_shape: tuple
    ) -> torch.Tensor:
        scaling = 1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings))
        # Add dimensions to match query shape: [seq_len] -> [seq_len, 1, 1] for packed or [seq_len, 1, 1, 1] for unpacked
        # Query can be either [seq_len, num_heads, head_dim] (packed) or [seq_len, batch, num_heads, head_dim] (unpacked)
        num_dims_to_add = len(query_shape) - 1
        for _ in range(num_dims_to_add):
            scaling = scaling.unsqueeze(-1)
        return scaling

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType,
        **kwargs,
    ):
        positions_ids = get_llama_4_scaling_positions(
            query.shape[0],
            packed_seq_params=kwargs.get("packed_seq_params"),
            cp_group=self.positions_cp_group,
            device=query.device,
        )
        query *= self._get_llama_4_attn_scale(positions_ids, self.beta, self.max_position_embeddings, query.shape).to(
            query.dtype
        )

        return super().forward(query, key, value, attention_mask, attn_mask_type, **kwargs)
