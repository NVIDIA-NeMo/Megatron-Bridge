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
Qwen3-ASR Model Provider configurations for Megatron-Core.

This module provides configuration classes for Qwen3-ASR audio speech recognition models
(audio+text), compatible with HuggingFace's Qwen3-ASR model configurations.
"""

from dataclasses import dataclass, field
from typing import Callable

import torch.nn.functional as F
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.qwen.qwen_hybrid import QwenHybridModelProvider, configure_qwen_hybrid_layers
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRThinkerConfig,
)
from megatron.bridge.models.qwen3_asr.modeling_qwen3_asr.model import Qwen3ASRModel
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.text_model import (
    Qwen3VLHybridModel,
    get_qwen3_vl_hybrid_stack_spec,
)
from megatron.bridge.models.qwen_vl.qwen3_vl_provider import _provide_qwen3_vl_language_model


@dataclass
class Qwen3ASRModelProvider(QwenHybridModelProvider):
    """
    Base model provider for Qwen3-ASR Models.
    Inherits language model configuration from HybridModelProvider with Qwen3-specific defaults.

    Key characteristics:
    - Audio-only (no vision, no video)
    - Qwen3-based LLM: qk_layernorm=True, no QKV bias, SwiGLU activation
    - mrope_section: [24, 20, 20]
    - rotary_base: 5000000.0
    - Simple RoPE: same position IDs across all 3 MRoPE dims
    """

    thinker_config: Qwen3ASRThinkerConfig = field(default_factory=Qwen3ASRThinkerConfig)

    # Token IDs matching Qwen3-ASR configuration
    audio_token_id: int = 151646
    audio_start_token_id: int = 151647

    # Qwen3 architecture defaults (previously inherited from Qwen3ModelProvider)
    activation_func: Callable = F.silu
    gated_linear_unit: bool = True
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    qk_layernorm: bool = True
    hidden_dropout: float = 0.0
    attention_softmax_in_fp32: bool = True
    attention_dropout: float = 0.0

    position_embedding_type: str = "mrope"
    apply_rotary_pos_emb_in_fp32: bool = False
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])
    rotary_base: float = 5000000.0

    scatter_embedding_sequence_parallel: bool = False

    # Freeze options
    freeze_language_model: bool = False
    freeze_audio_model: bool = False
    language_max_sequence_length: int = 2048

    normalization: str = "RMSNorm"
    persist_layer_norm: bool = True
    bias_activation_fusion: bool = True
    bias_dropout_fusion: bool = True
    masked_softmax_fusion: bool = False
    deallocate_pipeline_outputs: bool = True
    async_tensor_model_parallel_allreduce: bool = True
    distribute_saved_activations: bool = False
    cp_comm_type: str = "p2p"
    gradient_accumulation_fusion: bool = False
    mtp_num_layers: int | None = None
    hybrid_stack_spec: ModuleSpec | Callable = get_qwen3_vl_hybrid_stack_spec

    def finalize(self) -> None:
        if self.hybrid_layer_pattern is None:
            if self.num_layers is None:
                raise ValueError("num_layers must be configured for Qwen3-ASR")
            configure_qwen_hybrid_layers(
                self,
                num_logical_layers=self.num_layers,
                mlp_symbols=Symbols.MLP,
                mtp_mlp_symbol=Symbols.MLP,
            )
        super().finalize()

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Provide a Qwen3-ASR model instance with audio and language components."""
        if self.hybrid_layer_pattern is None:
            if self.num_layers is None:
                raise ValueError("num_layers must be configured for Qwen3-ASR")
            configure_qwen_hybrid_layers(
                self,
                num_logical_layers=self.num_layers,
                mlp_symbols=Symbols.MLP,
                mtp_mlp_symbol=Symbols.MLP,
            )
        language_transformer_config = self
        thinker_config = self.thinker_config

        language_transformer_layer_spec = self._resolve_hybrid_stack_spec()

        model = Qwen3ASRModel(
            language_transformer_config=language_transformer_config,
            language_transformer_layer_spec=language_transformer_layer_spec,
            thinker_transformer_config=thinker_config,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=self._pg_collection,
        )

        if self.freeze_language_model or self.freeze_audio_model:
            model.freeze(
                freeze_language_model=self.freeze_language_model,
                freeze_audio_model=self.freeze_audio_model,
            )

        return model

    def provide_language_model(self, pre_process=None, post_process=None, vp_stage=None) -> Qwen3VLHybridModel:
        """Provide just the language model component without audio."""
        return _provide_qwen3_vl_language_model(self, pre_process, post_process, vp_stage)
