# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
FLUX diffusion model implementation for DFM.

This module provides the FLUX model architecture, which is a state-of-the-art
text-to-image diffusion model using MMDiT-style transformer blocks.

Components:
    - Flux: Main FLUX model class
    - FluxProvider: Configuration and provider dataclass for FLUX models
    - MMDiTLayer: Multi-modal DiT layer for double blocks
    - FluxSingleTransformerBlock: Single transformer block for FLUX
    - JointSelfAttention: Joint self-attention for MMDiT layers
    - FluxSingleAttention: Self-attention for single blocks
    - EmbedND: N-dimensional rotary position embedding
    - MLPEmbedder: MLP embedding module
    - TimeStepEmbedder: Timestep embedding module
    - AdaLN: Adaptive Layer Normalization
    - AdaLNContinuous: Continuous Adaptive Layer Normalization
"""

from megatron.bridge.diffusion.models.common.normalization import RMSNorm
from megatron.bridge.diffusion.models.flux.flow_matching.flux_inference_pipeline import (
    ClipConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxInferencePipeline,
    T5Config,
)
from megatron.bridge.diffusion.models.flux.flux_attention import (
    FluxSingleAttention,
    JointSelfAttention,
    JointSelfAttentionSubmodules,
)
from megatron.bridge.diffusion.models.flux.flux_layer_spec import (
    AdaLN,
    AdaLNContinuous,
    FluxSingleTransformerBlock,
    MMDiTLayer,
    get_flux_double_transformer_engine_spec,
    get_flux_single_transformer_engine_spec,
)
from megatron.bridge.diffusion.models.flux.flux_model import Flux
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider
from megatron.bridge.diffusion.models.flux.layers import (
    EmbedND,
    MLPEmbedder,
    TimeStepEmbedder,
    Timesteps,
    rope,
)


__all__ = [
    # Main model
    "Flux",
    "FluxProvider",
    # Transformer layers
    "MMDiTLayer",
    "FluxSingleTransformerBlock",
    # Attention modules
    "JointSelfAttention",
    "JointSelfAttentionSubmodules",
    "FluxSingleAttention",
    # Normalization
    "AdaLN",
    "AdaLNContinuous",
    "RMSNorm",
    # Embeddings
    "EmbedND",
    "MLPEmbedder",
    "TimeStepEmbedder",
    "Timesteps",
    "rope",
    # Layer specs
    "get_flux_double_transformer_engine_spec",
    "get_flux_single_transformer_engine_spec",
    # Inference pipeline
    "FluxInferencePipeline",
    "FlowMatchEulerDiscreteScheduler",
    "T5Config",
    "ClipConfig",
    "AutoEncoderConfig",
]
