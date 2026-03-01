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

# Import model providers for easy access
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.models.gemma import (
    CodeGemmaModelProvider2B,
    CodeGemmaModelProvider7B,
    Gemma2ModelProvider,
    Gemma2ModelProvider2B,
    Gemma2ModelProvider9B,
    Gemma2ModelProvider27B,
    Gemma3ModelProvider,
    Gemma3ModelProvider1B,
    Gemma3ModelProvider4B,
    Gemma3ModelProvider12B,
    Gemma3ModelProvider27B,
    GemmaModelProvider,
    GemmaModelProvider2B,
    GemmaModelProvider7B,
)
from megatron.bridge.models.gemma_vl import (
    Gemma3VLBridge,
    Gemma3VLModel,
    Gemma3VLModelProvider,
)
from megatron.bridge.models.glm import (
    GLM45Bridge,
)
from megatron.bridge.models.glm_vl import (
    GLM45VBridge,
    GLM45VModelProvider,
)
from megatron.bridge.models.gpt_oss import (
    GPTOSSBridge,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.llama import (
    LlamaBridge,
)
from megatron.bridge.models.llama_nemotron import (
    LlamaNemotronBridge,
    LlamaNemotronHeterogeneousProvider,
)
from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
from megatron.bridge.models.mimo.mimo_bridge import MimoBridge
from megatron.bridge.models.minimax_m2 import (
    MiniMaxM2Bridge,
)
from megatron.bridge.models.ministral3 import (
    Ministral3Bridge,
    Ministral3Model,
    Ministral3ModelProvider,
    Ministral3ModelProvider3B,
    Ministral3ModelProvider8B,
    Ministral3ModelProvider14B,
)
from megatron.bridge.models.mistral import (
    MistralModelProvider,
    MistralSmall3ModelProvider24B,
)
from megatron.bridge.models.nemotron import (
    NemotronBridge,
)
from megatron.bridge.models.nemotron_vl import (
    NemotronNano12Bv2VLModelProvider,
    NemotronVLBridge,
    NemotronVLModel,
)
from megatron.bridge.models.nemotronh import (
    NemotronHBridge,
)
from megatron.bridge.models.olmoe import (
    OlMoEBridge,
    OlMoEModelProvider,
)
from megatron.bridge.models.qwen import (
    Qwen2ModelProvider,
    Qwen2ModelProvider1P5B,
    Qwen2ModelProvider7B,
    Qwen2ModelProvider72B,
    Qwen2ModelProvider500M,
    Qwen3ModelProvider,
    Qwen3ModelProvider1P7B,
    Qwen3ModelProvider4B,
    Qwen3ModelProvider8B,
    Qwen3ModelProvider14B,
    Qwen3ModelProvider32B,
    Qwen3ModelProvider600M,
    Qwen3MoEModelProvider,
    Qwen3MoEModelProvider30B_A3B,
    Qwen3MoEModelProvider235B_A22B,
    Qwen25ModelProvider1P5B,
    Qwen25ModelProvider3B,
    Qwen25ModelProvider7B,
    Qwen25ModelProvider14B,
    Qwen25ModelProvider32B,
    Qwen25ModelProvider72B,
    Qwen25ModelProvider500M,
)
from megatron.bridge.models.qwen_vl import (
    Qwen25VLBridge,
    Qwen25VLModel,
    Qwen25VLModelProvider,
)
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import (
    Qwen3VLBridge,
    Qwen3VLModel,
    Qwen3VLModelProvider,
    Qwen3VLMoEBridge,
    Qwen3VLMoEModelProvider,
)
from megatron.bridge.models.t5_provider import T5ModelProvider


__all__ = [
    "AutoBridge",
    "MegatronMappingRegistry",
    "MegatronModelBridge",
    "ColumnParallelMapping",
    "GatedMLPMapping",
    "MegatronParamMapping",
    "QKVMapping",
    "ReplicatedMapping",
    "RowParallelMapping",
    "AutoMapping",
    "Gemma3ModelProvider",
    "Gemma3ModelProvider1B",
    "Gemma3ModelProvider4B",
    "Gemma3ModelProvider12B",
    "Gemma3ModelProvider27B",
    "CodeGemmaModelProvider2B",
    "CodeGemmaModelProvider7B",
    "GemmaModelProvider",
    "GemmaModelProvider2B",
    "GemmaModelProvider7B",
    "Gemma2ModelProvider",
    "Gemma2ModelProvider2B",
    "Gemma2ModelProvider9B",
    "Gemma2ModelProvider27B",
    "GLM45Bridge",
    "GLM45VBridge",
    "GLM45VModelProvider",
    "GPTModelProvider",
    "GPTOSSBridge",
    "T5ModelProvider",
    "LlamaBridge",
    "LlamaNemotronHeterogeneousProvider",
    "LlamaNemotronBridge",
    "MistralModelProvider",
    "MistralSmall3ModelProvider24B",
    # Ministral 3 Models
    "Ministral3Bridge",
    "Ministral3Model",
    "Ministral3ModelProvider",
    "Ministral3ModelProvider3B",
    "Ministral3ModelProvider8B",
    "Ministral3ModelProvider14B",
    "MiniMaxM2Bridge",
    "OlMoEBridge",
    "OlMoEModelProvider",
    "Qwen2ModelProvider",
    "Qwen2ModelProvider500M",
    "Qwen2ModelProvider1P5B",
    "Qwen2ModelProvider7B",
    "Qwen2ModelProvider72B",
    "Qwen25ModelProvider500M",
    "Qwen25ModelProvider1P5B",
    "Qwen25ModelProvider3B",
    "Qwen25ModelProvider7B",
    "Qwen25ModelProvider14B",
    "Qwen25ModelProvider32B",
    "Qwen25ModelProvider72B",
    "Qwen3ModelProvider",
    "Qwen3ModelProvider600M",
    "Qwen3ModelProvider1P7B",
    "Qwen3ModelProvider4B",
    "Qwen3ModelProvider8B",
    "Qwen3ModelProvider14B",
    "Qwen3ModelProvider32B",
    "Qwen3MoEModelProvider",
    "Qwen3MoEModelProvider30B_A3B",
    "Qwen3MoEModelProvider235B_A22B",
    "NemotronHBridge",
    "MambaModelProvider",
    "MimoBridge",
    # Nemotron Models
    "NemotronBridge",
    # VL Models
    "Qwen25VLModel",
    "Qwen25VLBridge",
    "Qwen25VLModelProvider",
    "Qwen3VLModel",
    "Qwen3VLModelProvider",
    "Qwen3VLMoEModelProvider",
    "Qwen3VLBridge",
    "Qwen3VLMoEBridge",
    "Gemma3VLBridge",
    "Gemma3VLModel",
    "Gemma3VLModelProvider",
    "NemotronVLModel",
    "NemotronVLBridge",
    "NemotronNano12Bv2VLModelProvider",
]
