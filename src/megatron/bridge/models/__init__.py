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
from megatron.bridge.models.deepseek import (
    DeepSeekV2LiteProvider,
    DeepSeekV2Provider,
    DeepSeekV3Provider,
    MoonlightProvider,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.llama import (
    CodeLlamaModelProvider7B,
    CodeLlamaModelProvider13B,
    CodeLlamaModelProvider34B,
    CodeLlamaModelProvider70B,
    Llama2ModelProvider7B,
    Llama2ModelProvider13B,
    Llama2ModelProvider70B,
    Llama3ModelProvider,
    Llama3ModelProvider8B,
    Llama3ModelProvider70B,
    Llama4Experts16ModelProvider,
    Llama4Experts128ModelProvider,
    Llama4ModelProvider,
    Llama31ModelProvider,
    Llama31ModelProvider8B,
    Llama31ModelProvider70B,
    Llama31ModelProvider405B,
    Llama32ModelProvider1B,
    Llama32ModelProvider3B,
    LlamaModelProvider,
)
from megatron.bridge.models.mamba.mamba_provider import (
    MambaProvider,
    MambaProvider1_3B,
    MambaProvider2_7B,
    MambaProvider130M,
    MambaProvider370M,
    MambaProvider780M,
    NVIDIAMambaHybridProvider8B,
    NVIDIAMambaProvider8B,
)
from megatron.bridge.models.mamba.nemotron_h_provider import (
    NemotronHModel4BProvider,
    NemotronHModel8BProvider,
    NemotronHModel47BProvider,
    NemotronHModel56BProvider,
    NemotronHModelProvider,
    NemotronNano9Bv2Provider,
    NemotronNano12Bv2Provider,
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
    "GPTModelProvider",
    "T5ModelProvider",
    "LlamaModelProvider",
    "Llama2ModelProvider7B",
    "Llama2ModelProvider13B",
    "Llama2ModelProvider70B",
    "Llama3ModelProvider",
    "Llama3ModelProvider8B",
    "Llama3ModelProvider70B",
    "Llama31ModelProvider",
    "Llama31ModelProvider8B",
    "Llama31ModelProvider70B",
    "Llama31ModelProvider405B",
    "Llama32ModelProvider1B",
    "Llama32ModelProvider3B",
    "CodeLlamaModelProvider7B",
    "CodeLlamaModelProvider13B",
    "CodeLlamaModelProvider34B",
    "CodeLlamaModelProvider70B",
    "Llama4ModelProvider",
    "Llama4Experts16ModelProvider",
    "Llama4Experts128ModelProvider",
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
    "DeepSeekV2LiteProvider",
    "DeepSeekV2Provider",
    "DeepSeekV3Provider",
    "MoonlightProvider",
    "NemotronHModelProvider",
    "NemotronHModel4BProvider",
    "NemotronHModel8BProvider",
    "NemotronHModel47BProvider",
    "NemotronHModel56BProvider",
    "NemotronNano9Bv2Provider",
    "NemotronNano12Bv2Provider",
    "MambaProvider",
    "MambaProvider1_3B",
    "MambaProvider2_7B",
    "MambaProvider130M",
    "MambaProvider370M",
    "MambaProvider780M",
    "NVIDIAMambaHybridProvider8B",
    "NVIDIAMambaProvider8B",
    # VL Models
    "Qwen25VLModel",
    "Qwen25VLBridge",
    "Qwen25VLModelProvider",
]
