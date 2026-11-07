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
import importlib.util


_BAGEL_EXPORTS = []
if (
    importlib.util.find_spec("megatron.core.models.bagel") is not None
    and importlib.util.find_spec("megatron.core.models.bagel.bagel_mimo") is not None
):
    from megatron.bridge.models.bagel import BagelBridge, BagelConfig, BagelModelProvider

    _BAGEL_EXPORTS = ["BagelBridge", "BagelConfig", "BagelModelProvider"]

try:
    from megatron.bridge.models.bailing import (
        BailingMoeV2Bridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    BailingMoeV2Bridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.beam import (
        BeamBridge,
        BeamModelBuilder,
        BeamModelConfig,
        BeamModelProvider,
        BeamQKVGateMapping,
        BeamTransformerConfig,
    )
except (ImportError, ModuleNotFoundError, Exception):
    BeamBridge = BeamModelBuilder = BeamModelConfig = BeamModelProvider = None  # type: ignore[assignment,misc]
    BeamQKVGateMapping = BeamTransformerConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.bert import (
        BertModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    BertModelProvider = None  # type: ignore[assignment,misc]

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
    DeepSeekV3Bridge,
)

try:
    from megatron.bridge.models.diffusion_gemma import (
        DiffusionGemmaBridge,
        DiffusionGemmaModel,
        DiffusionGemmaModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    DiffusionGemmaBridge = DiffusionGemmaModel = DiffusionGemmaModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.ernie import (
        Ernie45Bridge,
        Ernie55Bridge,
        Ernie55Config,
        Ernie55ModelBuilder,
        Ernie55ModelConfig,
        Ernie55TransformerConfig,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Ernie45Bridge = Ernie55Bridge = Ernie55Config = None  # type: ignore[assignment,misc]
    Ernie55ModelBuilder = Ernie55ModelConfig = Ernie55TransformerConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.ernie_vl import (
        Ernie45VLBridge,
        Ernie45VLModel,
        Ernie45VLModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Ernie45VLBridge = Ernie45VLModel = Ernie45VLModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.exaone.exaone4 import (
        Exaone4Bridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Exaone4Bridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.exaone.exaone45 import (
        Exaone45Bridge,
        Exaone45Model,
        Exaone45ModelBuilder,
        Exaone45ModelConfig,
        Exaone45ModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Exaone45Bridge = Exaone45Model = Exaone45ModelBuilder = None  # type: ignore[assignment,misc]
    Exaone45ModelConfig = Exaone45ModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.exaone.exaone_moe import (
        ExaoneMoeBridge,
        ExaoneMoeModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    ExaoneMoeBridge = ExaoneMoeModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.falcon_h1 import (
        FalconH1Bridge,
        FalconH1ModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    FalconH1Bridge = FalconH1ModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.gemma import (
        Gemma3ModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Gemma3ModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.gemma_vl import (
        Gemma3VLBridge,
        Gemma3VLModel,
        Gemma3VLModelProvider,
        Gemma4DenseVLModelConfig,
        Gemma4VLBridge,
        Gemma4VLModel,
        Gemma4VLModelConfig,
        Gemma4VLModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Gemma3VLBridge = Gemma3VLModel = Gemma3VLModelProvider = None  # type: ignore[assignment,misc]
    Gemma4DenseVLModelConfig = Gemma4VLBridge = Gemma4VLModel = None  # type: ignore[assignment,misc]
    Gemma4VLModelConfig = Gemma4VLModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.glm import (
        GLM45Bridge,
        GLM47FlashBridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    GLM45Bridge = GLM47FlashBridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.glm5next import (
        GLM53FlashBridge,
        GLM53FlashModel,
        GLM53FlashModelBuilder,
        GLM53FlashModelConfig,
        GLM53FlashModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    GLM53FlashBridge = GLM53FlashModel = GLM53FlashModelBuilder = None  # type: ignore[assignment,misc]
    GLM53FlashModelConfig = GLM53FlashModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.glm_moe_dsa import (
        GLM5Bridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    GLM5Bridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.glm_vl import (
        GLM45VBridge,
        GLM45VModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    GLM45VBridge = GLM45VModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.gpt_oss import (
        GPTOSSBridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    GPTOSSBridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.gpt_provider import GPTModelProvider
except (ImportError, ModuleNotFoundError, Exception):
    GPTModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.hy_v3 import (
        HYV3Bridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    HYV3Bridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.hybrid import (
        HybridModelBuilder,
        HybridModelConfig,
        HybridModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    HybridModelBuilder = HybridModelConfig = HybridModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.kimi import (
        KimiK2Bridge,
        KimiK3Bridge,
        KimiK3ModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    KimiK2Bridge = KimiK3Bridge = KimiK3ModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.kimi_vl import (
        KimiK25VLBridge,
        KimiK25VLModel,
        KimiK25VLModelConfig,
        KimiK25VLModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    KimiK25VLBridge = KimiK25VLModel = KimiK25VLModelConfig = KimiK25VLModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.llama import (
        LlamaBridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    LlamaBridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.mamba.mamba_provider import MambaModelProvider
except (ImportError, ModuleNotFoundError, Exception):
    MambaModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.mimo.mimo_bridge import MimoBridge
except (ImportError, ModuleNotFoundError, Exception):
    MimoBridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.mimo_v2_flash import (
        MiMoV2FlashBridge,
        MiMoV2FlashModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    MiMoV2FlashBridge = MiMoV2FlashModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.minicpmo import MiniCPMOBridge, MiniCPMOModelConfig
except (ImportError, ModuleNotFoundError, Exception):
    MiniCPMOBridge = MiniCPMOModelConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.minimax_m2 import (
        MiniMaxM2Bridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    MiniMaxM2Bridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.minimax_m3 import (
        MiniMaxM3Bridge,
        MiniMaxM3ModelConfig,
        MiniMaxM3VLModel,
        MiniMaxM3VLModelConfig,
        MiniMaxM3VLModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    MiniMaxM3Bridge = MiniMaxM3ModelConfig = MiniMaxM3VLModel = None  # type: ignore[assignment,misc]
    MiniMaxM3VLModelConfig = MiniMaxM3VLModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.ministral3 import (
        Ministral3Bridge,
        Ministral3Model,
        Ministral3ModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Ministral3Bridge = Ministral3Model = Ministral3ModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.mistral import (
        MistralModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    MistralModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.muse_glimmer import (
        MuseGlimmerBridge,
        MuseGlimmerModel,
        MuseGlimmerModelBuilder,
        MuseGlimmerModelConfig,
    )
except (ImportError, ModuleNotFoundError, Exception):
    MuseGlimmerBridge = MuseGlimmerModel = MuseGlimmerModelBuilder = MuseGlimmerModelConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.muse_spark_open import (
        MuseSparkBridge,
        MuseSparkModel,
        MuseSparkModelBuilder,
        MuseSparkModelConfig,
    )
except (ImportError, ModuleNotFoundError, Exception):
    MuseSparkBridge = MuseSparkModel = MuseSparkModelBuilder = MuseSparkModelConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.nemotron_omni import (
        Nemotron35SuperVLBridge,
        NemotronOmniBridge,
        NemotronOmniModel,
        NemotronOmniModelBuilder,
        NemotronOmniModelConfig,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Nemotron35SuperVLBridge = NemotronOmniBridge = NemotronOmniModel = None  # type: ignore[assignment,misc]
    NemotronOmniModelBuilder = NemotronOmniModelConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.nemotron_vl import (
        NemotronVLBridge,
        NemotronVLModel,
    )
except (ImportError, ModuleNotFoundError, Exception):
    NemotronVLBridge = NemotronVLModel = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.nemotronh import (
        NemotronHBridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    NemotronHBridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.olmoe import (
        OlMoEBridge,
        OlMoEModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    OlMoEBridge = OlMoEModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.qwen3_asr import (
        Qwen3ASRBridge,
        Qwen3ASRModel,
        Qwen3ASRModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Qwen3ASRBridge = Qwen3ASRModel = Qwen3ASRModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.qwen_audio import (
        Qwen2AudioBridge,
        Qwen2AudioModel,
        Qwen2AudioModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Qwen2AudioBridge = Qwen2AudioModel = Qwen2AudioModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.qwen_omni import (
        Qwen3OmniBridge,
        Qwen3OmniModel,
        Qwen3OmniModelProvider,
        Qwen25OmniBridge,
        Qwen25OmniModel,
        Qwen25OmniModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Qwen3OmniBridge = Qwen3OmniModel = Qwen3OmniModelProvider = None  # type: ignore[assignment,misc]
    Qwen25OmniBridge = Qwen25OmniModel = Qwen25OmniModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.qwen_vl import (
        Qwen25VLBridge,
        Qwen25VLModel,
        Qwen25VLModelProvider,
        Qwen35TokenClassificationBridge,
        Qwen35TokenClassificationModelProvider,
        Qwen35VLBridge,
        Qwen35VLModelBuilder,
        Qwen35VLModelConfig,
        Qwen35VLModelProvider,
        Qwen35VLMoEBridge,
        Qwen35VLMoEModelProvider,
        Qwen35VLTransformerConfig,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Qwen25VLBridge = Qwen25VLModel = Qwen25VLModelProvider = None  # type: ignore[assignment,misc]
    Qwen35TokenClassificationBridge = Qwen35TokenClassificationModelProvider = None  # type: ignore[assignment,misc]
    Qwen35VLBridge = Qwen35VLModelBuilder = Qwen35VLModelConfig = None  # type: ignore[assignment,misc]
    Qwen35VLModelProvider = Qwen35VLMoEBridge = Qwen35VLMoEModelProvider = None  # type: ignore[assignment,misc]
    Qwen35VLTransformerConfig = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl import (
        Qwen3VLBridge,
        Qwen3VLForTokenClassification,
        Qwen3VLModel,
        Qwen3VLModelProvider,
        Qwen3VLMoEBridge,
        Qwen3VLMoEModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Qwen3VLBridge = Qwen3VLForTokenClassification = Qwen3VLModel = None  # type: ignore[assignment,misc]
    Qwen3VLModelProvider = Qwen3VLMoEBridge = Qwen3VLMoEModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.sarvam import (
        SarvamMLABridge,
        SarvamMoEBridge,
    )
except (ImportError, ModuleNotFoundError, Exception):
    SarvamMLABridge = SarvamMoEBridge = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.stepfun import (
        Step35Bridge,
        Step37Bridge,
        Step37Model,
        Step37ModelBuilder,
        Step37ModelConfig,
        Step37ModelProvider,
    )
except (ImportError, ModuleNotFoundError, Exception):
    Step35Bridge = Step37Bridge = Step37Model = None  # type: ignore[assignment,misc]
    Step37ModelBuilder = Step37ModelConfig = Step37ModelProvider = None  # type: ignore[assignment,misc]

try:
    from megatron.bridge.models.t5_provider import T5ModelProvider
except (ImportError, ModuleNotFoundError, Exception):
    T5ModelProvider = None  # type: ignore[assignment,misc]


__all__ = [
    "AutoBridge",
    "MiniCPMOBridge",
    "MiniCPMOModelConfig",
    "MegatronMappingRegistry",
    "MegatronModelBridge",
    "ColumnParallelMapping",
    "GatedMLPMapping",
    "MegatronParamMapping",
    "QKVMapping",
    "ReplicatedMapping",
    "RowParallelMapping",
    "AutoMapping",
    "BailingMoeV2Bridge",
    "BertModelProvider",
    # DeepSeek Models
    "DeepSeekV3Bridge",
    # ERNIE Text-Only Models
    "Ernie45Bridge",
    "Ernie55Bridge",
    "Ernie55Config",
    "Ernie55ModelBuilder",
    "Ernie55ModelConfig",
    "Ernie55TransformerConfig",
    # ERNIE VL Models
    "Ernie45VLBridge",
    "Ernie45VLModel",
    "Ernie45VLModelProvider",
    "Exaone4Bridge",
    "FalconH1Bridge",
    "FalconH1ModelProvider",
    "Gemma3ModelProvider",
    "GLM45Bridge",
    "GLM47FlashBridge",
    "GLM5Bridge",
    "GLM53FlashBridge",
    "GLM53FlashModel",
    "GLM53FlashModelBuilder",
    "GLM53FlashModelConfig",
    "GLM53FlashModelProvider",
    "GLM45VBridge",
    "GLM45VModelProvider",
    "GPTModelProvider",
    "GPTOSSBridge",
    "T5ModelProvider",
    "HYV3Bridge",
    "HybridModelBuilder",
    "HybridModelConfig",
    "HybridModelProvider",
    "KimiK2Bridge",
    "KimiK3Bridge",
    "KimiK3ModelProvider",
    "KimiK25VLModel",
    "KimiK25VLBridge",
    "KimiK25VLModelConfig",
    "KimiK25VLModelProvider",
    "LlamaBridge",
    "MistralModelProvider",
    # Ministral 3 Models
    "Ministral3Bridge",
    "Ministral3Model",
    "Ministral3ModelProvider",
    "MiniMaxM2Bridge",
    "MiniMaxM3Bridge",
    "MiniMaxM3ModelConfig",
    "MiniMaxM3VLModel",
    "MiniMaxM3VLModelConfig",
    "MiniMaxM3VLModelProvider",
    # Muse Glimmer
    "MuseGlimmerBridge",
    "MuseGlimmerModel",
    "MuseGlimmerModelBuilder",
    "MuseGlimmerModelConfig",
    "MuseSparkBridge",
    "MuseSparkModel",
    "MuseSparkModelBuilder",
    "MuseSparkModelConfig",
    "OlMoEBridge",
    "OlMoEModelProvider",
    "NemotronHBridge",
    "MambaModelProvider",
    "MimoBridge",
    # MiMo-V2-Flash
    "MiMoV2FlashBridge",
    "MiMoV2FlashModelProvider",
    # Nemotron Models
    # Audio-Language Models
    "Qwen2AudioBridge",
    "Qwen2AudioModel",
    "Qwen2AudioModelProvider",
    # Beam (Reflection AI) models
    "BeamBridge",
    "BeamModelBuilder",
    "BeamModelProvider",
    "BeamModelConfig",
    "BeamQKVGateMapping",
    "BeamTransformerConfig",
    # VL Models
    "Qwen25VLModel",
    "Qwen25VLBridge",
    "Qwen25VLModelProvider",
    "Qwen3VLModel",
    "Qwen3VLForTokenClassification",
    "Qwen3VLModelProvider",
    "Qwen3VLMoEModelProvider",
    "Qwen3VLBridge",
    "Qwen3VLMoEBridge",
    "Qwen35VLBridge",
    "Qwen35TokenClassificationBridge",
    "Qwen35TokenClassificationModelProvider",
    "Qwen35VLModelBuilder",
    "Qwen35VLModelConfig",
    "Qwen35VLModelProvider",
    "Qwen35VLMoEBridge",
    "Qwen35VLMoEModelProvider",
    "Qwen35VLTransformerConfig",
    "Gemma3VLBridge",
    "Gemma3VLModel",
    "Gemma3VLModelProvider",
    "Gemma4VLBridge",
    "Gemma4VLModel",
    "Gemma4VLModelConfig",
    "Gemma4VLModelProvider",
    "Gemma4DenseVLModelConfig",
    "DiffusionGemmaBridge",
    "DiffusionGemmaModel",
    "DiffusionGemmaModelProvider",
    "NemotronVLModel",
    "NemotronVLBridge",
    "NemotronOmniBridge",
    "NemotronOmniModel",
    "NemotronOmniModelBuilder",
    "NemotronOmniModelConfig",
    "Nemotron35SuperVLBridge",
    # ASR Models
    "Qwen3ASRBridge",
    "Qwen3ASRModel",
    "Qwen3ASRModelProvider",
    # Qwen Omni registrations
    "Qwen25OmniModel",
    "Qwen25OmniBridge",
    "Qwen25OmniModelProvider",
    "Qwen3OmniModel",
    "Qwen3OmniBridge",
    "Qwen3OmniModelProvider",
    "SarvamMLABridge",
    "SarvamMoEBridge",
    "Step35Bridge",
    "Step37Bridge",
    "Step37Model",
    "Step37ModelBuilder",
    "Step37ModelConfig",
    "Step37ModelProvider",
    # EXAONE
    "Exaone4Bridge",
    "Exaone45Bridge",
    "Exaone45Model",
    "Exaone45ModelBuilder",
    "Exaone45ModelConfig",
    "Exaone45ModelProvider",
    "ExaoneMoeBridge",
    "ExaoneMoeModelProvider",
] + _BAGEL_EXPORTS
