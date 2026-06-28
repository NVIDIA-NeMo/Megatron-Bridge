# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Completeness audit for registered bridge builder contracts."""

import ast
import importlib
from pathlib import Path

import pytest
from megatron.training.models.base import ModelBuilder, ModelConfig


pytestmark = pytest.mark.unit

SRC_ROOT = Path(__file__).parents[3] / "src"
BRIDGE_ROOT = SRC_ROOT / "megatron" / "bridge" / "models"

# Every registered bridge must be named here. Adding a registration without
# adding its builder/config contract to the audit is intentionally a test failure.
REGISTERED_BRIDGE_CONFIGS: dict[str, tuple[str, ...]] = {
    "BailingMoeV2Bridge": ("megatron.bridge.models.bailing.model_config.BailingMoeV2ModelConfig",),
    "DeepSeekV2Bridge": ("megatron.bridge.models.deepseek.model_config.DeepSeekV2ModelConfig",),
    "DeepSeekV3Bridge": ("megatron.bridge.models.deepseek.model_config.DeepSeekV3ModelConfig",),
    "DeepSeekV4Bridge": ("megatron.bridge.models.deepseek.deepseek_v4_model_config.DeepSeekV4ModelConfig",),
    "Ernie45Bridge": ("megatron.bridge.models.ernie.model_config.Ernie45ModelConfig",),
    "Ernie45VLBridge": ("megatron.bridge.models.ernie_vl.model_config.Ernie45VLModelConfig",),
    "Exaone4Bridge": ("megatron.bridge.models.exaone.model_config.Exaone4ModelConfig",),
    "FalconH1Bridge": ("megatron.bridge.models.falcon_h1.model_config.FalconH1ModelConfig",),
    "GLM45Bridge": ("megatron.bridge.models.glm.model_config.GLM45ModelConfig",),
    "GLM45VBridge": ("megatron.bridge.models.glm_vl.model_config.GLM45VModelConfig",),
    "GLM47FlashBridge": ("megatron.bridge.models.glm.model_config.GLM47FlashModelConfig",),
    "GLM5Bridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "GPTOSSBridge": ("megatron.bridge.models.gpt_oss.model_config.GPTOSSModelConfig",),
    "Gemma2Bridge": ("megatron.bridge.models.gemma.model_config.Gemma2ModelConfig",),
    "Gemma3ModelBridge": ("megatron.bridge.models.gemma.model_config.Gemma3ModelConfig",),
    "Gemma3VLBridge": ("megatron.bridge.models.gemma_vl.model_config.Gemma3VLModelConfig",),
    "Gemma4Bridge": (
        "megatron.bridge.models.gemma.model_config.Gemma4ModelConfig",
        "megatron.bridge.models.gemma.model_config.Gemma4DenseModelConfig",
    ),
    "Gemma4VLBridge": (
        "megatron.bridge.models.gemma_vl.model_config.Gemma4VLModelConfig",
        "megatron.bridge.models.gemma_vl.model_config.Gemma4DenseVLModelConfig",
    ),
    "GemmaBridge": ("megatron.bridge.models.gemma.model_config.GemmaModelConfig",),
    "KimiK2Bridge": ("megatron.bridge.models.kimi.model_config.KimiK2ModelConfig",),
    "KimiK25VLBridge": ("megatron.bridge.models.kimi_vl.model_config.KimiK25VLModelConfig",),
    "LlamaBridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "LlamaNemotronBridge": ("megatron.bridge.models.llama_nemotron.model_config.LlamaNemotronModelConfig",),
    "MiMoV2FlashBridge": ("megatron.bridge.models.mimo_v2_flash.model_config.MiMoV2FlashModelConfig",),
    "MimoBridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "MiniMaxM2Bridge": ("megatron.bridge.models.minimax_m2.model_config.MiniMaxM2ModelConfig",),
    "Ministral3Bridge": ("megatron.bridge.models.ministral3.model_config.Ministral3ModelConfig",),
    "MistralBridge": ("megatron.bridge.models.mistral.model_config.MistralModelConfig",),
    "NemotronBridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "NemotronHBridge": ("megatron.bridge.models.nemotronh.model_config.NemotronHModelConfig",),
    "NemotronOmniBridge": ("megatron.bridge.models.nemotron_omni.model_config.NemotronOmniModelConfig",),
    "NemotronVLBridge": ("megatron.bridge.models.nemotron_vl.model_config.NemotronVLModelConfig",),
    "OlMoEBridge": ("megatron.bridge.models.olmoe.model_config.OlMoEModelConfig",),
    "Qwen2AudioBridge": ("megatron.bridge.models.qwen_audio.model_config.Qwen2AudioModelConfig",),
    "Qwen2Bridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "Qwen25OmniBridge": ("megatron.bridge.models.qwen_omni.model_config.Qwen25OmniModelConfig",),
    "Qwen25VLBridge": ("megatron.bridge.models.qwen_vl.model_config.Qwen25VLModelConfig",),
    "Qwen35Bridge": ("megatron.bridge.models.qwen.model_config.QwenHybridModelConfig",),
    "Qwen35MoEBridge": ("megatron.bridge.models.qwen.model_config.QwenHybridModelConfig",),
    "Qwen35VLBridge": ("megatron.bridge.models.qwen_vl.model_config.Qwen35VLModelConfig",),
    "Qwen35VLMoEBridge": ("megatron.bridge.models.qwen_vl.model_config.Qwen35VLModelConfig",),
    "Qwen3ASRBridge": ("megatron.bridge.models.qwen3_asr.model_config.Qwen3ASRModelConfig",),
    "Qwen3Bridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "Qwen3MoEBridge": ("megatron.bridge.models.gpt.model_config.BridgeGPTModelConfig",),
    "Qwen3NextBridge": ("megatron.bridge.models.qwen.model_config.QwenHybridModelConfig",),
    "Qwen3OmniBridge": ("megatron.bridge.models.qwen_omni.model_config.Qwen3OmniModelConfig",),
    "Qwen3VLBridge": ("megatron.bridge.models.qwen_vl.model_config.Qwen3VLModelConfig",),
    "Qwen3VLMoEBridge": ("megatron.bridge.models.qwen_vl.model_config.Qwen3VLModelConfig",),
    "SarvamMLABridge": ("megatron.bridge.models.sarvam.model_config.SarvamMLAModelConfig",),
    "SarvamMoEBridge": ("megatron.bridge.models.sarvam.model_config.SarvamMoEModelConfig",),
    "Step35Bridge": ("megatron.bridge.models.stepfun.step35_bridge.Step35ModelConfig",),
    "Step37Bridge": ("megatron.bridge.models.stepfun.step37_model_config.Step37ModelConfig",),
}

# Keep the registration source expression explicit as part of the manifest. This
# catches a bridge silently being moved to a different HF architecture even when
# its Python class name and builder contract stay unchanged.
REGISTERED_BRIDGE_ARCHITECTURES: dict[str, str] = {
    "BailingMoeV2Bridge": "'BailingMoeV2ForCausalLM'",
    "DeepSeekV2Bridge": "'DeepseekV2ForCausalLM'",
    "DeepSeekV3Bridge": "'DeepseekV3ForCausalLM'",
    "DeepSeekV4Bridge": "'DeepseekV4ForCausalLM'",
    "Ernie45Bridge": "_ERNIE45_MOE_HF_CLASS_NAME",
    "Ernie45VLBridge": "_ERNIE45_VL_MOE_HF_CLASS_NAME",
    "Exaone4Bridge": "'Exaone4ForCausalLM'",
    "FalconH1Bridge": "'FalconH1ForCausalLM'",
    "GLM45Bridge": "Glm4MoeForCausalLM",
    "GLM45VBridge": "Glm4vMoeForConditionalGeneration",
    "GLM47FlashBridge": "'Glm4MoeLiteForCausalLM'",
    "GLM5Bridge": "GlmMoeDsaForCausalLM",
    "GPTOSSBridge": "GptOssForCausalLM",
    "Gemma2Bridge": "Gemma2ForCausalLM",
    "Gemma3ModelBridge": "Gemma3ForCausalLM",
    "Gemma3VLBridge": "Gemma3ForConditionalGeneration",
    "Gemma4Bridge": "'Gemma4ForCausalLM'",
    "Gemma4VLBridge": "'Gemma4ForConditionalGeneration'",
    "GemmaBridge": "GemmaForCausalLM",
    "KimiK25VLBridge": "'KimiK25ForConditionalGeneration'",
    "KimiK2Bridge": "'KimiK2ForCausalLM'",
    "LlamaBridge": "LlamaForCausalLM",
    "LlamaNemotronBridge": "'DeciLMForCausalLM'",
    "MiMoV2FlashBridge": "'MiMoV2FlashForCausalLM'",
    "MimoBridge": "'MiMoForCausalLM'",
    "MiniMaxM2Bridge": "'MiniMaxM2ForCausalLM'",
    "Ministral3Bridge": "Mistral3ForConditionalGeneration",
    "MistralBridge": "MistralForCausalLM",
    "NemotronBridge": "NemotronForCausalLM",
    "NemotronHBridge": "'NemotronHForCausalLM'",
    "NemotronOmniBridge": "'NemotronH_Nano_Omni_Reasoning_V3'",
    "NemotronVLBridge": "'NemotronH_Nano_VL_V2'",
    "OlMoEBridge": "OlmoeForCausalLM",
    "Qwen25OmniBridge": "Qwen2_5OmniForConditionalGeneration",
    "Qwen25VLBridge": "Qwen2_5_VLForConditionalGeneration",
    "Qwen2AudioBridge": "Qwen2AudioForConditionalGeneration",
    "Qwen2Bridge": "Qwen2ForCausalLM",
    "Qwen35Bridge": "Qwen3_5ForCausalLM",
    "Qwen35MoEBridge": "Qwen3_5MoeForCausalLM",
    "Qwen35VLBridge": "_QWEN3_5_DENSE_HF_CLASS_NAME",
    "Qwen35VLMoEBridge": "_QWEN3_5_MOE_HF_CLASS_NAME",
    "Qwen3ASRBridge": "'Qwen3ASRForConditionalGeneration'",
    "Qwen3Bridge": "Qwen3ForCausalLM",
    "Qwen3MoEBridge": "Qwen3MoeForCausalLM",
    "Qwen3NextBridge": "Qwen3NextForCausalLM",
    "Qwen3OmniBridge": "Qwen3OmniMoeForConditionalGeneration",
    "Qwen3VLBridge": "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoEBridge": "Qwen3VLMoeForConditionalGeneration",
    "SarvamMLABridge": "'SarvamMLAForCausalLM'",
    "SarvamMoEBridge": "'SarvamMoEForCausalLM'",
    "Step35Bridge": "'Step3p5ForCausalLM'",
    "Step37Bridge": "'Step3p7ForConditionalGeneration'",
}

REGISTRATION_SOURCE_ALIASES: dict[str, str] = {
    "_ERNIE45_MOE_HF_CLASS_NAME": "Ernie4_5_MoeForCausalLM",
    "_ERNIE45_VL_MOE_HF_CLASS_NAME": "Ernie4_5_VLMoeForConditionalGeneration",
    "_QWEN3_5_DENSE_HF_CLASS_NAME": "Qwen3_5ForConditionalGeneration",
    "_QWEN3_5_MOE_HF_CLASS_NAME": "Qwen3_5MoeForConditionalGeneration",
}


def _resolve(path: str) -> type:
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def _source_argument(call: ast.Call) -> str:
    source = next(keyword.value for keyword in call.keywords if keyword.arg == "source")
    return ast.unparse(source)


def _registered_bridge_architectures() -> dict[str, str]:
    architectures: dict[str, str] = {}
    for path in BRIDGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if any(
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "register_bridge"
                    for decorator in node.decorator_list
                ):
                    decorator = next(
                        decorator
                        for decorator in node.decorator_list
                        if isinstance(decorator, ast.Call)
                        and isinstance(decorator.func, ast.Attribute)
                        and decorator.func.attr == "register_bridge"
                    )
                    architectures[node.name] = _source_argument(decorator)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id == "register_bridge_implementation":
                    bridge_class = next(
                        (keyword.value for keyword in call.keywords if keyword.arg == "bridge_class"), None
                    )
                    if isinstance(bridge_class, ast.Name):
                        architectures[bridge_class.id] = _source_argument(call)
            if isinstance(node, ast.If):
                for child in ast.walk(node):
                    if not isinstance(child, ast.Assign) or not isinstance(child.value, ast.Call):
                        continue
                    registered = child.value.func
                    if not isinstance(registered, ast.Call) or not isinstance(registered.func, ast.Attribute):
                        continue
                    if registered.func.attr != "register_bridge":
                        continue
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            architectures[target.id] = _source_argument(registered)
    return architectures


def _registration_source_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for path in BRIDGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in REGISTRATION_SOURCE_ALIASES:
                    aliases[target.id] = value.value
    return aliases


def test_every_registered_bridge_has_a_builder_contract_manifest_entry() -> None:
    assert _registered_bridge_architectures() == REGISTERED_BRIDGE_ARCHITECTURES
    assert _registration_source_aliases() == REGISTRATION_SOURCE_ALIASES
    assert REGISTERED_BRIDGE_CONFIGS.keys() == REGISTERED_BRIDGE_ARCHITECTURES.keys()


@pytest.mark.parametrize(
    "config_path",
    sorted({path for paths in REGISTERED_BRIDGE_CONFIGS.values() for path in paths}),
)
def test_registered_model_config_resolves_a_stable_builder(config_path: str) -> None:
    config_class = _resolve(config_path)
    builder_class = _resolve(config_class.builder)

    assert issubclass(config_class, ModelConfig)
    assert issubclass(builder_class, ModelBuilder)
    assert _resolve(config_class.builder) is builder_class
    assert "provider" not in config_class.builder.lower()


def test_registered_bridges_do_not_declare_legacy_model_build_routes() -> None:
    violations: list[str] = []
    for path in BRIDGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == "LEGACY_MODEL_BUILD_ONLY" for target in targets):
                violations.append(f"{path}:{node.lineno}")

    assert not violations, "Legacy provider build routes remain: " + ", ".join(violations)
