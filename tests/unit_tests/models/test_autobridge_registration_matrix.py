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

"""Contract tests for public AutoBridge model registration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REGISTRATION_CHECK_SCRIPT = Path(__file__).with_name("autobridge_registration_check.py")


EXPECTED_REGISTRATIONS = {
    "BailingMoeV2ForCausalLM": "megatron.bridge.models.bailing.bailing_moe2_bridge.BailingMoeV2Bridge",
    "DeciLMForCausalLM": "megatron.bridge.models.llama_nemotron.llama_nemotron_bridge.LlamaNemotronBridge",
    "DeepseekV2ForCausalLM": "megatron.bridge.models.deepseek.deepseek_v2_bridge.DeepSeekV2Bridge",
    "DeepseekV3ForCausalLM": "megatron.bridge.models.deepseek.deepseek_v3_bridge.DeepSeekV3Bridge",
    "DeepseekV4ForCausalLM": "megatron.bridge.models.deepseek.deepseek_v4_bridge.DeepSeekV4Bridge",
    "Ernie4_5_MoeForCausalLM": "megatron.bridge.models.ernie.ernie_45_bridge.Ernie45Bridge",
    "Ernie4_5_VLMoeForConditionalGeneration": ("megatron.bridge.models.ernie_vl.ernie45_vl_bridge.Ernie45VLBridge"),
    "Exaone4ForCausalLM": "megatron.bridge.models.exaone.exaone4.exaone4_bridge.Exaone4Bridge",
    "Exaone4_5_ForConditionalGeneration": "megatron.bridge.models.exaone.exaone45.exaone45_bridge.Exaone45Bridge",
    "ExaoneMoEForCausalLM": "megatron.bridge.models.exaone.exaone_moe.exaone_moe_bridge.ExaoneMoeBridge",
    "ExaoneMoeForCausalLM": "megatron.bridge.models.exaone.exaone_moe.exaone_moe_bridge.ExaoneMoeBridge",
    "FalconH1ForCausalLM": "megatron.bridge.models.falcon_h1.falconh1_bridge.FalconH1Bridge",
    "Gemma3ForCausalLM": "megatron.bridge.models.gemma.gemma3_bridge.Gemma3ModelBridge",
    "Gemma3ForConditionalGeneration": "megatron.bridge.models.gemma_vl.gemma3_vl_bridge.Gemma3VLBridge",
    "Gemma4ForCausalLM": "megatron.bridge.models.gemma.gemma4_bridge.Gemma4Bridge",
    "Gemma4ForConditionalGeneration": "megatron.bridge.models.gemma_vl.gemma4_vl_bridge.Gemma4VLBridge",
    "Glm4MoeForCausalLM": "megatron.bridge.models.glm.glm45_bridge.GLM45Bridge",
    "Glm4MoeLiteForCausalLM": "megatron.bridge.models.glm.glm47_flash_bridge.GLM47FlashBridge",
    "Glm4vMoeForConditionalGeneration": "megatron.bridge.models.glm_vl.glm_45v_bridge.GLM45VBridge",
    "GlmMoeDsaForCausalLM": "megatron.bridge.models.glm_moe_dsa.glm5_bridge.GLM5Bridge",
    "GptOssForCausalLM": "megatron.bridge.models.gpt_oss.gpt_oss_bridge.GPTOSSBridge",
    "HYV3ForCausalLM": "megatron.bridge.models.hy_v3.hy_v3_bridge.HYV3Bridge",
    "KimiK25ForConditionalGeneration": "megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge.KimiK25VLBridge",
    "KimiK2ForCausalLM": "megatron.bridge.models.kimi.kimi_bridge.KimiK2Bridge",
    "LlamaForCausalLM": "megatron.bridge.models.llama.llama_bridge.LlamaBridge",
    "MiMoForCausalLM": "megatron.bridge.models.mimo.mimo_bridge.MimoBridge",
    "MiMoV2FlashForCausalLM": ("megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge.MiMoV2FlashBridge"),
    "MiniMaxM2ForCausalLM": "megatron.bridge.models.minimax_m2.minimax_m2_bridge.MiniMaxM2Bridge",
    "MiniMaxM3SparseForConditionalGeneration": ("megatron.bridge.models.minimax_m3.minimax_m3_bridge.MiniMaxM3Bridge"),
    "Mistral3ForConditionalGeneration": ("megatron.bridge.models.ministral3.ministral3_bridge.Ministral3Bridge"),
    "NemotronHForCausalLM": "megatron.bridge.models.nemotronh.nemotron_h_bridge.NemotronHBridge",
    "NemotronH_Nano_Omni_Reasoning_V3": (
        "megatron.bridge.models.nemotron_omni.nemotron_omni_bridge.NemotronOmniBridge"
    ),
    "NemotronH_Nano_VL_V2": "megatron.bridge.models.nemotron_vl.nemotron_vl_bridge.NemotronVLBridge",
    "NemotronLabsDiffusionModel": (
        "megatron.bridge.diffusion.conversion.nemotron_labs_diffusion."
        "nemotron_labs_diffusion_bridge.NemotronLabsDiffusionBridge"
    ),
    "OlmoeForCausalLM": "megatron.bridge.models.olmoe.olmoe_bridge.OlMoEBridge",
    "Qwen2AudioForConditionalGeneration": ("megatron.bridge.models.qwen_audio.qwen2_audio_bridge.Qwen2AudioBridge"),
    "Qwen2ForCausalLM": "megatron.bridge.models.qwen.qwen2_bridge.Qwen2Bridge",
    "Qwen2_5OmniForConditionalGeneration": ("megatron.bridge.models.qwen_omni.qwen25_omni_bridge.Qwen25OmniBridge"),
    "Qwen2_5_VLForConditionalGeneration": ("megatron.bridge.models.qwen_vl.qwen25_vl_bridge.Qwen25VLBridge"),
    "Qwen3ASRForConditionalGeneration": "megatron.bridge.models.qwen3_asr.qwen3_asr_bridge.Qwen3ASRBridge",
    "Qwen3ForCausalLM": "megatron.bridge.models.qwen.qwen3_bridge.Qwen3Bridge",
    "Qwen3MoeForCausalLM": "megatron.bridge.models.qwen.qwen3_moe_bridge.Qwen3MoEBridge",
    "Qwen3NextForCausalLM": "megatron.bridge.models.qwen.qwen3_next_bridge.Qwen3NextBridge",
    "Qwen3OmniMoeForConditionalGeneration": ("megatron.bridge.models.qwen_omni.qwen3_omni_bridge.Qwen3OmniBridge"),
    "Qwen3VLForConditionalGeneration": "megatron.bridge.models.qwen_vl.qwen3_vl_bridge.Qwen3VLBridge",
    "Qwen3VLMoeForConditionalGeneration": "megatron.bridge.models.qwen_vl.qwen3_vl_bridge.Qwen3VLMoEBridge",
    "Qwen3_5ForCausalLM": "megatron.bridge.models.qwen.qwen35_bridge.Qwen35Bridge",
    "Qwen3_5ForConditionalGeneration": "megatron.bridge.models.qwen_vl.qwen35_vl_bridge.Qwen35VLBridge",
    "Qwen3_5MoeForCausalLM": "megatron.bridge.models.qwen.qwen35_bridge.Qwen35MoEBridge",
    "Qwen3_5MoeForConditionalGeneration": ("megatron.bridge.models.qwen_vl.qwen35_vl_bridge.Qwen35VLMoEBridge"),
    "SarvamMLAForCausalLM": "megatron.bridge.models.sarvam.sarvam_mla_bridge.SarvamMLABridge",
    "SarvamMoEForCausalLM": "megatron.bridge.models.sarvam.sarvam_moe_bridge.SarvamMoEBridge",
    "Step3p5ForCausalLM": "megatron.bridge.models.stepfun.step35_bridge.Step35Bridge",
    "Step3p7ForConditionalGeneration": "megatron.bridge.models.stepfun.step37_bridge.Step37Bridge",
}

STRING_REGISTRATIONS = {
    "BailingMoeV2ForCausalLM",
    "DeciLMForCausalLM",
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    "DeepseekV4ForCausalLM",
    "Ernie4_5_MoeForCausalLM",
    "Ernie4_5_VLMoeForConditionalGeneration",
    "Exaone4ForCausalLM",
    "ExaoneMoEForCausalLM",
    "FalconH1ForCausalLM",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Glm4MoeLiteForCausalLM",
    "HYV3ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "KimiK2ForCausalLM",
    "MiMoForCausalLM",
    "MiMoV2FlashForCausalLM",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
    "NemotronHForCausalLM",
    "NemotronH_Nano_Omni_Reasoning_V3",
    "NemotronH_Nano_VL_V2",
    "NemotronLabsDiffusionModel",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
    "SarvamMLAForCausalLM",
    "SarvamMoEForCausalLM",
    "Step3p5ForCausalLM",
    "Step3p7ForConditionalGeneration",
}

DEPRECATED_REGISTRATIONS = {
    "Gemma2ForCausalLM",
    "GemmaForCausalLM",
    "MistralForCausalLM",
    "NemotronForCausalLM",
}


def test_public_autobridge_import_registers_every_supported_model() -> None:
    """The public package import must install every expected bridge registration."""
    result = subprocess.run(
        [
            sys.executable,
            str(REGISTRATION_CHECK_SCRIPT),
            json.dumps(EXPECTED_REGISTRATIONS, sort_keys=True),
            json.dumps(sorted(STRING_REGISTRATIONS)),
            json.dumps(sorted(DEPRECATED_REGISTRATIONS)),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
