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

"""Contract tests for AutoBridge model registration."""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import pytest
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge


pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class RegistrationCase:
    """Expected AutoBridge registration for one model family."""

    id: str
    module: str
    architecture: str
    bridge_class: str
    needs_auto_map: bool = False


REGISTRATION_CASES = (
    RegistrationCase(
        id="llama",
        module="megatron.bridge.models.llama.llama_bridge",
        architecture="LlamaForCausalLM",
        bridge_class="LlamaBridge",
    ),
    RegistrationCase(
        id="qwen3_vl",
        module="megatron.bridge.models.qwen_vl.qwen3_vl_bridge",
        architecture="Qwen3VLForConditionalGeneration",
        bridge_class="Qwen3VLBridge",
    ),
    RegistrationCase(
        id="qwen3_omni",
        module="megatron.bridge.models.qwen_omni.qwen3_omni_bridge",
        architecture="Qwen3OmniMoeForConditionalGeneration",
        bridge_class="Qwen3OmniBridge",
    ),
    RegistrationCase(
        id="exaone4",
        module="megatron.bridge.models.exaone.exaone4_bridge",
        architecture="Exaone4ForCausalLM",
        bridge_class="Exaone4Bridge",
        needs_auto_map=True,
    ),
    RegistrationCase(
        id="glm45",
        module="megatron.bridge.models.glm.glm45_bridge",
        architecture="Glm4MoeForCausalLM",
        bridge_class="GLM45Bridge",
    ),
    RegistrationCase(
        id="sarvam_moe",
        module="megatron.bridge.models.sarvam.sarvam_moe_bridge",
        architecture="SarvamMoEForCausalLM",
        bridge_class="SarvamMoEBridge",
        needs_auto_map=True,
    ),
    RegistrationCase(
        id="mimo_v2_flash",
        module="megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge",
        architecture="MiMoV2FlashForCausalLM",
        bridge_class="MiMoV2FlashBridge",
        needs_auto_map=True,
    ),
    RegistrationCase(
        id="step35",
        module="megatron.bridge.models.stepfun.step35_bridge",
        architecture="Step3p5ForCausalLM",
        bridge_class="Step35Bridge",
        needs_auto_map=True,
    ),
)


def _minimal_config(case: RegistrationCase) -> PretrainedConfig:
    kwargs = {"architectures": [case.architecture]}
    if case.needs_auto_map:
        kwargs["auto_map"] = {"AutoModelForCausalLM": f"modeling_{case.id}.{case.architecture}"}
    return PretrainedConfig(**kwargs)


@pytest.mark.parametrize("case", REGISTRATION_CASES, ids=lambda case: case.id)
def test_autobridge_selects_registered_bridge_for_minimal_hf_config(case: RegistrationCase) -> None:
    module = importlib.import_module(case.module)
    expected_bridge_class = getattr(module, case.bridge_class)
    config = _minimal_config(case)

    assert issubclass(expected_bridge_class, MegatronModelBridge)
    assert AutoBridge.supports(config)
    assert case.architecture in AutoBridge.list_supported_models()

    bridge = AutoBridge.from_hf_config(config)
    selected_bridge = bridge._model_bridge

    assert type(selected_bridge) is expected_bridge_class
    assert selected_bridge.hf_config is config
