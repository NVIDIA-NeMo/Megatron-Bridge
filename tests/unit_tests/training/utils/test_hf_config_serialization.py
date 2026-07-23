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

from dataclasses import dataclass

import pytest
import yaml
from transformers.models.llava.configuration_llava import LlavaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
    Qwen3OmniMoeTalkerTextConfig,
)
from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

from megatron.bridge.models.qwen_omni import Qwen3OmniModelProvider
from megatron.bridge.recipes.common import _sft_common_vlm
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
from megatron.bridge.utils.instantiate_utils import InstantiationMode


pytestmark = pytest.mark.unit


def _code_predictor_config() -> dict[str, object]:
    return {
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 5,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_code_groups": 16,
        "vocab_size": 2048,
    }


def _text_config() -> dict[str, object]:
    return {
        "hidden_size": 1024,
        "intermediate_size": 2048,
        "moe_intermediate_size": 384,
        "num_hidden_layers": 20,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "num_experts": 128,
        "num_experts_per_tok": 6,
        "vocab_size": 3072,
    }


def _talker_config() -> Qwen3OmniMoeTalkerConfig:
    return Qwen3OmniMoeTalkerConfig(
        code_predictor_config=_code_predictor_config(),
        text_config=_text_config(),
        thinker_hidden_size=2048,
        num_code_groups=16,
    )


@dataclass
class QwenTalkerConfigContainer(_ConfigContainerBase):
    talker_config: Qwen3OmniMoeTalkerConfig


@dataclass
class LlavaConfigContainer(_ConfigContainerBase):
    model_config: LlavaConfig


@dataclass
class RecursiveLeafConfig:
    value: int


@dataclass
class RecursiveBranchConfig:
    leaf: RecursiveLeafConfig


@dataclass
class RecursiveConfigContainer(_ConfigContainerBase):
    branch: RecursiveBranchConfig


@pytest.mark.parametrize("nested_as_dict", [False, True])
def test_qwen_talker_config_yaml_roundtrip(tmp_path, nested_as_dict):
    talker_config = _talker_config()
    if nested_as_dict:
        talker_config.code_predictor_config = _code_predictor_config()
        talker_config.text_config = _text_config()

    config = QwenTalkerConfigContainer(talker_config=talker_config)
    yaml_path = tmp_path / "run_config.yaml"
    config.to_yaml(str(yaml_path))

    serialized = yaml.safe_load(yaml_path.read_text())
    serialized_talker = serialized["talker_config"]
    assert serialized_talker["_target_"].endswith(".Qwen3OmniMoeTalkerConfig")
    assert "_target_" not in serialized_talker["code_predictor_config"]
    assert "_target_" not in serialized_talker["text_config"]

    loaded = QwenTalkerConfigContainer.from_yaml(str(yaml_path), mode=InstantiationMode.STRICT)
    assert isinstance(loaded.talker_config, Qwen3OmniMoeTalkerConfig)
    assert isinstance(loaded.talker_config.code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig)
    assert isinstance(loaded.talker_config.text_config, Qwen3OmniMoeTalkerTextConfig)
    assert loaded.talker_config.code_predictor_config.num_hidden_layers == 5
    assert loaded.talker_config.code_predictor_config.hidden_size == 1024
    assert loaded.talker_config.text_config.num_hidden_layers == 20
    assert loaded.talker_config.text_config.num_experts == 128


def test_qwen_provider_config_container_yaml_roundtrip(tmp_path):
    config = _sft_common_vlm()
    config.model = Qwen3OmniModelProvider(
        num_layers=2,
        hidden_size=128,
        num_attention_heads=4,
        talker_config=_talker_config(),
    )
    yaml_path = tmp_path / "run_config.yaml"
    config.to_yaml(str(yaml_path))

    loaded = ConfigContainer.from_yaml(str(yaml_path), mode=InstantiationMode.STRICT)
    assert isinstance(loaded.model, Qwen3OmniModelProvider)
    assert isinstance(loaded.model.talker_config, Qwen3OmniMoeTalkerConfig)
    assert isinstance(loaded.model.talker_config.code_predictor_config, Qwen3OmniMoeTalkerCodePredictorConfig)
    assert isinstance(loaded.model.talker_config.text_config, Qwen3OmniMoeTalkerTextConfig)
    assert loaded.model.talker_config.code_predictor_config.hidden_size == 1024
    assert loaded.model.talker_config.text_config.num_experts == 128


def test_other_composite_hf_config_yaml_roundtrip(tmp_path):
    config = LlavaConfigContainer(
        model_config=LlavaConfig(
            vision_config=SiglipVisionConfig(hidden_size=768, num_attention_heads=12),
            text_config=MistralConfig(hidden_size=1024, num_attention_heads=16, num_key_value_heads=8),
        )
    )
    yaml_path = tmp_path / "run_config.yaml"
    config.to_yaml(str(yaml_path))

    serialized = yaml.safe_load(yaml_path.read_text())
    assert serialized["model_config"]["_target_"].endswith(".LlavaConfig")
    assert "_target_" not in serialized["model_config"]["vision_config"]
    assert "_target_" not in serialized["model_config"]["text_config"]
    assert serialized["model_config"]["vision_config"]["model_type"] == "siglip_vision_model"
    assert serialized["model_config"]["text_config"]["model_type"] == "mistral"

    loaded = LlavaConfigContainer.from_yaml(str(yaml_path), mode=InstantiationMode.STRICT)
    assert isinstance(loaded.model_config, LlavaConfig)
    assert isinstance(loaded.model_config.vision_config, SiglipVisionConfig)
    assert isinstance(loaded.model_config.text_config, MistralConfig)
    assert loaded.model_config.vision_config.hidden_size == 768
    assert loaded.model_config.text_config.hidden_size == 1024


def test_non_hf_nested_configs_remain_recursive(tmp_path):
    config = RecursiveConfigContainer(branch=RecursiveBranchConfig(leaf=RecursiveLeafConfig(value=17)))
    yaml_path = tmp_path / "run_config.yaml"
    config.to_yaml(str(yaml_path))

    serialized = yaml.safe_load(yaml_path.read_text())
    assert serialized["branch"]["_target_"].endswith(".RecursiveBranchConfig")
    assert serialized["branch"]["leaf"]["_target_"].endswith(".RecursiveLeafConfig")

    loaded = RecursiveConfigContainer.from_yaml(str(yaml_path), mode=InstantiationMode.STRICT)
    assert isinstance(loaded.branch, RecursiveBranchConfig)
    assert isinstance(loaded.branch.leaf, RecursiveLeafConfig)
    assert loaded.branch.leaf.value == 17
