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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from megatron.bridge.models._deprecation import _deprecated_model_name
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.nemotronh import nemotronh_4b_pretrain_config


@pytest.mark.parametrize(
    ("config", "expected_name"),
    [
        (SimpleNamespace(architectures=["GemmaForCausalLM"]), "Gemma 1"),
        (SimpleNamespace(architectures=["Gemma2ForCausalLM"]), "Gemma 2"),
        (
            SimpleNamespace(architectures=["LlamaForCausalLM"], vocab_size=32000, max_position_embeddings=4096),
            "Llama 2",
        ),
        (
            SimpleNamespace(architectures=["MistralForCausalLM"], hidden_size=5120, num_hidden_layers=40),
            "Mistral 7B and Mistral Small 3 24B",
        ),
        (
            SimpleNamespace(architectures=["NemotronHForCausalLM"], hidden_size=4096, num_hidden_layers=52),
            "Nemotron H v1",
        ),
        (SimpleNamespace(architectures=["NemotronForCausalLM"]), "legacy Nemotron bridge"),
    ],
)
def test_deprecated_model_detection(config, expected_name):
    assert expected_name in _deprecated_model_name(config)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(architectures=["LlamaForCausalLM"], vocab_size=128256, max_position_embeddings=131072),
        SimpleNamespace(architectures=["NemotronHForCausalLM"], hidden_size=4480, num_hidden_layers=56),
        SimpleNamespace(architectures=["Mistral3ForConditionalGeneration"]),
    ],
)
def test_active_model_detection(config):
    assert _deprecated_model_name(config) is None


def test_auto_bridge_warns_for_deprecated_model():
    config = PretrainedConfig(architectures=["Gemma2ForCausalLM"])

    with pytest.warns(FutureWarning, match=r"Gemma 2.*removed in Megatron Bridge 0\.7\.0"):
        AutoBridge(config)


def test_legacy_nemotron_warns_before_config_load():
    with patch(
        "megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry",
        side_effect=OSError("config.json is unavailable"),
    ):
        with pytest.warns(FutureWarning, match=r"Nemotron-4 340B.*removed in Megatron Bridge 0\.7\.0"):
            with pytest.raises(OSError, match="config.json is unavailable"):
                AutoBridge.from_hf_pretrained("nvidia/Nemotron-4-340B-Instruct")


def test_nemotron_h_v1_recipe_warns():
    with pytest.warns(FutureWarning, match=r"Nemotron H v1.*removed in Megatron Bridge 0\.7\.0"):
        nemotronh_4b_pretrain_config()
