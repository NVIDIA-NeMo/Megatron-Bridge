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

from megatron.bridge.models._deprecation import _removed_model_name
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.recipes.nemotronh import nemotronh_4b_pretrain_config


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("fields", "expected_name"),
    [
        ({"architectures": ["DeepseekV2ForCausalLM"]}, "DeepSeek V2"),
        ({"architectures": ["DeciLMForCausalLM"]}, "Llama Nemotron"),
        (
            {"architectures": ["LlamaForCausalLM"], "name_or_path": "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"},
            "Llama Nemotron",
        ),
        ({"architectures": ["GemmaForCausalLM"]}, "Gemma 1"),
        ({"architectures": ["Gemma2ForCausalLM"]}, "Gemma 2"),
        ({"architectures": ["LlamaForCausalLM"], "vocab_size": 32000, "max_position_embeddings": 4096}, "Llama 2"),
        ({"architectures": ["LlamaForCausalLM"], "name_or_path": "meta-llama/Llama-2-70b-hf"}, "Llama 2"),
        (
            {
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "intermediate_size": 14336,
            },
            "Mistral 7B",
        ),
        (
            {
                "architectures": ["MistralForCausalLM"],
                "hidden_size": 5120,
                "num_hidden_layers": 40,
                "intermediate_size": 32768,
            },
            "Mistral 7B",
        ),
        (
            {"architectures": ["NemotronHForCausalLM"], "hidden_size": 4480, "num_hidden_layers": 56},
            "Nemotron Nano v2",
        ),
        (
            {"architectures": ["NemotronHForCausalLM"], "hidden_size": 5120, "num_hidden_layers": 62},
            "Nemotron Nano v2",
        ),
        ({"architectures": ["NemotronH_Nano_VL_V2"]}, "Nemotron Nano v2 VL"),
        ({"architectures": ["NemotronForCausalLM"]}, "legacy Nemotron bridge"),
    ],
)
def test_removed_models_are_rejected(fields, expected_name):
    config = PretrainedConfig(**fields)
    assert expected_name in _removed_model_name(config)
    assert not AutoBridge.supports(config)
    with pytest.raises(ValueError, match="removed"):
        AutoBridge(config)
    with pytest.raises(ValueError, match="removed"):
        AutoBridge.from_hf_config(config)
    with patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry", return_value=config):
        with pytest.raises(ValueError, match="removed"):
            AutoBridge.from_hf_pretrained("local-checkpoint")


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(architectures=["DeepseekV3ForCausalLM"]),
        SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
        SimpleNamespace(architectures=["Gemma3ForCausalLM"]),
        SimpleNamespace(architectures=["Gemma4ForCausalLM"]),
        SimpleNamespace(architectures=["LlamaForCausalLM"], vocab_size=128256, max_position_embeddings=131072),
        SimpleNamespace(architectures=["NemotronHForCausalLM"], hidden_size=2688, num_hidden_layers=52),
        SimpleNamespace(architectures=["MistralForCausalLM"], hidden_size=4096, num_hidden_layers=36),
        SimpleNamespace(architectures=["Mistral3ForConditionalGeneration"]),
        SimpleNamespace(
            architectures=["MistralForCausalLM"], hidden_size=5120, num_hidden_layers=40, intermediate_size=14336
        ),
        SimpleNamespace(architectures=["NemotronH_Nano_VL_V2"], llm_config=SimpleNamespace(n_routed_experts=128)),
    ],
)
def test_successor_configurations_remain_supported(config):
    assert _removed_model_name(config) is None
    assert AutoBridge.supports(config)


def test_legacy_nemotron_is_rejected_before_config_load():
    with patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry") as load_config:
        with pytest.raises(ValueError, match="Nemotron-4 340B.*removed"):
            AutoBridge.from_hf_pretrained("nvidia/Nemotron-4-340B-Instruct")
    load_config.assert_not_called()


def test_nemotron_h_v1_recipe_warns():
    with pytest.warns(FutureWarning, match=r"Nemotron H v1.*removed in Megatron Bridge 0\.7\.0"):
        nemotronh_4b_pretrain_config()


@pytest.mark.parametrize(
    "name",
    [
        "deepseek_v2_pretrain_config",
        "deepseek_v2_lite_pretrain_config",
        "gemma2_2b_pretrain_config",
        "gemma2_9b_sft_config",
        "gemma2_27b_peft_config",
        "llama2_7b_pretrain_config",
        "nemotron_nano_9b_v2_pretrain_config",
        "nemotron_nano_12b_v2_sft_config",
        "nemotron_nano_v2_vl_12b_peft_config",
        "deepseek_v2_lite_pretrain_8gpu_h100_bf16_config",
        "llama2_7b_pretrain_2gpu_h100_bf16_config",
        "gemma2_2b_sft_1gpu_h100_bf16_config",
    ],
)
def test_retired_recipe_factories_are_not_exported(name):
    import megatron.bridge.recipes as recipes

    assert not hasattr(recipes, name)


@pytest.mark.parametrize(
    ("hidden_size", "num_layers", "ffn_size"),
    [(4096, 36, 12288), (5120, 40, 14336)],
)
def test_shared_mistral_bridge_preserves_ministral_and_nemo(hidden_size, num_layers, ffn_size):
    from transformers import MistralConfig

    config = MistralConfig(
        architectures=["MistralForCausalLM"],
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        intermediate_size=ffn_size,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=131072,
        max_position_embeddings=131072,
        sliding_window=None,
    )
    provider = AutoBridge.from_hf_config(config).to_megatron_provider(load_weights=False)
    assert provider.hidden_size == hidden_size
    assert provider.num_layers == num_layers
    assert provider.ffn_hidden_size == ffn_size
