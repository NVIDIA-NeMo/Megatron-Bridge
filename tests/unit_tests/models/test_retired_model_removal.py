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

from unittest.mock import patch

import pytest
from transformers import AutoConfig, LlamaConfig, MistralConfig, PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "architecture",
    ["DeepseekV2ForCausalLM", "DeciLMForCausalLM", "GemmaForCausalLM", "Gemma2ForCausalLM", "NemotronForCausalLM"],
)
def test_retired_architectures_have_no_bridge(architecture):
    config = PretrainedConfig(architectures=[architecture])
    assert architecture not in AutoBridge.list_supported_models()
    with pytest.raises(ValueError, match="not yet supported"):
        AutoBridge.from_hf_config(config)
    with patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry", return_value=config):
        assert not AutoBridge.can_handle("local-checkpoint")
        with pytest.raises(ValueError, match="not yet supported"):
            AutoBridge.from_hf_pretrained("local-checkpoint")


@pytest.mark.parametrize(
    ("config", "relative_path"),
    [
        (
            MistralConfig(
                architectures=["MistralForCausalLM"],
                hidden_size=5120,
                num_hidden_layers=40,
                intermediate_size=14336,
                vocab_size=131072,
            ),
            "models/mistral-7b-comparison/nemo",
        ),
        (
            LlamaConfig(architectures=["LlamaForCausalLM"], vocab_size=128256, max_position_embeddings=131072),
            "data/nemotron-sft/Llama-3.1-8B-Instruct",
        ),
        (
            LlamaConfig(architectures=["LlamaForCausalLM"], vocab_size=128256, max_position_embeddings=131072),
            "models/nemotron-4-340b-comparison/llama3",
        ),
    ],
)
def test_supported_local_checkpoint_ignores_parent_directory_names(config, relative_path, tmp_path):
    checkpoint = tmp_path / relative_path
    config.save_pretrained(checkpoint)
    loaded_config = AutoConfig.from_pretrained(checkpoint)
    assert loaded_config.name_or_path == str(checkpoint)
    assert AutoBridge.supports(loaded_config)
    assert AutoBridge.can_handle(checkpoint)
    assert AutoBridge.from_hf_pretrained(checkpoint).to_megatron_provider(load_weights=False) is not None


def test_nemotron_3_finetune_name_is_not_a_retirement_signal():
    config = PretrainedConfig(
        architectures=["NemotronHForCausalLM"],
        name_or_path="my-org/nemotron-nano-30b-a3b-sft-v2",
        hidden_size=2688,
        num_hidden_layers=52,
        n_routed_experts=128,
    )
    assert AutoBridge.supports(config)
    assert AutoBridge.from_hf_config(config) is not None
    with patch("megatron.bridge.models.conversion.auto_bridge.safe_load_config_with_retry", return_value=config):
        assert AutoBridge.can_handle(config.name_or_path)
        assert AutoBridge.from_hf_pretrained(config.name_or_path) is not None


def test_nemotron_h_v1_factories_are_not_exported():
    from megatron.bridge import perf_recipes, recipes
    from megatron.bridge.perf_recipes import nemotronh as perf_nemotron
    from megatron.bridge.recipes import nemotronh as library_nemotron
    from megatron.bridge.recipes.nemotronh import h100

    for module in (recipes, library_nemotron, h100, perf_recipes, perf_nemotron):
        assert not any(name.startswith("nemotronh_") and callable(value) for name, value in vars(module).items())


@pytest.mark.parametrize(
    "name",
    [
        "nemotronh_4b_pretrain_config",
        "nemotronh_8b_sft_config",
        "nemotronh_47b_peft_config",
        "nemotronh_56b_pretrain_config",
        "nemotronh_56b_pretrain_8gpu_h100_bf16_config",
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
