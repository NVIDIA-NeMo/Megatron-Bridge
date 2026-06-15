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

"""Shared config-only AutoBridge/provider contracts for high-risk model families."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import pytest
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.configuration_qwen3_asr import Qwen3ASRConfig
from megatron.bridge.models.stepfun.configuration_step35 import Step35Config


pytestmark = [pytest.mark.unit]


@dataclass(frozen=True)
class ModelProviderContractCase:
    """Config-only bridge/provider contract for one HF architecture."""

    name: str
    architecture: str
    config_factory: Callable[[], PretrainedConfig]
    bridge_symbol: str
    provider_symbol: str
    expected_provider_attrs: Mapping[str, Any]


def _resolve_symbol(qualified_name: str) -> type:
    module_name, symbol_name = qualified_name.rsplit(".", 1)
    return getattr(import_module(module_name), symbol_name)


def _make_qwen3_asr_config() -> Qwen3ASRConfig:
    return Qwen3ASRConfig(
        architectures=["Qwen3ASRForConditionalGeneration"],
        thinker_config={
            "torch_dtype": "bfloat16",
            "audio_config": {
                "encoder_layers": 2,
            },
            "text_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "max_position_embeddings": 1024,
                "initializer_range": 0.02,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5000000.0,
                "tie_word_embeddings": False,
            },
        },
    )


def _make_step35_config() -> Step35Config:
    return Step35Config(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_attention_groups=2,
        num_hidden_layers=4,
        vocab_size=512,
        max_position_embeddings=1024,
        moe_intermediate_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        share_expert_dim=64,
        head_dim=32,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 32,
        },
        sliding_window=128,
        num_nextn_predict_layers=2,
        moe_layers_enum=(2, 3),
        torch_dtype="bfloat16",
    )


def _make_mimo_v2_flash_config() -> PretrainedConfig:
    return PretrainedConfig(
        architectures=["MiMoV2FlashForCausalLM"],
        model_type="mimo_v2_flash",
        num_hidden_layers=6,
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=1024,
        max_position_embeddings=2048,
        rope_theta=5000000,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
        layernorm_epsilon=1e-5,
        v_head_dim=16,
        hybrid_layer_pattern=[0, 1, 1, 1, 0, 1],
        sliding_window_size=128,
        sliding_window=128,
        attention_chunk_size=128,
        swa_rope_theta=10000,
        swa_num_key_value_heads=4,
        swa_num_attention_heads=8,
        swa_head_dim=32,
        swa_v_head_dim=16,
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=False,
        attention_value_scale=0.707,
        moe_layer_freq=[0, 1, 1, 1, 1, 1],
        n_routed_experts=8,
        moe_intermediate_size=128,
        num_experts_per_tok=2,
        scoring_func="sigmoid",
        n_shared_experts=None,
        n_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        norm_topk_prob=True,
        routed_scaling_factor=None,
        torch_dtype="bfloat16",
    )


def _make_nemotron_labs_diffusion_config() -> PretrainedConfig:
    text_config = PretrainedConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        tie_word_embeddings=True,
        rope_parameters={"rope_theta": 10000.0},
        vocab_size=512,
    )
    return PretrainedConfig(
        architectures=["NemotronLabsDiffusionModel"],
        model_type="nemotron_labs_diffusion",
        text_config=text_config,
    )


G_CONTRACT_CASES = (
    ModelProviderContractCase(
        name="qwen3_asr_nested_config",
        architecture="Qwen3ASRForConditionalGeneration",
        config_factory=_make_qwen3_asr_config,
        bridge_symbol="megatron.bridge.models.qwen3_asr.qwen3_asr_bridge.Qwen3ASRBridge",
        provider_symbol="megatron.bridge.models.qwen3_asr.qwen3_asr_provider.Qwen3ASRModelProvider",
        expected_provider_attrs={
            "hidden_size": 128,
            "num_layers": 2,
            "num_query_groups": 2,
            "vocab_size": 512,
            "audio_token_id": 151646,
            "share_embeddings_and_output_weights": False,
        },
    ),
    ModelProviderContractCase(
        name="step35_mtp_layer_types",
        architecture="Step3p5ForCausalLM",
        config_factory=_make_step35_config,
        bridge_symbol="megatron.bridge.models.stepfun.step35_bridge.Step35Bridge",
        provider_symbol="megatron.bridge.models.stepfun.step35_provider.Step35ModelProvider",
        expected_provider_attrs={
            "hidden_size": 128,
            "num_layers": 4,
            "num_query_groups": 2,
            "num_moe_experts": 4,
            "moe_router_topk": 2,
            "moe_layer_freq": [0, 0, 1, 1],
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
                "sliding_attention",
            ],
        },
    ),
    ModelProviderContractCase(
        name="mimo_v2_flash_registration",
        architecture="MiMoV2FlashForCausalLM",
        config_factory=_make_mimo_v2_flash_config,
        bridge_symbol="megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge.MiMoV2FlashBridge",
        provider_symbol="megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider.MiMoV2FlashModelProvider",
        expected_provider_attrs={
            "hidden_size": 256,
            "num_layers": 6,
            "num_query_groups": 2,
            "full_attn_num_query_groups": 2,
            "swa_num_query_groups": 4,
            "v_head_dim": 16,
            "window_size": 128,
            "mtp_num_layers": 0,
        },
    ),
    ModelProviderContractCase(
        name="nemotron_labs_diffusion_text_config",
        architecture="NemotronLabsDiffusionModel",
        config_factory=_make_nemotron_labs_diffusion_config,
        bridge_symbol=(
            "megatron.bridge.diffusion.conversion.nemotron_labs_diffusion."
            "nemotron_labs_diffusion_bridge.NemotronLabsDiffusionBridge"
        ),
        provider_symbol=(
            "megatron.bridge.diffusion.models.nemotron_labs_diffusion."
            "nemotron_labs_diffusion_provider.NemotronLabsDiffusionModelProvider"
        ),
        expected_provider_attrs={
            "hidden_size": 128,
            "ffn_hidden_size": 256,
            "num_layers": 2,
            "vocab_size": 512,
            "share_embeddings_and_output_weights": True,
            "rotary_base": 10000.0,
        },
    ),
)


@pytest.mark.parametrize("case", G_CONTRACT_CASES, ids=[case.name for case in G_CONTRACT_CASES])
def test_config_only_autobridge_provider_contract(case: ModelProviderContractCase) -> None:
    bridge_type = _resolve_symbol(case.bridge_symbol)
    provider_type = _resolve_symbol(case.provider_symbol)
    config = case.config_factory()

    assert AutoBridge.supports(config) is True
    assert case.architecture in AutoBridge.list_supported_models()

    bridge = AutoBridge.from_hf_config(config)
    assert isinstance(bridge._model_bridge, bridge_type)

    provider = bridge.to_megatron_provider(load_weights=False)
    assert isinstance(provider, provider_type)
    for attr_name, expected_value in case.expected_provider_attrs.items():
        assert getattr(provider, attr_name) == expected_value
