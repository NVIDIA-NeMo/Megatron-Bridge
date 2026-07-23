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

"""Shared config-only AutoBridge/provider contracts for high-risk model families.

This matrix is intentionally cheap: each case starts from an HF config, verifies
AutoBridge registration, and constructs the Megatron provider without weights.
It is the onboarding point for release bugs where the failure mode is a stale
architecture string, config mapping drift, or provider-construction contract.

When adding a case, keep the factory tiny, use the real HF architecture string,
assert at least one bug-specific provider field, and document the boundary in an
extra assertion. Weight conversion, export round trips, and runtime behavior
belong in narrower model-specific tests.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import pytest
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.utils import conform_config_to_reference
from megatron.bridge.models.qwen3_asr.hf_qwen3_asr.configuration_qwen3_asr import (
    Qwen3ASRConfig,
)
from megatron.bridge.models.stepfun.configuration_step35 import Step35Config


pytestmark = [pytest.mark.unit]

_STEP35_MAIN_LAYER_COUNT = 45
_STEP35_MTP_LAYER_COUNT = 3


@dataclass(frozen=True)
class ModelProviderContractCase:
    """Config-only bridge/provider contract for one HF architecture."""

    name: str
    architecture: str
    config_factory: Callable[[], PretrainedConfig]
    bridge_symbol: str
    provider_symbol: str
    expected_provider_attrs: Mapping[str, Any]
    extra_assertions: Callable[[PretrainedConfig, Any], None] | None = None


def _resolve_symbol(qualified_name: str) -> type:
    module_name, symbol_name = qualified_name.rsplit(".", 1)
    return getattr(import_module(module_name), symbol_name)


def _make_qwen3_asr_config() -> Qwen3ASRConfig:
    return Qwen3ASRConfig(
        architectures=["Qwen3ASRForConditionalGeneration"],
        thinker_config={
            "torch_dtype": "bfloat16",
            "audio_config": {
                "d_model": 1024,
                "encoder_layers": 2,
                "encoder_attention_heads": 16,
                "encoder_ffn_dim": 4096,
                "output_dim": 128,
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


def _make_step35_layer_types() -> list[str]:
    main_layer_types = [
        "full_attention",
        *["sliding_attention"] * 3,
    ] * 12
    return main_layer_types[:_STEP35_MAIN_LAYER_COUNT] + [
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]


def _make_step35_config() -> Step35Config:
    return Step35Config(
        architectures=["Step3p5ForCausalLM"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_attention_groups=2,
        num_hidden_layers=_STEP35_MAIN_LAYER_COUNT,
        vocab_size=512,
        max_position_embeddings=1024,
        moe_intermediate_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        share_expert_dim=64,
        head_dim=32,
        layer_types=_make_step35_layer_types(),
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "head_dim": 32,
        },
        sliding_window=128,
        num_nextn_predict_layers=_STEP35_MTP_LAYER_COUNT,
        moe_layers_enum=tuple(range(3, _STEP35_MAIN_LAYER_COUNT)),
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


def _make_nemotron_vl_config() -> PretrainedConfig:
    llm_config = PretrainedConfig(
        hybrid_override_pattern="M-M*-M-",
        hidden_size=256,
        intermediate_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        vocab_size=1024,
        max_position_embeddings=2048,
        hidden_act="relu2",
        torch_dtype="bfloat16",
    )
    return PretrainedConfig(
        architectures=["NemotronH_Nano_VL_V2"],
        model_type="nemotron_vl",
        llm_config=llm_config,
    )


def _assert_qwen3_asr_nested_config_contract(config: PretrainedConfig, provider: Any) -> None:
    assert isinstance(config, Qwen3ASRConfig)
    assert config.thinker_config.audio_config.d_model == 1024
    assert provider.thinker_config.audio_config.d_model == 1024

    run_config_dict = config.thinker_config.to_cfg_dict()
    assert run_config_dict["_target_"].endswith("Qwen3ASRThinkerConfig")
    assert run_config_dict["audio_config"]["d_model"] == 1024
    assert run_config_dict["text_config"]["hidden_size"] == 128

    reference_config_dict = config.to_dict()
    megatron_derived_config_dict = {
        "architectures": ["Qwen3ASRForConditionalGeneration"],
        "model_type": "qwen3_asr",
        "thinker_config": {
            "text_config": {
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "vocab_size": 512,
                "max_position_embeddings": 1024,
            }
        },
    }

    reconstructed_config_dict = conform_config_to_reference(megatron_derived_config_dict, reference_config_dict)
    reconstructed_config = Qwen3ASRConfig(**reconstructed_config_dict)

    # Historical NVBug 6314636: the non-recursive conform helper rebuilt the
    # audio encoder from defaults, changing d_model from 1024 to 1280.
    assert reconstructed_config.thinker_config.audio_config.d_model == 1024
    assert reconstructed_config.thinker_config.audio_config.encoder_attention_heads == 16
    assert reconstructed_config.thinker_config.audio_config.encoder_ffn_dim == 4096


def _assert_step35_published_layer_contract(config: PretrainedConfig, provider: Any) -> None:
    assert isinstance(config, Step35Config)
    assert len(_make_step35_layer_types()) == _STEP35_MAIN_LAYER_COUNT + _STEP35_MTP_LAYER_COUNT
    assert len(config.layer_types) == _STEP35_MAIN_LAYER_COUNT
    assert len(config.mtp_layer_types) == _STEP35_MTP_LAYER_COUNT
    assert provider.num_layers == _STEP35_MAIN_LAYER_COUNT
    assert provider.layer_types == config.layer_types + config.mtp_layer_types
    assert len(provider.moe_layer_freq) == _STEP35_MAIN_LAYER_COUNT
    assert provider.moe_layer_freq[:3] == [0, 0, 0]
    assert provider.moe_layer_freq[3:] == [1] * (_STEP35_MAIN_LAYER_COUNT - 3)


def _assert_mimo_v2_flash_coverage_boundary(config: PretrainedConfig, provider: Any) -> None:
    assert config.architectures == ["MiMoV2FlashForCausalLM"]
    assert provider.full_attn_num_query_groups == config.num_key_value_heads
    assert provider.swa_num_query_groups == config.swa_num_key_value_heads
    assert provider.v_head_dim == config.v_head_dim
    assert provider.mtp_num_layers == 0


def _assert_nemotron_labs_diffusion_coverage_boundary(config: PretrainedConfig, provider: Any) -> None:
    assert config.architectures == ["NemotronLabsDiffusionModel"]
    assert provider.hf_config is config
    assert provider.rotary_base == config.text_config.rope_parameters["rope_theta"]


def _assert_nemotron_vl_nested_llm_config(config: PretrainedConfig, provider: Any) -> None:
    assert config.architectures == ["NemotronH_Nano_VL_V2"]
    assert provider.make_vocab_size_divisible_by == 128
    assert provider.scatter_embedding_sequence_parallel is False
    assert provider.attention_softmax_in_fp32 is True


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
        extra_assertions=_assert_qwen3_asr_nested_config_contract,
    ),
    ModelProviderContractCase(
        name="step35_published_layer_types_with_mtp",
        architecture="Step3p5ForCausalLM",
        config_factory=_make_step35_config,
        bridge_symbol="megatron.bridge.models.stepfun.step35_bridge.Step35Bridge",
        provider_symbol="megatron.bridge.models.stepfun.step35_provider.Step35ModelProvider",
        expected_provider_attrs={
            "hidden_size": 128,
            "num_layers": _STEP35_MAIN_LAYER_COUNT,
            "num_query_groups": 2,
            "num_moe_experts": 4,
            "moe_router_topk": 2,
        },
        extra_assertions=_assert_step35_published_layer_contract,
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
        extra_assertions=_assert_mimo_v2_flash_coverage_boundary,
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
        extra_assertions=_assert_nemotron_labs_diffusion_coverage_boundary,
    ),
    ModelProviderContractCase(
        name="nemotron_vl_nested_llm_config",
        architecture="NemotronH_Nano_VL_V2",
        config_factory=_make_nemotron_vl_config,
        bridge_symbol="megatron.bridge.models.nemotron_vl.nemotron_vl_bridge.NemotronVLBridge",
        provider_symbol="megatron.bridge.models.nemotron_vl.nemotron_vl_provider.NemotronVLModelProvider",
        expected_provider_attrs={
            "hidden_size": 256,
            "ffn_hidden_size": 512,
            "num_attention_heads": 8,
            "num_query_groups": 2,
            "vocab_size": 1024,
            "seq_length": 2048,
        },
        extra_assertions=_assert_nemotron_vl_nested_llm_config,
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
    if case.extra_assertions is not None:
        case.extra_assertions(config, provider)
