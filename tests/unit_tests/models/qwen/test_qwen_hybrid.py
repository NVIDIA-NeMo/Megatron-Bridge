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

import pytest
from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

from megatron.bridge.models.qwen.qwen_hybrid import (
    QwenHybridModelProvider,
    configure_qwen_hybrid_layers,
    qwen_pipeline_layer_pattern,
)
from megatron.bridge.training.utils.config_utils import _ConfigContainerBase


def _provider() -> QwenHybridModelProvider:
    return QwenHybridModelProvider(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        bf16=False,
    )


def test_mtp_pattern_is_deferred_when_mtp_is_disabled():
    provider = _provider()
    configure_qwen_hybrid_layers(
        provider,
        num_logical_layers=2,
        mlp_symbols=Symbols.MLP,
        mtp_mlp_symbol=Symbols.MLP,
    )

    provider.finalize()

    assert provider.hybrid_layer_pattern == "*-*-"
    assert provider.mtp_hybrid_override_pattern == "*-"


def test_yarn_settings_are_serialized():
    provider = QwenHybridModelProvider(
        num_layers=2,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        bf16=False,
        position_embedding_type="yarn",
        yarn_rotary_scaling_factor=3.125,
        yarn_original_max_position_embeddings=40960,
        yarn_beta_fast=32.0,
        yarn_beta_slow=1.0,
        yarn_mscale=1.0,
        yarn_mscale_all_dim=1.0,
        yarn_correction_range_round_to_int=False,
    )
    expected_yarn_settings = {
        "position_embedding_type": "yarn",
        "yarn_rotary_scaling_factor": 3.125,
        "yarn_original_max_position_embeddings": 40960,
        "yarn_beta_fast": 32.0,
        "yarn_beta_slow": 1.0,
        "yarn_mscale": 1.0,
        "yarn_mscale_all_dim": 1.0,
        "yarn_correction_range_round_to_int": False,
    }

    serialized_provider = _ConfigContainerBase._convert_value_to_dict(provider)

    assert {key: serialized_provider[key] for key in expected_yarn_settings} == expected_yarn_settings


def test_mtp_pattern_honors_recipe_override_set_after_conversion():
    provider = _provider()
    configure_qwen_hybrid_layers(
        provider,
        num_logical_layers=2,
        mlp_symbols=Symbols.MLP,
        mtp_mlp_symbol=Symbols.MLP,
    )
    provider.mtp_num_layers = 1

    provider.finalize()

    assert provider.hybrid_layer_pattern == "*-*-/*-"
    assert provider.num_layers == 4


def test_pipeline_segmentation_preserves_logical_blocks_and_embedding_loss_balance():
    provider = QwenHybridModelProvider(
        num_layers=94,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_model_parallel_size=16,
        account_for_embedding_in_pipeline_split=True,
        account_for_loss_in_pipeline_split=True,
        bf16=False,
    )
    configure_qwen_hybrid_layers(
        provider,
        num_logical_layers=94,
        mlp_symbols=Symbols.MOE,
    )

    provider.finalize()

    segments = provider.hybrid_layer_pattern.split(Symbols.PIPE)
    assert [len(segment) for segment in segments] == [10] + [12] * 14 + [10]
    assert all(segment == "*E" * (len(segment) // 2) for segment in segments)
    assert provider.num_layers == 188
    assert provider.account_for_embedding_in_pipeline_split is False
    assert provider.account_for_loss_in_pipeline_split is False


def test_pipeline_segmentation_accounts_for_virtual_pipeline_stages():
    provider = QwenHybridModelProvider(
        num_layers=16,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        virtual_pipeline_model_parallel_size=4,
        bf16=False,
    )
    configure_qwen_hybrid_layers(
        provider,
        num_logical_layers=16,
        mlp_symbols=Symbols.MOE,
    )

    provider.finalize()

    segments = provider.hybrid_layer_pattern.split(Symbols.PIPE)
    assert len(segments) == 8
    assert all(segment == "*E*E" for segment in segments)


def test_pipeline_segmentation_rejects_stale_explicit_segments():
    with pytest.raises(ValueError, match="defines 2 pipeline segments"):
        qwen_pipeline_layer_pattern("*-*-|*-*-", pipeline_model_parallel_size=1)
