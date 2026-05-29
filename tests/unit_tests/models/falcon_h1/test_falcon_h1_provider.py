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

from unittest.mock import Mock, patch

import pytest

from megatron.bridge.models.falcon_h1.falconh1_provider import FalconH1ModelProvider


pytestmark = pytest.mark.unit


def _minimal_provider(**overrides):
    kwargs = {
        "num_layers": 2,
        "hidden_size": 128,
        "ffn_hidden_size": 256,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "kv_channels": 32,
        "vocab_size": 1000,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
    }
    kwargs.update(overrides)
    return FalconH1ModelProvider(**kwargs)


def test_provider_defaults_do_not_apply_mup_scaling():
    provider = _minimal_provider()

    assert provider.embedding_multiplier == 1.0
    assert provider.lm_head_multiplier == 1.0
    assert provider.key_multiplier == 1.0
    assert provider.attention_in_multiplier == 1.0
    assert provider.attention_out_multiplier == 1.0
    assert provider.ssm_in_multiplier == 1.0
    assert provider.ssm_out_multiplier == 1.0
    assert provider.mlp_multipliers == (1.0, 1.0)
    assert provider.ssm_multipliers == (1.0, 1.0, 1.0, 1.0, 1.0)


def test_provide_preserves_explicit_pipeline_stage_false_values():
    provider = _minimal_provider()

    with (
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.parallel_state.is_pipeline_first_stage") as first,
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.parallel_state.is_pipeline_last_stage") as last,
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.FalconH1Model") as model_cls,
    ):
        first.return_value = True
        last.return_value = True
        model_cls.return_value = Mock()

        provider.provide(pre_process=False, post_process=False)

    call_kwargs = model_cls.call_args.kwargs
    assert call_kwargs["pre_process"] is False
    assert call_kwargs["post_process"] is False


def test_provide_uses_pipeline_stage_defaults_when_unspecified():
    provider = _minimal_provider()

    with (
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.parallel_state.is_pipeline_first_stage") as first,
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.parallel_state.is_pipeline_last_stage") as last,
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.FalconH1Model") as model_cls,
    ):
        first.return_value = False
        last.return_value = True
        model_cls.return_value = Mock()

        provider.provide()

    call_kwargs = model_cls.call_args.kwargs
    assert call_kwargs["pre_process"] is False
    assert call_kwargs["post_process"] is True


def test_provide_calculates_padded_vocab_size_when_enabled():
    provider = _minimal_provider(vocab_size=1001, should_pad_vocab=True, make_vocab_size_divisible_by=128)

    with (
        patch(
            "megatron.bridge.models.falcon_h1.falconh1_provider.calculate_padded_vocab_size",
            return_value=1024,
        ) as calc_vocab,
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.FalconH1Model") as model_cls,
    ):
        model_cls.return_value = Mock()

        provider.provide(pre_process=True, post_process=True)

    calc_vocab.assert_called_once_with(1001, 128, 1)
    assert model_cls.call_args.kwargs["vocab_size"] == 1024


def test_provide_uses_original_vocab_size_without_padding():
    provider = _minimal_provider(vocab_size=1001, should_pad_vocab=False)

    with (
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.calculate_padded_vocab_size") as calc_vocab,
        patch("megatron.bridge.models.falcon_h1.falconh1_provider.FalconH1Model") as model_cls,
    ):
        model_cls.return_value = Mock()

        provider.provide(pre_process=True, post_process=True)

    calc_vocab.assert_not_called()
    assert model_cls.call_args.kwargs["vocab_size"] == 1001
