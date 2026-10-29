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
import torch
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TENorm,
)

from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer import (
    FalconH1MambaMixer,
    FalconH1MLP,
    FalconH1SelfAttention,
    SelfAttention,
    _run_mamba_mixer_with_static_cache_namespace,
)
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer_specs import falconh1_stack_spec
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Config


pytestmark = pytest.mark.unit


def _minimal_config(**overrides):
    kwargs = {
        "num_layers": 2,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_query_groups": 2,
        "ffn_hidden_size": 256,
    }
    kwargs.update(overrides)
    return FalconH1Config(**kwargs)


class _FakeInferenceContext:
    def __init__(self, *, is_static: bool):
        self._is_static = is_static
        self.key_value_memory_dict = {}

    def is_static_batching(self):
        return self._is_static


class _FakeMambaMixer:
    def __init__(self, *, layer_number: int):
        self.layer_number = layer_number
        self.called_layer_numbers = []

    def __call__(self, hidden_states, *, inference_context):
        self.called_layer_numbers.append(self.layer_number)
        inference_context.key_value_memory_dict[self.layer_number] = ("mamba-conv", "mamba-ssm")
        return hidden_states + 1, None


def test_config_rejects_invalid_a_init_distribution():
    with pytest.raises(ValueError, match="A_init_dist"):
        _minimal_config(A_init_dist="normal")


def test_config_rejects_mamba_inner_dim_not_divisible_by_head_dim():
    with pytest.raises(ValueError, match="must be divisible by mamba_head_dim"):
        _minimal_config(hidden_size=80, mamba_head_dim=64)


def test_mlp_applies_gate_and_down_multipliers():
    mlp = object.__new__(FalconH1MLP)
    object.__setattr__(
        mlp,
        "config",
        SimpleNamespace(
            activation_func=lambda x: x,
            activation_func_clamp_value=None,
            bias_activation_fusion=False,
            gated_linear_unit=True,
            glu_linear_offset=0.0,
            mlp_multipliers=(2.0, 0.5),
            use_te_activation_func=False,
        ),
    )
    object.__setattr__(mlp, "linear_fc1", lambda _: (torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]), None))
    object.__setattr__(mlp, "linear_fc2", lambda x: (x, torch.ones_like(x)))

    output, output_bias = mlp.forward(torch.zeros(1, 1, 2))

    torch.testing.assert_close(output, torch.tensor([[[3.0, 8.0]]]))
    torch.testing.assert_close(output_bias, torch.tensor([[[0.5, 0.5]]]))


def test_self_attention_applies_key_multiplier():
    attention = object.__new__(FalconH1SelfAttention)
    object.__setattr__(attention, "config", SimpleNamespace(key_multiplier=3.0))
    query = torch.tensor([1.0])
    key = torch.tensor([2.0])
    value = torch.tensor([4.0])

    with patch.object(SelfAttention, "get_query_key_value_tensors", return_value=(query, key, value)):
        scaled_query, scaled_key, scaled_value = attention.get_query_key_value_tensors(torch.zeros(1, 1, 1))

    torch.testing.assert_close(scaled_query, query)
    torch.testing.assert_close(scaled_key, torch.tensor([6.0]))
    torch.testing.assert_close(scaled_value, value)


def test_parallel_layer_normalizes_once_before_branch_input_multipliers():
    submodules = falconh1_stack_spec.submodules.falconh1_layer.submodules

    assert submodules.norm is TENorm
    assert submodules.mamba_mixer.submodules.in_proj is TEColumnParallelLinear
    assert submodules.self_attention.submodules.linear_qkv is TEColumnParallelLinear
    assert submodules.mlp.submodules.linear_fc1 is TELayerNormColumnParallelLinear


def test_static_mamba_cache_key_is_namespaced_when_layer_uses_attention():
    context = _FakeInferenceContext(is_static=True)
    context.key_value_memory_dict[7] = ("attention-key", "attention-value")
    mixer = _FakeMambaMixer(layer_number=7)
    hidden_states = torch.zeros(1, 1, 1)

    output, output_bias = _run_mamba_mixer_with_static_cache_namespace(
        mixer,
        hidden_states,
        context,
        use_attention=True,
    )

    torch.testing.assert_close(output, torch.ones(1, 1, 1))
    assert output_bias is None
    assert mixer.layer_number == 7
    assert mixer.called_layer_numbers == [("mamba", 7)]
    assert context.key_value_memory_dict[7] == ("attention-key", "attention-value")
    assert context.key_value_memory_dict[("mamba", 7)] == ("mamba-conv", "mamba-ssm")


@pytest.mark.parametrize(
    ("is_static", "use_attention", "expected_key"),
    [
        (True, False, 7),
        (False, True, 7),
    ],
)
def test_mamba_cache_key_is_not_namespaced_outside_static_attention_layers(is_static, use_attention, expected_key):
    context = _FakeInferenceContext(is_static=is_static)
    mixer = _FakeMambaMixer(layer_number=7)

    _run_mamba_mixer_with_static_cache_namespace(
        mixer,
        torch.zeros(1, 1, 1),
        context,
        use_attention=use_attention,
    )

    assert mixer.layer_number == 7
    assert mixer.called_layer_numbers == [expected_key]
    assert expected_key in context.key_value_memory_dict
    assert ("mamba", 7) not in context.key_value_memory_dict


def test_mamba_mixer_applies_ssm_multipliers():
    mixer = object.__new__(FalconH1MambaMixer)
    object.__setattr__(mixer, "config", SimpleNamespace(ssm_multipliers=(2.0, 3.0, 4.0, 5.0, 6.0)))
    object.__setattr__(mixer, "d_inner_local_tp", 2)
    object.__setattr__(mixer, "ngroups_local_tp", 1)
    object.__setattr__(mixer, "d_state", 2)
    object.__setattr__(mixer, "nheads_local_tp", 1)

    zxbc_dt = torch.ones(1, 1, 9)
    scaled = mixer._scale_zxbc_dt(zxbc_dt, use_context_parallel_dims=False)

    expected = torch.tensor([[[2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]]])
    torch.testing.assert_close(scaled, expected)
