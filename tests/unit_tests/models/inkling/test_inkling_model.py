# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Compare Inkling's native operators with the independent Transformers reference."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.inkling.modeling_inkling import (
    InklingShortConvolution as HFShortConvolution,
)
from transformers.models.inkling.modeling_inkling import (
    InklingTopkRouter,
    eager_attention_forward,
)

from megatron.bridge.models.inkling import InklingModelProvider
from megatron.bridge.models.inkling.modeling_inkling import (
    InklingShortConvolution,
    joint_router_weights,
    relative_attention,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires CUDA"),
        ),
    ],
)
def test_relative_attention_outputs_and_gradients_match_reference(window, device):
    torch.manual_seed(42)
    requires_grad = device == "cuda"
    inputs = [
        torch.randn(*shape, device=device, requires_grad=requires_grad)
        for shape in [(2, 4, 9, 16), (2, 2, 9, 16), (2, 2, 9, 16), (2, 4, 9, 4)]
    ]
    reference_inputs = [value.detach().clone().requires_grad_(requires_grad) for value in inputs]
    padding = torch.zeros(2, 9, dtype=torch.bool, device=device)
    padding[1, 6:] = True
    actual = relative_attention(*inputs, window_size=window, padding_mask=padding)

    positions = torch.arange(9, device=device)
    distance = positions[:, None] - positions[None, :]
    query, key, value, bank = reference_inputs
    bias = bank.gather(-1, distance.clamp(0, 3)[None, None].expand(2, 4, -1, -1))
    bias = bias.masked_fill((distance < 0) | (distance >= 4), 0.0)
    excluded = (distance < 0)[None, None] | padding[:, None, None, :]
    if window is not None:
        excluded = excluded | (distance >= window)
    mask = torch.zeros_like(excluded, dtype=torch.float32).masked_fill(excluded, torch.finfo(torch.float32).min)
    expected, _ = eager_attention_forward(
        SimpleNamespace(num_key_value_groups=2, training=True),
        query,
        key,
        value,
        mask,
        scaling=1 / 16,
        position_bias=bias,
    )
    actual = actual.transpose(1, 2)[~padding]
    expected = expected[~padding]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    if not requires_grad:
        return
    actual.square().sum().backward()
    expected.square().sum().backward()
    for native, reference in zip(inputs, reference_inputs):
        torch.testing.assert_close(native.grad, reference.grad, atol=5e-6, rtol=2e-4)


def test_joint_routed_and_shared_weights_match_transformers():
    torch.manual_seed(43)
    config = SimpleNamespace(
        n_routed_experts=4, n_shared_experts=2, hidden_size=8, route_scale=1.7, num_experts_per_tok=2
    )
    reference = InklingTopkRouter(config)
    with torch.no_grad():
        reference.weight.normal_()
        reference.e_score_correction_bias.normal_(std=0.1)
        reference.global_scale.fill_(0.7)
    hidden = torch.randn(2, 3, 8)
    _, routed, indices, shared = reference(hidden)
    native_indices, native_routed, native_shared = joint_router_weights(
        torch.nn.functional.linear(hidden.flatten(0, 1), reference.weight),
        reference.e_score_correction_bias,
        reference.global_scale,
        2,
        1.7,
    )
    assert torch.equal(native_indices, indices)
    torch.testing.assert_close(native_routed, routed, atol=0, rtol=0)
    torch.testing.assert_close(native_shared, shared, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Transformers causal-conv1d kernel requires CUDA")
def test_short_convolution_padding_and_gradients_match_transformers():
    torch.manual_seed(44)
    config = InklingModelProvider(num_layers=1, hidden_size=8, num_attention_heads=1, inkling_conv_kernel_size=3)
    config.finalize()
    native = InklingShortConvolution(config, 8).cuda()
    reference = HFShortConvolution(8, 3, layer_idx=0, conv_idx=0).cuda()
    reference.load_state_dict(native.state_dict())
    hidden = torch.randn(7, 2, 8, device="cuda", requires_grad=True)
    reference_hidden = hidden.detach().transpose(0, 1).clone().requires_grad_()
    padding = torch.zeros(2, 7, dtype=torch.bool, device="cuda")
    padding[1, :2] = True
    actual = native(hidden, padding)
    expected = reference(reference_hidden, conv_mask=~padding).transpose(0, 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(hidden.grad, reference_hidden.grad.transpose(0, 1))
    torch.testing.assert_close(native.conv1d.weight.grad, reference.conv1d.weight.grad)
