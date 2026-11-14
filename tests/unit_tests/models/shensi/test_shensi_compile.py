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

"""The opt-in ``shensi_compile_modules`` path reproduces the eager results and gradients."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.shensi import modeling_shensi
from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig
from megatron.bridge.models.shensi.modeling_shensi import ShensiAttentionResidual, ShensiTransformerLayer


def _hc_config() -> SimpleNamespace:
    # Three routed streams with two selectable: the top-k width differs from the stream count,
    # which also exercises the balancing loss over the full route logits.
    return SimpleNamespace(
        hidden_size=16,
        hc_mult=4,
        hc_active_streams=3,
        hc_fixed_streams=1,
        hc_adapter_rank=4,
        hc_conv_kernels=[3],
        layernorm_epsilon=1e-6,
        hc_router_aux_loss_coef=0.01,
        sequence_parallel=False,
    )


def _attn_res_config() -> SimpleNamespace:
    return SimpleNamespace(hidden_size=16, attn_res_rank=4, layernorm_epsilon=1e-6)


def test_compile_option_defaults_off() -> None:
    assert ShensiTransformerConfig(num_layers=1, hidden_size=16).shensi_compile_modules is False


def test_layer_dispatch_uses_the_compiled_wrapper(monkeypatch) -> None:
    """``shensi_compile_modules`` routes the short modules through the shared compiled graph."""
    calls = []

    def spy(module, prefix, delta, blocks, num_blocks, block_idx):
        calls.append((num_blocks, block_idx))
        visible = torch.cat([blocks[..., : 1 + block_idx, :], prefix.unsqueeze(-2)], dim=-2)
        return module(prefix, delta, visible, num_blocks)

    monkeypatch.setattr(modeling_shensi, "_compiled_attention_residual", spy)
    layer = SimpleNamespace(compile_short_modules=True, recompute_short_modules=False, training=False, block_idx=1)
    module = ShensiAttentionResidual(_attn_res_config())
    prefix = torch.randn(4, 1, 16)
    delta = torch.randn(4, 1, 16)
    blocks = torch.randn(4, 1, 3, 16)

    ShensiTransformerLayer._attention_residual(layer, module, prefix, delta, blocks, 3)
    assert calls == [(3, 1)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile runs on the GPU path")
def test_compiled_short_modules_match_eager() -> None:
    from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    torch.manual_seed(0)
    device = torch.cuda.current_device()

    hc = modeling_shensi.ShensiHyperConnection(_hc_config(), is_mlp=True).to(device)
    streams = torch.randn(8, 2, 4, 16, device=device)
    sublayer = torch.randn(8, 2, 16, device=device, requires_grad=True)
    eager_in = streams.clone().requires_grad_(True)
    compiled_in = streams.clone().requires_grad_(True)

    eager_out, eager_state = hc(eager_in)
    eager_streams = hc.write_back(sublayer, eager_state, sublayer)
    compiled_out, compiled_state = modeling_shensi._compiled_hyper_connection(hc, compiled_in)
    compiled_streams = hc.write_back(sublayer, compiled_state, sublayer)
    torch.testing.assert_close(compiled_out, eager_out)
    torch.testing.assert_close(compiled_streams, eager_streams)
    for key in ("residual", "mixed", "post", "indices", "route_weights"):
        torch.testing.assert_close(compiled_state[key], eager_state[key])

    (eager_out.float().sum() + eager_streams.float().sum()).backward()
    eager_grads = {name: param.grad.detach().clone() for name, param in hc.named_parameters()}
    hc.zero_grad()
    (compiled_out.float().sum() + compiled_streams.float().sum()).backward()
    torch.testing.assert_close(compiled_in.grad, eager_in.grad)
    for name, param in hc.named_parameters():
        torch.testing.assert_close(param.grad, eager_grads[name], rtol=1e-4, atol=1e-5, msg=name)

    attn_res = ShensiAttentionResidual(_attn_res_config()).to(device)
    prefix = torch.randn(8, 2, 16, device=device)
    delta = torch.randn(8, 2, 16, device=device)
    blocks = torch.randn(8, 2, 3, 16, device=device)

    def run(module, compiled):
        prefix_in, delta_in, blocks_in = (tensor.clone().requires_grad_(True) for tensor in (prefix, delta, blocks))
        if compiled:
            output = modeling_shensi._compiled_attention_residual(module, prefix_in, delta_in, blocks_in, 3, 1)[0]
        else:
            visible = torch.cat([blocks_in[..., :2, :], prefix_in.unsqueeze(-2)], dim=-2)
            output = module(prefix_in, delta_in, visible, 3)[0]
        output.float().sum().backward()
        return output.detach(), [prefix_in.grad, delta_in.grad, blocks_in.grad]

    eager_out, eager_grads = run(attn_res, compiled=False)
    compiled_out, compiled_grads = run(attn_res, compiled=True)
    torch.testing.assert_close(compiled_out, eager_out)
    for eager_grad, compiled_grad in zip(eager_grads, compiled_grads):
        torch.testing.assert_close(compiled_grad, eager_grad)
