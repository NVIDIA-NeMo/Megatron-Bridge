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

"""ERC trains the router and experts through MCore's auxiliary-loss autoscaler."""

from types import SimpleNamespace

import torch
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

from megatron.bridge.models.shensi.modeling_shensi import (
    ShensiMoELayer,
    attach_erc_loss,
    erc_loss_func,
)


def _reset_autoscaler() -> None:
    MoEAuxLossAutoScaler.main_loss_backward_scale = None


def test_erc_loss_func_reaches_router_and_expert_weights() -> None:
    torch.manual_seed(0)
    num_experts, hidden, latent, intermediate = 4, 6, 3, 5
    router = torch.randn(num_experts, hidden, requires_grad=True)
    down = torch.randn(latent, hidden, requires_grad=True)
    gate_up = torch.randn(num_experts, 2 * intermediate, latent, requires_grad=True)

    loss = erc_loss_func(router, down, gate_up, alpha=0.5)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()

    for tensor in (router, down, gate_up):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


def test_erc_loss_func_is_zero_for_an_identical_expert_pool() -> None:
    """With alpha=1 an all-identical pool has no cross-expert coupling to penalize."""
    router = torch.ones(1, 5).expand(4, 5).clone()
    down = torch.randn(3, 5)
    gate_up = torch.ones(1, 4, 3).expand(4, 4, 3).clone()
    loss = erc_loss_func(router, down, gate_up, alpha=1.0)
    assert float(loss) == 0.0


def test_attach_erc_loss_injects_the_scaled_gradient() -> None:
    _reset_autoscaler()
    weight = torch.tensor(1.0, requires_grad=True)
    hidden = torch.zeros(2, 3, 4, requires_grad=True)
    layer = SimpleNamespace(
        config=SimpleNamespace(calculate_per_token_loss=False),
        compute_erc_loss=lambda: weight * 2.0,
    )

    output, bias = attach_erc_loss(layer, (hidden, None))
    assert bias is None
    output.sum().backward()

    # The autoscaler's default backward scale is 1, so the ERC gradient is 2.
    torch.testing.assert_close(weight.grad, torch.tensor(2.0))
    _reset_autoscaler()


def test_attach_erc_loss_matches_per_token_loss_scaling() -> None:
    _reset_autoscaler()
    weight = torch.tensor(1.0, requires_grad=True)
    hidden = torch.zeros(1, 3, 4, requires_grad=True)
    layer = SimpleNamespace(
        config=SimpleNamespace(calculate_per_token_loss=True),
        compute_erc_loss=lambda: weight * 2.0,
        _erc_token_scale=lambda _hidden: torch.tensor(4.0),
    )

    output = attach_erc_loss(layer, hidden)
    output.sum().backward()
    torch.testing.assert_close(weight.grad, torch.tensor(8.0))
    _reset_autoscaler()


def test_erc_owner_gate_requires_training_owner_and_coefficient() -> None:
    layer = SimpleNamespace(erc_num_groups=2, erc_loss_coef=1.0, training=True)
    assert ShensiMoELayer._should_compute_erc(layer)
    assert not ShensiMoELayer._should_compute_erc(SimpleNamespace(erc_num_groups=0, erc_loss_coef=1.0, training=True))
    assert not ShensiMoELayer._should_compute_erc(SimpleNamespace(erc_num_groups=2, erc_loss_coef=0.0, training=True))
    assert not ShensiMoELayer._should_compute_erc(SimpleNamespace(erc_num_groups=2, erc_loss_coef=1.0, training=False))
    with torch.no_grad():
        assert not ShensiMoELayer._should_compute_erc(
            SimpleNamespace(erc_num_groups=2, erc_loss_coef=1.0, training=True)
        )


def test_tie_mlp_scores_each_tied_pool_once() -> None:
    """Only the pool owner carries the global group count, and siblings share its modules."""

    def _mlp():
        return SimpleNamespace(router=object(), experts=object(), deepemb=None)

    layers = [
        SimpleNamespace(
            layer_idx=i,
            layer_number=i + 1,
            is_dense=(i == 0),
            is_write=(i % 2 == 0),
            mlp=_mlp(),
            self_attention=SimpleNamespace(),
            rosa=None,
        )
        for i in range(4)
    ]
    model = SimpleNamespace(
        decoder=SimpleNamespace(layers=layers),
        config=SimpleNamespace(
            attn_res_block_layer_types=[
                "block_write_layer",
                "block_read_layer",
                "block_write_layer",
                "block_read_layer",
            ],
            mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
            rosa_embed_dim=None,
            hidden_size=16,
        ),
        embedding=None,
    )

    from megatron.bridge.models.shensi.modeling_shensi import ShensiModel

    ShensiModel.tie_mlp(model)

    # Layers 1 and 2 belong to block 0, layer 3 to block 2: two tied pools in total.
    assert model.moe_group_owners == {0: layers[1].mlp, 2: layers[2].mlp}
    assert layers[1].mlp.erc_num_groups == 2
    assert layers[2].mlp.erc_num_groups == 2
    assert layers[3].mlp.router is layers[2].mlp.router
    assert layers[3].mlp.experts is layers[2].mlp.experts
    assert not hasattr(layers[3].mlp, "erc_num_groups")
