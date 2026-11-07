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

"""ROSA suffix predictions and the auxiliary next-token objective."""

from types import SimpleNamespace

import torch
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler

from megatron.bridge.models.shensi.modeling_shensi import ShensiROSALayer, rosa


def test_rosa_predicts_the_token_after_the_longest_previous_suffix() -> None:
    tokens = torch.tensor([[1, 2, 3, 4, 1, 2, 3]])
    predictions = rosa(tokens)
    assert predictions.tolist() == [[-1, -1, -1, -1, 2, 3, 4]]
    # The 1-D form returns a 1-D tensor.
    assert rosa(tokens[0]).tolist() == [-1, -1, -1, -1, 2, 3, 4]
    # Padded positions never predict.
    masked = rosa(tokens, attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1, 0]]))
    assert masked.tolist() == [[-1, -1, -1, -1, 2, 3, -1]]


def _make_layer(coef: float) -> ShensiROSALayer:
    config = SimpleNamespace(
        hidden_size=8,
        hc_mult=2,
        actual_vocab_size=16,
        vocab_size=16,
        rosa_embed_dim=8,
        rosa_layer_ids=[0],
        rosa_loss_coef=coef,
        calculate_per_token_loss=False,
    )
    layer = ShensiROSALayer(config)
    layer.train()
    return layer


def _branch_grads(coef: float) -> dict[str, float]:
    MoEAuxLossAutoScaler.main_loss_backward_scale = None
    torch.manual_seed(0)
    layer = _make_layer(coef)
    streams = torch.randn(4, 2, 2, 8)
    rosa_ids = torch.tensor([[1, 2, 3, -1], [2, 3, 4, -1]])
    labels = torch.tensor([[2, 3, 4, 5], [3, 4, 5, 6]])
    # Labels are passed explicitly: the layer reads them from the call site so the auxiliary
    # loss survives activation checkpointing (the context variable is reset by then).
    out = layer(streams, rosa_ids, labels)
    out.sum().backward()
    return {name: float(param.grad.abs().sum()) for name, param in layer.named_parameters() if param.grad is not None}


def test_rosa_auxiliary_loss_trains_embedding_value_and_gate() -> None:
    baseline = _branch_grads(0.0)
    with_aux = _branch_grads(1.0)
    for name in ("embedding.weight", "value_proj.weight", "gate_proj.weight", "gate_proj.bias"):
        assert baseline[name] > 0
        assert with_aux[name] > 0
        assert with_aux[name] != baseline[name], name
