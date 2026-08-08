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

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from megatron.core.transformer.moe.moe_layer import MoELayer

from megatron.bridge.models.kimi_vl.kimi_k25_vl_spec import (
    KimiK25VLChunkedMoELayer,
    _replace_moe_builder,
    build_kimi_k25_vl_spec,
)


def _make_layer(*, chunks: int, training: bool) -> KimiK25VLChunkedMoELayer:
    layer = object.__new__(KimiK25VLChunkedMoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(legacy_prefix_moe_chunks=chunks, transformer_impl="transformer_engine")
    layer.train(training)
    return layer


@pytest.mark.unit
def test_legacy_prefix_eval_bounds_each_moe_forward() -> None:
    """Full-prefix eval must release each expert activation chunk before the next."""
    layer = _make_layer(chunks=4, training=False)
    hidden_states = torch.arange(8 * 2 * 3, dtype=torch.float32).view(8, 2, 3)
    forwarded_sequence_lengths: list[int] = []

    def fake_forward(self, chunk, intermediate_tensors=None, padding_mask=None):
        forwarded_sequence_lengths.append(chunk.shape[0])
        return chunk + 1, None

    with patch.object(MoELayer, "forward", new=fake_forward):
        output, bias = layer(hidden_states)

    assert forwarded_sequence_lengths == [2, 2, 2, 2]
    assert torch.equal(output, hidden_states + 1)
    assert bias is None


@pytest.mark.unit
def test_training_keeps_single_moe_forward() -> None:
    """The Bridge inference contract must not alter the training path."""
    layer = _make_layer(chunks=4, training=True)
    hidden_states = torch.zeros(8, 2, 3)
    forwarded_sequence_lengths: list[int] = []

    def fake_forward(self, chunk, intermediate_tensors=None, padding_mask=None):
        forwarded_sequence_lengths.append(chunk.shape[0])
        return chunk, None

    with patch.object(MoELayer, "forward", new=fake_forward):
        layer(hidden_states)

    assert forwarded_sequence_lengths == [8]


@pytest.mark.unit
def test_kimi_spec_replaces_only_stock_moe_builder() -> None:
    """The custom layer must preserve the MCore builder arguments and dense layers."""
    sentinel = object()
    moe_builder = partial(MoELayer, submodules=sentinel)
    dense_builder = partial(torch.nn.Linear, 2, 2)

    replaced = _replace_moe_builder(moe_builder)

    assert replaced.func is KimiK25VLChunkedMoELayer
    assert replaced.keywords == {"submodules": sentinel}
    assert _replace_moe_builder(dense_builder) is dense_builder


@pytest.mark.unit
def test_kimi_block_spec_preserves_dense_layers_and_replaces_moe_layers() -> None:
    dense_layer_spec = SimpleNamespace(submodules=SimpleNamespace(mlp=partial(torch.nn.Linear, 2, 2)))
    moe_layer_spec = SimpleNamespace(submodules=SimpleNamespace(mlp=partial(MoELayer, submodules="sentinel")))
    original_moe_builder = moe_layer_spec.submodules.mlp
    block_spec = SimpleNamespace(layer_specs=[dense_layer_spec, moe_layer_spec])
    config = SimpleNamespace()

    with patch(
        "megatron.bridge.models.kimi_vl.kimi_k25_vl_spec.get_gpt_decoder_block_spec",
        return_value=block_spec,
    ) as get_block_spec:
        result = build_kimi_k25_vl_spec(config, vp_stage=2, use_transformer_engine=True)

    get_block_spec.assert_called_once_with(config, use_transformer_engine=True, vp_stage=2)
    assert result.layer_specs[0].submodules.mlp.func is torch.nn.Linear
    assert result.layer_specs[1].submodules.mlp.func is KimiK25VLChunkedMoELayer
    assert result.layer_specs[1].submodules.mlp.keywords == {"submodules": "sentinel"}
    assert original_moe_builder.func is MoELayer
