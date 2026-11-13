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

"""CPU unit tests for the Inkling2 layers' process groups."""

from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.inkling.modeling_inkling2 import layers, ops


pytestmark = pytest.mark.unit


class _Group:
    """Process group stand-in of a given size."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


def test_moe_short_convolution_gathers_and_scatters_over_the_layer_tp_group(monkeypatch):
    calls = []

    def gather(tensor, tensor_parallel_output_grad=True, group=None):
        calls.append(("gather", group, tensor_parallel_output_grad))
        return tensor

    def scatter(tensor, group=None):
        calls.append(("scatter", group))
        return tensor

    monkeypatch.setattr(layers, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(layers, "scatter_to_sequence_parallel_region", scatter)
    tp_group = _Group(2)
    layer = SimpleNamespace(
        config=SimpleNamespace(sequence_parallel=True, inkling2_sconv_kernel_size=4, inkling2_compile_ops=False),
        pg_collection=SimpleNamespace(tp=tp_group, cp=_Group(1)),
        mlp_conv_weight=torch.randn(8, 1, 4),
    )
    output = torch.randn(6, 2, 8)

    result = layers.Inkling2MoELayer._mlp_conv(layer, output, None)

    # Megatron-Core would fall back to the global TP group without an explicit group.
    assert calls == [("gather", tp_group, False), ("scatter", tp_group)]
    positions = ops.document_positions(6, 2, output.device, None)
    torch.testing.assert_close(result, ops.static_short_conv(output, layer.mlp_conv_weight.squeeze(1), positions))
