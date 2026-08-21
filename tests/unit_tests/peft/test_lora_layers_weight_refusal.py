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

"""Pins for LoRALinear.weight's grouped-adapter refusal.

Grouped expert adapters have 3D per-expert weights, no single 2D ``to_wrap.weight`` to merge
into, and no ``tp_group`` attribute — before the guard, the property died on an accidental
``AttributeError`` (or worse, a group-less local-shard merge). The refusal makes that loud and
intentional; the export-side materializer remains the merge path for grouped adapters.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.peft.utils import GroupedExpertLinearAdapter


def _wrapper_with_adapter(adapter) -> LoRALinear:
    wrapper = object.__new__(LoRALinear)  # skip AdapterWrapper.__init__ plumbing
    nn.Module.__init__(wrapper)
    wrapper.to_wrap = SimpleNamespace(weight=torch.zeros(4, 4))
    wrapper.adapter = adapter
    wrapper._adapter_enabled = True
    return wrapper


def test_weight_property_refuses_grouped_expert_adapters():
    adapter = Mock(spec=GroupedExpertLinearAdapter)
    adapter.base_linear_name = "decoder.layers.0.mlp.experts.linear_fc2"
    wrapper = _wrapper_with_adapter(adapter)

    with pytest.raises(NotImplementedError, match="grouped expert adapters"):
        _ = wrapper.weight


def test_weight_property_still_merges_plain_adapters():
    adapter = SimpleNamespace(
        linear_in=SimpleNamespace(weight=torch.zeros(2, 4)),
        linear_out=SimpleNamespace(weight=torch.zeros(4, 2)),
        alpha=4,
        dim=2,
        tp_group=None,
    )
    wrapper = _wrapper_with_adapter(adapter)

    with patch("megatron.bridge.peft.lora_layers.LoRAMerge") as mock_merge:
        mock_merge.return_value.merge.return_value = torch.ones(4, 4)
        merged = wrapper.weight

    mock_merge.return_value.merge.assert_called_once()
    assert torch.equal(merged, torch.ones(4, 4))
