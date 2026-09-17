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
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.gtp import _gather_gtp_weight, _get_mapping_shape, _slice_gtp_weight
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    DirectMapping,
    GatedMLPMapping,
    RowParallelMapping,
)


pytestmark = pytest.mark.unit


class _Bridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        return None

    def mapping_registry(self):
        return MegatronMappingRegistry()


def _shard(shape, *, rank=1, size=2, padding=0):
    weight = torch.nn.Parameter(torch.zeros(shape))
    weight.is_gtp_weight_remat = True
    weight.group = Mock()
    weight.group.rank.return_value = rank
    weight.group.size.return_value = size
    weight.pad_length = padding
    return weight


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("padding", [0, 1, 3])
def test_import_slices_rows_and_zero_pads(rank, padding):
    param = _shard((3, 4), rank=rank, padding=padding)
    logical = torch.arange((6 - padding) * 4).reshape(6 - padding, 4).float()
    actual = _slice_gtp_weight(logical, param)
    expected = torch.cat((logical, torch.zeros(padding, 4)))[rank * 3 : (rank + 1) * 3]
    assert torch.equal(actual, expected)
    assert actual.is_contiguous()
    assert _get_mapping_shape(param) == logical.shape


def test_import_rejects_wrong_logical_shape():
    with pytest.raises(ValueError, match="TP-local GTP weight shape"):
        _slice_gtp_weight(torch.ones(3, 4), _shard((3, 4)))


def test_non_gtp_conversion_keeps_tensor_identity():
    param = torch.nn.Parameter(torch.ones(3, 4))
    source = torch.zeros_like(param)
    assert _slice_gtp_weight(source, param) is source
    assert _gather_gtp_weight(param) is param
    assert _gather_gtp_weight(None) is None
    assert _get_mapping_shape(param) == param.shape


def test_local_hf_views_reject_unrepresentable_gtp_sharding():
    param = _shard((3, 4))
    param.data = param.data.bfloat16()
    with pytest.raises(ValueError, match="cannot represent GTP sharding"):
        DirectMapping("weight", "hf.weight").local_hf_params(
            param, global_param_name="weight", megatron_module=torch.nn.Module()
        )


def test_export_gathers_rows_and_trims_padding(monkeypatch):
    param = _shard((3, 4), padding=1)
    logical_padded = torch.arange(24).reshape(6, 4).float()

    def gather(output, local, *, group):
        assert group is param.group
        assert type(local) is torch.Tensor
        assert local.shape == param.shape
        output.copy_(logical_padded)

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    assert torch.equal(_gather_gtp_weight(param), logical_padded[:5])


@pytest.mark.parametrize("mapping_type", [ColumnParallelMapping, RowParallelMapping, GatedMLPMapping])
def test_tp_scatter_uses_shape_before_gtp_sharding(mapping_type, monkeypatch):
    monkeypatch.setattr("megatron.bridge.models.conversion.param_mapping.get_pg_size", lambda group: group.size())
    monkeypatch.setattr("megatron.bridge.models.conversion.param_mapping.get_pg_rank", lambda group: group.rank())
    module = torch.nn.Module()
    module.weight = _shard((3, 4), padding=0)
    if mapping_type is GatedMLPMapping:
        mapping = mapping_type("weight", gate="gate", up="up")
        source = {"gate": torch.ones(6, 4), "up": torch.full((6, 4), 2.0)}
    else:
        mapping = mapping_type("weight", "hf.weight")
        source = torch.ones(12, 4) if mapping_type is ColumnParallelMapping else torch.ones(6, 8)
    group = Mock()
    group.size.return_value = 2
    group.rank.return_value = 0
    mapping._tp_group = group

    def scatter(splits, shape, dtype, device):
        assert shape == torch.Size((6, 4))
        assert splits[0].shape == shape
        return splits[0]

    mapping.scatter_to_tp_ranks = scatter
    result = mapping.hf_to_megatron(source, module)
    assert result.shape == (6, 4)


@pytest.mark.parametrize("streaming", [False, True])
def test_hf_import_loads_local_gtp_shard(monkeypatch, streaming):
    bridge = _Bridge()
    module = torch.nn.Module()
    module.weight = _shard((3, 4), rank=1, padding=1)
    source = torch.arange(20).reshape(5, 4).float()
    mapping = Mock()
    mapping.hf_param = "hf.weight"
    mapping.is_grouped_export = False
    mapping.hf_to_megatron.return_value = source
    task = WeightConversionTask(
        param_name="weight",
        global_param_name="weight",
        mapping=mapping,
        megatron_module=module,
        param_weight=module.weight,
        vp_stage=0,
    )
    hf = SimpleNamespace(state={"hf.weight": source}, model_name_or_path="toy")
    monkeypatch.setattr(bridge, "build_conversion_tasks", lambda *args: [task])
    monkeypatch.setattr(bridge, "finalize_hf_import", lambda *args: None)
    if streaming:
        [result] = bridge.stream_weights_hf_to_megatron(hf, [module], [task])
        actual = result.weight
    else:
        bridge.load_weights_hf_to_megatron(hf, [module])
        actual = module.weight
    expected = torch.cat((source[3:], torch.zeros(1, 4)))
    assert torch.equal(actual, expected)
