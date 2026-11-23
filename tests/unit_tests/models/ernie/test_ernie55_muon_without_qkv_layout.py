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

"""CPU tests of ERNIE 5.5's Muon tagging on a Megatron-Core build without per-parameter QKV layouts.

``QKVLayout`` is patched away, so these tests run on any Megatron-Core: the APE is still excluded from Muon,
no weight gets a ``qkv_layout``, one warning is logged per process, and building a layout raises.
"""

import logging
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.ernie.modeling_ernie55 import muon


class _Compressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ape = torch.nn.Parameter(torch.zeros(2, 4))


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.compressor = _Compressor()
        self.linear_fc1 = torch.nn.Linear(4, 8, bias=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            tensor_model_parallel_size=1, expert_tensor_parallel_size=1, gated_linear_unit=True
        )
        self.layer = _Layer()


@pytest.fixture
def no_qkv_layout(monkeypatch):
    monkeypatch.setattr(muon, "QKVLayout", None)
    monkeypatch.setattr(muon, "_warned_no_qkv_layout", False)


def test_tagging_excludes_the_ape_and_warns_once(no_qkv_layout, caplog):
    model = _Model()
    with caplog.at_level(logging.WARNING, logger=muon.__name__):
        muon.tag_ernie55_muon_params(model)
        muon.tag_ernie55_muon_params(model)
    warnings = [record for record in caplog.records if "no per-parameter QKV layouts" in record.getMessage()]
    assert len(warnings) == 1
    assert model.layer.compressor.ape.use_muon is False
    assert not any(hasattr(param, "qkv_layout") for param in model.parameters())


def test_building_a_layout_raises(no_qkv_layout):
    with pytest.raises(RuntimeError, match="QKVLayout"):
        muon.ernie55_muon_layout([2, 2])
