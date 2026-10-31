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

"""TrainingConfig.cuda_memory_fraction leaves device memory for allocations outside PyTorch."""

import pytest
import torch

from megatron.bridge.training import initialize
from megatron.bridge.training.config import TrainingConfig


pytestmark = pytest.mark.unit


def _train_config(fraction):
    return TrainingConfig(micro_batch_size=1, global_batch_size=8, train_iters=10, cuda_memory_fraction=fraction)


@pytest.mark.parametrize("fraction", [None, 0.5, 1.0])
def test_valid_fractions_finalize(fraction):
    config = _train_config(fraction)
    config.finalize()
    assert config.cuda_memory_fraction == fraction


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_invalid_fractions_are_rejected(fraction):
    with pytest.raises(ValueError, match="cuda_memory_fraction"):
        _train_config(fraction).finalize()


def test_fraction_is_applied_to_the_current_device_after_distributed_init(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: calls.append((f, d)))

    initialize._apply_cuda_memory_fraction(0.9)

    assert calls == [(0.9, 3)]


@pytest.mark.parametrize(("fraction", "initialized"), [(None, True), (0.9, False)])
def test_fraction_is_left_alone_when_unset_or_before_distributed_init(monkeypatch, fraction, initialized):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, d: calls.append((f, d)))

    initialize._apply_cuda_memory_fraction(fraction)

    assert calls == []
