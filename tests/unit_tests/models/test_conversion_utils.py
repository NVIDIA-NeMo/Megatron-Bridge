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

import pytest
import torch

from megatron.bridge.models.conversion import utils as conversion_utils
from megatron.bridge.models.conversion.utils import mcore_to_hf_window_size


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (None, None),
        (2048, 2048),
        ((2047, 0), 2048),
        ([2047, 0], 2048),
    ],
)
def test_mcore_to_hf_window_size(window_size, expected):
    assert mcore_to_hf_window_size(window_size) == expected


def test_mcore_to_hf_window_size_rejects_malformed_pair():
    with pytest.raises(ValueError, match="two-element MCore window"):
        mcore_to_hf_window_size([2047])


def test_unwrap_model_supports_mcore_fsdp_factory(monkeypatch):
    from megatron.core.distributed.fsdp import mcore_fsdp_adapter

    class FakeFullyShardedDataParallelV1(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    class FakeFullyShardedDataParallelV2(FakeFullyShardedDataParallelV1):
        pass

    def fsdp_factory(*args, **kwargs):
        return FakeFullyShardedDataParallelV2(*args, **kwargs)

    monkeypatch.setattr(mcore_fsdp_adapter, "FullyShardedDataParallel", fsdp_factory)
    monkeypatch.setattr(
        mcore_fsdp_adapter,
        "FullyShardedDataParallelV1",
        FakeFullyShardedDataParallelV1,
        raising=False,
    )
    monkeypatch.setattr(
        mcore_fsdp_adapter,
        "FullyShardedDataParallelV2",
        FakeFullyShardedDataParallelV2,
        raising=False,
    )

    model = torch.nn.Linear(2, 2)
    wrapped_model = FakeFullyShardedDataParallelV2(model)

    assert conversion_utils.unwrap_model(wrapped_model) is model
