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

from unittest.mock import Mock

import pytest

from megatron.bridge.models.conversion.utils import mcore_to_hf_window_size, unwrap_model


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


def test_unwrap_model_ignores_non_type_module_instances():
    class Wrapper:
        def __init__(self, module):
            self.module = module

    model = object()
    wrapper = Wrapper(model)

    assert unwrap_model(wrapper, module_instances=(Wrapper, Mock())) is model


@pytest.mark.parametrize("adapter_layout", ["versioned", "legacy"])
def test_unwrap_model_default_supports_fsdp_adapter_layout(monkeypatch, adapter_layout):
    import sys
    from types import ModuleType

    class Wrapper:
        def __init__(self, module):
            self.module = module

    class OtherWrapper(Wrapper):
        pass

    class DistributedDataParallel(Wrapper):
        pass

    class TorchFullyShardedDataParallel(Wrapper):
        pass

    class MegatronFSDP(Wrapper):
        pass

    class Float16Module(Wrapper):
        pass

    adapter = ModuleType("megatron.core.distributed.fsdp.mcore_fsdp_adapter")
    if adapter_layout == "versioned":
        adapter.FullyShardedDataParallelV1 = Wrapper
        adapter.FullyShardedDataParallelV2 = OtherWrapper
        adapter.FullyShardedDataParallel = Mock()
    else:
        adapter.FullyShardedDataParallel = Wrapper

    distributed = ModuleType("megatron.core.distributed")
    distributed.DistributedDataParallel = DistributedDataParallel
    distributed.TorchFullyShardedDataParallel = TorchFullyShardedDataParallel
    fsdp = ModuleType("megatron.core.distributed.fsdp")
    fsdp.mcore_fsdp_adapter = adapter
    raw_fsdp = ModuleType("megatron.core.distributed.fsdp.src.megatron_fsdp")
    raw_fsdp.MegatronFSDP = MegatronFSDP
    transformer_module = ModuleType("megatron.core.transformer.module")
    transformer_module.Float16Module = Float16Module

    monkeypatch.setitem(sys.modules, "megatron.core.distributed", distributed)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed.fsdp", fsdp)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed.fsdp.mcore_fsdp_adapter", adapter)
    monkeypatch.setitem(sys.modules, "megatron.core.distributed.fsdp.src.megatron_fsdp", raw_fsdp)
    monkeypatch.setitem(sys.modules, "megatron.core.transformer.module", transformer_module)

    model = object()

    assert unwrap_model(Wrapper(MegatronFSDP(model))) is model
