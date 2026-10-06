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

import math

import pytest
import torch

from megatron.bridge.peft.utils import ParallelLinearAdapter


FAN_IN, FAN_OUT = 4096, 32


def _std_of(init_fn, shape, seed=17):
    torch.manual_seed(seed)
    t = torch.empty(*shape)
    init_fn(t)
    return t.std().item()


class TestGetInitFnFullFan:
    """_get_init_fn with full fan dims must be independent of the tensor shape.

    When the adapter weight is sharded across TP ranks, Megatron initializes
    each shard locally; a fan-dependent init computed on the shard makes the
    init distribution a function of TP size. Resolving xavier/kaiming from the
    full dims removes that dependence.
    """

    def test_xavier_full_fan_is_shape_independent(self):
        fn = ParallelLinearAdapter._get_init_fn(None, "xavier", fan_in=FAN_IN, fan_out=FAN_OUT)
        expected = math.sqrt(2.0 / (FAN_IN + FAN_OUT))
        full = _std_of(fn, (FAN_OUT, FAN_IN))
        shard_tp4 = _std_of(fn, (FAN_OUT, FAN_IN // 4))
        shard_tp8 = _std_of(fn, (FAN_OUT, FAN_IN // 8))
        for got in (full, shard_tp4, shard_tp8):
            assert got == pytest.approx(expected, rel=0.05)

    def test_xavier_without_fan_depends_on_shape(self):
        # Documents the legacy behavior the full-fan path exists to avoid.
        fn = ParallelLinearAdapter._get_init_fn(None, "xavier")
        full = _std_of(fn, (FAN_OUT, FAN_IN))
        shard_tp4 = _std_of(fn, (FAN_OUT, FAN_IN // 4))
        assert shard_tp4 / full == pytest.approx(math.sqrt((FAN_IN + FAN_OUT) / (FAN_IN / 4 + FAN_OUT)), rel=0.05)

    def test_xavier_full_fan_matches_tensor_derived_at_full_shape(self):
        # At TP=1 (full tensor) the resolved init must equal xavier_normal_:
        # same std, and bit-identical draws (both are one normal_ call).
        fixed = ParallelLinearAdapter._get_init_fn(None, "xavier", fan_in=FAN_IN, fan_out=FAN_OUT)
        legacy = ParallelLinearAdapter._get_init_fn(None, "xavier")
        torch.manual_seed(3)
        a = torch.empty(FAN_OUT, FAN_IN)
        fixed(a)
        torch.manual_seed(3)
        b = torch.empty(FAN_OUT, FAN_IN)
        legacy(b)
        assert torch.equal(a, b)

    def test_kaiming_full_fan_is_shape_independent(self):
        fn = ParallelLinearAdapter._get_init_fn(None, "kaiming", fan_in=FAN_IN)
        bound = 1.0 / math.sqrt(FAN_IN)
        for shape in ((FAN_OUT, FAN_IN), (FAN_OUT, FAN_IN // 4)):
            torch.manual_seed(17)
            t = torch.empty(*shape)
            fn(t)
            assert t.abs().max().item() <= bound
            assert t.std().item() == pytest.approx(bound / math.sqrt(3.0), rel=0.05)

    def test_kaiming_full_fan_matches_tensor_derived_at_full_shape(self):
        fixed = ParallelLinearAdapter._get_init_fn(None, "kaiming", fan_in=FAN_IN)
        legacy = ParallelLinearAdapter._get_init_fn(None, "kaiming")
        torch.manual_seed(3)
        a = torch.empty(FAN_OUT, FAN_IN)
        fixed(a)
        torch.manual_seed(3)
        b = torch.empty(FAN_OUT, FAN_IN)
        legacy(b)
        assert torch.equal(a, b)

    def test_zero_and_normal_unaffected(self):
        zero = ParallelLinearAdapter._get_init_fn(None, "zero", fan_in=FAN_IN, fan_out=FAN_OUT)
        t = torch.rand(8, 8)
        zero(t)
        assert torch.equal(t, torch.zeros(8, 8))
        normal = ParallelLinearAdapter._get_init_fn(None, "normal", fan_in=FAN_IN, fan_out=FAN_OUT)
        assert _std_of(normal, (256, 256)) == pytest.approx(0.2, rel=0.05)
