# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the Megatron-FSDP hooks added to model conversion utilities.

Covers:
    * ``_module_uses_fsdp`` (param_mapping.py) — DTensor-based FSDP detection.
    * ``unwrap_model``     (conversion/utils.py) — MegatronFSDP-aware unwrapping.
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from megatron.bridge.models.conversion.param_mapping import _module_uses_fsdp
from megatron.bridge.models.conversion.utils import unwrap_model


class _FakeDTensor(torch.Tensor):
    """Stand-in class used with ``isinstance`` checks inside the code under test."""


class _Wrapper(nn.Module):
    """Generic wrapper module exposing ``.module`` like DDP/FSDP wrappers do."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.module = inner


class TestModuleUsesFsdp:
    """Behavior of ``_module_uses_fsdp``."""

    def test_returns_false_for_plain_module(self):
        module = nn.Linear(4, 4)
        assert _module_uses_fsdp(module) is False

    def test_returns_false_when_no_weight_prefixed_dtensor(self):
        module = nn.Linear(4, 4)
        module.register_parameter("bias_extra", nn.Parameter(torch.zeros(4)))
        assert _module_uses_fsdp(module) is False

    def test_returns_true_when_weight_param_is_dtensor(self):
        """Patch DTensor to ``_FakeDTensor`` so isinstance resolves against our sentinel."""
        module = nn.Linear(4, 4)
        fake_weight = nn.Parameter(_FakeDTensor(torch.zeros(4, 4)))
        module._parameters["weight"] = fake_weight
        with patch("megatron.bridge.models.conversion.param_mapping.DTensor", _FakeDTensor):
            assert _module_uses_fsdp(module) is True

    def test_ignores_non_weight_dtensor_params(self):
        module = nn.Linear(4, 4)
        fake_bias = nn.Parameter(_FakeDTensor(torch.zeros(4)))
        module._parameters["bias"] = fake_bias
        with patch("megatron.bridge.models.conversion.param_mapping.DTensor", _FakeDTensor):
            # Only ``weight``-prefixed params count; ``bias`` alone should yield False.
            assert _module_uses_fsdp(module) is False

    def test_accepts_weight_suffix_variants(self):
        module = nn.Linear(4, 4)
        # Simulate a quantization setup exposing ``weight_quant`` as DTensor.
        module._parameters["weight"] = nn.Parameter(torch.zeros(4, 4))  # regular
        module._parameters["weight_quant"] = nn.Parameter(_FakeDTensor(torch.zeros(4)))
        with patch("megatron.bridge.models.conversion.param_mapping.DTensor", _FakeDTensor):
            assert _module_uses_fsdp(module) is True


class TestUnwrapModel:
    """Behavior of the MegatronFSDP-aware ``unwrap_model``."""

    def test_unwraps_single_wrapper(self):
        inner = nn.Linear(4, 4)
        wrapped = _Wrapper(inner)
        result = unwrap_model(wrapped, module_instances=(_Wrapper,))
        assert result is inner

    def test_unwraps_nested_wrappers(self):
        inner = nn.Linear(4, 4)
        wrapped = _Wrapper(_Wrapper(inner))
        result = unwrap_model(wrapped, module_instances=(_Wrapper,))
        assert result is inner

    def test_list_in_list_out(self):
        a, b = nn.Linear(4, 4), nn.Linear(4, 4)
        result = unwrap_model([_Wrapper(a), _Wrapper(b)], module_instances=(_Wrapper,))
        assert isinstance(result, list)
        assert result == [a, b]

    def test_single_model_returns_single(self):
        inner = nn.Linear(4, 4)
        result = unwrap_model(_Wrapper(inner), module_instances=(_Wrapper,))
        assert not isinstance(result, list)
        assert result is inner

    def test_non_wrapped_model_returned_as_is(self):
        model = nn.Linear(4, 4)
        assert unwrap_model(model, module_instances=(_Wrapper,)) is model

    def test_default_module_instances_imports_megatron_fsdp(self):
        """When ``module_instances`` is None, the function resolves MegatronFSDP from mcore."""
        model = nn.Linear(4, 4)
        # We don't have mcore FSDP set up in unit tests, so the import path must
        # succeed but not actually unwrap anything when the model isn't a wrapper.
        try:
            result = unwrap_model(model)
        except ImportError as exc:
            pytest.skip(f"Megatron-Core FSDP imports unavailable in this env: {exc}")
        assert result is model
