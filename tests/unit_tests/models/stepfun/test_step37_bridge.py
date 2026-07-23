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

"""Unit tests for Step37Bridge (Step3.7) layer-spec wiring."""

from types import SimpleNamespace
from unittest.mock import patch

from megatron.bridge.models.stepfun import step37_bridge as _step37_bridge_mod
from megatron.bridge.models.stepfun.modelling_step37 import transformer_block as _step37_block_mod
from megatron.bridge.models.stepfun.modelling_step37.transformer_block import get_step37_text_layer_spec
from megatron.bridge.models.stepfun.step35_bridge import build_step35_layer_spec
from megatron.bridge.models.stepfun.step37_bridge import Step37Bridge


class _FakeProvider:
    """Stand-in for `Step37ModelProvider`; empty `__dataclass_fields__` filters
    away the CONFIG_MAPPING kwargs so no Megatron backend is needed."""

    __dataclass_fields__ = {}

    def __init__(self, **kwargs):
        self.num_layers = 4
        for key, value in kwargs.items():
            setattr(self, key, value)


def _make_hf_config(**text_overrides) -> SimpleNamespace:
    """Build a minimal Step3.7-shaped config: nested text_config + vision_config."""
    text = dict(
        num_hidden_layers=4,
        torch_dtype="bfloat16",
        moe_layers_enum="2,3",
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
    )
    text.update(text_overrides)
    return SimpleNamespace(
        text_config=SimpleNamespace(**text),
        vision_config=SimpleNamespace(),
    )


class TestStep37BridgeProviderBridge:
    """Verify the MoE-schedule branch of Step37Bridge.provider_bridge.

    `Step37ModelProvider` is patched out so the test runs without a real
    Megatron backend, mirroring the Step35Bridge unit tests.
    """

    def _run(self, **text_overrides):
        hf_config = _make_hf_config(**text_overrides)
        with patch.object(_step37_bridge_mod, "Step37ModelProvider", _FakeProvider):
            return Step37Bridge().provider_bridge(SimpleNamespace(config=hf_config))

    def test_moe_schedule_selects_public_layer_spec_builder(self):
        p = self._run()
        assert p.transformer_layer_spec is build_step35_layer_spec
        assert p.moe_layer_freq == [0, 0, 1, 1]

    def test_layer_spec_target_passes_checkpoint_load_validation(self):
        """The serialized spec target must be loadable by the checkpoint instantiate path."""
        from megatron.bridge.utils.instantiate_utils import _validate_target_prefix

        p = self._run()
        spec = p.transformer_layer_spec
        target = f"{spec.__module__}.{spec.__qualname__}"
        _validate_target_prefix(target=target, full_key="transformer_layer_spec")


class TestGetStep37TextLayerSpec:
    def test_delegates_to_step35_builder(self):
        """Step3.7's text decoder spec is Step-3.5's hybrid layer spec, passed through verbatim."""
        sentinel_cfg = object()
        sentinel_out = object()
        with patch.object(_step37_block_mod, "build_step35_layer_spec", return_value=sentinel_out) as builder:
            out = get_step37_text_layer_spec(sentinel_cfg, num_experts=None)

        builder.assert_called_once_with(sentinel_cfg, num_experts=None)
        assert out is sentinel_out

    def test_module_exposes_public_builder(self):
        assert _step37_block_mod.build_step35_layer_spec is build_step35_layer_spec
