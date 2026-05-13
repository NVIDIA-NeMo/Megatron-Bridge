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

"""Unit tests for the DeepSeek-V4 bridge mapping registry.

Locks in the MTP mapping layout: per-MTP-layer HC head, separate ``e_proj``
and ``h_proj`` mappings, and no deprecated concatenated ``eh_proj`` path.
"""

from types import SimpleNamespace

import pytest

from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import DeepSeekV4Bridge


@pytest.fixture
def bridge_with_mtp():
    """A DSv4 bridge with hf_config stubbed for a single MTP layer."""
    bridge = DeepSeekV4Bridge()
    # mapping_registry only reads num_nextn_predict_layers from hf_config.
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=1)
    return bridge


@pytest.fixture
def bridge_without_mtp():
    """A DSv4 bridge with hf_config that has zero MTP layers."""
    bridge = DeepSeekV4Bridge()
    bridge.hf_config = SimpleNamespace(num_nextn_predict_layers=0)
    return bridge


def _by_megatron(registry):
    """Index mappings by megatron_param for quick lookup in assertions."""
    return {m.megatron_param: m for m in registry.mappings}


class TestDecoderHCHeadMappings:
    """The global decoder HC-head triplet must be replicated mappings."""

    @pytest.mark.parametrize(
        "name",
        ["decoder.hc_head_fn", "decoder.hc_head_base", "decoder.hc_head_scale"],
    )
    def test_decoder_hc_head_replicated(self, bridge_with_mtp, name):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(name)
        assert mapping is not None, f"missing decoder HC-head mapping: {name}"
        assert isinstance(mapping, ReplicatedMapping)
        # HF side drops the 'decoder.' prefix.
        assert mapping.hf_param == name.removeprefix("decoder.")


class TestMTPHCHeadMappings:
    """Per-MTP-layer HC head must mirror the decoder pattern."""

    @pytest.mark.parametrize(
        "suffix",
        ["hc_head_fn", "hc_head_base", "hc_head_scale"],
    )
    def test_mtp_hc_head_replicated(self, bridge_with_mtp, suffix):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(f"mtp.layers.0.{suffix}")
        assert mapping is not None, f"missing MTP HC-head mapping: mtp.layers.0.{suffix}"
        assert isinstance(mapping, ReplicatedMapping)
        assert mapping.hf_param == f"mtp.0.{suffix}"

    def test_mtp_hc_head_absent_when_no_mtp(self, bridge_without_mtp):
        registry = bridge_without_mtp.mapping_registry()
        names = _by_megatron(registry)
        for suffix in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
            assert f"mtp.layers.0.{suffix}" not in names


class TestMTPEHProjSplit:
    """MTP e_proj and h_proj are separate ColumnParallelLinear modules.

    The bridge must use two AutoMappings (which auto-detect column parallelism),
    not the deprecated concatenated eh_proj path.
    """

    @pytest.mark.parametrize("name", ["e_proj", "h_proj"])
    def test_split_proj_automapping(self, bridge_with_mtp, name):
        registry = bridge_with_mtp.mapping_registry()
        mapping = _by_megatron(registry).get(f"mtp.layers.0.{name}.weight")
        assert mapping is not None, f"missing MTP projection: {name}"
        assert isinstance(mapping, AutoMapping)
        assert mapping.hf_param == f"mtp.0.{name}.weight"

    def test_eh_proj_not_in_registry(self, bridge_with_mtp):
        registry = bridge_with_mtp.mapping_registry()
        for mapping in registry.mappings:
            assert "eh_proj" not in mapping.megatron_param, (
                f"deprecated eh_proj reference found in megatron_param: {mapping.megatron_param}"
            )
            hf_param = mapping.hf_param
            if isinstance(hf_param, str):
                assert "eh_proj" not in hf_param, f"deprecated eh_proj reference found in hf_param: {hf_param}"
            elif isinstance(hf_param, dict):
                for v in hf_param.values():
                    assert "eh_proj" not in v, f"deprecated eh_proj reference found in hf_param dict value: {v}"
