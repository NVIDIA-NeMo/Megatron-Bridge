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
import torch

from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.param_mapping import AutoMapping, ReplicatedMapping
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    DeepSeekV4Bridge,
    _dsv4_compress_ratios,
    _dsv4_num_hash_layers,
)


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


def _dummy_task():
    return SimpleNamespace(param_name="", global_param_name="", mapping=None)


class TestNativeDeepSeekV4ConfigTranslation:
    """Native Transformers DSv4 config fields must map back to MCore fields."""

    def test_compress_ratios_from_native_layer_types(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            layer_types=[
                "sliding_attention",
                "sliding_attention",
                "compressed_sparse_attention",
                "heavily_compressed_attention",
            ],
            compress_rates={
                "compressed_sparse_attention": 4,
                "heavily_compressed_attention": 128,
            },
        )

        assert _dsv4_compress_ratios(hf_config) == [0, 0, 4, 128, 0]

    def test_legacy_compress_ratios_still_work(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=4,
            num_nextn_predict_layers=1,
            compress_ratios=[0, 0, 4, 128, 0],
        )

        assert _dsv4_compress_ratios(hf_config) == [0, 0, 4, 128, 0]

    def test_hash_layers_from_native_mlp_layer_types(self):
        hf_config = SimpleNamespace(
            mlp_layer_types=["hash_moe", "hash_moe", "hash_moe", "moe", "moe"],
        )

        assert _dsv4_num_hash_layers(hf_config) == 3

    def test_hash_layers_must_be_prefix(self):
        hf_config = SimpleNamespace(mlp_layer_types=["hash_moe", "moe", "hash_moe"])

        with pytest.raises(ValueError, match="contiguous prefix"):
            _dsv4_num_hash_layers(hf_config)


class TestDeepSeekV4QuantizedExport:
    """DSv4 export must regenerate quantized weights and scale tensors."""

    def test_export_quantizes_fp8_weight_and_emits_scale(self):
        bridge = DeepSeekV4Bridge()
        hf_param = "layers.0.attn.wq_a.weight"
        scale_key = "layers.0.attn.wq_a.scale"
        weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
        source_state = {scale_key: torch.ones((1, 1), dtype=torch.float32)}

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {hf_param: weight}, source_state)

        assert set(result) == {hf_param, scale_key}
        assert result[hf_param].dtype == torch.float8_e4m3fn
        assert result[scale_key].shape == source_state[scale_key].shape
        assert result[scale_key].dtype == source_state[scale_key].dtype

        restored = bridge.maybe_modify_loaded_hf_weight(hf_param, result)
        assert restored.dtype == torch.bfloat16
        assert torch.allclose(restored.float(), weight.float())

    def test_export_preserves_e8m0_scale_dtype(self):
        e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
        if e8m0_dtype is None:
            pytest.skip("torch.float8_e8m0fnu is unavailable")
        try:
            source_scale = torch.ones((1, 1), dtype=e8m0_dtype)
        except RuntimeError as exc:
            pytest.skip(f"torch.float8_e8m0fnu tensor creation is unavailable: {exc}")

        bridge = DeepSeekV4Bridge()
        hf_param = "layers.0.attn.wq_a.weight"
        scale_key = "layers.0.attn.wq_a.scale"
        weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {hf_param: weight}, {scale_key: source_scale})

        assert result[hf_param].dtype == torch.float8_e4m3fn
        assert result[scale_key].dtype == e8m0_dtype
        restored = bridge.maybe_modify_loaded_hf_weight(hf_param, result)
        assert torch.allclose(restored.float(), weight.float())

    def test_export_quantizes_routed_expert_to_mxfp4_and_emits_scale(self):
        bridge = DeepSeekV4Bridge()
        hf_param = "layers.0.ffn.experts.0.w1.weight"
        scale_key = "layers.0.ffn.experts.0.w1.scale"
        values = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
        ).repeat(2)
        weight = values.reshape(1, 32).to(torch.bfloat16)
        source_state = {scale_key: torch.ones((1, 1), dtype=torch.float32)}

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {hf_param: weight}, source_state)

        assert set(result) == {hf_param, scale_key}
        assert result[hf_param].dtype == torch.int8
        assert result[hf_param].shape == (1, 16)
        assert result[scale_key].shape == source_state[scale_key].shape
        assert result[scale_key].dtype == source_state[scale_key].dtype

        restored = quantization_utils.dequantize_mxfp4_e2m1_packed(result[hf_param], result[scale_key])
        assert torch.equal(restored.float(), weight.float())

    @pytest.mark.parametrize(
        "hf_param",
        [
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.experts.0.w1.weight",
        ],
    )
    def test_export_uses_fp8_for_non_mxfp4_expert_scale_geometry(self, hf_param):
        bridge = DeepSeekV4Bridge()
        scale_key = hf_param.removesuffix(".weight") + ".scale"
        weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(
            _dummy_task(), {hf_param: weight}, {scale_key: torch.ones(1, 1)}
        )

        assert result[hf_param].dtype == torch.float8_e4m3fn
        assert result[scale_key].shape == (1, 1)

    def test_export_leaves_unscaled_weight_unchanged(self):
        bridge = DeepSeekV4Bridge()
        weight = torch.ones(4, 4, dtype=torch.bfloat16)

        result = bridge.maybe_modify_converted_hf_weight(_dummy_task(), {"norm.weight": weight}, {})

        assert set(result) == {"norm.weight"}
        assert result["norm.weight"] is weight


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
