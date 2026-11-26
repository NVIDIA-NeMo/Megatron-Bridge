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

"""Unit tests for the MiMo-V2.6-Flash bridge."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from megatron.bridge.models.conversion.model_bridge import WeightConversionTask
from megatron.bridge.models.conversion.quantization_utils import dequantize_fp8_blockwise, dequantize_mxfp4_e2m1_packed
from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_bridge import MiMoV2FlashBridge, MiMoV2FlashQKVMapping
from megatron.bridge.models.mimo_v2_flash.mimo_v26_bridge import (
    MiMoV26Bridge,
    _fuse_qkv,
    _FusedQKVMapping,
    _split_fused_qkv,
)
from tests.unit_tests.models.mimo_v2_flash.test_mimo_v2_flash_bridge import _make_mock_pretrained


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """Pack E2M1 codes [rows, cols] into uint8 [rows, cols/2], low nibble first."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _fp8_slices_with_block_max(rows: int, cols: int) -> torch.Tensor:
    """FP8 weight whose every 128-row block, restarting at each ``rows``-row slice, reaches the E4M3 max 448."""
    torch.manual_seed(0)
    weight = torch.empty(2 * rows, cols).uniform_(-400, 400)
    for row in (0, 128, rows, rows + 128):
        weight[row, 0] = 448.0
    return weight.to(torch.float8_e4m3fn)


QKV = "model.mtp.layers.0.self_attn."


class _State(dict):
    """A source state dict with a ``source`` attribute, as the bridge's lazy HF state has."""


class TestMXFP4Dequant:
    def test_mxfp4_applies_power_of_two_scale_per_32(self):
        codes = torch.full((1, 64), 2, dtype=torch.uint8)  # E2M1 code 2 is 1.0
        scale = torch.tensor([[127, 129]], dtype=torch.uint8)  # 2^0 and 2^2
        out = dequantize_mxfp4_e2m1_packed(_pack(codes), scale)
        assert out.shape == (1, 64)
        assert torch.equal(out[0, :32].float(), torch.full((32,), 1.0))
        assert torch.equal(out[0, 32:].float(), torch.full((32,), 4.0))

    def test_bridge_dequantizes_mxfp4_experts(self):
        key = "model.layers.1.mlp.experts.0.down_proj.weight"
        state = {
            key: _pack(torch.full((1, 32), 2, dtype=torch.uint8)),
            key + "_scale": torch.tensor([[128]], dtype=torch.uint8),  # 2^1
        }
        assert torch.equal(MiMoV26Bridge()._load_and_dequant(key, state).float(), torch.full((1, 32), 2.0))

    def test_bridge_leaves_bf16_weights_to_v2_flash(self):
        key = "model.layers.0.input_layernorm.weight"
        weight = torch.ones(8, dtype=torch.bfloat16)
        assert MiMoV26Bridge()._load_and_dequant(key, {key: weight}) is weight


class TestFusedQKV:
    CONFIG = dict(
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        v_head_dim=32,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        swa_v_head_dim=32,
    )

    @pytest.fixture
    def config(self):
        return SimpleNamespace(**self.CONFIG)

    def test_split_returns_heads_in_order_with_per_slice_scales(self, config):
        slices, rows_per_slice = 2, 2 * 64 + 64 + 32  # per slice: 2 q heads, 1 k head, 1 v head (224 rows)
        weight = torch.arange(slices * rows_per_slice, dtype=torch.float32)[:, None].expand(-1, 128).contiguous()
        weight = weight.to(torch.float8_e4m3fn)
        # 224 rows -> 2 scale rows per slice (the second block is ragged); the blocks restart at each slice.
        scale = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        q, k, v = _split_fused_qkv(weight, scale, config, swa=False)
        assert (q.shape, k.shape, v.shape) == ((256, 128), (128, 128), (64, 128))
        row_scale = torch.cat(
            [torch.full((128,), 1.0), torch.full((96,), 2.0), torch.full((128,), 3.0), torch.full((96,), 4.0)]
        )
        expected = (weight.float() * row_scale[:, None]).to(torch.bfloat16)
        c1 = rows_per_slice
        assert torch.equal(q, torch.cat([expected[:128], expected[c1 : c1 + 128]]))
        assert torch.equal(k, torch.cat([expected[128:192], expected[c1 + 128 : c1 + 192]]))
        assert torch.equal(v, torch.cat([expected[192:224], expected[c1 + 192 :]]))

    def test_fuse_is_the_inverse_of_split(self, config):
        weight = _fp8_slices_with_block_max(224, 128)
        scale = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        fused = _fuse_qkv(*_split_fused_qkv(weight, scale, config, swa=False), slices=2)
        expected = torch.cat(
            [dequantize_fp8_blockwise(weight[:224], scale[:2]), dequantize_fp8_blockwise(weight[224:], scale[2:])]
        )
        assert torch.equal(fused, expected)

    def test_rejects_unexpected_layout(self, config):
        with pytest.raises(ValueError, match="unexpected fused qkv_proj layout"):
            _split_fused_qkv(torch.zeros(100, 128).to(torch.float8_e4m3fn), torch.ones(2, 1), config, swa=False)


class TestMiMoV26Bridge:
    def test_provider_inherits_zero_dropout_and_mtp(self):
        provider = MiMoV26Bridge().provider_bridge(_make_mock_pretrained())
        assert provider.hidden_dropout == 0.0 and provider.attention_dropout == 0.0
        assert provider.mtp_num_layers == MiMoV2FlashBridge().provider_bridge(_make_mock_pretrained()).mtp_num_layers

    def test_v2_flash_provider_disables_dropout(self):
        provider = MiMoV2FlashBridge().provider_bridge(_make_mock_pretrained())
        assert provider.hidden_dropout == 0.0 and provider.attention_dropout == 0.0

    def test_registry_keeps_mtp_and_fuses_every_qkv_mapping(self):
        v2_params = {m.megatron_param for m in MiMoV2FlashBridge().mapping_registry().mappings}
        mappings = MiMoV26Bridge().mapping_registry().mappings
        assert {m.megatron_param for m in mappings} == v2_params
        assert any(m.megatron_param.startswith("mtp.") for m in mappings)
        qkv = [m for m in mappings if isinstance(m, MiMoV2FlashQKVMapping)]
        assert qkv and all(isinstance(m, _FusedQKVMapping) and m.allow_hf_name_mismatch for m in qkv)

    def test_attention_type_per_layer_and_mtp(self):
        bridge = MiMoV26Bridge()
        bridge.hf_config = SimpleNamespace(hybrid_layer_pattern=[0, 1])
        assert not bridge._is_swa("model.layers.0.self_attn.qkv_proj.weight")
        assert bridge._is_swa("model.layers.1.self_attn.qkv_proj.weight")
        assert bridge._is_swa("model.mtp.layers.0.self_attn.qkv_proj.weight")

    @staticmethod
    def _task(mapping=None, weight_dtype=None):
        return WeightConversionTask(
            param_name="p", global_param_name="p", mapping=mapping or Mock(), weight_dtype=weight_dtype
        )

    @staticmethod
    def _bridge(num_key_value_heads=2):
        bridge = MiMoV26Bridge()
        bridge.hf_config = SimpleNamespace(num_key_value_heads=num_key_value_heads)
        return bridge

    def test_export_requantizes_fused_qkv_slices_to_source_bytes(self):
        config = SimpleNamespace(**TestFusedQKV.CONFIG)
        weight = _fp8_slices_with_block_max(224, 128)
        scale = torch.tensor([[1.0], [2.0], [4.0], [8.0]])  # powers of two keep the BF16 round trip exact
        q, k, v = _split_fused_qkv(weight, scale, config, swa=False)
        mapping = _FusedQKVMapping(
            "mtp.layers.0.self_attention.linear_qkv.weight",
            q=QKV + "q_proj.weight",
            k=QKV + "k_proj.weight",
            v=QKV + "v_proj.weight",
        )
        converted = {QKV + "v_proj.weight": v, QKV + "q_proj.weight": q, QKV + "k_proj.weight": k}
        source = {QKV + "qkv_proj.weight": weight, QKV + "qkv_proj.weight_scale_inv": scale}
        out = self._bridge().maybe_modify_converted_hf_weight(self._task(mapping), converted, source)
        assert list(out) == [QKV + "qkv_proj.weight", QKV + "qkv_proj.weight_scale_inv"]
        assert out[QKV + "qkv_proj.weight"].dtype == torch.float8_e4m3fn
        assert torch.equal(out[QKV + "qkv_proj.weight"].view(torch.uint8), weight.view(torch.uint8))
        assert torch.equal(out[QKV + "qkv_proj.weight_scale_inv"], scale)

    def test_export_requantizes_mxfp4_experts_to_source_bytes(self):
        name = "model.layers.1.mlp.experts.0.down_proj.weight"
        torch.manual_seed(0)
        codes = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])[torch.randint(15, (4, 64))]
        codes[:, 0::32] = 7  # each 32-value block reaches 6.0, the E2M1 max, so the source scale is reproduced
        packed = _pack(codes.to(torch.uint8))
        scale = torch.randint(120, 135, (4, 2), dtype=torch.uint8)
        weight = dequantize_mxfp4_e2m1_packed(packed, scale)
        out = self._bridge().maybe_modify_converted_hf_weight(
            self._task(), {name: weight}, {name: packed, name + "_scale": scale}
        )
        assert out[name].dtype == torch.uint8 and torch.equal(out[name], packed)
        assert torch.equal(out[name + "_scale"], scale)

    def test_export_requantizes_other_fp8_weights_as_one_block_grid(self):
        name = "model.layers.0.mlp.down_proj.weight"
        weight = torch.empty(256, 128).uniform_(-400, 400)
        weight[0, 0] = weight[128, 0] = 448.0  # one 128-row grid over the whole tensor, no slices
        weight = weight.to(torch.float8_e4m3fn)
        scale = torch.tensor([[2.0], [4.0]])
        out = self._bridge().maybe_modify_converted_hf_weight(
            self._task(), {name: dequantize_fp8_blockwise(weight, scale)}, {name: weight, name + "_scale_inv": scale}
        )
        assert torch.equal(out[name].view(torch.uint8), weight.view(torch.uint8))
        assert torch.equal(out[name + "_scale_inv"], scale)

    def test_export_rejects_weight_dtype_for_quantized_weights(self):
        name = "model.layers.0.mlp.down_proj.weight"
        source = {name + "_scale_inv": torch.ones(1, 1)}
        with pytest.raises(ValueError, match="source quantized format"):
            self._bridge().maybe_modify_converted_hf_weight(
                self._task(weight_dtype=torch.bfloat16), {name: torch.ones(2, 2)}, source
            )

    def test_export_leaves_bf16_weights_and_empty_ranks(self):
        weights = {"model.norm.weight": torch.ones(2)}
        assert self._bridge().maybe_modify_converted_hf_weight(self._task(), weights, {}) == weights
        qkv = self._task(_FusedQKVMapping("x", q="q", k="k", v="v"))
        assert self._bridge().maybe_modify_converted_hf_weight(qkv, {}, {}) == {}

    def test_export_copies_vision_audio_and_speech_from_source(self, monkeypatch):
        monkeypatch.setattr(
            MiMoV2FlashBridge,
            "stream_weights_megatron_to_hf",
            lambda *args, **kwargs: iter([("model.norm.weight", 0)]),
        )
        keys = [
            "model.norm.weight",
            "visual.patch_embed.weight",
            "audio_encoder.projection.weight",
            "speech_embeddings.0.weight",
        ]
        state = _State({key: torch.full((1,), float(i)) for i, key in enumerate(keys)})
        state.source = SimpleNamespace(get_all_keys=lambda: keys)
        hf_pretrained = SimpleNamespace(state=state)
        out = list(MiMoV26Bridge().stream_weights_megatron_to_hf([], hf_pretrained, cpu=True))
        assert [name for name, _ in out] == keys
        assert all(torch.equal(tensor, state[name]) for name, tensor in out[1:])

    def test_export_restores_source_quantization_config(self, tmp_path):
        quantization_config = {"quant_method": "fp8", "store_dtype": "mxfp4", "weight_block_size": [128, 128]}
        source, saved = tmp_path / "source", tmp_path / "saved"
        source.mkdir()
        saved.mkdir()
        (source / "config.json").write_text(
            json.dumps({"model_type": "mimo_v2", "quantization_config": quantization_config})
        )
        (saved / "config.json").write_text(json.dumps({"model_type": "mimo_v2"}))
        bridge = MiMoV26Bridge()
        bridge.hf_config = SimpleNamespace(_name_or_path=str(source))
        bridge.postprocess_hf_export_artifacts(saved)
        assert json.loads((saved / "config.json").read_text()) == {
            "model_type": "mimo_v2",
            "quantization_config": quantization_config,
        }

    def test_export_config_is_v26(self):
        provider = MiMoV26Bridge().provider_bridge(_make_mock_pretrained())
        config = MiMoV26Bridge.megatron_to_hf_config(provider)
        assert config["model_type"] == "mimo_v2"
        assert config["attention_projection_layout"] == "fused_qkv"
        assert config["auto_map"]["AutoModelForCausalLM"] == "modeling_mimo_v2.MiMoV2ForCausalLM"
