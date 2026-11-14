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

"""Unit tests for Inkling2 NVFP4/MXFP8 expert formats."""

import pytest
import torch

from megatron.bridge.models.inkling.inkling2_quantization import (
    align_precision_config,
    companion_keys,
    dequantize_mxfp8,
    dequantize_nvfp4,
    expert_format,
    quantize_mxfp8_like,
    quantize_nvfp4_like,
)


pytestmark = pytest.mark.unit

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def test_nvfp4_decodes_low_nibble_first_with_block_and_outer_scales():
    packed = torch.tensor([[0x31] + [0] * 7], dtype=torch.uint8)  # element 0 -> code 1, element 1 -> code 3
    scale = torch.tensor([[2.0]], dtype=torch.float8_e4m3fn)
    weight = dequantize_nvfp4(packed, scale, torch.tensor(6.0), torch.tensor([3.0]))
    assert weight.shape == (1, 16)
    assert weight[0, 0].item() == pytest.approx(E2M1[1] * 2.0 * 2.0)
    assert weight[0, 1].item() == pytest.approx(E2M1[3] * 2.0 * 2.0)


@pytest.mark.parametrize("amax", [1.0, 3.7, 448.0 * 6.0])
def test_nvfp4_bf16_round_trip_reproduces_codes(amax):
    torch.manual_seed(0)
    codes = torch.randint(0, 256, (8, 32), dtype=torch.uint8)
    scale = (torch.rand(8, 4) * 4 + 0.01).to(torch.float8_e4m3fn)
    denominator = torch.tensor([2688.0])
    weight = dequantize_nvfp4(codes, scale, torch.tensor(amax), denominator).to(torch.bfloat16)
    assert torch.equal(quantize_nvfp4_like(weight, scale, torch.tensor(amax), denominator), codes)


def test_nvfp4_preserves_negative_zero():
    codes = torch.tensor([[0x80, 0x08] + [0] * 6], dtype=torch.uint8)
    scale = torch.tensor([[1.0]], dtype=torch.float8_e4m3fn)
    weight = dequantize_nvfp4(codes, scale, torch.tensor(1.0), torch.tensor([1.0])).to(torch.bfloat16)
    assert torch.equal(quantize_nvfp4_like(weight, scale, torch.tensor(1.0), torch.tensor([1.0])), codes)


def test_nvfp4_rounds_midpoints_to_even_codes():
    scale = torch.ones(1, 1, dtype=torch.float8_e4m3fn)
    weight = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 9.0] + [0.0] * 8])
    packed = quantize_nvfp4_like(weight, scale, torch.tensor(1.0), torch.tensor([1.0]))
    codes = torch.stack((packed & 0xF, packed >> 4), -1).flatten(-2)[0, :8].tolist()
    assert codes == [0, 2, 2, 4, 4, 6, 6, 7]


def test_mxfp8_round_trip_is_exact_in_bf16():
    torch.manual_seed(1)
    values = (torch.randn(4, 64) * 40).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = torch.randint(100, 140, (4, 2), dtype=torch.uint8)
    weight = dequantize_mxfp8(values, scale)
    assert torch.equal(weight.to(torch.bfloat16).float(), weight)
    requantized = quantize_mxfp8_like(weight.to(torch.bfloat16), scale)
    assert torch.equal(requantized.view(torch.uint8), values.view(torch.uint8))


def test_expert_format_detection():
    name = "model.llm.layers.0.mlp.experts.w13_weight"
    nvfp4 = {name, f"{name}_scale", f"{name}_amax", f"{name}_outer_scale_denominator"}
    mxfp8 = {name, f"{name}_scale"}
    assert expert_format(name, nvfp4) == "nvfp4"
    assert expert_format(name, mxfp8) == "mxfp8"
    assert expert_format(name, {name}) is None
    assert companion_keys(name, "nvfp4") == (f"{name}_scale", f"{name}_amax", f"{name}_outer_scale_denominator")
    assert companion_keys(name, None) == ()


_QUERY = "model.llm.attn_res.output_query"
_EMBED = "model.llm.embed.weight"
_NVFP4 = "model.llm.layers.0.mlp.experts.w13_weight"
_MXFP8 = "model.llm.layers.1.mlp.experts.w2_weight"
_UNEMBED = "model.llm.unembed.weight"
_PLAIN = {"required_activation_quantization": None, "weight_block_shape": [], "weight_transform": "cast"}


def _precision_config() -> dict:
    """A miniature hf_quant_config.json with every Inkling2 group kind."""
    return {
        "model": {
            "parameter_groups": [
                {"compute_dtype": "float32", "output_dtype": "float32", "tensors": [_QUERY], **_PLAIN},
                {"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "tensors": [_EMBED], **_PLAIN},
                {
                    "compute_dtype": "nvfp4",
                    "output_dtype": "uint8",
                    "required_activation_quantization": {"format": "nvfp4_e2m1"},
                    "weight_block_shape": [1, 16],
                    "weight_scale_format": "amax_denominator",
                    "weight_transform": "nvfp4_nearest",
                    "tensors": [_NVFP4],
                },
                {
                    "compute_dtype": "mxfp8",
                    "output_dtype": "float8_e4m3fn",
                    "required_activation_quantization": {"format": "mxfp8_e4m3"},
                    "weight_block_shape": [32, 32],
                    "weight_transform": "mxfp8_nearest",
                    "tensors": [_MXFP8],
                },
                {
                    "compute_dtype": "bfloat16",
                    "fwd_output_dtype": "float32",
                    "output_dtype": "bfloat16",
                    "tensors": [_UNEMBED],
                    **_PLAIN,
                },
            ]
        },
        "mtp": {"parameter_groups": [{"compute_dtype": "bfloat16", "output_dtype": "bfloat16", "tensors": ["fc"]}]},
        "quant_method": "inkling2",
    }


def test_align_precision_config_lists_plain_experts_as_bf16():
    config = _precision_config()
    written = {_QUERY: "float32", _EMBED: "bfloat16", _NVFP4: "bfloat16", _MXFP8: "bfloat16", _UNEMBED: "bfloat16"}

    assert align_precision_config(config, written)

    groups = config["model"]["parameter_groups"]
    assert [group["compute_dtype"] for group in groups] == ["float32", "bfloat16", "bfloat16"]
    assert groups[0]["tensors"] == [_QUERY]
    assert groups[1] == {
        "compute_dtype": "bfloat16",
        "output_dtype": "bfloat16",
        "tensors": sorted([_EMBED, _NVFP4, _MXFP8]),
        **_PLAIN,
    }
    assert groups[2]["tensors"] == [_UNEMBED] and groups[2]["fwd_output_dtype"] == "float32"
    assert config["mtp"] == _precision_config()["mtp"]


def test_align_precision_config_keeps_a_quantized_export():
    config = _precision_config()
    written = {_QUERY: "float32", _EMBED: "bfloat16", _NVFP4: "uint8", _MXFP8: "float8_e4m3fn", _UNEMBED: "bfloat16"}

    assert not align_precision_config(config, written)
    assert config == _precision_config()


def test_align_precision_config_creates_missing_plain_groups():
    config = _precision_config()
    config["model"]["parameter_groups"].pop(0)
    written = {_QUERY: "float32", _EMBED: "bfloat16", _NVFP4: "float32", _MXFP8: "bfloat16", _UNEMBED: "bfloat16"}

    assert align_precision_config(config, written)

    by_dtype = {
        group["compute_dtype"]: group["tensors"]
        for group in config["model"]["parameter_groups"]
        if "fwd_output_dtype" not in group
    }
    assert by_dtype == {"bfloat16": sorted([_EMBED, _MXFP8]), "float32": [_NVFP4]}


def test_align_precision_config_rejects_fp16():
    config = _precision_config()
    with pytest.raises(ValueError, match="float16"):
        align_precision_config(config, {_NVFP4: "float16"})
