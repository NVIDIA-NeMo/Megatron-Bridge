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

from megatron.bridge.models.conversion.quantization_utils import (
    dequantize_fp8_blockwise,
    dequantize_fp8_e4m3fn_with_scale,
    dequantize_int4,
    dequantize_mxfp4,
    dequantize_mxfp4_e2m1_packed,
    is_mxfp4_e2m1_scale_geometry,
    maybe_dequantize_fp8,
    maybe_dequantize_fp8_blockwise,
    maybe_dequantize_hf_quantized_weight,
    quantize_fp8_e4m3fn_like_scale,
    quantize_mxfp4_e2m1_like_scale,
    quantize_to_int4,
    requantize_hf_weight_scale_pairs,
)


def test_dequantize_fp8_blockwise_applies_distinct_scales():
    weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    result = dequantize_fp8_blockwise(weight, scale_inv).float()

    assert result.dtype == torch.float32
    assert torch.all(result[:128, :128] == 1.0)
    assert torch.all(result[:128, 128:] == 2.0)
    assert torch.all(result[128:, :128] == 3.0)
    assert torch.all(result[128:, 128:] == 4.0)


def test_maybe_dequantize_fp8_blockwise_passthrough_and_fallback_cast():
    bf16_weight = torch.ones(4, 4, dtype=torch.bfloat16)
    assert maybe_dequantize_fp8_blockwise(bf16_weight) is bf16_weight

    fp8_weight = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
    result = maybe_dequantize_fp8_blockwise(fp8_weight)

    assert result.dtype == torch.bfloat16
    assert torch.all(result == 1.0)


def test_maybe_dequantize_fp8_applies_broadcastable_scale():
    fp8_weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    scale_inv = torch.tensor([2.0])

    result = maybe_dequantize_fp8(fp8_weight, scale_inv)

    assert result.dtype == torch.bfloat16
    assert torch.all(result == 2.0)


def test_dequantize_mxfp4_uses_low_then_high_nibbles():
    blocks = torch.tensor([[[0x21]]], dtype=torch.uint8)
    scales = torch.tensor([[127]], dtype=torch.uint8)

    result = dequantize_mxfp4(blocks, scales, dtype=torch.float32)

    assert result.shape == (1, 2)
    assert torch.equal(result, torch.tensor([[0.5, 1.0]]))


def test_quantize_fp8_e4m3fn_like_scale_roundtrips_scaled_weight():
    weight = torch.full((4, 4), 2.0, dtype=torch.bfloat16)
    source_scale = torch.ones((1, 1), dtype=torch.float32)

    q_weight, q_scale = quantize_fp8_e4m3fn_like_scale(weight, source_scale)
    result = dequantize_fp8_e4m3fn_with_scale(q_weight, q_scale)

    assert q_weight.dtype == torch.float8_e4m3fn
    assert q_scale.shape == source_scale.shape
    assert q_scale.dtype == source_scale.dtype
    assert torch.allclose(result.float(), weight.float())


@pytest.mark.parametrize(
    ("weight", "source_scale", "block_size"),
    [
        (
            torch.tensor([[1.0, -1.0], [2.0, -2.0]], dtype=torch.bfloat16),
            torch.ones(2),
            128,
        ),
        (
            torch.tensor(
                [
                    [2.0, -2.0, 2.0, -2.0],
                    [2.0, -2.0, 2.0, -2.0],
                    [4.0, -4.0, 4.0, -4.0],
                ],
                dtype=torch.bfloat16,
            ),
            torch.ones(2),
            2,
        ),
        (
            torch.tensor([[1.0, -1.0, 2.0, -2.0], [3.0, -3.0, 4.0, -4.0]], dtype=torch.bfloat16),
            torch.ones(2, 2),
            128,
        ),
    ],
)
def test_quantize_fp8_e4m3fn_like_scale_roundtrips_supported_scale_layouts(weight, source_scale, block_size):
    q_weight, q_scale = quantize_fp8_e4m3fn_like_scale(weight, source_scale, block_size=block_size)
    result = dequantize_fp8_e4m3fn_with_scale(q_weight, q_scale, block_size=block_size)

    assert q_weight.dtype == torch.float8_e4m3fn
    assert q_scale.shape == source_scale.shape
    assert q_scale.dtype == source_scale.dtype
    assert torch.allclose(result.float(), weight.float())


@pytest.mark.parametrize(
    ("weight", "source_scale", "message"),
    [
        (torch.ones(2, 2, 2), torch.ones(2), "expects a 2-D weight"),
        (torch.ones(2, 4), torch.ones(1, 1, 1), "Unsupported FP8 scale rank"),
        (torch.ones(3, 4), torch.ones(4), "Unsupported 1-D FP8 scale geometry"),
        (torch.ones(2, 5), torch.ones(2, 2), "Unsupported per-row FP8 scale geometry"),
        (torch.ones(4, 4), torch.ones(3, 1), "Unsupported FP8 scale geometry"),
    ],
)
def test_quantize_fp8_e4m3fn_like_scale_rejects_unsupported_geometry(weight, source_scale, message):
    with pytest.raises(RuntimeError, match=message):
        quantize_fp8_e4m3fn_like_scale(weight, source_scale, block_size=2)


def test_dequantize_fp8_e4m3fn_with_scale_rejects_unsupported_geometry():
    fp8_weight = torch.ones(2, 3, dtype=torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="Unsupported FP8 scale rank"):
        dequantize_fp8_e4m3fn_with_scale(fp8_weight, torch.ones(1, 1, 1))

    with pytest.raises(RuntimeError, match="FP8 dequant shape mismatch"):
        dequantize_fp8_e4m3fn_with_scale(fp8_weight, torch.ones(2, 2))


def test_quantize_dequantize_mxfp4_e2m1_packed_roundtrips_representable_values():
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
    source_scale = torch.ones((1, 1), dtype=torch.float32)

    packed, scale = quantize_mxfp4_e2m1_like_scale(weight, source_scale)
    result = dequantize_mxfp4_e2m1_packed(packed, scale)

    assert packed.dtype == torch.int8
    assert packed.shape == (1, 16)
    assert scale.shape == source_scale.shape
    assert torch.equal(result.float(), weight.float())


def test_mxfp4_scale_geometry_checks_logical_unpacked_shape():
    weight = torch.ones(2, 64)

    assert is_mxfp4_e2m1_scale_geometry(weight, torch.ones(2, 2))
    assert not is_mxfp4_e2m1_scale_geometry(weight, torch.ones(2, 1))
    assert not is_mxfp4_e2m1_scale_geometry(torch.ones(2, 2, 32), torch.ones(2, 2))


def test_mxfp4_helpers_reject_unsupported_geometry():
    with pytest.raises(RuntimeError, match="Unsupported MXFP4 scale geometry"):
        dequantize_mxfp4_e2m1_packed(torch.zeros(1, 16, dtype=torch.int8), torch.ones(2, 1))

    with pytest.raises(RuntimeError, match="expects a 2-D weight"):
        quantize_mxfp4_e2m1_like_scale(torch.ones(1, 1, 32), torch.ones(1, 1))

    with pytest.raises(RuntimeError, match="Unsupported MXFP4 geometry"):
        quantize_mxfp4_e2m1_like_scale(torch.ones(1, 30), torch.ones(1, 1))


def test_maybe_dequantize_hf_quantized_weight_dispatches_by_dtype_and_sibling_scale():
    fp8_weight = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    mxfp4_weight = torch.zeros(1, 16, dtype=torch.int8)
    hf_state_dict = {
        "fp8.weight": fp8_weight,
        "fp8.scale": torch.ones(2),
        "fp8.activation": fp8_weight,
        "mxfp4.weight": mxfp4_weight,
        "mxfp4.scale": torch.ones(1, 1),
        "int8_without_scale.weight": mxfp4_weight,
        "bf16.weight": torch.ones(1, dtype=torch.bfloat16),
    }

    result = maybe_dequantize_hf_quantized_weight(
        {"fp8": "fp8.weight", "mxfp4": "mxfp4.weight"},
        hf_state_dict,
        dtype=torch.float32,
    )

    assert set(result) == {"fp8", "mxfp4"}
    assert result["fp8"].dtype == torch.float32
    assert result["mxfp4"].dtype == torch.float32
    assert maybe_dequantize_hf_quantized_weight("fp8.activation", hf_state_dict).dtype == torch.bfloat16
    assert maybe_dequantize_hf_quantized_weight("int8_without_scale.weight", hf_state_dict).dtype == torch.bfloat16
    assert maybe_dequantize_hf_quantized_weight("bf16.weight", hf_state_dict) is hf_state_dict["bf16.weight"]


def test_requantize_hf_weight_scale_pairs_emits_scale_siblings():
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    scale_key = "layers.0.ffn.experts.0.w1.scale"
    weight = torch.ones((1, 32), dtype=torch.bfloat16)
    source_scale = torch.ones((1, 1), dtype=torch.float32)
    stale_scale = torch.full((1, 1), 9.0, dtype=torch.float32)

    result = requantize_hf_weight_scale_pairs(
        {weight_key: weight, scale_key: stale_scale},
        {scale_key: source_scale},
        use_mxfp4=lambda *_: True,
    )

    assert set(result) == {weight_key, scale_key}
    assert result[weight_key].dtype == torch.int8
    assert result[scale_key].shape == source_scale.shape
    assert not torch.equal(result[scale_key], stale_scale)


def test_requantize_hf_weight_scale_pairs_preserves_unscaled_weights_and_stale_scales():
    weight_key = "layers.0.self_attn.q_proj.weight"
    stale_scale_key = "layers.0.self_attn.k_proj.scale"
    scaled_weight_key = "layers.0.mlp.down_proj.weight"
    scaled_scale_key = "layers.0.mlp.down_proj.scale"
    unscaled_weight = torch.ones(1, 2, dtype=torch.bfloat16)
    stale_scale = torch.full((1, 1), 9.0)
    scaled_weight = torch.full((2, 4), 2.0, dtype=torch.bfloat16)
    source_scale = torch.ones(2)

    result = requantize_hf_weight_scale_pairs(
        {
            weight_key: unscaled_weight,
            stale_scale_key: stale_scale,
            scaled_weight_key: scaled_weight,
            scaled_scale_key: stale_scale,
            "metadata": torch.tensor(1.0),
        },
        {scaled_scale_key: source_scale},
    )

    assert result[weight_key] is unscaled_weight
    assert result["metadata"].item() == 1.0
    assert stale_scale_key in result
    assert scaled_scale_key in result
    assert result[scaled_weight_key].dtype == torch.float8_e4m3fn
    assert result[scaled_scale_key].shape == source_scale.shape


def test_quantize_dequantize_int4_preserves_shape_and_dtype():
    weight = torch.linspace(-1.0, 1.0, steps=32).view(1, 32).to(torch.bfloat16)

    packed, scale, shape = quantize_to_int4(weight)
    result = dequantize_int4(packed, scale, shape)

    assert packed.shape == (1, 4)
    assert shape.tolist() == [1, 32]
    assert result.shape == weight.shape
    assert result.dtype == torch.bfloat16
