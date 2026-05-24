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

import torch

from megatron.bridge.models.conversion.quantization_utils import (
    dequantize_fp8_blockwise,
    dequantize_int4,
    dequantize_mxfp4,
    maybe_dequantize_fp8,
    maybe_dequantize_fp8_blockwise,
    quantize_to_int4,
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


def test_quantize_dequantize_int4_preserves_shape_and_dtype():
    weight = torch.linspace(-1.0, 1.0, steps=32).view(1, 32).to(torch.bfloat16)

    packed, scale, shape = quantize_to_int4(weight)
    result = dequantize_int4(packed, scale, shape)

    assert packed.shape == (1, 4)
    assert shape.tolist() == [1, 32]
    assert result.shape == weight.shape
    assert result.dtype == torch.bfloat16
