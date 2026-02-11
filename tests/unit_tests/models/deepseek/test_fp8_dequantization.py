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

"""Unit tests for FP8 block-wise dequantization used by DeepSeek-V3 and Kimi-K2.5."""

import math

import pytest
import torch

from megatron.bridge.models.deepseek.common import (
    dequantize_fp8_blockwise,
    maybe_dequantize_fp8_weight,
)

_requires_fp8 = pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch.float8_e4m3fn not available in this PyTorch build",
)

FP8_E4M3_MAX = 448.0


def _quantize_to_fp8(tensor: torch.Tensor, block_size: int = 128):
    """Reference quantization: bf16 -> (fp8, scale_inv).

    Mirrors the logic in ``create_hf_toy_model.py --quantize-fp8``.
    """
    M, N = tensor.shape
    num_blocks_m = math.ceil(M / block_size)
    num_blocks_n = math.ceil(N / block_size)
    padded_M = num_blocks_m * block_size
    padded_N = num_blocks_n * block_size

    padded = torch.zeros(padded_M, padded_N, dtype=tensor.dtype)
    padded[:M, :N] = tensor

    blocks = padded.reshape(num_blocks_m, block_size, num_blocks_n, block_size)
    abs_max = blocks.abs().amax(dim=(1, 3))
    scale_inv = (abs_max / FP8_E4M3_MAX).clamp(min=1e-12).to(torch.float32)

    scaled = blocks / scale_inv[:, None, :, None]
    scaled = scaled.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).reshape(padded_M, padded_N)
    if M != padded_M or N != padded_N:
        scaled = scaled[:M, :N].contiguous()

    fp8_weight = scaled.to(torch.float8_e4m3fn)
    return fp8_weight, scale_inv


class TestDequantizeFp8Blockwise:
    """Tests for ``dequantize_fp8_blockwise``."""

    @_requires_fp8
    def test_roundtrip_divisible(self):
        """Quantize -> dequantize round-trip with dimensions divisible by block_size."""
        original = torch.randn(256, 512, dtype=torch.float32)
        fp8, scale_inv = _quantize_to_fp8(original, block_size=128)

        recovered = dequantize_fp8_blockwise(fp8, scale_inv, dtype=torch.float32)

        assert recovered.shape == original.shape
        assert recovered.dtype == torch.float32
        # FP8 e4m3 has ~3 bits of mantissa; relative error should be small
        rel_err = (recovered - original).abs() / (original.abs() + 1e-8)
        assert rel_err.mean() < 0.05, f"Mean relative error too large: {rel_err.mean():.4f}"

    @_requires_fp8
    def test_roundtrip_non_divisible(self):
        """Round-trip where dimensions are NOT multiples of block_size."""
        original = torch.randn(200, 300, dtype=torch.float32)
        fp8, scale_inv = _quantize_to_fp8(original, block_size=128)

        recovered = dequantize_fp8_blockwise(fp8, scale_inv, dtype=torch.float32)

        assert recovered.shape == (200, 300)
        rel_err = (recovered - original).abs() / (original.abs() + 1e-8)
        assert rel_err.mean() < 0.05

    @_requires_fp8
    def test_output_dtype(self):
        """Output dtype matches the requested dtype."""
        original = torch.randn(128, 128, dtype=torch.float32)
        fp8, scale_inv = _quantize_to_fp8(original, block_size=128)

        for dtype in (torch.bfloat16, torch.float32):
            result = dequantize_fp8_blockwise(fp8, scale_inv, dtype=dtype)
            assert result.dtype == dtype

    @_requires_fp8
    def test_zeros_preserved(self):
        """All-zero weight stays zero after round-trip."""
        original = torch.zeros(128, 256, dtype=torch.float32)
        fp8, scale_inv = _quantize_to_fp8(original, block_size=128)

        recovered = dequantize_fp8_blockwise(fp8, scale_inv, dtype=torch.float32)
        assert torch.allclose(recovered, original, atol=1e-6)


class TestMaybeDequantizeFp8Weight:
    """Tests for ``maybe_dequantize_fp8_weight`` (the bridge helper)."""

    @_requires_fp8
    def test_dequantizes_when_fp8_and_scale_present(self):
        """FP8 weight + matching scale_inv -> dequantized output."""
        original = torch.randn(256, 256, dtype=torch.float32)
        fp8, scale_inv = _quantize_to_fp8(original, block_size=128)

        state_dict = {
            "layer.weight": fp8,
            "layer.weight_scale_inv": scale_inv,
        }

        result = maybe_dequantize_fp8_weight("layer.weight", fp8, state_dict)

        assert result.dtype == torch.bfloat16
        assert result.shape == (256, 256)
        # Should be close to original
        rel_err = (result.float() - original).abs() / (original.abs() + 1e-8)
        assert rel_err.mean() < 0.05

    @_requires_fp8
    def test_passthrough_when_no_scale_inv(self):
        """FP8 weight without matching scale_inv is returned as-is."""
        fp8 = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        state_dict = {"layer.weight": fp8}

        result = maybe_dequantize_fp8_weight("layer.weight", fp8, state_dict)
        assert result.dtype == torch.float8_e4m3fn

    def test_passthrough_when_not_fp8(self):
        """Non-FP8 weight is returned unchanged regardless of scale_inv presence."""
        bf16_weight = torch.randn(128, 128, dtype=torch.bfloat16)
        state_dict = {
            "layer.weight": bf16_weight,
            "layer.weight_scale_inv": torch.ones(1, 1),
        }

        result = maybe_dequantize_fp8_weight("layer.weight", bf16_weight, state_dict)
        assert result is bf16_weight
