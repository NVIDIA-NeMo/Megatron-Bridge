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

import torch


FP8_BLOCK_SIZE = 128
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


def is_fp8_tensor(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* uses one of PyTorch's FP8 dtypes."""
    return tensor.dtype in FP8_DTYPES


def dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights with one scale per 2D block.

    DeepSeek-V3 and MiniMax-M2 store linear weights as FP8 tensors with a
    separate ``*_scale_inv`` tensor. Each scale applies to one 128x128 weight
    block by default.
    """
    M, N = weight.shape
    w = weight.float()
    out = torch.empty_like(w)
    sM, sN = scale_inv.shape
    for bi in range(sM):
        for bj in range(sN):
            r0, r1 = bi * block_size, min((bi + 1) * block_size, M)
            c0, c1 = bj * block_size, min((bj + 1) * block_size, N)
            out[r0:r1, c0:c1] = w[r0:r1, c0:c1] * scale_inv[bi, bj]
    return out.to(dtype)


def maybe_dequantize_fp8_blockwise(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None = None,
    *,
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 block-scaled weights, falling back to a plain cast."""
    if not is_fp8_tensor(weight):
        return weight
    if weight.ndim == 2 and scale_inv is not None:
        return dequantize_fp8_blockwise(weight, scale_inv, block_size=block_size, dtype=dtype)
    return weight.float().to(dtype)


def maybe_dequantize_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None = None,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 weights with a scalar or broadcastable scale tensor."""
    if not is_fp8_tensor(weight):
        return weight
    if scale_inv is None:
        return weight.to(dtype)
    return weight.to(dtype) * scale_inv.to(dtype)


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize GPT-OSS MXFP4 block/scales tensors."""
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    fp4_values = [
        +0.0,
        +0.5,
        +1.0,
        +1.5,
        +2.0,
        +3.0,
        +4.0,
        +6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(fp4_values, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    return out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)


def dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int = 32,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Dequantize Kimi INT4 packed weights to bfloat16.

    The checkpoint stores eight offset-binary INT4 values in each int32 slot and
    carries per-group scales beside the packed tensor.
    """
    del weight_shape, group_size

    local_out, local_packed_in = weight_packed.shape
    local_in = local_packed_in * 8

    target_device = weight_packed.device if device is None else torch.device(device)
    use_cuda = target_device.type == "cuda" and torch.cuda.is_available()

    if use_cuda:
        weight_packed = weight_packed.to(target_device)
        weight_scale = weight_scale.to(target_device)

    shifts = torch.arange(8, device=weight_packed.device) * 4

    packed_unsqueezed = weight_packed.unsqueeze(-1)
    unpacked = ((packed_unsqueezed >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(local_out, local_in)

    unpacked = unpacked - 8

    scale = weight_scale.float()
    if scale.ndim == 1:
        local_num_groups = scale.numel() // local_out
        scale = scale.view(local_out, local_num_groups)
    else:
        scale = scale.view(local_out, -1)

    local_num_groups = scale.shape[1]
    elements_per_group = local_in // local_num_groups

    scale_expanded = scale.repeat_interleave(elements_per_group, dim=1)

    if scale_expanded.shape[1] < local_in:
        scale_expanded = torch.nn.functional.pad(
            scale_expanded, (0, local_in - scale_expanded.shape[1]), value=scale_expanded[:, -1:].mean()
        )
    scale_expanded = scale_expanded[:, :local_in]
    result = unpacked * scale_expanded

    return result.to(torch.bfloat16)


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
    scale_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to Kimi INT4 packed format."""
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int32)

    w = weight.float()

    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)

    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)

    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"

    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)

    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    weight_packed = packed
    weight_scale = scale.to(scale_dtype)

    return weight_packed, weight_scale, weight_shape
