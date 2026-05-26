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
from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F


FP8_BLOCK_SIZE = 128
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
MXFP4_BLOCK_SIZE = 32

_FP4_E2M1_TABLE_VALUES = [
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
]


def is_fp8_tensor(tensor: torch.Tensor) -> bool:
    """Return whether *tensor* uses one of PyTorch's FP8 dtypes."""
    return tensor.dtype in FP8_DTYPES


def is_float8_e8m0_dtype(dtype: torch.dtype) -> bool:
    """Return whether *dtype* is PyTorch's E8M0 scale dtype."""
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return e8m0_dtype is not None and dtype == e8m0_dtype


def scale_from_amax(amax: torch.Tensor, max_quantized_value: float, scale_dtype: torch.dtype) -> torch.Tensor:
    """Build positive quantization scales in the same scale family as ``scale_dtype``."""
    scale = torch.where(amax > 0, amax / max_quantized_value, torch.ones_like(amax))
    if is_float8_e8m0_dtype(scale_dtype):
        scale = scale.clamp(min=2.0**-127, max=2.0**127)
        return torch.exp2(torch.ceil(torch.log2(scale)))
    return scale


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


def dequantize_fp8_e4m3fn_with_scale(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize FP8 E4M3 weights with a companion scale tensor.

    Supports three common HF checkpoint scale layouts:
    1-D per-row or row-block scales, 2-D per-row K tiles, and 2-D block scales.
    """
    weight_f32 = weight.to(torch.float32)
    scale_f32 = scale.to(torch.float32)
    if scale_f32.dim() == 1:
        if scale_f32.shape[0] == weight_f32.shape[0]:
            scale_exp = scale_f32.unsqueeze(1)
        else:
            scale_exp = scale_f32.repeat_interleave(block_size)[: weight_f32.shape[0]].unsqueeze(1)
    elif scale_f32.dim() != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"Unsupported FP8 scale rank{label}: scale={tuple(scale.shape)}")
    elif scale_f32.shape[0] == weight_f32.shape[0]:
        tile_k = weight_f32.shape[1] // scale_f32.shape[1]
        scale_exp = scale_f32.repeat_interleave(tile_k, dim=1)[:, : weight_f32.shape[1]]
    else:
        scale_exp = scale_f32.repeat_interleave(block_size, dim=0)[: weight_f32.shape[0]]
        scale_exp = scale_exp.repeat_interleave(block_size, dim=1)[:, : weight_f32.shape[1]]
    if scale_exp.shape != weight_f32.shape:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"FP8 dequant shape mismatch{label}: "
            f"weight={tuple(weight_f32.shape)} scale={tuple(scale.shape)} "
            f"scale_exp={tuple(scale_exp.shape)}"
        )
    return (weight_f32 * scale_exp).to(dtype)


def _quantize_fp8_2d_blocks(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    expected_shape = (
        (rows + block_size - 1) // block_size,
        (cols + block_size - 1) // block_size,
    )
    if (scale_rows, scale_cols) != expected_shape:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)} expected={expected_shape}"
        )

    weight_f32 = weight.to(torch.float32)
    pad_rows = scale_rows * block_size - rows
    pad_cols = scale_cols * block_size - cols
    if pad_rows or pad_cols:
        weight_f32 = F.pad(weight_f32, (0, pad_cols, 0, pad_rows))

    blocks = weight_f32.view(scale_rows, block_size, scale_cols, block_size).transpose(1, 2)
    scale_f32 = scale_from_amax(blocks.abs().amax(dim=(-1, -2)), FP8_E4M3_MAX, source_scale.dtype)
    q_blocks = (blocks / scale_f32[:, :, None, None]).to(torch.float8_e4m3fn)
    q_weight = q_blocks.transpose(1, 2).reshape(scale_rows * block_size, scale_cols * block_size)[:rows, :cols]
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_per_row_tiles(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = weight.shape
    scale_rows, scale_cols = source_scale.shape
    if scale_rows != rows or scale_cols <= 0 or cols % scale_cols != 0:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported per-row FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)}"
        )

    tile_cols = cols // scale_cols
    grouped = weight.to(torch.float32).reshape(rows, scale_cols, tile_cols)
    scale_f32 = scale_from_amax(grouped.abs().amax(dim=-1), FP8_E4M3_MAX, source_scale.dtype)
    q_weight = (grouped / scale_f32[:, :, None]).to(torch.float8_e4m3fn).reshape(rows, cols)
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def _quantize_fp8_1d_scale(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, _ = weight.shape
    scale_len = source_scale.numel()
    weight_f32 = weight.to(torch.float32)

    if scale_len == rows:
        scale_f32 = scale_from_amax(weight_f32.abs().amax(dim=1), FP8_E4M3_MAX, source_scale.dtype)
        q_weight = (weight_f32 / scale_f32[:, None]).to(torch.float8_e4m3fn)
        return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)

    expected_len = (rows + block_size - 1) // block_size
    if scale_len != expected_len:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported 1-D FP8 scale geometry{label}: "
            f"weight={tuple(weight.shape)} scale={tuple(source_scale.shape)} expected_len={expected_len}"
        )

    q_weight = torch.empty_like(weight_f32, dtype=torch.float8_e4m3fn)
    scale_f32 = torch.empty(scale_len, dtype=torch.float32, device=weight.device)
    for block_idx in range(scale_len):
        row_start = block_idx * block_size
        row_end = min(row_start + block_size, rows)
        block = weight_f32[row_start:row_end]
        block_scale = scale_from_amax(block.abs().amax().reshape(()), FP8_E4M3_MAX, source_scale.dtype)
        q_weight[row_start:row_end] = (block / block_scale).to(torch.float8_e4m3fn)
        scale_f32[block_idx] = block_scale
    return q_weight.contiguous(), scale_f32.to(dtype=source_scale.dtype)


def quantize_fp8_e4m3fn_like_scale(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = FP8_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to FP8 E4M3 using ``source_scale`` geometry and dtype."""
    if weight.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"FP8 quantized export expects a 2-D weight{label}, got {weight.ndim}D")

    if source_scale.ndim == 1:
        return _quantize_fp8_1d_scale(weight, source_scale, name=name, block_size=block_size)

    if source_scale.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"Unsupported FP8 scale rank{label}: scale={tuple(source_scale.shape)}")

    if source_scale.shape[0] == weight.shape[0]:
        return _quantize_fp8_per_row_tiles(weight, source_scale, name=name)

    return _quantize_fp8_2d_blocks(weight, source_scale, name=name, block_size=block_size)


def dequantize_mxfp4(
    blocks: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Dequantize GPT-OSS MXFP4 block/scales tensors."""
    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"
    scales = scales.to(torch.int32) - 127
    lut = torch.tensor(_FP4_E2M1_TABLE_VALUES, dtype=dtype, device=blocks.device)

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


def dequantize_mxfp4_e2m1_packed(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize MXFP4 E2M1 weights packed two values per byte.

    ``scale`` is expected to be one scale per row and per K tile. E8M0 scale
    tensors can be passed directly; ``.to(torch.float32)`` materializes their
    power-of-two values.
    """
    w_u8 = weight_packed.view(torch.uint8)
    lo = (w_u8 & 0xF).to(torch.int64)
    hi = (w_u8 >> 4).to(torch.int64)

    table = torch.tensor(_FP4_E2M1_TABLE_VALUES, dtype=torch.float32, device=weight_packed.device)
    logical = torch.stack([table[lo], table[hi]], dim=-1).reshape(weight_packed.shape[0], -1)

    scale_f32 = scale.to(torch.float32)
    if scale_f32.dim() != 2 or scale_f32.shape[0] != logical.shape[0] or logical.shape[1] % scale_f32.shape[1] != 0:
        raise RuntimeError(
            f"Unsupported MXFP4 scale geometry: "
            f"weight={tuple(weight_packed.shape)} logical={tuple(logical.shape)} scale={tuple(scale.shape)}"
        )
    block_size = logical.shape[1] // scale_f32.shape[1]
    scale_exp = scale_f32.repeat_interleave(block_size, dim=1)

    return (logical * scale_exp).to(dtype)


def is_mxfp4_e2m1_scale_geometry(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    block_size: int = MXFP4_BLOCK_SIZE,
) -> bool:
    """Return whether ``source_scale`` describes packed MXFP4 E2M1 K tiles."""
    return (
        weight.ndim == 2
        and source_scale.ndim == 2
        and source_scale.shape[0] == weight.shape[0]
        and source_scale.shape[1] * block_size == weight.shape[1]
    )


def quantize_mxfp4_e2m1_like_scale(
    weight: torch.Tensor,
    source_scale: torch.Tensor,
    *,
    name: str = "",
    block_size: int = MXFP4_BLOCK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2-D weight to packed MXFP4 E2M1 using source scale geometry."""
    if weight.ndim != 2:
        label = f" for {name!r}" if name else ""
        raise RuntimeError(f"MXFP4 quantized export expects a 2-D weight{label}, got {weight.ndim}D")

    rows, cols = weight.shape
    if cols % 2 != 0 or cols % block_size != 0 or source_scale.shape != (rows, cols // block_size):
        label = f" for {name!r}" if name else ""
        raise RuntimeError(
            f"Unsupported MXFP4 geometry{label}: weight={tuple(weight.shape)} scale={tuple(source_scale.shape)}"
        )

    weight_f32 = weight.to(torch.float32)
    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device=weight.device)
    scale_f32 = torch.empty(tuple(source_scale.shape), dtype=torch.float32, device=weight.device)
    boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=weight.device)

    max_chunk_elements = 16_000_000
    rows_per_chunk = max(1, min(rows, max_chunk_elements // max(cols, 1)))
    scale_cols = source_scale.shape[1]
    for row_start in range(0, rows, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, rows)
        chunk = weight_f32[row_start:row_end].reshape(-1, scale_cols, block_size)
        chunk_scale = scale_from_amax(chunk.abs().amax(dim=-1), FP4_E2M1_MAX, source_scale.dtype)
        scale_f32[row_start:row_end] = chunk_scale

        normalized = chunk / chunk_scale[:, :, None]
        codes = torch.bucketize(normalized.abs(), boundaries).to(torch.uint8)
        codes = (codes | ((normalized < 0).to(torch.uint8) * 8)).reshape(row_end - row_start, cols)

        lo = codes[:, 0::2].to(torch.int16)
        hi = codes[:, 1::2].to(torch.int16)
        packed[row_start:row_end] = (lo | (hi << 4)).to(torch.uint8)

    return packed.contiguous().view(torch.int8), scale_f32.to(dtype=source_scale.dtype)


def maybe_dequantize_hf_quantized_weight(
    hf_param: str | dict[str, str],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Load and dequantize HF ``*.weight`` tensors that carry sibling ``*.scale`` tensors."""
    if isinstance(hf_param, dict):
        return {k: maybe_dequantize_hf_quantized_weight(v, hf_state_dict, dtype=dtype) for k, v in hf_param.items()}

    weight = hf_state_dict[hf_param]

    if weight.dtype == torch.int8:
        scale_key = hf_param[: -len(".weight")] + ".scale" if hf_param.endswith(".weight") else None
        if scale_key is None or scale_key not in hf_state_dict:
            return weight.to(dtype)
        return dequantize_mxfp4_e2m1_packed(weight, hf_state_dict[scale_key], dtype=dtype)

    if weight.dtype != torch.float8_e4m3fn:
        return weight

    if not hf_param.endswith(".weight"):
        return weight.to(dtype)

    scale_key = hf_param[: -len(".weight")] + ".scale"
    if scale_key not in hf_state_dict:
        return weight.to(dtype)

    return dequantize_fp8_e4m3fn_with_scale(weight, hf_state_dict[scale_key], name=hf_param, dtype=dtype)


def requantize_hf_weight_scale_pairs(
    converted_weights_dict: Mapping[str, torch.Tensor],
    hf_state_dict: Mapping[str, torch.Tensor],
    *,
    use_mxfp4: Callable[[str, torch.Tensor, torch.Tensor], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Recreate quantized HF ``*.weight``/``*.scale`` pairs using source scale layout.

    ``use_mxfp4`` lets model bridges opt specific parameters into packed MXFP4
    output. Other scaled weights are emitted as FP8 E4M3.
    """
    result: dict[str, torch.Tensor] = {}
    for hf_param, weight in converted_weights_dict.items():
        if not hf_param.endswith(".weight"):
            if hf_param.endswith(".scale"):
                weight_key = hf_param[: -len(".scale")] + ".weight"
                if weight_key in converted_weights_dict:
                    continue
            result[hf_param] = weight
            continue

        scale_key = hf_param[: -len(".weight")] + ".scale"
        if scale_key not in hf_state_dict:
            result[hf_param] = weight
            continue

        source_scale = hf_state_dict[scale_key]
        if use_mxfp4 is not None and use_mxfp4(hf_param, weight, source_scale):
            q_weight, q_scale = quantize_mxfp4_e2m1_like_scale(weight, source_scale, name=hf_param)
        else:
            q_weight, q_scale = quantize_fp8_e4m3fn_like_scale(weight, source_scale, name=hf_param)

        result[hf_param] = q_weight
        result[scale_key] = q_scale
    return result


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
