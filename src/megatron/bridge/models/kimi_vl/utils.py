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


import torch


_INT4_SHIFTS: dict[str, torch.Tensor] = {}


def _get_int4_shifts(device: torch.device) -> torch.Tensor:
    """Return cached shift constants for INT4 unpacking on the given device."""
    key = str(device)
    if key not in _INT4_SHIFTS:
        _INT4_SHIFTS[key] = torch.arange(8, device=device, dtype=torch.int32) * 4
    return _INT4_SHIFTS[key]


def dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Dequantize INT4 packed weights to bfloat16.

    Extracts local tensors from DTensors before unpacking (bitwise ops don't work on DTensor).
    Both weight_packed and weight_scale should have matching sharding so .to_local() gives
    corresponding slices automatically.

    Args:
        weight_packed: INT4 packed weights [out_features, in_features // 8], may be DTensor
        weight_scale: Per-group scales [out_features, num_groups], should be DTensor with same sharding
        weight_shape: Original shape [2], stores global dimensions
        group_size: Elements per scale group (default 32)
        device: Target device for computation
    """

    local_out, local_packed_in = weight_packed.shape
    local_in = local_packed_in * 8  # 8 INT4 values per int32

    use_cuda = device == "cuda" and torch.cuda.is_available()

    if use_cuda:
        weight_packed = weight_packed.cuda()
        weight_scale = weight_scale.cuda()

    # Unpack INT4: [out, packed_in] -> [out, packed_in, 8] -> [out, in_features]
    shifts = _get_int4_shifts(weight_packed.device)

    # Unpack, convert to signed, and cast to float in one fused expression
    unpacked = (((weight_packed.unsqueeze(-1) >> shifts) & 0xF).to(torch.float32) - 8.0).reshape(local_out, local_in)

    # Apply per-group scale using broadcast (no repeat_interleave allocation)
    scale = weight_scale.float()
    if scale.ndim == 1:
        local_num_groups = scale.numel() // local_out
        scale = scale.view(local_out, local_num_groups)
    else:
        scale = scale.view(local_out, -1)

    local_num_groups = scale.shape[1]
    elements_per_group = local_in // local_num_groups

    # Reshape unpacked to [out, num_groups, elements_per_group], multiply by
    # scale [out, num_groups, 1] via broadcast, then flatten back — avoids
    # the expensive repeat_interleave allocation.
    unpacked = unpacked.view(local_out, local_num_groups, elements_per_group)
    result = (unpacked * scale.unsqueeze(-1)).reshape(local_out, local_in)

    return result.to(torch.bfloat16)


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to INT4 packed format.

    Returns:
        weight_packed: INT4 values packed into int32 (8 values per int32)
        weight_scale: Per-group scale factors (float16)
        weight_shape: Original tensor shape (int64)
    """
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

    # Convert to float32 for computation
    w = weight.float()

    # Compute per-group scales (group along in_features dimension)
    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    # Symmetric quantization: scale = max(abs(w)) / 7
    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)  # Avoid division by zero

    # Quantize: w_q = round(w / scale), clamp to [-8, 7]
    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)

    # Convert signed [-8, 7] to unsigned [0, 15] using offset binary
    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    # Pack 8 INT4 values into each int32 along the in_features dimension
    # HF format: [out_features, in_features//8] - 2D packed tensor
    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"

    # Reshape to [out_features, in_features//8, 8] for packing along dim 1
    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)

    # Pack 8 nibbles into 1 int32
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    weight_packed = packed
    weight_scale = scale.to(torch.float16)

    return weight_packed, weight_scale, weight_shape
