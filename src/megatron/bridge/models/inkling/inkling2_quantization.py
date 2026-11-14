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

"""NVFP4 and MXFP8 expert weight formats of Inkling2 checkpoints.

NVFP4 experts store packed E2M1 values (``uint8``, low nibble first), one
FP8 E4M3 scale per 16 input channels, and an outer scale
``amax[expert] / outer_scale_denominator``. MXFP8 experts store FP8 E4M3
values with one UE8M0 power-of-two scale per 32 input channels. Bridge imports
both as BF16 and, on export, quantizes them back against the source scales,
which reproduces the source bytes for unchanged weights. A plain export
(an explicit export dtype) writes the experts unquantized instead, and
``align_precision_config`` then lists them under their written dtype in
``hf_quant_config.json``.
"""

from collections.abc import Mapping

import torch


NVFP4_BLOCK = 16
MXFP8_BLOCK = 32
_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_table(device: torch.device) -> torch.Tensor:
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    return torch.cat((magnitudes, -magnitudes))


def dequantize_nvfp4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Dequantize one NVFP4 matrix to FP32.

    Args:
        packed: ``[N, K / 2]`` ``uint8`` codes, element ``2i`` in the low nibble.
        scale: ``[N, K / 16]`` FP8 E4M3 block scales.
        amax: Scalar FP32 outer amax of this expert.
        denominator: Scalar FP32 outer-scale denominator.

    Returns:
        ``[N, K]`` FP32 weights.
    """
    if packed.dtype != torch.uint8 or scale.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Unexpected NVFP4 dtypes: values {packed.dtype}, scales {scale.dtype}")
    table = _e2m1_table(packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2).long()
    values = table[codes]
    blocks = scale.float().repeat_interleave(NVFP4_BLOCK, dim=-1)
    if blocks.shape != values.shape:
        raise ValueError(f"NVFP4 scale shape {tuple(scale.shape)} does not match values {tuple(values.shape)}")
    outer = amax.float().reshape(()) / denominator.float().reshape(())
    return values * blocks * outer


def quantize_nvfp4_like(
    weight: torch.Tensor,
    scale: torch.Tensor,
    amax: torch.Tensor,
    denominator: torch.Tensor,
) -> torch.Tensor:
    """Quantize ``[N, K]`` weights to packed NVFP4 codes against existing scales.

    Each value is divided by its block's dequantization factor and rounded to the
    nearest E2M1 magnitude (saturating at 6), preserving the sign of zero.
    """
    blocks = scale.float().repeat_interleave(NVFP4_BLOCK, dim=-1) * (
        amax.float().reshape(()) / denominator.float().reshape(())
    )
    weight = weight.float()
    ratio = torch.where(blocks != 0, weight / torch.where(blocks != 0, blocks, 1.0), torch.zeros_like(weight))
    magnitudes = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=weight.device)
    midpoints = (magnitudes[1:] + magnitudes[:-1]) / 2
    # bucketize maps a value equal to midpoints[j] to the lower code j. Round half to even,
    # as cvt.rn does: an odd lower code on an exact midpoint moves up to the even code.
    index = torch.bucketize(ratio.abs(), midpoints)
    on_midpoint = (index < midpoints.numel()) & (ratio.abs() == midpoints[index.clamp(max=midpoints.numel() - 1)])
    index = torch.where(on_midpoint & (index % 2 == 1), index + 1, index)
    codes = (index | (torch.signbit(weight).long() << 3)).to(torch.uint8)
    pairs = codes.reshape(*codes.shape[:-1], -1, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4)


def dequantize_mxfp8(values: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantize one MXFP8 matrix (``[N, K]`` E4M3 values, ``[N, K / 32]`` UE8M0 scales) to FP32."""
    if values.dtype != torch.float8_e4m3fn or scale.dtype != torch.uint8:
        raise ValueError(f"Unexpected MXFP8 dtypes: values {values.dtype}, scales {scale.dtype}")
    exponents = torch.exp2(scale.float() - 127.0).repeat_interleave(MXFP8_BLOCK, dim=-1)
    return values.float() * exponents


def quantize_mxfp8_like(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``[N, K]`` weights to E4M3 against existing UE8M0 block scales."""
    exponents = torch.exp2(scale.float() - 127.0).repeat_interleave(MXFP8_BLOCK, dim=-1)
    return (weight.float() / exponents).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def expert_format(name: str, keys: Mapping[str, object] | set[str]) -> str | None:
    """Return ``"nvfp4"``, ``"mxfp8"``, or ``None`` for an expert weight key."""
    if f"{name}_scale" not in keys:
        return None
    return "nvfp4" if f"{name}_amax" in keys else "mxfp8"


def companion_keys(name: str, fmt: str | None) -> tuple[str, ...]:
    """Return the scale tensors stored next to a quantized expert weight."""
    if fmt == "nvfp4":
        return (f"{name}_scale", f"{name}_amax", f"{name}_outer_scale_denominator")
    if fmt == "mxfp8":
        return (f"{name}_scale",)
    return ()


def align_precision_config(config: dict, written_dtypes: Mapping[str, str]) -> bool:
    """Make an ``hf_quant_config.json`` precision config match the dtypes an export wrote.

    The config assigns every tensor to a parameter group whose ``output_dtype``
    is its storage dtype. A tensor written in another dtype, such as an NVFP4
    expert exported in BF16, moves to the plain group of that dtype; plain
    groups carry no quantization settings, so its scale entries go too. Groups
    left empty are removed, and the ``mtp`` section, which describes the
    separate draft file, is left unchanged.

    Args:
        config: Parsed ``hf_quant_config.json``, updated in place.
        written_dtypes: Torch dtype name (``"bfloat16"``) of every exported tensor.

    Returns:
        Whether ``config`` changed.

    Raises:
        ValueError: If a tensor was written in a dtype other than BF16 or FP32,
            which a plain parameter group cannot describe.
    """
    groups = config["model"]["parameter_groups"]
    moved: dict[str, list[str]] = {}
    for group in groups:
        kept = []
        for name in group["tensors"]:
            dtype = written_dtypes.get(name, group["output_dtype"])
            if dtype == group["output_dtype"]:
                kept.append(name)
            elif dtype in ("bfloat16", "float32"):
                moved.setdefault(dtype, []).append(name)
            else:
                raise ValueError(f"{name} was exported as {dtype}; hf_quant_config.json supports BF16 and FP32")
        group["tensors"] = kept
    if not moved:
        return False
    for dtype, names in moved.items():
        plain = next(
            (
                group
                for group in groups
                if group["compute_dtype"] == dtype
                and group["output_dtype"] == dtype
                and "fwd_output_dtype" not in group
            ),
            None,
        )
        if plain is None:
            plain = {
                "compute_dtype": dtype,
                "output_dtype": dtype,
                "required_activation_quantization": None,
                "weight_block_shape": [],
                "weight_transform": "cast",
                "tensors": [],
            }
            groups.append(plain)
        plain["tensors"] = sorted(plain["tensors"] + names)
    groups[:] = [group for group in groups if group["tensors"]]
    return True


class LazyQuantizedExperts:
    """Fused ``[E, N, K]`` expert weights that dequantize one expert at a time.

    Importing a whole fused expert tensor at once would hold every expert of every
    layer in memory. Indexing reads one expert's slice from the safetensors shard
    and returns it in BF16.
    """

    ndim = 3

    def __init__(self, source, name: str, fmt: str) -> None:
        self.source = source
        self.name = name
        self.format = fmt
        self._scalars: dict[str, torch.Tensor] = {}

    def _slice(self, key: str):
        path = self.source.path / self.source.key_to_filename_map[key]
        return self.source._get_handle(path).get_slice(key)

    def _small(self, key: str) -> torch.Tensor:
        if key not in self._scalars:
            self._scalars[key] = self.source.load_tensors([key])[key]
        return self._scalars[key]

    @property
    def shape(self) -> torch.Size:
        """Return the dequantized fused shape ``[E, N, K]``."""
        experts, rows, width = self._slice(self.name).get_shape()
        return torch.Size((experts, rows, width * 2 if self.format == "nvfp4" else width))

    def __getitem__(self, expert: int) -> torch.Tensor:
        values = self._slice(self.name)[expert]
        scale = self._slice(f"{self.name}_scale")[expert]
        if self.format == "nvfp4":
            amax = self._small(f"{self.name}_amax")[expert]
            denominator = self._small(f"{self.name}_outer_scale_denominator")
            weight = dequantize_nvfp4(values, scale, amax, denominator)
        else:
            weight = dequantize_mxfp8(values, scale)
        return weight.to(torch.bfloat16)
