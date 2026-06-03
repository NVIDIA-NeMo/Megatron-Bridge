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

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Iterator

import torch


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.model_bridge import WeightConversionTask

_NVFP4_AMAX_DENOMINATOR = 6.0 * 448.0
_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
)


@dataclass(frozen=True)
class QuantMeta:
    """ModelOpt quantization metadata for one Megatron parameter."""

    qformat: str
    block_size: int
    weight_amax: torch.Tensor | None


def _iter_quant_ignore_name_candidates(name: str) -> Iterator[str]:
    yield name
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if name.endswith(suffix):
            yield name[: -len(suffix)]
            break

    alternate = name.removeprefix("model.") if name.startswith("model.") else f"model.{name}"

    yield alternate
    for suffix in _QUANT_IGNORE_NAME_SUFFIXES:
        if alternate.endswith(suffix):
            yield alternate[: -len(suffix)]
            break


def matches_quant_ignore_pattern(name: str, patterns: list[str]) -> bool:
    """Return whether a parameter name matches any ModelOpt ignore pattern."""
    return any(
        fnmatchcase(candidate, pattern)
        for candidate in _iter_quant_ignore_name_candidates(name)
        for pattern in patterns
    )


def _iter_hf_param_names(mapping) -> Iterator[str]:
    hf_param = mapping.hf_param
    if isinstance(hf_param, str):
        yield hf_param
    else:
        yield from hf_param.values()


def build_hf_to_megatron_name_map(conversion_tasks: list[WeightConversionTask | None]) -> dict[str, str]:
    """Build a map from Hugging Face parameter names to Megatron parameter names."""
    return {
        hf_name: task.global_param_name
        for task in conversion_tasks
        if task is not None
        for hf_name in _iter_hf_param_names(task.mapping)
    }


def collect_modelopt_quant_metadata(conversion_tasks: list[WeightConversionTask | None]) -> dict[str, QuantMeta]:
    """Collect ModelOpt quantization metadata from conversion task modules."""
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NONE,
        get_quantization_format,
        get_weight_block_size,
    )

    metadata: dict[str, QuantMeta] = {}
    for task in conversion_tasks:
        if task is None or task.megatron_module is None or task.param_weight is None:
            continue

        qformat = get_quantization_format(task.megatron_module)
        if qformat == QUANTIZATION_NONE:
            continue

        block_size = get_weight_block_size(task.megatron_module)
        weight_quantizer = getattr(task.megatron_module, "weight_quantizer", None)
        if block_size == 0 or weight_quantizer is None:
            continue
        weight_amax = getattr(weight_quantizer, "_amax", None)

        metadata[task.global_param_name] = QuantMeta(
            qformat=qformat,
            block_size=block_size,
            weight_amax=weight_amax.clone().cpu() if weight_amax is not None else None,
        )
    return metadata


def sync_modelopt_quant_metadata(metadata: dict[str, QuantMeta], group) -> None:
    """Synchronize ModelOpt quantization metadata across a distributed group."""
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, QuantMeta] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, metadata, group=group)

    for rank_metadata in gathered:
        if rank_metadata:
            metadata.update(rank_metadata)


def compute_nvfp4_weight_scale(
    weight: torch.Tensor,
    block_size: int,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    """Compute the NVFP4 per-block weight scale tensor for ModelOpt export."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        block_size,
        weights_scaling_factor_2=weight_scale_2.to(weight.device),
        keep_high_precision=True,
    )[0]
    weight_scale = weight_scale.to(torch.float32).abs()
    weight_scale[weight_scale == 0] = 1.0
    return weight_scale.to(torch.float8_e4m3fn)


def quantize_nvfp4_weight(
    name: str,
    weight: torch.Tensor,
    meta: QuantMeta,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield NVFP4 quantized weight tensors and associated scale tensors."""
    from modelopt.torch.export.quant_utils import to_quantized_weight

    if not name.endswith(".weight"):
        raise ValueError(f"Expected '.weight' suffix for NVFP4 export parameter name: {name}")
    if meta.weight_amax is None:
        raise RuntimeError(f"Missing ModelOpt weight amax for quantized parameter {name}")

    weight_amax = meta.weight_amax.to(weight.device).float().abs()
    weight_scale_2 = weight_amax / _NVFP4_AMAX_DENOMINATOR
    weight_scale = compute_nvfp4_weight_scale(weight, meta.block_size, weight_scale_2)
    quantized = to_quantized_weight(
        weight,
        weight_scale,
        meta.qformat,
        weight_scale_2,
        meta.block_size,
    )

    weight_name = name[: -len(".weight")]
    yield name, quantized.detach()
    yield f"{weight_name}.weight_scale", weight_scale.detach()
    yield f"{weight_name}.weight_scale_2", weight_scale_2.detach()


def get_modelopt_quant_exporter(quant_mode: str):
    """Return the ModelOpt quantization format and exporter for a quantization mode."""
    from modelopt.torch.export.quant_utils import QUANTIZATION_NVFP4

    if quant_mode.lower() != "nvfp4":
        raise ValueError(f"Unsupported ModelOpt quant_mode: {quant_mode}")
    return QUANTIZATION_NVFP4, quantize_nvfp4_weight
