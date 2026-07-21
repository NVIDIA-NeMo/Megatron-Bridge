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

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING, Any, Iterator, cast

import torch
from megatron.core.utils import get_pg_size

from megatron.bridge.models.conversion import model_bridge as model_bridge_utils
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


if TYPE_CHECKING:
    from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge

HFExportHook = Callable[[str, torch.Tensor], Iterable[HFWeightTuple]]

_NVFP4_AMAX_DENOMINATOR = 6.0 * 448.0
_NVFP4_MAXBOUND = 6.0
_FP8_E4M3_MIN = 2**-9
_FP8_E4M3_MAX = 448.0
_QUANT_IGNORE_NAME_SUFFIXES = (
    ".weight",
    ".weight_scale",
    ".weight_scale_2",
)
_EXPERT_NUMBER_PATTERNS = (
    re.compile(r"(local_experts\.)(\d+)(\.)"),
    re.compile(r"((?:weight|bias))(\d+)(?=$|\.)"),
    re.compile(r"(experts\.)(\d+)(\.)"),
)
_FUSED_MOE_NVFP4_NAME_MAP = {
    ".experts.gate_up_proj": ".experts.w13_weight",
    # vLLM uses the w13 family for both gated and non-gated first projections;
    # ``is_act_and_mul=False`` selects a single projection shard.
    ".experts.up_proj": ".experts.w13_weight",
    ".experts.down_proj": ".experts.w2_weight",
}


@dataclass(frozen=True)
class QuantMeta:
    """ModelOpt quantization metadata for one Megatron parameter."""

    qformat: str
    block_size: int
    weight_amax: torch.Tensor | None
    weight_scale_2: torch.Tensor | None = None
    input_amax: torch.Tensor | None = None


@dataclass(frozen=True)
class _Nvfp4InputQuantizerView:
    """Minimal quantizer interface for ModelOpt activation-scale export."""

    input_amax: torch.Tensor
    is_enabled: bool = True
    maxbound: float = _NVFP4_MAXBOUND

    def export_amax(self) -> torch.Tensor:
        return self.input_amax


@dataclass(frozen=True)
class _Nvfp4WeightQuantizerView:
    """CPU-only quantizer state needed by ModelOpt's canonical scale export."""

    _amax: torch.Tensor
    block_sizes: object | None = None


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


def is_modelopt_quantizable_weight_name(name: str) -> bool:
    """Return whether an exported HF tensor name should be ModelOpt-quantized."""
    return name.endswith(".weight") or any(name.endswith(suffix) for suffix in _FUSED_MOE_NVFP4_NAME_MAP)


def _is_same_tensor(param_weight: object, weight: object) -> bool:
    if param_weight is weight:
        return True
    if not isinstance(param_weight, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return False
    if param_weight.device.type == "meta" or weight.device.type == "meta":
        return False
    if (
        param_weight.device != weight.device
        or param_weight.dtype != weight.dtype
        or param_weight.layout != torch.strided
        or weight.layout != torch.strided
        or tuple(param_weight.shape) != tuple(weight.shape)
        or tuple(param_weight.stride()) != tuple(weight.stride())
    ):
        return False
    return (
        param_weight.untyped_storage().data_ptr() == weight.untyped_storage().data_ptr()
        and param_weight.storage_offset() == weight.storage_offset()
    )


def _iter_modelopt_weight_quantizers(
    module: torch.nn.Module,
) -> Iterator[tuple[object, object, bool]]:
    iter_weights = getattr(module, "iter_weights_for_calibration", None)
    if iter_weights is not None:
        for weight, weight_quantizer in iter_weights():
            if not _is_enabled_quantizer(weight_quantizer):
                continue
            yield (
                weight,
                weight_quantizer,
                _is_same_tensor(
                    getattr(module, "weight", None),
                    weight,
                ),
            )
        return

    weight_quantizer = getattr(module, "weight_quantizer", None)
    if _is_enabled_quantizer(weight_quantizer):
        for weight_name, weight in module.named_parameters(recurse=False):
            if weight_name == "weight" or (weight_name.startswith("weight") and weight_name[6:].isdigit()):
                yield weight, weight_quantizer, weight_name == "weight"


def _is_enabled_quantizer(quantizer: object) -> bool:
    is_enabled = getattr(quantizer, "is_enabled", None)
    return bool(is_enabled)


def find_modelopt_weight_quantizer_and_module(
    module: torch.nn.Module,
    param_weight: object,
) -> tuple[object | None, torch.nn.Module | None]:
    """Find the enabled weight quantizer and owning module for ``param_weight``."""
    for _, candidate_module in module.named_modules():
        for weight, weight_quantizer, can_use_module in _iter_modelopt_weight_quantizers(candidate_module):
            if _is_same_tensor(param_weight, weight):
                if can_use_module:
                    return weight_quantizer, candidate_module
                quant_module = torch.nn.Module()
                quant_module.weight = weight
                quant_module.weight_quantizer = weight_quantizer
                input_quantizer = getattr(candidate_module, "input_quantizer", None)
                if input_quantizer is not None:
                    quant_module.input_quantizer = input_quantizer
                parallel_state = getattr(candidate_module, "parallel_state", None)
                if parallel_state is not None:
                    quant_module.parallel_state = parallel_state
                return weight_quantizer, quant_module

    return None, None


def _with_quant_meta_tensors(
    meta: QuantMeta,
    *,
    weight_amax: torch.Tensor | None,
    weight_scale_2: torch.Tensor | None,
) -> QuantMeta:
    return QuantMeta(
        qformat=meta.qformat,
        block_size=meta.block_size,
        weight_amax=weight_amax,
        weight_scale_2=weight_scale_2,
        input_amax=meta.input_amax,
    )


def _clone_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().cpu().float().clone()


def _clone_positive_cpu(value: torch.Tensor | None) -> torch.Tensor | None:
    cloned = _clone_cpu(value)
    return None if cloned is None else cloned.abs()


def _get_modelopt_weight_amax(weight_quantizer: object) -> tuple[torch.Tensor | None, bool]:
    """Read the raw global amax without deriving a scale on its CUDA device."""
    value = getattr(weight_quantizer, "global_amax", None)
    if isinstance(value, torch.Tensor):
        return value, True
    value = getattr(weight_quantizer, "_global_amax", None)
    if isinstance(value, torch.Tensor):
        return value, True
    value = getattr(weight_quantizer, "_amax", None)
    return (value, False) if isinstance(value, torch.Tensor) else (None, False)


def _compute_modelopt_weight_scale_2_cpu(
    weight_amax: torch.Tensor | None,
    weight_quantizer: object,
) -> torch.Tensor | None:
    """Run ModelOpt's quantizer-specific global-scale derivation on CPU."""
    if weight_amax is None:
        return None

    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    view = _Nvfp4WeightQuantizerView(
        _amax=weight_amax,
        block_sizes=getattr(weight_quantizer, "block_sizes", None),
    )
    return NVFP4QTensor.get_weights_scaling_factor_2_from_quantizer(view).detach().float().abs()


def _get_modelopt_tp_process_group(module: object) -> object | None:
    """Return the raw TP process group attached to a ModelOpt quantized module."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None

    parallel_state = getattr(module, "parallel_state", None)
    tp_group = getattr(parallel_state, "tensor_parallel_group", None)
    if tp_group is None:
        return None

    is_initialized = getattr(tp_group, "is_initialized", None)
    if callable(is_initialized) and not is_initialized():
        return None

    process_group = getattr(tp_group, "group", tp_group)
    if isinstance(process_group, int) and process_group == -1:
        return None

    get_world_size = getattr(tp_group, "world_size", None)
    world_size = (
        get_world_size() if callable(get_world_size) else torch.distributed.get_world_size(group=process_group)
    )
    return process_group if world_size > 1 else None


def _max_reduce_modelopt_tp_scalar(
    value: torch.Tensor | None,
    module: object,
    field_name: str,
) -> torch.Tensor | None:
    """MAX-reduce one per-tensor quantization value over its owning TP group."""
    if value is None:
        return None

    process_group = _get_modelopt_tp_process_group(module)
    if process_group is None:
        return value
    if value.numel() != 1:
        raise RuntimeError(f"Cannot TP-synchronize non-scalar ModelOpt {field_name} with shape {tuple(value.shape)}")

    reduced = value.detach().clone()
    torch.distributed.all_reduce(
        reduced,
        op=torch.distributed.ReduceOp.MAX,
        group=process_group,
    )
    return reduced


def _slice_optional_quant_tensor(
    value: torch.Tensor | None,
    split: slice,
    leading_dim: int,
) -> torch.Tensor | None:
    if value is None or value.dim() == 0:
        return value
    if value.shape[0] != leading_dim:
        return value
    return value[split].contiguous()


def _slice_gated_quant_meta(meta: QuantMeta, hf_key: str) -> QuantMeta:
    """Slice fused ``[gate; up]`` metadata to match a split HF tensor."""
    if hf_key not in {"gate", "up"} or meta.weight_amax is None or meta.weight_amax.dim() == 0:
        return meta

    leading_dim = meta.weight_amax.shape[0]
    if leading_dim % 2 != 0:
        return meta

    midpoint = leading_dim // 2
    split = slice(0, midpoint) if hf_key == "gate" else slice(midpoint, leading_dim)
    return _with_quant_meta_tensors(
        meta,
        weight_amax=_slice_optional_quant_tensor(meta.weight_amax, split, leading_dim),
        weight_scale_2=meta.weight_scale_2,
    )


def _stack_optional_quant_tensors(
    values: list[torch.Tensor | None],
    *,
    hf_name: str,
    field_name: str,
) -> torch.Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise RuntimeError(f"Incomplete ModelOpt {field_name} metadata for grouped parameter {hf_name}")
    return torch.stack(cast(list[torch.Tensor], values), dim=0).contiguous()


def _stack_grouped_quant_meta(hf_name: str, expert_meta: dict[int, QuantMeta]) -> QuantMeta:
    if not expert_meta:
        raise RuntimeError(f"Missing ModelOpt metadata for grouped parameter {hf_name}")

    expected_experts = set(range(max(expert_meta) + 1))
    missing_experts = sorted(expected_experts.difference(expert_meta))
    if missing_experts:
        raise RuntimeError(f"Missing ModelOpt metadata for experts {missing_experts} of grouped parameter {hf_name}")

    metas = [expert_meta[idx] for idx in sorted(expert_meta)]
    qformat = metas[0].qformat
    block_size = metas[0].block_size
    for meta in metas[1:]:
        if meta.qformat != qformat or meta.block_size != block_size:
            raise RuntimeError(f"Inconsistent ModelOpt metadata for grouped parameter {hf_name}")

    return QuantMeta(
        qformat=qformat,
        block_size=block_size,
        weight_amax=_stack_optional_quant_tensors(
            [meta.weight_amax for meta in metas],
            hf_name=hf_name,
            field_name="weight_amax",
        ),
        weight_scale_2=_stack_optional_quant_tensors(
            [meta.weight_scale_2 for meta in metas],
            hf_name=hf_name,
            field_name="weight_scale_2",
        ),
        input_amax=_stack_optional_quant_tensors(
            [meta.input_amax for meta in metas],
            hf_name=hf_name,
            field_name="input_amax",
        ),
    )


def _expert_param_template(param_name: str) -> str | None:
    for pattern in _EXPERT_NUMBER_PATTERNS:
        match = pattern.search(param_name)
        if match is None:
            continue
        return f"{param_name[: match.start(2)]}{{expert}}{param_name[match.end(2) :]}"
    return None


def _iter_grouped_quant_meta(
    task: WeightConversionTask,
    metadata: dict[str, QuantMeta],
) -> Iterator[tuple[int, QuantMeta]]:
    """Yield all synced per-expert metadata entries for a grouped export task."""
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    task_template = _expert_param_template(task.global_param_name)
    if task_template is None:
        raise ValueError(f"Expected expert parameter name for grouped export: {task.global_param_name}")

    for global_name, meta in metadata.items():
        if _expert_param_template(global_name) == task_template:
            yield extract_expert_number_from_param(global_name), meta


def build_hf_modelopt_quant_metadata(
    conversion_tasks: list[WeightConversionTask],
    metadata: dict[str, QuantMeta],
) -> dict[str, QuantMeta]:
    """Map Megatron ModelOpt metadata onto exported Hugging Face names."""
    hf_metadata: dict[str, QuantMeta] = {}
    grouped_metadata: dict[str, dict[int, QuantMeta]] = {}

    for task in conversion_tasks:
        if task.global_param_name not in metadata:
            continue

        meta = metadata[task.global_param_name]
        hf_param = task.mapping.hf_param
        if isinstance(hf_param, str):
            hf_items = (("", hf_param),)
        else:
            hf_items = tuple(hf_param.items())

        if getattr(task.mapping, "is_grouped_export", False):
            for expert_number, expert_meta in _iter_grouped_quant_meta(task, metadata):
                for hf_key, hf_name in hf_items:
                    grouped_metadata.setdefault(hf_name, {})[expert_number] = _slice_gated_quant_meta(
                        expert_meta,
                        hf_key,
                    )
            continue

        for hf_key, hf_name in hf_items:
            hf_metadata[hf_name] = _slice_gated_quant_meta(meta, hf_key)

    for hf_name, expert_meta in grouped_metadata.items():
        hf_metadata[hf_name] = _stack_grouped_quant_meta(hf_name, expert_meta)

    return hf_metadata


def _build_hf_modelopt_pre_ep_quant_metadata(
    conversion_tasks: list[WeightConversionTask],
    metadata: dict[str, QuantMeta],
) -> dict[str, QuantMeta]:
    """Build metadata for selected local expert batches before EP gathering."""
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    grouped_metadata: dict[str, dict[int, QuantMeta]] = {}
    for task in conversion_tasks:
        if not getattr(task.mapping, "is_modelopt_pre_ep_export", False):
            raise ValueError(f"Expected a pre-EP ModelOpt task, got {task.global_param_name}")
        if task.global_param_name not in metadata:
            raise RuntimeError(f"Missing ModelOpt metadata for pre-EP task {task.global_param_name}")
        if not isinstance(task.mapping.hf_param, str):
            raise ValueError(f"Expected one fused HF name for pre-EP task {task.global_param_name}")

        expert_number = extract_expert_number_from_param(task.global_param_name)
        grouped_metadata.setdefault(task.mapping.hf_param, {})[expert_number] = metadata[task.global_param_name]

    hf_metadata: dict[str, QuantMeta] = {}
    for hf_name, expert_meta in grouped_metadata.items():
        # Global expert ids on EP rank N start at N * experts_per_rank. Rebase
        # the sorted local batch before applying the usual dense validation.
        local_expert_meta = {local_idx: meta for local_idx, (_, meta) in enumerate(sorted(expert_meta.items()))}
        hf_metadata[hf_name] = _stack_grouped_quant_meta(hf_name, local_expert_meta)
    return hf_metadata


def collect_modelopt_quant_metadata(
    conversion_tasks: list[WeightConversionTask],
) -> dict[str, QuantMeta]:
    """Collect ModelOpt quantization metadata from conversion task modules."""
    from modelopt.torch.export import quant_utils
    from modelopt.torch.export.quant_utils import (
        QUANTIZATION_NONE,
        QUANTIZATION_NVFP4,
        get_quantization_format,
        get_weight_block_size,
    )

    w4a16_nvfp4 = getattr(quant_utils, "QUANTIZATION_W4A16_NVFP4", None)
    metadata: dict[str, QuantMeta] = {}
    quantizer_metadata: dict[tuple[int, int, str, int], QuantMeta | None] = {}
    for task in conversion_tasks:
        if task.megatron_module is None or task.param_weight is None:
            continue

        weight_quantizer, quant_module = find_modelopt_weight_quantizer_and_module(
            task.megatron_module,
            task.param_weight,
        )
        if weight_quantizer is None:
            continue

        qformat = get_quantization_format(quant_module)
        block_size = get_weight_block_size(quant_module)
        input_quantizer = getattr(quant_module, "input_quantizer", None)
        quantizer_key = (id(weight_quantizer), id(input_quantizer), qformat, block_size)
        if quantizer_key in quantizer_metadata:
            cached_meta = quantizer_metadata[quantizer_key]
            if cached_meta is not None:
                metadata[task.global_param_name] = cached_meta
            continue

        if qformat == QUANTIZATION_NONE:
            quantizer_metadata[quantizer_key] = None
            continue

        if block_size == 0:
            quantizer_metadata[quantizer_key] = None
            continue
        weight_amax, is_static_weight_quantizer = _get_modelopt_weight_amax(weight_quantizer)
        is_nvfp4 = qformat == QUANTIZATION_NVFP4 or (w4a16_nvfp4 is not None and qformat == w4a16_nvfp4)
        if is_nvfp4 and is_static_weight_quantizer:
            raise RuntimeError(
                "Static NVFP4 weight quantizers are not supported by CPU ModelOpt export: "
                "their calibrated per-block amax values cannot be reconstructed after TP/EP gathering. "
                "Use dynamic NVFP4 block scaling."
            )
        input_amax = None
        if qformat == QUANTIZATION_NVFP4:
            if _is_enabled_quantizer(input_quantizer):
                input_amax = getattr(input_quantizer, "_amax", None)
                if input_amax is None:
                    export_amax = getattr(input_quantizer, "export_amax", None)
                    if callable(export_amax):
                        input_amax = export_amax()

        weight_amax = _max_reduce_modelopt_tp_scalar(weight_amax, quant_module, "weight amax")
        input_amax = _max_reduce_modelopt_tp_scalar(input_amax, quant_module, "input amax")
        weight_amax = _clone_positive_cpu(weight_amax)
        weight_scale_2 = None
        if weight_amax is not None and is_nvfp4:
            weight_scale_2 = _compute_modelopt_weight_scale_2_cpu(
                weight_amax,
                weight_quantizer,
            )

        quant_meta = QuantMeta(
            qformat=qformat,
            block_size=block_size,
            weight_amax=weight_amax,
            weight_scale_2=weight_scale_2,
            input_amax=_clone_cpu(input_amax),
        )
        quantizer_metadata[quantizer_key] = quant_meta
        metadata[task.global_param_name] = quant_meta
    return metadata


def sync_modelopt_quant_metadata(metadata: dict[str, QuantMeta], group=None) -> None:
    """Synchronize ModelOpt quantization metadata across a distributed group."""
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, QuantMeta] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, metadata, group=group)

    for rank_metadata in gathered:
        if rank_metadata:
            metadata.update(rank_metadata)


def _reshape_nvfp4_weight_scale_2_for_compute(
    weight: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    if weight.dim() != 3 or weight_scale_2.dim() == 0:
        return weight_scale_2
    if weight_scale_2.dim() == 1:
        return weight_scale_2.view(-1, 1, 1)
    if weight_scale_2.dim() == 2:
        if weight_scale_2.shape[1] == 1:
            return weight_scale_2.view(weight_scale_2.shape[0], 1, 1)
        if weight.shape[1] % weight_scale_2.shape[1] == 0:
            repeat = weight.shape[1] // weight_scale_2.shape[1]
            return weight_scale_2.repeat_interleave(repeat, dim=1).unsqueeze(-1)
    return weight_scale_2


def compute_nvfp4_weight_scale(
    weight: torch.Tensor,
    block_size: int,
    weight_amax: torch.Tensor | None = None,
    weight_scale_2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the NVFP4 per-block weight scale tensor for ModelOpt export."""
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    if weight_scale_2 is not None:
        scale_2_for_export = weight_scale_2.to(weight.device).float().abs()
    elif weight_amax is not None:
        scale_2_for_export = weight_amax.to(weight.device).float().abs() / _NVFP4_AMAX_DENOMINATOR
    else:
        raise RuntimeError("Missing ModelOpt weight amax for NVFP4 export")

    scale_2_for_compute = _reshape_nvfp4_weight_scale_2_for_compute(
        weight,
        scale_2_for_export,
    )

    weight_scale = NVFP4QTensor.get_weights_scaling_factor(
        weight,
        block_size,
        weights_scaling_factor_2=scale_2_for_compute,
        keep_high_precision=False,
    )[0]
    weight_scale_for_validation = weight_scale.float()
    if not torch.isfinite(weight_scale_for_validation).all():
        raise RuntimeError(f"Invalid ModelOpt NVFP4 per-block weight scale: {weight_scale_for_validation}")
    # ModelOpt versions before its E4M3 clamp can underflow a tiny positive
    # per-block scale to zero. Normalize to the same representable range as the
    # current canonical exporter before the packed weight consumes this scale.
    weight_scale = (
        weight_scale_for_validation.abs().clamp(min=_FP8_E4M3_MIN, max=_FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    )
    return weight_scale, scale_2_for_export


def compute_nvfp4_input_scale(input_amax: torch.Tensor | None) -> torch.Tensor:
    """Compute a static NVFP4 activation scale from synchronized input amax."""
    if input_amax is None:
        raise RuntimeError("Missing ModelOpt input amax for NVFP4 W4A4 export")

    input_amax = input_amax.detach().float()
    if input_amax.numel() == 0 or not torch.isfinite(input_amax).all() or not torch.all(input_amax > 0):
        raise RuntimeError(f"Invalid ModelOpt input amax for NVFP4 W4A4 export: {input_amax}")

    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor

    canonical_export = getattr(NVFP4QTensor, "get_activation_scaling_factor", None)
    if callable(canonical_export):
        input_scale = canonical_export(_Nvfp4InputQuantizerView(input_amax))
    else:
        input_scale = input_amax / _NVFP4_AMAX_DENOMINATOR

    if (
        input_scale is None
        or input_scale.numel() == 0
        or not torch.isfinite(input_scale).all()
        or not torch.all(input_scale > 0)
    ):
        raise RuntimeError(f"Invalid ModelOpt input scale for NVFP4 W4A4 export: {input_scale}")
    return input_scale.detach().float()


def _nvfp4_export_names(name: str) -> tuple[str, str, str, str]:
    if name.endswith(".weight"):
        base = name[: -len(".weight")]
        return (
            name,
            f"{base}.weight_scale",
            f"{base}.weight_scale_2",
            f"{base}.input_scale",
        )
    for hf_suffix, vllm_suffix in _FUSED_MOE_NVFP4_NAME_MAP.items():
        if name.endswith(hf_suffix):
            base = name[: -len(hf_suffix)] + vllm_suffix
            input_scale_base = base.removesuffix("_weight")
            return (
                base,
                f"{base}_scale",
                f"{base}_scale_2",
                f"{input_scale_base}_input_scale",
            )
    raise ValueError(f"Expected quantizable NVFP4 export parameter name: {name}")


def _format_nvfp4_weight_scale_2_for_export(
    source_name: str,
    weight_name: str,
    weight: torch.Tensor,
    weight_scale_2: torch.Tensor,
) -> torch.Tensor:
    if source_name.endswith(".experts.up_proj"):
        if weight_scale_2.dim() == 0:
            return weight_scale_2.expand(weight.shape[0], 1).contiguous()
        if weight_scale_2.dim() == 1:
            return weight_scale_2[:, None].contiguous()
        if weight_scale_2.dim() == 2 and weight_scale_2.shape[1] == 1:
            return weight_scale_2.contiguous()
        raise RuntimeError(
            "Expected one non-gated W13 weight_scale_2 per expert for "
            f"{weight_name}, got shape {tuple(weight_scale_2.shape)}"
        )
    if weight_name.endswith("w13_weight"):
        if weight_scale_2.dim() == 0:
            weight_scale_2 = weight_scale_2.expand(weight.shape[0])
        if weight_scale_2.dim() == 1:
            return weight_scale_2[:, None].expand(-1, 2).contiguous()
        if weight_scale_2.dim() == 2 and weight_scale_2.shape[1] == 1:
            return weight_scale_2.expand(-1, 2).contiguous()
    if weight_name.endswith("w2_weight"):
        if weight_scale_2.dim() == 0:
            return weight_scale_2.expand(weight.shape[0]).contiguous()
        if weight_scale_2.dim() == 2 and weight_scale_2.shape[1] == 1:
            return weight_scale_2[:, 0].contiguous()
    return weight_scale_2


def _format_nvfp4_input_scale_for_export(
    source_name: str,
    weight_name: str,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
) -> torch.Tensor:
    """Shape static activation scales for dense and fused-MoE vLLM loaders."""
    if source_name.endswith(".experts.up_proj"):
        num_experts = weight.shape[0]
        if input_scale.numel() == 1:
            return input_scale.reshape(1, 1).expand(num_experts, 1).contiguous()
        if input_scale.shape == (num_experts,):
            return input_scale[:, None].contiguous()
        if input_scale.shape == (num_experts, 1):
            return input_scale.contiguous()
        raise RuntimeError(
            f"Expected one non-gated up-projection input scale per expert for "
            f"{weight_name}, got shape {tuple(input_scale.shape)}"
        )

    if weight_name.endswith("w13_weight"):
        num_experts = weight.shape[0]
        if input_scale.numel() == 1:
            expert_scales = input_scale.reshape(1).expand(num_experts)
        elif input_scale.shape == (num_experts,):
            expert_scales = input_scale
        elif input_scale.shape == (num_experts, 1):
            expert_scales = input_scale[:, 0]
        elif input_scale.shape == (num_experts, 2):
            return input_scale.contiguous()
        else:
            raise RuntimeError(
                f"Expected one or two W13 input scales per expert for {weight_name}, "
                f"got shape {tuple(input_scale.shape)}"
            )
        return expert_scales[:, None].expand(-1, 2).contiguous()

    if weight_name.endswith("w2_weight"):
        num_experts = weight.shape[0]
        if input_scale.numel() == 1:
            return input_scale.reshape(1).expand(num_experts).contiguous()
        if input_scale.shape == (num_experts,):
            return input_scale.contiguous()
        if input_scale.shape == (num_experts, 1):
            return input_scale[:, 0].contiguous()
        raise RuntimeError(
            f"Expected one input scale per expert for {weight_name}, got shape {tuple(input_scale.shape)}"
        )

    if input_scale.numel() != 1:
        raise RuntimeError(
            f"Expected one input scale for dense parameter {weight_name}, got shape {tuple(input_scale.shape)}"
        )
    return input_scale.reshape(())


def quantize_nvfp4_weight(
    name: str,
    weight: torch.Tensor,
    meta: QuantMeta,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield NVFP4 quantized weight tensors and associated scale tensors."""
    from modelopt.torch.export.quant_utils import QUANTIZATION_NVFP4, to_quantized_weight

    weight_name, weight_scale_name, weight_scale_2_name, input_scale_name = _nvfp4_export_names(name)
    weight_scale, weight_scale_2 = compute_nvfp4_weight_scale(
        weight,
        meta.block_size,
        weight_amax=meta.weight_amax,
        weight_scale_2=meta.weight_scale_2,
    )
    if not torch.isfinite(weight_scale_2).all() or not torch.all(weight_scale_2 > 0):
        raise RuntimeError(f"Invalid ModelOpt weight_scale_2 for quantized parameter {name}: {weight_scale_2}")
    weight_scale_2_for_quant = weight_scale_2
    if weight.dim() == 3:
        weight_scale_2_for_quant = _reshape_nvfp4_weight_scale_2_for_compute(
            weight,
            weight_scale_2,
        )
    if weight.dim() == 2 and weight_scale_2.numel() == 1:
        weight_scale_2_for_quant = weight_scale_2.reshape(())

    quantized = to_quantized_weight(
        weight,
        weight_scale,
        meta.qformat,
        weight_scale_2_for_quant,
        meta.block_size,
    )

    yield weight_name, quantized.detach()
    yield weight_scale_name, weight_scale.detach()
    yield (
        weight_scale_2_name,
        _format_nvfp4_weight_scale_2_for_export(
            name,
            weight_name,
            weight,
            weight_scale_2,
        ).detach(),
    )
    if meta.qformat == QUANTIZATION_NVFP4:
        input_scale = compute_nvfp4_input_scale(meta.input_amax).to(weight.device)
        yield (
            input_scale_name,
            _format_nvfp4_input_scale_for_export(
                name,
                weight_name,
                weight,
                input_scale,
            ).detach(),
        )


def get_modelopt_quant_exporter(quant_mode: str):
    """Return the ModelOpt quantization format and exporter for a quantization mode."""
    from modelopt.torch.export import quant_utils

    normalized_mode = quant_mode.lower()
    if normalized_mode == "nvfp4":
        qformat = quant_utils.QUANTIZATION_NVFP4
    elif normalized_mode == "w4a16_nvfp4":
        qformat = getattr(quant_utils, "QUANTIZATION_W4A16_NVFP4", None)
        if qformat is None:
            raise RuntimeError(
                "The installed nvidia-modelopt version does not support W4A16 NVFP4 export; "
                "install a version that exposes QUANTIZATION_W4A16_NVFP4."
            )
    else:
        raise ValueError(f"Unsupported ModelOpt quant_mode: {quant_mode}")
    return qformat, quantize_nvfp4_weight


def _grouped_expert_projection_name(hf_name: str) -> tuple[str, int] | None:
    """Remove a resolved expert index and terminal weight suffix from an HF name."""
    parts = hf_name.split(".")
    if parts and parts[-1] == "weight":
        parts.pop()
    if (
        len(parts) < 4
        or parts[-3] != "experts"
        or not parts[-2].isascii()
        or not parts[-2].isdecimal()
        or not parts[-1]
        or parts.count("experts") != 1
        or any(not part or "*" in part for part in parts)
    ):
        return None
    return ".".join((*parts[:-2], parts[-1])), int(parts[-2])


def _fuse_grouped_projection_names(gate_name: str, up_name: str) -> str | None:
    """Merge two projection leaves while preserving their common underscore suffix."""
    gate_parent, separator, gate_projection = gate_name.rpartition(".")
    up_parent, up_separator, up_projection = up_name.rpartition(".")
    if not separator or not up_separator or gate_parent != up_parent:
        return None

    gate_parts = gate_projection.split("_")
    up_parts = up_projection.split("_")
    if any(not part for part in (*gate_parts, *up_parts)):
        return None
    common_suffix = 0
    while (
        common_suffix < len(gate_parts)
        and common_suffix < len(up_parts)
        and gate_parts[-(common_suffix + 1)] == up_parts[-(common_suffix + 1)]
    ):
        common_suffix += 1
    if common_suffix == 0 or common_suffix == len(gate_parts) or common_suffix == len(up_parts):
        return None

    fused_projection = "_".join(
        (*gate_parts[:-common_suffix], *up_parts[:-common_suffix], *gate_parts[-common_suffix:])
    )
    return f"{gate_parent}.{fused_projection}"


class _LocalExpertMappingMixin:
    """Use expert TP while leaving the expert dimension local."""

    @property
    def tp_group(self):
        return self._etp_group

    @property
    def is_expert(self) -> bool:
        return False


class _LocalExpertColumnMapping(_LocalExpertMappingMixin, ColumnParallelMapping):
    pass


class _LocalExpertRowMapping(_LocalExpertMappingMixin, RowParallelMapping):
    pass


class _LocalExpertReplicatedMapping(_LocalExpertMappingMixin, ReplicatedMapping):
    pass


class _LocalExpertGatedMLPMapping(_LocalExpertMappingMixin, GatedMLPMapping):
    pass


class _ModelOptFusedExpertMapping(FusedExpertMapping):
    """Build one E/EP-local expert batch without a BF16 EP gather."""

    is_grouped_export = False
    is_modelopt_pre_ep_export = True

    def _get_or_create_mapping(self, parallelism_type: str):
        mapping_types = {
            "column": _LocalExpertColumnMapping,
            "row": _LocalExpertRowMapping,
            "replicated": _LocalExpertReplicatedMapping,
        }
        try:
            mapping_type = mapping_types[parallelism_type]
        except KeyError as error:
            raise ValueError(f"Unknown parallelism type: {parallelism_type}") from error
        mapping = mapping_type(self.megatron_param, self.hf_param)
        mapping.set_process_groups_from_pg_collection(self._pg_collection)
        return mapping


class _ModelOptFusedGatedExpertMapping(FusedGatedExpertMapping):
    """Fuse gate/up locally without a BF16 EP gather."""

    is_grouped_export = False
    is_modelopt_pre_ep_export = True

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        permute_dims: tuple[int, ...] | None = None,
        transpose_on_export: bool = False,
    ) -> None:
        super().__init__(megatron_param, hf_param, permute_dims, transpose_on_export)
        self._gated_mapping = _LocalExpertGatedMLPMapping(
            megatron_param=self.megatron_param,
            gate=f"{self.hf_param}.gate",
            up=f"{self.hf_param}.up",
        )


def _modelopt_pre_ep_mapping(
    mapping: Any,
    pg_collection: Any = None,
) -> tuple[Any, tuple[str, ...]] | None:
    """Build a fused local-expert mapping for ModelOpt expert projections."""
    if type(mapping) is GatedMLPMapping and isinstance(mapping.hf_param, dict):
        gate_name = mapping.hf_param.get("gate")
        up_name = mapping.hf_param.get("up")
        if isinstance(gate_name, str) and isinstance(up_name, str):
            grouped_gate = _grouped_expert_projection_name(gate_name)
            grouped_up = _grouped_expert_projection_name(up_name)
            if grouped_gate is not None and grouped_up is not None and grouped_gate[1] == grouped_up[1]:
                grouped_name = _fuse_grouped_projection_names(grouped_gate[0], grouped_up[0])
            else:
                grouped_name = None
            if grouped_name is not None and is_modelopt_quantizable_weight_name(grouped_name):
                replacement = _ModelOptFusedGatedExpertMapping(
                    mapping.megatron_param,
                    grouped_name,
                )
                replacement.set_process_groups_from_pg_collection(pg_collection)
                return replacement, (gate_name, up_name)

    if type(mapping) is not AutoMapping or not isinstance(mapping.hf_param, str):
        return None

    grouped_projection = _grouped_expert_projection_name(mapping.hf_param)
    if grouped_projection is None or not is_modelopt_quantizable_weight_name(grouped_projection[0]):
        return None

    replacement = _ModelOptFusedExpertMapping(
        mapping.megatron_param,
        grouped_projection[0],
        mapping.permute_dims,
    )
    replacement.set_process_groups_from_pg_collection(pg_collection)
    return replacement, (mapping.hf_param,)


def _stage_tensor_for_collective(tensor: torch.Tensor, group: Any) -> torch.Tensor:
    """Move a CPU tensor to CUDA only when its collective backend requires it."""
    backend = str(torch.distributed.get_backend(group)).lower()
    if backend != "nccl" or tensor.device.type != "cpu":
        return tensor
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL ModelOpt expert gather requires CUDA")
    return tensor.to(
        device=torch.device("cuda", torch.cuda.current_device()),
        non_blocking=tensor.is_pinned(),
    )


def _compose_export_hooks(exporter: HFExportHook, finalizer: HFExportHook | None) -> HFExportHook:
    if finalizer is None:
        return exporter

    def export_and_finalize(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        for exported_name, exported_tensor in exporter(hf_name, tensor):
            yield from finalizer(exported_name, exported_tensor)

    return export_and_finalize


def build_modelopt_export_plan(
    conversion_tasks: list[WeightConversionTask],
    *,
    model: list[torch.nn.Module],
    bridge: "MegatronModelBridge",
    quant_mode: str,
    ignore_patterns: list[str],
) -> list[WeightConversionTask]:
    """Prepare mapped conversion tasks and hooks for a ModelOpt export."""
    expected_qformat, export_weight = get_modelopt_quant_exporter(quant_mode)
    pg_collection = model_bridge_utils._get_pg_collection_from_model(model)
    local_metadata = collect_modelopt_quant_metadata(conversion_tasks)
    if torch.distributed.is_initialized():
        pp_group = model_bridge_utils._get_pp_group(model)
        ep_group = model_bridge_utils._get_ep_group(model)
    else:
        pp_group = None
        ep_group = None

    # PP stages need metadata for their EP-local experts, but pre-EP packing
    # must retain the E/EP-local view rather than a full global-E copy.
    pp_world_size = get_pg_size(pp_group)
    ep_world_size = pp_world_size if ep_group is pp_group else get_pg_size(ep_group)
    if pp_world_size > 1:
        sync_modelopt_quant_metadata(local_metadata, pp_group)

    # A shared non-singleton PP/EP group has already globalized metadata, so it
    # cannot provide the local view required by pre-EP packing.
    can_export_before_ep = not (pp_world_size > 1 and pp_group is not None and ep_group is pp_group)
    candidate_mappings: dict[int, Any] = {}
    candidate_groups: dict[str, list[int]] = {}
    eligible_groups: dict[str, bool] = {}
    for task_idx, task in enumerate(conversion_tasks):
        candidate = _modelopt_pre_ep_mapping(task.mapping, pg_collection) if can_export_before_ep else None
        if candidate is None:
            continue
        replacement, original_hf_names = candidate
        group_key = str(replacement.hf_param)
        candidate_mappings[task_idx] = replacement
        candidate_groups.setdefault(group_key, []).append(task_idx)

        meta = local_metadata.get(task.global_param_name)
        task_is_eligible = (
            meta is not None
            and meta.qformat == expected_qformat
            and not any(matches_quant_ignore_pattern(name, ignore_patterns) for name in original_hf_names)
        )
        eligible_groups[group_key] = eligible_groups.get(group_key, True) and task_is_eligible

    experts_per_rank = 0
    if candidate_groups:
        unwrapped_model = model_bridge_utils.unwrap_model(model)[0]
        num_experts = getattr(unwrapped_model.config, "num_moe_experts", None)
        valid_expert_layout = isinstance(num_experts, int) and num_experts > 0 and num_experts % ep_world_size == 0
        experts_per_rank = num_experts // ep_world_size if valid_expert_layout else 0
        expected_local_ids = set(range(experts_per_rank))
        for group_key, task_indices in candidate_groups.items():
            local_ids: list[int] = []
            if valid_expert_layout:
                try:
                    for task_idx in task_indices:
                        candidate_task = conversion_tasks[task_idx]
                        local_ids.append(
                            extract_expert_number_from_param(candidate_task.param_name) % experts_per_rank
                        )
                except ValueError:
                    local_ids = []
            eligible_groups[group_key] = eligible_groups[group_key] and (
                valid_expert_layout and len(local_ids) == experts_per_rank and set(local_ids) == expected_local_ids
            )

    if ep_world_size > 1:
        gathered_eligibility: list[dict[str, bool] | None] = [None] * ep_world_size
        torch.distributed.all_gather_object(gathered_eligibility, eligible_groups, group=ep_group)
        all_group_keys = set().union(*(rank_groups or {} for rank_groups in gathered_eligibility))
        eligible_groups = {
            group_key: all(bool((rank_groups or {}).get(group_key, False)) for rank_groups in gathered_eligibility)
            for group_key in all_group_keys
        }

    mapped_tasks = list(conversion_tasks)
    for group_key, task_indices in candidate_groups.items():
        if not eligible_groups[group_key]:
            continue
        for task_idx in task_indices:
            mapped_tasks[task_idx] = replace(
                conversion_tasks[task_idx],
                mapping=candidate_mappings[task_idx],
            )

    regular_metadata_tasks: list[WeightConversionTask] = []
    pre_ep_tasks: list[WeightConversionTask] = []
    for original_task, mapped_task in zip(conversion_tasks, mapped_tasks, strict=True):
        if getattr(mapped_task.mapping, "is_modelopt_pre_ep_export", False):
            pre_ep_tasks.append(mapped_task)
        else:
            regular_metadata_tasks.append(original_task)

    regular_metadata_names = {task.global_param_name for task in regular_metadata_tasks}
    pre_ep_only_metadata_names = {task.global_param_name for task in pre_ep_tasks}.difference(regular_metadata_names)
    regular_metadata = {name: meta for name, meta in local_metadata.items() if name not in pre_ep_only_metadata_names}
    if ep_world_size > 1 and ep_group is not pp_group:
        sync_modelopt_quant_metadata(regular_metadata, ep_group)

    hf_metadata = build_hf_modelopt_quant_metadata(regular_metadata_tasks, regular_metadata)
    pre_ep_hf_metadata = _build_hf_modelopt_pre_ep_quant_metadata(pre_ep_tasks, local_metadata)
    hf_metadata.update(pre_ep_hf_metadata)
    pre_ep_hf_names = set(pre_ep_hf_metadata)
    mapping_registry = None

    def modelopt_export_weight(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        nonlocal mapping_registry
        if "_quantizer." in hf_name:
            return

        meta = None
        if is_modelopt_quantizable_weight_name(hf_name) and (
            hf_name in pre_ep_hf_names or not matches_quant_ignore_pattern(hf_name, ignore_patterns)
        ):
            meta = hf_metadata.get(hf_name)
            if meta is None and regular_metadata:
                if mapping_registry is None:
                    mapping_registry = bridge.mapping_registry()
                    mapping_registry.set_process_groups_from_pg_collection(pg_collection)
                mapping = mapping_registry.hf_to_megatron_lookup(hf_name)
                if mapping is not None:
                    meta = regular_metadata.get(mapping.megatron_param)

        if meta is None:
            yield HFWeightTuple(hf_name, tensor)
            return
        if meta.qformat != expected_qformat:
            raise RuntimeError(f"Unsupported qformat for ModelOpt {quant_mode} export: {meta.qformat}")
        for quant_name, quant_tensor in export_weight(hf_name, tensor, meta):
            yield HFWeightTuple(quant_name, quant_tensor)

    def finalize_ep_weight(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        if get_pg_size(ep_group) == 1:
            yield HFWeightTuple(hf_name, tensor)
            return

        local_tensor = tensor.contiguous()
        if local_tensor.ndim == 0:
            raise ValueError("Pre-EP ModelOpt tensor must have an expert dimension")
        collective_tensor = _stage_tensor_for_collective(local_tensor, ep_group)
        local_bytes = collective_tensor.reshape(-1).view(torch.uint8)
        world_size = get_pg_size(ep_group)
        gathered_bytes = torch.empty(
            world_size * local_bytes.numel(),
            dtype=torch.uint8,
            device=collective_tensor.device,
        )
        torch.distributed.all_gather_into_tensor(gathered_bytes, local_bytes, group=ep_group)
        global_shape = (world_size * local_tensor.shape[0], *local_tensor.shape[1:])
        yield HFWeightTuple(
            hf_name,
            gathered_bytes.view(local_tensor.dtype).reshape(global_shape),
        )

    pre_ep_buffers: dict[str, dict[int, torch.Tensor]] = {}

    def build_pre_ep_hook(task: WeightConversionTask) -> HFExportHook:
        local_expert_id = extract_expert_number_from_param(task.param_name) % experts_per_rank

        def export_local_expert(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
            expert_tensors = pre_ep_buffers.setdefault(hf_name, {})
            if local_expert_id in expert_tensors:
                raise RuntimeError(f"Duplicate local expert {local_expert_id} for {hf_name}")
            expert_tensors[local_expert_id] = tensor
            if len(expert_tensors) != experts_per_rank:
                return

            missing = set(range(experts_per_rank)).difference(expert_tensors)
            if missing:
                raise RuntimeError(f"Missing local experts {sorted(missing)} for {hf_name}")
            local_batch = torch.stack(
                [expert_tensors[expert_id] for expert_id in range(experts_per_rank)],
                dim=0,
            )
            del pre_ep_buffers[hf_name]
            for exported_name, exported_tensor in modelopt_export_weight(hf_name, local_batch):
                yield from finalize_ep_weight(exported_name, exported_tensor)

        return export_local_expert

    regular_hook = _compose_export_hooks(modelopt_export_weight, None)
    export_tasks = []
    for task in mapped_tasks:
        export_hook = (
            build_pre_ep_hook(task) if getattr(task.mapping, "is_modelopt_pre_ep_export", False) else regular_hook
        )
        export_tasks.append(replace(task, export_hook=export_hook))
    return export_tasks
