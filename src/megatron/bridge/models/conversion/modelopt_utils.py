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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch
from megatron.core.transformer.moe.router import Router
from megatron.core.utils import get_pg_size

from megatron.bridge.models.conversion import model_bridge as model_bridge_utils
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask


if TYPE_CHECKING:
    from modelopt.torch.export.quantized_weight import QuantizedWeightExportState
else:
    QuantizedWeightExportState = Any

HFExportHook = Callable[[str, torch.Tensor], Iterable[HFWeightTuple]]
GroupedHFExportHook = Callable[[str, torch.Tensor, int], Iterable[HFWeightTuple]]
GroupedQuantizedWeightState = tuple[QuantizedWeightExportState, ...]
MappedQuantizedWeightState = QuantizedWeightExportState | GroupedQuantizedWeightState

_EXPERT_NUMBER_PATTERNS = (
    re.compile(r"(local_experts\.)(\d+)(\.)"),
    re.compile(r"((?:weight|bias))(\d+)(?=$|\.)"),
    re.compile(r"(experts\.)(\d+)(\.)"),
)


@dataclass(frozen=True)
class ModelOptExportPlan:
    """Prepared ModelOpt conversion tasks and their canonical HF config."""

    conversion_tasks: list[WeightConversionTask]
    quantization_config: dict[str, Any]


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


def _is_enabled_quantizer(quantizer: object) -> bool:
    return bool(getattr(quantizer, "is_enabled", False))


def _weight_name(module: torch.nn.Module, weight: torch.Tensor) -> str:
    for name, candidate in module.named_parameters(recurse=False):
        if _is_same_tensor(candidate, weight):
            return name
    raise RuntimeError(f"Cannot identify the parameter name for a quantized weight owned by {type(module).__name__}")


def _input_quantizer(module: torch.nn.Module, weight_name: str) -> object | None:
    from modelopt.torch.quantization.utils import quantizer_attr_names

    quantizer = getattr(module, quantizer_attr_names(weight_name).input_quantizer, None)
    if quantizer is None and weight_name != "weight":
        quantizer = getattr(module, "input_quantizer", None)
    return quantizer


def _iter_modelopt_weight_quantizers(
    module: torch.nn.Module,
    *,
    enabled_only: bool = True,
) -> Iterator[tuple[torch.Tensor, str, object, object | None]]:
    iter_weights = getattr(module, "iter_weights_for_calibration", None)
    if callable(iter_weights):
        for weight, weight_quantizer in iter_weights():
            if enabled_only and not _is_enabled_quantizer(weight_quantizer):
                continue
            weight_name = _weight_name(module, weight)
            yield (
                weight,
                weight_name,
                weight_quantizer,
                _input_quantizer(module, weight_name),
            )
        return

    weight_quantizer = getattr(module, "weight_quantizer", None)
    if weight_quantizer is None or (enabled_only and not _is_enabled_quantizer(weight_quantizer)):
        return
    for weight_name, weight in module.named_parameters(recurse=False):
        if weight_name == "weight" or (weight_name.startswith("weight") and weight_name[6:].isdigit()):
            yield weight, weight_name, weight_quantizer, _input_quantizer(module, weight_name)


def find_modelopt_quantizers(
    module: torch.nn.Module,
    param_weight: object,
    *,
    enabled_only: bool = True,
) -> tuple[torch.nn.Module, str, object, object | None] | None:
    """Find the exact ModelOpt quantizers that own ``param_weight``."""
    for _, candidate_module in module.named_modules():
        for weight, weight_name, weight_quantizer, input_quantizer in _iter_modelopt_weight_quantizers(
            candidate_module,
            enabled_only=enabled_only,
        ):
            if _is_same_tensor(param_weight, weight):
                return candidate_module, weight_name, weight_quantizer, input_quantizer
    return None


def _get_modelopt_tp_process_group(module: object) -> object | None:
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

    world_size = getattr(tp_group, "world_size", None)
    world_size = world_size() if callable(world_size) else torch.distributed.get_world_size(group=process_group)
    return process_group if world_size > 1 else None


def _max_reduce_modelopt_tp_scalar(
    value: torch.Tensor | None,
    module: object,
    field_name: str,
) -> torch.Tensor | None:
    if value is None:
        return None
    if value.numel() != 1:
        raise RuntimeError(f"Cannot TP-synchronize non-scalar ModelOpt {field_name} with shape {tuple(value.shape)}")

    process_group = _get_modelopt_tp_process_group(module)
    reduced = value.detach().clone()
    if process_group is not None:
        torch.distributed.all_reduce(
            reduced,
            op=torch.distributed.ReduceOp.MAX,
            group=process_group,
        )
    return reduced.detach().cpu().float()


def collect_modelopt_config_weights(
    conversion_tasks: list[WeightConversionTask | None],
) -> set[str]:
    """Return weight tasks represented in ModelOpt's deployment config."""
    config_weights = set()
    for task in conversion_tasks:
        if task is None or task.megatron_module is None or task.param_weight is None:
            continue
        has_quantizer = (
            find_modelopt_quantizers(
                task.megatron_module,
                task.param_weight,
                enabled_only=False,
            )
            is not None
        )
        if has_quantizer or isinstance(task.megatron_module, Router):
            config_weights.add(task.global_param_name)
    return config_weights


def collect_modelopt_quant_states(
    conversion_tasks: list[WeightConversionTask | None],
) -> dict[str, QuantizedWeightExportState]:
    """Capture scalar ModelOpt export state for every quantized Megatron weight."""
    from modelopt.torch.export.quantized_weight import (
        capture_quantized_weight_export_state,
    )

    states: dict[str, QuantizedWeightExportState] = {}
    cached_states: dict[tuple[int, int, str], QuantizedWeightExportState] = {}
    for task in conversion_tasks:
        if task is None or task.megatron_module is None or task.param_weight is None:
            continue

        found = find_modelopt_quantizers(task.megatron_module, task.param_weight)
        if found is None:
            continue
        module, weight_name, weight_quantizer, input_quantizer = found
        cache_key = (id(weight_quantizer), id(input_quantizer), weight_name)
        if cache_key in cached_states:
            states[task.global_param_name] = cached_states[cache_key]
            continue

        state = capture_quantized_weight_export_state(
            module,
            weight_name,
            weight_quantizer=weight_quantizer,
            input_quantizer=input_quantizer,
        )
        weight_amax = _max_reduce_modelopt_tp_scalar(
            state.weight_amax,
            module,
            "weight amax",
        )
        if weight_amax is None:
            raise RuntimeError(f"Missing ModelOpt weight amax for {task.global_param_name}")
        state = replace(
            state,
            weight_amax=weight_amax,
            input_amax=_max_reduce_modelopt_tp_scalar(
                state.input_amax,
                module,
                "input amax",
            ),
        )
        cached_states[cache_key] = state
        states[task.global_param_name] = state
    return states


def sync_modelopt_quant_states(
    states: dict[str, QuantizedWeightExportState],
    group: object | None = None,
) -> None:
    """Synchronize quantized-weight states across a distributed group."""
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, QuantizedWeightExportState] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, states, group=group)
    for rank_states in gathered:
        if not rank_states:
            continue
        for name, state in rank_states.items():
            existing = states.get(name)
            if existing is not None and not _same_quant_state(existing, state):
                raise RuntimeError(f"Conflicting ModelOpt state for {name}")
            states[name] = state


def _same_quant_state(
    lhs: QuantizedWeightExportState,
    rhs: QuantizedWeightExportState,
) -> bool:
    return (
        lhs.quantization_format == rhs.quantization_format
        and lhs.block_size == rhs.block_size
        and torch.equal(lhs.weight_amax, rhs.weight_amax)
        and (
            lhs.input_amax is rhs.input_amax
            or (
                lhs.input_amax is not None
                and rhs.input_amax is not None
                and torch.equal(lhs.input_amax, rhs.input_amax)
            )
        )
    )


def _expert_param_template(param_name: str) -> str | None:
    for pattern in _EXPERT_NUMBER_PATTERNS:
        match = pattern.search(param_name)
        if match is not None:
            return f"{param_name[: match.start(2)]}{{expert}}{param_name[match.end(2) :]}"
    return None


def _group_quant_states_by_expert(
    states: dict[str, QuantizedWeightExportState],
) -> dict[str, dict[int, QuantizedWeightExportState]]:
    from megatron.bridge.utils.common_utils import extract_expert_number_from_param

    grouped: dict[str, dict[int, QuantizedWeightExportState]] = {}
    for global_name, state in states.items():
        template = _expert_param_template(global_name)
        if template is not None:
            grouped.setdefault(template, {})[extract_expert_number_from_param(global_name)] = state
    return grouped


def _ordered_grouped_states(
    hf_name: str,
    expert_states: dict[int, QuantizedWeightExportState],
    num_experts: int,
) -> GroupedQuantizedWeightState:
    if not expert_states:
        raise RuntimeError(f"Missing ModelOpt state for grouped parameter {hf_name}")
    expected_experts = set(range(num_experts))
    missing_experts = sorted(expected_experts.difference(expert_states))
    if missing_experts:
        raise RuntimeError(f"Missing ModelOpt state for experts {missing_experts} of grouped parameter {hf_name}")
    unexpected_experts = sorted(set(expert_states).difference(expected_experts))
    if unexpected_experts:
        raise RuntimeError(
            f"Unexpected ModelOpt state for experts {unexpected_experts} of grouped parameter {hf_name}"
        )
    ordered = tuple(expert_states[index] for index in range(num_experts))
    first = ordered[0]
    if any(
        state.quantization_format != first.quantization_format or state.block_size != first.block_size
        for state in ordered[1:]
    ):
        raise RuntimeError(f"Inconsistent ModelOpt state for grouped parameter {hf_name}")
    return ordered


def _hf_weight_names(task: WeightConversionTask) -> tuple[str, ...]:
    hf_param = task.mapping.hf_param
    if isinstance(hf_param, str):
        return (hf_param,)
    return tuple(hf_param.values())


def build_hf_modelopt_quant_states(
    conversion_tasks: list[WeightConversionTask],
    states: dict[str, QuantizedWeightExportState],
    *,
    num_experts: int | None = None,
) -> dict[str, MappedQuantizedWeightState]:
    """Map Megatron quantized-weight states onto canonical HF weight names."""
    hf_states: dict[str, MappedQuantizedWeightState] = {}
    grouped_by_template = _group_quant_states_by_expert(states)
    for task in conversion_tasks:
        state = states.get(task.global_param_name)
        if state is None:
            continue
        if getattr(task.mapping, "is_grouped_export", False):
            if num_experts is None:
                raise RuntimeError("Grouped ModelOpt export requires model.config.num_moe_experts")
            template = _expert_param_template(task.global_param_name)
            if template is None:
                raise ValueError(f"Expected expert parameter name for grouped export: {task.global_param_name}")
            expert_states = grouped_by_template.get(template, {})
            for hf_name in _hf_weight_names(task):
                _set_hf_quant_state(
                    hf_states,
                    hf_name,
                    _ordered_grouped_states(hf_name, expert_states, num_experts),
                )
            continue
        for hf_name in _hf_weight_names(task):
            _set_hf_quant_state(hf_states, hf_name, state)
    return hf_states


def _set_hf_quant_state(
    hf_states: dict[str, MappedQuantizedWeightState],
    hf_name: str,
    state: MappedQuantizedWeightState,
) -> None:
    existing = hf_states.get(hf_name)
    if existing is None:
        hf_states[hf_name] = state
        return
    if isinstance(existing, tuple) != isinstance(state, tuple):
        raise RuntimeError(f"Conflicting ModelOpt state layouts for {hf_name}")
    existing_states = existing if isinstance(existing, tuple) else (existing,)
    new_states = state if isinstance(state, tuple) else (state,)
    if len(existing_states) != len(new_states) or any(
        not _same_quant_state(lhs, rhs) for lhs, rhs in zip(existing_states, new_states, strict=True)
    ):
        raise RuntimeError(f"Conflicting ModelOpt states for {hf_name}")


def sync_hf_modelopt_quant_states(
    hf_states: dict[str, MappedQuantizedWeightState],
    group: object | None = None,
) -> None:
    """Synchronize canonical HF quantization states across a distributed group."""
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, MappedQuantizedWeightState] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, hf_states, group=group)
    for rank_states in gathered:
        if not rank_states:
            continue
        for name, state in rank_states.items():
            _set_hf_quant_state(hf_states, name, state)


def _is_weight_task(task: WeightConversionTask) -> bool:
    return re.search(r"(?:^|\.)weight\d*$", task.global_param_name) is not None


def _quantization_layer_name(hf_weight_name: str) -> str:
    return hf_weight_name.removesuffix(".weight")


def build_modelopt_quantization_config(
    conversion_tasks: list[WeightConversionTask],
    hf_states: dict[str, MappedQuantizedWeightState],
    *,
    config_weights: set[str],
    sync_groups: Iterable[object | None] = (),
) -> dict[str, Any]:
    """Build the canonical ModelOpt HF config for mapped conversion tasks."""
    from modelopt.torch.export.quantized_weight import build_hf_quantization_config

    layer_states: dict[str, QuantizedWeightExportState | None] = {}
    for task in conversion_tasks:
        if not _is_weight_task(task) or task.global_param_name not in config_weights:
            continue
        for hf_name in _hf_weight_names(task):
            mapped_state = hf_states.get(hf_name)
            if isinstance(mapped_state, tuple):
                mapped_state = mapped_state[0]
            layer_name = _quantization_layer_name(hf_name)
            if layer_name not in layer_states or layer_states[layer_name] is None:
                layer_states[layer_name] = mapped_state
            elif mapped_state is not None and not _same_quant_state(
                layer_states[layer_name],
                mapped_state,
            ):
                raise RuntimeError(f"Conflicting ModelOpt configs for {layer_name}")
    for group in sync_groups:
        _sync_modelopt_layer_states(layer_states, group)
    if not any(state is not None for state in layer_states.values()):
        raise RuntimeError("No ModelOpt-quantized weights were found in the conversion tasks")
    return build_hf_quantization_config(layer_states)


def _sync_modelopt_layer_states(
    layer_states: dict[str, QuantizedWeightExportState | None],
    group: object | None,
) -> None:
    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, QuantizedWeightExportState | None] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, layer_states, group=group)
    for rank_states in gathered:
        if not rank_states:
            continue
        for name, state in rank_states.items():
            if name not in layer_states or layer_states[name] is None:
                layer_states[name] = state
            elif state is not None and not _same_quant_state(layer_states[name], state):
                raise RuntimeError(f"Conflicting ModelOpt configs for {name}")


def _named_export_tensors(
    hf_name: str,
    state: QuantizedWeightExportState,
    weight: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor]]:
    from modelopt.torch.export.quantized_weight import export_quantized_weight

    prefix, separator, weight_name = hf_name.rpartition(".")
    exported = export_quantized_weight(weight, state, dtype=weight.dtype)
    for relative_name, tensor in exported.named_tensors(weight_name).items():
        yield f"{prefix}{separator}{relative_name}", tensor.detach()


def _compose_export_hooks(exporter: HFExportHook, finalizer: HFExportHook | None) -> HFExportHook:
    if finalizer is None:
        return exporter

    def export_and_finalize(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        for exported_name, exported_tensor in exporter(hf_name, tensor):
            yield from finalizer(exported_name, exported_tensor)

    return export_and_finalize


def _compose_grouped_export_hooks(
    exporter: GroupedHFExportHook,
    finalizer: GroupedHFExportHook | None,
) -> GroupedHFExportHook:
    if finalizer is None:
        return exporter

    def export_and_finalize(
        hf_name: str,
        tensor: torch.Tensor,
        expert_number: int,
    ) -> Iterable[HFWeightTuple]:
        for exported_name, exported_tensor in exporter(hf_name, tensor, expert_number):
            yield from finalizer(exported_name, exported_tensor, expert_number)

    return export_and_finalize


def build_modelopt_export_plan(
    conversion_tasks: list[WeightConversionTask | None],
    *,
    model: list[torch.nn.Module],
) -> ModelOptExportPlan:
    """Prepare topology-aware conversion tasks for ModelOpt HF export."""
    concrete_tasks = [task for task in conversion_tasks if task is not None]
    config_weights = collect_modelopt_config_weights(conversion_tasks)
    states = collect_modelopt_quant_states(conversion_tasks)
    sync_groups: list[object | None] = []
    if torch.distributed.is_initialized():
        pp_group = model_bridge_utils._get_pp_group(model)
        ep_group = model_bridge_utils._get_ep_group(model)
        if get_pg_size(pp_group) > 1:
            sync_modelopt_quant_states(states, pp_group)
            sync_groups.append(pp_group)
        if get_pg_size(ep_group) > 1 and ep_group is not pp_group:
            sync_modelopt_quant_states(states, ep_group)
            sync_groups.append(ep_group)

    model_config = model_bridge_utils.unwrap_model(model)[0].config
    hf_states = build_hf_modelopt_quant_states(
        concrete_tasks,
        states,
        num_experts=getattr(model_config, "num_moe_experts", None),
    )
    for group in sync_groups:
        sync_hf_modelopt_quant_states(hf_states, group)

    def export_weight(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        state = hf_states.get(hf_name)
        if state is None:
            yield HFWeightTuple(hf_name, tensor)
        elif isinstance(state, tuple):
            raise RuntimeError(f"Grouped ModelOpt weight {hf_name} was not exported per expert")
        else:
            for name, exported_tensor in _named_export_tensors(hf_name, state, tensor):
                yield HFWeightTuple(name, exported_tensor)

    def export_grouped_weight(
        hf_name: str,
        tensor: torch.Tensor,
        expert_number: int,
    ) -> Iterable[HFWeightTuple]:
        states = hf_states.get(hf_name)
        if not isinstance(states, tuple):
            raise RuntimeError(f"Missing grouped ModelOpt state for {hf_name}")
        if expert_number >= len(states):
            raise RuntimeError(f"Missing ModelOpt state for expert {expert_number} of grouped parameter {hf_name}")
        for name, exported_tensor in _named_export_tensors(
            hf_name,
            states[expert_number],
            tensor,
        ):
            yield HFWeightTuple(name, exported_tensor)

    export_tasks = []
    for task in concrete_tasks:
        is_grouped = getattr(task.mapping, "is_grouped_export", False)
        has_grouped_state = is_grouped and any(
            isinstance(hf_states.get(hf_name), tuple) for hf_name in _hf_weight_names(task)
        )
        if has_grouped_state:
            export_tasks.append(
                replace(
                    task,
                    grouped_export_hook=_compose_grouped_export_hooks(
                        export_grouped_weight,
                        task.grouped_export_hook,
                    ),
                )
            )
        else:
            export_tasks.append(
                replace(
                    task,
                    export_hook=_compose_export_hooks(export_weight, task.export_hook),
                )
            )
    return ModelOptExportPlan(
        conversion_tasks=export_tasks,
        quantization_config=build_modelopt_quantization_config(
            concrete_tasks,
            hf_states,
            config_weights=config_weights,
            sync_groups=sync_groups,
        ),
    )
