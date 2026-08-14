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
from typing import Any

import torch
from megatron.core.transformer.moe.router import Router
from megatron.core.utils import get_pg_size

from megatron.bridge.models.conversion import model_bridge as model_bridge_utils
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
    RowParallelMapping,
)
from megatron.bridge.utils.common_utils import extract_expert_number_from_param


HFExportHook = Callable[[str, torch.Tensor], Iterable[HFWeightTuple]]
GroupedHFExportHook = Callable[[str, torch.Tensor, int], Iterable[HFWeightTuple]]
QuantizedWeightState = object
GroupedQuantizedWeightState = tuple[QuantizedWeightState, ...]
MappedQuantizedWeightState = QuantizedWeightState | GroupedQuantizedWeightState

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

    # Preserve an explicit TP-size-1 group: ModelOpt treats ``group=None`` as
    # the world group, which would mix calibration state across PP/EP ranks.
    return process_group


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
) -> dict[str, QuantizedWeightState]:
    """Capture scalar ModelOpt export state for every quantized Megatron weight."""
    from modelopt.torch.export.quantized_weight import (
        capture_quantized_weight_export_state,
        synchronize_quantized_weight_export_state,
    )

    states: dict[str, QuantizedWeightState] = {}
    cached_states: dict[tuple[int, int, str], QuantizedWeightState] = {}
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
        state = synchronize_quantized_weight_export_state(
            state,
            group=_get_modelopt_tp_process_group(module),
            device="cpu",
        )
        cached_states[cache_key] = state
        states[task.global_param_name] = state
    return states


def sync_modelopt_quant_states(
    states: dict[str, QuantizedWeightState],
    group: object | None = None,
) -> None:
    """Synchronize quantized-weight states across a distributed group."""
    world_size = torch.distributed.get_world_size(group=group)
    from modelopt.torch.export.quantized_weight import quantized_weight_export_states_equal

    gathered: list[dict[str, QuantizedWeightState] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, states, group=group)
    for rank_states in gathered:
        if not rank_states:
            continue
        for name, state in rank_states.items():
            existing = states.get(name)
            if existing is not None and not quantized_weight_export_states_equal(existing, state):
                raise RuntimeError(f"Conflicting ModelOpt state for {name}")
            states[name] = state


def _expert_param_template(param_name: str) -> str | None:
    for pattern in _EXPERT_NUMBER_PATTERNS:
        match = pattern.search(param_name)
        if match is not None:
            return f"{param_name[: match.start(2)]}{{expert}}{param_name[match.end(2) :]}"
    return None


def _group_quant_states_by_expert(
    states: dict[str, QuantizedWeightState],
) -> dict[str, dict[int, QuantizedWeightState]]:
    grouped: dict[str, dict[int, QuantizedWeightState]] = {}
    for global_name, state in states.items():
        template = _expert_param_template(global_name)
        if template is not None:
            grouped.setdefault(template, {})[extract_expert_number_from_param(global_name)] = state
    return grouped


def _ordered_grouped_states(
    hf_name: str,
    expert_states: dict[int, QuantizedWeightState],
    num_experts: int,
) -> GroupedQuantizedWeightState:
    from modelopt.torch.export.quantized_weight import (
        quantized_weight_export_states_compatible,
    )

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
    if any(not quantized_weight_export_states_compatible(first, state) for state in ordered[1:]):
        raise RuntimeError(f"Inconsistent ModelOpt state for grouped parameter {hf_name}")
    return ordered


def _hf_weight_names(task: WeightConversionTask) -> tuple[str, ...]:
    hf_param = task.mapping.hf_param
    if isinstance(hf_param, str):
        return (hf_param,)
    return tuple(hf_param.values())


def build_hf_modelopt_quant_states(
    conversion_tasks: list[WeightConversionTask],
    states: dict[str, QuantizedWeightState],
    *,
    num_experts: int | None = None,
) -> dict[str, MappedQuantizedWeightState]:
    """Map Megatron quantized-weight states onto canonical HF weight names."""
    from modelopt.torch.export.quantized_weight import (
        replicate_quantized_weight_export_state,
    )

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
            hf_names = _hf_weight_names(task)
            grouped_states = _ordered_grouped_states(hf_names[0], grouped_by_template.get(template, {}), num_experts)
            replicated_by_expert = [
                replicate_quantized_weight_export_state(expert_state, len(hf_names)) for expert_state in grouped_states
            ]
            for hf_index, hf_name in enumerate(hf_names):
                _set_hf_quant_state(
                    hf_states,
                    hf_name,
                    tuple(states[hf_index] for states in replicated_by_expert),
                )
            continue
        hf_names = _hf_weight_names(task)
        for hf_name, projected_state in zip(
            hf_names,
            replicate_quantized_weight_export_state(state, len(hf_names)),
            strict=True,
        ):
            _set_hf_quant_state(hf_states, hf_name, projected_state)
    return hf_states


def _set_hf_quant_state(
    hf_states: dict[str, MappedQuantizedWeightState],
    hf_name: str,
    state: MappedQuantizedWeightState,
) -> None:
    from modelopt.torch.export.quantized_weight import quantized_weight_export_states_equal

    existing = hf_states.get(hf_name)
    if existing is None:
        hf_states[hf_name] = state
        return
    if isinstance(existing, tuple) != isinstance(state, tuple):
        raise RuntimeError(f"Conflicting ModelOpt state layouts for {hf_name}")
    existing_states = existing if isinstance(existing, tuple) else (existing,)
    new_states = state if isinstance(state, tuple) else (state,)
    if len(existing_states) != len(new_states) or any(
        not quantized_weight_export_states_equal(lhs, rhs)
        for lhs, rhs in zip(existing_states, new_states, strict=True)
    ):
        raise RuntimeError(f"Conflicting ModelOpt states for {hf_name}")


def _is_weight_task(task: WeightConversionTask) -> bool:
    return re.search(r"(?:^|\.)weight\d*$", task.global_param_name) is not None


def _is_quantizer_state_task(task: WeightConversionTask) -> bool:
    """Return whether a task carries ModelOpt's internal quantizer state."""
    return any(segment.endswith(("quantizer", "quantizers")) for segment in task.global_param_name.split("."))


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
    from modelopt.torch.export.quantized_weight import (
        build_hf_quantization_config,
        quantized_weight_export_states_compatible,
    )

    layer_states: dict[str, QuantizedWeightState | None] = {}
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
            elif mapped_state is not None and not quantized_weight_export_states_compatible(
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
    layer_states: dict[str, QuantizedWeightState | None],
    group: object | None,
) -> None:
    from modelopt.torch.export.quantized_weight import (
        quantized_weight_export_states_compatible,
    )

    world_size = torch.distributed.get_world_size(group=group)
    gathered: list[dict[str, QuantizedWeightState | None] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered, layer_states, group=group)
    for rank_states in gathered:
        if not rank_states:
            continue
        for name, state in rank_states.items():
            if name not in layer_states or layer_states[name] is None:
                layer_states[name] = state
            elif state is not None and not quantized_weight_export_states_compatible(layer_states[name], state):
                raise RuntimeError(f"Conflicting ModelOpt configs for {name}")


def _named_export_tensors(
    hf_name: str,
    state: QuantizedWeightState,
    weight: torch.Tensor,
) -> Iterator[tuple[str, torch.Tensor]]:
    from modelopt.torch.export.quantized_weight import export_quantized_weight

    prefix, separator, weight_name = hf_name.rpartition(".")
    exported = export_quantized_weight(weight, state, dtype=weight.dtype)
    for relative_name, tensor in exported.named_tensors(weight_name).items():
        yield f"{prefix}{separator}{relative_name}", tensor.detach()


def _expert_hf_name_template(hf_name: str) -> tuple[str, int] | None:
    """Replace one resolved expert index with a formatting placeholder."""
    parts = hf_name.split(".")
    expert_segments = [index for index, part in enumerate(parts[:-1]) if part == "experts"]
    if len(expert_segments) != 1:
        return None
    expert_segment = expert_segments[0] + 1
    if (
        expert_segment >= len(parts)
        or not parts[expert_segment].isascii()
        or not parts[expert_segment].isdecimal()
        or any(not part or "*" in part for part in parts)
    ):
        return None
    expert = int(parts[expert_segment])
    parts[expert_segment] = "{expert}"
    return ".".join(parts), expert


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


class _ModelOptLocalExpertMapping(AutoMapping):
    """Export one canonical HF expert without a BF16 EP gather."""

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


class _ModelOptLocalGatedExpertMapping(_LocalExpertGatedMLPMapping):
    """Split one canonical HF gate/up expert without a BF16 EP gather."""

    is_modelopt_pre_ep_export = True


def _modelopt_pre_ep_mapping(mapping: Any, pg_collection: Any = None) -> Any | None:
    """Build a local-expert mapping while preserving canonical HF names."""
    if type(mapping) is GatedMLPMapping and isinstance(mapping.hf_param, dict):
        gate_name = mapping.hf_param.get("gate")
        up_name = mapping.hf_param.get("up")
        if isinstance(gate_name, str) and isinstance(up_name, str):
            gate_expert = _expert_hf_name_template(gate_name)
            up_expert = _expert_hf_name_template(up_name)
            if gate_expert is not None and up_expert is not None and gate_expert[1] == up_expert[1]:
                replacement = _ModelOptLocalGatedExpertMapping(
                    mapping.megatron_param,
                    gate=gate_name,
                    up=up_name,
                )
                replacement.set_process_groups_from_pg_collection(pg_collection)
                return replacement

    if type(mapping) is not AutoMapping or not isinstance(mapping.hf_param, str):
        return None
    if _expert_hf_name_template(mapping.hf_param) is None:
        return None
    replacement = _ModelOptLocalExpertMapping(
        mapping.megatron_param,
        mapping.hf_param,
        mapping.permute_dims,
    )
    replacement.set_process_groups_from_pg_collection(pg_collection)
    return replacement


def _pre_ep_group(mapping: Any) -> tuple[str, int] | None:
    """Return a stable projection-family key and its resolved expert index."""
    names = (mapping.hf_param,) if isinstance(mapping.hf_param, str) else tuple(mapping.hf_param.values())
    templated = [_expert_hf_name_template(name) for name in names]
    if not templated or any(item is None for item in templated):
        return None
    templates = tuple(item[0] for item in templated if item is not None)
    experts = {item[1] for item in templated if item is not None}
    if len(experts) != 1:
        return None
    return "|".join(templates), experts.pop()


def _stage_tensor_for_collective(tensor: torch.Tensor, group: Any) -> torch.Tensor:
    """Move a CPU tensor to CUDA only when an NCCL collective requires it."""
    if str(torch.distributed.get_backend(group)).lower() != "nccl" or tensor.device.type != "cpu":
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
    concrete_tasks = [task for task in conversion_tasks if task is not None and not _is_quantizer_state_task(task)]
    config_weights = collect_modelopt_config_weights(concrete_tasks)
    states = collect_modelopt_quant_states(concrete_tasks)

    pg_collection = model_bridge_utils._get_pg_collection_from_model(model)
    if torch.distributed.is_initialized():
        pp_group = model_bridge_utils._get_pp_group(model)
        ep_group = model_bridge_utils._get_ep_group(model)
    else:
        pp_group = None
        ep_group = None
    pp_world_size = get_pg_size(pp_group)
    ep_world_size = pp_world_size if ep_group is pp_group else get_pg_size(ep_group)
    if pp_world_size > 1:
        sync_modelopt_quant_states(states, pp_group)

    can_export_before_ep = not (pp_world_size > 1 and pp_group is not None and ep_group is pp_group)
    candidates: dict[int, Any] = {}
    candidate_groups: dict[str, list[int]] = {}
    candidate_experts: dict[int, int] = {}
    for task_index, task in enumerate(concrete_tasks):
        if task.global_param_name not in states or not can_export_before_ep:
            continue
        candidate = _modelopt_pre_ep_mapping(task.mapping, pg_collection)
        group = _pre_ep_group(task.mapping)
        if candidate is None or group is None:
            continue
        group_key, expert = group
        candidates[task_index] = candidate
        candidate_experts[task_index] = expert
        candidate_groups.setdefault(group_key, []).append(task_index)

    experts_per_rank = 0
    num_experts = None
    eligible_groups: set[str] = set()
    global_expert_orders: dict[str, tuple[int, ...]] = {}
    local_expert_orders: dict[str, tuple[int, ...]] = {}
    if candidate_groups:
        model_config = model_bridge_utils.unwrap_model(model)[0].config
        num_experts = getattr(model_config, "num_moe_experts", None)
        valid_layout = isinstance(num_experts, int) and num_experts > 0 and num_experts % ep_world_size == 0
        experts_per_rank = num_experts // ep_world_size if valid_layout else 0
        for group_key, task_indices in candidate_groups.items():
            experts = tuple(sorted(candidate_experts[task_index] for task_index in task_indices))
            if valid_layout and len(experts) == experts_per_rank and len(set(experts)) == experts_per_rank:
                local_expert_orders[group_key] = experts

    if ep_world_size > 1:
        gathered_experts: list[dict[str, tuple[int, ...]] | None] = [None] * ep_world_size
        torch.distributed.all_gather_object(
            gathered_experts,
            local_expert_orders,
            group=ep_group,
        )
        group_keys = set().union(*(rank_groups or {} for rank_groups in gathered_experts))
        for group_key in group_keys:
            rank_orders = [(rank_groups or {}).get(group_key, ()) for rank_groups in gathered_experts]
            global_order = tuple(expert for rank_order in rank_orders for expert in rank_order)
            if (
                isinstance(num_experts, int)
                and all(len(rank_order) == experts_per_rank for rank_order in rank_orders)
                and len(set(global_order)) == len(global_order)
                and set(global_order) == set(range(num_experts))
            ):
                eligible_groups.add(group_key)
                global_expert_orders[group_key] = global_order
    else:
        for group_key, local_order in local_expert_orders.items():
            if isinstance(num_experts, int) and set(local_order) == set(range(num_experts)):
                eligible_groups.add(group_key)
                global_expert_orders[group_key] = local_order

    mapped_tasks = list(concrete_tasks)
    pre_ep_groups_by_name: dict[str, str] = {}
    for group_key, task_indices in candidate_groups.items():
        if group_key not in eligible_groups:
            continue
        for task_index in task_indices:
            mapped_tasks[task_index] = replace(
                concrete_tasks[task_index],
                mapping=candidates[task_index],
            )
            pre_ep_groups_by_name[concrete_tasks[task_index].global_param_name] = group_key

    pre_ep_tasks = [task for task in mapped_tasks if getattr(task.mapping, "is_modelopt_pre_ep_export", False)]
    regular_tasks = [task for task in mapped_tasks if not getattr(task.mapping, "is_modelopt_pre_ep_export", False)]
    pre_ep_only_names = {task.global_param_name for task in pre_ep_tasks}.difference(
        task.global_param_name for task in regular_tasks
    )
    regular_states = {name: state for name, state in states.items() if name not in pre_ep_only_names}
    if ep_world_size > 1 and ep_group is not pp_group:
        sync_modelopt_quant_states(regular_states, ep_group)

    model_config = model_bridge_utils.unwrap_model(model)[0].config
    regular_hf_states = build_hf_modelopt_quant_states(
        regular_tasks,
        regular_states,
        num_experts=getattr(model_config, "num_moe_experts", None),
    )
    pre_ep_hf_states = build_hf_modelopt_quant_states(pre_ep_tasks, states)
    hf_states: dict[str, MappedQuantizedWeightState] = dict(regular_hf_states)
    for hf_name, state in pre_ep_hf_states.items():
        _set_hf_quant_state(hf_states, hf_name, state)

    def export_weight(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        state = regular_hf_states.get(hf_name)
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
        grouped_states = regular_hf_states.get(hf_name)
        if not isinstance(grouped_states, tuple):
            raise RuntimeError(f"Missing grouped ModelOpt state for {hf_name}")
        if expert_number >= len(grouped_states):
            raise RuntimeError(f"Missing ModelOpt state for expert {expert_number} of grouped parameter {hf_name}")
        for name, exported_tensor in _named_export_tensors(
            hf_name,
            grouped_states[expert_number],
            tensor,
        ):
            yield HFWeightTuple(name, exported_tensor)

    def gather_ep_experts(
        hf_name_template: str,
        tensor: torch.Tensor,
        group_key: str,
    ) -> Iterable[HFWeightTuple]:
        global_order = global_expert_orders[group_key]
        if ep_world_size == 1:
            for expert, expert_tensor in zip(global_order, tensor, strict=True):
                yield HFWeightTuple(hf_name_template.format(expert=expert), expert_tensor)
            return
        local_tensor = tensor.contiguous()
        collective_tensor = _stage_tensor_for_collective(local_tensor, ep_group)
        local_bytes = collective_tensor.reshape(-1).view(torch.uint8)
        gathered_bytes = torch.empty(
            ep_world_size * local_bytes.numel(),
            dtype=torch.uint8,
            device=collective_tensor.device,
        )
        torch.distributed.all_gather_into_tensor(
            gathered_bytes,
            local_bytes,
            group=ep_group,
        )
        gathered = gathered_bytes.view(local_tensor.dtype).reshape(
            ep_world_size * local_tensor.shape[0],
            *local_tensor.shape[1:],
        )
        for expert, expert_tensor in zip(global_order, gathered, strict=True):
            yield HFWeightTuple(hf_name_template.format(expert=expert), expert_tensor)

    pre_ep_buffers: dict[tuple[str, str], dict[int, torch.Tensor]] = {}

    def build_pre_ep_hook(task: WeightConversionTask) -> HFExportHook:
        group_key = pre_ep_groups_by_name[task.global_param_name]
        local_expert_order = local_expert_orders[group_key]

        def export_local_expert(
            hf_name: str,
            tensor: torch.Tensor,
        ) -> Iterable[HFWeightTuple]:
            state = pre_ep_hf_states.get(hf_name)
            templated_name = _expert_hf_name_template(hf_name)
            if state is None or isinstance(state, tuple) or templated_name is None:
                raise RuntimeError(f"Missing ModelOpt state for pre-EP parameter {hf_name}")
            _, expert = templated_name
            for exported_name, exported_tensor in _named_export_tensors(
                hf_name,
                state,
                tensor,
            ):
                exported_template = _expert_hf_name_template(exported_name)
                if exported_template is None or exported_template[1] != expert:
                    raise RuntimeError(f"Invalid expert checkpoint name {exported_name}")
                name_template, _ = exported_template
                expert_tensors = pre_ep_buffers.setdefault((group_key, name_template), {})
                if expert in expert_tensors:
                    raise RuntimeError(f"Duplicate local expert {expert} for {exported_name}")
                expert_tensors[expert] = exported_tensor
                if len(expert_tensors) != experts_per_rank:
                    continue
                missing = set(local_expert_order).difference(expert_tensors)
                if missing:
                    raise RuntimeError(f"Missing local experts {sorted(missing)} for {exported_name}")
                local_batch = torch.stack(
                    [expert_tensors[index] for index in local_expert_order],
                    dim=0,
                )
                del pre_ep_buffers[(group_key, name_template)]
                for final_name, final_tensor in gather_ep_experts(
                    name_template,
                    local_batch,
                    group_key,
                ):
                    if task.export_hook is None:
                        yield HFWeightTuple(final_name, final_tensor)
                    else:
                        yield from task.export_hook(final_name, final_tensor)

        return export_local_expert

    export_tasks = []
    for task in mapped_tasks:
        if getattr(task.mapping, "is_modelopt_pre_ep_export", False):
            export_tasks.append(replace(task, export_hook=build_pre_ep_hook(task)))
            continue
        is_grouped = getattr(task.mapping, "is_grouped_export", False)
        has_grouped_state = is_grouped and any(
            isinstance(regular_hf_states.get(hf_name), tuple) for hf_name in _hf_weight_names(task)
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
    config_sync_groups = []
    if pp_world_size > 1:
        config_sync_groups.append(pp_group)
    if ep_world_size > 1 and ep_group is not pp_group:
        config_sync_groups.append(ep_group)

    return ModelOptExportPlan(
        conversion_tasks=export_tasks,
        quantization_config=build_modelopt_quantization_config(
            mapped_tasks,
            hf_states,
            config_weights=config_weights,
            sync_groups=config_sync_groups,
        ),
    )
