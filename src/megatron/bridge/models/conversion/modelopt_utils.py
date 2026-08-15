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

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, NamedTuple

import torch
from megatron.core.utils import get_pg_size

from megatron.bridge.models.conversion import model_bridge as model_bridge_utils
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    RowParallelMapping,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.quant_mapping import AmaxMapping


HFExportHook = Callable[[str, torch.Tensor], Iterable[HFWeightTuple]]


class ModelOptExportPlan(NamedTuple):
    """Prepared conversion tasks and their canonical ModelOpt HF config."""

    conversion_tasks: list[WeightConversionTask]
    quantization_config: dict[str, Any]


@dataclass(frozen=True)
class _SourceState:
    state: object | None
    weight_shape: tuple[int, ...]
    parallelism: str | None = None
    qkv_layout: tuple[int, int, int, int, bool] | None = None


def _same_storage(left: object, right: object) -> bool:
    if left is right:
        return True
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return False
    if left.device.type == "meta" or right.device.type == "meta":
        return False
    return (
        left.device == right.device
        and left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
    )


def _direct_weight_name(module: torch.nn.Module, weight: torch.Tensor) -> str | None:
    for name, parameter in module.named_parameters(recurse=False):
        if _same_storage(parameter, weight):
            return name
    return None


def _mapping_parallelism(mapping: Any, module: torch.nn.Module) -> str:
    if getattr(mapping, "is_grouped_export", False):
        raise NotImplementedError(
            "ModelOpt real-quant export does not yet support grouped HF expert weights"
        )
    if isinstance(mapping, GatedMLPMapping):
        return "gated"
    if isinstance(mapping, QKVMapping):
        return mapping._tp_mapping._detect_parallelism_type(module)
    if isinstance(mapping, AutoMapping):
        return mapping._detect_parallelism_type(module)
    if isinstance(mapping, ColumnParallelMapping):
        return "column"
    if isinstance(mapping, RowParallelMapping):
        return "row"
    if isinstance(mapping, ReplicatedMapping):
        return "replicated"
    raise NotImplementedError(f"ModelOpt real-quant export does not support {type(mapping).__name__} topology")


def _qkv_layout(mapping: Any, module: torch.nn.Module) -> tuple[int, int, int, int, bool] | None:
    if not isinstance(mapping, QKVMapping):
        return None
    config = mapping._get_config(module)
    return (
        int(config.num_attention_heads),
        int(config.num_query_groups),
        int(config.kv_channels or 0),
        int(config.hidden_size),
        bool(getattr(config, "attention_output_gate", False)),
    )


def _capture_source_state(task: WeightConversionTask) -> _SourceState | None:
    if task.megatron_module is None or task.param_weight is None:
        return None
    if not isinstance(task.param_weight, torch.Tensor):
        raise NotImplementedError("ModelOpt real-quant export requires a tensor source weight")

    weight_name = _direct_weight_name(task.megatron_module, task.param_weight)
    if weight_name is None:
        return _SourceState(None, tuple(task.param_weight.shape))

    from modelopt.torch.export.quant_utils import capture_quantized_weight_export_state
    from modelopt.torch.quantization.nn import SequentialQuantizer
    from modelopt.torch.quantization.utils import representative_weight_quantizer

    quantizer = representative_weight_quantizer(task.megatron_module, weight_name)
    if isinstance(quantizer, SequentialQuantizer):
        quantizer = quantizer[0]
    if quantizer is None or not quantizer.is_enabled:
        return _SourceState(None, tuple(task.param_weight.shape))

    return _SourceState(
        capture_quantized_weight_export_state(task.megatron_module, weight_name),
        tuple(task.param_weight.shape),
        _mapping_parallelism(task.mapping, task.megatron_module),
        _qkv_layout(task.mapping, task.megatron_module),
    )


def _all_gather_objects(value: Any, group: Any) -> list[Any]:
    world_size = get_pg_size(group)
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, value, group=group)
    return gathered


def _raise_distributed_errors(error: str | None, group: Any, context: str) -> None:
    errors = [item for item in _all_gather_objects(error, group) if item is not None]
    if errors:
        raise RuntimeError(f"{context}: {'; '.join(dict.fromkeys(errors))}")


def _world_group() -> Any:
    if not torch.distributed.is_initialized():
        return None
    return torch.distributed.group.WORLD


def _sync_source_states(
    local_states: dict[str, _SourceState],
    group: Any,
) -> dict[str, _SourceState]:
    synchronized: dict[str, _SourceState] = {}
    for rank_states in _all_gather_objects(local_states, group):
        synchronized.update(rank_states)
    return synchronized


def _mapping_names(mapping: Any) -> dict[str, str]:
    if isinstance(mapping.hf_param, str):
        return {"": mapping.hf_param}
    return dict(mapping.hf_param)


def _merge_states(states: list[object], weight_dim: int) -> object:
    from modelopt.torch.export.quant_utils import merge_quantized_weight_export_states

    if len(states) == 1:
        return states[0]
    return merge_quantized_weight_export_states(states, weight_dim)


def _select_state(state: object, weight_dim: int, indices: torch.Tensor) -> object:
    from modelopt.torch.export.quant_utils import select_quantized_weight_export_state

    return select_quantized_weight_export_state(state, weight_dim, indices)


def _transform_source_state(
    task: WeightConversionTask,
    source: _SourceState,
) -> dict[str, object | None]:
    names = _mapping_names(task.mapping)
    shards: list[_SourceState] = _all_gather_objects(source, task.mapping.tp_group)
    quantized = [shard.state is not None for shard in shards]
    if any(quantized) and not all(quantized):
        raise RuntimeError(f"Inconsistent ModelOpt quantization across TP for {task.global_param_name}")
    if not any(quantized):
        return {name: None for name in names.values() if name.endswith(".weight")}

    states = [shard.state for shard in shards]
    assert all(state is not None for state in states)
    concrete_states = [state for state in states if state is not None]
    parallelism = source.parallelism

    if parallelism == "gated":
        if set(names) != {"gate", "up"}:
            raise ValueError(f"Expected gate/up HF names for {task.global_param_name}")
        gate_states = []
        up_states = []
        for shard, state in zip(shards, concrete_states, strict=True):
            fused_size = shard.weight_shape[0]
            if fused_size % 2:
                raise ValueError(f"Expected an even gated dimension for {task.global_param_name}")
            midpoint = fused_size // 2
            gate_states.append(_select_state(state, 0, torch.arange(midpoint)))
            up_states.append(_select_state(state, 0, torch.arange(midpoint, fused_size)))
        transformed = {
            names["gate"]: _merge_states(gate_states, 0),
            names["up"]: _merge_states(up_states, 0),
        }
    else:
        if parallelism == "column":
            transformed_state = _merge_states(concrete_states, 0)
            full_shape = (
                sum(shard.weight_shape[0] for shard in shards),
                *source.weight_shape[1:],
            )
        elif parallelism == "row":
            transformed_state = (
                concrete_states[0] if len(source.weight_shape) == 1 else _merge_states(concrete_states, 1)
            )
            full_shape = (
                source.weight_shape
                if len(source.weight_shape) == 1
                else (
                    source.weight_shape[0],
                    sum(shard.weight_shape[1] for shard in shards),
                    *source.weight_shape[2:],
                )
            )
        elif parallelism == "replicated":
            transformed_state = concrete_states[0]
            full_shape = source.weight_shape
        else:
            raise NotImplementedError(f"Unsupported ModelOpt topology {parallelism!r} for {task.global_param_name}")

        if isinstance(task.mapping, QKVMapping):
            if source.qkv_layout is None:
                raise RuntimeError(f"Missing QKV layout for {task.global_param_name}")
            heads, groups, channels, hidden, output_gate = source.qkv_layout
            config = SimpleNamespace(
                num_attention_heads=heads,
                num_query_groups=groups,
                kv_channels=channels or None,
                hidden_size=hidden,
                attention_output_gate=output_gate,
            )
            q_indices, k_indices, v_indices = split_qkv_weights(
                config,
                torch.arange(full_shape[0]),
            )
            transformed = {
                names["q"]: _select_state(transformed_state, 0, q_indices),
                names["k"]: _select_state(transformed_state, 0, k_indices),
                names["v"]: _select_state(transformed_state, 0, v_indices),
            }
        else:
            if len(names) != 1:
                raise ValueError(f"Expected one HF name for {task.global_param_name}")
            if isinstance(task.mapping, AutoMapping) and task.mapping.permute_dims is not None:
                from modelopt.torch.export.quant_utils import permute_quantized_weight_export_state

                transformed_state = permute_quantized_weight_export_state(
                    transformed_state,
                    task.mapping.permute_dims,
                )
            transformed = {next(iter(names.values())): transformed_state}

    return transformed


class _LocalExpertMappingMixin:
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


class _LocalExpertAutoMapping(_LocalExpertMappingMixin, AutoMapping):
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


class _LocalExpertGatedMLPMapping(_LocalExpertMappingMixin, GatedMLPMapping):
    pass


def _local_expert_mapping(mapping: Any, pg_collection: Any) -> Any | None:
    if type(mapping) is AutoMapping and isinstance(mapping.hf_param, str):
        replacement = _LocalExpertAutoMapping(
            mapping.megatron_param,
            mapping.hf_param,
            mapping.permute_dims,
        )
    elif type(mapping) is GatedMLPMapping and isinstance(mapping.hf_param, dict):
        replacement = _LocalExpertGatedMLPMapping(
            mapping.megatron_param,
            mapping.hf_param["gate"],
            mapping.hf_param["up"],
        )
    else:
        return None
    replacement.set_process_groups_from_pg_collection(pg_collection)
    return replacement


def _stage_tensor_for_collective(tensor: torch.Tensor, group: Any) -> torch.Tensor:
    backend = str(torch.distributed.get_backend(group)).lower()
    if backend != "nccl" or tensor.device.type != "cpu":
        return tensor
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL ModelOpt expert gather requires CUDA")
    return tensor.to(torch.device("cuda", torch.cuda.current_device()))


def _gather_expert_output(
    hf_name: str,
    tensor: torch.Tensor,
    group: Any,
) -> Iterable[HFWeightTuple]:
    names = _all_gather_objects(hf_name, group)
    if len(names) == 1:
        yield HFWeightTuple(hf_name, tensor)
        return

    tensor = _stage_tensor_for_collective(tensor.contiguous(), group)
    local_bytes = tensor.reshape(-1).view(torch.uint8)
    gathered = [torch.empty_like(local_bytes) for _ in names]
    torch.distributed.all_gather(gathered, local_bytes, group=group)
    for name, value in zip(names, gathered, strict=True):
        yield HFWeightTuple(name, value.view(tensor.dtype).reshape(tensor.shape))


def _compose_export_hooks(
    exporter: HFExportHook,
    finalizer: HFExportHook | None,
) -> HFExportHook:
    if finalizer is None:
        return exporter

    def export_and_finalize(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        for exported_name, exported_tensor in exporter(hf_name, tensor):
            yield from finalizer(exported_name, exported_tensor)

    return export_and_finalize


def build_modelopt_export_plan(
    conversion_tasks: list[WeightConversionTask | None],
    *,
    model: list[torch.nn.Module],
) -> ModelOptExportPlan:
    """Prepare canonical ModelOpt export state without encoding a quantization format."""
    # ModelOpt converts quantizer state into deployment scales from the owning
    # weight task. Amax mappings are only part of the fake-quant refit path.
    concrete_tasks = [
        task
        for task in conversion_tasks
        if task is not None
        and not isinstance(getattr(task, "mapping", None), AmaxMapping)
    ]
    local_states = {}
    capture_error = None
    try:
        local_states = {
            task.global_param_name: source
            for task in concrete_tasks
            if (source := _capture_source_state(task)) is not None
        }
    except Exception as error:
        capture_error = f"{type(error).__name__}: {error}"
    _raise_distributed_errors(
        capture_error,
        _world_group(),
        "ModelOpt state capture failed",
    )

    pp_group = model_bridge_utils._get_pp_group(model) if torch.distributed.is_initialized() else None
    source_states = _sync_source_states(local_states, pp_group)

    pg_collection = model_bridge_utils._get_pg_collection_from_model(model)
    local_expert_mappings = {}
    expert_mapping_error = None
    try:
        for task in concrete_tasks:
            source = source_states.get(task.global_param_name)
            if source is None or source.state is None or not task.mapping.is_expert:
                continue
            local_mapping = _local_expert_mapping(task.mapping, pg_collection)
            if local_mapping is None:
                raise NotImplementedError(
                    "ModelOpt real-quant export cannot pack expert mapping "
                    f"{type(task.mapping).__name__} before its EP gather"
                )
            local_expert_mappings[task.global_param_name] = local_mapping
    except Exception as error:
        expert_mapping_error = f"{type(error).__name__}: {error}"
    _raise_distributed_errors(
        expert_mapping_error,
        _world_group(),
        "ModelOpt expert mapping validation failed",
    )

    named_states: dict[str, object | None] = {}
    for task in concrete_tasks:
        source = source_states.get(task.global_param_name)
        if source is None:
            continue
        transformed = None
        transform_error = None
        try:
            transformed = _transform_source_state(task, source)
        except Exception as error:
            transform_error = f"{type(error).__name__}: {error}"

        if task.mapping.is_expert:
            expert_states: dict[str, object | None] = {}
            gathered = _all_gather_objects(
                (transformed, transform_error),
                task.mapping.ep_group,
            )
            errors = [error for _, error in gathered if error is not None]
            if errors:
                raise RuntimeError(
                    "ModelOpt expert state transform failed: "
                    + "; ".join(dict.fromkeys(errors))
                )
            for rank_states, _ in gathered:
                assert rank_states is not None
                overlap = expert_states.keys() & rank_states.keys()
                if overlap:
                    raise RuntimeError(
                        f"Duplicate ModelOpt expert states: {sorted(overlap)}"
                    )
                expert_states.update(rank_states)
            transformed = expert_states
        elif transform_error is not None:
            raise RuntimeError(
                f"ModelOpt state transform failed for {task.global_param_name}: "
                f"{transform_error}"
            )

        assert transformed is not None
        for name, state in transformed.items():
            previous = named_states.setdefault(name, state)
            if previous is not state and previous is not None and state is not None:
                raise RuntimeError(f"Duplicate ModelOpt state for {name}")

    if not any(state is not None for state in named_states.values()):
        raise RuntimeError("No supported ModelOpt quantized weights were found")

    from modelopt.torch.export.quant_utils import (
        build_hf_quantization_config,
        export_quantized_weight_tensors,
    )

    quantization_config = build_hf_quantization_config(named_states)

    def export_weight(hf_name: str, tensor: torch.Tensor) -> Iterable[HFWeightTuple]:
        state = named_states.get(hf_name)
        if state is None:
            yield HFWeightTuple(hf_name, tensor)
            return
        if not hf_name.endswith(".weight"):
            raise ValueError(f"Expected a canonical HF weight name, got {hf_name}")
        prefix = hf_name.removesuffix(".weight")
        for relative_name, exported_tensor in export_quantized_weight_tensors(
            tensor,
            state,
            tensor.dtype,
        ).items():
            exported_name = hf_name if relative_name == "weight" else f"{prefix}.{relative_name}"
            yield HFWeightTuple(exported_name, exported_tensor)

    export_tasks = []
    for task in concrete_tasks:
        source = source_states.get(task.global_param_name)
        mapping = task.mapping
        finalizer = None
        if task.mapping.is_expert:
            eligibility = _all_gather_objects(
                source is not None and source.state is not None,
                task.mapping.ep_group,
            )
            if any(eligibility) and not all(eligibility):
                raise RuntimeError(f"Inconsistent ModelOpt quantization across EP for {task.global_param_name}")
        else:
            eligibility = [False]
        if all(eligibility):
            local_mapping = local_expert_mappings.get(task.global_param_name)
            if local_mapping is not None:
                mapping = local_mapping

                def finalize(name: str, value: torch.Tensor, group=mapping.ep_group):
                    yield from _gather_expert_output(name, value, group)

                finalizer = finalize
        export_tasks.append(
            replace(
                task,
                mapping=mapping,
                export_hook=_compose_export_hooks(export_weight, finalizer),
            )
        )

    return ModelOptExportPlan(export_tasks, quantization_config)
