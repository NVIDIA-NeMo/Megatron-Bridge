# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO Training Loop for heterogeneous multi-module training.

This module provides the dedicated training loop for MegatronMIMO models with
heterogeneous parallelism. It uses MultiModulePipelineCommunicator for
cross-module communication and supports per-module gradient handling.

Key differences from standard train():
- Creates MultiModulePipelineCommunicator for cross-module communication
- Creates MultiModuleProcessGroupCollection for the schedule
- Uses forward_backward_pipelining_without_interleaving with multimodule support
- Uses zero_grad_buffer_for_multimodule() for gradient clearing
- Supports per-module optimizers

Note: Stub ranks are disallowed - validated at setup time.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.pipeline_parallel.schedules import (
    forward_backward_no_pipelining,
    forward_backward_pipelining_without_interleaving,
)
from megatron.core.utils import get_model_config

from megatron.bridge.training.checkpointing import CheckpointManager, DefaultCheckpointManager
from megatron.bridge.training.eval import evaluate_and_print_results
from megatron.bridge.training.megatron_mimo_parallel_utils import (
    build_pg_collection_for_schedule,
    get_module_to_grid_tuple,
    unwrap_megatron_mimo_model,
    zero_grad_buffer_for_multimodule,
)
from megatron.bridge.training.megatron_mimo_step import forward_backward_colocated_mimo_with_pp
from megatron.bridge.training.profiling import (
    handle_profiling_step,
    handle_profiling_stop,
    initialize_pytorch_profiler,
    should_profile_rank,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.train import checkpoint_and_decide_exit
from megatron.bridge.training.utils.train_utils import (
    prepare_forward_step_func,
    training_log,
)


if TYPE_CHECKING:
    from megatron.core.models.mimo import MimoModel
    from megatron.core.models.mimo.optimizer import MimoOptimizer
    from megatron.core.optimizer.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
    from megatron.core.process_groups_config import MultiModuleProcessGroupCollection, ProcessGroupCollection

    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOInfra


logger = logging.getLogger(__name__)


def _optimizer_has_params(optimizer: object) -> bool:
    """Return whether an optimizer owns any rank-local trainable parameters."""
    return any(group.get("params") for group in getattr(optimizer, "param_groups", ()))


def _first_scheduler_with_param_groups(
    schedulers: Dict[str, "OptimizerParamScheduler"],
) -> Optional["OptimizerParamScheduler"]:
    """Return the first scheduler backed by a non-empty optimizer."""
    for scheduler in schedulers.values():
        if scheduler is not None and _optimizer_has_params(scheduler.optimizer):
            return scheduler
    return None


def _learning_rate_for_logging(
    schedulers: Dict[str, "OptimizerParamScheduler"],
) -> float:
    """Return a globally visible learning rate for logging.

    Non-colocated MegatronMIMO can have ranks whose local module is fully frozen.
    Those ranks have no rank-local scheduler, but training_log still expects a
    scalar learning rate on every rank.
    """
    local_learning_rate = -1.0
    sched = _first_scheduler_with_param_groups(schedulers)
    if sched is not None:
        local_learning_rate = sched.get_lr(sched.optimizer.param_groups[0])

    if dist.is_available() and dist.is_initialized():
        device = (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        learning_rate_tensor = torch.tensor([local_learning_rate], dtype=torch.float32, device=device)
        dist.all_reduce(learning_rate_tensor, op=dist.ReduceOp.MAX)
        local_learning_rate = learning_rate_tensor.item()

    return max(local_learning_rate, 0.0)


def _get_single_encoder_module_name(infra: "MegatronMIMOInfra") -> str:
    """Return the one non-language module supported by colocated language PP."""
    encoder_module_names = [name for name in infra.module_to_grid_map if name != MIMO_LANGUAGE_MODULE_KEY]
    if len(encoder_module_names) != 1:
        raise RuntimeError(
            "Colocated MegatronMIMO with language PP>1 requires exactly one encoder module on this path. "
            f"Found encoder modules: {encoder_module_names}."
        )
    return encoder_module_names[0]


def _needs_colocated_language_pp(model: "MimoModel") -> bool:
    """Return whether the unwrapped MIMO model needs the colocated PP adapter."""
    megatron_mimo_model = unwrap_megatron_mimo_model(model)
    return megatron_mimo_model.role.mode is ModuleLayout.COLOCATED and megatron_mimo_model.lm_has_pp


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _mimo_wandb_debug_enabled() -> bool:
    """Return whether per-iteration L3 resume diagnostics should be logged to W&B."""
    return _env_flag("MIMO_WANDB_DEBUG_METRICS")


def _metric_name_part(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _iter_debug_modules(model: "MimoModel") -> Iterator[tuple[str, torch.nn.Module]]:
    mimo_model = unwrap_megatron_mimo_model(model)
    language_model = getattr(mimo_model, "language_model", None)
    if language_model is not None:
        yield MIMO_LANGUAGE_MODULE_KEY, language_model

    modality_submodules = getattr(mimo_model, "modality_submodules", None)
    if modality_submodules is None:
        return
    if hasattr(modality_submodules, "items"):
        for name, module in modality_submodules.items():
            if module is not None:
                yield str(name), module


def _iter_trainable_params(model: "MimoModel") -> Iterator[tuple[str, str, torch.nn.Parameter]]:
    for module_name, module in _iter_debug_modules(model):
        for param_name, param in module.named_parameters():
            if param.requires_grad:
                yield module_name, param_name, param


def _grad_tensor(param: torch.nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    if isinstance(param.grad, torch.Tensor):
        return param.grad
    return None


def _summarize_tensors(tensors: Iterator[torch.Tensor | None]) -> dict[str, float]:
    total_sum = 0.0
    total_sq_norm = 0.0
    max_abs = 0.0
    total_numel = 0
    tensor_count = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        flat = tensor.detach().float().reshape(-1)
        if flat.numel() == 0:
            continue
        total_sum += flat.sum().item()
        total_sq_norm += flat.pow(2).sum().item()
        max_abs = max(max_abs, flat.abs().max().item())
        total_numel += flat.numel()
        tensor_count += 1
    return {
        "sum": total_sum,
        "norm": max(total_sq_norm, 0.0) ** 0.5,
        "max_abs": max_abs,
        "numel": float(total_numel),
        "tensor_count": float(tensor_count),
    }


def _add_stats(metrics: dict[str, float], prefix: str, stats: dict[str, float]) -> None:
    for stat_name, value in stats.items():
        metrics[f"{prefix}/{stat_name}"] = float(value)


def _reduce_world_stats(stats: dict[str, float]) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return stats

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    sum_tensor = torch.tensor(
        [
            stats.get("sum", 0.0),
            stats.get("norm", 0.0) ** 2,
            stats.get("numel", 0.0),
            stats.get("tensor_count", 0.0),
        ],
        dtype=torch.float64,
        device=device,
    )
    max_tensor = torch.tensor([stats.get("max_abs", 0.0)], dtype=torch.float64, device=device)
    dist.all_reduce(sum_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(max_tensor, op=dist.ReduceOp.MAX)
    return {
        "sum": float(sum_tensor[0].item()),
        "norm": max(float(sum_tensor[1].item()), 0.0) ** 0.5,
        "max_abs": float(max_tensor[0].item()),
        "numel": float(sum_tensor[2].item()),
        "tensor_count": float(sum_tensor[3].item()),
    }


def _add_trainable_param_metrics(metrics: dict[str, float], model: "MimoModel", prefix: str) -> None:
    named_params = list(_iter_trainable_params(model))
    local_param_stats = _summarize_tensors(param for _, _, param in named_params)
    _add_stats(metrics, f"{prefix}/local_trainable_params", local_param_stats)
    _add_stats(metrics, f"{prefix}/global_trainable_params", _reduce_world_stats(local_param_stats))

    modules = sorted({module_name for module_name, _, _ in named_params})
    for module_name in modules:
        module_stats = _summarize_tensors(param for mod, _, param in named_params if mod == module_name)
        _add_stats(metrics, f"{prefix}/local_module/{_metric_name_part(module_name)}/params", module_stats)


def _add_trainable_grad_metrics(metrics: dict[str, float], model: "MimoModel", prefix: str) -> None:
    named_params = list(_iter_trainable_params(model))
    local_grad_stats = _summarize_tensors(_grad_tensor(param) for _, _, param in named_params)
    _add_stats(metrics, f"{prefix}/local_trainable_grads", local_grad_stats)
    _add_stats(metrics, f"{prefix}/global_trainable_grads", _reduce_world_stats(local_grad_stats))

    modules = sorted({module_name for module_name, _, _ in named_params})
    for module_name in modules:
        module_stats = _summarize_tensors(_grad_tensor(param) for mod, _, param in named_params if mod == module_name)
        _add_stats(metrics, f"{prefix}/local_module/{_metric_name_part(module_name)}/grads", module_stats)


def _iter_optimizer_objects(optimizer: object) -> Iterator[object]:
    seen: set[int] = set()
    stack = [optimizer]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        inner = getattr(current, "optimizer", None)
        if inner is not None:
            stack.append(inner)
        stack.extend(getattr(current, "chained_optimizers", ()) or ())


def _is_debug_optimizer_tensor(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in {"exp_avg", "exp_avg_sq", "momentum", "variance", "step"}
        or "exp_avg" in lowered
        or "fp32" in lowered
        or "main_param" in lowered
    )


def _add_optimizer_state_metrics(metrics: dict[str, float], optimizer: "MimoOptimizer", prefix: str) -> None:
    module_infos = getattr(optimizer, "module_infos", {})
    state_tensors_by_name: dict[str, list[torch.Tensor]] = {}
    step_values: list[float] = []

    for module_name, info in sorted(module_infos.items()):
        module_optimizer = getattr(info, "optimizer", None)
        if not getattr(info, "is_active", False) or module_optimizer is None:
            continue
        if getattr(module_optimizer, "is_stub_optimizer", False):
            continue

        for opt_obj in _iter_optimizer_objects(module_optimizer):
            state = getattr(opt_obj, "state", None)
            if not isinstance(state, dict):
                continue
            for state_entry in state.values():
                if not isinstance(state_entry, dict):
                    continue
                for state_name, state_value in state_entry.items():
                    if state_name == "step":
                        if isinstance(state_value, torch.Tensor):
                            step_values.append(float(state_value.detach().float().max().item()))
                        elif isinstance(state_value, (int, float)):
                            step_values.append(float(state_value))
                        continue
                    if isinstance(state_value, torch.Tensor) and _is_debug_optimizer_tensor(str(state_name)):
                        key = f"{_metric_name_part(module_name)}/{_metric_name_part(state_name)}"
                        state_tensors_by_name.setdefault(key, []).append(state_value)

    for name, tensors in sorted(state_tensors_by_name.items()):
        _add_stats(metrics, f"{prefix}/local_optimizer_state/{name}", _summarize_tensors(iter(tensors)))
    if step_values:
        metrics[f"{prefix}/local_optimizer_state/max_step"] = max(step_values)
        metrics[f"{prefix}/local_optimizer_state/min_step"] = min(step_values)


def _iter_optimizer_param_tensors(optimizer: object) -> Iterator[torch.Tensor]:
    seen: set[int] = set()
    for opt_obj in _iter_optimizer_objects(optimizer):
        for group in getattr(opt_obj, "param_groups", ()) or ():
            if not isinstance(group, dict):
                continue
            for param in group.get("params", ()) or ():
                if not isinstance(param, torch.Tensor):
                    continue
                tensor_id = id(param)
                if tensor_id in seen:
                    continue
                seen.add(tensor_id)
                yield param


def _add_optimizer_param_metrics(metrics: dict[str, float], optimizer: "MimoOptimizer", prefix: str) -> None:
    """Log optimizer-owned parameter tensors.

    With MCore DistributedOptimizer these are the local FP32 master parameter
    shards that drive the next optimizer update, distinct from BF16 model
    parameters and from Adam moment tensors.
    """
    module_infos = getattr(optimizer, "module_infos", {})
    local_tensors: list[torch.Tensor] = []

    for module_name, info in sorted(module_infos.items()):
        module_optimizer = getattr(info, "optimizer", None)
        if not getattr(info, "is_active", False) or module_optimizer is None:
            continue
        if getattr(module_optimizer, "is_stub_optimizer", False):
            continue

        module_tensors = list(_iter_optimizer_param_tensors(module_optimizer))
        local_tensors.extend(module_tensors)
        _add_stats(
            metrics,
            f"{prefix}/local_optimizer_master_params/{_metric_name_part(module_name)}",
            _summarize_tensors(iter(module_tensors)),
        )

    local_stats = _summarize_tensors(iter(local_tensors))
    _add_stats(metrics, f"{prefix}/local_optimizer_master_params", local_stats)
    _add_stats(metrics, f"{prefix}/global_optimizer_master_params", _reduce_world_stats(local_stats))


def _add_loss_debug_metrics(
    metrics: dict[str, float],
    *,
    losses_reduced: list[dict[str, torch.Tensor]],
    infra: "MegatronMIMOInfra",
) -> None:
    if not losses_reduced:
        return
    language_pg = infra.pg_collections.get(MIMO_LANGUAGE_MODULE_KEY) if infra.pg_collections else None

    for mb_idx, loss_entry in enumerate(losses_reduced):
        for loss_name, value in loss_entry.items():
            if not isinstance(value, torch.Tensor):
                continue
            flat = value.detach().float().view(-1)
            if flat.numel() == 2:
                reduced = flat.clone()
                if language_pg is not None and language_pg.dp_cp is not None:
                    dist.all_reduce(reduced, group=language_pg.dp_cp)
                total_loss = float(reduced[0].item())
                num_tokens = float(reduced[1].item())
                mean_loss = total_loss / num_tokens if num_tokens else 0.0
                loss_prefix = f"mimo_debug/loss/{_metric_name_part(loss_name)}/mb_{mb_idx:02d}"
                metrics[f"{loss_prefix}/total"] = total_loss
                metrics[f"{loss_prefix}/tokens"] = num_tokens
                metrics[f"{loss_prefix}/mean"] = mean_loss
            elif flat.numel() == 1:
                metrics[f"mimo_debug/loss/{_metric_name_part(loss_name)}/mb_{mb_idx:02d}/mean"] = float(flat.item())


def _log_wandb_debug_metrics(
    global_state: GlobalState,
    metrics: dict[str, float],
    *,
    iteration: int,
) -> None:
    if not metrics:
        return
    wandb_writer = global_state.wandb_logger
    if wandb_writer is None:
        return
    metrics = {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
    if metrics:
        wandb_writer.log(metrics, iteration)


def train_step_megatron_mimo(
    forward_step_func: Callable,
    data_iterator: Iterator,
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    global_state: GlobalState,
    multimodule_communicator: Optional["MultiModulePipelineCommunicator"],
    multimodule_pg_collection,
    infra: "MegatronMIMOInfra",
    module_to_grid_tuple: List,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
) -> Tuple[Dict[str, torch.Tensor], Optional[float], Optional[int]]:
    """Single MegatronMIMO training step.

    Args:
        forward_step_func: Forward step function (wrapped with GlobalState).
        data_iterator: Iterator over the dataset.
        model: MimoModel instance.
        optimizer: MimoOptimizer managing per-module optimizers.
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
        global_state: GlobalState containing timers, config, train_state.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P; unused for colocated layouts.
        multimodule_pg_collection: PG collection for schedule.
        infra: MegatronMIMOInfra with grids, topology, pg_collections.
        module_to_grid_tuple: List of (module, grid) tuples.
        num_microbatches: Number of microbatches per iteration.
        seq_length: Sequence length.
        micro_batch_size: Micro batch size.

    Returns:
        Tuple of (loss_dict, skipped_iter, grad_norm, num_zeros_in_grad).
    """
    timers = global_state.timers
    debug_metrics: dict[str, float] | None = {} if _mimo_wandb_debug_enabled() else None
    if debug_metrics is not None:
        debug_metrics["mimo_debug/train_state/step_before"] = float(global_state.train_state.step)
        debug_metrics["mimo_debug/train_state/consumed_samples_before"] = float(
            global_state.train_state.consumed_train_samples
        )
        debug_metrics["mimo_debug/checkpoint/save_rng"] = float(bool(global_state.cfg.checkpoint.save_rng))
        debug_metrics["mimo_debug/checkpoint/load_rng"] = float(bool(global_state.cfg.checkpoint.load_rng))
        _add_trainable_param_metrics(debug_metrics, model, "mimo_debug/pre_step")
        _add_optimizer_state_metrics(debug_metrics, optimizer, "mimo_debug/pre_step")
        _add_optimizer_param_metrics(debug_metrics, optimizer, "mimo_debug/pre_step")

    # Zero gradients for all modules
    zero_grad_buffer_for_multimodule(module_to_grid_tuple)

    # Schedule dispatch: the colocated path with LLM PP>1 needs the three-phase
    # schedule (encoder full-batch forward → LLM 1F1B pipeline → encoder
    # backward) to avoid deadlocking encoder collectives inside the pipeline
    # staggering. Other cases — non-colocated (any PP layout) and colocated
    # with LLM PP=1 — use the standard schedule; colocated PP=1 works because
    # MimoModel._forward_all_modules runs encoder+communicate+LLM in a single
    # forward on every rank, which is what the standard schedule expects.
    megatron_mimo_model = unwrap_megatron_mimo_model(model)
    is_colocated = megatron_mimo_model.role.mode is ModuleLayout.COLOCATED
    needs_three_phase = _needs_colocated_language_pp(model)

    # Run forward-backward schedule
    timers("forward-backward", log_level=1).start(barrier=False)

    if needs_three_phase:
        language_pg = infra.pg_collections[MIMO_LANGUAGE_MODULE_KEY]
        if language_pg is None:
            raise RuntimeError("Colocated language-PP schedule requires an active language pg_collection.")
        language_p2p_communicator = P2PCommunicator(
            pp_group=language_pg.pp,
            config=get_model_config(model),
        )
        losses_reduced = forward_backward_colocated_mimo_with_pp(
            model=model,
            data_iterator=data_iterator,
            infra=infra,
            encoder_module_name=_get_single_encoder_module_name(infra),
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            p2p_communicator=language_p2p_communicator,
            diagnostics=debug_metrics,
        )
    elif is_colocated:
        # Colocated LLM-PP=1: encoder→language flows through
        # MimoModel._forward_all_modules via ColocatedBridgeCommunicator —
        # no cross-module P2P at the schedule level. Use the no-pipelining
        # schedule with the language module's pg_collection. Matches mcore's
        # test_mimo_colocated_correctness.py:_run_forward_backward pattern.
        language_pg = infra.pg_collections[MIMO_LANGUAGE_MODULE_KEY]
        losses_reduced = forward_backward_no_pipelining(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=[model],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            pg_collection=language_pg,
        )
    else:
        if multimodule_communicator is None:
            raise RuntimeError("Non-colocated MegatronMIMO training requires a MultiModulePipelineCommunicator.")
        losses_reduced = forward_backward_pipelining_without_interleaving(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=[model],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
            p2p_communicator=multimodule_communicator,
            pg_collection=multimodule_pg_collection,
        )

    timers("forward-backward").stop()

    if debug_metrics is not None:
        _add_trainable_grad_metrics(debug_metrics, model, "mimo_debug/post_backward")
        _add_optimizer_state_metrics(debug_metrics, optimizer, "mimo_debug/pre_optimizer")
        _add_optimizer_param_metrics(debug_metrics, optimizer, "mimo_debug/pre_optimizer")

    # Optimizer step - MimoOptimizer handles all modules and computes global grad norm
    timers("optimizer", log_level=1).start(barrier=False)

    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

    timers("optimizer").stop()

    if debug_metrics is not None:
        _add_trainable_param_metrics(debug_metrics, model, "mimo_debug/post_optimizer")
        _add_optimizer_state_metrics(debug_metrics, optimizer, "mimo_debug/post_optimizer")
        _add_optimizer_param_metrics(debug_metrics, optimizer, "mimo_debug/post_optimizer")

    # Step learning rate schedulers
    if update_successful:
        increment = num_microbatches * micro_batch_size * global_state.cfg.data_parallel_size
        for module_name, scheduler in schedulers.items():
            if scheduler is not None:
                scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    loss_dict = {}
    if losses_reduced:
        is_last_stage = False
        # Access role from unwrapped model (handles Float16Module wrapper)
        megatron_mimo_model = unwrap_megatron_mimo_model(model)
        if megatron_mimo_model.role is None:
            is_last_stage = True
        elif megatron_mimo_model.role.has_language_module:
            is_last_stage = megatron_mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)

        if is_last_stage:
            if debug_metrics is not None:
                _add_loss_debug_metrics(debug_metrics, losses_reduced=losses_reduced, infra=infra)
            llm_pg = infra.pg_collections.get(MIMO_LANGUAGE_MODULE_KEY) if infra.pg_collections else None
            for key in losses_reduced[0].keys():
                val = [x[key].view(-1) for x in losses_reduced]
                if val[0].numel() == 2:
                    val = torch.vstack(val).sum(dim=0)
                    if llm_pg is not None and llm_pg.dp_cp is not None:
                        torch.distributed.all_reduce(val, group=llm_pg.dp_cp)
                    loss_dict[key] = val[0] / val[1]
                elif val[0].numel() == 1:
                    loss_dict[key] = torch.cat(val).mean()
                else:
                    raise ValueError(f"Invalid value shape: {val[0].shape} for key {key}")

    # Broadcast loss_dict to all ranks (the last rank is the logging rank for
    # W&B/TensorBoard). Use broadcast_object_list from the source rank so every
    # rank ends up with the same dict — no fragile P2P or GPU-side pickle needed.
    last_rank = dist.get_world_size() - 1
    my_rank = dist.get_rank()

    # All ranks agree on which rank holds the loss (pick highest rank with data).
    has_loss = 1 if loss_dict else 0
    source_tensor = torch.tensor([my_rank if has_loss else -1], dtype=torch.int32, device="cuda")
    torch.distributed.all_reduce(source_tensor, op=torch.distributed.ReduceOp.MAX)
    source_rank = int(source_tensor.item())

    # Only broadcast if the source and logging rank differ and a valid source exists.
    if source_rank >= 0 and source_rank != last_rank:
        obj = [loss_dict if my_rank == source_rank else None]
        torch.distributed.broadcast_object_list(obj, src=source_rank)
        if my_rank == last_rank:
            received = obj[0] or {}
            # Tensors inside the received dict carry the source rank's CUDA device;
            # move them to this rank's device so training_log arithmetic works.
            loss_dict = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in received.items()}

    if debug_metrics is not None:
        debug_metrics["mimo_debug/train_state/step_after"] = float(global_state.train_state.step + 1)
        debug_metrics["mimo_debug/train_state/consumed_samples_after"] = float(
            global_state.train_state.consumed_train_samples
            + num_microbatches * micro_batch_size * global_state.cfg.data_parallel_size
        )
        _log_wandb_debug_metrics(global_state, debug_metrics, iteration=global_state.train_state.step + 1)

    return loss_dict, skipped_iter, grad_norm, num_zeros_in_grad


def train_megatron_mimo(
    forward_step_func: Callable,
    model: "MimoModel",
    optimizer: "MimoOptimizer",
    schedulers: Dict[str, "OptimizerParamScheduler"],
    train_data_iterator: Iterator,
    valid_data_iterator: Optional[Iterator],
    global_state: GlobalState,
    megatron_mimo_infra: "MegatronMIMOInfra",
    multimodule_communicator: Optional["MultiModulePipelineCommunicator"],
    active_module_name: str,
    local_pg_collection: "ProcessGroupCollection",
    checkpoint_manager: Optional[CheckpointManager] = None,
    multimodule_pg_collection: Optional["MultiModuleProcessGroupCollection"] = None,
    module_to_grid_tuple: Optional[List] = None,
) -> None:
    """Main MegatronMIMO training loop.

    Key differences from standard train():
    - Uses MultiModuleProcessGroupCollection for the schedule
    - Uses forward_backward_pipelining_without_interleaving with multimodule support
    - Uses zero_grad_buffer_for_multimodule() for gradient clearing
    - Uses MimoOptimizer for coordinated gradient clipping with global norm

    Reuses from existing Bridge training:
    - GlobalState for timers, config, train_state
    - training_log() for metrics reporting
    - handle_profiling_step() and handle_profiling_stop() for profiler lifecycle
    - save_checkpoint_and_time() / checkpoint_and_decide_exit() for checkpointing
    - evaluate_and_print_results() for validation with multimodule support
    - maybe_finalize_async_save() for async checkpoint finalization

    Args:
        forward_step_func: Forward step function.
        model: MimoModel instance.
        optimizer: MimoOptimizer managing per-module optimizers.
        schedulers: Per-module learning rate schedulers {module_name: scheduler}.
        train_data_iterator: Training data iterator.
        valid_data_iterator: Validation data iterator (optional).
        global_state: GlobalState containing timers, config, train_state.
        megatron_mimo_infra: MegatronMIMOInfra with grids, topology, pg_collections.
        multimodule_communicator: MultiModulePipelineCommunicator for P2P; unused for colocated layouts.
        active_module_name: Canonical module name for this rank (from setup). In
            non-colocated mode this is the single active module; in colocated
            mode it defaults to the language module. Used for logging reductions
            and legacy consumers that require a single module identity.
        local_pg_collection: Canonical per-rank ProcessGroupCollection matching
            ``active_module_name`` (from setup). Per-module operations should
            still iterate ``megatron_mimo_infra.pg_collections`` directly.
        checkpoint_manager: CheckpointManager for save operations. Created by
            setup_megatron_mimo(). If None, a DefaultCheckpointManager is created.
        multimodule_pg_collection: Pre-built PG collection for the pipeline schedule.
            If None, built from megatron_mimo_infra.
        module_to_grid_tuple: Pre-built (module, grid) pairs for gradient ops.
            If None, built from model and megatron_mimo_infra.
    """
    timers = global_state.timers
    train_state = global_state.train_state
    cfg = global_state.cfg

    # Get training config
    train_config = cfg.train
    num_microbatches = get_num_microbatches()
    seq_length = cfg.dataset.seq_length
    micro_batch_size = train_config.micro_batch_size

    # Prepare forward step function with GlobalState injection
    wrapped_forward_step_func = prepare_forward_step_func(forward_step_func, global_state)

    # Use pre-built objects from setup_megatron_mimo if provided, otherwise build them.
    if module_to_grid_tuple is None:
        module_to_grid_tuple = get_module_to_grid_tuple(model, megatron_mimo_infra)
    if multimodule_pg_collection is None:
        multimodule_pg_collection = build_pg_collection_for_schedule(megatron_mimo_infra)

    # Guard against list fallback - MegatronMIMO training requires MultiModuleProcessGroupCollection
    if isinstance(multimodule_pg_collection, list):
        raise RuntimeError(
            "MultiModuleProcessGroupCollection is required for MegatronMIMO training. "
            "The list-based fallback is not supported. Ensure Megatron-LM PR 3212 is available."
        )

    if checkpoint_manager is None:
        checkpoint_manager = DefaultCheckpointManager(cfg.checkpoint)

    # Initialize tracking variables
    total_loss_dict = {}
    history_wct = []
    report_memory_flag = True

    # Get first scheduler for checkpoint saving.
    # All modules share the same LR schedule, so first scheduler state is representative.
    first_scheduler = _first_scheduler_with_param_groups(schedulers)

    # Profiler setup (mirrors train.py behavior)
    prof = None
    nsys_nvtx_context = None
    prof_config = cfg.profiling
    if prof_config and should_profile_rank(prof_config, dist.get_rank()):
        if prof_config.use_pytorch_profiler:
            prof = initialize_pytorch_profiler(prof_config, cfg.logger.tensorboard_dir)
            prof.start()

    logger.info(f"Rank {dist.get_rank()}: Starting MegatronMIMO training loop")

    # Main training loop
    timers("interval-time", log_level=0).start(barrier=True)

    while train_state.step < train_config.train_iters:
        # Finalize any pending async saves (non-blocking). Placed at the top
        # of the loop so async saves get a full iteration to complete.
        checkpoint_manager.finalize_async_saves(
            state=global_state,
            blocking=False,
        )

        # Handle profiling
        nsys_ctx = handle_profiling_step(
            prof_config,
            train_state.step,
            dist.get_rank(),
            prof,
        )
        if nsys_ctx is not None:
            nsys_nvtx_context = nsys_ctx

        # Start iteration timer
        timers("iteration-time", log_level=0).start(barrier=False)

        # Run single training step
        loss_dict, skipped_iter, grad_norm, num_zeros_in_grad = train_step_megatron_mimo(
            forward_step_func=wrapped_forward_step_func,
            data_iterator=train_data_iterator,
            model=model,
            optimizer=optimizer,
            schedulers=schedulers,
            global_state=global_state,
            multimodule_communicator=multimodule_communicator,
            multimodule_pg_collection=multimodule_pg_collection,
            infra=megatron_mimo_infra,
            module_to_grid_tuple=module_to_grid_tuple,
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
        )

        # Stop iteration timer
        timers("iteration-time").stop(barrier=False)
        iteration_time = timers("iteration-time").elapsed(reset=True, barrier=False)
        history_wct.append(iteration_time)

        # Update training state
        train_state.step += 1
        train_state.consumed_train_samples += micro_batch_size * num_microbatches * cfg.data_parallel_size

        # Get learning rate from the first active scheduler. Some non-colocated
        # ranks may serve only frozen modules, so the value is shared globally.
        learning_rate = _learning_rate_for_logging(schedulers)

        # Log training metrics
        if not cfg.logger.skip_train_metrics_log:
            # Get loss scale from MimoOptimizer
            if optimizer is not None and hasattr(optimizer, "get_loss_scale"):
                loss_scale = optimizer.get_loss_scale()
                if hasattr(loss_scale, "item"):
                    loss_scale = loss_scale.item()
            else:
                loss_scale = 1.0

            report_memory_flag = training_log(
                loss_dict=loss_dict,
                total_loss_dict=total_loss_dict,
                learning_rate=learning_rate,
                decoupled_learning_rate=None,
                loss_scale=loss_scale,
                report_memory_flag=report_memory_flag,
                skipped_iter=skipped_iter,
                grad_norm=grad_norm,
                params_norm=None,
                num_zeros_in_grad=num_zeros_in_grad,
                config=cfg,
                global_state=global_state,
                history_wct=history_wct,
                model=[model],
                pg_collection=local_pg_collection,
            )

            # Log iteration-time directly for MegatronMIMO models.
            # training_log only logs this inside a hasattr(config.model, "kv_channels")
            # block which MegatronMIMO models don't satisfy, so we log it here as a workaround.
            if cfg.logger.log_timers_to_tensorboard and train_state.step % cfg.logger.log_interval == 0:
                writer = global_state.tensorboard_logger
                if writer:
                    writer.add_scalar("iteration-time", iteration_time, train_state.step)
                wandb_writer = global_state.wandb_logger
                if wandb_writer:
                    wandb_writer.log({"iteration-time": iteration_time}, train_state.step)

        # Evaluation at specified intervals
        if (
            train_config.eval_interval is not None
            and train_state.step % train_config.eval_interval == 0
            and valid_data_iterator is not None
        ):
            timers("evaluate", log_level=0).start(barrier=True)
            evaluate_and_print_results(
                state=global_state,
                prefix=f"iteration {train_state.step}",
                forward_step_func=forward_step_func,
                data_iterator=valid_data_iterator,
                model=[model],
                config=cfg,
                verbose=False,
                write_to_tensorboard=True,
                p2p_communicator=multimodule_communicator,
                pg_collection=multimodule_pg_collection,
            )
            timers("evaluate").stop()

        # Checkpointing (interval, signal, duration, exit-interval) and exit decision.
        # TODO: MegatronMIMO FLOPs estimation is non-trivial (heterogeneous modules); pass 0 for now.
        should_exit = checkpoint_and_decide_exit(
            state=global_state,
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=first_scheduler,
            num_floating_point_operations_so_far=0,
            checkpoint_manager=checkpoint_manager,
            train_data_iterator=train_data_iterator,
            pg_collection=local_pg_collection,
            module_name=active_module_name,
            megatron_mimo_infra=megatron_mimo_infra,
        )
        if should_exit:
            break

    # Stop profiling
    handle_profiling_stop(
        prof_config,
        train_state.step,
        dist.get_rank(),
        prof,
        nsys_nvtx_context,
    )

    timers("interval-time").stop()

    logger.info(f"Rank {dist.get_rank()}: MegatronMIMO training completed")
