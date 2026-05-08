# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""MegatronMIMO-specific forward step function for use with pipeline schedules.

This module provides the forward step function for MegatronMIMO model training.
Key design notes (per PR 3212):
- The schedule expects dict-based outputs: {module_name: tensor} instead of single tensors
- The MimoModel's forward returns output tensors that the schedule sends via MultiModulePipelineCommunicator
- The schedule's backward_step_multimodule() handles dict-based backward pass automatically
- Only the LLM module produces a loss - encoders just produce activations
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.pipeline_parallel.schedules import forward_backward_pipelining_without_interleaving
from megatron.core.utils import get_model_config

from megatron.bridge.data.megatron_mimo.dp_utils import slice_batch_for_megatron_mimo_modules
from megatron.bridge.training.megatron_mimo_parallel_utils import unwrap_megatron_mimo_model
from megatron.bridge.training.state import GlobalState


if TYPE_CHECKING:
    from megatron.bridge.models.megatron_mimo.megatron_mimo_provider import MegatronMIMOInfra


logger = logging.getLogger(__name__)

_DATA_ALIGNMENT_CHECK_COUNT = 0
_PP_DEBUG_LOG_COUNT = 0


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _pp_debug_log_active() -> bool:
    """Gate verbose colocated language-PP debug logging behind ``MIMO_PP_DEBUG_LOG``.

    Logging is bounded to the first ``MIMO_PP_DEBUG_LOG_ITERS`` iterations
    (default 1) so a long run does not flood the slurm log. Used to diagnose
    encoder-embedding alignment, per-microbatch loss, and rank slicing issues.
    """
    if not _env_flag("MIMO_PP_DEBUG_LOG"):
        return False
    max_iters = int(os.environ.get("MIMO_PP_DEBUG_LOG_ITERS", "1"))
    return _PP_DEBUG_LOG_COUNT < max_iters


def _tensor_summary(value: Any) -> str:
    """Compact ``shape sum norm`` string for a tensor; ``None`` / non-tensor falls through.

    ``sum`` and ``norm`` are computed in fp32 to keep them stable across
    bf16/fp16 runs; both can act as cheap order-and-content checksums.
    """
    if not isinstance(value, torch.Tensor):
        return repr(value)
    flat = value.detach().float().reshape(-1)
    return f"shape={tuple(value.shape)} dtype={value.dtype} sum={flat.sum().item():.6e} norm={flat.norm().item():.6e}"


def _tensor_debug_stats(value: Any) -> dict[str, float]:
    """Return numeric tensor checksums suitable for scalar metric logging."""
    if not isinstance(value, torch.Tensor):
        return {}
    flat = value.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"sum": 0.0, "norm": 0.0, "max_abs": 0.0, "numel": 0.0}
    return {
        "sum": flat.sum().item(),
        "norm": flat.norm().item(),
        "max_abs": flat.abs().max().item(),
        "numel": float(flat.numel()),
    }


def _metric_name_part(value: object) -> str:
    return str(value).replace("/", "_").replace(" ", "_")


def _add_tensor_debug_metrics(metrics: dict[str, float], prefix: str, value: Any) -> None:
    for stat_name, stat_value in _tensor_debug_stats(value).items():
        metrics[f"{prefix}/{stat_name}"] = stat_value


def _tensor_collection_debug_stats(tensors: Iterable[torch.Tensor | None]) -> dict[str, float]:
    """Aggregate tensor checksums for local/global diagnostic metrics."""
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


def _add_debug_stats(metrics: dict[str, float], prefix: str, stats: dict[str, float]) -> None:
    for stat_name, value in stats.items():
        metrics[f"{prefix}/{stat_name}"] = float(value)


def _reduce_world_debug_stats(stats: dict[str, float]) -> dict[str, float]:
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


def _add_nested_tensor_debug_metrics(
    metrics: dict[str, float],
    prefix: str,
    value: Any,
    *,
    depth_limit: int = 4,
) -> None:
    if isinstance(value, torch.Tensor):
        _add_tensor_debug_metrics(metrics, prefix, value)
        return
    if isinstance(value, dict) and depth_limit > 0:
        for key, nested_value in value.items():
            _add_nested_tensor_debug_metrics(
                metrics,
                f"{prefix}/{_metric_name_part(key)}",
                nested_value,
                depth_limit=depth_limit - 1,
            )


def _modality_input_summary(value: Any, *, depth_limit: int = 3) -> str:
    """Recursively summarize a possibly-nested modality input dict for logging."""
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if isinstance(value, dict) and depth_limit > 0:
        parts = [
            f"{key}=<{_modality_input_summary(inner, depth_limit=depth_limit - 1)}>" for key, inner in value.items()
        ]
        return "{" + ", ".join(parts) + "}"
    return repr(value)


def _first_tensor_batch_dim(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        return value.size(0)
    if isinstance(value, dict):
        for nested in value.values():
            batch_dim = _first_tensor_batch_dim(nested)
            if batch_dim is not None:
                return batch_dim
    return None


def _rank_modules(grids: Dict[str, Any]) -> Dict[str, Any]:
    current_rank = dist.get_rank()
    return {
        name: grid for name, grid in grids.items() if grid.rank_offset <= current_rank < (grid.rank_offset + grid.size)
    }


def _maybe_check_colocated_data_alignment(
    original_batch: Dict[str, Any],
    sliced_batch: Dict[str, Any],
    grids: Dict[str, Any],
    *,
    increment_count: bool = True,
) -> None:
    """One-shot L3 guardrail for colocated heterogeneous-DP data slicing."""
    global _DATA_ALIGNMENT_CHECK_COUNT

    # TODO(liding): to be removed after L3 data-alignment validation is no longer needed.
    if not _env_flag("MIMO_CHECK_DATA_ALIGNMENT"):
        return
    max_checks = int(os.environ.get("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1"))
    if _DATA_ALIGNMENT_CHECK_COUNT >= max_checks:
        return
    if not dist.is_available() or not dist.is_initialized() or not grids:
        return

    rank_modules = _rank_modules(grids)
    if len(rank_modules) < 2:
        return

    language_grid = rank_modules.get(MIMO_LANGUAGE_MODULE_KEY)
    if language_grid is None:
        return

    language_dp = language_grid.get_pg(["dp"])
    language_dp_rank = language_dp.rank()
    language_dp_size = language_dp.size()
    global_batch = _first_tensor_batch_dim(original_batch.get("input_ids"))
    if global_batch is None:
        return
    if global_batch % language_dp_size != 0:
        raise RuntimeError(
            f"MegatronMIMO data alignment check failed: global language batch {global_batch} "
            f"is not divisible by language DP size {language_dp_size}."
        )

    expected_language_batch = global_batch // language_dp_size
    local_language_batch = _first_tensor_batch_dim(sliced_batch.get("input_ids"))
    if local_language_batch != expected_language_batch:
        raise RuntimeError(
            "MegatronMIMO data alignment check failed: "
            f"language local batch is {local_language_batch}, expected {expected_language_batch} "
            f"from global batch {global_batch}, language dp_rank={language_dp_rank}, "
            f"language dp_size={language_dp_size}."
        )

    modality_reports = []
    original_modalities = original_batch.get("modality_inputs") or {}
    sliced_modalities = sliced_batch.get("modality_inputs") or {}
    for modality_name, encoder_grid in sorted(rank_modules.items()):
        if modality_name == MIMO_LANGUAGE_MODULE_KEY or modality_name not in original_modalities:
            continue

        encoder_dp = encoder_grid.get_pg(["dp"])
        encoder_dp_rank = encoder_dp.rank()
        encoder_dp_size = encoder_dp.size()
        original_modality_batch = _first_tensor_batch_dim(original_modalities.get(modality_name))
        local_modality_batch = _first_tensor_batch_dim(sliced_modalities.get(modality_name))
        if original_modality_batch is None:
            continue
        if original_modality_batch != global_batch:
            raise RuntimeError(
                "MegatronMIMO data alignment check failed: "
                f"modality '{modality_name}' global batch is {original_modality_batch}, "
                f"language global batch is {global_batch}."
            )
        if original_modality_batch % encoder_dp_size != 0:
            raise RuntimeError(
                "MegatronMIMO data alignment check failed: "
                f"modality '{modality_name}' global batch {original_modality_batch} is not divisible "
                f"by encoder DP size {encoder_dp_size}."
            )

        expected_modality_batch = original_modality_batch // encoder_dp_size
        if local_modality_batch != expected_modality_batch:
            raise RuntimeError(
                "MegatronMIMO data alignment check failed: "
                f"modality '{modality_name}' local batch is {local_modality_batch}, "
                f"expected {expected_modality_batch} from encoder dp_rank={encoder_dp_rank}, "
                f"encoder dp_size={encoder_dp_size}."
            )

        if encoder_dp_size % language_dp_size == 0:
            fanin = encoder_dp_size // language_dp_size
            expected_language_dp_rank = encoder_dp_rank // fanin
            if expected_language_dp_rank != language_dp_rank:
                raise RuntimeError(
                    "MegatronMIMO data alignment check failed: "
                    f"modality '{modality_name}' encoder dp_rank={encoder_dp_rank} maps to "
                    f"language dp_rank={expected_language_dp_rank}, but this rank is language "
                    f"dp_rank={language_dp_rank}."
                )

        modality_reports.append(
            f"{modality_name}:dp={encoder_dp_rank}/{encoder_dp_size},local_batch={local_modality_batch}"
        )

    logger.info(
        "MegatronMIMO data alignment check passed on rank %s: global_batch=%s, "
        "language_dp=%s/%s, language_local_batch=%s, modalities=[%s]",
        dist.get_rank(),
        global_batch,
        language_dp_rank,
        language_dp_size,
        local_language_batch,
        "; ".join(modality_reports),
    )
    if increment_count:
        _DATA_ALIGNMENT_CHECK_COUNT += 1


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor) -> Tuple:
    """Loss function for MegatronMIMO model training.

    Called at the terminal stage (LLM's last PP stage).

    Args:
        loss_mask: Mask indicating which tokens contribute to the loss.
        output_tensor: Model output tensor (losses per token).

    Returns:
        Tuple of (total_loss, num_tokens, {'lm loss': reporting_loss}).

    Note:
        Only the LLM module produces a loss. Encoders produce activations
        that are consumed by the LLM, but don't have their own loss.
    """
    losses = output_tensor.float()

    loss_mask = loss_mask.contiguous().view(-1).float()

    total_tokens = loss_mask.sum().clone().detach().to(torch.int)
    total_loss = torch.sum(losses.view(-1) * loss_mask)
    reporting_loss = torch.cat([total_loss.clone().detach().view(1), total_tokens.view(1)])

    return (total_loss, total_tokens, {"lm loss": reporting_loss})


def get_batch(data_iterator: Iterable) -> Optional[Dict[str, torch.Tensor]]:
    """Get batch from data iterator.

    Returns dict with:
    - input_ids, labels, loss_mask, position_ids (for LLM)
    - modality_inputs: {modality_name: preprocessed_tensors} (for encoders)

    Uses existing MegatronMIMODataset format from Phase 3.

    Args:
        data_iterator: Iterator over the dataset.

    Returns:
        Batch dictionary or None if iterator is exhausted.
    """
    if data_iterator is None:
        return None

    try:
        batch = next(data_iterator)
    except StopIteration:
        return None

    # Move tensors to GPU if not already there
    def _move_to_cuda(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cuda(non_blocking=True) if not obj.is_cuda else obj
        if isinstance(obj, dict):
            return {k: _move_to_cuda(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            converted = [_move_to_cuda(v) for v in obj]
            return type(obj)(converted)
        return obj

    if batch is not None:
        batch = _move_to_cuda(batch)

    return batch


def forward_step(
    state: GlobalState,
    data_iterator: Iterable,
    model: MimoModel,
) -> Tuple[torch.Tensor, Optional[partial]]:
    """Forward step for MegatronMIMO model training.

    Uses 3-arg signature with GlobalState for Bridge compatibility.
    The training loop wraps this with prepare_forward_step_func() which:
    - Injects GlobalState automatically if forward_step accepts it
    - Provides access to state.timers, state.cfg, state.train_state

    The MimoModel handles dict-based tensor flow internally:
    - Encoder modules produce activations sent via BridgeCommunicator
    - LLM module receives encoder outputs and produces loss

    At terminal stage: returns (loss_tensor, loss_func)
    At intermediate stages: returns (output_dict, None) - schedule handles communication

    GUARDRAIL: At last stage, assert output is scalar tensor (not dict) to catch
    misconfigurations early with a clear error message.

    Args:
        state: GlobalState containing timers, config, train_state.
        data_iterator: Iterator over the dataset.
        model: MimoModel instance.

    Returns:
        Tuple of (output_tensor, loss_function or None).
    """
    # Get the model's role to determine if we're at first pipeline stage
    megatron_mimo_model = unwrap_megatron_mimo_model(model)

    # Determine if this rank needs data.
    # - LLM ranks: first stage needs input_ids; last stage needs labels/loss_mask.
    # - Modality ranks: only first stage needs raw modality inputs.
    needs_data = True
    if megatron_mimo_model.role is not None:
        if megatron_mimo_model.role.has_language_module:
            is_first_stage = megatron_mimo_model.role.is_first_stage(MIMO_LANGUAGE_MODULE_KEY)
            is_last_stage = megatron_mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)
            needs_data = is_first_stage or is_last_stage
        elif megatron_mimo_model.role.has_modality_modules:
            modality_modules = megatron_mimo_model.role.modality_module_names
            needs_data = any(megatron_mimo_model.role.is_first_stage(mod) for mod in modality_modules)

    if needs_data:
        data_batch = get_batch(data_iterator)
        if data_batch is None:
            raise RuntimeError(
                "get_batch returned None at a stage that requires data. "
                "This indicates a data-loading or parallelism misconfiguration."
            )
        # Slice the global micro-batch per-module. All data-loading ranks
        # receive identical batches (sampler dp_size=1); the helper sub-shards
        # language keys by language DP and modality_inputs by each encoder's
        # DP. In non-colocated mode this reduces to uniform slicing by the
        # rank's single module's DP.
        grids = getattr(megatron_mimo_model.mimo_config, "module_to_grid_map", None) or {}
        original_data_batch = data_batch
        data_batch = slice_batch_for_megatron_mimo_modules(data_batch, grids=grids)
        _maybe_check_colocated_data_alignment(original_data_batch, data_batch, grids)
    else:
        # Non-data stages consume hidden states from pipeline input tensors.
        data_batch = {
            "input_ids": None,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "loss_mask": None,
            "modality_inputs": None,
        }

    # Extract loss_mask before forward pass
    loss_mask = data_batch.get("loss_mask")

    # Run forward pass
    # MimoModel.forward() returns (output_tensor, loss_mask) or just output_tensor
    output = model(**data_batch)

    # Handle tuple return from model
    if isinstance(output, tuple):
        output_tensor, model_loss_mask = output
        # Use model-provided loss_mask if available
        if model_loss_mask is not None:
            loss_mask = model_loss_mask
    else:
        output_tensor = output

    # Check if we're at the last pipeline stage for the language module
    # megatron_mimo_model was already unwrapped at the start of this function
    if megatron_mimo_model.role is None:
        is_last_stage = True
    elif megatron_mimo_model.role.has_language_module:
        is_last_stage = megatron_mimo_model.role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)
    else:
        is_last_stage = False

    if is_last_stage:
        # GUARDRAIL: Verify scalar loss at last stage
        if isinstance(output_tensor, dict):
            raise ValueError(
                f"Last pipeline stage must return scalar loss tensor, got dict with keys: {output_tensor.keys()}. "
                f"Ensure the LLM module's final stage produces a loss, not activations."
            )

        # Return output and loss function
        if loss_mask is not None:
            return output_tensor, partial(loss_func, loss_mask)
        else:
            # Create default loss mask if not provided
            logger.warning("No loss_mask provided, using all-ones mask")
            default_mask = torch.ones_like(output_tensor)
            return output_tensor, partial(loss_func, default_mask)

    # Intermediate stage - return output for activation passing
    return output_tensor, None


def forward_backward_colocated_mimo_with_pp(
    *,
    model: torch.nn.Module,
    data_iterator: Iterator[Dict[str, Any]],
    infra: "MegatronMIMOInfra",
    encoder_module_name: str,
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    forward_only: bool,
    p2p_communicator: object,
    force_all_reduce: bool = False,
    diagnostics: dict[str, float] | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run Bridge's three-phase colocated MIMO schedule for language PP.

    The adapter keeps Bridge-specific contracts intact: every rank consumes the
    full global microbatch from the data iterator, module-aware DP slicing
    happens inside the forward path, the inner language pipeline uses Bridge's
    loss function, and gradient finalization is deferred until encoder backward
    has produced encoder gradients.
    """
    mimo_model = unwrap_megatron_mimo_model(model)
    language_grid = infra.module_to_grid_map[MIMO_LANGUAGE_MODULE_KEY]
    language_pg = infra.pg_collections[MIMO_LANGUAGE_MODULE_KEY]
    if language_pg is None:
        raise RuntimeError("Colocated language-PP schedule requires an active language pg_collection.")

    language_micro_batch_size = _language_micro_batch_size(
        micro_batch_size=micro_batch_size,
        language_grid=language_grid,
    )

    encoder_grid = infra.module_to_grid_map.get(encoder_module_name)
    if encoder_grid is None:
        raise RuntimeError(
            f"Colocated language-PP schedule requires an encoder grid for module "
            f"'{encoder_module_name}', but it is not registered in module_to_grid_map."
        )

    original_batches, sliced_batches = _load_and_slice_microbatches(
        data_iterator=data_iterator,
        grids=infra.module_to_grid_map,
        num_microbatches=num_microbatches,
    )
    for original_batch, sliced_batch in zip(original_batches, sliced_batches):
        _maybe_check_colocated_data_alignment(
            original_batch,
            sliced_batch,
            infra.module_to_grid_map,
            increment_count=False,
        )

    _pp_debug_log_data_slicing(
        original_batches=original_batches,
        sliced_batches=sliced_batches,
        encoder_module_name=encoder_module_name,
        encoder_grid=encoder_grid,
        language_grid=language_grid,
    )
    _record_wandb_debug_data_slicing(
        diagnostics=diagnostics,
        original_batches=original_batches,
        sliced_batches=sliced_batches,
        encoder_module_name=encoder_module_name,
        special_token_ids=mimo_model.special_token_ids,
    )

    _pp_debug_log_module_params(
        mimo_model=mimo_model,
        encoder_module_name=encoder_module_name,
    )

    # Build the per-rank encoder input so that after the colocated bridge's
    # fan-in all-gather, the gathered tensor lands in microbatch-major order
    # on each language DP rank. ``_split_encoder_output`` downstream chunks
    # that gathered tensor per microbatch by token count, so the rank-major
    # layout produced by slicing each microbatch up front would mis-pair
    # encoder embeddings with language samples.
    full_encoder_input = _build_pp_encoder_input(
        original_batches=original_batches,
        encoder_module_name=encoder_module_name,
        encoder_grid=encoder_grid,
        language_grid=language_grid,
    )
    _validate_encoder_concat_batch_size(
        sliced_batches=sliced_batches,
        concatenated_input=full_encoder_input,
        encoder_module_name=encoder_module_name,
    )
    _maybe_check_colocated_pp_concat_alignment(
        sliced_batches=sliced_batches,
        concatenated_input=full_encoder_input,
        encoder_module_name=encoder_module_name,
    )
    encoder_outputs = mimo_model.encode_and_communicate(
        {encoder_module_name: full_encoder_input} if full_encoder_input else {}
    )
    detached_encoder_outputs = {
        name: output.detach().requires_grad_(not forward_only) for name, output in encoder_outputs.items()
    }
    cached_language_microbatches = _build_cached_language_microbatches(
        detached_encoder_outputs=detached_encoder_outputs,
        sliced_batches=sliced_batches,
        encoder_module_name=encoder_module_name,
        special_token_ids=mimo_model.special_token_ids,
    )

    _pp_debug_log_encoder_split(
        full_encoder_input=full_encoder_input,
        detached_encoder_outputs=detached_encoder_outputs,
        cached_language_microbatches=cached_language_microbatches,
        sliced_batches=sliced_batches,
        encoder_module_name=encoder_module_name,
        special_token_ids=mimo_model.special_token_ids,
    )
    _record_wandb_debug_encoder_split(
        diagnostics=diagnostics,
        detached_encoder_outputs=detached_encoder_outputs,
        cached_language_microbatches=cached_language_microbatches,
        encoder_module_name=encoder_module_name,
    )

    schedule_kwargs = {
        "forward_step_func": _make_inner_language_forward_step(diagnostics=diagnostics),
        "data_iterator": iter(cached_language_microbatches),
        "model": [model],
        "num_microbatches": num_microbatches,
        "seq_length": seq_length,
        "micro_batch_size": language_micro_batch_size,
        "decoder_seq_length": seq_length,
        "forward_only": forward_only,
        "p2p_communicator": p2p_communicator,
        "pg_collection": language_pg,
        "force_all_reduce": force_all_reduce,
    }

    if forward_only:
        result = forward_backward_pipelining_without_interleaving(**schedule_kwargs)
        _pp_debug_log_iteration_done()
        return result

    config = get_model_config(model)
    with _deferred_finalize(config) as (original_finalize, finalize_capture):
        losses_reduced = forward_backward_pipelining_without_interleaving(**schedule_kwargs)

    _backward_encoder_outputs(
        detached_encoder_outputs=detached_encoder_outputs,
        encoder_outputs=encoder_outputs,
        encoder_module_name=encoder_module_name,
        pp_group=language_pg.pp,
        diagnostics=diagnostics,
    )
    _record_wandb_debug_projector_grads(
        diagnostics=diagnostics,
        mimo_model=mimo_model,
        encoder_module_name=encoder_module_name,
        prefix="mimo_debug/projector_grad/post_encoder_backward",
    )

    _pp_debug_log_finalize_capture(finalize_capture=finalize_capture)

    if original_finalize is not None and finalize_capture.called:
        original_finalize(
            [model],
            finalize_capture.num_tokens,
            pg_collection=language_pg,
            force_all_reduce=finalize_capture.force_all_reduce,
        )
    _record_wandb_debug_projector_grads(
        diagnostics=diagnostics,
        mimo_model=mimo_model,
        encoder_module_name=encoder_module_name,
        prefix="mimo_debug/projector_grad/post_finalize",
    )

    _pp_debug_log_iteration_done()
    return losses_reduced


def _load_and_slice_microbatches(
    *,
    data_iterator: Iterator[Dict[str, Any]],
    grids: Dict[str, Any],
    num_microbatches: int,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    original_batches: list[Dict[str, Any]] = []
    sliced_batches: list[Dict[str, Any]] = []
    for _ in range(num_microbatches):
        batch = get_batch(data_iterator)
        if batch is None:
            raise RuntimeError(
                "MegatronMIMO colocated PP data iterator was exhausted while building cached microbatches."
            )
        sliced = slice_batch_for_megatron_mimo_modules(batch, grids=grids)
        original_batches.append(batch)
        sliced_batches.append(sliced)
    return original_batches, sliced_batches


def _get_grid_pg(grid: object, dim_name: str) -> object:
    try:
        return grid.get_pg([dim_name])
    except TypeError:
        return grid.get_pg(dim_name)


def _process_group_size(group: object | None) -> int:
    if group is None:
        return 1
    if hasattr(group, "size"):
        return group.size()
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=group)
    return 1


def _process_group_rank(group: object | None) -> int:
    if group is None:
        return 0
    if hasattr(group, "rank"):
        return group.rank()
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group=group)
    return 0


def _language_micro_batch_size(*, micro_batch_size: int, language_grid: object) -> int:
    language_dp = _get_grid_pg(language_grid, "dp")
    language_dp_size = _process_group_size(language_dp)
    if micro_batch_size % language_dp_size != 0:
        raise ValueError(
            f"micro_batch_size ({micro_batch_size}) must be divisible by language data_parallel_size "
            f"({language_dp_size}) for colocated language PP."
        )
    return micro_batch_size // language_dp_size


def _build_pp_encoder_input(
    *,
    original_batches: list[Dict[str, Any]],
    encoder_module_name: str,
    encoder_grid: object,
    language_grid: object,
) -> Any:
    """Build the per-rank encoder input for the three-phase PP schedule.

    The colocated bridge's fan-in all-gather concatenates each rank's
    contribution along the batch dim in encoder-DP-rank order. The downstream
    splitter chunks that gathered tensor per microbatch, so the gather output
    must land in microbatch-major order on each language DP rank.

    Each encoder DP rank ``r`` feeds language partition ``r // scale`` and
    appears at slot ``r % scale`` inside its gather group (matching
    ``ColocatedBridgeCommunicator._build_gather_groups`` in mcore). To produce
    that layout we:
      1. take this rank's language-DP partition slice from each microbatch;
      2. concat those per-microbatch slices in microbatch order; and
      3. narrow the resulting mega-batch by the encoder DP slot.

    Step 1 is a no-op when language DP is 1; step 3 is a no-op when encoder
    DP equals language DP. The fan-out branch (encoder DP < language DP) only
    slices per microbatch by encoder DP — the bridge narrows in forward, so
    no all-gather ordering constraint applies.
    """
    per_mb_inputs: list[Any] = []
    for batch in original_batches:
        modality_inputs = batch.get("modality_inputs") or {}
        value = modality_inputs.get(encoder_module_name)
        if value is None:
            continue
        per_mb_inputs.append(value)
    if not per_mb_inputs:
        return {}
    if len(per_mb_inputs) != len(original_batches):
        raise ValueError(
            f"Only {len(per_mb_inputs)} of {len(original_batches)} microbatches contain "
            f"modality_inputs for encoder module '{encoder_module_name}'."
        )

    enc_dp_pg = _get_grid_pg(encoder_grid, "dp")
    lm_dp_pg = _get_grid_pg(language_grid, "dp")
    enc_dp_size = _process_group_size(enc_dp_pg)
    enc_dp_rank = _process_group_rank(enc_dp_pg)
    lm_dp_size = _process_group_size(lm_dp_pg)

    path = f"modality_inputs.{encoder_module_name}"

    if enc_dp_size >= lm_dp_size:
        if enc_dp_size % lm_dp_size != 0:
            raise ValueError(
                f"encoder DP ({enc_dp_size}) must be divisible by language DP ({lm_dp_size}) "
                f"for colocated language-PP encoder input layout."
            )
        scale = enc_dp_size // lm_dp_size
        lm_partition = enc_dp_rank // scale
        slot = enc_dp_rank % scale

        if lm_dp_size > 1:
            per_mb_inputs = [
                _narrow_along_batch(value, partition_idx=lm_partition, num_partitions=lm_dp_size, path=path)
                for value in per_mb_inputs
            ]
        full_lm_partition = _concat_batch_first_values(per_mb_inputs, path=path)
        if scale > 1:
            full_lm_partition = _narrow_along_batch(
                full_lm_partition, partition_idx=slot, num_partitions=scale, path=path
            )
        return full_lm_partition

    # Fan-out: encoder DP < language DP. Each encoder rank holds its own
    # encoder-DP slice of every microbatch; the bridge narrows in forward to
    # feed each language DP rank, so no gather-ordering constraint applies.
    if enc_dp_size > 1:
        per_mb_inputs = [
            _narrow_along_batch(value, partition_idx=enc_dp_rank, num_partitions=enc_dp_size, path=path)
            for value in per_mb_inputs
        ]
    return _concat_batch_first_values(per_mb_inputs, path=path)


def _narrow_along_batch(
    value: Any,
    *,
    partition_idx: int,
    num_partitions: int,
    path: str,
) -> Any:
    """Narrow tensors at ``value`` to one of ``num_partitions`` chunks on dim 0.

    Recurses through nested dicts (``modality_inputs`` may carry per-encoder
    sub-dicts) and passes scalar / metadata leaves through unchanged. Raises
    ``ValueError`` when a tensor's batch dim is not divisible so a misconfigured
    layout fails here rather than mis-slicing silently.
    """
    if num_partitions <= 1:
        return value
    if isinstance(value, torch.Tensor):
        batch_size = value.size(0)
        if batch_size % num_partitions != 0:
            raise ValueError(f"Batch dim {batch_size} at '{path}' is not divisible by {num_partitions}.")
        chunk = batch_size // num_partitions
        return value[partition_idx * chunk : (partition_idx + 1) * chunk].contiguous()
    if isinstance(value, dict):
        return {
            key: _narrow_along_batch(
                inner,
                partition_idx=partition_idx,
                num_partitions=num_partitions,
                path=f"{path}.{key}",
            )
            for key, inner in value.items()
        }
    return value


def _concat_batch_first_values(values: list[Any], *, path: str) -> Any:
    first = values[0]
    if isinstance(first, torch.Tensor):
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError(f"Cannot concatenate mixed tensor and non-tensor values at {path}.")
        return torch.cat(values, dim=0)

    if isinstance(first, dict):
        if any(not isinstance(value, dict) for value in values):
            raise TypeError(f"Cannot concatenate mixed dict and non-dict values at {path}.")
        keys = set(first)
        for value in values[1:]:
            if set(value) != keys:
                raise ValueError(f"All microbatches must have identical keys at {path}.")
        return {
            key: _concat_batch_first_values([value[key] for value in values], path=f"{path}.{key}") for key in first
        }

    for value in values[1:]:
        if value != first:
            raise ValueError(f"Non-tensor metadata differs across microbatches at {path}.")
    return first


def _first_tensor_batch_size(value: Any) -> int | None:
    if isinstance(value, torch.Tensor):
        return value.size(0)
    if isinstance(value, dict):
        for nested_value in value.values():
            batch_size = _first_tensor_batch_size(nested_value)
            if batch_size is not None:
                return batch_size
    return None


def _validate_encoder_concat_batch_size(
    *,
    sliced_batches: list[Dict[str, Any]],
    concatenated_input: dict[str, Any],
    encoder_module_name: str,
) -> None:
    if not concatenated_input:
        return

    expected_batch_size = 0
    for batch in sliced_batches:
        modality_input = (batch.get("modality_inputs") or {}).get(encoder_module_name)
        batch_size = _first_tensor_batch_size(modality_input)
        if batch_size is None:
            return
        expected_batch_size += batch_size

    actual_batch_size = _first_tensor_batch_size(concatenated_input)
    if actual_batch_size is not None and actual_batch_size != expected_batch_size:
        raise RuntimeError(
            "MegatronMIMO colocated PP encoder concat produced an unexpected batch size: "
            f"got {actual_batch_size}, expected {expected_batch_size}."
        )


def _maybe_check_colocated_pp_concat_alignment(
    *,
    sliced_batches: list[Dict[str, Any]],
    concatenated_input: dict[str, Any],
    encoder_module_name: str,
) -> None:
    """Debug-only guardrail for colocated PP multi-microbatch aggregation."""
    global _DATA_ALIGNMENT_CHECK_COUNT

    if not _env_flag("MIMO_CHECK_DATA_ALIGNMENT"):
        return
    max_checks = int(os.environ.get("MIMO_CHECK_DATA_ALIGNMENT_STEPS", "1"))
    if _DATA_ALIGNMENT_CHECK_COUNT >= max_checks:
        return
    if not concatenated_input:
        return

    encoder_batch_sizes = []
    language_batch_sizes = []
    for batch in sliced_batches:
        modality_input = (batch.get("modality_inputs") or {}).get(encoder_module_name)
        encoder_batch_size = _first_tensor_batch_size(modality_input)
        language_batch_size = _first_tensor_batch_size(batch.get("input_ids"))
        if encoder_batch_size is None or language_batch_size is None:
            return
        encoder_batch_sizes.append(encoder_batch_size)
        language_batch_sizes.append(language_batch_size)

    expected_encoder_batch_size = sum(encoder_batch_sizes)
    actual_encoder_batch_size = _first_tensor_batch_size(concatenated_input)
    if actual_encoder_batch_size != expected_encoder_batch_size:
        raise RuntimeError(
            "MegatronMIMO colocated PP data alignment check failed: "
            f"concatenated encoder batch is {actual_encoder_batch_size}, expected "
            f"{expected_encoder_batch_size} from per-microbatch encoder batches {encoder_batch_sizes}."
        )

    logger.info(
        "MegatronMIMO colocated PP aggregation alignment check passed: "
        "encoder_module=%s, concat_encoder_batch=%s, encoder_microbatches=%s, "
        "concat_language_reference_count=%s, language_microbatches=%s",
        encoder_module_name,
        actual_encoder_batch_size,
        encoder_batch_sizes,
        sum(language_batch_sizes),
        language_batch_sizes,
    )
    _DATA_ALIGNMENT_CHECK_COUNT += 1


def _count_modality_tokens(input_ids: torch.Tensor | None, *, special_token_id: int) -> int:
    if input_ids is None:
        return 0
    return int((input_ids == special_token_id).sum().item())


def _split_encoder_output(
    output: torch.Tensor,
    *,
    token_counts: list[int],
    language_batch_sizes: list[int],
    encoder_module_name: str,
) -> list[torch.Tensor]:
    if output.ndim == 2:
        total_tokens = sum(token_counts)
        if output.size(0) != total_tokens:
            raise ValueError(
                f"Encoder output for '{encoder_module_name}' has {output.size(0)} flattened tokens, "
                f"but language microbatches contain {total_tokens} placeholder tokens."
            )
        chunks = []
        start = 0
        for token_count in token_counts:
            end = start + token_count
            chunks.append(output[start:end])
            start = end
        return chunks

    if output.ndim == 3:
        total_batch_size = sum(language_batch_sizes)
        if output.size(1) != total_batch_size:
            raise ValueError(
                f"Encoder output for '{encoder_module_name}' has batch dimension {output.size(1)}, "
                f"but language microbatches contain {total_batch_size} samples."
            )
        chunks = []
        start = 0
        for batch_size in language_batch_sizes:
            end = start + batch_size
            chunks.append(output[:, start:end, :])
            start = end
        return chunks

    raise ValueError(
        f"Encoder output for '{encoder_module_name}' must be flattened [N, H] or SBH [S, B, H], "
        f"got shape {tuple(output.shape)}."
    )


def _build_cached_language_microbatches(
    *,
    detached_encoder_outputs: dict[str, torch.Tensor],
    sliced_batches: list[Dict[str, Any]],
    encoder_module_name: str,
    special_token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    special_token_id = special_token_ids.get(encoder_module_name)
    if detached_encoder_outputs and special_token_id is None:
        raise ValueError(
            f"Missing special token id for encoder module '{encoder_module_name}', "
            "required to split flattened encoder embeddings into language microbatches."
        )

    token_counts: list[int] = []
    language_batch_sizes: list[int] = []
    for batch in sliced_batches:
        input_ids = batch.get("input_ids")
        token_count = (
            0 if special_token_id is None else _count_modality_tokens(input_ids, special_token_id=special_token_id)
        )
        token_counts.append(token_count)
        language_batch_sizes.append(input_ids.size(0) if isinstance(input_ids, torch.Tensor) else 0)

    encoder_chunks_by_module: dict[str, list[torch.Tensor]] = {}
    for module_name, output in detached_encoder_outputs.items():
        encoder_chunks_by_module[module_name] = _split_encoder_output(
            output,
            token_counts=token_counts,
            language_batch_sizes=language_batch_sizes,
            encoder_module_name=module_name,
        )

    cached_microbatches = []
    for microbatch_index, batch in enumerate(sliced_batches):
        cached_microbatches.append(
            {
                "input_ids": batch.get("input_ids"),
                "position_ids": batch.get("position_ids"),
                "labels": batch.get("labels"),
                "loss_mask": batch.get("loss_mask"),
                "attention_mask": batch.get("attention_mask"),
                "encoder_embeddings": {
                    module_name: chunks[microbatch_index] for module_name, chunks in encoder_chunks_by_module.items()
                },
            }
        )
    return cached_microbatches


def _make_inner_language_forward_step(diagnostics: dict[str, float] | None = None):
    microbatch_index = [0]

    def _inner_language_forward_step(
        cached_iterator: Iterator[Dict[str, Any]],
        model_arg: torch.nn.Module,
        *_unused_args: object,
    ) -> tuple[torch.Tensor, partial | None]:
        cached = next(cached_iterator)
        mimo_model = unwrap_megatron_mimo_model(model_arg)
        role = mimo_model.role
        is_first_stage = role is None or role.is_first_stage(MIMO_LANGUAGE_MODULE_KEY)
        is_last_stage = role is None or role.is_last_stage(MIMO_LANGUAGE_MODULE_KEY)
        mb_index = microbatch_index[0]
        microbatch_index[0] += 1

        kwargs: dict[str, Any] = {
            "input_ids": cached["input_ids"] if is_first_stage else None,
            "position_ids": cached["position_ids"] if is_first_stage else None,
            "attention_mask": cached["attention_mask"],
            "labels": cached["labels"] if is_last_stage else None,
            "loss_mask": cached["loss_mask"] if is_last_stage else None,
            "modality_inputs": None,
            "encoder_embeddings": cached["encoder_embeddings"] if is_first_stage else None,
        }

        _pp_debug_log_first_stage_inputs(
            mb_index=mb_index,
            is_first_stage=is_first_stage,
            cached=cached,
            kwargs=kwargs,
        )

        output = model_arg(**kwargs)
        loss_mask = kwargs["loss_mask"]
        if isinstance(output, tuple):
            output_tensor, model_loss_mask = output
            if model_loss_mask is not None:
                loss_mask = model_loss_mask
        else:
            output_tensor = output

        if is_last_stage:
            if loss_mask is None:
                logger.warning("No loss_mask provided to colocated language-PP last stage; using all-ones mask.")
                loss_mask = torch.ones_like(output_tensor)
            _record_wandb_debug_language_output(
                diagnostics=diagnostics,
                output_tensor=output_tensor,
                loss_mask=loss_mask,
                mb_index=mb_index,
            )
            return output_tensor, _make_logging_loss_func(loss_mask, mb_index)

        return output_tensor, None

    return _inner_language_forward_step


def _record_wandb_debug_language_output(
    *,
    diagnostics: dict[str, float] | None,
    output_tensor: torch.Tensor,
    loss_mask: torch.Tensor,
    mb_index: int,
) -> None:
    if diagnostics is None:
        return

    prefix = f"mimo_debug/language/mb_{mb_index:02d}"
    _add_tensor_debug_metrics(diagnostics, f"{prefix}/output_tensor", output_tensor)
    _add_tensor_debug_metrics(diagnostics, f"{prefix}/loss_mask", loss_mask)

    if not (isinstance(output_tensor, torch.Tensor) and isinstance(loss_mask, torch.Tensor)):
        return
    output_flat = output_tensor.detach().float().reshape(-1)
    mask_flat = loss_mask.detach().float().reshape(-1)
    if output_flat.numel() != mask_flat.numel():
        return

    masked_total = torch.sum(output_flat * mask_flat)
    tokens = torch.sum(mask_flat)
    diagnostics[f"{prefix}/masked_total"] = float(masked_total.item())
    diagnostics[f"{prefix}/masked_tokens"] = float(tokens.item())
    diagnostics[f"{prefix}/masked_mean"] = float((masked_total / tokens).item()) if tokens.item() else 0.0


def _make_logging_loss_func(loss_mask: torch.Tensor, mb_index: int):
    """Wrap ``loss_func`` to log per-microbatch loss / token counts on the last PP stage."""

    def _loss_func_with_logging(output_tensor: torch.Tensor):
        result = loss_func(loss_mask, output_tensor)
        _pp_debug_log_last_stage_loss(
            mb_index=mb_index,
            loss_mask=loss_mask,
            result=result,
        )
        return result

    return _loss_func_with_logging


class _DeferredFinalizeCapture:
    def __init__(self) -> None:
        self.called = False
        self.num_tokens: torch.Tensor | None = None
        self.force_all_reduce: bool | None = None

    def __call__(
        self,
        model_list: list[torch.nn.Module],
        num_tokens: torch.Tensor | None,
        *args: object,
        force_all_reduce: bool | None = None,
        **kwargs: object,
    ) -> None:
        del model_list, args, kwargs
        self.called = True
        self.num_tokens = num_tokens
        self.force_all_reduce = force_all_reduce


@contextmanager
def _deferred_finalize(config: object) -> Iterator[tuple[object | None, _DeferredFinalizeCapture]]:
    original_finalize = getattr(config, "finalize_model_grads_func", None)
    capture = _DeferredFinalizeCapture()
    if original_finalize is not None:
        config.finalize_model_grads_func = capture
    try:
        yield original_finalize, capture
    finally:
        if original_finalize is not None:
            config.finalize_model_grads_func = original_finalize


def _backward_encoder_outputs(
    *,
    detached_encoder_outputs: dict[str, torch.Tensor],
    encoder_outputs: dict[str, torch.Tensor],
    encoder_module_name: str,
    pp_group: object | None,
    diagnostics: dict[str, float] | None = None,
) -> None:
    if not encoder_outputs:
        return

    _record_wandb_debug_encoder_output_grads(
        diagnostics=diagnostics,
        detached_encoder_outputs=detached_encoder_outputs,
        encoder_module_name=encoder_module_name,
        prefix="mimo_debug/encoder_output_grad/pre_broadcast",
    )
    _broadcast_encoder_grads(
        detached_encoder_outputs=detached_encoder_outputs,
        pp_group=pp_group,
    )
    _record_wandb_debug_encoder_output_grads(
        diagnostics=diagnostics,
        detached_encoder_outputs=detached_encoder_outputs,
        encoder_module_name=encoder_module_name,
        prefix="mimo_debug/encoder_output_grad/post_broadcast",
    )

    outputs = []
    grad_tensors = []
    for module_name, output in encoder_outputs.items():
        grad = detached_encoder_outputs[module_name].grad
        if grad is not None:
            outputs.append(output)
            grad_tensors.append(grad)

    if outputs:
        torch.autograd.backward(outputs, grad_tensors=grad_tensors)


def _broadcast_encoder_grads(
    *,
    detached_encoder_outputs: dict[str, torch.Tensor],
    pp_group: object | None,
) -> None:
    if _process_group_size(pp_group) <= 1:
        return

    is_pp_first_stage = _process_group_rank(pp_group) == 0
    src_rank = dist.get_global_rank(pp_group, 0)
    for module_name, detached_output in detached_encoder_outputs.items():
        if is_pp_first_stage:
            if detached_output.grad is None:
                raise RuntimeError(f"No encoder gradient available on language PP stage 0 for '{module_name}'.")
            dist.broadcast(detached_output.grad, src=src_rank, group=pp_group)
        else:
            grad = torch.zeros_like(detached_output)
            dist.broadcast(grad, src=src_rank, group=pp_group)
            detached_output.grad = grad


# ---------------------------------------------------------------------------
# Colocated language-PP debug logging
#
# Gated behind ``MIMO_PP_DEBUG_LOG=1`` and bounded to the first
# ``MIMO_PP_DEBUG_LOG_ITERS`` iterations (default 1). Log lines are prefixed
# with ``mimo_pp_debug`` so they are easy to grep out of slurm output.
# ---------------------------------------------------------------------------


def _pp_debug_log_iteration_done() -> None:
    global _PP_DEBUG_LOG_COUNT
    if _env_flag("MIMO_PP_DEBUG_LOG"):
        _PP_DEBUG_LOG_COUNT += 1


def _record_wandb_debug_data_slicing(
    *,
    diagnostics: dict[str, float] | None,
    original_batches: list[Dict[str, Any]],
    sliced_batches: list[Dict[str, Any]],
    encoder_module_name: str,
    special_token_ids: dict[str, int],
) -> None:
    """Record per-microbatch data checksums for optional W&B diagnostics."""
    if diagnostics is None:
        return

    special_token_id = special_token_ids.get(encoder_module_name)
    for mb_idx, (orig, sliced) in enumerate(zip(original_batches, sliced_batches)):
        prefix = f"mimo_debug/batch/mb_{mb_idx:02d}"
        _add_tensor_debug_metrics(diagnostics, f"{prefix}/global_input_ids", orig.get("input_ids"))
        _add_tensor_debug_metrics(diagnostics, f"{prefix}/local_input_ids", sliced.get("input_ids"))
        _add_tensor_debug_metrics(diagnostics, f"{prefix}/global_labels", orig.get("labels"))
        _add_tensor_debug_metrics(diagnostics, f"{prefix}/local_labels", sliced.get("labels"))
        _add_tensor_debug_metrics(diagnostics, f"{prefix}/global_loss_mask", orig.get("loss_mask"))
        _add_tensor_debug_metrics(diagnostics, f"{prefix}/local_loss_mask", sliced.get("loss_mask"))
        _add_nested_tensor_debug_metrics(
            diagnostics,
            f"{prefix}/global_modality/{_metric_name_part(encoder_module_name)}",
            (orig.get("modality_inputs") or {}).get(encoder_module_name),
        )
        _add_nested_tensor_debug_metrics(
            diagnostics,
            f"{prefix}/local_modality/{_metric_name_part(encoder_module_name)}",
            (sliced.get("modality_inputs") or {}).get(encoder_module_name),
        )
        if special_token_id is not None:
            diagnostics[f"{prefix}/modality_token_count"] = float(
                _count_modality_tokens(orig.get("input_ids"), special_token_id=special_token_id)
            )


def _record_wandb_debug_encoder_split(
    *,
    diagnostics: dict[str, float] | None,
    detached_encoder_outputs: dict[str, torch.Tensor],
    cached_language_microbatches: list[dict[str, Any]],
    encoder_module_name: str,
) -> None:
    """Record encoder output / per-microbatch embedding checksums for W&B diagnostics."""
    if diagnostics is None:
        return

    module_key = _metric_name_part(encoder_module_name)
    _add_tensor_debug_metrics(
        diagnostics,
        f"mimo_debug/encoder/{module_key}/gathered_output",
        detached_encoder_outputs.get(encoder_module_name),
    )
    for mb_idx, cached in enumerate(cached_language_microbatches):
        chunk = (cached.get("encoder_embeddings") or {}).get(encoder_module_name)
        _add_tensor_debug_metrics(
            diagnostics,
            f"mimo_debug/encoder/{module_key}/mb_{mb_idx:02d}/chunk",
            chunk,
        )


def _record_wandb_debug_encoder_output_grads(
    *,
    diagnostics: dict[str, float] | None,
    detached_encoder_outputs: dict[str, torch.Tensor],
    encoder_module_name: str | None = None,
    prefix: str,
) -> None:
    """Record gradients on detached encoder outputs around the language-PP handoff."""
    if diagnostics is None:
        return

    local_tensors: list[torch.Tensor] = []
    module_names = [encoder_module_name] if encoder_module_name is not None else sorted(detached_encoder_outputs)
    for module_name in module_names:
        detached_output = detached_encoder_outputs.get(module_name)
        if not isinstance(detached_output, torch.Tensor):
            grad = None
        else:
            grad = detached_output.grad if isinstance(detached_output.grad, torch.Tensor) else None
        if grad is not None:
            local_tensors.append(grad)
        local_stats = _tensor_collection_debug_stats([grad])
        module_key = _metric_name_part(module_name)
        _add_debug_stats(diagnostics, f"{prefix}/{module_key}/local", local_stats)
        _add_debug_stats(diagnostics, f"{prefix}/{module_key}/global", _reduce_world_debug_stats(local_stats))

    local_all_stats = _tensor_collection_debug_stats(local_tensors)
    _add_debug_stats(diagnostics, f"{prefix}/local", local_all_stats)
    _add_debug_stats(diagnostics, f"{prefix}/global", _reduce_world_debug_stats(local_all_stats))


def _parameter_grad_tensor(param: torch.nn.Parameter) -> torch.Tensor | None:
    main_grad = getattr(param, "main_grad", None)
    if isinstance(main_grad, torch.Tensor):
        return main_grad
    if isinstance(param.grad, torch.Tensor):
        return param.grad
    return None


def _record_wandb_debug_projector_grads(
    *,
    diagnostics: dict[str, float] | None,
    mimo_model: MimoModel,
    encoder_module_name: str,
    prefix: str,
) -> None:
    """Record trainable encoder-module parameter gradients during the PP handoff."""
    if diagnostics is None:
        return

    modality_submodules = getattr(mimo_model, "modality_submodules", None)
    module = None
    if modality_submodules is not None:
        try:
            module = modality_submodules[encoder_module_name]
        except (KeyError, TypeError):
            module = modality_submodules.get(encoder_module_name) if hasattr(modality_submodules, "get") else None
    if module is None:
        local_stats = _tensor_collection_debug_stats([])
    else:
        grads = [_parameter_grad_tensor(param) for param in module.parameters() if param.requires_grad]
        local_stats = _tensor_collection_debug_stats(grads)

    module_key = _metric_name_part(encoder_module_name)
    _add_debug_stats(diagnostics, f"{prefix}/{module_key}/local", local_stats)
    _add_debug_stats(diagnostics, f"{prefix}/{module_key}/global", _reduce_world_debug_stats(local_stats))


def _pp_debug_global_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _pp_debug_log_data_slicing(
    *,
    original_batches: list[Dict[str, Any]],
    sliced_batches: list[Dict[str, Any]],
    encoder_module_name: str,
    encoder_grid: object,
    language_grid: object,
) -> None:
    """Log per-rank language/vision DP-TP-PP coords and per-microbatch checksums."""
    if not _pp_debug_log_active():
        return

    enc_dp = _get_grid_pg(encoder_grid, "dp")
    enc_tp = _get_grid_pg(encoder_grid, "tp")
    lm_dp = _get_grid_pg(language_grid, "dp")
    lm_tp = _get_grid_pg(language_grid, "tp")
    try:
        lm_pp = _get_grid_pg(language_grid, "pp")
        lm_pp_str = f"{_process_group_rank(lm_pp)}/{_process_group_size(lm_pp)}"
    except Exception:
        lm_pp_str = "n/a"

    rank = _pp_debug_global_rank()
    logger.info(
        "mimo_pp_debug data_slicing rank=%d enc(dp=%d/%d,tp=%d/%d) lm(dp=%d/%d,tp=%d/%d,pp=%s) num_microbatches=%d",
        rank,
        _process_group_rank(enc_dp),
        _process_group_size(enc_dp),
        _process_group_rank(enc_tp),
        _process_group_size(enc_tp),
        _process_group_rank(lm_dp),
        _process_group_size(lm_dp),
        _process_group_rank(lm_tp),
        _process_group_size(lm_tp),
        lm_pp_str,
        len(original_batches),
    )

    for mb_idx, (orig, sliced) in enumerate(zip(original_batches, sliced_batches)):
        orig_input_ids = orig.get("input_ids")
        sliced_input_ids = sliced.get("input_ids")
        orig_modality = (orig.get("modality_inputs") or {}).get(encoder_module_name)
        sliced_modality = (sliced.get("modality_inputs") or {}).get(encoder_module_name)
        logger.info(
            "mimo_pp_debug data_slicing rank=%d mb=%d "
            "global_input_ids=%s local_input_ids=%s "
            "global_modality=%s local_modality=%s",
            rank,
            mb_idx,
            _tensor_summary(orig_input_ids),
            _tensor_summary(sliced_input_ids),
            _modality_input_summary(orig_modality),
            _modality_input_summary(sliced_modality),
        )


def _pp_debug_log_encoder_split(
    *,
    full_encoder_input: Any,
    detached_encoder_outputs: dict[str, torch.Tensor],
    cached_language_microbatches: list[dict[str, Any]],
    sliced_batches: list[Dict[str, Any]],
    encoder_module_name: str,
    special_token_ids: dict[str, int],
) -> None:
    """Log gathered encoder output shape, token counts, and per-microbatch chunk checksums."""
    if not _pp_debug_log_active():
        return

    rank = _pp_debug_global_rank()
    encoder_output = detached_encoder_outputs.get(encoder_module_name)
    special_token_id = special_token_ids.get(encoder_module_name)

    token_counts = []
    for batch in sliced_batches:
        input_ids = batch.get("input_ids")
        token_counts.append(
            0 if special_token_id is None else _count_modality_tokens(input_ids, special_token_id=special_token_id)
        )

    logger.info(
        "mimo_pp_debug encoder_split rank=%d encoder_module=%s per_rank_input=<%s> gathered_output=%s token_counts=%s",
        rank,
        encoder_module_name,
        _modality_input_summary(full_encoder_input),
        _tensor_summary(encoder_output),
        token_counts,
    )

    for mb_idx, cached in enumerate(cached_language_microbatches):
        chunk = (cached.get("encoder_embeddings") or {}).get(encoder_module_name)
        logger.info(
            "mimo_pp_debug encoder_split rank=%d mb=%d chunk=%s",
            rank,
            mb_idx,
            _tensor_summary(chunk),
        )


def _pp_debug_log_first_stage_inputs(
    *,
    mb_index: int,
    is_first_stage: bool,
    cached: dict[str, Any],
    kwargs: dict[str, Any],
) -> None:
    """Log per-microbatch language input + encoder embedding alignment on first PP stage."""
    if not _pp_debug_log_active():
        return
    if not is_first_stage:
        return

    rank = _pp_debug_global_rank()
    input_ids = kwargs.get("input_ids")
    encoder_embeddings = kwargs.get("encoder_embeddings") or {}
    embedding_summary = {module_name: _tensor_summary(tensor) for module_name, tensor in encoder_embeddings.items()}
    logger.info(
        "mimo_pp_debug first_stage_inputs rank=%d mb=%d input_ids=%s encoder_embeddings=%s",
        rank,
        mb_index,
        _tensor_summary(input_ids),
        embedding_summary,
    )


def _pp_debug_log_last_stage_loss(
    *,
    mb_index: int,
    loss_mask: torch.Tensor,
    result: tuple,
) -> None:
    """Log per-microbatch loss / token count on the last PP stage."""
    if not _pp_debug_log_active():
        return

    rank = _pp_debug_global_rank()
    total_loss = result[0] if len(result) > 0 else None
    total_tokens = result[1] if len(result) > 1 else None
    loss_value = float(total_loss.detach().item()) if isinstance(total_loss, torch.Tensor) else None
    tokens_value = int(total_tokens.detach().item()) if isinstance(total_tokens, torch.Tensor) else None
    mean_loss = (loss_value / tokens_value) if (loss_value is not None and tokens_value) else None
    mask_sum = float(loss_mask.detach().float().sum().item()) if isinstance(loss_mask, torch.Tensor) else None
    logger.info(
        "mimo_pp_debug last_stage_loss rank=%d mb=%d loss_mask_sum=%s total_loss=%s num_tokens=%s mean_loss=%s",
        rank,
        mb_index,
        mask_sum,
        loss_value,
        tokens_value,
        mean_loss,
    )


def _pp_debug_log_module_params(*, mimo_model: object, encoder_module_name: str) -> None:
    """Log per-rank parameter checksums for the vision and language modules.

    Diagnoses cross-rank weight divergence — specifically the
    ``_get_global_seed_and_module`` (megatron_mimo_provider.py) seed-source
    behavior, which seeds CPU init with ``base_seed + 100 * lm_pp_rank``. In
    colocated PP=2 that splits the vision projector's CPU init RNG between LM
    PP stages, so vision-DP siblings on different LM PP stages end up with
    different randomly initialized projector weights even though the colocated
    layout intends a single vision-DP=4 group.

    The expected pattern when the divergence is present:
      - Trainable vision parameters (projector) cluster into two checksums —
        one for ranks at LM PP stage 0, another for ranks at LM PP stage 1.
      - Frozen vision parameters (CLIP encoder) match across all ranks
        (loaded from a single checkpoint, no random init).
      - Language parameters match within each LM PP stage and differ between
        stages by design (each stage owns different layer indices).
    """
    if not _pp_debug_log_active():
        return

    rank = _pp_debug_global_rank()

    vision_submodule = None
    modality_submodules = getattr(mimo_model, "modality_submodules", None)
    if modality_submodules is not None:
        try:
            vision_submodule = modality_submodules[encoder_module_name]
        except (KeyError, TypeError):
            vision_submodule = None

    if vision_submodule is not None:
        for name, param in vision_submodule.named_parameters():
            if not isinstance(param, torch.Tensor):
                continue
            flat = param.detach().float().reshape(-1)
            logger.info(
                "mimo_pp_debug module_param rank=%d module=%s requires_grad=%s name=%s shape=%s sum=%.6e norm=%.6e",
                rank,
                encoder_module_name,
                param.requires_grad,
                name,
                tuple(param.shape),
                flat.sum().item(),
                flat.norm().item(),
            )

    language_model = getattr(mimo_model, "language_model", None)
    if language_model is not None:
        # Language LM has many parameters; cap at ~10 representative entries
        # so the log stays readable. The first 10 entries cover the embedding
        # and the first decoder layer's main weights, which is enough to
        # confirm checkpoint loading is symmetric inside each LM PP stage.
        language_param_cap = int(os.environ.get("MIMO_PP_DEBUG_LM_PARAM_CAP", "10"))
        for idx, (name, param) in enumerate(language_model.named_parameters()):
            if idx >= language_param_cap:
                break
            if not isinstance(param, torch.Tensor):
                continue
            flat = param.detach().float().reshape(-1)
            logger.info(
                "mimo_pp_debug module_param rank=%d module=language requires_grad=%s name=%s shape=%s "
                "sum=%.6e norm=%.6e",
                rank,
                param.requires_grad,
                name,
                tuple(param.shape),
                flat.sum().item(),
                flat.norm().item(),
            )


def _pp_debug_log_finalize_capture(*, finalize_capture: "_DeferredFinalizeCapture") -> None:
    if not _pp_debug_log_active():
        return
    rank = _pp_debug_global_rank()
    num_tokens = finalize_capture.num_tokens
    tokens_value = int(num_tokens.detach().item()) if isinstance(num_tokens, torch.Tensor) else num_tokens
    logger.info(
        "mimo_pp_debug finalize_capture rank=%d called=%s num_tokens=%s force_all_reduce=%s",
        rank,
        finalize_capture.called,
        tokens_value,
        finalize_capture.force_all_reduce,
    )
