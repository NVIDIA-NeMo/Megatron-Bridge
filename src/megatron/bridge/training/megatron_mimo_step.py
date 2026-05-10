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


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


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
    """Optional data-alignment guard for colocated heterogeneous-DP slicing."""
    global _DATA_ALIGNMENT_CHECK_COUNT

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
) -> list[dict[str, torch.Tensor]]:
    """Run Bridge's colocated MIMO schedule for language PP.

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

    schedule_kwargs = {
        "forward_step_func": _make_inner_language_forward_step(),
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
        return forward_backward_pipelining_without_interleaving(**schedule_kwargs)

    config = get_model_config(model)
    with _deferred_finalize(config) as (original_finalize, finalize_capture):
        losses_reduced = forward_backward_pipelining_without_interleaving(**schedule_kwargs)

    _backward_encoder_outputs(
        detached_encoder_outputs=detached_encoder_outputs,
        encoder_outputs=encoder_outputs,
        pp_group=language_pg.pp,
    )

    if original_finalize is not None and finalize_capture.called:
        original_finalize(
            [model],
            finalize_capture.num_tokens,
            pg_collection=language_pg,
            force_all_reduce=finalize_capture.force_all_reduce,
        )
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
    """Build the per-rank encoder input for the colocated language-PP schedule.

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

    The language-DP partition slice is a no-op when language DP is 1; the
    encoder-DP slot narrow is a no-op when encoder DP equals language DP. The
    fan-out branch (encoder DP < language DP) only slices per microbatch by
    encoder DP — the bridge narrows in forward, so no all-gather ordering
    constraint applies.
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
    """Optional guardrail for colocated PP multi-microbatch aggregation."""
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


def _make_inner_language_forward_step():
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

        kwargs: dict[str, Any] = {
            "input_ids": cached["input_ids"] if is_first_stage else None,
            "position_ids": cached["position_ids"] if is_first_stage else None,
            "attention_mask": cached["attention_mask"],
            "labels": cached["labels"] if is_last_stage else None,
            "loss_mask": cached["loss_mask"] if is_last_stage else None,
            "modality_inputs": None,
            "encoder_embeddings": cached["encoder_embeddings"] if is_first_stage else None,
        }

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
            return output_tensor, partial(loss_func, loss_mask)

        return output_tensor, None

    return _inner_language_forward_step


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
    pp_group: object | None,
) -> None:
    if not encoder_outputs:
        return

    _broadcast_encoder_grads(
        detached_encoder_outputs=detached_encoder_outputs,
        pp_group=pp_group,
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
