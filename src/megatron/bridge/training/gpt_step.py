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

import logging
from functools import partial
from typing import Iterable

import modelopt.torch.distill as mtd
import torch
from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_model_config,
    get_pg_rank,
    get_pg_size,
    is_te_min_version,
    unwrap_model,
)

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.post_training.distillation import loss_func_kd
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params
from megatron.bridge.training.utils.pg_utils import get_pg_collection


logger = logging.getLogger(__name__)


def _trim_padded_cu_seqlens_for_cp(cu_seqlens: torch.Tensor, cu_seqlens_argmin: torch.Tensor | None) -> torch.Tensor:
    """Trim padded THD cu_seqlens without introducing a CUDA sync."""
    if cu_seqlens_argmin is not None:
        if cu_seqlens_argmin.is_cuda:
            raise ValueError("Packed CP batches expect cu_seqlens_argmin on CPU to avoid device-to-host sync")
        return cu_seqlens[: int(cu_seqlens_argmin.item())]

    if cu_seqlens.is_cuda:
        raise ValueError("Packed CP batches require cu_seqlens_argmin to trim cu_seqlens without GPU synchronization")

    # Packed dataset padding uses -1 sentinels. Match the first negative entry
    # instead of argmin so this stays correct for any negative sentinel value.
    padding_indices = torch.nonzero(cu_seqlens < 0, as_tuple=True)[0]
    if padding_indices.numel() == 0:
        return cu_seqlens
    return cu_seqlens[: int(padding_indices[0].item())]


def _uses_packed_sequence_metadata(cfg: ConfigContainer) -> bool:
    """Return whether the dataset is expected to provide packed sequence metadata."""
    dataset_cfg = getattr(cfg, "dataset", None)
    offline_packing_specs = getattr(dataset_cfg, "offline_packing_specs", None)
    if getattr(dataset_cfg, "enable_offline_packing", False):
        packed_sequence_size = getattr(offline_packing_specs, "packed_sequence_size", None)
        return packed_sequence_size is None or packed_sequence_size > 0

    return getattr(dataset_cfg, "enable_in_batch_packing", False)


def _middle_pp_stage_needs_batch(cfg: ConfigContainer) -> bool:
    """Return whether middle PP stages need batch metadata for attention."""
    dataset_cfg = getattr(cfg, "dataset", None)
    uses_custom_attention_mask = not getattr(dataset_cfg, "skip_getting_attention_mask_from_dataset", True)
    return uses_custom_attention_mask or _uses_packed_sequence_metadata(cfg)


def _layout_stage_has_mtp(layout, *, pp_rank: int, pp_size: int, vp_stage: int) -> bool:
    """Return whether a parsed or raw pipeline layout stage owns MTP layers."""
    if isinstance(layout, str):
        layout = PipelineParallelLayerLayout.from_str(layout, pp_size)

    if isinstance(layout, PipelineParallelLayerLayout):
        stage_layout = layout.layout[pp_rank][vp_stage]
    elif isinstance(layout, list):
        stage_layout = layout[vp_stage * pp_size + pp_rank]
    else:
        return False

    return any(
        layer == "mtp" or layer == LayerType.mtp or getattr(layer, "name", None) == "mtp" for layer in stage_layout
    )


def _current_stage_has_mtp_from_layout(cfg: ConfigContainer, *, pg_collection, vp_stage: int | None = None) -> bool:
    """Return whether the current PP/VPP stage owns the configured MTP block, derived from layout."""
    model_cfg = getattr(cfg, "model", None)
    layout = getattr(model_cfg, "pipeline_model_parallel_layout", None)
    if layout is None:
        return False

    pp_group = getattr(pg_collection, "pp", None)
    pp_rank = get_pg_rank(pp_group)
    pp_size = get_pg_size(pp_group)
    if vp_stage is None:
        vp_stage = parallel_state.get_virtual_pipeline_model_parallel_rank()
    if vp_stage is None:
        vp_stage = 0

    return _layout_stage_has_mtp(layout, pp_rank=pp_rank, pp_size=pp_size, vp_stage=vp_stage)


def _current_stage_needs_mtp_inputs_from_layout(
    cfg: ConfigContainer, *, pg_collection, is_last: bool, vp_stage: int | None = None
) -> bool:
    """Return whether this stage needs token ids for MTP embedding lookup, derived from layout."""
    model_cfg = getattr(cfg, "model", None)
    layout = getattr(model_cfg, "pipeline_model_parallel_layout", None)
    if layout is None:
        return is_last

    return _current_stage_has_mtp_from_layout(cfg, pg_collection=pg_collection, vp_stage=vp_stage)


def _model_chunk_vp_stage(model: GPTModel) -> int | None:
    """Return the virtual pipeline stage owned by the current model chunk."""
    try:
        vp_stage = get_attr_wrapped_model(model, "vp_stage", allow_none=False)
    except RuntimeError:
        return None
    return vp_stage if isinstance(vp_stage, int) else None


def _partition_packed_batch_for_cp(batch: dict[str, torch.Tensor], cp_size: int) -> dict[str, torch.Tensor]:
    """Partition THD/packed batches across context-parallel ranks.

    Uses transformer_engine's `thd_get_partitioned_indices` to slice sequence
    dimension aligned with packed cu_seqlens. This avoids the generic
    `get_batch_on_this_cp_rank` slicing which assumes contiguous sequence tokens.
    """

    err_msg = "Please update Transformer Engine to >= 1.10 to use Context Parallel with THD format data"
    try:
        import transformer_engine_torch as tex

        if not is_te_min_version("1.10.0"):
            logger.error(err_msg)
            raise RuntimeError(err_msg)
    except ModuleNotFoundError as e:
        logger.error(err_msg)
        raise e

    cp_rank = parallel_state.get_context_parallel_rank()
    cu_seqlens = batch["cu_seqlens"]
    if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
        raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
    cu_seqlens = cu_seqlens.squeeze()
    cu_seqlens = _trim_padded_cu_seqlens_for_cp(cu_seqlens, batch.get("cu_seqlens_argmin"))

    cu_seqlens_unpadded = batch.get("cu_seqlens_unpadded")
    if cu_seqlens_unpadded is not None:
        batch["cu_seqlens_unpadded"] = cu_seqlens_unpadded.squeeze()

    skip_keys = {
        "cu_seqlens",
        "cu_seqlens_unpadded",
        "cu_seqlens_argmin",
        "cu_seqlens_unpadded_argmin",
        "max_seqlen",
        "token_count",
    }

    for key, val in batch.items():
        if val is None or key in skip_keys:
            continue
        index = tex.thd_get_partitioned_indices(cu_seqlens, val.size(1), cp_size, cp_rank)
        batch[key] = val.index_select(1, index)

    return batch


def get_batch_from_iterator(
    data_iterator: Iterable,
    include_mtp_inputs: bool = False,
    skip_getting_attention_mask_from_dataset: bool = True,
    *,
    is_first_pp_stage: bool,
    is_last_pp_stage: bool,
    include_full_batch_fields: bool = False,
) -> dict[str, torch.Tensor]:
    """Get a batch of data from the iterator.

    Args:
        data_iterator: The data iterator to get the batch from.
        include_mtp_inputs: Whether this PP stage needs Multi-Token Prediction input tensors.
        skip_getting_attention_mask_from_dataset: If set, the dataset will pass a None attention mask.
        include_full_batch_fields: Whether to include all standard training tensors regardless of PP stage.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the batch data.
    """
    batch = next(data_iterator)

    required_device_keys = set()
    required_host_keys = set()

    if include_full_batch_fields:
        required_device_keys.update(("tokens", "labels", "loss_mask", "attention_mask", "position_ids"))
    elif not skip_getting_attention_mask_from_dataset:
        required_device_keys.add("attention_mask")

    if "cu_seqlens" in batch:
        required_device_keys.add("cu_seqlens")
        if "cu_seqlens_unpadded" in batch:
            required_device_keys.add("cu_seqlens_unpadded")
        required_host_keys.add("cu_seqlens_argmin")
        required_host_keys.add("max_seqlen")
        if "cu_seqlens_unpadded_argmin" in batch:
            required_host_keys.add("cu_seqlens_unpadded_argmin")

    if not include_full_batch_fields:
        if is_first_pp_stage or include_mtp_inputs:
            required_device_keys.update(("tokens", "position_ids"))
        if is_last_pp_stage:
            required_device_keys.update(("labels", "loss_mask"))

    _batch_required_keys = {}
    for key, val in batch.items():
        if key in required_device_keys:
            _batch_required_keys[key] = val.cuda(non_blocking=True) if val is not None else None
        elif key in required_host_keys:
            _batch_required_keys[key] = val.cpu() if val is not None else None
        else:
            _batch_required_keys[key] = None

    return _batch_required_keys


def get_batch(
    data_iterator: Iterable,
    cfg: ConfigContainer,
    use_mtp: bool = False,
    *,
    pg_collection,
    vp_stage: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled
        vp_stage: Virtual pipeline stage for the current model chunk.

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask, position_ids,
        cu_seqlens, cu_seqlens_argmin, max_seqlen, cu_seqlens_unpadded, and
        cu_seqlens_unpadded_argmin
    """
    # Determine pipeline stage role via process group collection
    model_cfg = getattr(cfg, "model", None)
    vp_size = getattr(model_cfg, "virtual_pipeline_model_parallel_size", None)
    is_first = is_pp_first_stage(pg_collection.pp) and (
        vp_stage is None or is_vp_first_stage(vp_stage=vp_stage, vp_size=vp_size)
    )
    is_last = is_pp_last_stage(pg_collection.pp) and (
        vp_stage is None or is_vp_last_stage(vp_stage=vp_stage, vp_size=vp_size)
    )
    is_middle = (not is_first) and (not is_last)
    include_full_batch_fields = is_middle and _middle_pp_stage_needs_batch(cfg)
    include_mtp_inputs = use_mtp and _current_stage_needs_mtp_inputs_from_layout(
        cfg, pg_collection=pg_collection, is_last=is_last, vp_stage=vp_stage
    )
    if is_middle and not include_full_batch_fields and not include_mtp_inputs:
        return None, None, None, None, None, None, None, None, None, None

    batch = get_batch_from_iterator(
        data_iterator,
        include_mtp_inputs=include_mtp_inputs,
        skip_getting_attention_mask_from_dataset=getattr(
            cfg.dataset, "skip_getting_attention_mask_from_dataset", True
        ),
        is_first_pp_stage=is_first,
        is_last_pp_stage=is_last,
        include_full_batch_fields=include_full_batch_fields,
    )

    cp_size = pg_collection.cp.size()
    has_packed = batch.get("cu_seqlens") is not None
    if has_packed and cp_size > 1:
        batch = _partition_packed_batch_for_cp(batch, cp_size)
    else:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch, is_hybrid_cp=False, cp_group=pg_collection.cp)

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch.get(
            "attention_mask"
        ),  # Attention_mask is optional for pre-training as a casual mask is generated automatically.
        batch["position_ids"],
        batch.get("cu_seqlens"),
        batch.get("cu_seqlens_argmin"),
        batch.get("max_seqlen"),
        batch.get("cu_seqlens_unpadded"),
        batch.get("cu_seqlens_unpadded_argmin"),
    )


def _forward_step_common(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and loss mask
    """
    timers = state.timers
    straggler_timer = state.straggler_timer

    config = get_model_config(model)
    pg_collection = get_pg_collection(model)
    use_mtp = (getattr(config, "mtp_num_layers", None) or 0) > 0

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            cu_seqlens,
            cu_seqlens_argmin,
            max_seqlen,
            cu_seqlens_unpadded,
            cu_seqlens_unpadded_argmin,
        ) = get_batch(
            data_iterator,
            state.cfg,
            use_mtp,
            pg_collection=pg_collection,
            vp_stage=_model_chunk_vp_stage(model),
        )
    timers("batch-generator").stop()

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Add packed sequence support
    if cu_seqlens is not None:
        packed_seq_params = {
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_argmin": cu_seqlens_argmin,
            "max_seqlen": max_seqlen,
            "cu_seqlens_unpadded": cu_seqlens_unpadded,
            "cu_seqlens_unpadded_argmin": cu_seqlens_unpadded_argmin,
        }
        # total_tokens drives seq_idx computation in PackedSeqParams.__post_init__,
        # which is only needed for Mamba/hybrid SSM layers. Skip it for pure
        # transformer models to avoid per-step CUDA overhead.
        if getattr(config, "is_hybrid_model", False):
            if tokens is not None:
                packed_seq_params["total_tokens"] = tokens.size(1)
            elif labels is not None:
                packed_seq_params["total_tokens"] = labels.size(1)
            else:
                packed_seq_params["total_tokens"] = getattr(config, "seq_length", None)
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_params)

    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            schedule_plan = model.build_schedule_plan(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )
            return schedule_plan, loss_mask
        else:
            output_tensor = model(**forward_args)

    return output_tensor, loss_mask


def forward_step(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = _create_loss_function(
        loss_mask,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )

    return output, loss_function


def _create_loss_function(loss_mask: torch.Tensor, check_for_nan_in_loss: bool, check_for_spiky_loss: bool) -> partial:
    """Create a partial loss function with the specified configuration.

    Args:
        loss_mask: Used to mask out some portions of the loss
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        A partial function that can be called with output_tensor to compute the loss
    """
    return partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )


def forward_step_modelopt(
    state: GlobalState, data_iterator: Iterable, model: GPTModel, return_schedule_plan: bool = False
) -> tuple[torch.Tensor, partial]:
    """Forward training step with ModelOpt required modifications.

    Args:
        state: Global state for the run
        data_iterator: Input data iterator
        model: The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor

    Returns:
        tuple containing the output tensor and the loss function
    """
    output, loss_mask = _forward_step_common(state, data_iterator, model, return_schedule_plan)

    loss_function = _create_loss_function_modelopt(
        loss_mask,
        model,
        check_for_nan_in_loss=state.cfg.rerun_state_machine.check_for_nan_in_loss,
        check_for_spiky_loss=state.cfg.rerun_state_machine.check_for_spiky_loss,
    )

    return output, loss_function


def _create_loss_function_modelopt(
    loss_mask: torch.Tensor, model: GPTModel, check_for_nan_in_loss: bool, check_for_spiky_loss: bool
) -> partial:
    """Create a partial loss function with the specified configuration.

    Kept here for backward compatibility with tests and callers that patch
    `megatron.bridge.training.gpt_step.masked_next_token_loss`.

    Args:
        loss_mask: Used to mask out some portions of the loss
        model: The GPT Model
        check_for_nan_in_loss: Whether to check for NaN values in the loss
        check_for_spiky_loss: Whether to check for spiky loss values

    Returns:
        A partial function that can be called with output_tensor to compute the loss
    """
    mnt_loss_func = partial(
        masked_next_token_loss,
        loss_mask,
        check_for_nan_in_loss=check_for_nan_in_loss,
        check_for_spiky_loss=check_for_spiky_loss,
    )
    unwrapped_model = unwrap_model(model)
    if isinstance(unwrapped_model, mtd.DistillationModel):
        return partial(loss_func_kd, loss_mask=loss_mask, original_loss_fn=mnt_loss_func, model=unwrapped_model)
    else:
        return mnt_loss_func
