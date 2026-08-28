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
from functools import partial, wraps
from typing import Iterable

import modelopt.torch.distill as mtd
import torch
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_model_config,
    get_pg_rank,
    get_pg_size,
    unwrap_model,
)

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.losses import masked_next_token_loss
from megatron.bridge.training.post_training.distillation import loss_func_kd
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.utils.flop_utils import accumulate_flops_metadata, get_model_chunk_vp_stage
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params, get_thd_cp_partition_indices
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.utils.cuda_graph import cuda_graph_module_names


logger = logging.getLogger(__name__)


_CURRENT_PACKED_SEQ_DEVICE_KEYS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
_CURRENT_PACKED_SEQ_HOST_KEYS = ("max_seqlen_q", "max_seqlen_kv", "pad_between_seqs")
_CURRENT_PACKED_SEQ_PARAM_KEYS = (*_CURRENT_PACKED_SEQ_DEVICE_KEYS, *_CURRENT_PACKED_SEQ_HOST_KEYS, "total_tokens")
_LEGACY_PACKED_SEQ_DEVICE_KEYS = ("cu_seqlens", "cu_seqlens_unpadded")
_LEGACY_PACKED_SEQ_HOST_KEYS = ("cu_seqlens_argmin", "max_seqlen", "cu_seqlens_unpadded_argmin")
_LEGACY_PACKED_SEQ_PARAM_KEYS = (*_LEGACY_PACKED_SEQ_DEVICE_KEYS, *_LEGACY_PACKED_SEQ_HOST_KEYS, "total_tokens")
_PackedMetadataValue = torch.Tensor | int | None
_MCORE_EXPERT_BIAS_PADDING_MASK_PATCHED = "_mbridge_expert_bias_padding_mask_compatible"
_MCORE_SCHEDULE_PADDING_MASK_PATCHED = "_mbridge_schedule_padding_mask_compatible"


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


def _has_packed_sequence_metadata(batch: dict[str, torch.Tensor]) -> bool:
    """Return whether a dataloader batch contains packed-sequence metadata."""
    return batch.get("cu_seqlens_q") is not None or batch.get("cu_seqlens") is not None


def _packed_metadata_for_forward(batch: dict[str, torch.Tensor]) -> dict[str, _PackedMetadataValue] | None:
    """Extract packed-sequence metadata needed by the forward step."""
    if batch.get("cu_seqlens_q") is not None:
        metadata: dict[str, _PackedMetadataValue] = {
            key: batch[key] for key in _CURRENT_PACKED_SEQ_PARAM_KEYS if batch.get(key) is not None
        }
        if batch.get("padding_mask") is not None:
            metadata["padding_mask"] = batch["padding_mask"]
        return metadata
    if batch.get("cu_seqlens") is not None:
        return {key: batch[key] for key in _LEGACY_PACKED_SEQ_PARAM_KEYS if batch.get(key) is not None}
    return None


def _patch_mcore_expert_bias_padding_mask() -> None:
    """Adapt MCore expert-bias routing to accept a flat padding mask.

    TODO(https://github.com/NVIDIA/Megatron-LM/issues/6111): Remove this
    compatibility patch after the MCore dev fix reaches the pinned main commit.
    """
    current_apply_expert_bias = TopKRouter._apply_expert_bias
    if getattr(current_apply_expert_bias, _MCORE_EXPERT_BIAS_PADDING_MASK_PATCHED, False):
        return

    def _apply_expert_bias(
        self: TopKRouter,
        routing_map: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> None:
        if padding_mask is not None:
            flat_mask = padding_mask.reshape(-1)
            assert flat_mask.shape[0] == routing_map.shape[0], (
                f"padding_mask flat {flat_mask.shape} vs routing_map {routing_map.shape}"
            )
            padding_mask = flat_mask.unsqueeze(-1)
        current_apply_expert_bias(self, routing_map, padding_mask=padding_mask)

    setattr(_apply_expert_bias, _MCORE_EXPERT_BIAS_PADDING_MASK_PATCHED, True)
    TopKRouter._apply_expert_bias = _apply_expert_bias


def _patch_mcore_schedule_plan_padding_mask() -> None:
    """Forward the schedule-plan chunk mask into the pinned MCore MoE router.

    TODO: Remove this compatibility patch after MCore dev's schedule-plan
    padding-mask forwarding reaches the pinned main commit.
    """
    try:
        from megatron.core.models.common import fine_grained_callables
    except ImportError:
        from megatron.core.models.gpt import fine_grained_callables

    current_builder = fine_grained_callables.build_transformer_layer_callables
    if getattr(current_builder, _MCORE_SCHEDULE_PADDING_MASK_PATCHED, False):
        return

    @wraps(current_builder)
    def build_transformer_layer_callables(layer):
        forward_funcs, backward_dw = current_builder(layer)
        if not hasattr(layer.mlp, "route"):
            return forward_funcs, backward_dw

        forward_funcs = list(forward_funcs)
        current_pre_dispatch = forward_funcs[0]

        @wraps(current_pre_dispatch)
        def pre_dispatch_with_padding_mask(node, *args, **kwargs):
            chunk_padding_mask = getattr(node.chunk_state, "padding_mask", None)
            if chunk_padding_mask is None:
                return current_pre_dispatch(node, *args, **kwargs)

            had_instance_route = "route" in layer.mlp.__dict__
            instance_route = layer.mlp.__dict__.get("route")
            current_route = layer.mlp.route

            @wraps(current_route)
            def route_with_padding_mask(hidden_states, padding_mask=None, *args, **kwargs):
                if padding_mask is None:
                    padding_mask = chunk_padding_mask
                return current_route(hidden_states, padding_mask, *args, **kwargs)

            layer.mlp.route = route_with_padding_mask
            try:
                return current_pre_dispatch(node, *args, **kwargs)
            finally:
                if had_instance_route:
                    layer.mlp.route = instance_route
                else:
                    delattr(layer.mlp, "route")

        forward_funcs[0] = pre_dispatch_with_padding_mask
        return forward_funcs, backward_dw

    setattr(build_transformer_layer_callables, _MCORE_SCHEDULE_PADDING_MASK_PATCHED, True)
    fine_grained_callables.build_transformer_layer_callables = build_transformer_layer_callables


def _validate_packed_moe_cuda_graph(config) -> None:
    """Reject router-scoped TE graphs that cannot consume packed padding masks."""
    if (
        not getattr(config, "num_moe_experts", None)
        or getattr(config, "cuda_graph_impl", "none") != "transformer_engine"
    ):
        return

    graph_modules = set(cuda_graph_module_names(config))
    captures_router = not graph_modules or bool(graph_modules & {"moe", "moe_router", "moe_preprocess"})
    if captures_router:
        raise ValueError(
            "Packed MoE padding masks do not support router-scoped Transformer Engine CUDA graphs in the pinned "
            "MCore. Use an attention-only CUDA graph scope or disable scoped CUDA graphs."
        )


def _prepare_packed_padding_mask(
    padding_mask: torch.Tensor | None,
    *,
    config,
    model: GPTModel,
    pg_collection,
) -> torch.Tensor | None:
    """Prepare an alignment-padding mask for the current model stage."""
    if padding_mask is None:
        return None
    if getattr(config, "moe_router_enable_expert_bias", False):
        _patch_mcore_expert_bias_padding_mask()

    # A pre-process GPT stage scatters this mask alongside its embeddings. HybridModel
    # scatters only its embeddings in the pinned MCore, while later PP stages already
    # receive SP-local activations, so both paths need the matching local mask here.
    needs_sp_scatter = not unwrap_model(model).pre_process or getattr(config, "is_hybrid_model", False)
    if getattr(config, "sequence_parallel", False) and needs_sp_scatter:
        padding_mask = (
            tensor_parallel.scatter_to_sequence_parallel_region(
                padding_mask.transpose(0, 1).contiguous(), group=pg_collection.tp
            )
            .transpose(0, 1)
            .contiguous()
        )
    return padding_mask


def _cu_seqlens_for_cp_partition(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the cu-seqlens tensor TE should use to partition packed THD tokens."""
    if batch.get("cu_seqlens_q") is not None:
        cu_seqlens = batch.get("cu_seqlens_q_padded")
        if cu_seqlens is None:
            cu_seqlens = batch["cu_seqlens_q"]
        if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
            raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
        return cu_seqlens.squeeze()

    cu_seqlens = batch["cu_seqlens"]
    if cu_seqlens.dim() > 1 and cu_seqlens.size(0) != 1:
        raise ValueError("Packed THD batches expect micro-batch size 1 for context-parallel slicing (THD layout)")
    cu_seqlens = cu_seqlens.squeeze()
    return _trim_padded_cu_seqlens_for_cp(cu_seqlens, batch.get("cu_seqlens_argmin"))


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


def _partition_packed_batch_for_cp(
    batch: dict[str, torch.Tensor], cp_group: torch.distributed.ProcessGroup
) -> dict[str, torch.Tensor]:
    """Partition THD/packed batches across context-parallel ranks.

    Uses MCore's packed-sequence partitioning to slice sequence dimensions
    aligned with packed cu_seqlens.
    """
    cu_seqlens = _cu_seqlens_for_cp_partition(batch)

    skip_keys = {
        "cu_seqlens",
        "cu_seqlens_unpadded",
        "cu_seqlens_argmin",
        "cu_seqlens_unpadded_argmin",
        "max_seqlen",
        "cu_seqlens_q",
        "cu_seqlens_kv",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "max_seqlen_q",
        "max_seqlen_kv",
        "pad_between_seqs",
        "token_count",
        # THD/packed attention is driven by cu_seqlens (PackedSeqParams), so the dense
        # attention_mask is unused here. It is also not sequence-partitionable: it is
        # either None or a degenerate placeholder without a slice-able seq dim at index 1.
        "attention_mask",
    }

    indices: dict[tuple[int, torch.device], torch.Tensor] = {}
    for key, val in batch.items():
        if val is None or key in skip_keys:
            continue
        index_key = (val.size(1), val.device)
        if index_key not in indices:
            indices[index_key] = get_thd_cp_partition_indices(
                cu_seqlens,
                total_tokens=val.size(1),
                cp_group=cp_group,
                device=val.device,
            )
        batch[key] = val.index_select(1, indices[index_key])

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

    if "cu_seqlens_q" in batch:
        required_device_keys.update(key for key in _CURRENT_PACKED_SEQ_DEVICE_KEYS if key in batch)
        required_host_keys.update(key for key in _CURRENT_PACKED_SEQ_HOST_KEYS if key in batch)
        if batch.get("padding_mask") is not None:
            required_device_keys.add("padding_mask")
    elif "cu_seqlens" in batch:
        required_device_keys.update(key for key in _LEGACY_PACKED_SEQ_DEVICE_KEYS if key in batch)
        required_host_keys.update(key for key in _LEGACY_PACKED_SEQ_HOST_KEYS if key in batch)

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
            _batch_required_keys[key] = val.cpu() if isinstance(val, torch.Tensor) else val
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
    dict[str, _PackedMetadataValue] | None,
]:
    """Generate a batch.

    Args:
        data_iterator: Input data iterator
        cfg: Configuration container
        use_mtp: Whether Multi-Token Prediction layers are enabled
        vp_stage: Virtual pipeline stage for the current model chunk.

    Returns:
        tuple of tensors containing tokens, labels, loss_mask, attention_mask,
        position_ids, and optional packed-sequence metadata.
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
        return None, None, None, None, None, None

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
    has_packed = _has_packed_sequence_metadata(batch)
    if has_packed and cp_size > 1:
        batch = _partition_packed_batch_for_cp(batch, pg_collection.cp)
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
        _packed_metadata_for_forward(batch),
    )


def _forward_step_common(
    state: GlobalState,
    data_iterator: Iterable,
    model: GPTModel,
    return_schedule_plan: bool = False,
    *,
    _get_batch_fn=None,
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
    vp_stage = get_model_chunk_vp_stage(model)

    timers("batch-generator", log_level=2).start()
    with straggler_timer(bdata=True):
        (
            tokens,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            packed_seq_metadata,
        ) = (_get_batch_fn or get_batch)(
            data_iterator,
            state.cfg,
            use_mtp,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
    timers("batch-generator").stop()

    # Accumulate FLOPS metadata across micro-batches. The THD attention term Σᵢ sᵢ² is
    # derived inline from cu_seqlens (kept on-device, sync-free); see
    # accumulate_flops_metadata. Falls back to BSHD when cu_seqlens is absent.
    #
    # The cu_seqlens-driven THD path is only wired/validated for CP == 1 in this PR.
    # Under context parallelism the batch (and its cu_seqlens) is CP-partitioned per
    # rank, so the per-rank Σᵢ sᵢ² accounting here is not yet correct — that is the
    # follow-up tracked in #4161. Until then, forward cu_seqlens only for CP == 1 so
    # CP > 1 stays on the BSHD term (the behavior this test passed on before the THD
    # change), instead of running the not-yet-CP-safe cu_seqlens path.
    cp_use_thd = pg_collection.cp.size() == 1
    cu_seqlens = None
    cu_seqlens_argmin = None
    cu_seqlens_unpadded = None
    cu_seqlens_unpadded_argmin = None
    if packed_seq_metadata is not None:
        if packed_seq_metadata.get("cu_seqlens_q") is not None:
            cu_seqlens_q = packed_seq_metadata.get("cu_seqlens_q")
            cu_seqlens_q_padded = packed_seq_metadata.get("cu_seqlens_q_padded")
            cu_seqlens = cu_seqlens_q_padded if cu_seqlens_q_padded is not None else cu_seqlens_q
            cu_seqlens_unpadded = cu_seqlens_q if cu_seqlens_q_padded is not None else None
        else:
            cu_seqlens = packed_seq_metadata.get("cu_seqlens")
            cu_seqlens_argmin = packed_seq_metadata.get("cu_seqlens_argmin")
            cu_seqlens_unpadded = packed_seq_metadata.get("cu_seqlens_unpadded")
            cu_seqlens_unpadded_argmin = packed_seq_metadata.get("cu_seqlens_unpadded_argmin")
    accumulate_flops_metadata(
        state,
        tokens,
        vp_stage=vp_stage,
        config_seq_len=getattr(config, "seq_length", None),
        cu_seqlens=cu_seqlens if cp_use_thd else None,
        cu_seqlens_argmin=cu_seqlens_argmin if cp_use_thd else None,
        cu_seqlens_unpadded=cu_seqlens_unpadded if cp_use_thd else None,
        cu_seqlens_unpadded_argmin=cu_seqlens_unpadded_argmin if cp_use_thd else None,
    )

    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    # Add packed sequence support
    if packed_seq_metadata is not None:
        padding_mask = packed_seq_metadata.get("padding_mask")
        # Supported producers make mask presence configuration-driven and stable
        # across DP ranks. Never branch on mask contents here: an all-false local
        # batch must follow the same path as a padded batch on another rank.
        if padding_mask is not None:
            _validate_packed_moe_cuda_graph(config)
        padding_mask = _prepare_packed_padding_mask(
            padding_mask,
            config=config,
            model=model,
            pg_collection=pg_collection,
        )
        packed_seq_metadata = {key: value for key, value in packed_seq_metadata.items() if key != "padding_mask"}
        # total_tokens drives seq_idx computation in PackedSeqParams.__post_init__,
        # which is only needed for Mamba/hybrid SSM layers. Skip it for pure
        # transformer models to avoid per-step CUDA overhead.
        if getattr(config, "is_hybrid_model", False):
            if tokens is not None:
                packed_seq_metadata["total_tokens"] = tokens.size(1)
            elif labels is not None:
                packed_seq_metadata["total_tokens"] = labels.size(1)
            else:
                packed_seq_metadata["total_tokens"] = getattr(config, "seq_length", None)
        forward_args["packed_seq_params"] = get_packed_seq_params(packed_seq_metadata)
        if padding_mask is not None:
            forward_args["padding_mask"] = padding_mask

    with straggler_timer:
        if return_schedule_plan:
            assert config.overlap_moe_expert_parallel_comm, (
                "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
            )
            if forward_args.get("padding_mask") is not None:
                _patch_mcore_schedule_plan_padding_mask()
            schedule_plan = model.build_schedule_plan(
                tokens,
                position_ids,
                attention_mask,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=forward_args.get("packed_seq_params"),
                padding_mask=forward_args.get("padding_mask"),
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
