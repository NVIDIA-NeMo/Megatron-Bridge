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

import inspect
import logging
import math
import re
from contextlib import nullcontext
from dataclasses import dataclass, fields
from functools import cache
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import packaging
import torch
import torch.nn as nn
from megatron.core import ModelParallelConfig, dist_checkpointing, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict, ShardedTensor, ShardedTensorFactory
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear, set_tensor_model_parallel_attributes
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.random import is_checkpointing
from megatron.core.transformer.mlp import apply_swiglu_sharded_factory
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.utils import get_pg_rank, get_pg_size

from megatron.bridge.utils.activation_map import str_to_dtype
from megatron.bridge.utils.import_utils import safe_import_from


logger = logging.getLogger(__name__)

ModelList = list[MegatronModule]
ModelHook = Callable[[ModelList], ModelList | None]
CheckpointPath = str | Path

_LEGACY_SHARED_EXPERT_ADAPTER_CHECKPOINT_ATTR = "use_legacy_shared_expert_adapter_checkpoint"


TEColumnParallelLinear, HAVE_TE_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelLinear"
)
TELayerNormColumnParallelLinear, HAVE_TE_LN_COL_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine",
    "TELayerNormColumnParallelLinear",
)
TEColumnParallelGroupedLinear, HAVE_TE_COL_GRP_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TEColumnParallelGroupedLinear"
)
TEFP8GlobalStateManager, HAVE_TE_FP8_GLOBAL_STATE_MANAGER = safe_import_from(
    "megatron.core.extensions.transformer_engine", "FP8GlobalStateManager"
)
TEPytorchGroupedLinear, HAVE_TE_PYTORCH_GROUPED_LINEAR = safe_import_from(
    "transformer_engine.pytorch", "GroupedLinear"
)
TEPytorchGroupedLinearAutograd, HAVE_TE_PYTORCH_GROUPED_LINEAR_AUTOGRAD = safe_import_from(
    "transformer_engine.pytorch.module.grouped_linear", "_GroupedLinear"
)
TEPytorchIsCPUOffloadEnabled, HAVE_TE_PYTORCH_CPU_OFFLOAD_STATUS = safe_import_from(
    "transformer_engine.pytorch.cpu_offload", "is_cpu_offload_enabled"
)
TERowParallelLinear, HAVE_TE_ROW_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelLinear"
)
TERowParallelGroupedLinear, HAVE_TE_ROW_GRP_LINEAR = safe_import_from(
    "megatron.core.extensions.transformer_engine", "TERowParallelGroupedLinear"
)
TELinear, HAVE_TE_LINEAR = safe_import_from("megatron.core.extensions.transformer_engine", "TELinear")
HAVE_TE = all(
    (
        HAVE_TE_COL_LINEAR,
        HAVE_TE_LN_COL_LINEAR,
        HAVE_TE_ROW_LINEAR,
        HAVE_TE_LINEAR,
        HAVE_TE_COL_GRP_LINEAR,
        HAVE_TE_ROW_GRP_LINEAR,
    )
)


@cache
def _te_grouped_linear_uses_explicit_m_splits(
    autograd_function: type[torch.autograd.Function],
) -> bool:
    """Return whether TE's grouped-linear autograd takes a separate splits tensor."""

    return "m_splits" in inspect.signature(autograd_function.forward).parameters


def _get_pg_collection_from_module(module: object | None) -> ProcessGroupCollection | None:
    """Return the process-group collection attached to a module or its config."""

    for owner in (module, getattr(module, "config", None)):
        if owner is None:
            continue
        for attr in ("pg_collection", "_pg_collection"):
            pg_collection = getattr(owner, attr, None)
            if pg_collection is not None:
                return pg_collection
    return None


def _get_pg_collection(
    pg_collection: ProcessGroupCollection | None = None,
    source: object | None = None,
    *,
    required_pgs: List[str],
) -> ProcessGroupCollection | None:
    """Return the explicit PG collection or MCore's default collection fallback."""

    pg_collection = pg_collection or _get_pg_collection_from_module(source)
    if pg_collection is None:
        # TODO: Once LoRA/DoRA transforms carry the model-level ProcessGroupCollection,
        # pass it into adapter constructors explicitly and remove this default-MPU fallback.
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=required_pgs)
    return pg_collection


def _iter_sharded_tensor_factories(state_dict: object) -> list[ShardedTensorFactory]:
    """Return all sharded tensor factories in a nested state dict."""

    if isinstance(state_dict, ShardedTensorFactory):
        return [state_dict]
    if isinstance(state_dict, Mapping):
        factories = []
        for value in state_dict.values():
            factories.extend(_iter_sharded_tensor_factories(value))
        return factories
    if isinstance(state_dict, list | tuple):
        factories = []
        for value in state_dict:
            factories.extend(_iter_sharded_tensor_factories(value))
        return factories
    return []


def _checkpoint_tensor_shape(checkpoint_metadata: Mapping[str, ShardedTensor], key: str) -> tuple[int, ...] | None:
    """Return checkpoint global tensor shape for a key, tolerating model-section prefixes."""

    for candidate in (key, f"model.{key}"):
        metadata = checkpoint_metadata.get(candidate)
        if metadata is not None:
            return tuple(metadata.global_shape)
    return None


def _legacy_shared_expert_adapter_key(factory: ShardedTensorFactory) -> str | None:
    """Return the adapter module key if a factory represents a shared expert LoRA tensor."""

    for suffix in (".linear_in.weight", ".linear_out.weight"):
        if not factory.key.endswith(suffix):
            continue
        built = factory.build()
        shards = built if isinstance(built, list) else [built]
        if not shards or not isinstance(shards[0], ShardedTensor):
            continue
        expected_shape = tuple(shards[0].global_shape)
        local_shape = tuple(factory.data.shape)
        if len(expected_shape) == len(local_shape) + 1:
            return factory.key[: -len(suffix)]
    return None


def _legacy_shared_expert_adapter_matches(
    adapters_by_name: Mapping[str, "ParallelLinearAdapter"], adapter_key: str
) -> list["ParallelLinearAdapter"]:
    """Return adapter modules matching a legacy shared-expert checkpoint key."""

    adapter = adapters_by_name.get(adapter_key)
    if adapter is not None:
        return [adapter]

    adapter_base_key = adapter_key.removesuffix(".adapter")
    matched_adapters = []
    for module_name, module in adapters_by_name.items():
        module_base_key = module_name.removesuffix(".adapter")
        base_linear_name = module.base_linear_name
        if (
            adapter_key.endswith(module_name)
            or module_name.endswith(adapter_key)
            or adapter_base_key.endswith(module_base_key)
            or module_base_key.endswith(adapter_base_key)
            or adapter_base_key.endswith(base_linear_name)
            or base_linear_name.endswith(adapter_base_key)
        ):
            matched_adapters.append(module)

    if matched_adapters:
        return matched_adapters

    return list(adapters_by_name.values())


def enable_legacy_shared_expert_adapter_loading(
    megatron_model: list[nn.Module] | nn.Module,
    sharded_state_dict: ShardedStateDict,
    checkpoint_path: str | Path,
) -> bool:
    """Enable legacy 2D checkpoint loading for old shared grouped-expert adapters.

    New shared grouped-expert LoRA checkpoints expose a leading global expert axis
    so they can be resharded across EP changes. Older checkpoints saved the same
    shared adapter as a plain 2D tensor. This helper detects that old metadata
    shape and marks only the matching shared adapter modules to emit the legacy
    2D sharded state dict for loading.

    Args:
        megatron_model: Model module or model chunks containing PEFT adapters.
        sharded_state_dict: Current adapter-only sharded state dict.
        checkpoint_path: Distributed checkpoint directory to inspect.

    Returns:
        True if at least one shared expert adapter was marked for legacy loading.
    """

    checkpoint_metadata = dist_checkpointing.load_tensors_metadata(str(checkpoint_path))
    models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    adapters_by_name: dict[str, ParallelLinearAdapter] = {}
    for model in models:
        for name, module in model.named_modules():
            if isinstance(module, ParallelLinearAdapter) and module._uses_grouped_expert_sharding():
                adapters_by_name[name.removeprefix("module.")] = module

    enabled = False
    for factory in _iter_sharded_tensor_factories(sharded_state_dict):
        adapter_key = _legacy_shared_expert_adapter_key(factory)
        if adapter_key is None:
            continue
        built = factory.build()
        shards = built if isinstance(built, list) else [built]
        expected_shape = tuple(shards[0].global_shape)
        legacy_shape = expected_shape[1:]
        if _checkpoint_tensor_shape(checkpoint_metadata, factory.key) == legacy_shape:
            for adapter in _legacy_shared_expert_adapter_matches(adapters_by_name, adapter_key):
                setattr(adapter, _LEGACY_SHARED_EXPERT_ADAPTER_CHECKPOINT_ATTR, True)
                enabled = True

    return enabled


def _get_process_group(pg_collection: ProcessGroupCollection | None, *names: str) -> object | None:
    """Return the first named process group available on a collection."""

    if pg_collection is None:
        return None
    for name in names:
        group = getattr(pg_collection, name, None)
        if group is not None:
            return group
    return None


def _process_group_size(group: object | None, fallback: int = 1) -> int:
    """Return a process-group size without consulting global parallel state."""

    if group is None:
        return int(fallback or 1)
    size = None
    size_attr = getattr(group, "size", None)
    try:
        size = size_attr() if callable(size_attr) else size_attr
    except (RuntimeError, ValueError, TypeError):
        size = None
    if size is None:
        try:
            size = get_pg_size(group)
        except (RuntimeError, ValueError, TypeError):
            size = None
    return int(size if size is not None else fallback or 1)


def _process_group_rank(group: object | None, fallback: int = 0) -> int:
    """Return this rank within a process group without consulting global parallel state."""

    if group is None:
        return int(fallback or 0)
    rank = None
    rank_attr = getattr(group, "rank", None)
    try:
        rank = rank_attr() if callable(rank_attr) else rank_attr
    except (RuntimeError, ValueError, TypeError):
        rank = None
    if rank is None:
        try:
            rank = get_pg_rank(group)
        except (RuntimeError, ValueError, TypeError):
            rank = None
    return int(rank if rank is not None else fallback or 0)


def _get_tensor_parallel_group(
    pg_collection: ProcessGroupCollection | None, *, is_expert: bool = False
) -> object | None:
    """Return the tensor-parallel group for dense or expert linear layers."""

    if is_expert:
        return _get_process_group(pg_collection, "expt_tp", "etp")
    return _get_process_group(pg_collection, "tp")


def _get_tensor_parallel_group_from_module(
    module: nn.Module, *, is_expert: bool = False, pg_collection: ProcessGroupCollection | None = None
) -> object | None:
    """Return the TP group passed to the wrapped module, falling back to its collection."""

    pg_collection = _get_pg_collection(
        pg_collection,
        module,
        required_pgs=["expt_tp"] if is_expert else ["tp"],
    )
    group = _get_tensor_parallel_group(pg_collection, is_expert=is_expert)
    if group is not None:
        return group
    return getattr(module, "_tp_group", None) or getattr(module, "tp_group", None)


MixedFusedLayerNorm, HAVE_APEX = safe_import_from("apex.normalization.fused_layer_norm", "MixedFusedLayerNorm")
ModelOptLinear, HAVE_MODELOPT_LINEAR = safe_import_from("megatron.core.post_training.modelopt.layers", "Linear")

TECL = (TEColumnParallelLinear, TELayerNormColumnParallelLinear, TEColumnParallelGroupedLinear)
TERL = (TERowParallelLinear, TERowParallelGroupedLinear)


def create_peft(config: Mapping[str, Any], dtype: torch.dtype | str | int | None = None) -> object | None:
    """Create a Bridge PEFT object from a small config mapping."""
    kwargs = dict(config)
    peft_type = kwargs.pop("type", "lora")
    if "rank" in kwargs:
        kwargs["dim"] = kwargs.pop("rank")
    if kwargs.get("dim", 0) <= 0:
        return None

    peft_cls = _import_peft_class(peft_type)

    peft_fields = {field.name for field in fields(peft_cls) if field.init}
    config_dtype = kwargs.pop("dtype", None)
    if "lora_dtype" not in kwargs:
        kwargs["lora_dtype"] = config_dtype if config_dtype is not None else dtype

    if kwargs.get("lora_dtype") is None or "lora_dtype" not in peft_fields:
        kwargs.pop("lora_dtype", None)
    else:
        kwargs["lora_dtype"] = str_to_dtype(str(kwargs["lora_dtype"]).lower())

    kwargs = {key: value for key, value in kwargs.items() if key in peft_fields}

    return peft_cls(**kwargs)


def create_peft_hook(
    peft: object,
    training: bool = True,
) -> ModelHook:
    """Create a provider pre-wrap hook that applies PEFT."""

    def hook(model: ModelList) -> ModelList:
        model = _apply_peft(peft, model, training=training)

        return model

    return hook


def load_peft_adapter_checkpoint(
    model: ModelList | MegatronModule,
    adapter_checkpoint_path: CheckpointPath,
    peft: object,
    strict: bool = False,
    model_sd_kwargs: Mapping[str, object] | None = None,
    ckpt_format: str = "torch_dist",
    pg_collection: ProcessGroupCollection | None = None,
    fully_parallel_load: bool = True,
    load_strategy: object | None = None,
) -> None:
    """Load a PEFT adapter checkpoint into an already transformed model."""
    from megatron.core import dist_checkpointing
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper
    from megatron.core.dist_checkpointing.strategies.torch import TorchDistLoadShardedStrategy

    from megatron.bridge.training.checkpointing import apply_peft_adapter_filter_to_state_dict

    model_chunks = _ensure_model_list(model)
    sharded_state_dict = _model_state_dict(
        model_chunks,
        model_sd_kwargs,
        ckpt_format,
        pg_collection=pg_collection,
    )
    sharded_state_dict = apply_peft_adapter_filter_to_state_dict(sharded_state_dict, peft)

    checkpoint_path = str(adapter_checkpoint_path)
    if load_strategy is None:
        load_strategy = TorchDistLoadShardedStrategy()
        if pg_collection is None and fully_parallel_load:
            try:
                pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["dp_cp"])
            except AssertionError:
                pg_collection = None
        dp_cp_group = _get_process_group(pg_collection, "dp_cp")
        if fully_parallel_load and dp_cp_group is not None:
            load_strategy = FullyParallelLoadStrategyWrapper(load_strategy, dp_cp_group)

    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_path, load_strategy)
    for vpp_rank, model_chunk in enumerate(model_chunks):
        model_key = "model" if len(model_chunks) == 1 else f"model{vpp_rank}"
        if model_key not in loaded_state_dict:
            if len(model_chunks) == 1:
                fallback_model_key = next((key for key in loaded_state_dict if key.startswith("model")), None)
                if fallback_model_key is not None:
                    model_key = fallback_model_key
                else:
                    raise KeyError(
                        "Expected adapter checkpoint to contain a top-level 'model' or 'model*' key, "
                        f"but found keys: {list(loaded_state_dict.keys())}"
                    )
            else:
                expected_model_keys = [f"model{rank}" for rank in range(len(model_chunks))]
                raise KeyError(
                    f"Expected adapter checkpoint to contain top-level key {model_key!r} "
                    f"for virtual pipeline model chunk {vpp_rank} "
                    f"(expected keys: {expected_model_keys}), "
                    f"but found keys: {list(loaded_state_dict.keys())}"
                )
        model_chunk.load_state_dict(loaded_state_dict[model_key], strict=strict)


def _apply_peft(peft: object, model: ModelList, training: bool = True) -> ModelList:
    """Apply PEFT and mark adapter parameters for checkpointing."""
    transformed_model = peft(model, training=training)
    peft.set_params_to_save(transformed_model)
    return transformed_model


def _import_peft_class(peft_type: str) -> type[Any]:
    peft_classes = {
        "lora": ("megatron.bridge.peft.lora", "LoRA"),
        "vlm_lora": ("megatron.bridge.peft.lora", "VLMLoRA"),
        "canonical_lora": ("megatron.bridge.peft.canonical_lora", "CanonicalLoRA"),
        "dora": ("megatron.bridge.peft.dora", "DoRA"),
    }
    if peft_type not in peft_classes:
        supported_types = ", ".join(sorted(peft_classes))
        raise ValueError(f"Unsupported PEFT type {peft_type!r}. Supported types: {supported_types}.")

    module_name, class_name = peft_classes[peft_type]
    try:
        module = import_module(module_name)
    except ImportError as err:
        message = f"Failed to import PEFT type {peft_type!r} ({module_name}:{class_name})."
        if peft_type in {"lora", "vlm_lora", "canonical_lora"}:
            message += " Install Megatron Bridge with the [te] extra for Transformer Engine support."
        raise ImportError(message) from err

    return getattr(module, class_name)


def _model_state_dict(
    model: ModelList,
    model_sd_kwargs: Mapping[str, object] | None = None,
    ckpt_format: str = "torch_dist",
    pg_collection: ProcessGroupCollection | None = None,
) -> dict[str, Any]:
    """Generate Bridge model checkpoint sections for an external trainer."""
    from megatron.bridge.training.checkpointing import _generate_model_state_dict

    return _generate_model_state_dict(
        model,
        dict(model_sd_kwargs or {}),
        ckpt_format,
        pg_collection=pg_collection,
    )


def _ensure_model_list(model: ModelList | MegatronModule) -> ModelList:
    return model if isinstance(model, list) else [model]


def is_modelopt_linear(m: nn.Module) -> bool:
    """Return whether a module is ModelOpt's local Megatron Linear."""
    return HAVE_MODELOPT_LINEAR and isinstance(m, ModelOptLinear)


@dataclass(frozen=True)
class AdapterAttributes:
    """Container for base linear adapter attributes."""

    input_is_parallel: bool
    in_features: int
    out_features: int
    disable_tensor_parallel_comm: bool
    disable_sequence_parallel_comm: bool
    base_linear_is_parallel: bool


def get_adapter_attributes_from_linear(
    m: nn.Module,
    is_expert: bool = False,
    pg_collection: ProcessGroupCollection | None = None,
    sequence_parallel_input_regather: bool = False,
) -> AdapterAttributes:
    """Returns attributes from the base layer as an AdapterAttributes dataclass.

    input_is_parallel, in_features, out_features, disable_tensor_parallel_comm,
    disable_sequence_parallel_comm, base_linear_is_parallel

    This function analyzes a linear module and extracts key attributes needed for adapter configuration,
    particularly for PEFT adapters in distributed training scenarios.

    Args:
        m: The linear module to analyze (should have a config attribute).
        is_expert: Whether the linear belongs to an expert module.
        pg_collection: Optional process-group collection associated with the module.
        sequence_parallel_input_regather: Whether LoRA-A should re-gather its sequence-parallel input in backward.

    Returns:
        AdapterAttributes containing:
            - input_is_parallel: Whether the input is already parallelized
            - in_features: Input feature dimension
            - out_features: Output feature dimension
            - disable_tensor_parallel_comm: Whether to disable tensor parallel communication
            - disable_sequence_parallel_comm: Whether to disable sequence parallel communication
            - base_linear_is_parallel: Whether the base linear layer uses parallelization

    Raises:
        NotImplementedError: If the layer type is not recognized for LoRA adaptation.
    """
    disable_sequence_parallel_comm = not m.config.sequence_parallel
    base_linear_is_parallel = True

    # In some modules (notably MoE shared_experts when moe_shared_expert_overlap is enabled),
    # Megatron disables TP-related communications on the base linear layer by
    # setting `parallel_mode=None` (TE) or `explicit_expert_comm=True` (legacy).
    # https://github.com/NVIDIA/Megatron-LM/blob/5b1ef0703184299fbf71f6131bf2f9a5331e7238/megatron/core/transformer/moe/shared_experts.py#L95-L104
    # The weights are still TP-sharded though, so we must keep using the real TP size
    disable_tensor_parallel_comm = getattr(m, "parallel_mode", "") is None or getattr(m, "explicit_expert_comm", False)
    if disable_tensor_parallel_comm:
        disable_sequence_parallel_comm = True

    if is_modelopt_linear(m):
        return AdapterAttributes(
            input_is_parallel=False,
            in_features=m.in_features,
            out_features=m.out_features,
            disable_tensor_parallel_comm=False,
            disable_sequence_parallel_comm=True,
            base_linear_is_parallel=False,
        )

    tp_group = _get_tensor_parallel_group_from_module(m, is_expert=is_expert, pg_collection=pg_collection)
    tp_size = _process_group_size(
        tp_group,
        getattr(
            m.config,
            "expert_tensor_parallel_size" if is_expert else "tensor_model_parallel_size",
            1,
        ),
    )
    if isinstance(m, TopKRouter):
        input_is_parallel = False
        in_features = m.weight.shape[1]
        out_features = m.weight.shape[0]
        base_linear_is_parallel = False
        disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_column_parallel) for te_column_parallel in TECL):
        input_is_parallel = False
        # m.in_features and m.out_features are divided by tp_size already,
        # but in_features and out_features passed to ParallelLinearAdapter are not.
        in_features = m.in_features
        out_features = m.out_features * tp_size

        if isinstance(m, TELayerNormColumnParallelLinear):
            # LoRA is applied after layernorm, so layernorm output must be returned
            m.return_layernorm_output = True
            # perf optimization for LoRA + SP
            if hasattr(m, "ub_overlap_ag"):
                ub_overlap_ag = m.ub_overlap_ag
            elif hasattr(m, "ub_overlap_ag_fprop"):
                ub_overlap_ag = m.ub_overlap_ag_fprop
            else:
                ub_overlap_ag = False
            if hasattr(m, "config") and m.config.sequence_parallel and not ub_overlap_ag:
                # The adapter's MCore SP linear must own the gather so it can retain the local shard.
                # Preserve the existing TE optimization for the default path.
                m.return_layernorm_output_gathered = not sequence_parallel_input_regather
                te_version = packaging.version.Version(version("transformer-engine"))
                if (
                    not sequence_parallel_input_regather
                    and te_version >= packaging.version.Version("1.5.0dev")
                    and (
                        not getattr(m.config, "tp_comm_overlap", False)
                        or getattr(m.config, "tp_comm_overlap_disable_qkv", False)
                    )
                ):
                    # TE 1.5 introduces the option `return_layernorm_output_gathered`, so the all gather
                    # in the forward method is not needed, so disable sp communications
                    # unless TP communication overlap is used
                    disable_sequence_parallel_comm = True
    elif HAVE_TE and any(isinstance(m, te_row_parallel) for te_row_parallel in TERL):
        input_is_parallel = True
        in_features = m.in_features * tp_size
        out_features = m.out_features
    elif HAVE_TE and isinstance(m, TELinear):  # parallel_mode="duplicated"
        input_is_parallel = False
        in_features = m.in_features
        out_features = m.out_features
        base_linear_is_parallel = False
    elif isinstance(m, ColumnParallelLinear):
        input_is_parallel = False
        in_features = m.input_size
        out_features = m.output_size
    elif isinstance(m, RowParallelLinear):
        input_is_parallel = True
        in_features = m.input_size
        out_features = m.output_size
    else:
        raise NotImplementedError(f"Layer type is unrecognized for LoRA: {type(m)}")

    return AdapterAttributes(
        input_is_parallel=input_is_parallel,
        in_features=in_features,
        out_features=out_features,
        disable_tensor_parallel_comm=disable_tensor_parallel_comm,
        disable_sequence_parallel_comm=disable_sequence_parallel_comm,
        base_linear_is_parallel=base_linear_is_parallel,
    )


def is_expert_linear(fqn: str) -> bool:
    """Return whether the current base module is an expert linear module.

    This function checks if a fully qualified name (FQN) corresponds to an expert linear
    module in a Mixture of Experts (MoE) architecture.

    Args:
        fqn: Fully qualified name of the module.

    Returns:
        True if the module is an expert linear module, False otherwise.

    Example:
        >>> is_expert_linear("model.layers.0.mlp.experts.0.linear_fc1")
        True
        >>> is_expert_linear("model.layers.0.mlp.linear_fc1")
        False
    """
    return re.match(r".*mlp\..*experts.*\.linear_fc[1-2]$", fqn) is not None and ".shared_experts." not in fqn


def is_grouped_expert_linear(fqn: str) -> bool:
    """Return whether the current base module is a grouped expert linear module."""

    return is_expert_linear(fqn) and ".local_experts." not in fqn


def get_effective_lora_dim(module: nn.Module, *, dim: int, normalize_moe_lora: bool, is_expert: bool) -> int:
    """Return the LoRA rank to use, reduced for expert layers when ``normalize_moe_lora`` is enabled."""

    if not normalize_moe_lora or not is_expert:
        return dim
    topk = module.config.moe_router_topk
    if topk is None or topk <= 0:
        raise ValueError(
            f"normalize_moe_lora is enabled but moe_router_topk is {topk!r}; "
            f"it must be set to a positive integer on the model config"
        )
    if dim % topk != 0:
        raise ValueError(
            f"LoRA dim={dim} must be divisible by moe_router_topk={topk} when normalize_moe_lora is enabled"
        )
    return dim // topk


def align_expert_dim_for_tp(
    module: nn.Module,
    dim: int,
    *,
    normalize_moe_lora: bool,
    is_expert: bool,
    input_is_parallel: bool,
    pg_collection: ProcessGroupCollection | None = None,
) -> int:
    """Round normalized expert LoRA ranks up to the expert-TP granularity when needed."""

    if not normalize_moe_lora or not is_expert or input_is_parallel:
        return dim

    expert_tp_group = _get_tensor_parallel_group_from_module(module, is_expert=True, pg_collection=pg_collection)
    expert_tp_size = _process_group_size(expert_tp_group, module.config.expert_tensor_parallel_size or 1)
    if expert_tp_size <= 1 or dim % expert_tp_size == 0:
        return dim

    return ((dim + expert_tp_size - 1) // expert_tp_size) * expert_tp_size


def wildcard_match(pattern: str, key: Optional[str]) -> Optional[bool]:
    """Return whether the pattern (target module to add LoRA) matches the key (model weight name).

    This function performs wildcard matching using '*' as a placeholder for any substring.

    Args:
        pattern: Pattern string with wildcards (*) to match against.
        key: Key string to test against the pattern.

    Returns:
        True if the pattern matches the key, False if it doesn't, None if key is None.

    Example:
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.0.self_attention.linear_qkv")
        True
        >>> wildcard_match("*.layers.0.*.linear_qkv", "decoder.layers.1.self_attention.linear_qkv")
        False
    """
    if key is None:
        return None
    regex_pattern = re.compile("^" + pattern.replace("*", "(.*)") + "$")
    match = regex_pattern.match(key)
    return match is not None


def init_method_normal(sigma: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on normal distribution N(0, sigma).

    Args:
        sigma: Standard deviation for the normal distribution.

    Returns:
        Initialization function that applies normal distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def init_method_kaiming_uniform(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method based on Kaiming uniform distribution.

    Args:
        val: The 'a' parameter for Kaiming uniform initialization.

    Returns:
        Initialization function that applies Kaiming uniform distribution to a tensor.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.kaiming_uniform_(tensor, a=val)

    return init_


def init_method_const(val: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create an initialization method that sets all values to a constant.

    Args:
        val: Constant value to initialize the tensor with.

    Returns:
        Initialization function that sets tensor to constant value.
    """

    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return nn.init.constant_(tensor, val)

    return init_


def pad_seq_to_mult(x: torch.Tensor, mult: int) -> Tuple[torch.Tensor, int]:
    """Pad sequence length to be a multiple of mult.

    This function pads the first dimension of the tensor to ensure it's divisible by mult.
    Used primarily for MoE (Mixture of Experts) operations that require specific sequence lengths.

    Args:
        x: Input tensor to pad.
        mult: Multiple that the sequence length should be divisible by.

    Returns:
        A tuple containing:
            - Padded tensor
            - Number of padding elements added
    """
    if x.shape[0] % mult == 0:
        return x, 0
    pad_len = mult - (x.shape[0] % mult)
    x = nn.functional.pad(x, (0, 0, 0, pad_len))
    return x, pad_len


def unpad_seq_to_mult(x: torch.Tensor, pad_len: int) -> torch.Tensor:
    """Remove sequence padding that was added by pad_seq_to_mult.

    Args:
        x: Padded tensor to unpad.
        pad_len: Number of padding elements to remove from the end.

    Returns:
        Unpadded tensor with pad_len elements removed from the first dimension.
    """
    if pad_len <= 0:
        return x
    return x[:-pad_len, :]


class _All2AllHp2Sp(torch.autograd.Function):
    """All-2-All from Hidden Parallel to Sequence Parallel.

    This is a temporary workaround for distributed communication patterns and can be updated in the future.
    It performs all-to-all communication to transform from hidden parallel to sequence parallel layout.

    TODO: Move the functionality to MCore
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: object | None) -> torch.Tensor:
        """Forward pass: All-to-All from Hidden Parallel to Sequence Parallel.

        Args:
            ctx: Autograd context (unused but required by Function interface).
            input_: Input tensor in hidden parallel layout.

        Returns:
            Output tensor in sequence parallel layout.
        """
        ctx.group = group
        world_size = _process_group_size(group)
        send_list = list(input_.chunk(world_size, dim=0))
        send_list = [tensor.contiguous() for tensor in send_list]
        receive_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(receive_list, send_list, group=group)
        x = torch.cat(receive_list, dim=-1)

        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass: All-to-All from Sequence Parallel to Hidden Parallel.

        Args:
            ctx: Autograd context (unused but required by Function interface).
            grad_output: Gradient tensor in sequence parallel layout.

        Returns:
            Gradient tensor in hidden parallel layout.
        """
        group = ctx.group
        world_size = _process_group_size(group)
        send_list = list(grad_output.chunk(world_size, dim=-1))
        send_list = [tensor.contiguous() for tensor in send_list]
        receive_list = [torch.empty_like(send_list[0]) for _ in range(world_size)]
        torch.distributed.all_to_all(receive_list, send_list, group=group)
        x = torch.cat(receive_list, dim=0)

        return x, None


def all2all_hp2sp(input_: torch.Tensor, tensor_parallel_group: object | None = None) -> torch.Tensor:
    """Perform All-to-All communication from Hidden Parallel to Sequence Parallel.

    Args:
        input_: Input tensor in hidden parallel layout.

    Returns:
        Output tensor in sequence parallel layout.
    """
    return _All2AllHp2Sp.apply(input_, tensor_parallel_group)


class ParallelLinearAdapter(nn.Module):
    """Parallel Linear Adapter for Parameter-Efficient Fine-Tuning (PEFT) in distributed settings.

    This adapter implements a low-rank adaptation pattern using two linear layers with configurable
    parallelization strategies. It supports both tensor and sequence parallelism patterns used in
    large language model training.

    The adapter follows the pattern: input -> linear_in -> activation -> linear_out -> scaling
    where linear_in and linear_out are parallelized according to the base layer configuration.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        dim: Adapter bottleneck dimension (rank).
        base_linear_name: Name of the base linear layer being adapted.
        activation: Activation function name (default: 'swish').
        column_init_method: Initialization method for column parallel layer (default: 'xavier').
        row_init_method: Initialization method for row parallel layer (default: 'zero').
        input_is_parallel: Whether input is already parallelized (default: False).
        dropout: Dropout probability (default: 0.0).
        model_parallel_config: Configuration for model parallelism (default: None).
        alpha: Scaling factor for adapter output (default: None, uses dim).
        dropout_position: Where to apply dropout ('pre' or 'post', default: 'pre').
        a2a_experimental: Whether to use experimental all-to-all communication (default: False).
        is_expert: Whether this adapter is for expert layers in MoE (default: False).
        disable_sequence_parallel_comm: Whether to disable sequence parallel communication (default: True).
        base_linear_is_parallel: Whether the base linear layer uses parallelization (default: True).
        sequence_parallel_input_regather: Whether eligible LoRA-A projections retain the sequence-local input and
            re-gather it in backward using MCore's sequence-parallel linear path (default: False).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        base_linear_name: str,
        activation: str = "swish",
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: Optional[float] = None,
        dropout_position: str = "pre",
        a2a_experimental: bool = False,
        is_expert: bool = False,
        disable_tensor_parallel_comm: bool = False,
        disable_sequence_parallel_comm: bool = True,
        base_linear_is_parallel: bool = True,
        pg_collection: ProcessGroupCollection | None = None,
        sequence_parallel_input_regather: bool = False,
    ) -> None:
        """Initialize the ParallelLinearAdapter.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            dim: Adapter bottleneck dimension.
            base_linear_name: Name of the base linear layer.
            activation: Activation function name.
            column_init_method: Initialization for column parallel layers.
            row_init_method: Initialization for row parallel layers.
            input_is_parallel: Whether input is already parallelized.
            dropout: Dropout probability.
            model_parallel_config: Model parallelism configuration.
            alpha: Scaling factor (uses dim if None).
            dropout_position: When to apply dropout.
            a2a_experimental: Use experimental all-to-all communication.
            is_expert: Whether for expert layers in MoE.
            disable_tensor_parallel_comm: Disable tensor parallel communication.
            disable_sequence_parallel_comm: Disable sequence parallel communication.
            sequence_parallel_input_regather: Re-gather eligible LoRA-A sequence-parallel inputs in backward.
        """
        super().__init__()
        self.base_linear_name = base_linear_name
        self.activation = self._get_activation_fn(activation)
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.input_is_parallel = input_is_parallel
        self.dropout_position = dropout_position
        self.use_a2a = a2a_experimental
        self.is_expert = is_expert
        self.base_linear_is_parallel = base_linear_is_parallel
        self.sequence_parallel_input_regather = sequence_parallel_input_regather
        self._sequence_parallel_input_regather_fallback_logged = False
        self.use_legacy_shared_expert_adapter_checkpoint = False

        # megatron_gpt_peft_models will provide this arg, but deprecated ones do not.
        # in case this arg is not provided, use the dummy default config.
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        # TODO: When the PEFT transform API has explicit PG plumbing, pass the
        # model-level collection here instead of relying on config/default discovery.
        self.pg_collection = _get_pg_collection(
            pg_collection,
            model_parallel_config,
            required_pgs=["ep", "expt_tp", "expt_dp"] if is_expert else ["tp"],
        )
        self.tp_group = _get_tensor_parallel_group(self.pg_collection, is_expert=is_expert)
        self.ep_group = _get_process_group(self.pg_collection, "ep")
        self.expert_dp_group = _get_process_group(self.pg_collection, "expt_dp")
        _sequence_parallel = model_parallel_config.sequence_parallel
        model_parallel_config.sequence_parallel = False  # SP is irrelevant for the lora linear layer
        self.config = model_parallel_config

        # Ensure adapter parameters are initialized when creating adapter layers.
        # In some flows (e.g., after import), perform_initialization may be False to skip heavy init.
        model_parallel_config.perform_initialization = True

        if input_is_parallel:
            self.linear_in = RowParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                input_is_parallel=True,
                skip_bias_add=True,
                bias=False,
                init_method=self._get_init_fn(column_init_method),
                is_expert=is_expert,
                tp_group=self.tp_group,
            )
        else:
            self.linear_in = ColumnParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                bias=False,
                gather_output=True,
                init_method=self._get_init_fn(column_init_method),
                disable_grad_reduce=_sequence_parallel,
                is_expert=is_expert,
                tp_group=self.tp_group,
            )

        # (@adithyare) we use this option to mirror the behavior
        # a column parallel layer with two low-rank column parallel layers
        # if the original column parallel layer uses gather_output=False,
        # then we will use the self.liner_out layer defined below.
        lin_out_gather_output = True if input_is_parallel else False
        if (
            self.use_a2a
            and input_is_parallel
            and _sequence_parallel
            or (disable_tensor_parallel_comm and not input_is_parallel)
        ):
            lin_out_gather_output = False

        if not base_linear_is_parallel:
            lin_out_gather_output = True

        self.linear_out = ColumnParallelLinear(
            dim,
            out_features,
            config=model_parallel_config,
            bias=False,
            gather_output=lin_out_gather_output,
            init_method=self._get_init_fn(row_init_method),
            is_expert=is_expert,
            tp_group=self.tp_group,
        )

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # cast all parameters when using amp O2 training
        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        if self._uses_grouped_expert_sharding():
            self._register_shared_expert_grad_sync_hooks()

        # revert config change in case it is read elsewhere
        model_parallel_config.sequence_parallel = _sequence_parallel
        self.disable_sequence_parallel_comm = disable_sequence_parallel_comm
        if not _sequence_parallel:
            self.disable_sequence_parallel_comm = True

        if not base_linear_is_parallel:
            self.disable_sequence_parallel_comm = True

    def _sequence_parallel_input_regather_eligibility(self, x: torch.Tensor) -> tuple[bool, str | None]:
        """Return whether the targeted sequence-parallel linear path is safe."""
        if not self.sequence_parallel_input_regather:
            return False, None
        if not self.training or not torch.is_grad_enabled():
            return False, None
        if is_checkpointing():
            return False, None
        if not self.linear_in.weight.requires_grad:
            return False, "the LoRA-A weight is frozen"
        if self.input_is_parallel:
            return False, "the adapter is row-parallel"
        if self.disable_sequence_parallel_comm:
            return False, "sequence-parallel communication is disabled"
        if _process_group_size(self.tp_group, self.config.tensor_model_parallel_size or 1) <= 1:
            return False, "tensor parallel size is one"
        if self.is_expert:
            return False, "expert adapters use a separate path"
        if self.use_a2a:
            return False, "all-to-all adapters use a separate path"
        if not isinstance(self.activation, nn.Identity):
            return False, "the adapter activation is not identity"
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            return False, "CPU activation offload is enabled"

        surface = self.base_linear_name.rsplit(".", maxsplit=1)[-1]
        if surface not in {"linear_qkv", "linear_fc1"}:
            return False, f"{surface} is not an eligible column-parallel LoRA surface"

        cuda_graph_impl = getattr(self.config, "cuda_graph_impl", "none")
        if cuda_graph_impl not in (None, "none"):
            return False, "CUDA graphs are enabled"

        recompute_granularity = getattr(self.config, "recompute_granularity", None)
        if recompute_granularity == "full":
            return False, "full-layer activation recompute already covers the adapter"
        recompute_modules = getattr(self.config, "recompute_modules", None) or []
        if recompute_granularity == "selective" and surface == "linear_fc1" and "mlp" in recompute_modules:
            return False, "selective MLP recompute already covers linear_fc1"
        return True, None

    def _log_sequence_parallel_input_regather_fallback(self, reason: str | None) -> None:
        """Log the first static fallback reason at debug level."""
        if reason is None or self._sequence_parallel_input_regather_fallback_logged:
            return
        logger.debug(
            "LoRA sequence-parallel input re-gather requested for %s but using the existing path: %s",
            self.base_linear_name,
            reason,
        )
        self._sequence_parallel_input_regather_fallback_logged = True

    def _get_activation_fn(self, activation: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation: Name of the activation function.

        Returns:
            PyTorch activation module.

        Note:
            Defaults to Identity if activation name is not recognized.
        """
        activation_map = {
            "identity": nn.Identity(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "swish": nn.SiLU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return activation_map.get(activation, nn.Identity())

    def _get_init_fn(self, init_method: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Get initialization function by method name.

        Args:
            init_method: Name of the initialization method.

        Returns:
            Initialization function.

        Raises:
            NotImplementedError: If init_method is not supported.
        """
        if init_method == "xavier":
            init_fn = nn.init.xavier_normal_
        elif init_method == "normal":
            init_fn = init_method_normal(0.2)
        elif init_method == "kaiming":
            init_fn = init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            init_fn = init_method_const(0.0)
        else:
            raise NotImplementedError("out_init_method should be zero, normal, kaiming or xavier")
        return init_fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass of the parallel linear adapter.

        Performs the adaptation computation with proper handling of parallel communication
        patterns, dropout, and expert routing for MoE scenarios.

        Args:
            x: Input tensor.

        Returns:
            Adapted output tensor with scaling applied.
        """
        del args, kwargs

        if self.dropout_position == "pre":
            x = self.dropout(x)

        pad_len = 0
        if self.is_expert:
            x, pad_len = pad_seq_to_mult(x, self.config.expert_tensor_parallel_size)

        use_sequence_parallel_input_regather, fallback_reason = self._sequence_parallel_input_regather_eligibility(x)
        if not self.input_is_parallel:
            # MCore's SP linear keeps the local input in ctx, launches the
            # backward all-gather asynchronously before dgrad, and waits only
            # before wgrad. The baseline path keeps communication external.
            self.linear_in.sequence_parallel = use_sequence_parallel_input_regather
        if use_sequence_parallel_input_regather:
            # ColumnParallelLinear returns output and bias; adapters do not use the bias.
            x, _ = self.linear_in(x)
        else:
            self._log_sequence_parallel_input_regather_fallback(fallback_reason)
            if not self.disable_sequence_parallel_comm and not self.input_is_parallel and not self.is_expert:
                # for attention_qkv and linear_fc1
                # layernorm before lora is impacted by sequence parallel,
                # hence seq dim need to be gathered right before lora linear layers
                # this function also handles the backward pass correctly
                x = gather_from_sequence_parallel_region(x, group=self.tp_group)

            if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                x.activation_offloading = True
            # ColumnParallelLinear returns output and bias; adapters do not use the bias.
            x, _ = self.linear_in(x)

        x = self.activation(x)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            x.activation_offloading = True
        x, _ = self.linear_out(x)

        if not self.disable_sequence_parallel_comm and self.input_is_parallel and not self.is_expert:
            # for attention_dense and linear_fc2
            # layernorm after lora is impacted by sequence parallel,
            # hence seq dim need to be scattered right after lora linear layers
            # this function also handles the backward pass correctly
            if self.use_a2a:
                # all2all hidden_size / TP to seq_len / TP
                x = all2all_hp2sp(x, self.tp_group)
            else:
                x = scatter_to_sequence_parallel_region(x, group=self.tp_group)

        # Add dropout if available
        if self.dropout_position == "post":
            x = self.dropout(x)

        x = x * (self.alpha / self.dim)

        if pad_len > 0:
            # Remove MoE padding.
            x = unpad_seq_to_mult(x, pad_len)

        return x

    def local_experts_per_rank(self) -> int:
        """Return the number of global expert slots owned by this EP rank."""

        ep_size = _process_group_size(self.ep_group, self.config.expert_model_parallel_size or 1)
        num_global_experts = getattr(self.config, "num_moe_experts", None)
        if num_global_experts is None:
            return 1
        if int(num_global_experts) % int(ep_size) != 0:
            raise ValueError(
                f"num_moe_experts={num_global_experts} must be divisible by expert_model_parallel_size={ep_size}"
            )
        return int(num_global_experts) // int(ep_size)

    def _uses_grouped_expert_sharding(self) -> bool:
        """Return whether this shared adapter needs an explicit expert axis."""

        return self.is_expert and is_grouped_expert_linear(self.base_linear_name)

    def _allreduce_shared_expert_grad(self, grad: torch.Tensor) -> torch.Tensor:
        """Sum shared expert adapter grads across EP before expert-DP reduction."""

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return grad
        if self.ep_group is None or _process_group_size(self.ep_group) <= 1:
            return grad
        # Sum across EP first; MCore expert DDP then reduces across expert-DP
        # and scales expert buffers by 1 / dp_cp_group.size(), i.e. the full
        # EP x expert-DP data-parallel world, not just expert-DP.
        torch.distributed.all_reduce(grad, group=self.ep_group)
        return grad

    def _register_shared_expert_grad_sync_hooks(self) -> None:
        """Keep shared grouped-expert adapters synchronized across EP ranks."""

        for module in (self.linear_in, self.linear_out):
            weight = getattr(module, "weight", None)
            if isinstance(weight, torch.Tensor) and weight.requires_grad:
                weight.register_hook(self._allreduce_shared_expert_grad)

    def _expert_axis_info(self, sharded_offsets: Tuple) -> Tuple[int, int, int]:
        """Return the global expert-axis sharding metadata for this rank."""

        ep_size = _process_group_size(self.ep_group, self.config.expert_model_parallel_size or 1)
        ep_rank = _process_group_rank(self.ep_group)
        local_experts = self.local_experts_per_rank()
        if local_experts <= 0:
            raise ValueError(f"local_experts_per_rank must be positive, got {local_experts}")

        num_global_experts = getattr(self.config, "num_moe_experts", None)
        if num_global_experts is None:
            num_global_experts = ep_size * local_experts
        num_global_experts = int(num_global_experts)
        first_expert_slot = ep_rank * local_experts
        if first_expert_slot >= num_global_experts:
            raise ValueError(
                f"Invalid expert adapter sharding for {self.base_linear_name}: "
                f"ep_rank={ep_rank}, local_experts_per_rank={local_experts}, "
                f"num_global_experts={num_global_experts}"
            )

        expert_axis = len(sharded_offsets)
        return expert_axis, first_expert_slot, num_global_experts

    def _keep_expert_extra_state(self) -> bool:
        """Keep one unsharded adapter extra-state entry."""

        tp_rank = _process_group_rank(self.tp_group)
        ep_rank = _process_group_rank(self.ep_group)
        return tp_rank == 0 and ep_rank == 0

    def _set_expert_replica_ids(self, *state_dicts: ShardedStateDict) -> None:
        """Mark expert adapter replicas across expert data-parallel ranks."""

        edp_rank = _process_group_rank(self.expert_dp_group)
        for state_dict in state_dicts:
            for value in state_dict.values():
                if not hasattr(value, "replica_id"):
                    continue
                replica_id = value.replica_id
                if isinstance(replica_id, int):
                    replica_id = (0, 0, replica_id)
                if len(replica_id) != 3:
                    raise ValueError(
                        f"Expected replica_id for {self.base_linear_name} to be in "
                        f"(PP, TP, DP) format, got: {replica_id}"
                    )
                dp_replica_id = 0 if getattr(value, "is_data_parallel_fully_shard", False) else edp_rank
                value.replica_id = (*replica_id[:2], dp_replica_id)

    def _apply_expert_axis_factory(
        self,
        sharded_tensor: ShardedTensor,
        sharded_offsets: Tuple,
        *,
        split_swiglu: bool = False,
    ) -> ShardedTensorFactory:
        """Map one shared 2D adapter tensor to this rank's global expert slots."""

        expert_axis, first_expert_slot, num_global_experts = self._expert_axis_info(sharded_offsets)
        local_experts = self.local_experts_per_rank()
        base_prepend_axis_num = len(sharded_offsets)
        output_prepend_axis_num = base_prepend_axis_num + 1
        swiglu_shard_axis = 0

        preserved_rank_offsets = []
        for axis, local_axis_shape in enumerate(sharded_tensor.local_shape):
            base_global_axis_idx = axis + base_prepend_axis_num
            output_global_axis_idx = base_global_axis_idx + 1
            axis_fragments = sharded_tensor.axis_fragmentations[base_global_axis_idx]
            if axis_fragments <= 1:
                continue
            global_offset = sharded_tensor.global_offset[base_global_axis_idx]
            if global_offset % local_axis_shape != 0:
                raise ValueError(
                    f"Cannot preserve non-integral sharding for {sharded_tensor.key}: "
                    f"offset={global_offset}, local_axis_shape={local_axis_shape}"
                )
            preserved_rank_offsets.append((output_global_axis_idx, global_offset // local_axis_shape, axis_fragments))

        swiglu_axis_frag = None
        swiglu_rank_offset = None
        base_swiglu_global_axis = swiglu_shard_axis + base_prepend_axis_num
        output_swiglu_global_axis = swiglu_shard_axis + output_prepend_axis_num
        if split_swiglu:
            local_axis_size = sharded_tensor.local_shape[swiglu_shard_axis]
            if sharded_tensor.global_offset[base_swiglu_global_axis] % local_axis_size != 0:
                raise ValueError(
                    f"Cannot split SwiGLU tensor {sharded_tensor.key}: "
                    f"offset={sharded_tensor.global_offset[base_swiglu_global_axis]}, local_axis_size={local_axis_size}"
                )
            swiglu_rank_offset = sharded_tensor.global_offset[base_swiglu_global_axis] // local_axis_size
            swiglu_axis_frag = sharded_tensor.axis_fragmentations[base_swiglu_global_axis]
            preserved_rank_offsets = [
                rank_offset for rank_offset in preserved_rank_offsets if rank_offset[0] != output_swiglu_global_axis
            ]

        @torch.no_grad()
        def sh_ten_build_fn(key: str, tensor: torch.Tensor, replica_id, flattened_range):
            del flattened_range
            sharded_tensors = []
            for expert_index in range(local_experts):
                expert_offset = (expert_axis, first_expert_slot + expert_index, num_global_experts)
                if not split_swiglu:
                    sharded_tensors.append(
                        ShardedTensor.from_rank_offsets(
                            key,
                            tensor,
                            *sharded_offsets,
                            *preserved_rank_offsets,
                            expert_offset,
                            replica_id=replica_id,
                            prepend_axis_num=output_prepend_axis_num,
                        )
                    )
                    continue

                tensor_w, tensor_v = torch.chunk(tensor, 2, dim=swiglu_shard_axis)
                offset_w = (output_swiglu_global_axis, swiglu_rank_offset, swiglu_axis_frag * 2)
                offset_v = (
                    output_swiglu_global_axis,
                    swiglu_rank_offset + swiglu_axis_frag,
                    swiglu_axis_frag * 2,
                )
                for tensor_part, swiglu_offset in ((tensor_w, offset_w), (tensor_v, offset_v)):
                    sharded_tensors.append(
                        ShardedTensor.from_rank_offsets(
                            key,
                            tensor_part,
                            *sharded_offsets,
                            *preserved_rank_offsets,
                            expert_offset,
                            swiglu_offset,
                            replica_id=replica_id,
                            prepend_axis_num=output_prepend_axis_num,
                        )
                    )
            return sharded_tensors

        def sh_ten_merge_fn(sub_state_dict):
            if not isinstance(sub_state_dict, list):
                sub_state_dict = [sub_state_dict]
            if split_swiglu:
                if len(sub_state_dict) % 2 != 0:
                    raise ValueError(f"Expected even number of SwiGLU shards for {sharded_tensor.key}")
                sub_state_dict = [
                    torch.cat(sub_state_dict[index : index + 2], dim=swiglu_shard_axis)
                    for index in range(0, len(sub_state_dict), 2)
                ]
            if len(sub_state_dict) == 1:
                return sub_state_dict[0]
            return torch.stack(sub_state_dict, dim=0).mean(dim=0)

        return ShardedTensorFactory(
            sharded_tensor.key,
            sharded_tensor.data,
            sh_ten_build_fn,
            sh_ten_merge_fn,
            sharded_tensor.replica_id,
            flattened_range=sharded_tensor.flattened_range,
        )

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple = (),
        metadata: Optional[Dict] = None,
        mamba_dim_info: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Create sharded state dictionary for distributed checkpointing.

        Special treatment is given to the linear_fc1 adapter since tensor parallelism is
        sharded separately for the two logical matrices (gate and up) in SwiGLU.

        Args:
            prefix: Prefix for parameter names.
            sharded_offsets: Offsets for sharded parameters.
            metadata: Additional metadata for sharding.

        Returns:
            Sharded state dictionary for distributed checkpointing.
        """
        sharded_state_dict = {}
        # Shared grouped-expert adapters have one 2D weight per EP rank, but the
        # checkpoint must expose the global expert axis so EP changes can reshard it.
        # Non-grouped expert adapters already sit under .local_experts.* and keep
        # their existing expert-DP replica metadata instead.
        use_expert_axis = self._uses_grouped_expert_sharding() and not self.use_legacy_shared_expert_adapter_checkpoint
        split_swiglu = "linear_fc1" in self.base_linear_name and getattr(self.config, "gated_linear_unit", False)
        linear_in_sd = self.linear_in.sharded_state_dict(f"{prefix}linear_in.", sharded_offsets, metadata)
        linear_out_sd = self.linear_out.sharded_state_dict(f"{prefix}linear_out.", sharded_offsets, metadata)

        if use_expert_axis:
            keep_extra_state = self._keep_expert_extra_state()
            if not keep_extra_state:
                for sd in (linear_in_sd, linear_out_sd):
                    for key in [k for k in sd if "_extra_state" in k]:
                        del sd[key]
            for key, value in list(linear_in_sd.items()):
                if isinstance(value, ShardedTensor):
                    linear_in_sd[key] = self._apply_expert_axis_factory(value, sharded_offsets)
            for key, value in list(linear_out_sd.items()):
                if isinstance(value, ShardedTensor):
                    linear_out_sd[key] = self._apply_expert_axis_factory(
                        value,
                        sharded_offsets,
                        split_swiglu=split_swiglu,
                    )
        elif self.is_expert:
            if _process_group_rank(self.tp_group) > 0:
                for state_dict in (linear_in_sd, linear_out_sd):
                    for key in [k for k in state_dict if "_extra_state" in k]:
                        del state_dict[key]

        if split_swiglu and not use_expert_axis:
            for k, v in linear_out_sd.items():
                if k in (f"{prefix}linear_out.weight", f"{prefix}linear_out.bias"):
                    linear_out_sd[k] = apply_swiglu_sharded_factory(v, sharded_offsets)

        # Special handling for Mamba in_proj layer which needs to be split into 5 tensors
        if mamba_dim_info is not None:
            from megatron.core.ssm.mamba_mixer import _split_tensor_factory

            # Split linear_out.weight into 5 parts: z, x, B, C, dt
            # The in_proj output dimension is: d_inner * 2 + 2 * ngroups * d_state + nheads
            # After TP sharding: d_inner_local_tp * 2 + 2 * ngroups_local_tp * d_state + nheads_local_tp
            for k, v in linear_out_sd.items():
                if k == f"{prefix}linear_out.weight" and isinstance(v, ShardedTensor):
                    in_proj_dim_local = (
                        mamba_dim_info["d_inner_local_tp"] * 2
                        + 2 * mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"]
                        + mamba_dim_info["nheads_local_tp"]
                    )
                    # Verify the dimension matches
                    if v.data.size(0) == in_proj_dim_local:
                        linear_out_sd[k] = _split_tensor_factory(
                            v,
                            [
                                mamba_dim_info["d_inner_local_tp"],  # z
                                mamba_dim_info["d_inner_local_tp"],  # x
                                mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"],  # B
                                mamba_dim_info["ngroups_local_tp"] * mamba_dim_info["d_state"],  # C
                                mamba_dim_info["nheads_local_tp"],  # dt
                            ],
                            ["z", "x", "B", "C", "dt"],
                            0,  # split along dimension 0
                        )

        if self.is_expert:
            self._set_expert_replica_ids(linear_in_sd, linear_out_sd)

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict


def _divide_exact(value: int, divisor: int, name: str) -> int:
    """Divide ``value`` by ``divisor`` and raise when the result would be fractional."""

    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by expert TP size={divisor}")
    return value // divisor


def _apply_grouped_expert_swiglu_sharded_factory(
    original_sh_ten: ShardedTensor,
    sharded_offsets: Tuple,
    singleton_local_shards: bool = False,
) -> ShardedTensorFactory:
    """Split grouped-expert SwiGLU tensors along the fused hidden axis for checkpointing."""

    if original_sh_ten.axis_fragmentations is None:
        raise ValueError("Grouped-expert SwiGLU sharding requires regular-grid sharded tensor metadata.")

    swiglu_shard_axis = 1
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    local_axis_size = original_shape[swiglu_shard_axis]
    global_axis = swiglu_shard_axis + prepend_axis_num
    assert original_sh_ten.global_offset[global_axis] % local_axis_size == 0
    rank_offset = original_sh_ten.global_offset[global_axis] // local_axis_size
    axis_frag = original_sh_ten.axis_fragmentations[global_axis]

    preserved_rank_offsets = []
    for axis, local_axis_shape in enumerate(original_shape):
        if axis == swiglu_shard_axis:
            continue
        global_axis_idx = axis + prepend_axis_num
        axis_fragm = original_sh_ten.axis_fragmentations[global_axis_idx]
        if axis_fragm <= 1:
            continue
        global_offset = original_sh_ten.global_offset[global_axis_idx]
        assert global_offset % local_axis_shape == 0
        preserved_rank_offsets.append((global_axis_idx, global_offset // local_axis_shape, axis_fragm))

    @torch.no_grad()
    def sh_ten_build_fn(key: str, tensor: torch.Tensor, replica_id, flattened_range):
        del flattened_range

        if singleton_local_shards:
            offset_w = (global_axis, rank_offset, axis_frag)
            offset_v = (global_axis, rank_offset, axis_frag)
            w_key = f"{key}_w"
            v_key = f"{key}_v"
        else:
            offset_w = (global_axis, rank_offset, axis_frag * 2)
            offset_v = (global_axis, rank_offset + axis_frag, axis_frag * 2)
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(tensor, 2, dim=swiglu_shard_axis)
        rank_offsets = (*sharded_offsets, *preserved_rank_offsets)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *rank_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *rank_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    def sh_ten_merge_fn(sub_state_dict):
        if not singleton_local_shards and len(sub_state_dict) > 1:
            # Dist checkpoint load reconstructs one local fused shard per expert-TP
            # rank, so the incoming tensors look like [gate_0|up_0, gate_1|up_1, ...].
            # Restore the fused [gate_0, gate_1, ..., up_0, up_1, ...] layout before
            # concatenating back along the SwiGLU axis.
            gate_parts = []
            up_parts = []
            for tensor in sub_state_dict:
                gate_part, up_part = torch.chunk(tensor, 2, dim=swiglu_shard_axis)
                gate_parts.append(gate_part)
                up_parts.append(up_part)
            sub_state_dict = [*gate_parts, *up_parts]
        try:
            return torch.cat(sub_state_dict, dim=swiglu_shard_axis)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            logger.warning(
                "CUDA OutOfMemoryError encountered during grouped-expert SwiGLU merge. "
                "Switching to CPU merge. (Error: %s)",
                exc,
            )
            merged_sub_state_dict = torch.cat([tensor.cpu() for tensor in sub_state_dict], dim=swiglu_shard_axis)
            torch.cuda.empty_cache()
            return merged_sub_state_dict

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )


def _append_rank_offset(
    rank_offsets: List[Tuple[int, int, int]],
    axis: int,
    rank: int,
    axis_fragments: int,
) -> None:
    """Append a sharding offset, combining fragmentations when the axis is already sharded."""

    if axis_fragments <= 1:
        return

    for index, (existing_axis, existing_rank, existing_fragments) in enumerate(rank_offsets):
        if existing_axis != axis:
            continue
        rank_offsets[index] = (
            axis,
            existing_rank * axis_fragments + rank,
            existing_fragments * axis_fragments,
        )
        return

    rank_offsets.append((axis, rank, axis_fragments))


def _make_grouped_expert_sharded_tensor(
    tensor: torch.Tensor,
    key: str,
    *,
    tp_axis: Optional[int],
    sharded_offsets: Tuple,
    pg_collection: ProcessGroupCollection | None,
    ep_size_fallback: int = 1,
    etp_size_fallback: int = 1,
) -> ShardedTensor:
    """Build a sharded tensor for packed grouped-expert weights.

    Grouped-expert LoRA weights shard two independent local axes: the packed
    expert axis across EP and the adapter matrix axis across expert TP.
    """

    prepend_axis_num = len(sharded_offsets)
    rank_offsets = list(sharded_offsets)

    ep_group = _get_process_group(pg_collection, "ep")
    ep_size = _process_group_size(ep_group, ep_size_fallback)
    _append_rank_offset(
        rank_offsets,
        prepend_axis_num,
        _process_group_rank(ep_group),
        ep_size,
    )

    if tp_axis is not None:
        etp_group = _get_tensor_parallel_group(pg_collection, is_expert=True)
        etp_size = _process_group_size(etp_group, etp_size_fallback)
        _append_rank_offset(
            rank_offsets,
            prepend_axis_num + tp_axis,
            _process_group_rank(etp_group),
            etp_size,
        )

    expt_dp_group = _get_process_group(pg_collection, "expt_dp")
    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *rank_offsets,
        replica_id=(0, 0, _process_group_rank(expt_dp_group)),
        prepend_axis_num=prepend_axis_num,
    )


class _GroupedExpertAdapterWeight(nn.Module):
    """Callable parameter container so DDP forward pre-hooks see grouped LoRA weights."""

    # Overlapped param gather is driven by module forward pre-hooks. Calling this
    # container before reading the weight makes expert-DP LoRA params participate
    # in the normal training-time gather instead of only forced eval/checkpoint sync.

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight)

    def forward(self, indices: Optional[List[int]] = None) -> torch.Tensor:
        if indices is None:
            return self.weight
        return self.weight[indices]


class GroupedExpertLinearAdapter(nn.Module):
    """LoRA adapter with one low-rank pair per local grouped MoE expert."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        *,
        num_local_experts: int,
        base_linear_name: str,
        activation: str = "swish",
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: Optional[float] = None,
        dropout_position: str = "pre",
        base_linear_is_parallel: bool = True,
        params_device: Optional[torch.device] = None,
        params_dtype: Optional[torch.dtype] = None,
        pg_collection: ProcessGroupCollection | None = None,
    ) -> None:
        """Initialize grouped-expert LoRA weights for one adapter per local expert."""

        super().__init__()

        self.base_linear_name = base_linear_name
        self.activation = ParallelLinearAdapter._get_activation_fn(self, activation)
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.input_is_parallel = input_is_parallel
        self.dropout_position = dropout_position
        self.base_linear_is_parallel = base_linear_is_parallel
        self.is_expert = True
        self.num_local_experts = num_local_experts
        # Keep TE helpers outside the module tree so their meta parameters do not
        # appear in adapter state dicts or optimizer parameter groups.
        self._te_grouped_linear_helpers: Dict[Tuple[str, int, int, int, torch.dtype, torch.device], nn.Module] = {}
        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        self.config = model_parallel_config
        # TODO: When the PEFT transform API has explicit PG plumbing, pass the
        # model-level collection here instead of relying on config/default discovery.
        self.pg_collection = _get_pg_collection(
            pg_collection,
            model_parallel_config,
            required_pgs=["ep", "expt_tp", "expt_dp"],
        )
        self.expert_tp_group = _get_tensor_parallel_group(self.pg_collection, is_expert=True)
        self.ep_group = _get_process_group(self.pg_collection, "ep")
        self.expert_dp_group = _get_process_group(self.pg_collection, "expt_dp")

        model_parallel_config.perform_initialization = True

        expert_tp_size = _process_group_size(
            self.expert_tp_group,
            model_parallel_config.expert_tensor_parallel_size or 1,
        )
        linear_in_tp_axis = 2 if input_is_parallel else 1
        linear_out_tp_axis = 1

        if input_is_parallel:
            linear_in_shape = (
                num_local_experts,
                dim,
                _divide_exact(in_features, expert_tp_size, "in_features"),
            )
        else:
            linear_in_shape = (
                num_local_experts,
                _divide_exact(dim, expert_tp_size, "dim"),
                in_features,
            )
        linear_out_shape = (
            num_local_experts,
            _divide_exact(out_features, expert_tp_size, "out_features"),
            dim,
        )

        if params_device is None:
            distributed_initialized = torch.distributed.is_available() and torch.distributed.is_initialized()
            params_device = (
                torch.device("cpu")
                if model_parallel_config.use_cpu_initialization
                or not torch.cuda.is_available()
                or not distributed_initialized
                else torch.device("cuda", torch.cuda.current_device())
            )
        dtype = params_dtype or model_parallel_config.params_dtype

        linear_in_weight = torch.empty(linear_in_shape, device=params_device, dtype=dtype)
        linear_out_weight = torch.empty(linear_out_shape, device=params_device, dtype=dtype)
        ParallelLinearAdapter._get_init_fn(self, column_init_method)(linear_in_weight)
        ParallelLinearAdapter._get_init_fn(self, row_init_method)(linear_out_weight)

        expert_parallel = (
            _process_group_size(
                self.ep_group,
                model_parallel_config.expert_model_parallel_size or 1,
            )
            > 1
        )
        self._linear_in_tp_axis = linear_in_tp_axis
        self._linear_out_tp_axis = linear_out_tp_axis
        self.linear_in = _GroupedExpertAdapterWeight(linear_in_weight)
        self.linear_out = _GroupedExpertAdapterWeight(linear_out_weight)
        for weight, tp_axis in (
            (self.linear_in.weight, linear_in_tp_axis),
            (self.linear_out.weight, linear_out_tp_axis),
        ):
            setattr(weight, "allreduce", not expert_parallel)
            if tp_axis is not None:
                set_tensor_model_parallel_attributes(weight, True, tp_axis, 1)

        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def _extract_expert_splits(self, args: Tuple, kwargs: Dict) -> List[int]:
        """Extract grouped-expert token splits from wrapped-module call arguments."""

        expert_splits = kwargs.get("m_splits")
        if expert_splits is None:
            expert_splits = kwargs.get("tokens_per_expert")
        if expert_splits is None and args:
            expert_splits = args[0]
        if isinstance(expert_splits, torch.Tensor):
            expert_splits = expert_splits.tolist()
        if expert_splits is None:
            raise ValueError(f"Per-expert LoRA on {self.base_linear_name} requires grouped expert token splits.")
        if len(expert_splits) != self.num_local_experts:
            raise ValueError(
                f"Expected {self.num_local_experts} expert splits for {self.base_linear_name}, "
                f"got {len(expert_splits)}"
            )
        splits = [int(split) for split in expert_splits]
        if any(split < 0 for split in splits):
            raise ValueError(f"Expert splits for {self.base_linear_name} must be non-negative, got {splits}")
        return splits

    def _gather_along_last_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a tensor across expert TP ranks by concatenating its last dimension."""

        expert_tp_size = _process_group_size(self.expert_tp_group, self.config.expert_tensor_parallel_size or 1)
        if expert_tp_size == 1:
            return tensor
        expert_tp_group = self.expert_tp_group
        if expert_tp_group is None:
            raise ValueError(
                f"{self.base_linear_name} requires initialized expert tensor parallel state "
                f"when expert_tensor_parallel_size={expert_tp_size}."
            )
        gathered = [torch.empty_like(tensor) for _ in range(expert_tp_size)]
        torch.distributed.all_gather(
            gathered,
            tensor,
            group=expert_tp_group,
        )
        return torch.cat(gathered, dim=-1)

    def _can_use_grouped_mm(self, x: torch.Tensor) -> bool:
        """Return whether the grouped GEMM fast path is supported for this input."""

        if getattr(nn.functional, "grouped_mm", None) is None:
            return False
        if not x.is_cuda or x.dtype != torch.bfloat16:
            return False
        if self.linear_in.weight.dtype != torch.bfloat16 or self.linear_out.weight.dtype != torch.bfloat16:
            return False
        if (
            not x.is_contiguous()
            or not self.linear_in.weight.is_contiguous()
            or not self.linear_out.weight.is_contiguous()
        ):
            return False
        # grouped_mm forward and weight-gradient kernels require matrix strides
        # to be multiples of 16 bytes, so every local matrix dimension in both
        # projections must be divisible by eight for BF16 tensors.
        grouped_mm_dimensions = self.linear_in.weight.shape[-2:] + self.linear_out.weight.shape[-2:]
        if any(dimension % 8 != 0 for dimension in grouped_mm_dimensions):
            return False
        return torch.cuda.get_device_capability(x.device) >= (8, 0)

    def _is_te_fp8_enabled(self) -> bool:
        """Return whether Transformer Engine's FP8 autocast context is active."""

        return HAVE_TE_FP8_GLOBAL_STATE_MANAGER and TEFP8GlobalStateManager.is_fp8_enabled()

    def _can_use_te_grouped_linear_fp8(self, x: torch.Tensor) -> bool:
        """Return whether TE can run the grouped adapter weights through FP8."""

        if not (
            HAVE_TE_PYTORCH_GROUPED_LINEAR
            and HAVE_TE_PYTORCH_GROUPED_LINEAR_AUTOGRAD
            and HAVE_TE_PYTORCH_CPU_OFFLOAD_STATUS
        ):
            return False
        if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float16):
            return False
        # TE binds FP8 autocast state to the current CUDA device before this
        # module runs. An input on another device must use the safe fallback.
        if x.device.index != torch.cuda.current_device():
            return False
        if self.linear_in.weight.dtype != x.dtype or self.linear_out.weight.dtype != x.dtype:
            return False
        fp8_grouped_dimensions = self.linear_in.weight.shape[-2:] + self.linear_out.weight.shape[-2:]
        return all(dimension % 16 == 0 for dimension in fp8_grouped_dimensions)

    def _get_te_grouped_linear_helper(
        self,
        *,
        projection: str,
        num_gemms: int,
        in_features: int,
        out_features: int,
        params_dtype: torch.dtype,
        device: torch.device,
    ) -> nn.Module:
        """Create or reuse a meta-device TE helper that owns FP8 runtime state."""

        # Each helper owns delayed-scaling state per GEMM slot. Keep one fixed
        # slot per local expert and logical projection so routing changes cannot
        # transfer FP8 history between weights or grow the helper cache.
        key = (projection, num_gemms, in_features, out_features, params_dtype, device)
        helper = self._te_grouped_linear_helpers.get(key)
        if helper is None:
            helper = TEPytorchGroupedLinear(
                num_gemms=num_gemms,
                in_features=in_features,
                out_features=out_features,
                sequence_parallel=False,
                fuse_wgrad_accumulation=False,
                tp_group=None,
                tp_size=1,
                bias=False,
                return_bias=False,
                params_dtype=params_dtype,
                parallel_mode=None,
                device="meta",
            )
            self._te_grouped_linear_helpers[key] = helper
        helper.train(self.training)
        return helper

    def _forward_te_grouped_linear_fp8(
        self,
        x: torch.Tensor,
        *,
        weight: torch.Tensor,
        m_splits: List[int],
        projection: str,
        active_expert_indices: Tuple[int, ...],
    ) -> torch.Tensor:
        """Apply externally owned grouped adapter weights through TE's FP8 kernel."""

        device_context = torch.cuda.device(x.device) if x.is_cuda else nullcontext()
        with device_context:
            return self._forward_te_grouped_linear_fp8_on_current_device(
                x,
                weight=weight,
                m_splits=m_splits,
                projection=projection,
                active_expert_indices=active_expert_indices,
            )

    def _forward_te_grouped_linear_fp8_on_current_device(
        self,
        x: torch.Tensor,
        *,
        weight: torch.Tensor,
        m_splits: List[int],
        projection: str,
        active_expert_indices: Tuple[int, ...],
    ) -> torch.Tensor:
        """Run TE FP8 grouped linear after selecting the input tensor's CUDA device."""

        helper = self._get_te_grouped_linear_helper(
            projection=projection,
            num_gemms=self.num_local_experts,
            in_features=weight.shape[-1],
            out_features=weight.shape[-2],
            params_dtype=weight.dtype,
            device=x.device,
        )
        x = helper.prepare_forward(x, num_gemms=self.num_local_experts)
        try:
            quantizer_groups = helper._get_quantizers()
            (
                input_quantizers,
                weight_quantizers,
                output_quantizers,
                grad_input_quantizers,
                grad_weight_quantizers,
                grad_output_quantizers,
            ) = ([quantizers[expert_idx] for expert_idx in active_expert_indices] for quantizers in quantizer_groups)
            common_non_tensor_args = (
                helper.apply_bias,
                None,
                helper.fp8,
                helper.fp8_calibration,
                helper.wgrad_store,
                input_quantizers,
                weight_quantizers,
                output_quantizers,
                grad_input_quantizers,
                grad_weight_quantizers,
                grad_output_quantizers,
                helper.fuse_wgrad_accumulation,
                TEPytorchIsCPUOffloadEnabled(),
                helper.sequence_parallel,
                helper.activation_dtype,
                torch.is_grad_enabled(),
                [None] * weight.shape[0],
                False,
                None,
                helper.save_original_input,
                False,
            )
            empty_biases = [x.new_empty(0) for _ in range(weight.shape[0])]
            weights_and_biases = (*[weight[i] for i in range(weight.shape[0])], *empty_biases)

            # TODO: Replace this private FP8 dispatch once Transformer Engine exposes
            # a public functional grouped-linear API for externally owned weights
            # (NVIDIA/TransformerEngine#3191).
            if _te_grouped_linear_uses_explicit_m_splits(TEPytorchGroupedLinearAutograd):
                te_m_splits = torch.tensor(m_splits, dtype=torch.int64, device="cpu")
                autograd_args = (x, te_m_splits, common_non_tensor_args, *weights_and_biases)
            else:
                non_tensor_args = (m_splits, *common_non_tensor_args)
                autograd_args = (x, non_tensor_args, *weights_and_biases)

            if torch.is_grad_enabled():
                output, _ = TEPytorchGroupedLinearAutograd.apply(*autograd_args)
            else:
                output, _ = TEPytorchGroupedLinearAutograd.forward(None, *autograd_args)
            return output
        finally:
            helper.end_forward()

    def _build_grouped_mm_offsets(self, m_splits: List[int], *, device: torch.device) -> torch.Tensor:
        """Build inclusive grouped_mm offsets from per-expert split sizes."""

        return torch.tensor(m_splits, device=device, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)

    def _forward_grouped_projection(
        self,
        x: torch.Tensor,
        *,
        weight: torch.Tensor,
        m_splits: List[int],
        use_te_fp8: bool,
        projection: str,
        active_expert_indices: Tuple[int, ...],
        offs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply one grouped expert projection through the selected grouped backend."""

        if use_te_fp8:
            return self._forward_te_grouped_linear_fp8(
                x,
                weight=weight,
                m_splits=m_splits,
                projection=projection,
                active_expert_indices=active_expert_indices,
            )
        if offs is None:
            offs = self._build_grouped_mm_offsets(m_splits, device=x.device)
        return nn.functional.grouped_mm(x, weight.transpose(1, 2), offs=offs)

    def _forward_per_expert(
        self,
        x: torch.Tensor,
        *,
        expert_splits: List[int],
        expert_tp_size: int,
    ) -> torch.Tensor:
        """Apply the adapter using the per-expert fallback path."""

        linear_in_weight = self.linear_in()
        linear_out_weight = self.linear_out()
        outputs = []
        start = 0
        for expert_idx, split_size in enumerate(expert_splits):
            expert_input = x.narrow(0, start, split_size)
            start += split_size

            pad_len = 0
            if expert_input.numel() > 0:
                expert_input, pad_len = pad_seq_to_mult(expert_input, expert_tp_size)
                if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                    expert_input.activation_offloading = True

            hidden = nn.functional.linear(expert_input, linear_in_weight[expert_idx])
            if not self.input_is_parallel:
                hidden = self._gather_along_last_dim(hidden)
            hidden = self.activation(hidden)

            if self.config.cpu_offloading and self.config.cpu_offloading_activations:
                hidden.activation_offloading = True
            expert_output = nn.functional.linear(hidden, linear_out_weight[expert_idx])
            if self.input_is_parallel:
                expert_output = self._gather_along_last_dim(expert_output)

            if self.dropout_position == "post":
                expert_output = self.dropout(expert_output)
            if pad_len > 0:
                expert_output = unpad_seq_to_mult(expert_output, pad_len)
            outputs.append(expert_output)

        return torch.cat(outputs, dim=0)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Apply the local expert-specific LoRA update to grouped expert inputs."""

        expert_splits = self._extract_expert_splits(args, kwargs)
        total_tokens = sum(expert_splits)
        if total_tokens != x.shape[0]:
            raise ValueError(
                f"Expert splits for {self.base_linear_name} sum to {total_tokens}, but received {x.shape[0]} tokens"
            )
        if self.dropout_position == "pre":
            x = self.dropout(x)

        expert_tp_size = _process_group_size(self.expert_tp_group, self.config.expert_tensor_parallel_size or 1)
        output_features = self.linear_out.weight.shape[1]
        if self.input_is_parallel:
            output_features *= expert_tp_size
        if x.shape[0] == 0:
            linear_in_weight = self.linear_in()
            linear_out_weight = self.linear_out()
            grad_anchor = linear_in_weight.reshape(-1)[0] + linear_out_weight.reshape(-1)[0]
            return (x.new_empty((0, output_features)) + grad_anchor * 0.0) * (self.alpha / self.dim)

        fp8_enabled = self._is_te_fp8_enabled()
        use_te_fp8 = fp8_enabled and self._can_use_te_grouped_linear_fp8(x)
        use_grouped_mm = not fp8_enabled and self._can_use_grouped_mm(x)
        if not use_te_fp8 and not use_grouped_mm:
            return self._forward_per_expert(x, expert_splits=expert_splits, expert_tp_size=expert_tp_size) * (
                self.alpha / self.dim
            )

        active_expert_indices = []
        grouped_inputs = []
        padded_splits = []
        pad_lengths = []
        grouped_padding_multiple = math.lcm(expert_tp_size, 16) if use_te_fp8 else expert_tp_size
        start = 0
        for expert_idx, split_size in enumerate(expert_splits):
            if split_size == 0:
                continue
            expert_input = x.narrow(0, start, split_size)
            start += split_size
            expert_input, pad_len = pad_seq_to_mult(expert_input, grouped_padding_multiple)
            active_expert_indices.append(expert_idx)
            grouped_inputs.append(expert_input)
            padded_splits.append(expert_input.shape[0])
            pad_lengths.append(pad_len)

        grouped_input = grouped_inputs[0] if len(grouped_inputs) == 1 else torch.cat(grouped_inputs, dim=0)
        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            grouped_input.activation_offloading = True

        offs = None if use_te_fp8 else self._build_grouped_mm_offsets(padded_splits, device=x.device)

        active_linear_in = self.linear_in(active_expert_indices)
        active_expert_tuple = tuple(active_expert_indices)
        hidden = self._forward_grouped_projection(
            grouped_input,
            weight=active_linear_in,
            m_splits=padded_splits,
            use_te_fp8=use_te_fp8,
            projection="linear_in",
            active_expert_indices=active_expert_tuple,
            offs=offs,
        )
        if not self.input_is_parallel:
            hidden = self._gather_along_last_dim(hidden)
        hidden = self.activation(hidden)

        if self.config.cpu_offloading and self.config.cpu_offloading_activations:
            hidden.activation_offloading = True
        active_linear_out = self.linear_out(active_expert_indices)
        expert_output = self._forward_grouped_projection(
            hidden,
            weight=active_linear_out,
            m_splits=padded_splits,
            use_te_fp8=use_te_fp8,
            projection="linear_out",
            active_expert_indices=active_expert_tuple,
            offs=offs,
        )
        if self.input_is_parallel:
            expert_output = self._gather_along_last_dim(expert_output)

        if self.dropout_position == "post":
            expert_output = self.dropout(expert_output)

        if all(pad_len == 0 for pad_len in pad_lengths):
            return expert_output * (self.alpha / self.dim)

        outputs = []
        start = 0
        for padded_size, pad_len in zip(padded_splits, pad_lengths):
            output_chunk = expert_output.narrow(0, start, padded_size)
            outputs.append(unpad_seq_to_mult(output_chunk, pad_len) if pad_len > 0 else output_chunk)
            start += padded_size

        return torch.cat(outputs, dim=0) * (self.alpha / self.dim)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Create sharded state dictionary for grouped-expert adapter weights."""

        sharded_state_dict = {}
        linear_in_sd = {
            f"{prefix}linear_in.weight": _make_grouped_expert_sharded_tensor(
                self.linear_in.weight,
                f"{prefix}linear_in.weight",
                tp_axis=self._linear_in_tp_axis,
                sharded_offsets=sharded_offsets,
                pg_collection=self.pg_collection,
                ep_size_fallback=self.config.expert_model_parallel_size or 1,
                etp_size_fallback=self.config.expert_tensor_parallel_size or 1,
            )
        }
        linear_out_sd = {
            f"{prefix}linear_out.weight": _make_grouped_expert_sharded_tensor(
                self.linear_out.weight,
                f"{prefix}linear_out.weight",
                tp_axis=self._linear_out_tp_axis,
                sharded_offsets=sharded_offsets,
                pg_collection=self.pg_collection,
                ep_size_fallback=self.config.expert_model_parallel_size or 1,
                etp_size_fallback=self.config.expert_tensor_parallel_size or 1,
            )
        }

        if "linear_fc1" in self.base_linear_name and getattr(self.config, "gated_linear_unit", False):
            singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
            linear_out_key = f"{prefix}linear_out.weight"
            linear_out_sd[linear_out_key] = _apply_grouped_expert_swiglu_sharded_factory(
                linear_out_sd[linear_out_key],
                sharded_offsets,
                singleton_local_shards,
            )

        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict


def _make_cross_ep_replicated(weight: nn.Parameter) -> None:
    """Mark a weight as logically replicated across the intra-PP-stage group.

    Megatron's DDP routes ``is_expert=True`` parameters through the expert
    data-parallel group only, which does not span the EP axis. A weight
    that must stay bit-identical across all EP ranks (e.g., the shared
    side of :class:`SharedOuterGroupedExpertAdapter`, which a serving
    engine consumes as a single global LoRA tensor) is otherwise left
    unsynced. This helper closes that gap with two primitives:

      * a one-shot broadcast from group rank 0 so every rank starts with
        bit-identical values despite per-rank RNG forks;
      * a backward hook that SUM all-reduces the gradient across the group
        so the optimizer step on every rank applies the same update.

    SUM is the correct reduction: each rank's local gradient is the partial
    loss gradient over its (token, expert) subset, and the total gradient
    is the sum of those partials. AVG would train at 1/N the intended rate.

    The intra-PP-stage group is ``tensor_and_data_parallel_group`` with
    context parallel included, which by Megatron's construction equals
    ETP × EP × EDP — all ranks within the current pipeline stage.

    Args:
        weight: The parameter to keep replicated across the group. Must
            be a leaf parameter so the backward hook fires when its
            gradient is computed.
    """

    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    try:
        group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    except AssertionError:
        return
    if torch.distributed.get_world_size(group=group) <= 1:
        return

    if weight.is_cuda:
        # NCCL requires CUDA tensors; pre-GPU construction relies on
        # deterministic init matching across ranks.
        src_rank = torch.distributed.get_global_rank(group, 0)
        with torch.no_grad():
            torch.distributed.broadcast(weight.data, src=src_rank, group=group)

    def _all_reduce_grad(grad: torch.Tensor) -> torch.Tensor:
        grad = grad.contiguous()
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=group)
        return grad

    weight.register_hook(_all_reduce_grad)


class PackedPerExpertLinear(nn.Module):
    """Per-expert linear with a packed 3D weight ``[N_local, out, in]``.

    Used as the per-expert side of :class:`SharedOuterGroupedExpertAdapter`.
    Stores one ``nn.Parameter`` (3D) so Bridge's adapter export sees a single
    ``.weight`` per side, matching the ``linear_in.weight`` / ``linear_out.weight``
    convention in :mod:`megatron.bridge.models.conversion.peft_bridge`. Forward
    dispatches to :func:`torch._grouped_mm` (the same grouped GEMM kernel TE's
    :class:`te.pytorch.GroupedLinear` calls) via a single fused op with native
    autograd, which keeps rank kernel launch counts in lockstep so CP's ring
    P2P does not deadlock.
    """

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        *,
        init_method: Optional[Callable] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not hasattr(torch, "_grouped_mm"):
            raise RuntimeError("PackedPerExpertLinear requires torch._grouped_mm (torch >= 2.9).")
        self.num_local_experts = num_local_experts
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.empty(num_local_experts, out_features, in_features, dtype=dtype, device=device)
        if init_method is not None:
            for e in range(num_local_experts):
                init_method(weight[e])
        else:
            nn.init.zeros_(weight)
        self.weight = nn.Parameter(weight)
        # DDP routes ``is_expert`` weights through the EDP group; the cross-EP
        # axis is naturally distinct here (different experts on each EP rank).
        setattr(self.weight, "allreduce", False)

    def forward(self, x: torch.Tensor, m_splits) -> Tuple[torch.Tensor, None]:
        # torch._grouped_mm expects mat2 as [num_groups, K, N]; our weight is
        # [N_local, out, in] so transpose the last two dims.
        if isinstance(m_splits, torch.Tensor):
            m_splits_i32 = m_splits.to(device=x.device, dtype=torch.int32)
        else:
            m_splits_i32 = torch.tensor(m_splits, device=x.device, dtype=torch.int32)
        offs = torch.cumsum(m_splits_i32, dim=0, dtype=torch.int32)
        out = torch._grouped_mm(x, self.weight.transpose(1, 2), offs=offs)
        return out, None

    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: Tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Shard the packed 3D weight along dim 0 (experts) across EP ranks."""
        key = f"{prefix}weight"
        return {
            key: _make_grouped_expert_sharded_tensor(
                self.weight.data, key, tp_axis=None, sharded_offsets=sharded_offsets
            )
        }


class SharedOuterGroupedExpertAdapter(nn.Module):
    """LoRA adapter for grouped expert MLP with shared-outer semantics.

    Matches SGLang PR #21466's ``experts_shared_outer_loras=True`` contract:

    * fc1 (gate_up):  linear_in  = SHARED     (hidden -> rank)
                      linear_out = PER-EXPERT (rank -> 2*intermediate)
    * fc2 (down):     linear_in  = PER-EXPERT (intermediate -> rank)
                      linear_out = SHARED     (rank -> hidden)

    The shared side is an ``is_expert=True`` ``ColumnParallelLinear`` (fc1)
    or ``RowParallelLinear`` (fc2): the TP group is ETP (ETP=1 → local
    forward), DDP routes the weight through the EDP group, and the
    logically-replicated cross-EP axis is covered by
    :func:`_make_cross_ep_replicated`.

    The per-expert side is :class:`PackedPerExpertLinear` (packed 3D weight
    + :func:`torch._grouped_mm`) — kept as a single ``.weight`` Parameter so
    Bridge's adapter-export materializer (which reads ``linear_in.weight`` /
    ``linear_out.weight``) sees a standard single-weight linear per side.

    Differs from ``ParallelLinearAdapter`` in ``__init__`` and ``forward``;
    ``sharded_state_dict`` is specialized for the packed 3D per-expert side.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int,
        *,
        num_local_experts: int,
        base_linear_name: str,
        activation: str = "swish",
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        input_is_parallel: bool = False,
        dropout: float = 0.0,
        model_parallel_config: Optional[ModelParallelConfig] = None,
        alpha: Optional[float] = None,
        dropout_position: str = "pre",
        base_linear_is_parallel: bool = True,
        params_device: Optional[torch.device] = None,
        params_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize shared-outer LoRA weights with one shared and one per-expert side."""

        super().__init__()
        self.base_linear_name = base_linear_name
        self.activation = ParallelLinearAdapter._get_activation_fn(self, activation)
        self.dim = dim
        self.alpha = alpha if alpha is not None else self.dim
        self.dropout_position = dropout_position
        self.num_local_experts = num_local_experts
        self.base_linear_is_parallel = base_linear_is_parallel
        # ``is_expert=True`` is observed by param_mapping.py and by inherited
        # checkpoint helpers; the per-expert side's grad routing is set on its
        # 3D weight directly inside :class:`PackedPerExpertLinear`.
        self.is_expert = True

        if model_parallel_config is None:
            model_parallel_config = ModelParallelConfig()
        model_parallel_config.perform_initialization = True
        self.config = model_parallel_config

        # ``input_is_parallel`` selects fc1 (column-parallel base) vs fc2
        # (row-parallel base). Mirrors :class:`ParallelLinearAdapter` and
        # :class:`GroupedExpertLinearAdapter`.
        self._is_fc1 = not input_is_parallel

        column_init = ParallelLinearAdapter._get_init_fn(self, column_init_method)
        row_init = ParallelLinearAdapter._get_init_fn(self, row_init_method)
        if self._is_fc1:
            # Shared A (hidden → rank); per-expert B (rank → 2*intermediate).
            self.linear_in = ColumnParallelLinear(
                in_features,
                dim,
                config=model_parallel_config,
                bias=False,
                gather_output=True,
                init_method=column_init,
                is_expert=True,
            )
            self.linear_out = PackedPerExpertLinear(
                num_local_experts,
                dim,
                out_features,
                init_method=row_init,
                device=params_device,
                dtype=params_dtype,
            )
        else:
            # Per-expert A (intermediate → rank); shared B (rank → hidden).
            self.linear_in = PackedPerExpertLinear(
                num_local_experts,
                in_features,
                dim,
                init_method=column_init,
                device=params_device,
                dtype=params_dtype,
            )
            self.linear_out = RowParallelLinear(
                dim,
                out_features,
                config=model_parallel_config,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=True,
                init_method=row_init,
                is_expert=True,
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if model_parallel_config.bf16:
            self.bfloat16()
        elif model_parallel_config.fp16:
            self.half()

        # The shared weight is logically replicated across EP; close the gap
        # that Megatron's expert-DDP routing leaves open.
        shared_weight = self.linear_in.weight if self._is_fc1 else self.linear_out.weight
        _make_cross_ep_replicated(shared_weight)

    def forward(self, x: torch.Tensor, m_splits=None) -> torch.Tensor:
        """Forward. ``m_splits`` is the tokens-per-expert split passed through
        from the base TEGroupedLinear; required for the per-expert side.
        """
        if self.dropout_position == "pre":
            x = self.dropout(x)

        if self._is_fc1:
            # Shared A → activation → per-expert B.
            x, _ = self.linear_in(x)
            x = self.activation(x)
            x, _ = self.linear_out(x, m_splits)
        else:
            # Per-expert A → activation → shared B.
            x, _ = self.linear_in(x, m_splits)
            x = self.activation(x)
            x, _ = self.linear_out(x)

        if self.dropout_position == "post":
            x = self.dropout(x)

        return x * (self.alpha / self.dim)

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple = (),
        metadata: Optional[Dict] = None,
    ) -> ShardedStateDict:
        """Create sharded state dictionary for mixed shared/per-expert adapter weights."""

        linear_in_sd = self.linear_in.sharded_state_dict(f"{prefix}linear_in.", sharded_offsets, metadata)
        linear_out_sd = self.linear_out.sharded_state_dict(f"{prefix}linear_out.", sharded_offsets, metadata)

        if self._is_fc1:
            singleton_local_shards = (metadata or {}).get("singleton_local_shards", False)
            linear_out_key = f"{prefix}linear_out.weight"
            linear_out_sd[linear_out_key] = _apply_grouped_expert_swiglu_sharded_factory(
                linear_out_sd[linear_out_key],
                sharded_offsets,
                singleton_local_shards,
            )

        sharded_state_dict = {}
        sharded_state_dict.update(linear_in_sd)
        sharded_state_dict.update(linear_out_sd)
        return sharded_state_dict
