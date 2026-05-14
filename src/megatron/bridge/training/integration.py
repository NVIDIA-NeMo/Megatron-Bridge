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

"""Small public helpers for integrating Bridge with external trainers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule


ModelList = list[MegatronModule]
ModelHook = Callable[[ModelList], ModelList | None]
CheckpointPath = str | Path


def _apply_peft(peft: object, model: ModelList, *, training: bool = True) -> ModelList:
    """Apply PEFT and mark adapter parameters for checkpointing."""
    transformed_model = peft(model, training=training)
    peft.set_params_to_save(transformed_model)
    return transformed_model


def create_peft(config: Mapping[str, Any], *, dtype: torch.dtype | str | int | None = None) -> object | None:
    """Create a Bridge PEFT object from a small config mapping."""
    kwargs = dict(config)
    peft_type = kwargs.pop("type", "lora")
    if "rank" in kwargs:
        kwargs["dim"] = kwargs.pop("rank")
    if kwargs.get("dim", 0) <= 0:
        return None

    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.dora import DoRA
    from megatron.bridge.peft.lora import LoRA, VLMLoRA

    peft_classes = {
        "lora": LoRA,
        "vlm_lora": VLMLoRA,
        "canonical_lora": CanonicalLoRA,
        "dora": DoRA,
    }
    if peft_type not in peft_classes:
        supported_types = ", ".join(sorted(peft_classes))
        raise ValueError(f"Unsupported PEFT type {peft_type!r}. Supported types: {supported_types}.")
    peft_cls = peft_classes[peft_type]

    peft_fields = {field.name for field in fields(peft_cls) if field.init}
    config_dtype = kwargs.pop("dtype", None)
    if "lora_dtype" not in kwargs:
        kwargs["lora_dtype"] = config_dtype if config_dtype is not None else dtype

    if kwargs.get("lora_dtype") is None or "lora_dtype" not in peft_fields:
        kwargs.pop("lora_dtype", None)
    else:
        kwargs["lora_dtype"] = _to_torch_dtype(kwargs["lora_dtype"])

    kwargs = {key: value for key, value in kwargs.items() if key in peft_fields}

    return peft_cls(**kwargs)


def create_peft_hook(
    peft: object,
    *,
    base_checkpoint_loader: Callable[[ModelList], ModelList | None] | None = None,
    adapter_checkpoint_loader: Callable[[ModelList], None] | None = None,
    training: bool = True,
) -> ModelHook:
    """Create a provider pre-wrap hook that loads base weights, applies PEFT, and loads adapters."""

    def hook(model: ModelList) -> ModelList:
        if base_checkpoint_loader is not None:
            loaded_model = base_checkpoint_loader(model)
            if loaded_model is not None:
                model = loaded_model

        model = _apply_peft(peft, model, training=training)

        if adapter_checkpoint_loader is not None:
            adapter_checkpoint_loader(model)

        return model

    return hook


def load_peft_adapter_checkpoint(
    model: ModelList | MegatronModule,
    adapter_checkpoint_path: CheckpointPath,
    *,
    peft: object,
    strict: bool = False,
    model_sd_kwargs: Mapping[str, object] | None = None,
    ckpt_format: str = "torch_dist",
    pg_collection: ProcessGroupCollection | None = None,
    fully_parallel_load: bool = True,
    load_strategy: object | None = None,
) -> None:
    """Load a PEFT adapter checkpoint into an already transformed model."""
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
    from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper

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
        load_strategy = get_default_load_sharded_strategy(checkpoint_path)
        if fully_parallel_load and parallel_state.is_initialized():
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy,
                parallel_state.get_data_parallel_group(with_context_parallel=True),
            )

    loaded_state_dict = dist_checkpointing.load(sharded_state_dict, checkpoint_path, load_strategy)
    for vpp_rank, model_chunk in enumerate(model_chunks):
        model_key = "model" if len(model_chunks) == 1 else f"model{vpp_rank}"
        if model_key not in loaded_state_dict and len(model_chunks) == 1:
            fallback_model_key = next((key for key in loaded_state_dict if key.startswith("model")), None)
            if fallback_model_key is None:
                raise KeyError(
                    "Expected adapter checkpoint to contain a top-level 'model' or 'model*' key, "
                    f"but found keys: {list(loaded_state_dict.keys())}"
                )
            model_key = fallback_model_key
        model_chunk.load_state_dict(loaded_state_dict[model_key], strict=strict)


def _model_state_dict(
    model: ModelList,
    model_sd_kwargs: Mapping[str, object] | None = None,
    ckpt_format: str = "torch_dist",
    *,
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


def create_ddp_config(
    *,
    wrap_with_ddp: bool = True,
    use_distributed_optimizer: bool = True,
    use_megatron_fsdp: bool = False,
    overrides: Mapping[str, object] | None = None,
    finalize: bool = True,
) -> object | None:
    """Create a finalized Bridge DDP config for external model construction."""
    if not wrap_with_ddp:
        return None

    from megatron.bridge.training.config import DistributedDataParallelConfig

    ddp_config = {
        "use_distributed_optimizer": use_distributed_optimizer,
    }
    if use_megatron_fsdp:
        ddp_config.update(
            {
                "use_distributed_optimizer": True,
                "check_for_nan_in_grad": True,
                "use_megatron_fsdp": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
                "overlap_grad_reduce": True,
            }
        )
    ddp_config.update(overrides or {})

    config = DistributedDataParallelConfig(**ddp_config)
    if finalize:
        config.finalize()
    return config


def _to_torch_dtype(dtype: torch.dtype | str | int | None) -> torch.dtype | None:
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    dtype_key = dtype.lower() if isinstance(dtype, str) else dtype
    dtype_map = {
        16: torch.float16,
        "16": torch.float16,
        "fp16": torch.float16,
        "float16": torch.float16,
        32: torch.float32,
        "32": torch.float32,
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_key not in dtype_map:
        supported_dtypes = ", ".join(str(key) for key in dtype_map)
        raise ValueError(f"Unsupported torch dtype {dtype!r}. Supported dtypes: {supported_dtypes}.")
    return dtype_map[dtype_key]


def _ensure_model_list(model: ModelList | MegatronModule) -> ModelList:
    return model if isinstance(model, list) else [model]


__all__ = [
    "create_ddp_config",
    "create_peft",
    "create_peft_hook",
    "load_peft_adapter_checkpoint",
]
