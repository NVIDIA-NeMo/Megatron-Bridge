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
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from megatron.core import tensor_parallel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule


ModelList = list[MegatronModule]
ModelHook = Callable[[ModelList], ModelList | None]
CheckpointPath = str | Path


class LinearForLastLayer(torch.nn.Linear):
    """Final replicated projection head compatible with Megatron output-layer calls.

    Megatron-Core output layers receive a few runtime-only arguments. This head
    accepts those arguments for call-site compatibility while using a standard
    replicated linear projection.
    """

    def __init__(self, *, input_size: int, output_size: int, sequence_parallel: bool) -> None:
        """Initialize a replicated final projection.

        Args:
            input_size: Hidden dimension of the transformer output.
            output_size: Output dimension of the value/reward head.
            sequence_parallel: Whether to gather sequence-parallel activations.
        """
        super().__init__(in_features=input_size, out_features=output_size, bias=False)
        self.sequence_parallel = sequence_parallel
        if sequence_parallel:
            setattr(self.weight, "sequence_parallel", True)

    def forward(
        self,
        input_: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ) -> tuple[torch.Tensor, None]:
        """Run the final projection and return Megatron-style ``(output, bias)``."""
        del weight, runtime_gather_output
        logits = super().forward(input_).float()
        if self.sequence_parallel:
            logits = tensor_parallel.gather_from_sequence_parallel_region(
                logits,
                tensor_parallel_output_grad=False,
            )
        return logits, None


def create_value_head_hook(*, hidden_size: int, sequence_parallel: bool, output_size: int = 1) -> ModelHook:
    """Create a pre-wrap hook that replaces the final pipeline stage output head.

    Args:
        hidden_size: Hidden dimension of the transformer output.
        sequence_parallel: Whether the model uses sequence parallelism.
        output_size: Number of outputs produced by the final head.

    Returns:
        A model hook suitable for external trainer provider construction.
    """
    from megatron.core import parallel_state

    _register_linear_for_last_layer_mapping()

    def hook(model: ModelList | MegatronModule) -> ModelList:
        model_chunks = _ensure_model_list(model)
        model_post_process: list[bool] = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for vp_stage in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(
                    parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
                )
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        if len(model_post_process) != len(model_chunks):
            raise ValueError(
                "Model list length and pipeline post-process list length must match. "
                f"Got {len(model_chunks)} model chunks and {len(model_post_process)} post-process flags."
            )

        for index, model_chunk in enumerate(model_chunks):
            if model_post_process[index]:
                model_chunk.output_layer = LinearForLastLayer(
                    input_size=hidden_size,
                    output_size=output_size,
                    sequence_parallel=sequence_parallel,
                )

        return model_chunks

    return hook


def make_value_model(hidden_size: int, sequence_parallel: bool) -> ModelHook:
    """Create a value-head hook compatible with existing external trainer code."""
    return create_value_head_hook(hidden_size=hidden_size, sequence_parallel=sequence_parallel)


def freeze_moe_router(model: ModelList | MegatronModule) -> ModelList:
    """Freeze MoE router and shared-expert gate parameters in model chunks.

    Args:
        model: Single Megatron module or list of virtual-pipeline model chunks.

    Returns:
        The normalized model chunk list with router parameters frozen in place.
    """
    model_chunks = _ensure_model_list(model)
    for model_chunk in model_chunks:
        decoder = getattr(model_chunk, "decoder", None)
        layers = getattr(decoder, "layers", None)
        if layers is None:
            continue
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            router = getattr(mlp, "router", None)
            if router is not None:
                _freeze_parameter_if_present(router, "weight")
                _freeze_parameter_if_present(router, "bias")

            shared_experts = getattr(mlp, "shared_experts", None)
            if shared_experts is not None:
                _freeze_parameter_if_present(shared_experts, "gate_weight")
                _freeze_parameter_if_present(shared_experts, "gate_bias")

    return model_chunks


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

    peft_cls = _import_peft_class(peft_type)

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
        message = f"Failed to import PEFT type {peft_type!r} from {module_name}.{class_name}."
        if peft_type in {"lora", "vlm_lora", "canonical_lora"}:
            message += " Install Megatron Bridge with the [te] extra for Transformer Engine support."
        raise ImportError(message) from err

    return getattr(module, class_name)


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


def _freeze_parameter_if_present(module: object, name: str) -> None:
    parameter = getattr(module, name, None)
    if parameter is not None:
        parameter.requires_grad = False


def _register_linear_for_last_layer_mapping() -> None:
    from megatron.bridge.models.conversion.param_mapping import AutoMapping

    AutoMapping.register_module_type("LinearForLastLayer", "replicated")


__all__ = [
    "LinearForLastLayer",
    "create_value_head_hook",
    "create_ddp_config",
    "create_peft",
    "create_peft_hook",
    "freeze_moe_router",
    "load_peft_adapter_checkpoint",
    "make_value_model",
]
