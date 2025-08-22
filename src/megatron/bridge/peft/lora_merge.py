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
from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora_layers import LoRALinear
from megatron.bridge.training.checkpointing import save_checkpoint
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer
from megatron.bridge.training.model_load_save import (
    load_megatron_model,
    temporary_distributed_context,
)


class LoRAMerge(PEFT):
    """
    Implements the LoRA weight merge for parameter-efficient fine-tuning.
    """

    @torch.no_grad()
    def transform(self, module: nn.Module, name: Optional[str] = None, prefix: Optional[str] = None) -> nn.Module:
        """
        Merges the LoRA adapter with the base model weights.

        Args:
            module (nn.Module): The module to apply LoRA merge to.
            name (str, optional): Name of the module to merge. Defaults to None.
            prefix (str, optional): Prefix for the module name. Defaults to None.

        Returns:
            nn.Module: The modified module with the LoRA adapter merged into the base model weights.
        """
        if not isinstance(module, LoRALinear):
            return module

        logging.info(f"merging {(prefix if prefix else '') + '.' + (name if name else '')}")
        lora_weight = (
            module.adapter.alpha
            / module.adapter.dim
            * module.adapter.linear_out.weight
            @ module.adapter.linear_in.weight
        )
        if hasattr(module.to_wrap, "weight"):
            base_weight = module.to_wrap.weight
            merged_weight = base_weight + lora_weight.to(base_weight.device)
            module.to_wrap.weight.data = merged_weight
        else:  # TE Grouped Linear
            for i in range(module.to_wrap.num_gemms):
                base_weight = getattr(module.to_wrap, f"weight{i}")
                merged_weight = base_weight + lora_weight.to(base_weight.device)
                getattr(module.to_wrap, f"weight{i}").data = merged_weight
        return module


def merge_lora(lora_checkpoint_path: str, output_path: str) -> None:
    """
    Merge LoRA adapter weights into base model weights, preserving all metadata.

    This function loads a LoRA checkpoint, extracts the base model and adapter weights,
    merges them into a single model, and saves the result as a Megatron-Bridge checkpoint
    that can be used for inference or further training without PEFT.

    Args:
        lora_checkpoint_path: Path to LoRA checkpoint (specific iteration directory like iter_0000100)
                             containing run_config.yaml, adapter weights and metadata
        output_path: Path to save merged checkpoint directory

    Example:
        >>> from megatron.bridge.peft.lora_merge import merge_lora
        >>> merge_lora("/path/to/lora_checkpoint", "/path/to/merged_checkpoint")

    Note:
        - Uses CPU initialization and Gloo backend for memory efficiency
        - Preserves all original model metadata and configuration
        - Removes PEFT configuration from merged checkpoint
        - Output checkpoint is in torch_dist format
        - Automatically resolves base pretrained directories to latest iteration
    """

    def _merge_lora_internal():
        config_from_checkpoint = _load_full_config_container_from_checkpoint(lora_checkpoint_path)

        # Extract pretrained checkpoint path and PEFT config
        pretrained_path = config_from_checkpoint.checkpoint.pretrained_checkpoint
        peft_config = config_from_checkpoint.peft

        if not pretrained_path:
            raise ValueError(f"No pretrained_checkpoint found in {lora_checkpoint_path}")
        if not peft_config:
            raise ValueError(f"No PEFT configuration found in {lora_checkpoint_path}")

        # Resolve pretrained path to specific iteration if it's a base directory
        resolved_pretrained_path = _resolve_checkpoint_path(pretrained_path)
        base_model = load_megatron_model(
            checkpoint_path=resolved_pretrained_path,
            use_cpu_init=True,
            return_state_dict=False,
            skip_temp_dist_context=True,
        )

        # Ensure base_model is a list for consistent handling
        if not isinstance(base_model, list):
            base_model = [base_model]

        # Apply PEFT transformation and load adapter weights
        _apply_adapters_to_model_with_full_config(base_model, config_from_checkpoint)

        # Apply LoRAMerge transformation
        lora_merge = LoRAMerge()
        merged_model = lora_merge(base_model, training=False)

        merged_config = _prepare_merged_config(config_from_checkpoint, output_path)

        _save_merged_model_with_metadata(merged_model, merged_config, output_path)

    # Check if distributed is already initialized
    skip_temp_context = torch.distributed.is_available() and torch.distributed.is_initialized()

    if skip_temp_context:
        # Already in distributed context, but ensure Megatron parallel state is initialized
        _ensure_megatron_parallel_state_initialized()
        _merge_lora_internal()
    else:
        # Use temporary distributed context for CPU-based merging
        with temporary_distributed_context(backend="gloo"):
            _merge_lora_internal()


def _ensure_megatron_parallel_state_initialized() -> None:
    """Ensure Megatron model parallel state is initialized for merge operations."""
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel()


def _load_full_config_container_from_checkpoint(lora_checkpoint_path: str) -> ConfigContainer:
    """Load the complete ConfigContainer from PEFT checkpoint, preserving all metadata."""
    from megatron.bridge.training.config import ConfigContainer
    from megatron.bridge.training.utils.checkpoint_utils import (
        file_exists,
        get_checkpoint_run_config_filename,
        read_run_config,
    )

    # Read run config from LoRA checkpoint
    run_config_filename = get_checkpoint_run_config_filename(lora_checkpoint_path)
    if not file_exists(run_config_filename):
        raise ValueError(
            f"Run config not found in {lora_checkpoint_path}. Expected Megatron-Bridge checkpoint format."
        )

    run_config = read_run_config(run_config_filename)

    from megatron.bridge.utils.instantiate_utils import InstantiationMode

    config_container = ConfigContainer.from_dict(run_config, mode=InstantiationMode.LENIENT)

    # Modify checkpoint config for merging
    config_container.checkpoint = _modify_checkpoint_config_for_merge(
        config_container.checkpoint, lora_checkpoint_path
    )

    return config_container


def _modify_checkpoint_config_for_merge(
    original_checkpoint_config: CheckpointConfig, lora_checkpoint_path: str
) -> CheckpointConfig:
    """Modify checkpoint config for merge operation while preserving metadata."""

    # Create a modified checkpoint config for merging
    # - load from the LoRA checkpoint path (to get adapter weights)
    # - don't load optimizer/rng (not needed for merging)
    modified_config = replace(
        original_checkpoint_config,
        load=lora_checkpoint_path,  # Load from LoRA checkpoint
        load_optim=False,  # Don't need optimizer
        load_rng=False,  # Don't need RNG state
    )

    return modified_config


def _prepare_merged_config(original_config: ConfigContainer, output_path: str) -> ConfigContainer:
    """Prepare config for merged checkpoint, removing PEFT but preserving everything else."""

    # Update checkpoint config for the merged model
    merged_checkpoint_config = replace(
        original_config.checkpoint,
        save=output_path,  # New save location
        pretrained_checkpoint=None,  # Merged model is now self-contained
        save_optim=False,  # Don't save optimizer in merged checkpoint
        save_rng=False,  # Don't save RNG in merged checkpoint
        ckpt_format="torch_dist",
    )

    # Create merged config (same as original but without PEFT)
    merged_config = replace(
        original_config,
        checkpoint=merged_checkpoint_config,
        peft=None,  # Remove PEFT - merged model doesn't need adapters anymore
    )

    return merged_config


def _save_merged_model_with_metadata(model: list, config: ConfigContainer, output_path: str) -> None:
    """Save merged model using the full config to preserve all metadata."""
    from megatron.bridge.training.checkpointing import init_checkpointing_context
    from megatron.bridge.training.state import GlobalState, TrainState

    # Create GlobalState with config and train state initialized
    state = GlobalState()
    state.cfg = config
    state.train_state = TrainState()

    # Initialize checkpointing context
    checkpointing_context = init_checkpointing_context(config.checkpoint)

    # Use the same save_checkpoint logic as training to preserve metadata
    save_checkpoint(
        state=state,
        model=model,
        optimizer=None,  # No optimizer for merged model
        opt_param_scheduler=None,  # No scheduler for merged model
        num_floating_point_operations_so_far=0,
        checkpointing_context=checkpointing_context,
    )


def _apply_adapters_to_model_with_full_config(model: list[MegatronModule], full_config: ConfigContainer) -> None:
    """Apply PEFT transformation and load adapter weights using full config."""
    peft_config = full_config.peft
    lora_checkpoint_path = full_config.checkpoint.load

    # Apply original PEFT transformation from LoRA checkpoint config
    transformed_model = peft_config(model, training=False)

    # Load only adapter weights
    _load_adapter_weights_only(transformed_model, lora_checkpoint_path, peft_config)


def _load_adapter_weights_only(model: list[MegatronModule], lora_checkpoint_path: str, peft_config: PEFT) -> None:
    """Load only adapter weights directly using distributed checkpoint loading."""
    from megatron.core import dist_checkpointing

    from megatron.bridge.training.checkpointing import (
        _generate_model_state_dict,
        _load_state_dict_into_model_list,
        apply_peft_adapter_filter_to_state_dict,
        get_default_load_sharded_strategy,
    )

    # Ensure model is a list for consistent handling
    if not isinstance(model, list):
        model = [model]

    # Generate model state dict template from transformed model (with adapters)
    complete_sharded_state_dict = _generate_model_state_dict(model)

    # Filter sharded state dict keys to only load adapter states
    filtered_sharded_state_dict = apply_peft_adapter_filter_to_state_dict(complete_sharded_state_dict, peft_config)

    # Load adapter weights directly using distributed checkpoint
    load_strategy = get_default_load_sharded_strategy(lora_checkpoint_path)
    loaded_state_dict = dist_checkpointing.load(
        filtered_sharded_state_dict,
        lora_checkpoint_path,
        load_strategy,
        strict=dist_checkpointing.validation.StrictHandling.LOG_UNEXPECTED,
    )

    # Load adapter weights into model using shared utility
    _load_state_dict_into_model_list(model, loaded_state_dict, strict=False)


def _resolve_checkpoint_path(checkpoint_path: str) -> str:
    """
    Resolve checkpoint path to specific iteration directory.

    If checkpoint_path is a base directory, resolve to the latest iteration.
    If checkpoint_path is already a specific iteration directory, return as-is.

    Args:
        checkpoint_path: Either base checkpoint directory or specific iteration directory

    Returns:
        Path to specific iteration directory containing weights and config
    """
    import os

    from megatron.bridge.training.checkpointing import TRACKER_PREFIX
    from megatron.bridge.training.utils.checkpoint_utils import (
        CONFIG_FILE,
        file_exists,
        get_checkpoint_name,
        get_checkpoint_train_state_filename,
        read_train_state,
    )

    # Check if this is already a specific iteration directory (contains run config)
    run_config_file = os.path.join(checkpoint_path, CONFIG_FILE)
    if file_exists(run_config_file):
        # This is already a specific iteration directory
        return checkpoint_path

    tracker_filename = get_checkpoint_train_state_filename(checkpoint_path, prefix=TRACKER_PREFIX)
    if not file_exists(tracker_filename):
        raise ValueError(
            f"Cannot resolve checkpoint path {checkpoint_path}. "
            f"Expected either a specific iteration directory (containing run_config.yaml) "
            f"or a base directory with tracker file (latest_train_state.pt)"
        )

    train_state = read_train_state(tracker_filename)
    iteration = train_state.step
    resolved_path = get_checkpoint_name(checkpoint_path, iteration, release=False)
    return resolved_path
