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
from dataclasses import dataclass, fields
from typing import Callable, Optional

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.models import GPTModelProvider, T5ModelProvider


@dataclass(kw_only=True)
class MixedPrecisionConfig:
    """Mixed precision configuration for models.

    Handles conversion of model parameters and inputs/outputs between different precisions,
    and manages mixed precision training settings.
    """

    fp32: bool = False
    fp16: bool = False
    bf16: bool = False
    params_dtype: Optional[torch.dtype] = None
    pipeline_dtype: Optional[torch.dtype] = None
    autocast_dtype: Optional[torch.dtype] = None
    autocast_enabled: bool = False
    grad_reduce_in_fp32: bool = True
    # fp8 related
    fp8: Optional[str] = None
    fp8_recipe: str = (
        "tensorwise"  # "tensorwise", "delayed", "mxfp8" (for Blackwell only), "blockwise" (for Hopper only)
    )
    first_last_layers_bf16: bool = False
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_dot_product_attention: bool = False
    fp8_multi_head_attention: bool = False
    fp8_param: Optional[bool] = None
    fp8_param_gather: bool = False
    # FP16 Loss scaling
    loss_scale: Optional[float] = None
    initial_loss_scale: Optional[float] = 4294967296  # 2**32
    min_loss_scale: float = 1.0
    loss_scale_window: float = 1000
    hysteresis: int = 2
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0
    reuse_grad_buf_for_mxfp8_param_ag: bool = False

    def __setattr__(self, name: str, value) -> None:
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, name, value)

        # Keep fp8_param and fp8_param_gather in sync
        if name == "fp8_param_gather" and hasattr(self, "fp8_param"):
            if self.fp8_param != value:
                object.__setattr__(self, "fp8_param", value)
        elif name == "fp8_param" and hasattr(self, "fp8_param_gather"):
            if self.fp8_param_gather != value:
                object.__setattr__(self, "fp8_param_gather", value)

    def finalize(self):
        # If fp8_param is None, initialize it from fp8_param_gather
        if self.fp8_param is None:
            self.fp8_param = self.fp8_param_gather

        # Validate that mxfp8 recipe requires reuse_grad_buf_for_mxfp8_param_ag=True when fp8_param_gather=True
        if self.fp8_param_gather and self.fp8_recipe == "mxfp8":
            assert self.reuse_grad_buf_for_mxfp8_param_ag, (
                "When fp8_param_gather=True and fp8_recipe='mxfp8', "
                "reuse_grad_buf_for_mxfp8_param_ag must be set to True"
            )

    def setup(
        self,
        model_config: GPTModelProvider | T5ModelProvider,
        optimizer_config: Optional[OptimizerConfig] = None,
        ddp_config: Optional[DistributedDataParallelConfig] = None,
    ) -> None:
        """Apply mixed precision configs to model, optimizer, and DDP configs.

        Args:
            model_config: Model configuration to update with dtype settings
            optimizer_config: Optional optimizer configuration to update
            ddp_config: Optional DDP configuration to update
        """
        # Update model config
        model_config = update_config_with_precision_overrides(self, model_config)

        # Update optimizer config if provided
        if optimizer_config is not None:
            optimizer_config = update_config_with_precision_overrides(self, optimizer_config)

        # Update DDP config if provided
        if ddp_config is not None:
            ddp_config = update_config_with_precision_overrides(self, ddp_config)


def update_config_with_precision_overrides(mixed_precision_config: MixedPrecisionConfig, config):
    """Update a config object with precision settings from mixed_precision_config.

    Args:
        mixed_precision_config: Source of precision settings
        config: Config object to update

    Returns:
        Updated config object
    """
    for field in fields(mixed_precision_config):
        if not hasattr(config, field.name):
            continue
        # If we overwrote a value, log a debug message.
        old_val = getattr(config, field.name)
        new_val = getattr(mixed_precision_config, field.name)
        if old_val != new_val:
            setattr(config, field.name, new_val)
            logging.debug(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
    return config


# ----------------------------------------------------------------------------
# Recipe functions for common mixed precision configurations
# ----------------------------------------------------------------------------

MIXED_PRECISION_RECIPES: dict[str, Callable[[], "MixedPrecisionConfig"]] = {}


def register(func: Callable[[], "MixedPrecisionConfig"]):
    """Decorator that registers a mixed-precision recipe factory by its function name.

    Automatically registers both underscore and hyphen versions (e.g., 'bf16_mixed' and 'bf16-mixed')
    to simplify migrating from NeMo2.
    """
    name = func.__name__
    MIXED_PRECISION_RECIPES[name] = func

    # Also register hyphen version if the name contains underscores
    if "_" in name:
        hyphen_name = name.replace("_", "-")
        MIXED_PRECISION_RECIPES[hyphen_name] = func

    return func


@register
def bf16_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 mixed precision training
    """
    return MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=True,
    )


@register
def fp16_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 mixed precision training
    """
    return MixedPrecisionConfig(
        fp16=True,
        params_dtype=torch.half,
        pipeline_dtype=torch.half,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )


@register
def bf16_with_fp8_delayed_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8.

    Note: FP8 recipes are experimental and have not been tested for training convergence.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "delayed"
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_param_gather = True
    return cfg


@register
def fp16_with_fp8_delayed_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with FP8.

    Note: FP8 recipes are experimental and have not been tested for training convergence.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with FP8 mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "delayed"
    cfg.fp8_margin = 0
    cfg.fp8_amax_history_len = 1024
    cfg.fp8_amax_compute_algo = "max"
    cfg.fp8_param_gather = True
    return cfg


@register
def bf16_with_mxfp8_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with MXFP8 mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "e4m3"
    cfg.fp8_recipe = "mxfp8"
    cfg.fp8_param_gather = True
    cfg.reuse_grad_buf_for_mxfp8_param_ag = True
    return cfg


@register
def fp16_with_mxfp8_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with MXFP8.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with MXFP8 mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "e4m3"
    cfg.fp8_recipe = "mxfp8"
    cfg.fp8_param_gather = True
    cfg.reuse_grad_buf_for_mxfp8_param_ag = True
    return cfg


@register
def bf16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.fp8_param_gather = True
    return cfg


@register
def nemotron_h_bf16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 2
    cfg.num_layers_at_end_in_bf16 = 2
    cfg.fp8_param_gather = True
    return cfg


@register
def nanov2_bf16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses BF16 in the first and last Transformer layers. The user
    can choose to disable the BF16 layers or apply BF16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "blockwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 2
    cfg.num_layers_at_end_in_bf16 = 2
    cfg.fp8_param_gather = True
    return cfg


@register
def fp16_with_fp8_current_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with FP8
    per-tensor current scaling.

    Note: The baseline current scaling recipe uses FP16 in the first and last Transformer layers. The user
    can choose to disable the FP16 layers or apply FP16 to more Transformer layers.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with FP8 per-tensor current scaling mixed
        precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "tensorwise"
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 1
    cfg.num_layers_at_end_in_bf16 = 1
    cfg.fp8_param_gather = True
    return cfg


@register
def bf16_with_fp8_subchannel_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using BF16 with FP8
    NV Subchannel scaling. This recipe uses 128x128 blockwise quantization for weight and 1x128 blockwise
    quantization for activation.

    Returns:
        MixedPrecisionConfig: Configuration for BF16 with FP8 subchannel scaling mixed precision training
    """
    cfg = bf16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "blockwise"
    cfg.fp8_param_gather = False
    return cfg


@register
def fp16_with_fp8_subchannel_scaling_mixed() -> MixedPrecisionConfig:
    """Create a MixedPrecisionConfig for mixed precision training using FP16 with FP8
    NV Subchannel scaling. This recipe uses 128x128 blockwise quantization for weight and 1x128 blockwise
    quantization for activation.

    Returns:
        MixedPrecisionConfig: Configuration for FP16 with FP8 subchannel scaling mixed precision training
    """
    cfg = fp16_mixed()
    cfg.fp8 = "hybrid"
    cfg.fp8_recipe = "blockwise"
    cfg.fp8_param_gather = False
    return cfg


def get_mixed_precision_config(name: str | MixedPrecisionConfig) -> MixedPrecisionConfig:
    """Return a :class:`MixedPrecisionConfig` for *name*.

    Args:
        name: Key of the recipe in :pydata:`MIXED_PRECISION_RECIPES` or a :class:`MixedPrecisionConfig` instance.

    Raises:
        ValueError: If *name* is not a known recipe.
    """
    if isinstance(name, MixedPrecisionConfig):
        return name
    try:
        return MIXED_PRECISION_RECIPES[name]()
    except KeyError as err:
        valid = ", ".join(sorted(MIXED_PRECISION_RECIPES.keys()))
        raise ValueError(f"Unknown mixed-precision recipe '{name}'. Available recipes: {valid}.") from err
