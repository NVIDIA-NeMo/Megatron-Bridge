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
from typing import Literal, Optional

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo_lm.models.gpt import GPTConfig
from nemo_lm.models.t5 import T5Config


@dataclass(kw_only=True)
class DtypeConfig:
    """Configuration class for mixed precision training settings.

    Contains settings for FP32/FP16/BF16 training, FP8 training.
    """

    fp32: bool = False
    fp16: bool = False
    bf16: bool = False
    params_dtype: torch.dtype = None
    pipeline_dtype: torch.dtype = None
    autocast_dtype: torch.dtype = None
    autocast_enabled: bool = False
    grad_reduce_in_fp32: bool = True
    # fp8 related
    fp8: str = None
    fp8_recipe: str = "delayed"
    first_last_layers_bf16: bool = False
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_dot_product_attention: bool = False
    fp8_multi_head_attention: bool = False
    fp8_param: bool = True
    fp8_param_gather: bool = True
    # FP16 Loss scaling
    loss_scale: float = (None,)
    initial_loss_scale: float = (None,)
    min_loss_scale: float = (None,)
    loss_scale_window: float = (None,)
    hysteresis: float = (None,)
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0


@dataclass(kw_only=True)
class MegatronMixedPrecisionConfig:
    """Mixed precision configuration for Megatron models.

    Handles conversion of model parameters and inputs/outputs between different precisions,
    and manages mixed precision training settings.
    """

    precision: Literal["16-mixed", "bf16-mixed", "32"]
    params_dtype: Optional[torch.dtype] = None
    pipeline_dtype: Optional[torch.dtype] = None
    autocast_dtype: Optional[torch.dtype] = None
    autocast_enabled: bool = False
    grad_reduce_in_fp32: bool = True
    # fp8 related
    fp8: Optional[str] = None
    fp8_recipe: str = "delayed"  # "tensorwise", "delayed", "mxfp8" (for Blackwell only)
    first_last_layers_bf16: bool = False
    fp8_margin: int = 0
    fp8_amax_history_len: int = 1
    fp8_amax_compute_algo: str = "most_recent"
    fp8_wgrad: bool = True
    fp8_dot_product_attention: bool = False
    fp8_multi_head_attention: bool = False
    fp8_param_gather: bool = False
    fp16_loss_scale: Optional[float] = None
    fp16_initial_loss_scale: float = 4294967296
    fp16_min_loss_scale: float = 1.0
    fp16_loss_scale_window: int = 1000
    fp16_hysteresis: int = 2
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0

    def __post_init__(self):
        # Convert precision string if needed
        if isinstance(self.precision, int):
            self.precision = str(self.precision)

        # Determine default dtype based on precision
        dtype = torch.bfloat16 if self.precision in ["bf16", "bf16-mixed"] else torch.float32

        # Create internal dtype config
        self.dtype_config = DtypeConfig(
            fp32=self.precision in ["fp32", "32"],
            fp16=self.precision in ["fp16", "fp16-mixed", "16", "16-mixed"],
            bf16=self.precision in ["bf16", "bf16-mixed"],
            params_dtype=self.params_dtype or torch.float32,
            pipeline_dtype=self.pipeline_dtype or dtype,
            autocast_dtype=self.autocast_dtype or dtype,
            autocast_enabled=self.autocast_enabled,
            grad_reduce_in_fp32=self.grad_reduce_in_fp32,
            fp8=self.fp8,
            fp8_recipe=self.fp8_recipe,
            first_last_layers_bf16=self.first_last_layers_bf16,
            fp8_margin=self.fp8_margin,
            fp8_amax_history_len=self.fp8_amax_history_len,
            fp8_amax_compute_algo=self.fp8_amax_compute_algo,
            fp8_wgrad=self.fp8_wgrad,
            fp8_dot_product_attention=self.fp8_dot_product_attention,
            fp8_multi_head_attention=self.fp8_multi_head_attention,
            fp8_param=self.fp8_param_gather,
            fp8_param_gather=self.fp8_param_gather,
            num_layers_at_start_in_bf16=self.num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=self.num_layers_at_end_in_bf16,
            # fp16 loss scale
            loss_scale=self.fp16_loss_scale,
            initial_loss_scale=self.fp16_initial_loss_scale,
            min_loss_scale=self.fp16_min_loss_scale,
            loss_scale_window=self.fp16_loss_scale_window,
            hysteresis=self.fp16_hysteresis,
        )

    def setup(
        self,
        model_config: GPTConfig | T5Config,
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
        model_config = update_config_with_dtype_overrides(self.dtype_config, model_config)

        # Update optimizer config if provided
        if optimizer_config is not None:
            optimizer_config = update_config_with_dtype_overrides(self.dtype_config, optimizer_config)

        # Update DDP config if provided
        if ddp_config is not None:
            ddp_config = update_config_with_dtype_overrides(self.dtype_config, ddp_config)


def update_config_with_dtype_overrides(dtype_config: DtypeConfig, config):
    """Update a config object with dtype settings from dtype_config.

    Args:
        dtype_config: Source of dtype settings
        config: Config object to update

    Returns:
        Updated config object
    """
    if hasattr(config, "__io__"):
        config.__io__ = update_config_with_dtype_overrides(dtype_config, config.__io__)
    for field in fields(dtype_config):
        if not hasattr(config, field.name):
            continue
        # If we overwrote a value, log a debug message.
        old_val = getattr(config, field.name)
        new_val = getattr(dtype_config, field.name)
        if old_val != new_val:
            setattr(config, field.name, new_val)
            logging.debug(f"Overwrote {type(config).__name__}.{field.name}  {old_val} -> {new_val}")
    return config
