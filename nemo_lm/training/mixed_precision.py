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
from typing import Optional

import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo_lm.models.gpt import GPTConfig
from nemo_lm.models.t5 import T5Config


@dataclass(kw_only=True)
class MegatronMixedPrecisionConfig:
    """Mixed precision configuration for Megatron models.

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
    fp8_recipe: str = "delayed"  # "tensorwise", "delayed", "mxfp8" (for Blackwell only)
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
    loss_scale: Optional[float] = None
    initial_loss_scale: Optional[float] = None
    min_loss_scale: Optional[float] = None
    loss_scale_window: Optional[float] = None
    hysteresis: Optional[float] = None
    num_layers_at_start_in_bf16: int = 0
    num_layers_at_end_in_bf16: int = 0

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
        model_config = update_config_with_precision_overrides(self, model_config)

        # Update optimizer config if provided
        if optimizer_config is not None:
            optimizer_config = update_config_with_precision_overrides(self, optimizer_config)

        # Update DDP config if provided
        if ddp_config is not None:
            ddp_config = update_config_with_precision_overrides(self, ddp_config)


def update_config_with_precision_overrides(mixed_precision_config: MegatronMixedPrecisionConfig, config):
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
