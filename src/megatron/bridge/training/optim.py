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

from typing import Dict, Optional, Union

from megatron.core.optimizer import (
    MegatronOptimizer,
    OptimizerConfig,
    ParamKey,
    get_megatron_optimizer,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler, ParamGroupOverride
from megatron.core.transformer.module import MegatronModule

from megatron.bridge.training.config import SchedulerConfig


def _build_config_overrides(
    scheduler_config: SchedulerConfig,
    model: Union[MegatronModule, list[MegatronModule]],
) -> Optional[Dict[ParamKey, ParamGroupOverride]]:
    """Build config overrides for weight decay based on scheduler configuration.

    This function creates parameter-specific overrides for weight decay behavior.
    By default, weight decay is skipped for bias parameters and 1D parameters.
    For Qwen3-Next models, weight decay is applied to q_layernorm and k_layernorm.

    Args:
        scheduler_config: Scheduler configuration containing weight decay settings
        model: The model or list of model chunks to collect parameter names from

    Returns:
        Dictionary of ParamKey to ParamGroupOverride for the optimizer
    """
    config_overrides: Dict[ParamKey, ParamGroupOverride] = {}

    # Collect param names that should skip weight decay
    no_wd_names: list[str] = []
    is_qwen3_next = scheduler_config.no_weight_decay_cond_type == "qwen3_next"

    model_list = model if isinstance(model, list) else [model]
    for model_chunk in model_list:
        for name, param in model_chunk.named_parameters():
            # Skip weight decay for bias parameters
            if name.endswith(".bias"):
                no_wd_names.append(name)
                continue

            # Skip weight decay for 1D parameters
            if len(param.shape) == 1:
                if is_qwen3_next:
                    # Qwen3-Next: apply weight decay to qk layernorm (don't add to skip list)
                    if "q_layernorm" in name or "k_layernorm" in name:
                        continue
                no_wd_names.append(name)

    # Create a single ParamKey with all names that should skip weight decay
    if no_wd_names:
        no_wd_key = ParamKey(name=tuple(no_wd_names))
        config_overrides[no_wd_key] = ParamGroupOverride(wd_mult=0.0)

    return config_overrides if config_overrides else None


def setup_optimizer(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    model: Union[MegatronModule, list[MegatronModule]],
    use_gloo_process_groups: bool = False,
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    """Set up the optimizer and scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        model: The model to optimize
        use_gloo_process_groups: Whether to use Gloo process groups

    Returns:
        tuple containing the optimizer and scheduler
    """
    # Build config overrides for weight decay based on scheduler config and model params
    config_overrides = _build_config_overrides(scheduler_config, model)

    if "muon" not in optimizer_config.optimizer and "soap" not in optimizer_config.optimizer:
        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
        )
    else:
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in optimizer_config.optimizer,
        )

    scheduler = _get_scheduler(optimizer_config, scheduler_config, optimizer)

    return optimizer, scheduler


def _get_scheduler(
    optimizer_config: OptimizerConfig, scheduler_config: SchedulerConfig, optimizer: MegatronOptimizer
) -> OptimizerParamScheduler:
    """Get the optimizer parameter scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        optimizer: The optimizer to schedule

    Returns:
        The optimizer parameter scheduler
    """
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=scheduler_config.lr_warmup_init,
        max_lr=optimizer_config.lr,
        min_lr=optimizer_config.min_lr,
        lr_warmup_steps=scheduler_config.lr_warmup_steps,
        lr_decay_steps=scheduler_config.lr_decay_steps,
        lr_decay_style=scheduler_config.lr_decay_style,
        start_wd=scheduler_config.start_weight_decay,
        end_wd=scheduler_config.end_weight_decay,
        wd_incr_steps=scheduler_config.wd_incr_steps,
        wd_incr_style=scheduler_config.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=scheduler_config.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=scheduler_config.override_opt_param_scheduler,
        wsd_decay_steps=scheduler_config.wsd_decay_steps,
        lr_wsd_decay_style=scheduler_config.lr_wsd_decay_style,
    )

    return scheduler
