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
from typing import Optional, Union

from megatron.core.optimizer import (
    MegatronOptimizer,
    OptimizerConfig,
    get_megatron_optimizer,
    get_mup_config_overrides,
)
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.utils import get_model_config

from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
    SchedulerConfig,
)


G_LOGGER = logging.getLogger(__name__)


def setup_optimizer(
    optimizer_config: OptimizerConfig,
    scheduler_config: SchedulerConfig,
    model: Union[MegatronModule, list[MegatronModule]],
    use_gloo_process_groups: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
    optimizer_config_override_provider: Optional[OptimizerConfigOverrideProvider] = None,
) -> tuple[MegatronOptimizer, OptimizerParamScheduler]:
    """Set up the optimizer and scheduler.

    Args:
        optimizer_config: Configuration for the optimizer
        scheduler_config: Configuration for the scheduler
        model: The model to optimize
        use_gloo_process_groups: Whether to use Gloo process groups
        pg_collection: Optional process group collection for distributed training

    Returns:
        tuple containing the optimizer and scheduler
    """
    if optimizer_config_override_provider is None:
        optimizer_config_override_provider = OptimizerConfigOverrideProvider()

    # Build config overrides for weight decay based on scheduler config and model params
    config_overrides = optimizer_config_override_provider.build_config_overrides(
        OptimizerConfigOverrideProviderContext(scheduler_config, optimizer_config, model)
    )

    # Apply μP optimizer scaling if enabled on the model config.
    model_chunks = model if isinstance(model, list) else [model]
    model_config = get_model_config(model_chunks[0])
    if getattr(model_config, "use_mup", False):
        mup_overrides = get_mup_config_overrides(
            config=optimizer_config,
            mup_width_mult=model_config.mup_width_mult,
            optimizer_type=optimizer_config.optimizer,
        )
        if mup_overrides:
            config_overrides = {**(config_overrides or {}), **mup_overrides}
            G_LOGGER.info(
                f"μP enabled (width_mult={model_config.mup_width_mult:.4g}): "
                f"applied {len(mup_overrides)} optimizer param-group override(s)."
            )

    if hasattr(optimizer_config, "provide"):
        optimizer = optimizer_config.provide(
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            pg_collection=pg_collection,
        )
    elif "muon" not in optimizer_config.optimizer and "soap" not in optimizer_config.optimizer:
        optimizer = get_megatron_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            pg_collection=pg_collection,
        )
    else:
        optimizer = get_megatron_muon_optimizer(
            config=optimizer_config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=use_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in optimizer_config.optimizer,
            pg_collection=pg_collection,
        )

    scheduler = _get_scheduler(optimizer_config, scheduler_config, optimizer)

    return optimizer, scheduler


def sync_hybrid_device_optimizer_fp32_master_copies(optimizer: MegatronOptimizer | None) -> bool:
    """Refresh ``HybridDeviceOptimizer`` FP32 master copies from BF16 model parameters.

    Workaround for an upstream Megatron-Core gap: when a checkpoint is loaded
    into the BF16 model parameters, ``reload_model_params()`` only refreshes
    the level-1 FP32 GPU shards inside ``HybridDeviceOptimizer``.  The level-2
    CPU clones (``gpu_params_map_cpu_copy``) and the level-3 FP32 working copy
    (``param_to_fp32_param``) keep their default initialisation values.

    Without this helper, the first optimizer step on a fine-tuning run that
    combines ``optimizer_cpu_offload=True`` + distributed optimizer + BF16
    runs Adam on stale FP32 masters and writes the result back into the BF16
    model, effectively reverting the loaded weights to fresh ``nn.Module``
    random init.  Training loss looks plausible at step 1 and collapses at
    step 2 because the model is no longer the one loaded from the checkpoint.

    Mirrors the workaround in NVIDIA-NeMo/RL PR #2372.  Once mcore's
    ``reload_model_params()`` walks all three FP32 levels, this helper can be
    removed from both Bridge and RL.

    Args:
        optimizer: The Megatron optimizer returned by :func:`setup_optimizer`.
            No-op when ``None`` or when no sub-optimizer wraps a
            ``HybridDeviceOptimizer`` (i.e. CPU offload is not enabled).

    Returns:
        ``True`` when at least one ``HybridDeviceOptimizer`` sub-optimizer was
        synced; ``False`` otherwise.
    """
    if optimizer is None:
        return False

    try:
        from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
    except ImportError:
        return False

    def _sync_one(distrib_opt: MegatronOptimizer) -> bool:
        """Sync the three FP32 master levels for one DistributedOptimizer wrapping an HDO."""
        inner = getattr(distrib_opt, "optimizer", None)
        if not isinstance(inner, HybridDeviceOptimizer):
            return False

        # Level 1: per-DP-rank FP32 GPU shards (Adam master parameters).
        for model_group, shard_main_group in zip(
            distrib_opt.model_float16_groups,
            distrib_opt.shard_fp32_from_float16_groups,
        ):
            for model_param, shard_main_param in zip(model_group, shard_main_group):
                if shard_main_param is None:
                    continue
                param_range_map = distrib_opt._get_model_param_range_map(model_param)
                param_range = param_range_map["param"]
                shard_model_param = model_param.view(-1)[param_range.start : param_range.end]
                shard_main_param.data.copy_(shard_model_param)

        # Level 2: CPU clones the CPU sub-optimizer steps against.
        if hasattr(inner, "gpu_params_map_cpu_copy"):
            for gpu_param, cpu_clone in inner.gpu_params_map_cpu_copy.items():
                cpu_clone.data.copy_(gpu_param.data)

        # Level 3: FP32 working copy kept for the async D2H/H2D dance.
        if hasattr(inner, "update_fp32_param_by_new_param"):
            inner.update_fp32_param_by_new_param()

        return True

    synced = False
    if hasattr(optimizer, "chained_optimizers"):
        for sub_opt in optimizer.chained_optimizers:
            synced |= _sync_one(sub_opt)
    else:
        synced = _sync_one(optimizer)

    if synced:
        G_LOGGER.info(
            "Synced HybridDeviceOptimizer FP32 master copies from BF16 model parameters "
            "after checkpoint load (workaround for upstream mcore reload_model_params() gap)."
        )
    return synced


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
