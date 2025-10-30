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
from typing import Any, Optional

from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_delayed_scaling_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)


logger = logging.getLogger(__name__)


def get_user_parallelism_and_batch_size_configs(kwargs: Any):
    """Extract user-specified parallelism and batch size overrides from kwargs."""
    tp = kwargs.get("tensor_model_parallel_size", None)
    pp = kwargs.get("pipeline_model_parallel_size", None)
    cp = kwargs.get("context_parallel_size", None)
    vp = kwargs.get("virtual_pipeline_model_parallel_size", None)
    ep = kwargs.get("expert_model_parallel_size", None)
    etp = kwargs.get("expert_tensor_parallel_size", None)

    mbs = kwargs.get("micro_batch_size", None)
    gbs = kwargs.get("global_batch_size", None)
    return tp, pp, cp, vp, ep, etp, mbs, gbs


def set_basic_perf_overrides(recipe: Any, max_steps: Optional[int] = 50) -> None:
    """Apply common performance overrides shared across recipes."""
    recipe.train.train_iters = max_steps
    recipe.train.eval_iters = 0

    recipe.checkpoint.save = None

    recipe.logger.log_interval = 1
    recipe.logger.tensorboard_dir = None

    recipe.ddp.check_for_nan_in_grad = False
    recipe.ddp.check_for_large_grads = False

    recipe.rerun_state_machine.check_for_nan_in_loss = False

    recipe.scheduler.lr_decay_iters = recipe.train.train_iters
    recipe.scheduler.lr_warmup_iters = 10


def set_megatron_fsdp_overrides(recipe: Any, perf_overrides: Any) -> None:
    """Set the mcore fsdp overrides from the performance matrix."""
    use_megatron_fsdp = perf_overrides.get("use_megatron_fsdp", False)
    if use_megatron_fsdp:
        recipe.ddp.use_megatron_fsdp = True
        recipe.ddp.data_parallel_sharding_strategy = "optim_grads_params"
        recipe.ddp.keep_fp8_transpose_cache = False
        # average_in_collective is not supported with Megatron FSDP
        recipe.ddp.average_in_collective = False

        recipe.model.init_model_with_meta_device = True
        recipe.model.gradient_accumulation_fusion = True

        if recipe.comm_overlap is not None and isinstance(recipe.comm_overlap, CommOverlapConfig):
            if recipe.comm_overlap.defer_embedding_wgrad_compute:
                logger.warning(
                    "Disabling deferring embedding wgrad compute because it cannot work with FSDP together."
                )
                recipe.comm_overlap.defer_embedding_wgrad_compute = False

        if recipe.optimizer.use_precision_aware_optimizer:
            recipe.optimizer.use_precision_aware_optimizer = False
            logger.warning("Disabling precision aware optimizer because it cannot work with FSDP together.")

        recipe.checkpoint.load = None


def get_precision_config(compute_dtype: str, fp8_recipe: Optional[str] = None):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    if compute_dtype == "fp8":
        if fp8_recipe == "ds":
            return bf16_with_fp8_delayed_scaling_mixed()
        elif fp8_recipe == "cs":
            current_scaling_cfg = bf16_with_fp8_current_scaling_mixed()
            # Disable BF16 Transformer layers in the performance config
            current_scaling_cfg.first_last_layers_bf16 = False
            return current_scaling_cfg
        elif fp8_recipe == "mx":
            return bf16_with_mxfp8_mixed()
        elif fp8_recipe == "ss":
            return bf16_with_fp8_subchannel_scaling_mixed()
        else:
            raise ValueError(f"Invalid FP8 recipe: {fp8_recipe}")
    elif compute_dtype == "bf16":
        return bf16_mixed()
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype}")


def set_cuda_graph_overrides(
    recipe: Any, cuda_graph_impl: Optional[str] = None, cuda_graph_scope: Optional[str] = None
) -> None:
    """Set the CUDA graph overrides from the performance matrix."""
    recipe.model.cuda_graph_impl = cuda_graph_impl
    recipe.model.cuda_graph_scope = cuda_graph_scope

    if cuda_graph_impl is not None:
        recipe.model.use_te_rng_tracker = True
        recipe.rng.te_rng_tracker = True

    if cuda_graph_impl == "transformer_engine":
        assert cuda_graph_scope in ["full", "attn"], (
            f"Invalid cuda graph scope: {cuda_graph_scope}. Valid options are: full, attn"
        )


def set_recompute_overrides(recipe: Any, perf_overrides: Any) -> None:
    """Set the recompute num layers overrides from the performance matrix."""
    recompute_num_layers = perf_overrides.get("recompute_num_layers", None)
    if recompute_num_layers is not None:
        recipe.model.recompute_granularity = "full"
        recipe.model.recompute_method = "block"
        recipe.model.recompute_num_layers = recompute_num_layers

    cpu_offloading_num_layers = perf_overrides.get("cpu_offloading_num_layers", 0)
    if cpu_offloading_num_layers > 0:
        recipe.model.cpu_offloading = True
        recipe.model.cpu_offloading_weights = False
        recipe.model.cpu_offloading_num_layers = cpu_offloading_num_layers


def moe_a2a_1f1b_overrides(recipe: Any) -> None:
    """Tune configuration for MoE A2A 1F1B communication overlap."""
    recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
    recipe.comm_overlap.delay_wgrad_compute = True
    recipe.model.moe_shared_expert_overlap = False
