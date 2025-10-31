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
from typing import Any, Dict, Optional

from utils.utils import get_model_recipe

from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_delayed_scaling_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)


logger = logging.getLogger(__name__)


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


def set_basic_perf_overrides(recipe: ConfigContainer) -> None:
    """Apply common performance overrides shared across recipes."""
    recipe.train.train_iters = 50
    recipe.train.eval_iters = 0

    recipe.checkpoint.save = None

    recipe.logger.log_interval = 1
    recipe.logger.tensorboard_dir = None

    recipe.ddp.check_for_nan_in_grad = False
    recipe.ddp.check_for_large_grads = False

    recipe.rerun_state_machine.check_for_nan_in_loss = False

    recipe.scheduler.lr_decay_iters = recipe.train.train_iters
    recipe.scheduler.lr_warmup_iters = 10

    recipe.mixed_precision.grad_reduce_in_fp32 = False
    recipe.ddp.grad_reduce_in_fp32 = False


def set_megatron_fsdp_overrides(recipe: ConfigContainer) -> None:
    """Set the Megatron FSDP overrides."""
    recipe.ddp.use_megatron_fsdp = True
    recipe.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    recipe.ddp.keep_fp8_transpose_cache = False
    # average_in_collective is not supported with Megatron FSDP
    recipe.ddp.average_in_collective = False

    recipe.model.init_model_with_meta_device = True
    recipe.model.gradient_accumulation_fusion = True

    if recipe.comm_overlap is not None and isinstance(recipe.comm_overlap, CommOverlapConfig):
        if recipe.comm_overlap.defer_embedding_wgrad_compute:
            logger.warning("Disabling deferring embedding wgrad compute because it cannot work with FSDP together.")
            recipe.comm_overlap.defer_embedding_wgrad_compute = False

    if recipe.optimizer.use_precision_aware_optimizer:
        recipe.optimizer.use_precision_aware_optimizer = False
        logger.warning("Disabling precision aware optimizer because it cannot work with FSDP together.")

    recipe.checkpoint.load = None


def set_cuda_graph_overrides(
    recipe: Any, cuda_graph_impl: Optional[str] = None, cuda_graph_scope: str = "full"
) -> None:
    """Set the CUDA graph overrides."""
    if cuda_graph_impl is not None:
        recipe.model.cuda_graph_impl = cuda_graph_impl
        if cuda_graph_impl != "none":
            recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = True
        else:
            recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = False

    if cuda_graph_impl == "transformer_engine":
        assert cuda_graph_scope in ["full", "attn"], (
            f"Invalid cuda graph scope: {cuda_graph_scope}. Valid options are: full, attn"
        )

    recipe.model.cuda_graph_scope = cuda_graph_scope


def set_recompute_overrides(
    recipe: Any,
    recompute_num_layers: Optional[int] = None,
    cpu_offloading_num_layers: Optional[int] = None,
) -> None:
    """Set the recompute num layers overrides."""
    if recompute_num_layers is not None:
        recipe.model.recompute_granularity = "full"
        recipe.model.recompute_method = "block"
        recipe.model.recompute_num_layers = recompute_num_layers
    if cpu_offloading_num_layers is not None:
        recipe.model.cpu_offloading = True
        recipe.model.cpu_offloading_weights = False
        recipe.model.cpu_offloading_num_layers = cpu_offloading_num_layers


def set_moe_a2a_1f1b_overrides(recipe: ConfigContainer) -> None:
    """Tune configuration for MoE A2A 1F1B communication overlap."""
    recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
    recipe.comm_overlap.delay_wgrad_compute = True
    recipe.model.moe_shared_expert_overlap = False


def set_user_overrides(recipe: ConfigContainer, kwargs: Dict[str, Any]) -> None:
    """Set the user overrides."""
    set_basic_perf_overrides(recipe)
    if kwargs.get("max_steps") is not None:
        recipe.train.train_iters = kwargs.get("max_steps")

    use_megatron_fsdp = kwargs.get("use_megatron_fsdp")
    if use_megatron_fsdp:
        set_megatron_fsdp_overrides(recipe)

    cuda_graph_impl = kwargs.get("cuda_graph_impl")
    cuda_graph_scope = kwargs.get("cuda_graph_scope")
    set_cuda_graph_overrides(recipe, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope=cuda_graph_scope)

    recompute_num_layers = kwargs.get("recompute_num_layers")
    cpu_offloading_num_layers = kwargs.get("activation_offload_layers")
    set_recompute_overrides(
        recipe, recompute_num_layers=recompute_num_layers, cpu_offloading_num_layers=cpu_offloading_num_layers
    )

    moe_a2a = kwargs.get("moe_a2a")
    if moe_a2a:
        set_moe_a2a_1f1b_overrides(recipe)

    use_tokendrop = kwargs.get("use_tokendrop")
    if use_tokendrop:
        recipe.model = apply_moe_token_drop(recipe.model)
    if use_tokendrop is not None and not use_tokendrop:  # explicitly set to False by user
        recipe.model.moe_router_force_load_balancing = True

    if kwargs.get("tensor_model_parallel_size") is not None:
        recipe.model.tensor_model_parallel_size = kwargs.get("tensor_model_parallel_size")
    if kwargs.get("pipeline_model_parallel_size") is not None:
        recipe.model.pipeline_model_parallel_size = kwargs.get("pipeline_model_parallel_size")
    if kwargs.get("context_parallel_size") is not None:
        recipe.model.context_parallel_size = kwargs.get("context_parallel_size")
    if kwargs.get("virtual_pipeline_model_parallel_size") is not None:
        recipe.model.virtual_pipeline_model_parallel_size = kwargs.get("virtual_pipeline_model_parallel_size")
    if kwargs.get("expert_model_parallel_size") is not None:
        recipe.model.expert_model_parallel_size = kwargs.get("expert_model_parallel_size")
    if kwargs.get("expert_tensor_parallel_size") is not None:
        recipe.model.expert_tensor_parallel_size = kwargs.get("expert_tensor_parallel_size")
    if kwargs.get("global_batch_size") is not None:
        recipe.train.global_batch_size = kwargs.get("global_batch_size")
    if kwargs.get("micro_batch_size") is not None:
        recipe.train.micro_batch_size = kwargs.get("micro_batch_size")

    if kwargs.get("compute_dtype") == "bf16":
        recipe.optimizer.use_precision_aware_optimizer = True
    if recipe.model.use_transformer_engine_op_fuser:
        recipe.model.use_transformer_engine_op_fuser = False
    recipe.model.apply_rope_fusion = True

    tp = recipe.model.tensor_model_parallel_size
    pp = recipe.model.pipeline_model_parallel_size
    cp = recipe.model.context_parallel_size
    vp = recipe.model.virtual_pipeline_model_parallel_size or 1

    dp = int(kwargs.get("num_gpus") / (tp * pp * cp))
    logger.info(f"DP: {dp}; TP: {tp}; PP: {pp}; CP: {cp}; VP: {vp}")
    if dp > 1 and pp > 1 and vp > 1:
        recipe.optimizer.overlap_param_gather_with_optimizer_step = True
        recipe.comm_overlap.overlap_param_gather_with_optimizer_step = True

    return recipe


def get_model_recipe_with_user_overrides(**kwargs) -> ConfigContainer:
    """Get the model recipe with user overrides."""
    model_name = kwargs.get("model_name")
    model_size = kwargs.get("model_size")
    gpu = kwargs.get("gpu")
    num_gpus = kwargs.get("num_gpus")
    compute_dtype = kwargs.get("compute_dtype")
    fp8_recipe = kwargs.get("fp8_recipe")

    recipe = get_model_recipe(model_name, model_size, gpu, num_gpus, compute_dtype, fp8_recipe)

    recipe = set_user_overrides(recipe, kwargs)

    return recipe
