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
from typing import Any, Dict, List, Optional

from utils.utils import WorkloadBaseConfig, get_model_recipe, get_workload_base_config

from megatron.bridge.training.comm_overlap import *
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_fp8_subchannel_scaling_mixed,
    bf16_with_mxfp8_mixed,
)
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop


logger = logging.getLogger(__name__)


def get_precision_config(compute_dtype: str):
    """Get the precision configs for the given compute dtype and FP8 recipe."""
    if compute_dtype == "fp8_cs":
        current_scaling_cfg = bf16_with_fp8_current_scaling_mixed()
        # Disable BF16 Transformer layers in the performance config
        current_scaling_cfg.first_last_layers_bf16 = False
        return current_scaling_cfg
    elif compute_dtype == "fp8_mx":
        return bf16_with_mxfp8_mixed()
    elif compute_dtype == "fp8_sc":
        return bf16_with_fp8_subchannel_scaling_mixed()
    elif compute_dtype == "bf16":
        return bf16_mixed()
    else:
        raise ValueError(f"Invalid compute dtype: {compute_dtype}")


def set_workload_base_configs(cfg: ConfigContainer, settings: WorkloadBaseConfig) -> None:
    """Set workload base configs."""
    cfg.model.tensor_model_parallel_size = settings.tensor_model_parallel_size
    cfg.model.pipeline_model_parallel_size = settings.pipeline_model_parallel_size
    cfg.model.context_parallel_size = settings.context_parallel_size
    cfg.model.virtual_pipeline_model_parallel_size = settings.virtual_pipeline_model_parallel_size
    cfg.model.expert_model_parallel_size = settings.expert_model_parallel_size
    cfg.model.expert_tensor_parallel_size = settings.expert_tensor_parallel_size
    cfg.model.sequence_parallel = settings.sequence_parallel

    cfg.train.global_batch_size = settings.global_batch_size
    cfg.train.micro_batch_size = settings.micro_batch_size

    if settings.use_megatron_fsdp:
        set_megatron_fsdp_overrides(cfg)
    if settings.cuda_graph_impl is not None or settings.cuda_graph_scope is not None:
        set_cuda_graph_overrides(
            cfg,
            cuda_graph_impl=settings.cuda_graph_impl,
            cuda_graph_scope=settings.cuda_graph_scope,
        )
    set_recompute_overrides(
        cfg,
        recompute_modules=settings.recompute_modules,
        cpu_offloading_num_layers=settings.cpu_offloading_num_layers,
        recompute_num_layers=settings.recompute_num_layers,
    )


def set_common_perf_overrides(recipe: ConfigContainer) -> None:
    """Set common performance overrides shared across recipes."""
    recipe.train.train_iters = 50
    recipe.train.eval_iters = 0

    recipe.checkpoint.save = None

    recipe.logger.log_interval = 1
    recipe.logger.tensorboard_dir = None
    recipe.logger.save_config_filepath = "/nemo_run/configs/ConfigContainer.yaml"

    recipe.ddp.check_for_nan_in_grad = False
    recipe.ddp.check_for_large_grads = False

    recipe.rerun_state_machine.check_for_nan_in_loss = False

    recipe.scheduler.lr_decay_iters = recipe.train.train_iters
    recipe.scheduler.lr_warmup_iters = 10

    if hasattr(recipe.model, "use_transformer_engine_op_fuser") and recipe.model.use_transformer_engine_op_fuser:
        recipe.model.use_transformer_engine_op_fuser = False
    recipe.model.apply_rope_fusion = True
    recipe.model.cross_entropy_fusion_impl = "te"


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
        else:  # this condition ensures we unset in case of user override to "none" from default
            recipe.rng.te_rng_tracker = recipe.model.use_te_rng_tracker = False

        if cuda_graph_impl == "transformer_engine":
            valid_te_scopes = ["attn", "mlp", "moe", "moe_router", "moe_preprocess", "mamba"]
            assert all(scope in valid_te_scopes for scope in cuda_graph_scope), (
                f"Invalid cuda graph scope: {cuda_graph_scope}. Valid options are: {valid_te_scopes}"
            )

    if cuda_graph_scope is not None:
        recipe.model.cuda_graph_scope = cuda_graph_scope


def set_recompute_overrides(
    recipe: Any,
    cpu_offloading_num_layers: Optional[int] = None,
    recompute_num_layers: Optional[int] = None,
    recompute_modules: Optional[List[str]] = None,
) -> None:
    """Set the recompute and CPU offloading overrides."""
    if cpu_offloading_num_layers is not None:
        recipe.model.cpu_offloading = True
        recipe.model.cpu_offloading_weights = False
        recipe.model.cpu_offloading_num_layers = cpu_offloading_num_layers
    if recompute_num_layers is not None:
        recipe.model.recompute_granularity = "full"
        recipe.model.recompute_method = "block"
        recipe.model.recompute_num_layers = recompute_num_layers
    if recompute_modules is not None:
        recipe.model.recompute_modules = recompute_modules


def set_moe_a2a_overlap_overrides(recipe: ConfigContainer) -> None:
    """Tune configuration for MoE A2A communication overlap."""
    recipe.comm_overlap.overlap_moe_expert_parallel_comm = True
    recipe.comm_overlap.delay_wgrad_compute = True
    recipe.model.moe_shared_expert_overlap = False


def set_user_overrides(recipe: ConfigContainer, kwargs: Dict[str, Any]) -> None:
    """Set the user overrides."""
    if kwargs.get("wandb_key") is not None:
        recipe.logger.wandb_project = kwargs.get("wandb_prj_name")
        recipe.logger.wandb_exp_name = kwargs.get("wandb_exp_name")
        recipe.logger.wandb_save_dir = "/nemo_run/wandb"

    if kwargs.get("max_steps") is not None:
        recipe.train.train_iters = kwargs.get("max_steps")

    use_megatron_fsdp = kwargs.get("use_megatron_fsdp")
    if use_megatron_fsdp:
        set_megatron_fsdp_overrides(recipe)

    cuda_graph_impl = kwargs.get("cuda_graph_impl")
    cuda_graph_scope = kwargs.get("cuda_graph_scope")
    if cuda_graph_impl is not None or cuda_graph_scope is not None:
        set_cuda_graph_overrides(
            recipe,
            cuda_graph_impl=cuda_graph_impl,
            cuda_graph_scope=cuda_graph_scope,
        )

    recompute_num_layers = kwargs.get("recompute_num_layers")
    cpu_offloading_num_layers = kwargs.get("activation_offload_layers")
    recompute_modules = kwargs.get("recompute_modules")
    set_recompute_overrides(
        recipe,
        recompute_num_layers=recompute_num_layers,
        cpu_offloading_num_layers=cpu_offloading_num_layers,
        recompute_modules=recompute_modules,
    )

    moe_a2a_overlap = kwargs.get("moe_a2a_overlap")
    if moe_a2a_overlap:
        set_moe_a2a_overlap_overrides(recipe)

    use_tokendrop = kwargs.get("use_tokendrop")
    if use_tokendrop:
        recipe.model = apply_moe_token_drop(recipe.model)
    if use_tokendrop is not None and not use_tokendrop:  # explicitly set to False by user
        recipe.model = apply_moe_token_drop(
            recipe.model, moe_expert_capacity_factor=-1.0, moe_pad_expert_input_to_capacity=False
        )
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

    tp = recipe.model.tensor_model_parallel_size
    pp = recipe.model.pipeline_model_parallel_size
    cp = recipe.model.context_parallel_size
    vp = recipe.model.virtual_pipeline_model_parallel_size or 1

    dp = int(kwargs.get("num_gpus") / (tp * pp * cp))
    logger.info(f"DP: {dp}; TP: {tp}; PP: {pp}; CP: {cp}; VP: {vp}")
    if dp > 1 and pp > 1 and vp > 1:
        recipe.optimizer.overlap_param_gather_with_optimizer_step = True
        recipe.comm_overlap.overlap_param_gather_with_optimizer_step = True


def get_model_recipe_with_user_overrides(**kwargs) -> ConfigContainer:
    """Get the model recipe with user overrides."""
    model_name = kwargs.get("model_name")
    model_size = kwargs.get("model_size")
    gpu = kwargs.get("gpu")
    num_gpus = kwargs.get("num_gpus")
    compute_dtype = kwargs.get("compute_dtype")

    recipe = get_model_recipe(model_name, model_size, gpu, compute_dtype)
    set_common_perf_overrides(recipe)
    set_user_overrides(recipe, kwargs)

    # Scale global batch size based on the number of GPUs IF GBS is not specified by the use 0 r
    workload_base_config = get_workload_base_config(model_name, model_size, gpu, compute_dtype)
    default_num_gpus = workload_base_config.num_gpus
    user_gbs = kwargs.get("global_batch_size")
    if user_gbs is None:
        if num_gpus != default_num_gpus:
            new_gbs = int(workload_base_config.gbs_scaling_factor * num_gpus)
            recipe.train.global_batch_size = new_gbs
            logger.info(
                f"Scaled global batch size from {workload_base_config.global_batch_size} to {new_gbs} based on {num_gpus} GPUs."
            )

    return recipe
