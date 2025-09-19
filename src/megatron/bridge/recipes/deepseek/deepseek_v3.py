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
import os
from typing import List, Optional, Union

import torch

from megatron.bridge.models.deepseek import DeepSeekV3Provider
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


logger = logging.getLogger(__name__)


def model_config(
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 16,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    expert_parallelism: int = 64,
    sequence_parallelism: bool = True,
    # MTP support
    mtp_num_layers: Optional[int] = 1,
    mtp_loss_scaling_factor: Optional[float] = 0.1,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    enable_deepep: bool = False,
    apply_rope_fusion: bool = True,
) -> DeepSeekV3Provider:
    """
    Configure the DeepSeek-V3 (671B) model.

    Args:
        tensor_parallelism: Degree of tensor model parallelism.
        pipeline_parallelism: Degree of pipeline model parallelism.
        pipeline_parallelism_dtype: Data type for pipeline parallelism.
        virtual_pipeline_parallelism: Size of virtual pipeline parallelism.
        context_parallelism: Degree of context parallelism.
        expert_parallelism: Degree of expert model parallelism.
        sequence_parallelism: Whether to use sequence parallelism.
        mtp_num_layers: Number of MTP layers.
        mtp_loss_scaling_factor: Loss scaling factor for MTP.
        recompute_granularity: Recomputation granularity. For V3 we recommend "selective".
        recompute_modules: Modules to selectively recompute when granularity is "selective".
        recompute_method: Method for activation recomputation.
        recompute_num_layers: Number of layers to recompute.
        apply_rope_fusion: Whether to apply MLA Yarn fusion.
    Returns:
        DeepSeekV3Provider: Configuration for the DeepSeek-V3 model.
    """
    cfg = DeepSeekV3Provider(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        expert_model_parallel_size=expert_parallelism,
        expert_tensor_parallel_size=1,
        sequence_parallel=sequence_parallelism,
        # MTP
        mtp_num_layers=mtp_num_layers,
        mtp_loss_scaling_factor=mtp_loss_scaling_factor,
        # Recomputation
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
    )

    # Pipeline split for asymmetric stages are specified with map_pp_vp_to_layout below
    cfg.account_for_embedding_in_pipeline_split = False
    cfg.account_for_loss_in_pipeline_split = False
    cfg.num_layers_in_first_pipeline_stage = None
    cfg.num_layers_in_last_pipeline_stage = None

    # Performance optimization knobs
    cfg.moe_permute_fusion = True
    if apply_rope_fusion:
        cfg.apply_rope_fusion = True

    # Pipeline parallelism configs. We infer PP layout from the provided PP and VP size
    if mtp_num_layers is None:
        mtp_num_layers = 0
    last_layer = ["mtp"] * mtp_num_layers + ["loss"]
    map_pp_vp_to_layout = {
        (1, 1): None,
        (4, 1): [["embedding"] + ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 16, ["decoder"] * 13 + last_layer],
        (8, 1): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + last_layer],
        (4, 2): [["embedding"] + ["decoder"] * 8] + [["decoder"] * 8] * 6 + [["decoder"] * 5 + last_layer],
        (16, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
        (8, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
        (4, 4): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 14 + [["decoder"] + last_layer],
    }
    pp_size = pipeline_parallelism or 1
    vp_size = virtual_pipeline_parallelism or 1
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for DeepSeek V3. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )

    layout = map_pp_vp_to_layout[(pp_size, vp_size)]

    if layout is not None:
        layout = list([list(x) for x in layout])  # yield all the elements
    cfg.pipeline_model_parallel_layout = layout

    if enable_deepep:
        cfg.moe_token_dispatcher_type = "flex"
        cfg.moe_enable_deepep = True
        cfg.moe_shared_expert_overlap = False

    return cfg


def pretrain_config(
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 16,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    expert_parallelism: int = 64,
    sequence_parallelism: bool = True,
    use_megatron_fsdp: bool = False,
    mtp_num_layers: Optional[int] = 1,
    mtp_loss_scaling_factor: Optional[float] = 0.1,
    # Training hyperparameters
    train_iters: int = 1_000_000,
    global_batch_size: int = 4096,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    enable_deepep: bool = False,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    apply_rope_fusion: bool = False,
) -> ConfigContainer:
    """
    Create a pre-training configuration for DeepSeek-V3 (671B) model.

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    model_cfg = model_config(
        tensor_parallelism=tensor_parallelism,
        pipeline_parallelism=pipeline_parallelism,
        pipeline_parallelism_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_parallelism=virtual_pipeline_parallelism,
        context_parallelism=context_parallelism,
        expert_parallelism=expert_parallelism,
        sequence_parallelism=sequence_parallelism,
        mtp_num_layers=mtp_num_layers,
        mtp_loss_scaling_factor=mtp_loss_scaling_factor,
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        enable_deepep=enable_deepep,
        apply_rope_fusion=apply_rope_fusion,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )
    opt_config.use_precision_aware_optimizer = True
    opt_config.main_params_dtype = torch.float32
    opt_config.main_grads_dtype = torch.bfloat16
    opt_config.exp_avg_dtype = torch.bfloat16
    opt_config.exp_avg_sq_dtype = torch.bfloat16

    if precision_config is None:
        precision_config = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=5,
            manual_gc_eval=5,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,  # V3 recipe sets this to False
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            sequence_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
    if apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg


def pretrain_config_32nodes(**kwargs):
    """
    Create a pre-training configuration for DeepSeek-V3 (671B) model with minimal number of nodes (32).

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    return pretrain_config(
        pipeline_parallelism=8,
        expert_parallelism=32,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        **kwargs,
    )
