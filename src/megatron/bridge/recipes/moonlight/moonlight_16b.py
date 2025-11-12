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
from megatron.core.distributed import DistributedDataParallelConfig
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.deepseek import MoonlightModelProvider16B
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


logger = logging.getLogger(__name__)


class MoonlightCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Moonlight family recipe helpers."""

    # Core identifiers
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[List[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    per_split_data_args_path: Optional[str]
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: int
    sequence_parallel: bool
    # Recomputation
    recompute_granularity: str
    recompute_modules: Optional[List[str]]
    recompute_method: Optional[str]
    recompute_num_layers: Optional[int]
    enable_deepep: bool
    apply_rope_fusion: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    optimizer_type: str
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]


def moonlight_16b_pretrain_config(**user_kwargs: Unpack[MoonlightCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Moonlight-16B.

    See `_moonlight_common` for the full list of parameters.
    """
    recommended_kwargs: MoonlightCommonKwargs = {
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "sequence_parallel": True,
        "recompute_granularity": "selective",
        "enable_deepep": False,
        "apply_rope_fusion": False,
        "train_iters": 500_000,
        "global_batch_size": 2048,
        "micro_batch_size": 1,
        "seq_length": 4096,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "lr_warmup_iters": 2000,
        "optimizer_type": "adam",
        "eval_interval": 2000,
        "save_interval": 2000,
    }
    combined_kwargs: MoonlightCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _moonlight_common(**combined_kwargs)


def _moonlight_common(
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
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 2,
    pipeline_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 4,
    sequence_parallel: bool = True,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    enable_deepep: bool = False,
    apply_rope_fusion: bool = False,
    # Training hyperparameters
    train_iters: int = 500_000,
    global_batch_size: int = 2048,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    optimizer_type: str = "adam",
    eval_interval: int = 2000,
    save_interval: int = 2000,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Moonlight-16B model.

    Args:
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size (int): Degree of tensor model parallelism.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism.
        pipeline_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size (Optional[int]): Size of virtual pipeline parallelism.
        context_parallel_size (int): Degree of context parallelism.
        expert_model_parallel_size (int): Degree of expert model parallelism.
        sequence_parallel (bool): Whether to use sequence parallelism.
        recompute_granularity (str): Recomputation granularity.
        recompute_modules (Optional[List[str]]): Modules to recompute.
        recompute_method (Optional[str]): Recomputation method.
        recompute_num_layers (Optional[int]): Number of layers to recompute.
        enable_deepep (bool): Whether to use DeePEP.
        apply_rope_fusion (bool): Whether to apply RoPE fusion.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        optimizer_type (str): Type of optimizer to use.
        eval_interval (int): Interval for evaluation.
        save_interval (int): Interval for saving checkpoints.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration.

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

    model_cfg = _model_config(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        enable_deepep=enable_deepep,
        apply_rope_fusion=apply_rope_fusion,
    )

    if optimizer_type == "adam":
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
    else:
        # TODO: Add support for muon optimizer once mcore supports it
        raise ValueError(f"Invalid optimizer type: {optimizer_type}")

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
            eval_interval=eval_interval,
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
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
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
            split=split or "99990,8,2",
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
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=model_cfg.vocab_size),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
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
        cfg.dist.enable_megatron_core_experimental = True  # for mla rope fusion

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg


def _model_config(
    tensor_model_parallel_size: int = 2,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 8,
    sequence_parallel: bool = True,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    enable_deepep: bool = False,
    apply_rope_fusion: bool = False,
) -> MoonlightModelProvider16B:
    """
    Configure the Moonlight-16B model.

    Args:
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_dtype: Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size: Size of virtual pipeline parallelism.
        context_parallel_size: Degree of context parallelism.
        expert_model_parallel_size: Degree of expert model parallelism.
        sequence_parallel: Whether to use sequence parallelism.
        recompute_granularity: Recomputation granularity.
        recompute_modules: Modules to recompute.
        recompute_method: Recomputation method.
        recompute_num_layers: Number of layers to recompute.
        enable_deepep: Whether to use DeePEP.
        apply_rope_fusion: Whether to apply RoPE fusion.

    Returns:
        MoonlightModelProvider16B: Configuration for the Moonlight-16B model.
    """
    cfg = MoonlightModelProvider16B(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        expert_tensor_parallel_size=1,  # Do not use ETP
        # Recomputation
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
    )

    # Pipeline split for asymmetric stages as used in NeMo recipe
    cfg.account_for_embedding_in_pipeline_split = False
    cfg.account_for_loss_in_pipeline_split = False
    cfg.num_layers_in_first_pipeline_stage = None
    cfg.num_layers_in_last_pipeline_stage = None

    # Performance optimization knobs
    cfg.moe_permute_fusion = True
    if apply_rope_fusion:
        cfg.apply_rope_fusion = True

    # Pipeline parallelism configs. We infer PP layout from the provided PP and VP size
    map_pp_vp_to_layout = {
        (1, 1): None,
        (2, 1): [["embedding"] + ["decoder"] * 14, ["decoder"] * 13 + ["loss"]],
        (4, 1): [["embedding"] + ["decoder"] * 7] + [["decoder"] * 7] * 2 + [["decoder"] * 6 + ["loss"]],
        (8, 1): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 6 + [["decoder"] * 3 + ["loss"]],
        (2, 2): [["embedding"] + ["decoder"] * 7] + [["decoder"] * 7] * 2 + [["decoder"] * 6 + ["loss"]],
        (4, 2): [["embedding"] + ["decoder"] * 4] + [["decoder"] * 4] * 6 + [["decoder"] * 3 + ["loss"]],
    }
    pp_size = pipeline_model_parallel_size or 1
    vp_size = virtual_pipeline_model_parallel_size or 1
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for Moonlight-16B. Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
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
