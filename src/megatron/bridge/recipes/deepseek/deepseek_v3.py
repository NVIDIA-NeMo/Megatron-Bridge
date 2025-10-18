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

import os
from typing import List, Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
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


class DeepSeekV3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by DeepSeek V3 recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str
    # Dataset configuration
    data_paths: List[str] | None
    data_args_path: str | None
    train_data_path: List[str] | None
    valid_data_path: List[str] | None
    test_data_path: List[str] | None
    per_split_data_args_path: str | None
    mock: bool
    # Model configuration
    tensor_parallelism: int
    pipeline_parallelism: int
    pipeline_parallelism_dtype: torch.dtype | None
    virtual_pipeline_parallelism: int | None
    context_parallelism: int
    expert_parallelism: int
    sequence_parallelism: bool
    use_megatron_fsdp: bool
    check_for_nan_in_grad: bool
    # Recompute configuration
    recompute_granularity: str | None
    recompute_modules: List[str] | None
    recompute_method: str | None
    recompute_num_layers: int | None
    # MTP support
    mtp_num_layers: int | None
    mtp_loss_scaling_factor: float | None
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    enable_deepep: bool
    apply_rope_fusion: bool
    layout: str | List[List[str]] | None


def deepseek_v3_pretrain_config(**user_kwargs: Unpack[DeepSeekV3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for DeepSeek-V3.

    See `_deepseek_v3_common` for the full list of parameters.
    """
    recommended_kwargs: DeepSeekV3CommonKwargs = {
        "hf_path": "deepseek-ai/DeepSeek-V3",
        "tensor_parallelism": 2,
        "pipeline_parallelism": 16,
        "expert_parallelism": 64,
        "pipeline_parallelism_dtype": torch.bfloat16,
        # Old recipe-compatible defaults passed via wrapper
        "recompute_granularity": "selective",
        "precision_config": MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
    }
    combined_kwargs: DeepSeekV3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _deepseek_v3_common(**combined_kwargs)


def deepseek_v3_pretrain_config_32nodes(**user_kwargs: Unpack[DeepSeekV3CommonKwargs]) -> ConfigContainer:
    """
    Create a pre-training configuration for DeepSeek-V3 (671B) model with minimal number of nodes (32).

    Returns:
        ConfigContainer: Configuration for pre-training.
    """
    recommended_kwargs: DeepSeekV3CommonKwargs = {
        "hf_path": "deepseek-ai/DeepSeek-V3",
        "tensor_parallelism": 2,
        "pipeline_parallelism": 8,
        "expert_parallelism": 32,
        # Maintain old recipe defaults via wrapper overrides
        "precision_config": MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        ),
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
    }
    combined_kwargs: DeepSeekV3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return deepseek_v3_pretrain_config(**combined_kwargs)


def _deepseek_v3_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    data_paths: List[str] | None = None,
    data_args_path: str | None = None,
    train_data_path: List[str] | None = None,
    valid_data_path: List[str] | None = None,
    test_data_path: List[str] | None = None,
    per_split_data_args_path: str | None = None,
    mock: bool = False,
    # Model configuration
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 16,
    pipeline_parallelism_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_parallelism: int | None = None,
    context_parallelism: int = 1,
    expert_parallelism: int = 64,
    sequence_parallelism: bool = True,
    use_megatron_fsdp: bool = False,
    check_for_nan_in_grad: bool = True,
    # Recompute configuration
    recompute_granularity: str | None = "selective",
    recompute_modules: List[str] | None = None,
    recompute_method: str | None = None,
    recompute_num_layers: int | None = None,
    # MTP support
    mtp_num_layers: int | None = 1,
    mtp_loss_scaling_factor: float | None = 0.1,
    # Training hyperparameters
    train_iters: int = 1_000_000,
    global_batch_size: int = 4096,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    eval_interval: int = 2000,
    save_interval: int = 2000,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = None,
    comm_overlap_config: CommOverlapConfig | None = None,
    enable_deepep: bool = False,
    apply_rope_fusion: bool = False,
    layout: str | List[List[str]] | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for DeepSeek-V3 models using a given HuggingFace path.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = pipeline_parallelism
    model_cfg.pipeline_dtype = pipeline_parallelism_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_parallelism
    model_cfg.context_parallel_size = context_parallelism
    model_cfg.expert_model_parallel_size = expert_parallelism
    model_cfg.sequence_parallel = sequence_parallelism
    model_cfg.seq_length = seq_length

    model_cfg.expert_tensor_parallel_size = 1
    # MTP configuration (allow None to disable by setting to 0)
    model_cfg.mtp_num_layers = 0 if mtp_num_layers is None else mtp_num_layers
    model_cfg.mtp_loss_scaling_factor = mtp_loss_scaling_factor
    model_cfg.init_method_std = 0.006
    model_cfg.rotary_base = 10000.0
    model_cfg.rotary_scaling_factor = 40
    model_cfg.rotary_base = float(model_cfg.rotary_base)
    model_cfg.rotary_scaling_factor = int(model_cfg.rotary_scaling_factor)

    model_cfg.recompute_granularity = recompute_granularity
    model_cfg.recompute_modules = recompute_modules
    model_cfg.recompute_method = recompute_method
    model_cfg.recompute_num_layers = recompute_num_layers

    mtp_layers = getattr(model_cfg, "mtp_num_layers", 1) or 0
    last_layer = ["mtp"] * mtp_layers + ["loss"]
    layout_map = {
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
    if layout is not None:
        # Allow overriding the automatically selected layout
        model_cfg.pipeline_model_parallel_layout = layout
    elif (pp_size, vp_size) in layout_map:
        model_cfg.pipeline_model_parallel_layout = layout_map[(pp_size, vp_size)]

    # Pipeline split for asymmetric stages are specified with map_pp_vp_to_layout below
    model_cfg.account_for_embedding_in_pipeline_split = False
    model_cfg.account_for_loss_in_pipeline_split = False
    model_cfg.num_layers_in_first_pipeline_stage = None
    model_cfg.num_layers_in_last_pipeline_stage = None

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True
    if enable_deepep:
        model_cfg.moe_token_dispatcher_type = "flex"
        model_cfg.moe_enable_deepep = True
        model_cfg.moe_shared_expert_overlap = False

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
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
            check_for_nan_in_grad=check_for_nan_in_grad,
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
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=hf_path if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
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
        cfg.dist.enable_megatron_core_experimental = True  # mla rope fusion is experimental

    # Ensure comm_overlap exists with old default tp_comm_overlap=False when not provided
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    return cfg
