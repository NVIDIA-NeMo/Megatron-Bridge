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
from typing import Optional, Union

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.nemotronh import Nemotron3NanoProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
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

class Nemotron3NanoCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron Next 3B v2 recipe helper functions."""

    # Core identifiers
    model_provider: Nemotron3NanoProvider
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[list[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[list[str]]
    valid_data_path: Optional[list[str]]
    test_data_path: Optional[list[str]]
    per_split_data_args_path: Optional[str]
    path_to_cache: Optional[str]
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_parallelism_dtype: Optional[torch.dtype]
    virtual_pipeline_parallelism: Optional[int]
    context_parallelism: int
    sequence_parallelism: bool
    expert_tensor_parallelism: int
    expert_model_parallelism: int
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    # MoE
    enable_deepep: bool


def nemotron_3_nano_pretrain_config(**user_kwargs: Unpack[Nemotron3NanoCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Nano.

    This recipe is designed for multi-node training.
    Default parallelism: TP=4, PP=1, SP=True, with DeepEP enabled.

    See `_nemotron_3_nano_common` for the full list of parameters.
    """
    recommended_kwargs: Nemotron3NanoCommonKwargs = {
        "model_provider": Nemotron3NanoProvider,
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "pipeline_parallelism_dtype": torch.bfloat16,
        "context_parallelism": 1,
        "sequence_parallelism": True,
        "enable_deepep": True,
        "expert_tensor_parallelism": 1,
        "expert_model_parallelism": 8,
        "precision_config": "bf16_mixed",
    }
    combined_kwargs: Nemotron3NanoCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron_3_nano_common(**combined_kwargs)


def _nemotron_3_nano_common(
    model_provider: type[Nemotron3NanoProvider],
    dir: Optional[str] = None,
    name: str = "default",
    # Dataset configuration
    data_paths: Optional[list[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[list[str]] = None,
    valid_data_path: Optional[list[str]] = None,
    test_data_path: Optional[list[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    path_to_cache: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    expert_tensor_parallelism: int = 1,
    expert_model_parallelism: int = 8,
    # Training hyperparameters
    train_iters: int = 39735,
    global_batch_size: int = 3072,
    micro_batch_size: int = 2,
    seq_length: int = 8192,
    eval_interval: int = 1000,
    save_interval: int = 200,
    # Optimizer
    lr: float = 1.6e-3,
    min_lr: float = 1.6e-5,
    lr_warmup_iters: int = 333,
    lr_decay_iters: Optional[int] = None,
    # Tokenizer
    tokenizer_model: Optional[str] = None,
    vocab_size: Optional[int] = 131072,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # MoE
    enable_deepep: bool = True,
    # W&B
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Nemotron Next 3B v2 model.

    Args:
        model_provider: The model provider class for the Nemotron Next 3B v2 variant.
        dir: Base directory for saving logs and checkpoints.
        name: Name of the pre-training run.
        data_paths: List of paths to dataset files. If None, mock data will be used.
        data_args_path: Path to file containing data arguments.
        train_data_path: List of training data paths.
        valid_data_path: List of validation data paths.
        test_data_path: List of test data paths.
        per_split_data_args_path: Path to JSON file with per-split data configuration.
        path_to_cache: Path to cache directory.
        mock: Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_parallelism_dtype: Data type for pipeline parallelism.
        virtual_pipeline_parallelism: Size of virtual pipeline parallelism.
        context_parallelism: Degree of context parallelism to be passed to model_config.
        sequence_parallelism: Whether to use sequence parallelism.
        expert_tensor_parallelism: Degree of expert tensor parallelism.
        expert_model_parallelism: Degree of expert model parallelism.
        train_iters: Total number of training iterations.
        global_batch_size: Global batch size for training.
        micro_batch_size: Micro batch size for training.
        seq_length: Sequence length for training data.
        lr: Learning rate.
        min_lr: Minimum learning rate for cosine decay.
        lr_warmup_iters: Number of warmup iterations for the learning rate.
        lr_decay_iters: Number of iterations for learning rate decay.
        precision_config: Precision configuration for the model.
        comm_overlap_config: Communication overlap configuration for the model.
        enable_deepep: Whether to enable DeepEP for MoE.

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

    # Configure the model (integrating the old model_config functionality)
    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_tensor_parallel_size=expert_tensor_parallelism,
        expert_model_parallel_size=expert_model_parallelism,
        apply_rope_fusion=False,
        async_tensor_model_parallel_allreduce=True,
        attention_backend="fused",
        gradient_accumulation_fusion=True,
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
    )

    if enable_deepep:
        model_cfg.moe_token_dispatcher_type = "flex"
        model_cfg.moe_shared_expert_overlap = False
        model_cfg.moe_flex_dispatcher_backend = "deepep"

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        lr_decay_style="cosine",
    )

    tokenizer_config=TokenizerConfig(
            tokenizer_type= "HuggingFaceTokenizer",
            tokenizer_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
            vocab_size=None
        )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
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
            split=split,
            path_to_cache=path_to_cache,
            # Dataloader config parameters
            data_sharding=True,
            dataloader_type="single",
            num_workers=1,
            skip_getting_attention_mask_from_dataset=True,
            mmap_bin_files=False,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_exp_name=wandb_exp_name,
        ),
        tokenizer=tokenizer_config,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
            ckpt_assume_constant_structure=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=True,
        )

    return cfg


class Nemotron3NanoFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron Next 3B v2 finetune recipe helpers."""

    # Core identifiers
    model_provider: Nemotron3NanoProvider
    dir: Optional[str]
    name: str
    # Model parallelism
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_parallelism_dtype: Optional[torch.dtype]
    virtual_pipeline_parallelism: Optional[int]
    context_parallelism: int
    sequence_parallelism: bool
    expert_tensor_parallelism: int
    expert_model_parallelism: int
    # Finetuning specifics
    pretrained_checkpoint: Optional[str]
    peft: Optional[Union[str, PEFT]]
    packed_sequence: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: Optional[int]
    micro_batch_size: int
    seq_length: int
    finetune_lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    # MoE
    enable_deepep: bool
    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]


def nemotron_3_nano_finetune_config(**user_kwargs: Unpack[Nemotron3NanoFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Nemotron 3 Nano.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1, EP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, EP=8, lower LR (5e-6)
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Nemotron3NanoFinetuneKwargs = {
        "model_provider": Nemotron3NanoProvider,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_parallelism_dtype": torch.bfloat16,
        "context_parallelism": 1,
        "sequence_parallelism": False,
        "expert_tensor_parallelism": 1,
        "expert_model_parallelism": 8,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "enable_deepep": True,
        "precision_config": "bf16_mixed",
    }
    combined_kwargs: Nemotron3NanoFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron_3_nano_finetune_common(**combined_kwargs)


def _nemotron_3_nano_finetune_common(
    model_provider: type[Nemotron3NanoProvider],
    dir: Optional[str] = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = True,
    expert_tensor_parallelism: int = 1,
    expert_model_parallelism: int = 1,
    # Finetuning-specific params
    pretrained_checkpoint: Optional[str] = None,
    peft: Optional[Union[str, PEFT]] = "lora",
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 500,
    save_interval: int = 200,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: Optional[int] = None,
    # Precision / overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # MoE
    enable_deepep: bool = True,
    # W&B
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Common finetuning configuration for Nemotron Next 3B v2 models."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Configure the model
    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        expert_tensor_parallel_size=expert_tensor_parallelism,
        expert_model_parallel_size=expert_model_parallelism,
        apply_rope_fusion=False,
        async_tensor_model_parallel_allreduce=True,
        attention_backend="fused",
        gradient_accumulation_fusion=True,
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
    )

    if enable_deepep:
        model_cfg.moe_token_dispatcher_type = "flex"
        model_cfg.moe_shared_expert_overlap = False
        model_cfg.moe_flex_dispatcher_backend = "deepep"

    # Optimizer and LR scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        max_lr=finetune_lr,
        min_lr=min_lr,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        lr_decay_style="cosine",
    )

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    peft_config = default_peft_config(peft, target_modules=mamba_target_modules)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            use_distributed_optimizer=True,
        ),
        dataset=default_squad_config(seq_length, packed_sequence),
        logger=logger_cfg,
        tokenizer = TokenizerConfig(tokenizer_type="HuggingFaceTokenizer",tokenizer_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"),
        checkpoint = CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
            ckpt_assume_constant_structure=True,
        ),
        rng=RNGConfig(seed=1234),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=True,
        )

    return cfg
