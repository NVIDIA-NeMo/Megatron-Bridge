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

import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetProvider,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class GLM45CommonKwargs(TypedDict, total=False):
    """Typed options accepted by GLM 4.5 recipe helpers."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str
    # Dataset configuration
    data_paths: list[str] | None
    data_args_path: str | None
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    per_split_data_args_path: str | None
    mock: bool
    # Provide dataset directly
    dataset: GPTDatasetConfig | FinetuningDatasetConfig | DatasetProvider | None
    # Model configuration
    num_layers: int  # for ci testing
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    expert_model_parallel_size: int | None
    sequence_parallel: bool
    use_megatron_fsdp: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
    cp_comm_type: str | None
    # Recompute configuration
    recompute_granularity: str | None
    recompute_modules: list[str] | None
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
    # Checkpointing
    pretrained_checkpoint: str | None


class GLM45FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by GLM 4.5 finetune recipe helpers."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str
    # Model parallelism
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    expert_model_parallel_size: int | None
    sequence_parallel: bool
    use_megatron_fsdp: bool
    # Finetuning specifics
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int | None
    micro_batch_size: int
    seq_length: int
    finetune_lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None


def glm45_355b_pretrain_config(**user_kwargs: Unpack[GLM45CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for GLM 4.5 355B-A32B variant."""
    recommended: GLM45CommonKwargs = {
        "hf_path": "zai-org/GLM-4.5",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 8,
        "expert_model_parallel_size": 16,
        "sequence_parallel": True,
        "use_null_tokenizer": True,
        "recompute_granularity": "selective",
    }
    kwargs: GLM45CommonKwargs = {**recommended, **user_kwargs}
    return _glm45_common(**kwargs)


def glm45_air_106b_pretrain_config(**user_kwargs: Unpack[GLM45CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for GLM 4.5 Air 106B-A12B variant."""
    recommended: GLM45CommonKwargs = {
        "hf_path": "zai-org/GLM-4.5-Air",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 4,
        "expert_model_parallel_size": 8,
        "sequence_parallel": True,
        "use_null_tokenizer": True,
        "recompute_granularity": "selective",
    }
    kwargs: GLM45CommonKwargs = {**recommended, **user_kwargs}
    return _glm45_common(**kwargs)


def _glm45_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    data_paths: list[str] | None = None,
    data_args_path: str | None = None,
    train_data_path: list[str] | None = None,
    valid_data_path: list[str] | None = None,
    test_data_path: list[str] | None = None,
    per_split_data_args_path: str | None = None,
    mock: bool = False,
    # Dataset override option
    dataset: GPTDatasetConfig | FinetuningDatasetConfig | DatasetProvider | None = None,
    # Model configuration
    num_layers: int = None,  # for ci testing
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    cp_comm_type: str | None = None,
    # Recompute configuration
    recompute_granularity: str | None = None,
    recompute_modules: list[str] | None = None,
    recompute_method: str | None = None,
    recompute_num_layers: int | None = None,
    # MTP support (GLM models use MTP)
    mtp_num_layers: int | None = 1,
    mtp_loss_scaling_factor: float | None = 0.3,
    # Training hyperparameters
    train_iters: int = 1000000,
    global_batch_size: int = 2048,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 1e-4,
    min_lr: float = 1e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    eval_interval: int = 2000,
    save_interval: int = 500,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # Checkpointing
    pretrained_checkpoint: str | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for GLM 4.5 family models using a given HuggingFace path.
    Mirrors the structure used in gpt_oss recipes for consistency.
    """

    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    if num_layers is not None:
        model_cfg.num_layers = num_layers
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = 1
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True
    model_cfg.cp_comm_type = cp_comm_type

    # Recompute configuration
    model_cfg.recompute_granularity = recompute_granularity
    model_cfg.recompute_modules = recompute_modules
    model_cfg.recompute_method = recompute_method
    model_cfg.recompute_num_layers = recompute_num_layers

    # MTP configuration (GLM models support MTP)
    model_cfg.mtp_num_layers = 0 if mtp_num_layers is None else mtp_num_layers
    model_cfg.mtp_loss_scaling_factor = mtp_loss_scaling_factor

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
    )

    # Build dataset config if not supplied directly
    if dataset is None:
        blend, blend_per_split, split = get_blend_fields_from_data_paths(
            data_paths,
            data_args_path,
            train_data_path,
            valid_data_path,
            test_data_path,
            per_split_data_args_path,
            mock,
        )
        dataset_cfg = GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            data_sharding=True,
            dataloader_type="single",
            skip_getting_attention_mask_from_dataset=True,
        )
    else:
        dataset_cfg = dataset

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,
        ),
        dataset=dataset_cfg,
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
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


def glm45_355b_finetune_config(**user_kwargs: Unpack[GLM45FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for GLM 4.5 355B-A32B variant.

    Default configuration:
    - LoRA/DoRA: TP=2, PP=4, EP=4 (32 GPUs), LR=1e-4
    - Full SFT: TP=2, PP=8, EP=16 (256 GPUs, same as pretrain), LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended: GLM45FinetuneKwargs = {
        "hf_path": "zai-org/GLM-4.5",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 8 if is_full_sft else 4,
        "expert_model_parallel_size": 16 if is_full_sft else 4,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    kwargs: GLM45FinetuneKwargs = {**recommended, **user_kwargs}
    return _glm45_finetune_common(**kwargs)


def glm45_air_106b_finetune_config(**user_kwargs: Unpack[GLM45FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for GLM 4.5 Air 106B-A12B variant.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=2, EP=4 (8 GPUs, 1 node), LR=1e-4
    - Full SFT: TP=1, PP=4, EP=8 (32 GPUs, same as pretrain), LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended: GLM45FinetuneKwargs = {
        "hf_path": "zai-org/GLM-4.5-Air",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 4 if is_full_sft else 2,
        "expert_model_parallel_size": 8 if is_full_sft else 4,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    kwargs: GLM45FinetuneKwargs = {**recommended, **user_kwargs}
    return _glm45_finetune_common(**kwargs)


def _glm45_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 50,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # Precision / overlap
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # W&B
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for GLM 4.5 models using a given HuggingFace path."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    assert not packed_sequence, "Packed sequence is not supported for GLM 4.5 finetuning"

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    # Optimizer and LR scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
    )

    # PEFT config
    peft_config = default_peft_config(peft)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Always use HF tokenizer for finetuning
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    pad_seq_to_mult = context_parallel_size * 2 if packed_sequence and context_parallel_size > 1 else 1

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(check_for_nan_in_grad=True, use_megatron_fsdp=use_megatron_fsdp),
        dataset=default_squad_config(seq_length, packed_sequence, pad_seq_to_mult),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
