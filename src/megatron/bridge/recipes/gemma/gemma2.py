#!/usr/bin/env python3
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
from megatron.core.distributed import DistributedDataParallelConfig
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
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class Gemma2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Gemma2 recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[List[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[str]
    per_split_data_args_path: Optional[str]
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]


class Gemma2FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Gemma2 finetuning recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str

    # Finetuning-specific
    pretrained_checkpoint: Optional[str]
    peft: Union[str, PEFT, None]
    packed_sequence: bool

    # Training hyperparameters
    train_iters: int
    global_batch_size: Optional[int]
    micro_batch_size: int
    seq_length: Optional[int]
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: Optional[float]
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]

    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]

    # Precision
    precision_config: Optional[Union[MixedPrecisionConfig, str]]


# Pretrain Configs
def gemma2_2b_pretrain_config(**user_kwargs: Unpack[Gemma2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Gemma2 2B.

    Default parallelism: TP=2, PP=1
    """
    recommended_kwargs: Gemma2CommonKwargs = {
        "hf_path": "google/gemma-2-2b",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
    }
    combined_kwargs: Gemma2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _gemma2_common(**combined_kwargs)


def gemma2_9b_pretrain_config(**user_kwargs: Unpack[Gemma2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Gemma2 9B.

    Default parallelism: TP=8, PP=1
    """
    recommended_kwargs: Gemma2CommonKwargs = {
        "hf_path": "google/gemma-2-9b",
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
    }
    combined_kwargs: Gemma2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _gemma2_common(**combined_kwargs)


def gemma2_27b_pretrain_config(**user_kwargs: Unpack[Gemma2CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Gemma2 27B.

    Default parallelism: TP=8, PP=2
    """
    recommended_kwargs: Gemma2CommonKwargs = {
        "hf_path": "google/gemma-2-27b",
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
    }
    combined_kwargs: Gemma2CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _gemma2_common(**combined_kwargs)


def _gemma2_common(
    hf_path: str,
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
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 500,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 500,
    save_interval: int = 500,
    use_null_tokenizer: bool = False,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """Create a pre-training configuration for Gemma2 models."""
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    blend, blend_per_split, split = get_blend_fields_from_data_paths(
        data_paths, data_args_path, train_data_path, valid_data_path, test_data_path, per_split_data_args_path, mock
    )

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
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
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,
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
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


# Finetune Configs
def gemma2_2b_finetune_config(**user_kwargs: Unpack[Gemma2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Gemma2 2B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    return _gemma2_finetune_common(hf_path="google/gemma-2-2b", **user_kwargs)


def gemma2_9b_finetune_config(**user_kwargs: Unpack[Gemma2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Gemma2 9B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=4, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 4 if is_full_sft else 1

    return _gemma2_finetune_common(hf_path="google/gemma-2-9b", **user_kwargs)


def gemma2_27b_finetune_config(**user_kwargs: Unpack[Gemma2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Gemma2 27B.

    Default configuration: 2 nodes (SFT) or 1 node (LoRA), 8 GPUs per node
    - LoRA/DoRA: TP=4, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=2, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    if "tensor_model_parallel_size" not in user_kwargs:
        user_kwargs["tensor_model_parallel_size"] = 8 if is_full_sft else 4
    if "pipeline_model_parallel_size" not in user_kwargs:
        user_kwargs["pipeline_model_parallel_size"] = 2 if is_full_sft else 1

    return _gemma2_finetune_common(hf_path="google/gemma-2-27b", **user_kwargs)


def _gemma2_finetune_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Core model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: Optional[str] = None,
    peft: Union[str, PEFT, None] = "lora",
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 100,
    global_batch_size: Optional[int] = None,
    micro_batch_size: int = 1,
    seq_length: Optional[int] = None,
    eval_interval: int = 50,
    save_interval: int = 100,
    # Optimizer
    finetune_lr: Optional[float] = None,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 10,
    lr_decay_iters: Optional[int] = None,
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
    # Precision
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
) -> ConfigContainer:
    """Common finetuning configuration for all Gemma2 models."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select sequence length
    if seq_length is None:
        seq_length = 2048 if packed_sequence else 4096

    # Auto-select global_batch_size
    if global_batch_size is None:
        global_batch_size = 128

    # Auto-select learning rate
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    # Adjust vocab size for Gemma2 (model vocab < tokenizer vocab)
    if hasattr(model_cfg, "vocab_size") and hf_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        if len(tokenizer) > model_cfg.vocab_size:
            model_cfg.vocab_size = len(tokenizer)

    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    # Precision configuration
    if precision_config is None:
        precision_config = bf16_mixed()
    elif isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    # Optimizer and scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
    )

    # PEFT config
    peft_config = default_peft_config(peft) if not is_full_sft else None

    # For PEFT, disable distributed optimizer in optimizer config
    if not is_full_sft:
        opt_cfg.use_distributed_optimizer = False

    # Dataset config
    dataset_config = default_squad_config(seq_length, packed_sequence)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Tokenizer
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    # DDP config
    ddp_cfg = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=False if is_full_sft else True,
        overlap_grad_reduce=True if is_full_sft else False,
        overlap_param_gather=True if is_full_sft else False,
        average_in_collective=True if is_full_sft else False,
        use_distributed_optimizer=True if is_full_sft else False,
    )

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=10,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=ddp_cfg,
        dataset=dataset_config,
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
        mixed_precision=precision_config,
    )
