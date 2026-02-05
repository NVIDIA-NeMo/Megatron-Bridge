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

from megatron.bridge.models import NemotronHModelProvider
from megatron.bridge.models.nemotronh import (
    NemotronHModelProvider4B,
    NemotronHModelProvider8B,
    NemotronHModelProvider47B,
    NemotronHModelProvider56B,
)
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


class NemotronHCommonKwargs(TypedDict, total=False):
    """Typed options accepted by NemotronH recipe helper functions."""

    # Core identifiers
    model_provider: type[NemotronHModelProvider]
    tokenizer_model: str | None
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
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    # CommOverlap setting
    enable_default_comm_overlap: bool


class NemotronHFinetuneKwargs(NemotronHCommonKwargs, total=False):
    """Typed options accepted by NemotronH finetuning recipe helper functions."""

    # Core finetuning options
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Training params
    finetune_lr: float

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None


def nemotronh_4b_pretrain_config(**user_kwargs: Unpack[NemotronHCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for NemotronH 4B.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=1, PP=1, SP=False.

    See `_nemotronh_common` for the full list of parameters.
    """
    recommended_kwargs: NemotronHCommonKwargs = {
        "model_provider": NemotronHModelProvider4B,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": False,
        "precision_config": "bf16_mixed",
        "enable_default_comm_overlap": True,
    }
    combined_kwargs: NemotronHCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_common(tokenizer_model="nvidia/Nemotron-H-4B-Base-8K", **combined_kwargs)


def nemotronh_8b_pretrain_config(**user_kwargs: Unpack[NemotronHCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for NemotronH 8B.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=2, PP=1, SP=True.

    See `_nemotronh_common` for the full list of parameters.
    """
    recommended_kwargs: NemotronHCommonKwargs = {
        "model_provider": NemotronHModelProvider8B,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": True,
        "precision_config": "bf16_mixed",
        "enable_default_comm_overlap": True,
    }
    combined_kwargs: NemotronHCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_common(tokenizer_model="nvidia/Nemotron-H-8B-Base-8K", **combined_kwargs)


def nemotronh_47b_pretrain_config(**user_kwargs: Unpack[NemotronHCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for NemotronH 47B.

    This recipe is designed for single-node training (1 node with 8 GPUs).
    Default parallelism: TP=8, PP=1, SP=True.

    Note: Uses FP8 precision by default. Communication overlap is disabled by default
    due to known issues with FP8 current scaling.

    See `_nemotronh_common` for the full list of parameters.
    """
    recommended_kwargs: NemotronHCommonKwargs = {
        "model_provider": NemotronHModelProvider47B,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": True,
        "precision_config": "nemotron_h_bf16_with_fp8_current_scaling_mixed",
        "enable_default_comm_overlap": True,
    }
    combined_kwargs: NemotronHCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_common(tokenizer_model="nvidia/Nemotron-H-47B-Base-8K", **combined_kwargs)


def nemotronh_56b_pretrain_config(**user_kwargs: Unpack[NemotronHCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for NemotronH 56B.

    This recipe is designed for single-node training (1 node with 8 GPUs).
    Default parallelism: TP=8, PP=1, SP=True.

    Note: Uses FP8 precision by default. Communication overlap is disabled by default
    due to known issues with FP8 current scaling.

    See `_nemotronh_common` for the full list of parameters.
    """
    recommended_kwargs: NemotronHCommonKwargs = {
        "model_provider": NemotronHModelProvider56B,
        "tensor_model_parallel_size": 8,
        "pipeline_model_parallel_size": 1,
        "sequence_parallel": True,
        "precision_config": "nemotron_h_bf16_with_fp8_current_scaling_mixed",
        "enable_default_comm_overlap": True,
    }
    combined_kwargs: NemotronHCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_common(tokenizer_model="nvidia/Nemotron-H-8B-Base-8K", **combined_kwargs)


def _nemotronh_common(
    model_provider: type[NemotronHModelProvider],
    tokenizer_model: str | None = None,
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
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 1_168_251,
    global_batch_size: int = 768,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # CommOverlap setting
    enable_default_comm_overlap: bool = True,
) -> ConfigContainer:
    """
    Create a pre-training configuration for NemotronH and Nemotron Nano v2 models.

    Args:
        model_provider: The model provider class for the specific NemotronH or Nemotron Nano v2 variant.
        tokenizer_model: HuggingFace tokenizer model name (only used when use_null_tokenizer=False).
        dir: Base directory for saving logs and checkpoints.
        name: Name of the pre-training run.
        data_paths: List of paths to dataset files. If None, mock data will be used.
        data_args_path: Path to file containing data arguments.
        train_data_path: List of training data paths.
        valid_data_path: List of validation data paths.
        test_data_path: List of test data paths.
        per_split_data_args_path: Path to JSON file with per-split data configuration.
        mock: Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_dtype: Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size: Size of virtual pipeline parallelism.
        context_parallel_size: Degree of context parallelism to be passed to model_config.
        sequence_parallel: Whether to use sequence parallelism.
        train_iters: Total number of training iterations.
        global_batch_size: Global batch size for training.
        micro_batch_size: Micro batch size for training.
        seq_length: Sequence length for training data.
        lr: Learning rate.
        min_lr: Minimum learning rate for cosine decay.
        lr_warmup_iters: Number of warmup iterations for the learning rate.
        lr_decay_iters: Number of iterations for learning rate decay.
        use_null_tokenizer: Whether to use NullTokenizer instead of HuggingFaceTokenizer.
        precision_config: Precision configuration for the model.
        comm_overlap_config: Communication overlap configuration for the model.
        enable_default_comm_overlap: Whether to enable default comm overlap config if none is provided.

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

    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallel,
    )

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=10,
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
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=seq_length,
            num_dataset_builder_threads=1,
            blend=blend,
            blend_per_split=blend_per_split,
            split=split,
            # Dataloader config parameters
            data_sharding=True,
            dataloader_type="single",
            num_workers=8,
            skip_getting_attention_mask_from_dataset=True,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
        ),
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=tokenizer_model if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
        checkpoint=CheckpointConfig(
            save_interval=10,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if cfg.comm_overlap is None and enable_default_comm_overlap:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_bootstrap_backend="nccl",
            tp_comm_overlap=True,
        )

    return cfg


def nemotronh_4b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 4B.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider4B,
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "sequence_parallelism": False,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "precision_config": "bf16_mixed",
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-4B-Base-8K", **combined_kwargs)


def nemotronh_8b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 8B.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider8B,
        "tensor_parallelism": 2 if is_full_sft else 1,
        "pipeline_parallelism": 1,
        "sequence_parallelism": is_full_sft,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "precision_config": "bf16_mixed",
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-8B-Base-8K", **combined_kwargs)


def nemotronh_47b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 47B.

    Default configuration:
    - LoRA/DoRA: TP=4, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=1, LR=5e-6

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider47B,
        "tensor_parallelism": 8 if is_full_sft else 4,
        "pipeline_parallelism": 2 if is_full_sft else 1,
        "sequence_parallelism": is_full_sft,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-47B-Base-8K", **combined_kwargs)


def nemotronh_56b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 56B.

    Default configuration:
    - LoRA/DoRA: TP=4, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=1, LR=5e-6

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider56B,
        "tensor_parallelism": 8 if is_full_sft else 4,
        "pipeline_parallelism": 1,
        "sequence_parallelism": is_full_sft,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-56B-Base-8K", **combined_kwargs)


def _nemotronh_finetune_common(
    model_provider: type[NemotronHModelProvider],
    tokenizer_model: str | None = None,
    dir: str | None = None,
    name: str = "default",
    # Model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_parallelism: int | None = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    eval_interval: int = 50,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 1e-5,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # Tokenizer kwargs (for model-specific tokenizer settings)
    hf_tokenizer_kwargs: dict | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for NemotronH and Nemotron Nano v2 models.

    Args:
        model_provider: The model provider class for the specific NemotronH or Nemotron Nano v2 variant.
        tokenizer_model: HuggingFace tokenizer model name.
        dir: Base directory for saving logs and checkpoints.
        name: Name of the finetuning run.
        tensor_parallelism: Degree of tensor model parallelism.
        pipeline_parallelism: Degree of pipeline model parallelism. Default: 1.
        pipeline_parallelism_dtype: Data type for pipeline parallelism. Default: torch.bfloat16.
        virtual_pipeline_parallelism: Size of virtual pipeline parallelism.
        context_parallelism: Degree of context parallelism. Default: 1.
        sequence_parallelism: Whether to use sequence parallelism.
        pretrained_checkpoint: Path to pretrained checkpoint to load from.
        peft: PEFT configuration (e.g., "lora", "dora") or PEFT object. None for full SFT. Default: "lora".
        packed_sequence: Whether to use packed sequences. Default: False.
        train_iters: Total number of training iterations. Default: 1000.
        global_batch_size: Global batch size. Default: 128.
        micro_batch_size: Micro batch size. Default: 1.
        seq_length: Sequence length. Default: 8192.
        eval_interval: Evaluation interval in iterations. Default: 50.
        save_interval: Checkpoint save interval in iterations. Default: 50.
        finetune_lr: Learning rate for finetuning. Default: 1e-4.
        min_lr: Minimum learning rate. Default: 1e-5.
        lr_warmup_iters: Number of warmup iterations. Default: 50.
        lr_decay_iters: Number of LR decay iterations.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity name.
        wandb_exp_name: Weights & Biases experiment name.
        precision_config: Precision configuration.
        comm_overlap_config: Communication overlap configuration.
        hf_tokenizer_kwargs: Additional kwargs for HuggingFace tokenizer (e.g., {"eos_token": "<SPECIAL_12>"}).

    Returns:
        ConfigContainer: Configuration for finetuning.

    Note:
        - 4B model: TP=1, SP=False, BF16 mixed precision
        - 8B model: TP=2 (full SFT) or TP=1 (LoRA), SP=True (full SFT), BF16 mixed precision
        - 9B Nano v2: TP=2 (full SFT) or TP=1 (LoRA), SP=True (full SFT), BF16 mixed precision
        - 12B Nano v2: TP=4 (full SFT) or TP=1 (LoRA), SP=True (full SFT), BF16 mixed precision
        - 47B model: TP=8 (full SFT) or TP=4 (LoRA), SP=True (full SFT), FP8 precision
        - 56B model: TP=8 (full SFT) or TP=4 (LoRA), SP=True (full SFT), FP8 precision
        - Uses SQuAD dataset format for finetuning
    """
    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Create model config
    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        seq_length=seq_length,
    )

    # Optimizer and scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
    )

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    peft_config = default_peft_config(peft, target_modules=mamba_target_modules)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Tokenizer config
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )

    pad_seq_to_mult = (
        model_cfg.context_parallel_size * 2 if packed_sequence and model_cfg.context_parallel_size > 1 else 1
    )

    cfg = ConfigContainer(
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
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=default_squad_config(seq_length, packed_sequence, pad_seq_to_mult),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


__all__ = [
    # Pretrain configs
    "nemotronh_4b_pretrain_config",
    "nemotronh_8b_pretrain_config",
    "nemotronh_47b_pretrain_config",
    "nemotronh_56b_pretrain_config",
    # Finetune configs
    "nemotronh_4b_finetune_config",
    "nemotronh_8b_finetune_config",
    "nemotronh_47b_finetune_config",
    "nemotronh_56b_finetune_config",
]
