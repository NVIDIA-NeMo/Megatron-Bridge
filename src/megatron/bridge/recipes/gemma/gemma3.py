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
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider1B
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
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class Gemma3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Gemma3 family recipe helpers."""

    # Core identifiers
    provider_class: type
    hf_path: str | None
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
    use_megatron_fsdp: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
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


class Gemma3FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Gemma3 finetuning recipe helper functions.

    This is separate from Gemma3CommonKwargs to avoid confusion - finetuning
    uses SQuAD dataset by default, not the data path fields.
    """

    # Core identifiers
    dir: str | None
    name: str

    # Finetuning-specific
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Training hyperparameters
    train_iters: int
    global_batch_size: int | None
    micro_batch_size: int
    seq_length: int | None
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: float | None
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None

    # Precision
    precision_config: MixedPrecisionConfig | str | None


# Sequence length constants
SEQUENCE_LENGTH_32K: int = 32768
SEQUENCE_LENGTH_128K: int = 131072


# Gemma3 models
def gemma3_1b_pretrain_config(**user_kwargs: Unpack[Gemma3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Gemma3 1B.

    See `_gemma3_common` for the full list of parameters.
    """
    recommended_kwargs: Gemma3CommonKwargs = {
        "provider_class": Gemma3ModelProvider1B,
        "hf_path": "google/gemma-3-1b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "sequence_parallel": False,
        "seq_length": SEQUENCE_LENGTH_32K,
    }
    combined_kwargs: Gemma3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _gemma3_common(**combined_kwargs)


def _gemma3_common(
    provider_class: type,
    hf_path: str | None = None,
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
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    # Training hyperparameters
    train_iters: int = 1168251,
    global_batch_size: int = 512,
    micro_batch_size: int = 1,
    seq_length: int = 131072,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 2000,
    lr_decay_iters: int | None = None,
    eval_interval: int = 2000,
    save_interval: int = 500,
    use_null_tokenizer: bool = True,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Gemma3 family models.

    Args:
        provider_class (type): Gemma3 model provider class (e.g., Gemma3ModelProvider1B).
        hf_path (str | None): HuggingFace model path (e.g., "google/gemma-3-1b").
        dir (str | None): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (list[str] | None): List of paths to dataset files. If None, mock data will be used.
        data_args_path (str | None): Path to file containing data arguments.
        train_data_path (list[str] | None): List of training data paths.
        valid_data_path (list[str] | None): List of validation data paths.
        test_data_path (list[str] | None): List of test data paths.
        per_split_data_args_path (str | None): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_model_parallel_size (int): Degree of tensor model parallelism.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism.
        pipeline_dtype (torch.dtype | None): Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size (int | None): Size of virtual pipeline parallelism.
        context_parallel_size (int): Degree of context parallelism.
        sequence_parallel (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        account_for_embedding_in_pipeline_split (bool): Whether to account for embedding in pipeline split.
        account_for_loss_in_pipeline_split (bool): Whether to account for loss in pipeline split.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (int | None): Number of iterations over which to decay the LR.
        eval_interval (int): Evaluation interval.
        save_interval (int): Checkpoint save interval.
        use_null_tokenizer (bool): Whether to use null tokenizer for synthetic data.
        precision_config (MixedPrecisionConfig | str | None): Precision configuration for the model.
        comm_overlap_config (CommOverlapConfig | None): Communication overlap configuration.

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

    # Instantiate the model provider
    model_cfg = provider_class()
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    # Large model specific pipeline split configurations
    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
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
            # Dataloader config parameters
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


# ============================================================================
# Finetuning Configurations
# ============================================================================


def gemma3_1b_finetune_config(**user_kwargs: Unpack[Gemma3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Gemma3 1B.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - DoRA: TP=1, PP=1, LR=1e-4, dim=8, alpha=16
    - Full SFT (peft=None): TP=1, PP=1, LR=5e-6

    Matches NeMo2 recipe at nemo/collections/llm/recipes/gemma3_1b.py
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _gemma3_finetune_common(hf_path="google/gemma-3-1b-pt", **user_kwargs)

    # Model-specific parallelism settings
    config.model.tensor_model_parallel_size = 1
    config.model.pipeline_model_parallel_size = 1
    config.model.context_parallel_size = 1
    config.model.sequence_parallel = False

    # PEFT or Full SFT specific settings
    if is_full_sft:
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
        else:
            config.peft = peft
        config.model.cross_entropy_loss_fusion = False
        config.optimizer.use_distributed_optimizer = False

    return config


def _gemma3_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Finetuning-specific
    pretrained_checkpoint: str | None = None,
    packed_sequence: bool = False,
    # Training hyperparameters
    train_iters: int = 100,
    global_batch_size: int | None = None,
    micro_batch_size: int = 1,
    seq_length: int | None = None,
    eval_interval: int = 50,
    save_interval: int = 100,
    # Optimizer
    finetune_lr: float | None = None,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 10,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = None,
) -> ConfigContainer:
    """
    Create a finetuning configuration for Gemma3 models.

    Args:
        hf_path (str): HuggingFace model path (e.g., "google/gemma-3-1b-pt").
        dir (str | None): Base directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        pretrained_checkpoint (str | None): Path to pretrained checkpoint to load.
        packed_sequence (bool): Whether to use packed sequences for training efficiency.
        train_iters (int): Total number of training iterations.
        global_batch_size (int | None): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int | None): Sequence length for training data.
        eval_interval (int): Evaluation interval.
        save_interval (int): Checkpoint save interval.
        finetune_lr (float | None): Learning rate for finetuning.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (int | None): Number of iterations over which to decay the LR.
        wandb_project (str | None): Weights & Biases project name.
        wandb_entity (str | None): Weights & Biases entity name.
        wandb_exp_name (str | None): Weights & Biases experiment name.
        precision_config (MixedPrecisionConfig | str | None): Precision configuration for the model.

    Returns:
        ConfigContainer: Configuration for finetuning.
    """
    # Default sequence length for finetuning
    if seq_length is None:
        seq_length = 4096 if packed_sequence else 2048

    # Default global batch size
    if global_batch_size is None:
        global_batch_size = 32

    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Create model config using AutoBridge (like Qwen3)
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    # Adjust vocab size for Gemma3 (model vocab < tokenizer vocab)
    # Gemma3 uses a smaller vocab size than the tokenizer, so we need to pad
    if hasattr(model_cfg, "vocab_size") and hf_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
        if len(tokenizer) > model_cfg.vocab_size:
            model_cfg.vocab_size = len(tokenizer)

    model_cfg.seq_length = seq_length

    # Precision configuration
    if precision_config is None:
        precision_config = bf16_mixed()
    elif isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    # Optimizer and scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=finetune_lr if finetune_lr is not None else 1e-4,
        min_lr=min_lr,
    )

    # Dataset configuration (SQuAD by default)
    dataset_config = default_squad_config(seq_length=seq_length, packed_sequence=packed_sequence)

    # W&B logger configuration
    logger_config = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
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
            eval_iters=10,
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
        ),
        dataset=dataset_config,
        logger=logger_config,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_path,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),  # Different seed for finetuning
        mixed_precision=precision_config,
    )

    return cfg
