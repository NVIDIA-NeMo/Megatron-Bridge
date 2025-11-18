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


class Qwen3MoeCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 MoE recipe helpers."""

    # Core identifiers
    hf_path: str
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
    expert_model_parallel_size: Optional[int]
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    enable_recompute: bool
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
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]


class Qwen3MoeFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 MoE finetuning recipe helper functions.

    This is separate from Qwen3MoeCommonKwargs to avoid confusion - finetuning
    uses SQuAD dataset by default, not the data path fields.
    """

    # Core identifiers
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


def qwen3_30b_a3b_pretrain_config(**user_kwargs: Unpack[Qwen3MoeCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3-30B-A3B MoE.

    See `_qwen3_moe_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3MoeCommonKwargs = {
        "hf_path": "Qwen/Qwen3-30B-A3B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 4,
        "sequence_parallel": True,
        "enable_recompute": True,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3MoeCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_moe_common(**combined_kwargs)


def qwen3_235b_a22b_pretrain_config(**user_kwargs: Unpack[Qwen3MoeCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3-235B-A22B MoE.

    See `_qwen3_moe_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3MoeCommonKwargs = {
        "hf_path": "Qwen/Qwen3-235B-A22B",
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 16,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 2,
        "expert_model_parallel_size": 8,
        "sequence_parallel": True,
        "micro_batch_size": 1,
        "account_for_embedding_in_pipeline_split": True,
        "account_for_loss_in_pipeline_split": True,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3MoeCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_moe_common(**combined_kwargs)


def _qwen3_moe_common(
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
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 2,
    pipeline_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: Optional[int] = 4,
    expert_tensor_parallel_size: int = 1,
    sequence_parallel: bool = True,
    use_megatron_fsdp: bool = False,
    enable_recompute: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
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
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Qwen3 MoE models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B").
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
        context_parallel_size (int): Degree of context parallelism to be passed to model_config.
        expert_model_parallel_size (Optional[int]): Degree of expert parallelism for MoE.
        expert_tensor_parallel_size (int): Expert tensor parallelism for MoE.
        sequence_parallel (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        enable_recompute (bool): Whether to enable recompute for memory optimization.
        account_for_embedding_in_pipeline_split (bool): Whether to account for embedding in pipeline split.
        account_for_loss_in_pipeline_split (bool): Whether to account for loss in pipeline split.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of iterations over which to decay the LR.
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

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = expert_tensor_parallel_size
    model_cfg.sequence_parallel = sequence_parallel

    if precision_config is None:
        precision_config = bf16_mixed()

    # MoE-specific pipeline split configurations
    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True

    # Add recompute settings for memory optimization (used by some MoE models)
    if enable_recompute:
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1
    model_cfg.seq_length = seq_length
    model_cfg.cross_entropy_fusion_impl = "te"

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
            average_in_collective=True,  # Not supported for Megatron FSDP for now, need to be set to False if using Megatron FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For Megatron FSDP only
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


def qwen3_30b_a3b_finetune_config(**user_kwargs: Unpack[Qwen3MoeFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3-30B-A3B MoE.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=4, PP=1, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
    - DoRA: TP=4, PP=1, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
    - Full SFT (peft=None): TP=4, PP=2, EP=4, LR=5e-6, SP=True

    Matches NeMo2 recipe at nemo/collections/llm/recipes/qwen3_30b_a3b.py
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _qwen3_moe_finetune_common(hf_path="Qwen/Qwen3-30B-A3B", **user_kwargs)

    # Model-specific parallelism settings (match NeMo pattern)
    if is_full_sft:
        config.model.tensor_model_parallel_size = 4
        config.model.expert_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 2
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        config.model.tensor_model_parallel_size = 4
        config.model.expert_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 1
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True

        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
            config.peft.target_modules = ["linear_qkv", "linear_proj"]
        else:
            config.peft = peft

    return config


def qwen3_235b_a22b_finetune_config(**user_kwargs: Unpack[Qwen3MoeFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3-235B-A22B MoE.

    Default configuration: 8 nodes (LoRA) or 16 nodes (Full SFT), 8 GPUs per node
    - LoRA (default): TP=4, PP=4, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
      Total: 64 GPUs (8 nodes)
    - DoRA: TP=4, PP=4, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
      Total: 64 GPUs (8 nodes)
    - Full SFT (peft=None): TP=4, PP=16, EP=4, LR=5e-6, SP=True
      Total: 64 GPUs (8 nodes)

    Matches NeMo2 recipe at nemo/collections/llm/recipes/qwen3_235b_a22b.py

    Note: Uses account_for_embedding_in_pipeline_split and account_for_loss_in_pipeline_split
    for proper layer distribution in pipeline parallelism.
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _qwen3_moe_finetune_common(hf_path="Qwen/Qwen3-235B-A22B", **user_kwargs)

    # Enable pipeline split accounting (required for 235B model)
    config.model.account_for_embedding_in_pipeline_split = True
    config.model.account_for_loss_in_pipeline_split = True

    # Model-specific parallelism settings (match NeMo pattern)
    if is_full_sft:
        config.model.tensor_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 16
        config.model.expert_model_parallel_size = 4
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        config.model.tensor_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 4
        config.model.expert_model_parallel_size = 4
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True

        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
            config.peft.target_modules = ["linear_qkv", "linear_proj"]
        else:
            config.peft = peft

    return config


def _qwen3_moe_finetune_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Finetuning-specific
    pretrained_checkpoint: Optional[str] = None,
    packed_sequence: bool = False,
    # Training hyperparameters
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
    """
    Create a finetuning configuration for Qwen3 MoE models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B").
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        pretrained_checkpoint (Optional[str]): Path to pretrained checkpoint to load.
        packed_sequence (bool): Whether to use packed sequences for training efficiency.
        train_iters (int): Total number of training iterations.
        global_batch_size (Optional[int]): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (Optional[int]): Sequence length for training data.
        eval_interval (int): Evaluation interval.
        save_interval (int): Checkpoint save interval.
        finetune_lr (Optional[float]): Learning rate for finetuning.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of iterations over which to decay the LR.
        wandb_project (Optional[str]): Weights & Biases project name.
        wandb_entity (Optional[str]): Weights & Biases entity name.
        wandb_exp_name (Optional[str]): Weights & Biases experiment name.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.

    Returns:
        ConfigContainer: Configuration for finetuning.
    """
    # Default sequence length for finetuning
    if seq_length is None:
        seq_length = 2048 if packed_sequence else 4096

    # Default global batch size
    if global_batch_size is None:
        global_batch_size = 32

    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    # Precision configuration
    if precision_config is None:
        precision_config = bf16_mixed()
    elif isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    # Sequence length
    model_cfg.seq_length = seq_length
    model_cfg.cross_entropy_fusion_impl = "te"

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
