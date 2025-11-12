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
    DistributedInitConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed


class Qwen3NextCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3-Next recipe helpers."""

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
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    expert_model_parallel_size: int | None
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    enable_recompute: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
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
    # Performance optimization knobs
    enable_deepep: bool
    disable_jit_fuser: bool


class Qwen3NextFinetuneKwargs(Qwen3NextCommonKwargs, total=False):
    """Typed options accepted by Qwen3-Next finetuning recipe helper functions."""

    # Core finetuning options
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Dataset configuration
    dataset_path: str | None

    # Training params
    finetune_lr: float

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None


def qwen3_next_80b_a3b_pretrain_config(**user_kwargs: Unpack[Qwen3NextCommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3-Next 80B-A3B.

    See `_qwen3_next_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3NextCommonKwargs = {
        "hf_path": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 4,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "sequence_parallel": False,
        "enable_recompute": True,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3NextCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_next_common(**combined_kwargs)


def _qwen3_next_common(
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
    path_to_cache: str | None = None,
    # Model configuration
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 2,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int | None = 4,
    expert_tensor_parallel_size: int = 1,
    sequence_parallel: bool = True,
    use_megatron_fsdp: bool = False,
    enable_recompute: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
    # MTP support
    mtp_num_layers: int | None = 1,
    mtp_loss_scaling_factor: float | None = 0.1,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 500,
    lr_decay_iters: int | None = None,
    eval_interval: int = 500,
    save_interval: int = 500,
    use_null_tokenizer: bool = False,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = None,
    comm_overlap_config: CommOverlapConfig | None = None,
    enable_deepep: bool = False,
    disable_jit_fuser: bool | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Qwen3-Next models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "Qwen/Qwen3-Next-80B-A3B-Instruct").
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
        context_parallel_size (int): Degree of context parallelism to be passed to model_config.
        expert_model_parallel_size (int | None): Degree of expert parallelism for MoE.
        expert_tensor_parallel_size (int): Expert tensor parallelism for MoE.
        sequence_parallel (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        enable_recompute (bool): Whether to enable recompute for memory optimization.
        account_for_embedding_in_pipeline_split (bool): Whether to account for embedding in pipeline split.
        account_for_loss_in_pipeline_split (bool): Whether to account for loss in pipeline split.
        mtp_num_layers (int | None): Number of layers for MTP.
        mtp_loss_scaling_factor (float | None): Loss scaling factor for MTP.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        lr (float): Learning rate.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (int | None): Number of iterations over which to decay the LR.
        precision_config (MixedPrecisionConfig | str | None): Precision configuration for the model.
        comm_overlap_config (CommOverlapConfig | None): Communication overlap configuration.
        enable_deepep (bool): Whether to enable DEEPEP for MoE.
        disable_jit_fuser (bool): Whether to disable the JIT fuser. Necessary for Qwen3-Next to work on Blackwell.

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
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = expert_tensor_parallel_size

    model_cfg.mtp_num_layers = 0 if mtp_num_layers is None else mtp_num_layers
    model_cfg.mtp_loss_scaling_factor = mtp_loss_scaling_factor

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True
    model_cfg.moe_grouped_gemm = True
    if enable_deepep:
        model_cfg.moe_token_dispatcher_type = "flex"
        # TODO: Remove moe_enable_deepep since it is deprecated
        model_cfg.moe_enable_deepep = True
        model_cfg.moe_shared_expert_overlap = False
        model_cfg.moe_flex_dispatcher_backend = "deepep"

    if precision_config is None:
        precision_config = bf16_mixed()
    if isinstance(precision_config, MixedPrecisionConfig):
        precision_config.grad_reduce_in_fp32 = False

    # MoE-specific pipeline split configurations
    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True

    # Add recompute settings for memory optimization (used by some MoE models)
    if enable_recompute:
        model_cfg.recompute_granularity = "selective"
        model_cfg.recompute_modules = ["layernorm", "moe", "moe_act"]
        model_cfg.recompute_method = None
        model_cfg.recompute_num_layers = None
    model_cfg.seq_length = seq_length

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
    )
    scheduler.no_weight_decay_cond_type = "qwen3_next"

    # If user does not specify, check if we are on Blackwell.
    if disable_jit_fuser is None:
        disable_jit_fuser = torch.cuda.get_device_properties(0).major == 10

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
        dist=DistributedInitConfig(disable_jit_fuser=disable_jit_fuser),
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
            path_to_cache=path_to_cache,
            mmap_bin_files=False,
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


def qwen3_next_80b_a3b_finetune_config(**user_kwargs: Unpack[Qwen3NextFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3-Next 80B-A3B.

    Default configuration: 8 nodes, 64 GPUs total
    - Full SFT: TP=1, PP=1, EP=8, LR=5e-6 (with recompute)
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")
    if not is_full_sft:
        raise ValueError("Only full SFT is supported for Qwen3-Next at the moment")

    # Check if user enables sequence packing since it is not supported for Qwen3-Next at the moment
    packed_sequence = user_kwargs.get("packed_sequence", False)
    if packed_sequence:
        raise ValueError("Sequence packing is not supported for Qwen3-Next at the moment")

    recommended_kwargs: Qwen3NextFinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "peft": peft_value,
        "finetune_lr": 5e-6,
        "min_lr": 5e-6,
        "enable_recompute": True,
    }
    combined_kwargs: Qwen3NextFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    config = _qwen3_next_finetune_common(**combined_kwargs)

    return config


def _qwen3_next_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    dataset_path: str | None = None,
    # Core model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int | None = 8,
    expert_tensor_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    enable_recompute: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = None,
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int | None = None,  # Auto-select based on packed_sequence if None
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 5e-6,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,  # Let config handle this
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    enable_deepep: bool = False,
    disable_jit_fuser: bool | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for Qwen3-Next model."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select global_batch_size based on packed_sequence
    if global_batch_size is None:
        global_batch_size = 8 if packed_sequence else 64

    if dataset_path is not None:
        dataset = FinetuningDatasetConfig(
            dataloader_type="single",
            dataset_root=dataset_path,
            seq_length=seq_length,
            seed=5678,
            memmap_workers=1,
            max_train_samples=None,
            packed_sequence_specs=None,  # TODO: add packed_sequence_specs if packed_sequence is True, currently not supported for Qwen3-Next
            dataset_kwargs=None,
            do_validation=True,
            do_test=True,
        )
    else:
        dataset = default_squad_config(seq_length, packed_sequence)

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = expert_tensor_parallel_size
    model_cfg.seq_length = seq_length

    # Add recompute settings for memory optimization
    if enable_recompute:
        model_cfg.recompute_granularity = "selective"
        model_cfg.recompute_modules = ["layernorm", "moe", "moe_act"]
        model_cfg.recompute_method = None
        model_cfg.recompute_num_layers = None

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True
    model_cfg.moe_grouped_gemm = True
    if enable_deepep:
        model_cfg.moe_token_dispatcher_type = "flex"
        # TODO: Remove moe_enable_deepep since it is deprecated
        model_cfg.moe_enable_deepep = True
        model_cfg.moe_shared_expert_overlap = False
        model_cfg.moe_flex_dispatcher_backend = "deepep"

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )
    scheduler_cfg.no_weight_decay_cond_type = "qwen3_next"

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

    # If user does not specify, check if we are on Blackwell.
    if disable_jit_fuser is None:
        disable_jit_fuser = torch.cuda.get_device_properties(0).major == 10

    return ConfigContainer(
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
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        dist=DistributedInitConfig(disable_jit_fuser=disable_jit_fuser),
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
        dataset=dataset,
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
