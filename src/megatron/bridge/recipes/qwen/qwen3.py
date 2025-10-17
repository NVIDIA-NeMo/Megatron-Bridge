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
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class Qwen3CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 recipe helper functions."""

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
    tensor_parallelism: int
    pipeline_parallelism: int
    pipeline_parallelism_dtype: torch.dtype | None
    virtual_pipeline_parallelism: int | None
    context_parallelism: int
    sequence_parallelism: bool
    use_megatron_fsdp: bool
    use_null_tokenizer: bool
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
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


class Qwen3FinetuneKwargs(Qwen3CommonKwargs, total=False):
    """Typed options accepted by Qwen3 finetuning recipe helper functions."""

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


def qwen3_600m_pretrain_config(**user_kwargs: Unpack[Qwen3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3 0.6B.

    See `_qwen3_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3CommonKwargs = {
        "hf_path": "Qwen/Qwen3-0.6B",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_common(**combined_kwargs)


def qwen3_1p7b_pretrain_config(**user_kwargs: Unpack[Qwen3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3 1.7B.

    See `_qwen3_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3CommonKwargs = {
        "hf_path": "Qwen/Qwen3-1.7B",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_common(**combined_kwargs)


def qwen3_4b_pretrain_config(**user_kwargs: Unpack[Qwen3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3 4B.

    See `_qwen3_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3CommonKwargs = {
        "hf_path": "Qwen/Qwen3-4B",
        "tensor_parallelism": 2,
        "pipeline_parallelism": 1,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_common(**combined_kwargs)


def qwen3_8b_pretrain_config(**user_kwargs: Unpack[Qwen3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3 8B.

    See `_qwen3_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3CommonKwargs = {
        "hf_path": "Qwen/Qwen3-8B",
        "tensor_parallelism": 4,
        "pipeline_parallelism": 1,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_common(**combined_kwargs)


def qwen3_14b_pretrain_config(**user_kwargs: Unpack[Qwen3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3 14B.

    See `_qwen3_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3CommonKwargs = {
        "hf_path": "Qwen/Qwen3-14B",
        "tensor_parallelism": 8,
        "pipeline_parallelism": 1,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_common(**combined_kwargs)


def qwen3_32b_pretrain_config(**user_kwargs: Unpack[Qwen3CommonKwargs]) -> ConfigContainer:
    """Return a pre-training config for Qwen3 32B.

    See `_qwen3_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen3CommonKwargs = {
        "hf_path": "Qwen/Qwen3-32B",
        "tensor_parallelism": 8,
        "pipeline_parallelism": 2,
        "pipeline_parallelism_dtype": torch.bfloat16,
        "enable_recompute": True,
    }
    # Combine defaults with user kwargs; user values take precedence.
    combined_kwargs: Qwen3CommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_common(**combined_kwargs)


def _qwen3_common(
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
    # Model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: torch.dtype | None = None,
    virtual_pipeline_parallelism: int | None = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
    use_null_tokenizer: bool = False,
    enable_recompute: bool = False,
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
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Qwen3 models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "Qwen/Qwen3-1.7B").
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the pre-training run.
        data_paths (Optional[List[str]]): List of paths to dataset files. If None, mock data will be used.
        data_args_path (Optional[str]): Path to file containing data arguments.
        train_data_path (Optional[List[str]]): List of training data paths.
        valid_data_path (Optional[List[str]]): List of validation data paths.
        test_data_path (Optional[List[str]]): List of test data paths.
        per_split_data_args_path (Optional[str]): Path to JSON file with per-split data configuration.
        mock (bool): Whether to use mock data. If True, ignores data_paths.
        tensor_parallelism (int): Degree of tensor model parallelism.
        pipeline_parallelism (int): Degree of pipeline model parallelism.
        pipeline_parallelism_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_parallelism (Optional[int]): Size of virtual pipeline parallelism.
        context_parallelism (int): Degree of context parallelism to be passed to model_config.
        sequence_parallelism (bool): Whether to use sequence parallelism.
        use_megatron_fsdp (bool): Whether to use Megatron FSDP.
        use_null_tokenizer (bool): Whether to use NullTokenizer instead of HuggingFaceTokenizer.
        enable_recompute (bool): Whether to enable recompute for memory optimization.
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
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = pipeline_parallelism
    model_cfg.pipeline_dtype = pipeline_parallelism_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_parallelism
    model_cfg.context_parallel_size = context_parallelism
    model_cfg.sequence_parallel = sequence_parallelism
    model_cfg.seq_length = seq_length

    # Add recompute settings for memory optimization (used by larger models like 32B)
    if enable_recompute:
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Config Container
    cfg_container = ConfigContainer(
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
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,  # Not supported for custom FSDP for now, need to be set to False if using FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For custom FSDP only
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

    return cfg_container


def qwen3_600m_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 600M.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-0.6B",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_1p7b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 1.7B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-1.7B",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_4b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 4B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-4B",
        "tensor_parallelism": 2 if is_full_sft else 1,  # Match NeMo2: higher TP for SFT, TP=1 for LoRA
        "pipeline_parallelism": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_8b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 8B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=4, PP=1, LR=5e-6
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-8B",
        "tensor_parallelism": 4 if is_full_sft else 1,  # Match NeMo2: TP=4 for SFT, TP=1 for LoRA
        "pipeline_parallelism": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_14b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 14B.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=1, LR=5e-6
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-14B",
        "tensor_parallelism": 8 if is_full_sft else 1,  # Match NeMo2: TP=8 for SFT, TP=1 for LoRA
        "pipeline_parallelism": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen3_finetune_common(**combined_kwargs)


def qwen3_32b_finetune_config(**user_kwargs: Unpack[Qwen3FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3 32B.

    Default configuration: 2 nodes, 16 GPUs total
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4 (with recompute)
    - Full SFT: TP=8, PP=2, LR=5e-6 (with recompute)
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen3FinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-32B",
        "tensor_parallelism": 8 if is_full_sft else 1,  # Match NeMo2: TP=8 for SFT, TP=1 for LoRA
        "pipeline_parallelism": 2 if is_full_sft else 1,  # PP=2 for SFT, PP=1 for LoRA
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,  # Match NeMo2: lower LR for SFT
    }
    combined_kwargs: Qwen3FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    config = _qwen3_finetune_common(**combined_kwargs)

    # Enable recompute for 32B model
    config.model.recompute_granularity = "full"
    config.model.recompute_method = "uniform"
    config.model.recompute_num_layers = 1

    return config


def _qwen3_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Core model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: torch.dtype | None = None,
    virtual_pipeline_parallelism: int | None = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int | None = None,  # Auto-select based on packed_sequence if None
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
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
) -> ConfigContainer:
    """Common finetuning configuration for all Qwen3 models."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select global_batch_size based on packed_sequence
    if global_batch_size is None:
        global_batch_size = 8 if packed_sequence else 128

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = pipeline_parallelism
    model_cfg.pipeline_dtype = pipeline_parallelism_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_parallelism
    model_cfg.context_parallel_size = context_parallelism
    model_cfg.sequence_parallel = sequence_parallelism
    model_cfg.seq_length = seq_length

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,  # None is fine
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
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
        ddp=DistributedDataParallelConfig(check_for_nan_in_grad=True),
        dataset=default_squad_config(seq_length, packed_sequence),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
