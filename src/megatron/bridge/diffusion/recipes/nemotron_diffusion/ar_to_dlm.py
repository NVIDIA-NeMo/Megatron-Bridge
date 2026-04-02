# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Optional

import torch
from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge import AutoBridge
from megatron.bridge.diffusion.models.nemotron_diffusion.nemotron_diffusion_provider import (
    NemotronDiffusionModelProvider,
)
from megatron.bridge.recipes.utils.dataset_utils import get_blend_fields_from_data_paths
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    GPTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def nemotron_diffusion3_pretrain_config(**user_kwargs) -> ConfigContainer:
    """Return a pre-training config for NemotronDiffusion.

    See `_nemotron_diffusion3_common` for the full list of parameters.
    """

    See `_dqwen3_common` for the full list of parameters.
    """
    # Combine defaults with user kwargs; user values take precedence.
    return _nemotron_diffusion3_common(**user_kwargs)


def _nemotron_diffusion3_common(
    model_provider: NemotronDiffusionModelProvider | None = None,
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
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: torch.dtype | None = None,
    virtual_pipeline_parallelism: int | None = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
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
    pretrained_checkpoint: str | None = None,
    # Precision recipe
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
) -> ConfigContainer:
    """
    Create a pre-training configuration for NemotronDiffusion models using a given model provider.

    Args:
        hf_path (Optional[str]): HuggingFace model path (e.g., "Qwen/Qwen3-1.7B").
        model_provider (NemotronDiffusionModelProvider): Model provider for the model.
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
    if hf_path is not None:
        bridge = AutoBridge.from_hf_pretrained(hf_path)
        model_cfg = bridge.to_megatron_provider(
            load_weights=bool(pretrained_checkpoint),
            hf_path=pretrained_checkpoint,
        )
    else:
        model_cfg = model_provider()

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

    model_cfg.cross_entropy_fusion_impl = "te"

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing_dllm(
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
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=False,  # Not supported for custom FSDP for now, need to be set to False if using FSDP
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
            dataloader_type="cyclic",
            skip_getting_attention_mask_from_dataset=False,
        ),
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        # Tokenizer will get overwritten using the yaml config
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer",
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE,
        ),
        checkpoint=CheckpointConfig(
            finetune=True,
            pretrained_checkpoint=pretrained_checkpoint,
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


def distributed_fused_adam_with_cosine_annealing_dllm(
    precision: str = "bf16-mixed",
    lr_warmup_iters: int = 2000,
    lr_decay_iters: Optional[int] = None,
    weight_decay: float = 0.1,
    max_lr: float = 1e-4,
    min_lr: Optional[float] = None,
    clip_grad: float = 1.0,
) -> tuple[OptimizerConfig, SchedulerConfig]:
    """
    Creates a distributed fused Adam optimizer with cosine annealing scheduler.
    Here we use all default parameters from Megatron-Bridge
    Args:
        precision: Mixed precision type ("bf16-mixed", "16-mixed", etc.)
        lr_warmup_iters: Number of iterations for learning rate warmup
        lr_decay_iters: Number of iterations for learning rate decay. If None,
            defaults to train_iters during training.
        adam_beta1: Adam optimizer beta1 parameter
        adam_beta2: Adam optimizer beta2 parameter
        adam_eps: Adam optimizer epsilon parameter
        weight_decay: Weight decay coefficient
        max_lr: Maximum learning rate
        min_lr: Minimum learning rate (defaults to 0.1 * max_lr)
        clip_grad: Gradient clipping value

    Returns:
        Tuple of (OptimizerConfig, SchedulerConfig)
    """
    min_lr = min_lr if min_lr is not None else (0.1 * max_lr)
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=max_lr,
        min_lr=min_lr,
        weight_decay=weight_decay,
        bf16=precision == "bf16-mixed",
        fp16=precision == "16-mixed",
        use_distributed_optimizer=True,
        clip_grad=clip_grad,
    )

    scheduler = SchedulerConfig(
        start_weight_decay=weight_decay,
        end_weight_decay=weight_decay,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_warmup_iters=lr_warmup_iters,
        lr_warmup_init=0.0,
        lr_decay_iters=lr_decay_iters,
    )

    return optimizer, scheduler
