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
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class Qwen3FinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 finetuning recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    # Finetuning-specific options
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int | None
    micro_batch_size: int
    seq_length: int
    eval_interval: int
    save_interval: int
    finetune_lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None


# =============================================================================
# Pretrain Recipes
# =============================================================================
# Each pretrain recipe returns a ConfigContainer with all settings directly visible.
# Users can modify the returned config as needed before training.


def qwen3_600m_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 0.6B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-0.6B").to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 1  # --tensor-model-parallel-size
    cfg.model.pipeline_model_parallel_size = 1  # --pipeline-model-parallel-size
    cfg.model.virtual_pipeline_model_parallel_size = None  # --num-layers-per-virtual-pipeline-stage
    cfg.model.context_parallel_size = 1  # --context-parallel-size
    cfg.model.sequence_parallel = False  # --sequence-parallel
    cfg.model.seq_length = 4096

    # Training optimizations
    cfg.model.cross_entropy_loss_fusion = True  # --cross-entropy-loss-fusion
    cfg.model.cross_entropy_fusion_impl = "te"  # --cross-entropy-fusion-impl
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Tokenizer (--tokenizer-type, --tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-0.6B"  # --tokenizer-model

    # =========================================================================
    # Training config (--train-samples, --exit-duration-in-mins, --manual-gc, etc.)
    # =========================================================================
    # cfg.train.train_samples = 268554688  # --train-samples (alternative to train_iters)
    # cfg.train.exit_duration_in_mins = 230  # --exit-duration-in-mins
    cfg.train.manual_gc_interval = 5  # --manual-gc-interval

    # =========================================================================
    # Dataset config (--data-path, --split, --num-workers, --no-mmap-bin-files)
    # =========================================================================
    # cfg.dataset.blend = (["path/to/data"], None)  # --data-path
    cfg.dataset.split = "99,1,0"  # --split
    cfg.dataset.mmap_bin_files = False  # --no-mmap-bin-files
    cfg.dataset.num_workers = 6  # --num-workers

    # =========================================================================
    # Logger config (--log-interval, --log-timers-to-tensorboard, --wandb-project, etc.)
    # =========================================================================
    cfg.logger.log_interval = 1  # --log-interval
    cfg.logger.log_memory_to_tensorboard = True  # --log-memory-to-tensorboard
    cfg.logger.log_params_norm = True  # --log-params-norm
    cfg.logger.log_validation_ppl_to_tensorboard = True  # --log-validation-ppl-to-tensorboard
    cfg.logger.log_throughput = True  # --log-throughput
    # cfg.logger.wandb_project = "my_project"  # --wandb-project
    # cfg.logger.wandb_exp_name = "Qwen3-0.6B-experiment"  # --wandb-exp-name

    # =========================================================================
    # Checkpoint config (--save, --load, --save-interval, --finetune, etc.)
    # =========================================================================
    cfg.checkpoint.finetune = False  # --finetune
    cfg.checkpoint.dist_ckpt_strictness = "log_all"  # --dist-ckpt-strictness

    # =========================================================================
    # Distributed config (--distributed-timeout-minutes, --enable-experimental)
    # =========================================================================
    cfg.dist.distributed_timeout_minutes = 60  # --distributed-timeout-minutes
    cfg.dist.enable_megatron_core_experimental = True  # --enable-experimental

    return cfg


def qwen3_1p7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 1.7B.

    Recommended parallelism: TP=1, PP=1 (fits on a single GPU).
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-1.7B").to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 1  # --tensor-model-parallel-size
    cfg.model.pipeline_model_parallel_size = 1  # --pipeline-model-parallel-size
    cfg.model.virtual_pipeline_model_parallel_size = None  # --num-layers-per-virtual-pipeline-stage
    cfg.model.context_parallel_size = 1  # --context-parallel-size
    cfg.model.sequence_parallel = False  # --sequence-parallel
    cfg.model.seq_length = 4096

    # Training optimizations
    cfg.model.cross_entropy_loss_fusion = True  # --cross-entropy-loss-fusion
    cfg.model.cross_entropy_fusion_impl = "te"  # --cross-entropy-fusion-impl
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Tokenizer (--tokenizer-type, --tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-1.7B"  # --tokenizer-model

    # =========================================================================
    # Training config (--train-samples, --exit-duration-in-mins, --manual-gc, etc.)
    # =========================================================================
    # cfg.train.train_samples = 268554688  # --train-samples (alternative to train_iters)
    # cfg.train.exit_duration_in_mins = 230  # --exit-duration-in-mins
    cfg.train.manual_gc_interval = 5  # --manual-gc-interval

    # =========================================================================
    # Dataset config (--data-path, --split, --num-workers, --no-mmap-bin-files)
    # =========================================================================
    # cfg.dataset.blend = (["path/to/data"], None)  # --data-path
    cfg.dataset.split = "99,1,0"  # --split
    cfg.dataset.mmap_bin_files = False  # --no-mmap-bin-files
    cfg.dataset.num_workers = 6  # --num-workers

    # =========================================================================
    # Logger config (--log-interval, --log-timers-to-tensorboard, --wandb-project, etc.)
    # =========================================================================
    cfg.logger.log_interval = 1  # --log-interval
    cfg.logger.log_memory_to_tensorboard = True  # --log-memory-to-tensorboard
    cfg.logger.log_params_norm = True  # --log-params-norm
    cfg.logger.log_validation_ppl_to_tensorboard = True  # --log-validation-ppl-to-tensorboard
    cfg.logger.log_throughput = True  # --log-throughput
    # cfg.logger.wandb_project = "my_project"  # --wandb-project
    # cfg.logger.wandb_exp_name = "Qwen3-1.7B-experiment"  # --wandb-exp-name

    # =========================================================================
    # Checkpoint config (--save, --load, --save-interval, --finetune, etc.)
    # =========================================================================
    cfg.checkpoint.finetune = False  # --finetune
    cfg.checkpoint.dist_ckpt_strictness = "log_all"  # --dist-ckpt-strictness

    # =========================================================================
    # Distributed config (--distributed-timeout-minutes, --enable-experimental)
    # =========================================================================
    cfg.dist.distributed_timeout_minutes = 60  # --distributed-timeout-minutes
    cfg.dist.enable_megatron_core_experimental = True  # --enable-experimental

    return cfg


def qwen3_4b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 4B.

    Recommended parallelism: TP=2, PP=1.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-4B").to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 2  # --tensor-model-parallel-size
    cfg.model.pipeline_model_parallel_size = 1  # --pipeline-model-parallel-size
    cfg.model.virtual_pipeline_model_parallel_size = None  # --num-layers-per-virtual-pipeline-stage
    cfg.model.context_parallel_size = 1  # --context-parallel-size
    cfg.model.sequence_parallel = True  # --sequence-parallel (enable for TP > 1)
    cfg.model.seq_length = 4096

    # Training optimizations
    cfg.model.cross_entropy_loss_fusion = True  # --cross-entropy-loss-fusion
    cfg.model.cross_entropy_fusion_impl = "te"  # --cross-entropy-fusion-impl
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Tokenizer (--tokenizer-type, --tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-4B"  # --tokenizer-model

    # =========================================================================
    # Training config (--train-samples, --exit-duration-in-mins, --manual-gc, etc.)
    # =========================================================================
    # cfg.train.train_samples = 268554688  # --train-samples (alternative to train_iters)
    # cfg.train.exit_duration_in_mins = 230  # --exit-duration-in-mins
    cfg.train.manual_gc_interval = 5  # --manual-gc-interval

    # =========================================================================
    # Dataset config (--data-path, --split, --num-workers, --no-mmap-bin-files)
    # =========================================================================
    # cfg.dataset.blend = (["path/to/data"], None)  # --data-path
    cfg.dataset.split = "99,1,0"  # --split
    cfg.dataset.mmap_bin_files = False  # --no-mmap-bin-files
    cfg.dataset.num_workers = 6  # --num-workers

    # =========================================================================
    # Logger config (--log-interval, --log-timers-to-tensorboard, --wandb-project, etc.)
    # =========================================================================
    cfg.logger.log_interval = 1  # --log-interval
    cfg.logger.log_memory_to_tensorboard = True  # --log-memory-to-tensorboard
    cfg.logger.log_params_norm = True  # --log-params-norm
    cfg.logger.log_validation_ppl_to_tensorboard = True  # --log-validation-ppl-to-tensorboard
    cfg.logger.log_throughput = True  # --log-throughput
    # cfg.logger.wandb_project = "my_project"  # --wandb-project
    # cfg.logger.wandb_exp_name = "Qwen3-4B-experiment"  # --wandb-exp-name

    # =========================================================================
    # Checkpoint config (--save, --load, --save-interval, --finetune, etc.)
    # =========================================================================
    cfg.checkpoint.finetune = False  # --finetune
    cfg.checkpoint.dist_ckpt_strictness = "log_all"  # --dist-ckpt-strictness

    # =========================================================================
    # Distributed config (--distributed-timeout-minutes, --enable-experimental)
    # =========================================================================
    cfg.dist.distributed_timeout_minutes = 60  # --distributed-timeout-minutes
    cfg.dist.enable_megatron_core_experimental = True  # --enable-experimental

    return cfg


def qwen3_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 8B.

    Recommended parallelism: TP=4, PP=1.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-8B").to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 4  # --tensor-model-parallel-size
    cfg.model.pipeline_model_parallel_size = 1  # --pipeline-model-parallel-size
    cfg.model.virtual_pipeline_model_parallel_size = None  # --num-layers-per-virtual-pipeline-stage
    cfg.model.context_parallel_size = 1  # --context-parallel-size
    cfg.model.sequence_parallel = True  # --sequence-parallel (enable for TP > 1)
    cfg.model.seq_length = 4096

    # Training optimizations
    cfg.model.cross_entropy_loss_fusion = True  # --cross-entropy-loss-fusion
    cfg.model.cross_entropy_fusion_impl = "te"  # --cross-entropy-fusion-impl
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Tokenizer (--tokenizer-type, --tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-8B"  # --tokenizer-model

    # =========================================================================
    # Training config (--train-samples, --exit-duration-in-mins, --manual-gc, etc.)
    # =========================================================================
    # cfg.train.train_samples = 268554688  # --train-samples (alternative to train_iters)
    # cfg.train.exit_duration_in_mins = 230  # --exit-duration-in-mins
    cfg.train.manual_gc_interval = 5  # --manual-gc-interval

    # =========================================================================
    # Dataset config (--data-path, --split, --num-workers, --no-mmap-bin-files)
    # =========================================================================
    # cfg.dataset.blend = (["path/to/data"], None)  # --data-path
    cfg.dataset.split = "99,1,0"  # --split
    cfg.dataset.mmap_bin_files = False  # --no-mmap-bin-files
    cfg.dataset.num_workers = 6  # --num-workers

    # =========================================================================
    # Logger config (--log-interval, --log-timers-to-tensorboard, --wandb-project, etc.)
    # =========================================================================
    cfg.logger.log_interval = 1  # --log-interval
    cfg.logger.log_memory_to_tensorboard = True  # --log-memory-to-tensorboard
    cfg.logger.log_params_norm = True  # --log-params-norm
    cfg.logger.log_validation_ppl_to_tensorboard = True  # --log-validation-ppl-to-tensorboard
    cfg.logger.log_throughput = True  # --log-throughput
    # cfg.logger.wandb_project = "my_project"  # --wandb-project
    # cfg.logger.wandb_exp_name = "Qwen3-8B-experiment"  # --wandb-exp-name

    # =========================================================================
    # Checkpoint config (--save, --load, --save-interval, --finetune, etc.)
    # =========================================================================
    cfg.checkpoint.finetune = False  # --finetune
    cfg.checkpoint.dist_ckpt_strictness = "log_all"  # --dist-ckpt-strictness

    # =========================================================================
    # Distributed config (--distributed-timeout-minutes, --enable-experimental)
    # =========================================================================
    cfg.dist.distributed_timeout_minutes = 60  # --distributed-timeout-minutes
    cfg.dist.enable_megatron_core_experimental = True  # --enable-experimental

    return cfg


def qwen3_14b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 14B.

    Recommended parallelism: TP=8, PP=1.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-14B").to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 8  # --tensor-model-parallel-size
    cfg.model.pipeline_model_parallel_size = 1  # --pipeline-model-parallel-size
    cfg.model.virtual_pipeline_model_parallel_size = None  # --num-layers-per-virtual-pipeline-stage
    cfg.model.context_parallel_size = 1  # --context-parallel-size
    cfg.model.sequence_parallel = True  # --sequence-parallel (enable for TP > 1)
    cfg.model.seq_length = 4096

    # Training optimizations
    cfg.model.cross_entropy_loss_fusion = True  # --cross-entropy-loss-fusion
    cfg.model.cross_entropy_fusion_impl = "te"  # --cross-entropy-fusion-impl
    cfg.model.init_method_std = 0.02  # --init-method-std

    # =========================================================================
    # Tokenizer (--tokenizer-type, --tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-14B"  # --tokenizer-model

    # =========================================================================
    # Training config (--train-samples, --exit-duration-in-mins, --manual-gc, etc.)
    # =========================================================================
    # cfg.train.train_samples = 268554688  # --train-samples (alternative to train_iters)
    # cfg.train.exit_duration_in_mins = 230  # --exit-duration-in-mins
    cfg.train.manual_gc_interval = 5  # --manual-gc-interval

    # =========================================================================
    # Dataset config (--data-path, --split, --num-workers, --no-mmap-bin-files)
    # =========================================================================
    # cfg.dataset.blend = (["path/to/data"], None)  # --data-path
    cfg.dataset.split = "99,1,0"  # --split
    cfg.dataset.mmap_bin_files = False  # --no-mmap-bin-files
    cfg.dataset.num_workers = 6  # --num-workers

    # =========================================================================
    # Logger config (--log-interval, --log-timers-to-tensorboard, --wandb-project, etc.)
    # =========================================================================
    cfg.logger.log_interval = 1  # --log-interval
    cfg.logger.log_memory_to_tensorboard = True  # --log-memory-to-tensorboard
    cfg.logger.log_params_norm = True  # --log-params-norm
    cfg.logger.log_validation_ppl_to_tensorboard = True  # --log-validation-ppl-to-tensorboard
    cfg.logger.log_throughput = True  # --log-throughput
    # cfg.logger.wandb_project = "my_project"  # --wandb-project
    # cfg.logger.wandb_exp_name = "Qwen3-14B-experiment"  # --wandb-exp-name

    # =========================================================================
    # Checkpoint config (--save, --load, --save-interval, --finetune, etc.)
    # =========================================================================
    cfg.checkpoint.finetune = False  # --finetune
    cfg.checkpoint.dist_ckpt_strictness = "log_all"  # --dist-ckpt-strictness

    # =========================================================================
    # Distributed config (--distributed-timeout-minutes, --enable-experimental)
    # =========================================================================
    cfg.dist.distributed_timeout_minutes = 60  # --distributed-timeout-minutes
    cfg.dist.enable_megatron_core_experimental = True  # --enable-experimental

    return cfg


def qwen3_32b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3 32B.

    Recommended parallelism: TP=8, PP=2 with recompute enabled for memory optimization.
    """
    cfg = _pretrain_common()

    # =========================================================================
    # Model config (--tensor-model-parallel-size, --pipeline-model-parallel-size, etc.)
    # =========================================================================
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-32B").to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 8  # --tensor-model-parallel-size
    cfg.model.pipeline_model_parallel_size = 2  # --pipeline-model-parallel-size
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None  # --num-layers-per-virtual-pipeline-stage
    cfg.model.context_parallel_size = 1  # --context-parallel-size
    cfg.model.sequence_parallel = True  # --sequence-parallel (enable for TP > 1)
    cfg.model.seq_length = 4096

    # Training optimizations
    cfg.model.cross_entropy_loss_fusion = True  # --cross-entropy-loss-fusion
    cfg.model.cross_entropy_fusion_impl = "te"  # --cross-entropy-fusion-impl
    cfg.model.init_method_std = 0.02  # --init-method-std

    # Enable recompute for memory optimization (large models)
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    # =========================================================================
    # Tokenizer (--tokenizer-type, --tokenizer-model)
    # =========================================================================
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-32B"  # --tokenizer-model

    # =========================================================================
    # Training config (--train-samples, --exit-duration-in-mins, --manual-gc, etc.)
    # =========================================================================
    # cfg.train.train_samples = 268554688  # --train-samples (alternative to train_iters)
    # cfg.train.exit_duration_in_mins = 230  # --exit-duration-in-mins
    cfg.train.manual_gc_interval = 5  # --manual-gc-interval

    # =========================================================================
    # Dataset config (--data-path, --split, --num-workers, --no-mmap-bin-files)
    # =========================================================================
    # cfg.dataset.blend = (["path/to/data"], None)  # --data-path
    cfg.dataset.split = "99,1,0"  # --split
    cfg.dataset.mmap_bin_files = False  # --no-mmap-bin-files
    cfg.dataset.num_workers = 6  # --num-workers

    # =========================================================================
    # Logger config (--log-interval, --log-timers-to-tensorboard, --wandb-project, etc.)
    # =========================================================================
    cfg.logger.log_interval = 1  # --log-interval
    cfg.logger.log_memory_to_tensorboard = True  # --log-memory-to-tensorboard
    cfg.logger.log_params_norm = True  # --log-params-norm
    cfg.logger.log_validation_ppl_to_tensorboard = True  # --log-validation-ppl-to-tensorboard
    cfg.logger.log_throughput = True  # --log-throughput
    # cfg.logger.wandb_project = "my_project"  # --wandb-project
    # cfg.logger.wandb_exp_name = "Qwen3-32B-experiment"  # --wandb-exp-name

    # =========================================================================
    # Checkpoint config (--save, --load, --save-interval, --finetune, etc.)
    # =========================================================================
    cfg.checkpoint.finetune = False  # --finetune
    cfg.checkpoint.dist_ckpt_strictness = "log_all"  # --dist-ckpt-strictness

    # =========================================================================
    # Distributed config (--distributed-timeout-minutes, --enable-experimental)
    # =========================================================================
    cfg.dist.distributed_timeout_minutes = 60  # --distributed-timeout-minutes
    cfg.dist.enable_megatron_core_experimental = True  # --enable-experimental

    return cfg


def _pretrain_common() -> ConfigContainer:
    """Create a base pre-training ConfigContainer with common defaults for any language model.

    This function returns a ConfigContainer template with sensible defaults.
    The caller MUST set `cfg.model` and `cfg.tokenizer.tokenizer_model` before use.

    Returns:
        ConfigContainer: Base configuration template for pre-training.
    """
    # Default output directories
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Default optimizer and scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=None,  # Defaults to train_iters during validation
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg = ConfigContainer(
        # Model - MUST be set by each recipe before use
        model=None,  # type: ignore[arg-type]
        # Training config
        train=TrainingConfig(
            train_iters=300000,
            eval_interval=500,
            eval_iters=32,
            global_batch_size=32,
            micro_batch_size=2,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        # Optimizer and scheduler
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        # DDP config - these are the commonly overridden settings
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
        ),
        # Dataset config - uses mock data by default
        dataset=GPTDatasetConfig(
            random_seed=1234,
            reset_attention_mask=False,
            reset_position_ids=False,
            eod_mask_loss=False,
            seq_length=4096,
            num_dataset_builder_threads=1,
            blend=None,  # Mock data mode
            blend_per_split=None,
            split="1,1,1",
            data_sharding=True,
            dataloader_type="single",
            skip_getting_attention_mask_from_dataset=True,
        ),
        # Logger config
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        # Tokenizer - placeholder, each recipe should set tokenizer_model
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=None,  # Must be set by each recipe
        ),
        # Checkpoint config
        checkpoint=CheckpointConfig(
            save_interval=500,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        # RNG config
        rng=RNGConfig(seed=1234),
        # Distributed init config
        dist=DistributedInitConfig(),
        # Mixed precision - bf16 by default
        mixed_precision="bf16_mixed",
    )

    return cfg


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
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
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
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
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
        "tensor_model_parallel_size": 2 if is_full_sft else 1,  # Match NeMo2: higher TP for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 1,
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
        "tensor_model_parallel_size": 4 if is_full_sft else 1,  # Match NeMo2: TP=4 for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 1,
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
        "tensor_model_parallel_size": 8 if is_full_sft else 1,  # Match NeMo2: TP=8 for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 1,
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
        "tensor_model_parallel_size": 8 if is_full_sft else 1,  # Match NeMo2: TP=8 for SFT, TP=1 for LoRA
        "pipeline_model_parallel_size": 2 if is_full_sft else 1,  # PP=2 for SFT, PP=1 for LoRA
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
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = None,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
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
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
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
