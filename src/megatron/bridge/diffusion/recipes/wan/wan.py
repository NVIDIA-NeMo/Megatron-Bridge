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

from megatron.bridge.diffusion.data.wan.wan_energon_datamodule import WanDatasetConfig
from megatron.bridge.diffusion.models.wan.wan_provider import WanModelProvider1_3B, WanModelProvider14B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


def wan_1_3b_pretrain_config() -> ConfigContainer:
    """
    Return a pre-training configuration for WAN 1.3B model.

    Default parallelism: TP=1, PP=1, CP=8. Uses mock/synthetic data when dataset.path
    is not set. To use real data, override via CLI: dataset.path=/path/to/wds
    """
    # Deferred imports to avoid circular import
    from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
    from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Output directories
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    name = "default"
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Model configuration
    model_cfg = WanModelProvider1_3B()
    model_cfg.tensor_model_parallel_size = 1
    model_cfg.pipeline_model_parallel_size = 1
    model_cfg.pipeline_dtype = torch.bfloat16
    model_cfg.virtual_pipeline_model_parallel_size = None
    model_cfg.context_parallel_size = 8
    model_cfg.sequence_parallel = False

    # Training hyperparameters
    train_iters = 10000
    global_batch_size = 2
    micro_batch_size = 1
    lr = 0.9e-4
    lr_warmup_iters = 2000

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
    )
    opt_config.use_precision_aware_optimizer = False

    precision_config = get_mixed_precision_config("bf16_mixed")
    precision_config.grad_reduce_in_fp32 = False

    # Dataset configuration (path=None => mock/synthetic data)
    dataset = WanDatasetConfig(
        path=None,
        seq_length=1024,
        packing_buffer_size=None,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_workers=16,
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
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
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=False,
        ),
        dataset=dataset,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=None,
        mixed_precision=precision_config,
    )

    return cfg


def wan_14b_pretrain_config() -> ConfigContainer:
    """
    Return a pre-training configuration for WAN 14B model.

    Default parallelism: TP=2, PP=1, CP=4, SP=True. Uses mock/synthetic data when
    dataset.path is not set. To use real data, override via CLI: dataset.path=/path/to/wds
    """
    # Deferred imports to avoid circular import
    from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
    from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Output directories
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    name = "default"
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Model configuration
    model_cfg = WanModelProvider14B()
    model_cfg.tensor_model_parallel_size = 2
    model_cfg.pipeline_model_parallel_size = 1
    model_cfg.pipeline_dtype = torch.bfloat16
    model_cfg.virtual_pipeline_model_parallel_size = None
    model_cfg.context_parallel_size = 4
    model_cfg.sequence_parallel = True
    model_cfg.recompute_granularity = "full"
    model_cfg.recompute_method = "uniform"
    model_cfg.recompute_num_layers = 1

    # Training hyperparameters
    train_iters = 10000
    global_batch_size = 1
    micro_batch_size = 1
    lr = 0.9e-4
    lr_warmup_iters = 2000

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=train_iters,
        max_lr=lr,
    )
    opt_config.use_precision_aware_optimizer = False

    precision_config = get_mixed_precision_config("bf16_mixed")
    precision_config.grad_reduce_in_fp32 = False

    # Dataset configuration (path=None => mock/synthetic data)
    dataset = WanDatasetConfig(
        path=None,
        seq_length=1024,
        packing_buffer_size=None,
        micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size,
        num_workers=16,
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=2000,
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
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            use_distributed_optimizer=True,
            use_megatron_fsdp=False,
        ),
        dataset=dataset,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            save_interval=2000,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        comm_overlap=None,
        mixed_precision=precision_config,
    )

    return cfg


def wan_1_3b_sft_config(pretrained_checkpoint: str | None = None) -> ConfigContainer:
    """
    Return a fine-tuning configuration for WAN 1.3B model.

    Uses the same defaults as wan_1_3b_pretrain_config() and overrides checkpoint to load from
    pretrained_checkpoint when provided.
    """
    cfg = wan_1_3b_pretrain_config()
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")

    cfg.checkpoint = CheckpointConfig(
        save_interval=2000,
        save=checkpoint_dir,
        load=checkpoint_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )
    return cfg


def wan_14b_sft_config(pretrained_checkpoint: str | None = None) -> ConfigContainer:
    """
    Return a fine-tuning configuration for WAN 14B model.

    Uses the same defaults as wan_14b_pretrain_config() and overrides checkpoint to load from
    pretrained_checkpoint when provided.
    """
    cfg = wan_14b_pretrain_config()
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")

    cfg.checkpoint = CheckpointConfig(
        save_interval=2000,
        save=checkpoint_dir,
        load=checkpoint_dir,
        pretrained_checkpoint=pretrained_checkpoint,
        ckpt_format="torch_dist",
        fully_parallel_save=True,
    )
    return cfg
