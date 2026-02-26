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

from megatron.core.distributed import DistributedDataParallelConfig

from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.utils.finetune_utils import default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
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
            split="9999,8,2",
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
        comm_overlap=None,
        # Mixed precision - bf16 by default
        mixed_precision="bf16_mixed",
    )

    return cfg


def _sft_common() -> ConfigContainer:
    """Create a base SFT (Supervised Fine-Tuning) ConfigContainer with common defaults.

    This function returns a ConfigContainer template with sensible defaults for full SFT
    (not LoRA/DoRA). The caller MUST set `cfg.model` and `cfg.tokenizer.tokenizer_model`
    before use.

    Key differences from pre-training:
    - Uses HFDatasetConfig with SQuAD as default dataset
    - Lower learning rate (5e-6) suitable for full fine-tuning
    - Fewer training iterations (1000)
    - Smaller batch sizes
    - Supports pretrained_checkpoint loading
    - No PEFT (full parameter training)

    Returns:
        ConfigContainer: Base configuration template for full SFT.
    """
    # Default output directories
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Default sequence length for SFT
    seq_length = 2048

    # Packed sequence is enabled by default for training efficiency
    # pad_seq_to_mult should be set to context_parallel_size * 2 if CP > 1
    packed_sequence = True
    pad_seq_to_mult = 1  # Override in model config if context_parallel_size > 1

    # Optimizer and scheduler with lower LR for full SFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,  # Defaults to train_iters during validation
        max_lr=5e-6,  # Lower LR for full fine-tuning
        min_lr=0.0,
        adam_beta2=0.98,  # Common for fine-tuning
    )

    cfg = ConfigContainer(
        # Model - MUST be set by each recipe before use
        model=None,  # type: ignore[arg-type]
        # Training config - shorter training for SFT
        train=TrainingConfig(
            train_iters=1000,
            global_batch_size=128,
            micro_batch_size=1,
            eval_interval=100,
            eval_iters=32,
        ),
        # Optimizer and scheduler
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        # DDP config - minimal settings, model-specific configs can override
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
        ),
        # Dataset config - uses SQuAD with packed sequences by default
        dataset=default_squad_config(
            seq_length=seq_length, packed_sequence=packed_sequence, pad_seq_to_mult=pad_seq_to_mult
        ),
        # Logger config
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        # Tokenizer - placeholder, each recipe should set tokenizer_model
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=None,  # Must be set by each recipe
        ),
        # Checkpoint config with pretrained_checkpoint support
        checkpoint=CheckpointConfig(
            save_interval=100,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=None,  # Set to load from pretrained weights
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        # RNG config - different seed from pretrain
        rng=RNGConfig(seed=5678),
        # Distributed init config
        dist=DistributedInitConfig(),
        comm_overlap=None,
        # Mixed precision - bf16 by default
        mixed_precision="bf16_mixed",
        # No PEFT for full SFT
        peft=None,
    )

    return cfg


def _peft_common() -> ConfigContainer:
    """Create a base PEFT (Parameter-Efficient Fine-Tuning) ConfigContainer with LoRA defaults.

    This function returns a ConfigContainer template with sensible defaults for PEFT
    using LoRA. The caller MUST set `cfg.model` and `cfg.tokenizer.tokenizer_model`
    before use.

    Key differences from full SFT:
    - Higher learning rate (1e-4) suitable for adapter training
    - LoRA enabled by default with standard settings (dim=32, alpha=32)
    - Targets all linear layers: linear_qkv, linear_proj, linear_fc1, linear_fc2

    Returns:
        ConfigContainer: Base configuration template for PEFT with LoRA.
    """
    # Default output directories
    base_output_dir = os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, "default")
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Default sequence length for PEFT
    seq_length = 2048

    # Packed sequence is enabled by default for training efficiency
    # pad_seq_to_mult should be set to context_parallel_size * 2 if CP > 1
    packed_sequence = True
    pad_seq_to_mult = 1  # Override in model config if context_parallel_size > 1

    # Optimizer and scheduler with higher LR for PEFT (only training adapters)
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=50,
        lr_decay_iters=None,  # Defaults to train_iters during validation
        max_lr=1e-4,  # Higher LR for adapter training
        min_lr=0.0,
        adam_beta2=0.98,  # Common for fine-tuning
    )

    cfg = ConfigContainer(
        # Model - MUST be set by each recipe before use
        model=None,  # type: ignore[arg-type]
        # Training config - shorter training for PEFT
        train=TrainingConfig(
            train_iters=1000,
            global_batch_size=128,
            micro_batch_size=1,
            eval_interval=100,
            eval_iters=32,
        ),
        # Optimizer and scheduler
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        # DDP config - minimal settings for PEFT
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
        ),
        # Dataset config - uses SQuAD with packed sequences by default
        dataset=default_squad_config(
            seq_length=seq_length, packed_sequence=packed_sequence, pad_seq_to_mult=pad_seq_to_mult
        ),
        # Logger config
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        # Tokenizer - placeholder, each recipe should set tokenizer_model
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=None,  # Must be set by each recipe
        ),
        # Checkpoint config with pretrained_checkpoint support
        checkpoint=CheckpointConfig(
            save_interval=100,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=None,  # Set to load from pretrained weights
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        # RNG config - different seed from pretrain
        rng=RNGConfig(seed=5678),
        # Distributed init config
        dist=DistributedInitConfig(),
        comm_overlap=None,
        # Mixed precision - bf16 by default
        mixed_precision="bf16_mixed",
        # LoRA config with standard defaults
        peft=LoRA(
            target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
            dim=32,
            alpha=32,
            dropout=0.0,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
            a2a_experimental=False,
            lora_dtype=None,  # Uses model's dtype
        ),
    )

    return cfg
