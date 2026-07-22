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
from typing import Optional, Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import ChatSFTPreprocessingConfig, DirectHFSFTDatasetConfig, HFDatasetSourceConfig
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


def qwen2_audio_7b_sft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a fine-tuning config for Qwen2-Audio 7B Instruct.

    Default configuration: 1 node, TP=1, PP=1
    - Full SFT: LR=5e-6
    """
    cfg = _qwen2_audio_common(
        hf_path="Qwen/Qwen2-Audio-7B-Instruct",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        peft=None,
        finetune_lr=5e-6,
    )

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen2_audio_7b_peft_1gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2-Audio 7B Instruct."""
    cfg = _qwen2_audio_common(
        hf_path="Qwen/Qwen2-Audio-7B-Instruct",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        peft=peft_scheme,
        finetune_lr=1e-4,
    )

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def _qwen2_audio_common(
    hf_path: str,
    output_dir: str | None = None,
    name: str = "qwen2_audio_finetune",
    pretrained_checkpoint: Optional[str] = None,
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Training hyperparameters
    train_iters: int = 2000,
    global_batch_size: int = 32,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 5,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 500,
    save_interval: int = 200,
    # Precision
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    # Freeze options
    freeze_language_model: bool = False,
    freeze_audio_model: bool = False,
    freeze_audio_projection: bool = False,
    # PEFT options
    peft: Optional[Union[str, PEFT]] = None,
    finetune_lr: Optional[float] = None,
    # Dataset
    source: HFDatasetSourceConfig | None = None,
    validation_source: HFDatasetSourceConfig | None = None,
    test_source: HFDatasetSourceConfig | None = None,
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Create a fine-tuning configuration for Qwen2-Audio models."""
    base_output_dir = output_dir if output_dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Build provider via AutoBridge
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.freeze_language_model = freeze_language_model
    model_cfg.freeze_audio_model = freeze_audio_model
    model_cfg.freeze_audio_projection = freeze_audio_projection
    model_cfg.seq_length = seq_length

    # Optimizer and scheduler
    effective_lr = finetune_lr if finetune_lr is not None else lr
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=effective_lr,
        min_lr=min_lr,
    )

    # PEFT config
    peft_config = default_peft_config(peft)

    # Dataset: named CommonVoice source preset owns its Hub path and schema adapter.
    if source is None:
        source = HFDatasetSourceConfig(dataset_name="cv17")
    if validation_source is None:
        validation_source = source.with_split("validation")
    dataset_cfg = DirectHFSFTDatasetConfig(
        seq_length=seq_length,
        preprocessing=ChatSFTPreprocessingConfig(),
        hf_processor_path=hf_path,
        source=source,
        validation_source=validation_source,
        test_source=test_source,
        num_workers=2,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=0,
        ),
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=1,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_exp_name=wandb_exp_name,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
        checkpoint=CheckpointConfig(
            pretrained_checkpoint=pretrained_checkpoint,
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=1234),
        peft=peft_config,
        mixed_precision=precision_config,
    )

    return cfg


__all__ = [
    "qwen2_audio_7b_peft_1gpu_h100_bf16_config",
    "qwen2_audio_7b_sft_1gpu_h100_bf16_config",
]
