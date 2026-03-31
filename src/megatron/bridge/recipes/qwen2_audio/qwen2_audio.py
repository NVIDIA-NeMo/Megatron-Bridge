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
from typing import Optional, Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import HFDatasetConversationProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
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


def qwen2_audio_7b_finetune_config(**user_kwargs) -> ConfigContainer:
    """Return a fine-tuning config for Qwen2-Audio 7B Instruct.

    Default configuration: 1 node, TP=1, PP=1
    - LoRA/DoRA: LR=1e-4
    - Full SFT: LR=5e-6

    See `_qwen2_audio_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs = {
        "hf_path": "Qwen/Qwen2-Audio-7B-Instruct",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    combined_kwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen2_audio_common(**combined_kwargs)


def _qwen2_audio_common(
    hf_path: str,
    dir: Optional[str] = None,
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
    maker_name: str = "make_default_audio_dataset",
    maker_kwargs: Optional[dict] = None,  # defaults applied below
    val_maker_kwargs: Optional[dict] = None,  # per-split overrides for validation
    test_maker_kwargs: Optional[dict] = None,  # per-split overrides for test
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Create a fine-tuning configuration for Qwen2-Audio models."""
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
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

    # Dataset: HF conversation provider with audio maker
    if maker_kwargs is None:
        maker_kwargs = {
            "path_or_dataset": "yuekai/aishell",
            "subset": "train",
            "split": "train",
        }
    if val_maker_kwargs is None:
        val_maker_kwargs = {
            "subset": "dev",
            "split": "test",
        }
    dataset_cfg = HFDatasetConversationProvider(
        seq_length=seq_length,
        hf_processor_path=hf_path,
        maker_name=maker_name,
        maker_kwargs=maker_kwargs,
        val_maker_kwargs=val_maker_kwargs,
        test_maker_kwargs=test_maker_kwargs,
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
