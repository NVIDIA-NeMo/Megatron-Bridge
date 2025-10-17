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

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import (
    HFDatasetConversationProvider,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig

from src.megatron.bridge.peft.lora import VLMLoRA


def pretrain_config(
    dir: Optional[str] = None,
    name: str = "nemotron_nano_v2_vl_pretrain",
    # Dataset configuration
    data_paths: Optional[List[str]] = None,
    data_args_path: Optional[str] = None,
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    per_split_data_args_path: Optional[str] = None,
    mock: bool = False,
    use_preloaded: bool = False,
    image_folder: Optional[str] = None,
    dataset_maker_name: str = "make_cord_v2_dataset",
    # Model configuration
    tensor_parallelism: int = 4,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    use_megatron_fsdp: bool = False,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 2,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 500,
    lr_decay_iters: Optional[int] = None,
    # Precision and comm overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # Tokenizer
    tokenizer_model: str = "/lustre/fs1/portfolios/coreai/users/chcui/pretrained_models/vlm-hf-code/nano_vl_v2",
    # Freeze options
    freeze_language_model: bool = False,
    freeze_vision_model: bool = False,
    freeze_vision_projection: bool = False,
    # Checkpointing
    save_interval: Optional[int] = 500,
) -> ConfigContainer:
    """
    Create a pre-training configuration for Nemotron Nano V2 VL.

    Note: Current dataset pipeline is text-centric. To train multimodal tokens,
    your preprocessed data should include placeholder tokens (e.g., <image>) as needed.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Build provider via AutoBridge and set parallel/seq params here
    bridge = AutoBridge.from_hf_pretrained("/lustre/fs1/portfolios/coreai/users/chcui/pretrained_models/vlm-hf-code/nano_vl_v2", trust_remote_code=True)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = pipeline_parallelism
    model_cfg.pipeline_dtype = pipeline_parallelism_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_parallelism
    model_cfg.context_parallel_size = context_parallelism
    model_cfg.sequence_parallel = sequence_parallelism
    model_cfg.seq_length = seq_length

    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Use HF-based VLM conversation dataset provider
    dataset_cfg = HFDatasetConversationProvider(
        sequence_length=seq_length,
        hf_processor_path=tokenizer_model,
        maker_name=dataset_maker_name,
        # Dataloader config parameters
        num_workers=2,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=500,
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
            average_in_collective=False,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
            # use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
        ),
        tokenizer=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE),
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


def finetune_config(
    *,
    pretrained_checkpoint: str,
    peft_on_language_model: bool = False,
    peft_on_vision_model: bool = False,
    save_checkpoint_dir: Optional[str] = None,
    **pretrain_kwargs,
) -> ConfigContainer:
    """Create a finetuning configuration for Nemotron Nano V2 VL.

    This helper wraps :func:`pretrain_config`, forwarding all keyword arguments to it
    while additionally wiring up the :class:`CheckpointConfig` for finetuning from a
    given *``pretrained_checkpoint``*.

    Parameters
    ----------
    pretrained_checkpoint: str
        Path to a Megatron-Bridge checkpoint (or a directory produced by
        ``convert_ckpt_hf_to_megatron``) that will be loaded before training.
    save_checkpoint_dir: str | None, default ``run_output_dir / "checkpoints"``
        Directory where new checkpoints will be saved / resumed from.  If not
        provided, we reuse the default path chosen by *pretrain_config*.
    peft_on_language_model: bool = True
        Whether to apply PEFT to the language model.
    peft_on_vision_model: bool = True
        Whether to apply PEFT to the vision model.
    **pretrain_kwargs: Any
        Additional keyword arguments are forwarded verbatim to
        :func:`pretrain_config` to customise the base recipe (e.g. batch size,
        learning rate, parallelism).
    """

    cfg = pretrain_config(**pretrain_kwargs)

    # Override Train hyper-parameters suitable for finetuning if the caller did
    # not explicitly pass them via **pretrain_kwargs.
    if pretrain_kwargs.get("train_iters") is None:
        cfg.train.train_iters = 10_000
    if pretrain_kwargs.get("lr") is None and hasattr(cfg.optimizer, "lr"):
        cfg.optimizer.lr = 1e-5  # type: ignore[attr-defined]
    if pretrain_kwargs.get("min_lr") is None and hasattr(cfg.optimizer, "min_lr"):
        cfg.optimizer.min_lr = 1e-6  # type: ignore[attr-defined]

    # Update CheckpointConfig for finetuning.
    ckpt_dir = save_checkpoint_dir or cfg.checkpoint.save or cfg.checkpoint.load  # type: ignore[attr-defined]
    cfg.checkpoint = CheckpointConfig(
        pretrained_checkpoint=pretrained_checkpoint,
        save=ckpt_dir,
        load=ckpt_dir,
        ckpt_format=cfg.checkpoint.ckpt_format,  # preserve existing choice
        fully_parallel_save=cfg.checkpoint.fully_parallel_save,
        save_interval=cfg.checkpoint.save_interval,
    )
    if peft_on_language_model:
        if peft_on_vision_model:
            cfg.peft = VLMLoRA(
                target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
                dim=16,
                alpha=32,
            )
        else:
            cfg.peft = VLMLoRA(
                target_modules=[
                    "*language_model*.linear_qkv", 
                    "*language_model*.linear_proj", 
                    "*language_model*.linear_fc1", 
                    "*language_model*.linear_fc2"
                ],
                dim=16,
                alpha=32,
                freeze_vision_model=False,
                freeze_vision_projection=False,
            )
            
        cfg.optimizer.lr = 5e-5
        cfg.optimizer.min_lr = 5e-6
        cfg.model.tensor_model_parallel_size = 1

    return cfg