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
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import (
    HFDatasetConversationProvider,
    MockVLMConversationProvider,
    PreloadedVLMConversationProvider,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetProvider,
    DistributedDataParallelConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class Qwen25VLCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen2.5-VL recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: str | None
    name: str
    # Dataset configuration
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    dataset_type: str | None
    image_folder: str | None
    tokenizer_model: str | None
    # Model configuration
    tensor_parallelism: int
    pipeline_parallelism: int
    pipeline_parallelism_dtype: torch.dtype
    virtual_pipeline_parallelism: int | None
    context_parallelism: int
    sequence_parallelism: bool
    use_megatron_fsdp: bool
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
    # Freeze options
    freeze_language_model: bool
    freeze_vision_model: bool
    freeze_vision_projection: bool
    # Checkpoint options
    pretrained_checkpoint: str | None


def qwen25_vl_3b_finetune_config(**user_kwargs: Unpack[Qwen25VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen2.5-VL 3B Instruct.

    See `_qwen25_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen25VLCommonKwargs = {
        "hf_path": "Qwen/Qwen2.5-VL-3B-Instruct",
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
    }
    combined_kwargs: Qwen25VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen25_vl_common(**combined_kwargs)


def qwen25_vl_7b_finetune_config(**user_kwargs: Unpack[Qwen25VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen2.5-VL 7B Instruct.

    See `_qwen25_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen25VLCommonKwargs = {
        "hf_path": "Qwen/Qwen2.5-VL-7B-Instruct",
        "tensor_parallelism": 2,
        "pipeline_parallelism": 1,
    }
    combined_kwargs: Qwen25VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen25_vl_common(**combined_kwargs)


def qwen25_vl_32b_finetune_config(**user_kwargs: Unpack[Qwen25VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen2.5-VL 32B Instruct.

    See `_qwen25_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen25VLCommonKwargs = {
        "hf_path": "Qwen/Qwen2.5-VL-32B-Instruct",
        "tensor_parallelism": 8,
        "pipeline_parallelism": 2,
        "pipeline_parallelism_dtype": torch.bfloat16,
    }
    combined_kwargs: Qwen25VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen25_vl_common(**combined_kwargs)


def qwen25_vl_72b_finetune_config(**user_kwargs: Unpack[Qwen25VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen2.5-VL 72B Instruct.

    See `_qwen25_vl_common` for the full list of parameters.
    """
    recommended_kwargs: Qwen25VLCommonKwargs = {
        "hf_path": "Qwen/Qwen2.5-VL-72B-Instruct",
        "tensor_parallelism": 8,
        "pipeline_parallelism": 4,
        "pipeline_parallelism_dtype": torch.bfloat16,
    }
    combined_kwargs: Qwen25VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen25_vl_common(**combined_kwargs)


def _qwen25_vl_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "qwen25_vl_finetune",
    pretrained_checkpoint: str | None = None,
    # Dataset configuration
    train_data_path: list[str] | None = None,
    valid_data_path: list[str] | None = None,
    test_data_path: list[str] | None = None,
    dataset_type: str | None = None,
    image_folder: str | None = None,
    tokenizer_model: str | None = None,
    # Model configuration
    tensor_parallelism: int = 2,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: torch.dtype = None,
    virtual_pipeline_parallelism: int | None = None,
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
    lr_decay_iters: int | None = None,
    eval_interval: int = 500,
    save_interval: int = 500,
    # Precision and comm overlap
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # Freeze options
    freeze_language_model: bool = False,
    freeze_vision_model: bool = False,
    freeze_vision_projection: bool = False,
) -> ConfigContainer:
    """
    Create a fine-tuning configuration for Qwen2.5-VL models using a given HuggingFace path.

    The dataset pipeline is conversation-based. To train multimodal tokens, ensure your
    preprocessed data includes placeholders (e.g., <image>) as needed.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Build provider via AutoBridge and set parallel/seq params here
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_parallelism
    model_cfg.pipeline_model_parallel_size = pipeline_parallelism
    model_cfg.pipeline_dtype = pipeline_parallelism_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_parallelism
    model_cfg.context_parallel_size = context_parallelism
    model_cfg.sequence_parallel = sequence_parallelism
    model_cfg.freeze_language_model = freeze_language_model
    model_cfg.freeze_vision_model = freeze_vision_model
    model_cfg.freeze_vision_projection = freeze_vision_projection
    model_cfg.seq_length = seq_length

    # Optimizer and scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=lr,
        min_lr=min_lr,
    )

    # Determine dataset selection strategy.
    _dataset_choice = dataset_type or "mock"
    _processor_model = tokenizer_model or hf_path

    if _dataset_choice == "mock":
        dataset_cfg: DatasetProvider = MockVLMConversationProvider(
            sequence_length=seq_length,
            hf_processor_path=_processor_model,
            prompt="Describe this image.",
            num_workers=1,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
            create_attention_mask=True,
            pad_to_max_length=True,
        )
    elif _dataset_choice == "preloaded":
        dataset_cfg = PreloadedVLMConversationProvider(
            sequence_length=seq_length,
            hf_processor_path=_processor_model,
            train_data_path=train_data_path[0] if isinstance(train_data_path, list) else train_data_path,
            valid_data_path=valid_data_path[0] if isinstance(valid_data_path, list) else valid_data_path,
            test_data_path=test_data_path[0] if isinstance(test_data_path, list) else test_data_path,
            image_folder=image_folder,
            num_workers=2,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
        )
    elif _dataset_choice == "hf":
        dataset_cfg = HFDatasetConversationProvider(
            sequence_length=seq_length,
            hf_processor_path=_processor_model,
            maker_name="make_cord_v2_dataset",
            num_workers=2,
            dataloader_type="single",
            data_sharding=True,
            pin_memory=True,
            persistent_workers=False,
        )
    else:
        raise ValueError(f"Unsupported dataset_type '{_dataset_choice}'. Expected one of ['mock', 'preloaded', 'hf'].")

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
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            average_in_collective=True,
            data_parallel_sharding_strategy="optim_grads_params",
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,
        ),
        dataset=dataset_cfg,
        logger=LoggerConfig(
            log_interval=10,
            tensorboard_dir=tensorboard_dir,
            log_timers_to_tensorboard=True,
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
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg
