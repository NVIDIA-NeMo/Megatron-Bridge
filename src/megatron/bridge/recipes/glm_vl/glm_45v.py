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
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.data.vlm_datasets import (
    HFDatasetConversationProvider,
    MockVLMConversationProvider,
    PreloadedVLMConversationProvider,
)
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
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


def set_glm_45v_pipeline_model_parallel_layout(
    model_cfg: GPTModelProvider, layout: Optional[Union[str, List[List[str]]]] = None
) -> None:
    """Set the GLM-4.5V pipeline model parallel layout.

    GLM-4.5V (based on GLM-4.5 Air) has 46 decoder layers and no MTP layers.
    This function sets up predefined layouts for common PP/VP combinations.

    Args:
        model_cfg: The model provider configuration to modify.
        layout: Optional custom layout. If None, uses predefined layouts based on PP/VP sizes.
    """
    # GLM-4.5V has no MTP layers
    last_layer = ["loss"]
    pp_size = model_cfg.pipeline_model_parallel_size or 1
    vp_size = model_cfg.virtual_pipeline_model_parallel_size or 1

    # GLM-4.5 Air has 46 decoder layers
    # Layout maps for common PP/VP combinations
    layout_map = {
        (1, 1): None,
        (2, 1): [["embedding"] + ["decoder"] * 23, ["decoder"] * 23 + last_layer],
        (4, 1): [["embedding"] + ["decoder"] * 11, ["decoder"] * 12, ["decoder"] * 12, ["decoder"] * 11 + last_layer],
        (8, 1): [["embedding"] + ["decoder"] * 5] + [["decoder"] * 6] * 6 + [["decoder"] * 5 + last_layer],
        (16, 1): [["embedding"] + ["decoder"] * 2] + [["decoder"] * 3] * 14 + [["decoder"] * 2 + last_layer],
    }

    if layout is not None:
        model_cfg.pipeline_model_parallel_layout = layout
    elif (pp_size, vp_size) in layout_map:
        model_cfg.pipeline_model_parallel_layout = layout_map[(pp_size, vp_size)]


class GLM45VCommonKwargs(TypedDict, total=False):
    """Typed options accepted by GLM-4.5V recipe helper functions."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    dataset_type: Optional[str]
    image_folder: Optional[str]
    tokenizer_model: Optional[str]
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    expert_model_parallel_size: int
    context_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    # Freeze options
    freeze_language_model: bool
    freeze_vision_model: bool
    freeze_vision_projection: bool
    # Checkpoint options
    pretrained_checkpoint: Optional[str]
    # Pipeline layout
    layout: Optional[Union[str, List[List[str]]]]
    # PEFT options
    peft: Optional[Union[str, PEFT]]
    finetune_lr: float
    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]


def glm_45v_finetune_config(**user_kwargs: Unpack[GLM45VCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for GLM-4.5V (based on GLM-4.5 Air 106B).

    Default configuration: 4 nodes, 32 GPUs total
    - LoRA/DoRA: TP=1, PP=8, EP=4 (32 GPUs, 4 nodes), LR=1e-4
    - Full SFT: TP=1, PP=8, EP=16 (128 GPUs, 16 nodes), LR=5e-6

    GLM-4.5V is a Vision-Language model with:
    - 106B total parameters (based on GLM-4.5 Air)
    - Sparse MoE with shared experts
    - Multi-modality support for images and videos

    See `_glm_45v_common` for the full list of parameters.
    """
    # Check if user is doing full SFT or PEFT
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: GLM45VCommonKwargs = {
        "hf_path": "zai-org/GLM-4.5V",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 4,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 16 if is_full_sft else 2,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    combined_kwargs: GLM45VCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _glm_45v_common(**combined_kwargs)


def _glm_45v_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "glm_45v_finetune",
    pretrained_checkpoint: Optional[str] = None,
    # Dataset configuration
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    dataset_type: Optional[str] = None,
    image_folder: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 2,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    expert_model_parallel_size: int = 4,
    context_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    # Training hyperparameters
    train_iters: int = 300000,
    global_batch_size: int = 32,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    lr_warmup_iters: int = 500,
    lr_decay_iters: Optional[int] = None,
    eval_interval: int = 500,
    save_interval: int = 500,
    # Precision and comm overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # Freeze options
    freeze_language_model: bool = False,
    freeze_vision_model: bool = False,
    freeze_vision_projection: bool = False,
    # Pipeline layout
    layout: Optional[Union[str, List[List[str]]]] = None,
    # PEFT options
    peft: Optional[Union[str, PEFT]] = None,
    finetune_lr: Optional[float] = None,
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """
    Create a fine-tuning configuration for GLM-4.5V models using a given HuggingFace path.

    The dataset pipeline is conversation-based. To train multimodal tokens, ensure your
    preprocessed data includes placeholders (e.g., <image>) as needed.

    GLM-4.5V is a Vision-Language model based on GLM-4.5 Air (106B parameters) with:
    - Sparse MoE architecture with shared experts
    - Multi-modal support for images and videos
    - MRoPE (Multi-Resolution Rotary Position Embedding)
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Build provider via AutoBridge and set parallel/seq params here
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.freeze_language_model = freeze_language_model
    model_cfg.freeze_vision_model = freeze_vision_model
    model_cfg.freeze_vision_projection = freeze_vision_projection
    model_cfg.seq_length = seq_length

    # Set pipeline model parallel layout for asymmetric stages
    set_glm_45v_pipeline_model_parallel_layout(model_cfg, layout)

    # Pipeline split for asymmetric stages are specified with the layout above
    model_cfg.account_for_embedding_in_pipeline_split = False
    model_cfg.account_for_loss_in_pipeline_split = False
    model_cfg.num_layers_in_first_pipeline_stage = None
    model_cfg.num_layers_in_last_pipeline_stage = None

    # Optimizer and scheduler - use finetune_lr if provided, otherwise use lr
    # Ensure min_lr does not exceed max_lr (use 10% of effective_lr as default min)
    effective_lr = finetune_lr if finetune_lr is not None else lr
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=effective_lr,
        min_lr=min(min_lr, effective_lr * 0.1),
    )

    # PEFT config
    peft_config = default_peft_config(peft)

    # Determine dataset selection strategy.
    _dataset_choice = dataset_type or "mock"
    _processor_model = tokenizer_model or hf_path

    if _dataset_choice == "mock":
        dataset_cfg: DatasetProvider = MockVLMConversationProvider(
            seq_length=seq_length,
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
            seq_length=seq_length,
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
            seq_length=seq_length,
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
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg
