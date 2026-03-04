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

"""
Fine-tuning recipes for Qwen3.5 Vision-Language Models.

Qwen3.5 is a family of VLMs that combine a hybrid Gated DeltaNet (GDN) + Gated
Attention language model with a vision encoder.  Two variants are supported:

- **Dense** (e.g., Qwen3.5-27B): standard dense MLP
- **MoE** (e.g., Qwen3.5-35B-A3B, 122B-A10B, 397B-A17B): Mixture of Experts
  with shared experts

Each public function returns a ready-to-use :class:`ConfigContainer` for
fine-tuning.  Pass ``peft="lora"`` for parameter-efficient
fine-tuning, or leave ``peft=None`` for full supervised fine-tuning (SFT).
"""

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
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config as _default_peft_config
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
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed


class Qwen35VLCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3.5 VL recipe helpers."""

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
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: Optional[int]
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    enable_recompute: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
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
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    # Freeze options
    pretrained_checkpoint: Optional[str]
    freeze_language_model: bool
    freeze_vision_model: bool
    freeze_vision_projection: bool
    # PEFT options
    peft: Optional[Union[str, PEFT]]
    finetune_lr: float
    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]


# ---------------------------------------------------------------------------
# Dense variant: Qwen3.5-800M
# ---------------------------------------------------------------------------


def qwen35_vl_800m_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-800M (dense).

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1  (1 node), LR=1e-4
    - Full SFT:  TP=1, PP=1  (1 node), LR=5e-6

    Note: num_kv_heads=2, so max TP=2.

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-0.8B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# Dense variant: Qwen3.5-2B
# ---------------------------------------------------------------------------


def qwen35_vl_2b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-2B (dense).

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1  (1 node), LR=1e-4
    - Full SFT:  TP=1, PP=1  (1 node), LR=5e-6

    Note: num_kv_heads=2, so max TP=2.

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-2B",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# Dense variant: Qwen3.5-4B
# ---------------------------------------------------------------------------


def qwen35_vl_4b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-4B (dense).

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1  (1 node), LR=1e-4
    - Full SFT:  TP=2, PP=1  (1 node), LR=5e-6

    Note: num_kv_heads=4, so max TP=4.

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-4B",
        "tensor_model_parallel_size": 2 if is_full_sft else 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# Dense variant: Qwen3.5-9B
# ---------------------------------------------------------------------------


def qwen35_vl_9b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-9B (dense).

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1  (1 node), LR=1e-4
    - Full SFT:  TP=4, PP=1  (1 node), LR=5e-6

    Note: num_kv_heads=4, so max TP=4.

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-9B",
        "tensor_model_parallel_size": 4 if is_full_sft else 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# Dense variant: Qwen3.5-27B
# ---------------------------------------------------------------------------


def qwen35_vl_27b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-27B (dense).

    Default configuration:
    - LoRA/DoRA: TP=2, PP=1       (1 node),  LR=1e-4
    - Full SFT:  TP=4, PP=4       (2 nodes), LR=5e-6

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-27B",
        "tensor_model_parallel_size": 4 if is_full_sft else 2,
        "pipeline_model_parallel_size": 4 if is_full_sft else 1,
        "pipeline_dtype": torch.bfloat16 if is_full_sft else None,
        "expert_model_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# MoE variant: Qwen3.5-35B-A3B
# ---------------------------------------------------------------------------


def qwen35_vl_35b_a3b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-35B-A3B (MoE).

    This is a small Mixture-of-Experts model.  Recommended to use with expert
    parallelism (EP) for efficient training.

    Default configuration:
    - LoRA/DoRA: TP=2, PP=1, EP=4   (1 node),  LR=2e-4
    - Full SFT:  TP=2, PP=1, EP=16  (2 nodes), LR=2e-5

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-35B-A3B",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 16 if is_full_sft else 4,
        "expert_tensor_parallel_size": 1,
        "peft": peft_value,
        "finetune_lr": 2e-5 if is_full_sft else 2e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "min_lr": 2e-6 if is_full_sft else 1e-4,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# MoE variant: Qwen3.5-122B-A10B
# ---------------------------------------------------------------------------


def qwen35_vl_122b_a10b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-122B-A10B (MoE).

    This is a medium-sized Mixture-of-Experts model.  Recommended to use with
    expert parallelism (EP) for efficient training.

    Default configuration:
    - LoRA/DoRA: TP=2, PP=1, EP=8   (1 node),  LR=2e-4
    - Full SFT:  TP=2, PP=4, EP=8  (4 nodes), LR=2e-5

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-122B-A10B",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 6 if is_full_sft else 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 8,
        "expert_tensor_parallel_size": 1,
        "peft": peft_value,
        "enable_recompute": is_full_sft,
        "finetune_lr": 2e-5 if is_full_sft else 2e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 36,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# MoE variant: Qwen3.5-397B-A17B
# ---------------------------------------------------------------------------
# TODO note this down somewhere
# For multinode training, if you encounter a file lock issue, you can replace hf_path with the local
# path to the model, e.g hf_home/hub/models--Qwen--Qwen3.5-397B-A17B/snapshots/... directory


def qwen35_vl_397b_a17b_finetune_config(**user_kwargs: Unpack[Qwen35VLCommonKwargs]) -> ConfigContainer:
    """Return a fine-tuning config for Qwen3.5-397B-A17B (MoE).

    This is a Mixture-of-Experts model with 512 experts and top-10 routing.
    Recommended to use with expert parallelism (EP) for efficient training.

    Default configuration:
    - LoRA/DoRA: TP=2, PP=1, EP=32  (4 nodes),  LR=2e-4
    - Full SFT:  TP=2, PP=4, EP=32  (16 nodes), LR=2e-5

    See `_qwen35_vl_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Qwen35VLCommonKwargs = {
        "hf_path": "Qwen/Qwen3.5-397B-A17B",
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 4 if is_full_sft else 1,
        "pipeline_dtype": torch.bfloat16,
        "expert_model_parallel_size": 32,
        "expert_tensor_parallel_size": 1,
        "peft": peft_value,
        "enable_recompute": is_full_sft,
        "finetune_lr": 2e-5 if is_full_sft else 2e-4,
        "freeze_language_model": False,
        "freeze_vision_model": False,
        "freeze_vision_projection": False,
        "lr_warmup_iters": 200,
        "micro_batch_size": 1,
        "global_batch_size": 32,
    }
    combined_kwargs: Qwen35VLCommonKwargs = {**recommended_kwargs, **user_kwargs}
    return _qwen35_vl_common(**combined_kwargs)


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------


def _qwen35_vl_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "qwen35_vl_finetune",
    # Dataset configuration
    train_data_path: Optional[List[str]] = None,
    valid_data_path: Optional[List[str]] = None,
    test_data_path: Optional[List[str]] = None,
    dataset_type: Optional[str] = None,
    image_folder: Optional[str] = None,
    tokenizer_model: Optional[str] = None,
    mock: bool = False,
    # Model configuration
    tensor_model_parallel_size: int = 4,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: Optional[int] = 1,
    expert_tensor_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    enable_recompute: bool = False,
    account_for_embedding_in_pipeline_split: bool = False,
    account_for_loss_in_pipeline_split: bool = False,
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
    use_null_tokenizer: bool = False,
    # Precision recipe
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # Freeze options
    pretrained_checkpoint: Optional[str] = None,
    freeze_language_model: bool = True,
    freeze_vision_model: bool = True,
    freeze_vision_projection: bool = False,
    # PEFT options
    peft: Optional[Union[str, PEFT]] = None,
    finetune_lr: Optional[float] = None,
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Create a fine-tuning configuration for Qwen3.5 VL models.

    Supports the dense (Qwen3.5-27B) and MoE (Qwen3.5-35B-A3B,
    Qwen3.5-122B-A10B, Qwen3.5-397B-A17B) variants.  The model
    architecture is automatically determined from ``hf_path`` via
    :class:`AutoBridge`.

    Args:
        hf_path: HuggingFace model path.
        dir: Base directory for logs and checkpoints.
        name: Name of the training run.
        train_data_path: Training data paths.
        valid_data_path: Validation data paths.
        test_data_path: Test data paths.
        dataset_type: One of ``"mock"``, ``"hf"``, ``"preloaded"``.
        image_folder: Path to image folder (for preloaded datasets).
        tokenizer_model: Path or HF name for the tokenizer/processor.
        mock: If *True*, equivalent to ``dataset_type="mock"``.
        tensor_model_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        pipeline_dtype: Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size: Virtual pipeline parallelism.
        context_parallel_size: Context parallelism degree.
        expert_model_parallel_size: Expert parallelism degree (MoE).
        expert_tensor_parallel_size: Expert tensor parallelism (MoE).
        sequence_parallel: Whether to use sequence parallelism.
        use_megatron_fsdp: Whether to use Megatron FSDP.
        enable_recompute: Whether to enable activation recomputation.
        account_for_embedding_in_pipeline_split: Account for embedding in PP split.
        account_for_loss_in_pipeline_split: Account for loss in PP split.
        train_iters: Total training iterations.
        global_batch_size: Global batch size.
        micro_batch_size: Micro batch size.
        seq_length: Sequence length.
        lr: Learning rate.
        min_lr: Minimum learning rate for cosine decay.
        lr_warmup_iters: Warmup iterations.
        lr_decay_iters: LR decay iterations (defaults to *train_iters*).
        eval_interval: Evaluation interval.
        save_interval: Checkpoint save interval.
        use_null_tokenizer: Use NullTokenizer instead of HuggingFace tokenizer.
        precision_config: Precision configuration (default: bf16 mixed).
        comm_overlap_config: Communication overlap configuration.
        pretrained_checkpoint: Path to a pretrained checkpoint.
        freeze_language_model: Freeze the language model weights.
        freeze_vision_model: Freeze the vision encoder weights.
        freeze_vision_projection: Freeze the vision projection weights.
        peft: PEFT configuration (``"lora"``, ``"dora"``, or a PEFT object).
        finetune_lr: Learning rate override for fine-tuning.
        wandb_project: W&B project name.
        wandb_entity: W&B entity name.
        wandb_exp_name: W&B experiment name.

    Returns:
        ConfigContainer ready for training.
    """
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = expert_tensor_parallel_size
    if not sequence_parallel and tensor_model_parallel_size > 1 and (expert_model_parallel_size or 1) > 1:
        sequence_parallel = True
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.freeze_language_model = freeze_language_model
    model_cfg.freeze_vision_model = freeze_vision_model
    model_cfg.freeze_vision_projection = freeze_vision_projection
    model_cfg.seq_length = seq_length

    if precision_config is None:
        precision_config = bf16_mixed()

    if account_for_embedding_in_pipeline_split:
        model_cfg.account_for_embedding_in_pipeline_split = True
    if account_for_loss_in_pipeline_split:
        model_cfg.account_for_loss_in_pipeline_split = True

    if enable_recompute:
        model_cfg.recompute_granularity = "full"
        model_cfg.recompute_method = "uniform"
        model_cfg.recompute_num_layers = 1

    model_cfg.validate_parallelism()

    # Optimizer and scheduler
    effective_lr = finetune_lr if finetune_lr is not None else lr
    if min_lr > effective_lr:
        min_lr = effective_lr * 0.1
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=effective_lr,
        min_lr=min_lr,
    )

    peft_config = _default_peft_config(peft)

    # Dataset selection
    _processor_model = tokenizer_model or hf_path
    _dataset_choice = dataset_type or ("mock" if mock else "hf")

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
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=100,
            manual_gc_eval=100,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=32,
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
        tokenizer=TokenizerConfig(
            tokenizer_type="NullTokenizer" if use_null_tokenizer else "HuggingFaceTokenizer",
            tokenizer_model=hf_path if not use_null_tokenizer else None,
            vocab_size=DEFAULT_NULL_TOKENIZER_VOCAB_SIZE if use_null_tokenizer else None,
        ),
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
