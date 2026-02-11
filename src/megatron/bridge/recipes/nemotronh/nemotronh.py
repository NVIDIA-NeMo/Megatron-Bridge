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

from megatron.bridge.models import NemotronHModelProvider
from megatron.bridge.models.nemotronh import (
    NemotronHModelProvider4B,
    NemotronHModelProvider8B,
    NemotronHModelProvider47B,
    NemotronHModelProvider56B,
)
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
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
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class NemotronHCommonKwargs(TypedDict, total=False):
    """Typed options accepted by NemotronH recipe helper functions."""

    # Core identifiers
    model_provider: type[NemotronHModelProvider]
    tokenizer_model: str | None
    dir: str | None
    name: str
    # Dataset configuration
    data_paths: list[str] | None
    data_args_path: str | None
    train_data_path: list[str] | None
    valid_data_path: list[str] | None
    test_data_path: list[str] | None
    per_split_data_args_path: str | None
    mock: bool
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: torch.dtype | None
    virtual_pipeline_model_parallel_size: int | None
    context_parallel_size: int
    sequence_parallel: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: int | None
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    # CommOverlap setting
    enable_default_comm_overlap: bool


class NemotronHFinetuneKwargs(NemotronHCommonKwargs, total=False):
    """Typed options accepted by NemotronH finetuning recipe helper functions."""

    # Core finetuning options
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Training params
    finetune_lr: float

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None


def nemotronh_4b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for NemotronH 4B.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=1, PP=1, SP=False.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = NemotronHModelProvider4B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
    )

    # Parallel settings
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 10

    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.train.manual_gc_eval = True

    # Optimizer
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = False

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - bf16_mixed
    cfg.mixed_precision = "bf16_mixed"
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap - enabled by default
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )

    # Checkpoint config
    cfg.checkpoint.save_interval = 10
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def nemotronh_8b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for NemotronH 8B.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=2, PP=1, SP=True.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = NemotronHModelProvider8B(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 10

    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.train.manual_gc_eval = True

    # Optimizer
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = False

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - bf16_mixed
    cfg.mixed_precision = "bf16_mixed"
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap - enabled by default
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )

    # Checkpoint config
    cfg.checkpoint.save_interval = 10
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def nemotronh_47b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for NemotronH 47B.

    This recipe is designed for single-node training (1 node with 8 GPUs).
    Default parallelism: TP=8, PP=1, SP=True.

    Note: Uses FP8 precision by default.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = NemotronHModelProvider47B(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 10

    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.train.manual_gc_eval = True

    # Optimizer
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = False

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - FP8 with current scaling
    cfg.mixed_precision = "nemotron_h_bf16_with_fp8_current_scaling_mixed"
    # FP8 settings (commented - already enabled via precision string above)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap - enabled by default
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )

    # Checkpoint config
    cfg.checkpoint.save_interval = 10
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (DIFFERENT from _pretrain_common)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def nemotronh_56b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for NemotronH 56B.

    This recipe is designed for single-node training (1 node with 8 GPUs).
    Default parallelism: TP=8, PP=1, SP=True.

    Note: Uses FP8 precision by default.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = NemotronHModelProvider56B(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 10

    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.train.manual_gc_eval = True

    # Optimizer
    cfg.scheduler.lr_warmup_iters = 2000

    cfg.logger.log_timers_to_tensorboard = False

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - FP8 with current scaling
    cfg.mixed_precision = "nemotron_h_bf16_with_fp8_current_scaling_mixed"
    # FP8 settings (commented - already enabled via precision string above)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap - enabled by default
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )

    # Checkpoint config
    cfg.checkpoint.save_interval = 10
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    return cfg


def nemotronh_4b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 4B.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider4B,
        "tensor_parallelism": 1,
        "pipeline_parallelism": 1,
        "sequence_parallelism": False,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "precision_config": "bf16_mixed",
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-4B-Base-8K", **combined_kwargs)


def nemotronh_8b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 8B.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider8B,
        "tensor_parallelism": 2 if is_full_sft else 1,
        "pipeline_parallelism": 1,
        "sequence_parallelism": is_full_sft,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "precision_config": "bf16_mixed",
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-8B-Base-8K", **combined_kwargs)


def nemotronh_47b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 47B.

    Default configuration:
    - LoRA/DoRA: TP=4, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=1, LR=5e-6

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider47B,
        "tensor_parallelism": 8 if is_full_sft else 4,
        "pipeline_parallelism": 2 if is_full_sft else 1,
        "sequence_parallelism": is_full_sft,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-47B-Base-8K", **combined_kwargs)


def nemotronh_56b_finetune_config(**user_kwargs: Unpack[NemotronHFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for NemotronH 56B.

    Default configuration:
    - LoRA/DoRA: TP=4, PP=1, LR=1e-4
    - Full SFT: TP=8, PP=1, LR=5e-6

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: NemotronHFinetuneKwargs = {
        "model_provider": NemotronHModelProvider56B,
        "tensor_parallelism": 8 if is_full_sft else 4,
        "pipeline_parallelism": 1,
        "sequence_parallelism": is_full_sft,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_11>"},  # Correct eos token for Nemotron H
    }
    combined_kwargs: NemotronHFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/Nemotron-H-56B-Base-8K", **combined_kwargs)


def _nemotronh_finetune_common(
    model_provider: type[NemotronHModelProvider],
    tokenizer_model: str | None = None,
    dir: str | None = None,
    name: str = "default",
    # Model configuration
    tensor_parallelism: int = 1,
    pipeline_parallelism: int = 1,
    pipeline_parallelism_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_parallelism: int | None = None,
    context_parallelism: int = 1,
    sequence_parallelism: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = "lora",
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 8192,
    eval_interval: int = 50,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 1e-5,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    # Tokenizer kwargs (for model-specific tokenizer settings)
    hf_tokenizer_kwargs: dict | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for NemotronH and Nemotron Nano v2 models.

    Args:
        model_provider: The model provider class for the specific NemotronH or Nemotron Nano v2 variant.
        tokenizer_model: HuggingFace tokenizer model name.
        dir: Base directory for saving logs and checkpoints.
        name: Name of the finetuning run.
        tensor_parallelism: Degree of tensor model parallelism.
        pipeline_parallelism: Degree of pipeline model parallelism. Default: 1.
        pipeline_parallelism_dtype: Data type for pipeline parallelism. Default: torch.bfloat16.
        virtual_pipeline_parallelism: Size of virtual pipeline parallelism.
        context_parallelism: Degree of context parallelism. Default: 1.
        sequence_parallelism: Whether to use sequence parallelism.
        pretrained_checkpoint: Path to pretrained checkpoint to load from.
        peft: PEFT configuration (e.g., "lora", "dora") or PEFT object. None for full SFT. Default: "lora".
        packed_sequence: Whether to use packed sequences. Default: True.
        train_iters: Total number of training iterations. Default: 1000.
        global_batch_size: Global batch size. Default: 128.
        micro_batch_size: Micro batch size. Default: 1.
        seq_length: Sequence length. Default: 8192.
        eval_interval: Evaluation interval in iterations. Default: 50.
        save_interval: Checkpoint save interval in iterations. Default: 50.
        finetune_lr: Learning rate for finetuning. Default: 1e-4.
        min_lr: Minimum learning rate. Default: 1e-5.
        lr_warmup_iters: Number of warmup iterations. Default: 50.
        lr_decay_iters: Number of LR decay iterations.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity name.
        wandb_exp_name: Weights & Biases experiment name.
        precision_config: Precision configuration.
        comm_overlap_config: Communication overlap configuration.
        hf_tokenizer_kwargs: Additional kwargs for HuggingFace tokenizer (e.g., {"eos_token": "<SPECIAL_12>"}).

    Returns:
        ConfigContainer: Configuration for finetuning.

    Note:
        - 4B model: TP=1, SP=False, BF16 mixed precision
        - 8B model: TP=2 (full SFT) or TP=1 (LoRA), SP=True (full SFT), BF16 mixed precision
        - 9B Nano v2: TP=2 (full SFT) or TP=1 (LoRA), SP=True (full SFT), BF16 mixed precision
        - 12B Nano v2: TP=4 (full SFT) or TP=1 (LoRA), SP=True (full SFT), BF16 mixed precision
        - 47B model: TP=8 (full SFT) or TP=4 (LoRA), SP=True (full SFT), FP8 precision
        - 56B model: TP=8 (full SFT) or TP=4 (LoRA), SP=True (full SFT), FP8 precision
        - Uses SQuAD dataset format for finetuning
    """
    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Create model config
    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_parallelism,
        pipeline_model_parallel_size=pipeline_parallelism,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallelism,
        sequence_parallel=sequence_parallelism,
        seq_length=seq_length,
    )

    # Optimizer and scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
    )

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    peft_config = default_peft_config(peft, target_modules=mamba_target_modules)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Tokenizer config
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=tokenizer_model,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )

    pad_seq_to_mult = (
        model_cfg.context_parallel_size * 2 if packed_sequence and model_cfg.context_parallel_size > 1 else 1
    )

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=10,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
            use_distributed_optimizer=True,
        ),
        dataset=default_squad_config(seq_length, packed_sequence, pad_seq_to_mult),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg


__all__ = [
    # Pretrain configs
    "nemotronh_4b_pretrain_config",
    "nemotronh_8b_pretrain_config",
    "nemotronh_47b_pretrain_config",
    "nemotronh_56b_pretrain_config",
    # Finetune configs
    "nemotronh_4b_finetune_config",
    "nemotronh_8b_finetune_config",
    "nemotronh_47b_finetune_config",
    "nemotronh_56b_finetune_config",
]
