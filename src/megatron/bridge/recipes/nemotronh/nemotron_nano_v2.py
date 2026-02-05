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


import torch
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models import NemotronNanoModelProvider9Bv2
from megatron.bridge.models.nemotronh import (
    NemotronNanoModelProvider9Bv2,
    NemotronNanoModelProvider12Bv2,
)
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    ConfigContainer,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class NemotronNanoV2CommonKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron Nano v2 recipe helper functions."""

    # Core identifiers
    model_provider: NemotronNanoModelProvider9Bv2 | NemotronNanoModelProvider12Bv2
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


class NemotronNanoV2FinetuneKwargs(NemotronNanoV2CommonKwargs, total=False):
    """Typed options accepted by Nemotron Nano v2 finetuning recipe helper functions."""

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


def nemotron_nano_9b_v2_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron Nano 9B v2.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=2, PP=1, SP=True.
    """
    cfg = _pretrain_common()

    # Model config - uses NemotronNanoModelProvider9Bv2
    cfg.model = NemotronNanoModelProvider9Bv2(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings (already set in model provider above)
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 10

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

    # Memory saving (recompute & offloading)
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

    # Communication overlap
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


def nemotron_nano_12b_v2_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron Nano 12B v2.

    This recipe is designed for single-node training (1 node).
    Default parallelism: TP=4, PP=1, SP=True.

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    cfg = _pretrain_common()

    # Model config - uses NemotronNanoModelProvider12Bv2
    cfg.model = NemotronNanoModelProvider12Bv2(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
    )

    # Parallel settings (already set in model provider above)
    cfg.model.pipeline_model_parallel_layout = None

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 8192
    cfg.dataset.num_workers = 8

    # Training config
    cfg.train.train_iters = 1_168_251
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 10

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

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - FP8 with current scaling
    cfg.mixed_precision = "nanov2_bf16_with_fp8_current_scaling_mixed"
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

    # Communication overlap - disabled by default for 12B (FP8 compatibility)
    cfg.comm_overlap = None

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


def nemotron_nano_9b_v2_finetune_config(**user_kwargs: Unpack[NemotronNanoV2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Nemotron Nano 9B v2.

    Default configuration: 8 nodes, 64 GPUs
    - LoRA/DoRA: TP=2, PP=1, LR=1e-4
    - Full SFT: TP=2, PP=1, LR=5e-6
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    from megatron.bridge.recipes.nemotronh.nemotronh import _nemotronh_finetune_common

    recommended_kwargs: NemotronNanoV2FinetuneKwargs = {
        "model_provider": NemotronNanoModelProvider9Bv2,
        "tensor_parallelism": 2 if is_full_sft else 1,
        "pipeline_parallelism": 1,
        "sequence_parallelism": is_full_sft,
        "seq_length": 2048,  # Default seq_length for Nano v2
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "min_lr": 1e-6 if is_full_sft else 1e-5,
        "precision_config": "bf16_mixed",
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_12>"},  # Correct eos token for Nemotron Nano v2
    }
    combined_kwargs: NemotronNanoV2FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base", **combined_kwargs)


def nemotron_nano_12b_v2_finetune_config(**user_kwargs: Unpack[NemotronNanoV2FinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Nemotron Nano 12B v2.

    Default configuration: 8 nodes, 64 GPUs
    - LoRA/DoRA: TP=4, PP=1, LR=1e-4
    - Full SFT: TP=4, PP=1, LR=5e-6

    Note: Uses FP8 precision by default. Communication overlap is disabled by default.
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    from megatron.bridge.recipes.nemotronh.nemotronh import _nemotronh_finetune_common

    recommended_kwargs: NemotronNanoV2FinetuneKwargs = {
        "model_provider": NemotronNanoModelProvider12Bv2,
        "tensor_parallelism": 4 if is_full_sft else 1,
        "pipeline_parallelism": 1,
        "sequence_parallelism": is_full_sft,
        "seq_length": 2048,  # Default seq_length for Nano v2
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "precision_config": "bf16_mixed",
        "hf_tokenizer_kwargs": {"eos_token": "<SPECIAL_12>"},  # Correct eos token for Nemotron Nano v2
    }
    combined_kwargs: NemotronNanoV2FinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotronh_finetune_common(tokenizer_model="nvidia/NVIDIA-Nemotron-Nano-12B-v2-Base", **combined_kwargs)


__all__ = [
    # Pretrain configs
    "nemotron_nano_9b_v2_pretrain_config",
    "nemotron_nano_12b_v2_pretrain_config",
    # Finetune configs
    "nemotron_nano_9b_v2_finetune_config",
    "nemotron_nano_12b_v2_finetune_config",
]
