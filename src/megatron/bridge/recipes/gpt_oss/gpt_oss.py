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
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.recipes.utils.tokenizer_utils import DEFAULT_NULL_TOKENIZER_VOCAB_SIZE
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DatasetProvider,
    DistributedDataParallelConfig,
    FinetuningDatasetConfig,
    GPTDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class GPTOSSCommonKwargs(TypedDict, total=False):
    """Typed options accepted by GPT-OSS recipe helpers."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Dataset configuration
    data_paths: Optional[List[str]]
    data_args_path: Optional[str]
    train_data_path: Optional[List[str]]
    valid_data_path: Optional[List[str]]
    test_data_path: Optional[List[str]]
    per_split_data_args_path: Optional[str]
    mock: bool
    # Provide dataset directly
    dataset: Optional[Union[GPTDatasetConfig, FinetuningDatasetConfig, DatasetProvider]]
    # Model configuration
    num_layers: int  # for ci testing
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: Optional[int]
    sequence_parallel: bool
    use_megatron_fsdp: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
    cp_comm_type: Optional[str]
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
    # Checkpointing
    pretrained_checkpoint: Optional[str]


class GPTOSSFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by GPT-OSS finetune recipe helpers."""

    # Core identifiers
    hf_path: str
    dir: Optional[str]
    name: str
    # Model parallelism
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: Optional[int]
    sequence_parallel: bool
    use_megatron_fsdp: bool
    # Finetuning specifics
    pretrained_checkpoint: Optional[str]
    peft: Optional[Union[str, PEFT]]
    packed_sequence: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: Optional[int]
    micro_batch_size: int
    seq_length: int
    finetune_lr: float
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]
    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]


def gpt_oss_20b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for GPT-OSS 20B variant.

    Recommended parallelism: TP=2, PP=4, EP=4
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("openai/gpt-oss-20b").to_megatron_provider(load_weights=False)

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # When CP>1 is used (e.g. via CLI override), a2a is required for TE attention backends with learnable softmax.
    cfg.model.cp_comm_type = "a2a"

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False

    # When CP>1 (e.g. via CLI override), per-token loss avoids issues on CP ranks; harmless when CP=1.
    cfg.model.calculate_per_token_loss = True

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"  # Default
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16  # Number of SMs for hybridep backend

    # Training config
    cfg.train.train_iters = 1000000
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - uses "bf16_mixed" from _pretrain_common
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False  # Pad router for FP8 alignment

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap (default None, can pass CommOverlapConfig for advanced overlap)
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False  # GPT-OSS default

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config (matches _pretrain_common)
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False  # Must be False when calculate_per_token_loss=True (used for CP>1)
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    return cfg


def gpt_oss_120b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for GPT-OSS 120B variant.

    Recommended parallelism: TP=2, PP=4, EP=16
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("openai/gpt-oss-120b").to_megatron_provider(load_weights=False)

    # Tokenizer - uses NullTokenizer by default
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = DEFAULT_NULL_TOKENIZER_VOCAB_SIZE

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    # Parallelism settings (MoE-specific)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16  # Larger EP for 120B
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    # When CP>1 is used (e.g. via CLI override), a2a is required for TE attention backends with learnable softmax.
    cfg.model.cp_comm_type = "a2a"

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = False
    cfg.model.account_for_loss_in_pipeline_split = False
    # When CP>1 (e.g. via CLI override), per-token loss avoids issues on CP ranks; harmless when CP=1.
    cfg.model.calculate_per_token_loss = True

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config (DIFFERENT from _pretrain_common)
    cfg.train.train_iters = 1000000
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 2000

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # GPT-OSS uses native

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap (default None, can pass CommOverlapConfig for advanced overlap)
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False  # Must be False when calculate_per_token_loss=True (used for CP>1)
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    return cfg


def gpt_oss_20b_finetune_config(**user_kwargs: Unpack[GPTOSSFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for GPT-OSS 20B variant.

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, EP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, EP=8, lower LR (5e-6)
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended: GPTOSSFinetuneKwargs = {
        "hf_path": "openai/gpt-oss-20b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 8 if is_full_sft else 1,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    kwargs: GPTOSSFinetuneKwargs = {**recommended, **user_kwargs}
    return _gpt_oss_finetune_common(**kwargs)


def gpt_oss_120b_finetune_config(**user_kwargs: Unpack[GPTOSSFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for GPT-OSS 120B variant.

    Default configuration: 2 nodes, 16 GPUs total
    - LoRA/DoRA: TP=1, PP=1, EP=8, LR=1e-4
    - Full SFT: TP=1, PP=4, EP=8, lower LR (5e-6)
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended: GPTOSSFinetuneKwargs = {
        "hf_path": "openai/gpt-oss-120b",
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 4 if is_full_sft else 1,
        "expert_model_parallel_size": 8,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    kwargs: GPTOSSFinetuneKwargs = {**recommended, **user_kwargs}
    return _gpt_oss_finetune_common(**kwargs)


def _gpt_oss_finetune_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: Optional[str] = None,
    peft: Optional[Union[str, PEFT]] = "lora",
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 50,
    save_interval: int = 250,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: Optional[int] = None,
    # Precision / overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # W&B
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Common finetuning configuration for GPT-OSS models using a given HuggingFace path."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = 1  # Required: bias in MoE only supported when ETP==1
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.seq_length = seq_length
    if context_parallel_size > 1:
        model_cfg.calculate_per_token_loss = True
        model_cfg.cp_comm_type = "a2a"  # only a2a cp is supported for sink attention.

    # Optimizer and LR scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )

    # PEFT config
    peft_config = default_peft_config(peft)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=1,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Always use HF tokenizer for finetuning
    tokenizer_cfg = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=hf_path,
    )

    pad_seq_to_mult = context_parallel_size * 2 if packed_sequence and context_parallel_size > 1 else 1

    return ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
        ),
        validation=ValidationConfig(
            eval_interval=eval_interval,
            eval_iters=32,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(check_for_nan_in_grad=True, use_megatron_fsdp=use_megatron_fsdp),
        dataset=default_squad_config(seq_length, packed_sequence, pad_seq_to_mult),
        logger=logger_cfg,
        tokenizer=tokenizer_cfg,
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )
