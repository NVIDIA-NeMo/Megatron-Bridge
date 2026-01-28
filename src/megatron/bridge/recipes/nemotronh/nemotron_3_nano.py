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
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.nemotronh import Nemotron3NanoProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
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


def nemotron_3_nano_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Nano (30B-A3B MoE).

    This is a MoE (Mixture of Experts) model with the following default parallelism:
    - TP=4, PP=1, EP=8, SP=True
    - DeepEP enabled for MoE token dispatch

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Nano.
    """
    cfg = _pretrain_common()

    # Model Configuration (MoE)
    cfg.model = Nemotron3NanoProvider(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
        seq_length=8192,
    )

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Dataset Configuration
    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    # Parallelism Settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training Configuration
    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 3072
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory Saving (recompute & offloading)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # =========================================================================
    # FP8 & MXFP8 (Mixed Precision Settings)
    # =========================================================================
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # FP8 settings (disabled by default, uncomment to enable)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer Precision Settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Optimizer hyperparameters
    cfg.optimizer.lr = 1.6e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.min_lr = 1.6e-5
    cfg.scheduler.warmup_iters = 333

    # Communication Overlap
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint Configuration
    # Paths are set in _pretrain_common by default. Override here if needed:
    # cfg.checkpoint.load = "path/to/load"
    # cfg.checkpoint.save = "path/to/save"
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP Configuration
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    cfg.model.init_method_std = 0.0173
    cfg.model.apply_rope_fusion = False
    cfg.model.async_tensor_model_parallel_allreduce = True
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True

    return cfg


class Nemotron3NanoFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Nemotron 3 Nano finetune recipe helpers."""

    # Core identifiers
    model_provider: Nemotron3NanoProvider
    dir: Optional[str]
    name: str
    # Model parallelism
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_parallelism_dtype: Optional[torch.dtype]
    virtual_pipeline_parallelism: Optional[int]
    context_parallel_size: int
    sequence_parallelism: bool
    expert_tensor_parallelism: int
    expert_model_parallelism: int
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
    # MoE
    enable_deepep: bool
    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]


def nemotron_3_nano_finetune_config(**user_kwargs: Unpack[Nemotron3NanoFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Nemotron 3 Nano.

    Default configuration:
    - LoRA/DoRA: TP=1, PP=1, EP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, EP=8, lower LR (5e-6)
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    recommended_kwargs: Nemotron3NanoFinetuneKwargs = {
        "model_provider": Nemotron3NanoProvider,
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_parallelism_dtype": torch.bfloat16,
        "context_parallel_size": 1,
        "sequence_parallelism": False,
        "expert_tensor_parallelism": 1,
        "expert_model_parallelism": 8,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
        "enable_deepep": True,
        "precision_config": "bf16_mixed",
    }
    combined_kwargs: Nemotron3NanoFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    return _nemotron_3_nano_finetune_common(**combined_kwargs)


def _nemotron_3_nano_finetune_common(
    model_provider: type[Nemotron3NanoProvider],
    dir: Optional[str] = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_parallelism_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_parallelism: Optional[int] = None,
    context_parallel_size: int = 1,
    sequence_parallelism: bool = True,
    expert_tensor_parallelism: int = 1,
    expert_model_parallelism: int = 1,
    # Finetuning-specific params
    pretrained_checkpoint: Optional[str] = None,
    peft: Optional[Union[str, PEFT]] = "lora",
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 500,
    save_interval: int = 200,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: Optional[int] = None,
    # Precision / overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = "bf16_mixed",
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # MoE
    enable_deepep: bool = True,
    # W&B
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """Common finetuning configuration for Nemotron 3 Nano models.

    Args:
        model_provider: The model provider class for the Nemotron 3 Nano variant.
        dir: Base directory for saving logs and checkpoints.
        name: Name of the finetuning run.
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_parallelism_dtype: Data type for pipeline parallelism.
        virtual_pipeline_parallelism: Size of virtual pipeline parallelism.
        context_parallel_size: Degree of context parallelism.
        sequence_parallelism: Whether to use sequence parallelism.
        expert_tensor_parallelism: Degree of expert tensor parallelism.
        expert_model_parallelism: Degree of expert model parallelism.
        pretrained_checkpoint: Path to the pretrained checkpoint.
        peft: PEFT configuration (e.g., "lora", "dora", "none" or PEFT object).
        packed_sequence: Whether to use packed sequences.
        train_iters: Total number of training iterations.
        global_batch_size: Global batch size for training.
        micro_batch_size: Micro batch size for training.
        seq_length: Sequence length for training data.
        eval_interval: Interval (in iterations) between evaluations.
        save_interval: Interval (in iterations) between checkpoints.
        finetune_lr: Learning rate for finetuning.
        min_lr: Minimum learning rate.
        lr_warmup_iters: Number of warmup iterations for the learning rate.
        lr_decay_iters: Number of iterations for learning rate decay.
        precision_config: Precision configuration for the model.
        comm_overlap_config: Communication overlap configuration for the model.
        enable_deepep: Whether to enable DeepEP for MoE.
        wandb_project: Weights & Biases project name.
        wandb_entity: Weights & Biases entity name.
        wandb_exp_name: Weights & Biases experiment name.

    Returns:
        ConfigContainer: Configuration for finetuning.
    """

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Configure the model
    model_cfg = model_provider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_parallelism_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_parallelism,
        context_parallel_size=context_parallel_size,
        sequence_parallel=sequence_parallelism,
        expert_tensor_parallel_size=expert_tensor_parallelism,
        expert_model_parallel_size=expert_model_parallelism,
        apply_rope_fusion=False,
        async_tensor_model_parallel_allreduce=True,
        attention_backend="fused",
        gradient_accumulation_fusion=True,
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
        seq_length=seq_length,
        calculate_per_token_loss=True,
    )

    if enable_deepep:
        model_cfg.moe_token_dispatcher_type = "flex"
        model_cfg.moe_shared_expert_overlap = False
        model_cfg.moe_flex_dispatcher_backend = "deepep"

    # Optimizer and LR scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=0.1,
        max_lr=finetune_lr,
        min_lr=min_lr,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        lr_decay_style="cosine",
    )

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    peft_config = default_peft_config(peft, target_modules=mamba_target_modules)

    # Logger
    logger_cfg = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    pad_seq_to_mult = (
        model_cfg.context_parallel_size * 2 if packed_sequence and model_cfg.context_parallel_size > 1 else 1
    )

    # Config Container
    cfg = ConfigContainer(
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
        optimizer=opt_config,
        scheduler=scheduler,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            use_distributed_optimizer=True,
        ),
        dataset=default_squad_config(seq_length, packed_sequence, pad_seq_to_mult),
        logger=logger_cfg,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer", tokenizer_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            dist_ckpt_strictness="log_all",
            ckpt_assume_constant_structure=True,
        ),
        rng=RNGConfig(seed=1234),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    return cfg
