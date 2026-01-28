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
from megatron.core.distributed import DistributedDataParallelConfig
from typing_extensions import TypedDict, Unpack

from megatron.bridge.models.olmoe import OlMoEModelProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class OLMoECommonKwargs(TypedDict, total=False):
    """Typed options accepted by OLMoE family recipe helpers."""

    # Core identifiers
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
    # Model configuration
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: int
    sequence_parallel: bool
    # Recomputation
    recompute_granularity: str
    recompute_modules: Optional[List[str]]
    recompute_method: Optional[str]
    recompute_num_layers: Optional[int]
    apply_rope_fusion: bool
    # Training hyperparameters
    train_iters: int
    global_batch_size: int
    micro_batch_size: int
    seq_length: int
    lr: float
    min_lr: float
    lr_warmup_iters: int
    optimizer_type: str
    eval_interval: int
    save_interval: int
    # Precision / overlap configs
    precision_config: Optional[Union[MixedPrecisionConfig, str]]
    comm_overlap_config: Optional[CommOverlapConfig]


class OLMoEFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by OLMoE finetune recipe helpers."""

    # Core identifiers
    tokenizer_path: str
    dir: Optional[str]
    name: str
    # Model parallelism
    tensor_model_parallel_size: int
    pipeline_model_parallel_size: int
    pipeline_dtype: Optional[torch.dtype]
    virtual_pipeline_model_parallel_size: Optional[int]
    context_parallel_size: int
    expert_model_parallel_size: int
    sequence_parallel: bool
    # Recomputation
    recompute_granularity: str
    recompute_modules: Optional[List[str]]
    recompute_method: Optional[str]
    recompute_num_layers: Optional[int]
    apply_rope_fusion: bool
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


def _get_olmoe_pipeline_layout(pp_size: int, vp_size: int):
    """Get pipeline layout for OLMoE-7B based on PP and VP size."""
    # OLMoE has 16 layers
    map_pp_vp_to_layout = {
        (1, 1): None,
        (2, 1): [["embedding"] + ["decoder"] * 8, ["decoder"] * 8 + ["loss"]],
        (4, 1): [["embedding"] + ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
        (2, 2): [["embedding"] + ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4, ["decoder"] * 4 + ["loss"]],
    }
    if (pp_size, vp_size) not in map_pp_vp_to_layout:
        raise ValueError(
            f"Invalid PP and VP size: {pp_size} and {vp_size} to infer PP layout "
            f"for OLMoE (7B). Known PP and VP combinations: {map_pp_vp_to_layout.keys()}"
        )
    layout = map_pp_vp_to_layout[(pp_size, vp_size)]
    if layout is not None:
        layout = list([list(x) for x in layout])
    return layout


def olmoe_7b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for OLMoE-7B (7B total, ~1B active).

    Recommended parallelism: TP=1, PP=1, EP=8
    Uses precision-aware optimizer with bf16 gradients/moments.
    """
    cfg = _pretrain_common()

    # Model config - uses OlMoEModelProvider
    cfg.model = OlMoEModelProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        expert_model_parallel_size=8,
        sequence_parallel=False,
        recompute_granularity="selective",
        recompute_modules=None,
        recompute_method=None,
        recompute_num_layers=None,
    )

    # Pipeline layout
    cfg.model.pipeline_model_parallel_layout = _get_olmoe_pipeline_layout(1, 1)

    # Performance optimization knobs
    cfg.model.moe_permute_fusion = True

    # Tokenizer - uses NullTokenizer with model vocab_size
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.tokenizer_model = None
    cfg.tokenizer.vocab_size = cfg.model.vocab_size

    # Dataset config
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.seq_length = 4096
    cfg.dataset.num_workers = 8

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.train_iters = 500_000
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1
    cfg.train.eval_interval = 2000
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 5
    cfg.train.manual_gc_eval = 5

    # Optimizer
    cfg.scheduler.lr_warmup_iters = 2000
    cfg.scheduler.lr_decay_iters = cfg.train.train_iters
    cfg.optimizer.adam_eps = 1e-8

    # Precision-aware optimizer settings
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading) - already set in OlMoEModelProvider
    # cfg.model.recompute_granularity = "selective"
    # cfg.model.recompute_modules = None
    cfg.model.apply_rope_fusion = False  # Set to True for RoPE fusion (requires experimental flag)
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Mixed precision - OLMoE uses custom MixedPrecisionConfig (NOT "bf16_mixed" string)
    cfg.mixed_precision = MixedPrecisionConfig(
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        autocast_enabled=False,
        grad_reduce_in_fp32=False,
    )
    # FP8 settings (commented - enable if using FP8)
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Communication overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 2000
    cfg.checkpoint.async_save = False
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "no_shard"

    if cfg.model.apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    return cfg


def _model_config(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = None,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 8,
    sequence_parallel: bool = False,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    apply_rope_fusion: bool = False,
) -> OlMoEModelProvider:
    """
    Configure the OLMoE-7B model (7B total, ~1B active).

    Args:
        tensor_model_parallel_size: Degree of tensor model parallelism.
        pipeline_model_parallel_size: Degree of pipeline model parallelism.
        pipeline_dtype: Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size: Size of virtual pipeline parallelism.
        context_parallel_size: Degree of context parallelism.
        expert_model_parallel_size: Degree of expert model parallelism.
        sequence_parallel: Whether to use sequence parallelism.
        recompute_granularity: Recomputation granularity.
        recompute_modules: Modules to recompute.
        recompute_method: Recomputation method.
        recompute_num_layers: Number of layers to recompute.
        apply_rope_fusion: Whether to apply RoPE fusion.

    Returns:
        OlMoEModelProvider: Configuration for the OLMoE-7B model (7B total, ~1B active).
    """
    cfg = OlMoEModelProvider(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        # Recomputation
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
    )

    # Performance optimization knobs
    cfg.moe_permute_fusion = True
    if apply_rope_fusion:
        cfg.apply_rope_fusion = True

    # Pipeline parallelism configs. We infer PP layout from the provided PP and VP size
    pp_size = pipeline_model_parallel_size or 1
    vp_size = virtual_pipeline_model_parallel_size or 1
    layout = _get_olmoe_pipeline_layout(pp_size, vp_size)
    cfg.pipeline_model_parallel_layout = layout

    return cfg


def olmoe_7b_finetune_config(**user_kwargs: Unpack[OLMoEFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for OLMoE-7B (7B total, ~1B active).

    Default configuration: 1 node, 8 GPUs
    - LoRA/DoRA: TP=1, PP=1, EP=1, LR=1e-4
    - Full SFT: TP=1, PP=1, EP=8, lower LR (5e-6)

    See `_olmoe_finetune_common` for the full list of parameters.
    """
    peft_value = user_kwargs.get("peft", "lora")
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")

    # Convert string "none" to None for PEFT parameter
    if isinstance(peft_value, str) and peft_value.lower() == "none":
        peft_value = None

    recommended: OLMoEFinetuneKwargs = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "pipeline_dtype": torch.bfloat16,
        "virtual_pipeline_model_parallel_size": None,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8 if is_full_sft else 1,
        "sequence_parallel": False,
        "recompute_granularity": "selective",
        "apply_rope_fusion": False,
        "peft": peft_value,
        "finetune_lr": 5e-6 if is_full_sft else 1e-4,
    }
    kwargs: OLMoEFinetuneKwargs = {**recommended, **user_kwargs}
    return _olmoe_finetune_common(**kwargs)


def _olmoe_finetune_common(
    tokenizer_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: Optional[torch.dtype] = torch.bfloat16,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    sequence_parallel: bool = False,
    # Recomputation
    recompute_granularity: str = "selective",
    recompute_modules: Optional[List[str]] = None,
    recompute_method: Optional[str] = None,
    recompute_num_layers: Optional[int] = None,
    apply_rope_fusion: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: Optional[str] = None,
    peft: Optional[Union[str, PEFT]] = "lora",
    packed_sequence: bool = False,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int = 128,
    micro_batch_size: int = 1,
    seq_length: int = 4096,
    eval_interval: int = 50,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 1e-4,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: Optional[int] = None,
    # Precision / overlap
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    comm_overlap_config: Optional[CommOverlapConfig] = None,
    # W&B
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
) -> ConfigContainer:
    """
    Create a finetuning configuration for OLMoE-7B model (7B total, ~1B active).

    Args:
        tokenizer_path (str): Path to the tokenizer (HuggingFace tokenizer).
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        tensor_model_parallel_size (int): Degree of tensor model parallelism.
        pipeline_model_parallel_size (int): Degree of pipeline model parallelism.
        pipeline_dtype (Optional[torch.dtype]): Data type for pipeline parallelism.
        virtual_pipeline_model_parallel_size (Optional[int]): Size of virtual pipeline parallelism.
        context_parallel_size (int): Degree of context parallelism.
        expert_model_parallel_size (int): Degree of expert model parallelism.
        sequence_parallel (bool): Whether to use sequence parallelism.
        recompute_granularity (str): Recomputation granularity.
        recompute_modules (Optional[List[str]]): Modules to recompute.
        recompute_method (Optional[str]): Recomputation method.
        recompute_num_layers (Optional[int]): Number of layers to recompute.
        apply_rope_fusion (bool): Whether to apply RoPE fusion.
        pretrained_checkpoint (Optional[str]): Path to pretrained checkpoint.
        peft (Optional[Union[str, PEFT]]): PEFT configuration (e.g., "lora", "dora", or None for full SFT).
        packed_sequence (bool): Whether to use packed sequences.
        train_iters (int): Total number of training iterations.
        global_batch_size (int): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (int): Sequence length for training data.
        eval_interval (int): Interval for evaluation.
        save_interval (int): Interval for saving checkpoints.
        finetune_lr (float): Learning rate for finetuning.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of decay iterations for the learning rate.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        comm_overlap_config (Optional[CommOverlapConfig]): Communication overlap configuration.
        wandb_project (Optional[str]): Weights & Biases project name.
        wandb_entity (Optional[str]): Weights & Biases entity name.
        wandb_exp_name (Optional[str]): Weights & Biases experiment name.

    Returns:
        ConfigContainer: Configuration for finetuning.
    """
    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Create model config
    model_cfg = _model_config(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        pipeline_dtype=pipeline_dtype,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        sequence_parallel=sequence_parallel,
        recompute_granularity=recompute_granularity,
        recompute_modules=recompute_modules,
        recompute_method=recompute_method,
        recompute_num_layers=recompute_num_layers,
        apply_rope_fusion=apply_rope_fusion,
    )

    # Update seq_length in model config
    model_cfg.seq_length = seq_length

    # Optimizer and LR scheduler
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters or train_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        weight_decay=0.1,
    )

    # Set precision-aware optimizer settings similar to pretrain
    opt_cfg.use_precision_aware_optimizer = True
    opt_cfg.main_params_dtype = torch.float32
    opt_cfg.main_grads_dtype = torch.bfloat16
    opt_cfg.exp_avg_dtype = torch.bfloat16
    opt_cfg.exp_avg_sq_dtype = torch.bfloat16

    # PEFT config
    peft_config = default_peft_config(peft)

    # Precision config
    if precision_config is None:
        precision_config = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

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
        tokenizer_model=tokenizer_path,
        hf_tokenizer_kwargs={"trust_remote_code": True},
    )

    pad_seq_to_mult = context_parallel_size * 2 if packed_sequence and context_parallel_size > 1 else 1

    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=32,
            global_batch_size=global_batch_size,
            micro_batch_size=micro_batch_size,
            manual_gc=True,
            manual_gc_interval=5,
            manual_gc_eval=5,
        ),
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
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
            fully_parallel_save=True,
            async_save=False,
        ),
        rng=RNGConfig(seed=5678),
        peft=peft_config,
        comm_overlap=comm_overlap_config,
        mixed_precision=precision_config,
    )

    if apply_rope_fusion:
        cfg.dist.enable_megatron_core_experimental = True  # for rope fusion

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(
            tp_comm_overlap=False,
        )

    return cfg
