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
from megatron.core.distributed import DistributedDataParallelConfig
from typing_extensions import TypedDict, Unpack

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedInitConfig,
    FinetuningDatasetConfig,
    LoggerConfig,
    RNGConfig,
    TokenizerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig


class Qwen3NextCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3-Next recipe helpers."""

    # Core identifiers
    hf_path: str
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
    expert_model_parallel_size: int | None
    expert_tensor_parallel_size: int
    sequence_parallel: bool
    use_megatron_fsdp: bool
    enable_recompute: bool
    account_for_embedding_in_pipeline_split: bool
    account_for_loss_in_pipeline_split: bool
    # MTP support
    mtp_num_layers: int | None
    mtp_loss_scaling_factor: float | None
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
    use_null_tokenizer: bool
    # Precision / overlap configs
    precision_config: MixedPrecisionConfig | str | None
    comm_overlap_config: CommOverlapConfig | None
    # Performance optimization knobs
    moe_flex_dispatcher_backend: str | None
    disable_jit_fuser: bool


class Qwen3NextFinetuneKwargs(Qwen3NextCommonKwargs, total=False):
    """Typed options accepted by Qwen3-Next finetuning recipe helper functions."""

    # Core finetuning options
    pretrained_checkpoint: str | None
    peft: str | PEFT | None
    packed_sequence: bool

    # Dataset configuration
    dataset_path: str | None

    # Training params
    finetune_lr: float

    # W&B logging
    wandb_project: str | None
    wandb_entity: str | None
    wandb_exp_name: str | None


def qwen3_next_80b_a3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-Next 80B-A3B.

    Recommended parallelism: TP=1, PP=4, EP=8.
    Note: Qwen3-Next supports Multi-Token Prediction (MTP) with mtp_num_layers and mtp_loss_scaling_factor.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct").to_megatron_provider(
        load_weights=False
    )

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False  # Qwen3-Next specific setting

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Multi-Token Prediction (MTP) settings - Qwen3-Next specific
    cfg.model.mtp_num_layers = 1  # Number of MTP layers (0 to disable)
    cfg.model.mtp_loss_scaling_factor = 0.1  # Loss scaling factor for MTP

    # MoE Token Dispatcher settings
    # Note: moe_token_dispatcher_type may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_token_dispatcher_type = "alltoall"  # Options: alltoall, allgather, flex
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # Options: None, deepep, hybridep
    cfg.model.moe_hybridep_num_sms = 16  # Number of SMs for hybridep backend

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

    # Scheduler config - Qwen3-Next specific
    cfg.scheduler.no_weight_decay_cond_type = "qwen3_next"

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections (includes MoE-specific kernels)
    cfg.model.attention_backend = None
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (recompute & offloading)
    cfg.model.recompute_granularity = "selective"  # Qwen3-Next uses selective recompute
    cfg.model.recompute_modules = ["layernorm", "moe", "moe_act"]  # Qwen3-Next specific modules
    cfg.model.recompute_method = None  # Not used for selective recompute
    cfg.model.recompute_num_layers = None  # Not used for selective recompute
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Communication overlap (default None, can pass CommOverlapConfig for advanced overlap)
    # cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # Uncomment to enable
    # cfg.comm_overlap.delay_wgrad_compute = False  # Delay wgrad compute for overlap
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False  # MoE-specific: Overlap EP communication
    # Note: moe_shared_expert_overlap may be overridden by apply_flex_dispatcher_backend at the end
    cfg.model.moe_shared_expert_overlap = False  # Overlap shared expert computation

    # Checkpoint config
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _pretrain_common. To override them, set them here.Ex:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)

    return cfg


def qwen3_next_80b_a3b_finetune_config(**user_kwargs: Unpack[Qwen3NextFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3-Next 80B-A3B.

    Default configuration: 8 nodes, 64 GPUs total
    - Full SFT: TP=1, PP=1, EP=8, LR=5e-6 (with recompute)
    """
    # Check if user is doing full SFT or PEFT (matches NeMo2 behavior)
    peft_value = user_kwargs.get("peft", None)
    is_full_sft = peft_value is None or (isinstance(peft_value, str) and peft_value.lower() == "none")
    if not is_full_sft:
        raise ValueError("Only full SFT is supported for Qwen3-Next at the moment")

    # Check if user enables sequence packing since it is not supported for Qwen3-Next at the moment
    packed_sequence = user_kwargs.get("packed_sequence", False)
    if packed_sequence:
        raise ValueError("Sequence packing is not supported for Qwen3-Next at the moment")

    recommended_kwargs: Qwen3NextFinetuneKwargs = {
        "hf_path": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "tensor_model_parallel_size": 1,
        "sequence_parallel": False,
        "pipeline_model_parallel_size": 2,
        "pipeline_dtype": torch.bfloat16,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 8,
        "peft": peft_value,
        "finetune_lr": 5e-6,
        "min_lr": 5e-6,
        "enable_recompute": True,
        "packed_sequence": False,  # Sequence packing is not supported for Qwen3-Next
    }
    combined_kwargs: Qwen3NextFinetuneKwargs = {**recommended_kwargs, **user_kwargs}
    config = _qwen3_next_finetune_common(**combined_kwargs)

    return config


def _qwen3_next_finetune_common(
    hf_path: str,
    dir: str | None = None,
    name: str = "default",
    # Dataset configuration
    dataset_path: str | None = None,
    # Core model configuration
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    pipeline_dtype: torch.dtype | None = torch.bfloat16,
    virtual_pipeline_model_parallel_size: int | None = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int | None = 8,
    expert_tensor_parallel_size: int = 1,
    sequence_parallel: bool = False,
    use_megatron_fsdp: bool = False,
    enable_recompute: bool = False,
    # Finetuning-specific params
    pretrained_checkpoint: str | None = None,
    peft: str | PEFT | None = None,
    packed_sequence: bool = True,
    # Training params
    train_iters: int = 1000,
    global_batch_size: int | None = None,  # Auto-select based on packed_sequence if None
    micro_batch_size: int = 1,
    seq_length: int = 2048,
    eval_interval: int = 30,
    save_interval: int = 50,
    # Optimizer
    finetune_lr: float = 5e-6,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 50,
    lr_decay_iters: int | None = None,  # Let config handle this
    # W&B logging
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_exp_name: str | None = None,
    # Precision
    precision_config: MixedPrecisionConfig | str | None = "bf16_mixed",
    comm_overlap_config: CommOverlapConfig | None = None,
    moe_flex_dispatcher_backend: str | None = None,
    disable_jit_fuser: bool | None = None,
) -> ConfigContainer:
    """Common finetuning configuration for Qwen3-Next model."""

    # Setup directories
    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    # Auto-select global_batch_size based on packed_sequence
    if global_batch_size is None:
        global_batch_size = 8 if packed_sequence else 64

    if dataset_path is not None:
        dataset = FinetuningDatasetConfig(
            dataloader_type="single",
            dataset_root=dataset_path,
            seq_length=seq_length,
            seed=5678,
            memmap_workers=1,
            max_train_samples=None,
            packed_sequence_specs=None,  # TODO: add packed_sequence_specs if packed_sequence is True, currently not supported for Qwen3-Next
            dataset_kwargs=None,
            do_validation=True,
            do_test=True,
        )
    else:
        pad_seq_to_mult = context_parallel_size * 2 if packed_sequence and context_parallel_size > 1 else 1
        dataset = default_squad_config(seq_length, packed_sequence, pad_seq_to_mult)

    # Create model config
    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)
    model_cfg.tensor_model_parallel_size = tensor_model_parallel_size
    model_cfg.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_cfg.pipeline_dtype = pipeline_dtype
    model_cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    model_cfg.context_parallel_size = context_parallel_size
    model_cfg.sequence_parallel = sequence_parallel
    model_cfg.expert_model_parallel_size = expert_model_parallel_size
    model_cfg.expert_tensor_parallel_size = expert_tensor_parallel_size
    model_cfg.seq_length = seq_length

    # Add recompute settings for memory optimization
    if enable_recompute:
        model_cfg.recompute_granularity = "selective"
        model_cfg.recompute_modules = ["layernorm", "moe", "moe_act"]
        model_cfg.recompute_method = None
        model_cfg.recompute_num_layers = None

    # Performance optimization knobs
    model_cfg.moe_permute_fusion = True
    model_cfg.moe_grouped_gemm = True
    apply_flex_dispatcher_backend(model_cfg, moe_flex_dispatcher_backend)

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters,
        max_lr=finetune_lr,
        min_lr=min_lr,
        adam_beta2=0.98,
    )
    scheduler_cfg.no_weight_decay_cond_type = "qwen3_next"

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

    # If user does not specify, check if we are on Blackwell.
    if disable_jit_fuser is None:
        disable_jit_fuser = torch.cuda.get_device_properties(0).major == 10

    return ConfigContainer(
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
        optimizer=opt_cfg,
        scheduler=scheduler_cfg,
        dist=DistributedInitConfig(disable_jit_fuser=disable_jit_fuser),
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,  # Not supported for Megatron FSDP for now, need to be set to False if using Megatron FSDP
            data_parallel_sharding_strategy="optim_grads_params",  # For Megatron FSDP only
            use_distributed_optimizer=True,
            use_megatron_fsdp=use_megatron_fsdp,  # need use_distributed_optimizer=True
        ),
        dataset=dataset,
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
