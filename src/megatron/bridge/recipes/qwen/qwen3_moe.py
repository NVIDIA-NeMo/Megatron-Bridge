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

from megatron.bridge import AutoBridge
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
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, get_mixed_precision_config


class Qwen3MoeCommonKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 MoE recipe helpers."""

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
    moe_flex_dispatcher_backend: str | None


class Qwen3MoeFinetuneKwargs(TypedDict, total=False):
    """Typed options accepted by Qwen3 MoE finetuning recipe helper functions.

    This is separate from Qwen3MoeCommonKwargs to avoid confusion - finetuning
    uses SQuAD dataset by default, not the data path fields.
    """

    # Core identifiers
    dir: Optional[str]
    name: str

    # Finetuning-specific
    pretrained_checkpoint: Optional[str]
    peft: Union[str, PEFT, None]
    packed_sequence: bool

    # Training hyperparameters
    train_iters: int
    global_batch_size: Optional[int]
    micro_batch_size: int
    seq_length: Optional[int]
    eval_interval: int
    save_interval: int

    # Optimizer
    finetune_lr: Optional[float]
    min_lr: float
    lr_warmup_iters: int
    lr_decay_iters: Optional[int]

    # W&B logging
    wandb_project: Optional[str]
    wandb_entity: Optional[str]
    wandb_exp_name: Optional[str]

    # Precision
    precision_config: Optional[Union[MixedPrecisionConfig, str]]


def qwen3_30b_a3b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-30B-A3B MoE.

    Recommended parallelism: TP=4, PP=2, EP=4.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-30B-A3B").to_megatron_provider(load_weights=False)

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-30B-A3B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings (MoE-specific: includes expert_model_parallel_size)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

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
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading) - ENABLED for 30B MoE
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8
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
    cfg.model.moe_shared_expert_overlap = False  # Overlap shared expert computation

    # Checkpoint config (paths set in _pretrain_common)
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

    return cfg


def qwen3_235b_a22b_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3-235B-A22B MoE.

    Recommended parallelism: TP=4, PP=16, CP=2, EP=8.
    Note: Uses account_for_embedding_in_pipeline_split and account_for_loss_in_pipeline_split
    for proper layer distribution in pipeline parallelism.
    """
    cfg = _pretrain_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-235B-A22B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-235B-A22B"

    # Dataset config - mock data by default
    cfg.dataset.blend = None  # Pass the path to the dataset here if not using mock data, along with weight. Ex: (["path/to/data1"], 0.2), [("path/to/data2", 0.8)]
    cfg.dataset.num_workers = 8

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096
    cfg.model.init_method_std = 0.02

    # Pipeline split accounting
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16

    # Training config
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100

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
    cfg.model.cross_entropy_fusion_impl = "te"

    # Memory saving (recompute & offloading)
    # Enable if needed for memory optimization
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _pretrain_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default
    # cfg.mixed_precision.fp8 = None  # not enabled
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

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

    return cfg


def qwen3_30b_a3b_finetune_config(**user_kwargs: Unpack[Qwen3MoeFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3-30B-A3B MoE.

    Default configuration: 1 node, 8 GPUs, LoRA
    - LoRA (default): TP=4, PP=1, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
    - DoRA: TP=4, PP=1, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
    - Full SFT (peft=None): TP=4, PP=2, EP=4, LR=5e-6, SP=True

    Matches NeMo2 recipe at nemo/collections/llm/recipes/qwen3_30b_a3b.py
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _qwen3_moe_finetune_common(hf_path="Qwen/Qwen3-30B-A3B", **user_kwargs)

    # Model-specific parallelism settings (match NeMo pattern)
    if is_full_sft:
        config.model.tensor_model_parallel_size = 4
        config.model.expert_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 2
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        config.model.tensor_model_parallel_size = 4
        config.model.expert_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 1
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True

        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
            config.peft.target_modules = ["linear_qkv", "linear_proj"]
        else:
            config.peft = peft

    return config


def qwen3_235b_a22b_finetune_config(**user_kwargs: Unpack[Qwen3MoeFinetuneKwargs]) -> ConfigContainer:
    """Return a finetuning config for Qwen3-235B-A22B MoE.

    Default configuration: 8 nodes (LoRA) or 16 nodes (Full SFT), 8 GPUs per node
    - LoRA (default): TP=4, PP=4, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
      Total: 64 GPUs (8 nodes)
    - DoRA: TP=4, PP=4, EP=4, LR=1e-4, dim=8, alpha=16, target_modules=['linear_qkv', 'linear_proj']
      Total: 64 GPUs (8 nodes)
    - Full SFT (peft=None): TP=4, PP=16, EP=4, LR=5e-6, SP=True
      Total: 64 GPUs (8 nodes)

    Matches NeMo2 recipe at nemo/collections/llm/recipes/qwen3_235b_a22b.py

    Note: Uses account_for_embedding_in_pipeline_split and account_for_loss_in_pipeline_split
    for proper layer distribution in pipeline parallelism.
    """
    peft = user_kwargs.pop("peft", "lora")
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    # Auto-select LR if not specified
    finetune_lr = user_kwargs.get("finetune_lr")
    if finetune_lr is None:
        finetune_lr = 5e-6 if is_full_sft else 1e-4
        user_kwargs["finetune_lr"] = finetune_lr

    # Build base config
    config = _qwen3_moe_finetune_common(hf_path="Qwen/Qwen3-235B-A22B", **user_kwargs)

    # Enable pipeline split accounting (required for 235B model)
    config.model.account_for_embedding_in_pipeline_split = True
    config.model.account_for_loss_in_pipeline_split = True

    # Model-specific parallelism settings (match NeMo pattern)
    if is_full_sft:
        config.model.tensor_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 16
        config.model.expert_model_parallel_size = 4
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True
        config.peft = None
    else:
        # PEFT (LoRA, DoRA, or custom)
        config.model.tensor_model_parallel_size = 4
        config.model.pipeline_model_parallel_size = 4
        config.model.expert_model_parallel_size = 4
        config.model.expert_tensor_parallel_size = 1
        config.model.sequence_parallel = True

        if isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
            config.peft = default_peft_config(peft)
            config.peft.dim = 8
            config.peft.alpha = 16
            config.peft.target_modules = ["linear_qkv", "linear_proj"]
        else:
            config.peft = peft

    return config


def _qwen3_moe_finetune_common(
    hf_path: str,
    dir: Optional[str] = None,
    name: str = "default",
    # Finetuning-specific
    pretrained_checkpoint: Optional[str] = None,
    packed_sequence: bool = False,
    # Training hyperparameters
    train_iters: int = 100,
    global_batch_size: Optional[int] = None,
    micro_batch_size: int = 1,
    seq_length: Optional[int] = None,
    eval_interval: int = 50,
    save_interval: int = 100,
    # Optimizer
    finetune_lr: Optional[float] = None,
    min_lr: float = 0.0,
    lr_warmup_iters: int = 10,
    lr_decay_iters: Optional[int] = None,
    # W&B logging
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_exp_name: Optional[str] = None,
    # Precision
    precision_config: Optional[Union[MixedPrecisionConfig, str]] = None,
    moe_flex_dispatcher_backend: Optional[str] = None,
) -> ConfigContainer:
    """
    Create a finetuning configuration for Qwen3 MoE models using a given HuggingFace path.

    Args:
        hf_path (str): HuggingFace model path (e.g., "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B").
        dir (Optional[str]): Base directory for saving logs and checkpoints.
        name (str): Name of the finetuning run.
        pretrained_checkpoint (Optional[str]): Path to pretrained checkpoint to load.
        packed_sequence (bool): Whether to use packed sequences for training efficiency.
        train_iters (int): Total number of training iterations.
        global_batch_size (Optional[int]): Global batch size for training.
        micro_batch_size (int): Micro batch size for training.
        seq_length (Optional[int]): Sequence length for training data.
        eval_interval (int): Evaluation interval.
        save_interval (int): Checkpoint save interval.
        finetune_lr (Optional[float]): Learning rate for finetuning.
        min_lr (float): Minimum learning rate for cosine decay.
        lr_warmup_iters (int): Number of warmup iterations for the learning rate.
        lr_decay_iters (Optional[int]): Number of iterations over which to decay the LR.
        wandb_project (Optional[str]): Weights & Biases project name.
        wandb_entity (Optional[str]): Weights & Biases entity name.
        wandb_exp_name (Optional[str]): Weights & Biases experiment name.
        precision_config (Optional[Union[MixedPrecisionConfig, str]]): Precision configuration for the model.
        moe_flex_dispatcher_backend (str | None): Token dispatcher type [deepep, hybridep].
    Returns:
        ConfigContainer: Configuration for finetuning.
    """
    # Default sequence length for finetuning
    if seq_length is None:
        seq_length = 2048 if packed_sequence else 4096

    # Default global batch size
    if global_batch_size is None:
        global_batch_size = 32

    base_output_dir = dir if dir is not None else os.path.join(os.getcwd(), "nemo_experiments")
    run_output_dir = os.path.join(base_output_dir, name)
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    tensorboard_dir = os.path.join(run_output_dir, "tb_logs")

    bridge = AutoBridge.from_hf_pretrained(hf_path)
    model_cfg = bridge.to_megatron_provider(load_weights=False)

    # Precision configuration
    if precision_config is None:
        precision_config = bf16_mixed()
    elif isinstance(precision_config, str):
        precision_config = get_mixed_precision_config(precision_config)

    # Sequence length
    model_cfg.seq_length = seq_length
    model_cfg.cross_entropy_fusion_impl = "te"

    apply_flex_dispatcher_backend(model_cfg, moe_flex_dispatcher_backend)

    # Optimizer and scheduler
    opt_config, scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=lr_warmup_iters,
        lr_decay_iters=lr_decay_iters if lr_decay_iters is not None else train_iters,
        max_lr=finetune_lr if finetune_lr is not None else 1e-4,
        min_lr=min_lr,
    )

    pad_seq_to_mult = (
        model_cfg.context_parallel_size * 2 if packed_sequence and model_cfg.context_parallel_size > 1 else 1
    )
    # Dataset configuration (SQuAD by default)
    dataset_config = default_squad_config(
        seq_length=seq_length, packed_sequence=packed_sequence, pad_seq_to_mult=pad_seq_to_mult
    )

    # W&B logger configuration
    logger_config = LoggerConfig(
        log_interval=10,
        tensorboard_dir=tensorboard_dir,
        log_timers_to_tensorboard=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_exp_name=wandb_exp_name,
    )

    # Config Container
    cfg = ConfigContainer(
        model=model_cfg,
        train=TrainingConfig(
            train_iters=train_iters,
            eval_interval=eval_interval,
            eval_iters=10,
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
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            use_distributed_optimizer=True,
        ),
        dataset=dataset_config,
        logger=logger_config,
        tokenizer=TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_path,
        ),
        checkpoint=CheckpointConfig(
            save_interval=save_interval,
            save=checkpoint_dir,
            load=checkpoint_dir,
            pretrained_checkpoint=pretrained_checkpoint,
            ckpt_format="torch_dist",
            fully_parallel_save=True,
        ),
        rng=RNGConfig(seed=5678),  # Different seed for finetuning
        mixed_precision=precision_config,
    )

    return cfg
