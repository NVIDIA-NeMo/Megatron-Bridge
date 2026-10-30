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

"""GB200 training recipes for Nemotron 3 Super.

For BF16 finetuning through ``scripts/training/run_recipe.py``, select
``--recipe nemotron_3_super_sft_64gpu_gb200_bf16_config --mode sft`` on 64 GPUs
or ``--recipe nemotron_3_super_peft_16gpu_gb200_bf16_config --mode lora`` on
16 GPUs, together with a pretrained checkpoint and an SFT dataset. These
recipes include the HybridEP NVL72 environment; no topology environment
variable overrides are needed when all ranks occupy one NVL72 domain.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def nemotron_3_super_pretrain_64gpu_gb200_bf16_config(
    *,
    hf_path: str | None = None,
    text_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Return the Nemotron 3 Super BF16 pretraining config for 64 GB200 GPUs.

    This is the convergence-oriented counterpart of the canonical 64-GPU
    performance recipe. It uses the same GB200 parallel layout, HybridEP
    dispatcher, CUDA graph scopes, and process environment while retaining
    natural expert routing, runtime validation, and checkpointing.

    Returns:
        GB200 BF16 pretraining configuration.
    """
    cfg = _pretrain_common()
    kwargs = {}
    if text_only:
        kwargs["text_only"] = True
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
        cfg.checkpoint.hf_trust_remote_code = True
    bridge = AutoBridge.from_hf_pretrained(hf_path or "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", **kwargs)

    cfg.model = bridge.get_model_config()

    cfg.tokenizer.tokenizer_model = hf_path or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    if hf_path:
        cfg.tokenizer.hf_tokenizer_kwargs = dict(cfg.tokenizer.hf_tokenizer_kwargs or {})
        cfg.tokenizer.hf_tokenizer_kwargs["trust_remote_code"] = trust_remote_code
        tokenizer_revision = bridge.hf_model_revision or revision
        if tokenizer_revision:
            cfg.tokenizer.hf_tokenizer_kwargs["revision"] = tokenizer_revision

    # Model parallelism, kernels, routing, and MTP.
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "fused"
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.use_te_rng_tracker = True
    # Reuse the source's shared MTP block for two prediction depths.
    cfg.model.mtp_num_layers = 2
    cfg.model.calculate_per_token_loss = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.init_method_std = 0.014
    cfg.model.apply_rope_fusion = True
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = 32
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.model.recompute_granularity = None

    # Dataset.
    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 1
    cfg.dataset.mmap_bin_files = False

    # Training schedule and batch size.
    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Validation.
    cfg.validation.eval_interval = 1000

    # Optimizer state and hyperparameters.
    cfg.optimizer.lr = 0.00045
    cfg.optimizer.min_lr = 4.5e-06
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-08
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.optimizer_offload_fraction = 0.0

    # Learning-rate and weight-decay schedule.
    cfg.scheduler.lr_warmup_iters = 333
    cfg.scheduler.lr_decay_iters = None
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "WSD"

    # Precision policy.
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # Checkpoint policy.
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.async_save = True

    # Gradient reduction and communication.
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False

    # GB200 NVL72 process environment.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }

    return cfg


def nemotron_3_super_sft_64gpu_gb200_bf16_config(
    *,
    hf_path: str | None = None,
    text_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Return the BF16 SFT config for 64 GB200 GPUs in one NVL72 domain.

    Uses TP8/EP64 and multi-node NVLink HybridEP, FP32 optimizer state and
    gradient reduction, synchronous saves, and numerical validation.
    Retains the H100 recipe's objective, batch size, and learning-rate policy.
    The allocation must place all expert-parallel ranks in one NVLink domain.

    Args:
        hf_path: Hugging Face model identifier or local source directory.
        text_only: Use only the language model of a multimodal source.
        revision: Immutable Hugging Face source revision.
        trust_remote_code: Allow custom code from the selected source.

    Returns:
        GB200 SFT configuration with its HybridEP environment included.
    """
    cfg = _sft_common()
    kwargs = {}
    if text_only:
        kwargs["text_only"] = True
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
        cfg.checkpoint.hf_trust_remote_code = True
    bridge = AutoBridge.from_hf_pretrained(hf_path or "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", **kwargs)

    cfg.model = bridge.get_model_config()

    cfg.tokenizer.tokenizer_model = hf_path or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    if hf_path:
        cfg.tokenizer.hf_tokenizer_kwargs = dict(cfg.tokenizer.hf_tokenizer_kwargs or {})
        cfg.tokenizer.hf_tokenizer_kwargs["trust_remote_code"] = trust_remote_code
        tokenizer_revision = bridge.hf_model_revision or revision
        if tokenizer_revision:
            cfg.tokenizer.hf_tokenizer_kwargs["revision"] = tokenizer_revision

    # Model parallelism, kernels, routing, and MTP.
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 2048
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.014
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    # Packed SFT masks are incompatible with Mamba CUDA graph capture.
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    # Reuse the source's shared MTP block for two prediction depths.
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.moe_router_force_load_balancing = False

    # Training schedule and batch size.
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    # Optimizer state and hyperparameters.
    cfg.optimizer.lr = 5e-06
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-08
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32

    # Learning-rate and weight-decay schedule.
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    # Precision policy.
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True

    # Checkpoint policy.
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.async_save = False
    cfg.checkpoint.exit_on_missing_checkpoint = True

    # Gradient reduction and communication.
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.check_for_large_grads = True

    # Logging.
    cfg.logger.log_interval = 10

    # Random seed.
    cfg.rng.seed = 1234

    # Numerical validation.
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    # GB200 NVL72 process environment.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }

    return cfg


def nemotron_3_super_peft_16gpu_gb200_bf16_config(
    peft_scheme: str | PEFT = "lora",
    *,
    hf_path: str | None = None,
    text_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Return the BF16 PEFT config for 16 GB200 GPUs in one NVL72 domain.

    Uses TP8/EP16 and multi-node NVLink HybridEP, FP32 optimizer state and
    gradient reduction, synchronous saves, and numerical validation.
    Retains the H100 recipe's adapters, objective, and learning-rate policy.
    The allocation must place all expert-parallel ranks in one NVLink domain.

    Args:
        peft_scheme: LoRA, DoRA, or a custom PEFT configuration.
        hf_path: Hugging Face model identifier or local source directory.
        text_only: Use only the language model of a multimodal source.
        revision: Immutable Hugging Face source revision.
        trust_remote_code: Allow custom code from the selected source.

    Returns:
        GB200 PEFT configuration with its HybridEP environment included.
    """
    cfg = _peft_common()
    kwargs = {}
    if text_only:
        kwargs["text_only"] = True
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
        cfg.checkpoint.hf_trust_remote_code = True
    bridge = AutoBridge.from_hf_pretrained(hf_path or "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16", **kwargs)

    cfg.model = bridge.get_model_config()

    cfg.tokenizer.tokenizer_model = hf_path or "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
    if hf_path:
        cfg.tokenizer.hf_tokenizer_kwargs = dict(cfg.tokenizer.hf_tokenizer_kwargs or {})
        cfg.tokenizer.hf_tokenizer_kwargs["trust_remote_code"] = trust_remote_code
        tokenizer_revision = bridge.hf_model_revision or revision
        if tokenizer_revision:
            cfg.tokenizer.hf_tokenizer_kwargs["revision"] = tokenizer_revision

    # Model parallelism, kernels, routing, and MTP.
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 2048
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.014
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    # Packed SFT masks are incompatible with Mamba CUDA graph capture.
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    # Reuse the source's shared MTP block for two prediction depths.
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True
    cfg.model.moe_router_force_load_balancing = False

    # Native language and Mamba adapter targets.
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
    else:
        cfg.peft = LoRA(
            target_modules=mamba_target_modules,
            dim=32,
            alpha=32,
            dropout=0.0,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
        )

    # Training schedule and batch size.
    cfg.train.global_batch_size = 16

    # Optimizer state and hyperparameters.
    cfg.optimizer.lr = 0.0001
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-08
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Learning-rate and weight-decay schedule.
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    # Precision policy.
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = True

    # Checkpoint policy.
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.async_save = False
    cfg.checkpoint.exit_on_missing_checkpoint = True

    # Gradient reduction and communication.
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.check_for_large_grads = True

    # Logging.
    cfg.logger.log_interval = 10

    # Random seed.
    cfg.rng.seed = 1234

    # Numerical validation.
    cfg.rerun_state_machine.check_for_nan_in_loss = True

    # GB200 NVL72 process environment.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }

    return cfg


__all__ = [
    "nemotron_3_super_pretrain_64gpu_gb200_bf16_config",
    "nemotron_3_super_sft_64gpu_gb200_bf16_config",
    "nemotron_3_super_peft_16gpu_gb200_bf16_config",
]
