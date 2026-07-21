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


import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import (
    default_openmathinstruct2_config,
    default_peft_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_3_5_NANO_HF_MODEL_ID = "nvidia/nemotron-nano-3.5-ea2"
NEMOTRON_3_5_NANO_HF_REVISION = "68f54a60ad3c68abaeb585d57a29f1da5021665e"  # pragma: allowlist secret


def nemotron_3_5_nano_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the bounded pre-training config for 16 H100 GPUs.

    This recipe follows the model-card convergence contract: 100 steps at
    sequence length 4096 and global batch size 1024 with natural MoE routing.
    The H100-specific execution policy uses micro batch size 1, a narrow CUDA
    graph scope, and selective recompute of MoE and layernorm modules.

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3.5 Nano.
    """
    cfg = _pretrain_common()

    # Model Configuration (Hybrid Mamba + MoE with MTP) — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(
        NEMOTRON_3_5_NANO_HF_MODEL_ID,
        revision=NEMOTRON_3_5_NANO_HF_REVISION,
    ).to_megatron_provider(load_weights=False)

    # Parallelism Settings — H100 BF16 perf preset (also valid on Blackwell, see docstring)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 4096

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = NEMOTRON_3_5_NANO_HF_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": NEMOTRON_3_5_NANO_HF_REVISION, "use_fast": True}

    # Dataset Configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.random_seed = 1234
    cfg.dataset.num_workers = 1
    cfg.dataset.mmap_bin_files = False

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = False

    # Training Configuration — micro_batch_size=1 matches the H100 perf preset
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0
    cfg.rng.seed = 1234

    # Validation
    cfg.validation.eval_interval = 0
    cfg.validation.eval_iters = 0

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph — H100 BF16 preset uses the narrower scope to keep memory bounded
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba"]
    cfg.model.cuda_graph_warmup_steps = 3

    # Activation Recompute — H100 BF16 preset selectively recomputes MoE + layernorm
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe", "layernorm"]

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.use_te_rng_tracker = True

    # MTP Settings (HF config has num_nextn_predict_layers=1 for the shared block;
    # mtp_num_layers=2 controls forward-pass repetitions with mtp_use_repeated_layer)
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True

    # Mixed Precision
    cfg.mixed_precision = "bf16_mixed"

    # Optimizer hyperparameters
    cfg.optimizer.lr = 3.0e-4
    cfg.optimizer.min_lr = 3.0e-5
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.scheduler.lr_warmup_iters = 40
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.lr_decay_style = "cosine"

    # Communication Overlap
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # Checkpoint Configuration
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.load = None
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.async_save = True

    # DDP Configuration
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False

    cfg.model.init_method_std = 0.0173
    cfg.model.apply_rope_fusion = False
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True

    cfg.logger.log_interval = 1
    cfg.logger.log_throughput = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


# =============================================================================
# SFT Config
# =============================================================================


def nemotron_3_5_nano_sft_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron 3.5 Nano (30B-A3B Hybrid MoE + MTP).

    Default parallelism: TP=1, PP=1, EP=8, SP=True.

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3.5 Nano SFT.
    """
    cfg = _sft_common()

    # Model config — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(
        NEMOTRON_3_5_NANO_HF_MODEL_ID,
        revision=NEMOTRON_3_5_NANO_HF_REVISION,
    ).to_megatron_provider(load_weights=False)

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 2048

    # Training-specific model overrides
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.0173
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"

    # CUDA Graph disabled — packed-sequence SFT passes explicit attention masks that
    # are incompatible with CUDA graph capture/replay in Mamba layers.
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []

    # MTP Settings
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True

    # Optimizer overrides
    cfg.optimizer.lr = 5e-6
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    # Tokenizer
    cfg.tokenizer.tokenizer_model = NEMOTRON_3_5_NANO_HF_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": NEMOTRON_3_5_NANO_HF_REVISION, "use_fast": True}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.async_save = True

    # Logger config
    cfg.logger.log_interval = 10

    # RNG config
    cfg.rng.seed = 1234

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


# =============================================================================
# PEFT Config
# =============================================================================


def nemotron_3_5_nano_peft_8gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Nemotron 3.5 Nano (30B-A3B Hybrid MoE + MTP).

    Default parallelism: TP=1, PP=1, EP=1, SP=True.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3.5 Nano PEFT.
    """
    cfg = _peft_common()

    # Model config — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(
        NEMOTRON_3_5_NANO_HF_MODEL_ID,
        revision=NEMOTRON_3_5_NANO_HF_REVISION,
    ).to_megatron_provider(load_weights=False)

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 2048

    # Training-specific model overrides
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.0173
    cfg.model.use_fused_weighted_squared_relu = True
    cfg.model.calculate_per_token_loss = True

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"

    # CUDA Graph disabled — packed-sequence PEFT passes explicit attention masks that
    # are incompatible with CUDA graph capture/replay in Mamba layers.
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []

    # MTP Settings
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True

    # PEFT config - Nemotron uses Mamba-specific target modules
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

    # Optimizer overrides
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "cosine"

    # Tokenizer
    cfg.tokenizer.tokenizer_model = NEMOTRON_3_5_NANO_HF_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": NEMOTRON_3_5_NANO_HF_REVISION, "use_fast": True}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.async_save = True

    # Logger config
    cfg.logger.log_interval = 10

    # RNG config
    cfg.rng.seed = 1234

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 1,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


# =============================================================================
# OpenMathInstruct-2 dataset wrappers (SFT + PEFT)
# =============================================================================
# Thin wrappers that swap the dataset config to the generic
# ``default_openmathinstruct2_config`` builder (no chat-template channel split,
# unlike gpt-oss's ``thinking_packed`` variant which depends on gpt-oss-specific
# tokens). Bumps seq_length to 4096 because OpenMathInstruct-2 solutions are
# longer than SQuAD QA pairs.


def nemotron_3_5_nano_sft_16gpu_h100_bf16_openmathinstruct2_packed_config() -> ConfigContainer:
    """SFT config for Nemotron 3.5 Nano on nvidia/OpenMathInstruct-2 (packed sequences).

    Inherits every model/optimizer/checkpoint knob from
    ``nemotron_3_5_nano_sft_16gpu_h100_bf16_config()`` and only overrides
    ``seq_length`` and ``cfg.dataset``.
    """
    cfg = nemotron_3_5_nano_sft_16gpu_h100_bf16_config()
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset = default_openmathinstruct2_config(seq_length=seq_length, enable_offline_packing=True)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def nemotron_3_5_nano_peft_8gpu_h100_bf16_openmathinstruct2_packed_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """PEFT/LoRA config for Nemotron 3.5 Nano on nvidia/OpenMathInstruct-2 (packed sequences).

    Inherits LoRA target modules and optimizer settings from
    ``nemotron_3_5_nano_peft_8gpu_h100_bf16_config()`` and only overrides
    ``seq_length`` and ``cfg.dataset``.
    """
    cfg = nemotron_3_5_nano_peft_8gpu_h100_bf16_config(peft_scheme=peft_scheme)
    seq_length = 4096
    cfg.model.seq_length = seq_length
    cfg.dataset = default_openmathinstruct2_config(seq_length=seq_length, enable_offline_packing=True)
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 1,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


__all__ = [
    "nemotron_3_5_nano_peft_8gpu_h100_bf16_config",
    "nemotron_3_5_nano_peft_8gpu_h100_bf16_openmathinstruct2_packed_config",
    "nemotron_3_5_nano_pretrain_16gpu_h100_bf16_config",
    "nemotron_3_5_nano_sft_16gpu_h100_bf16_config",
    "nemotron_3_5_nano_sft_16gpu_h100_bf16_openmathinstruct2_packed_config",
]
