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
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


NEMOTRON_3_SUPER_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def nemotron_3_super_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Super (120B-A12B LatentMoE).

    This is a Latent MoE model with Multi-Token Prediction (MTP). Default parallelism:
    - TP=8, PP=1, EP=16, SP=True

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Super.
    """
    cfg = _pretrain_common()

    # Model Configuration (LatentMoE with MTP) — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_SUPER_HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Parallelism Settings
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192

    # Tokenizer (--tokenizer-model)
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Dataset Configuration
    cfg.dataset.seq_length = 8192
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 1
    cfg.dataset.mmap_bin_files = False

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"

    # Training Configuration. Bounded 100-step cohort schedule, matching the
    # other H100 verification recipes (GBS 1024 / MBS 1 / 100 steps).
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Validation
    cfg.validation.eval_interval = 1000

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph (TE impl + partial scopes: ~40% throughput gain over disabled)
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3

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

    # Mixed Precision. Gradients reduce in BF16 to bound the gradient buffer;
    # grad_reduce_in_fp32 is a MixedPrecisionConfig field that setup() copies
    # onto the DDP config, so it must be set here rather than on cfg.ddp alone.
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # Optimizer hyperparameters
    cfg.optimizer.lr = 4.5e-4
    cfg.optimizer.min_lr = 4.5e-6
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.scheduler.lr_warmup_iters = 40
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "WSD"

    # Checkpoint Configuration
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.async_save = True

    # DDP Configuration. Overlap stays off so full-model checkpoint saves are
    # ordered after all DDP collectives.
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = False

    # Reduced-memory optimizer state.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.model.init_method_std = 0.014
    cfg.model.apply_rope_fusion = False
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def _nemotron_3_super_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return the convergence-oriented Nemotron 3 Super config for 64 H100 GPUs.

    The public hardware-agnostic ``nemotron_3_super_pretrain_config`` alias
    exposes this configuration. Its execution layout matches the canonical
    H100 performance recipe while it retains natural routing, runtime checks,
    and checkpointing for real-data training evidence.

    Returns:
        BF16 pretraining configuration for 64 H100 GPUs.
    """
    cfg = nemotron_3_super_pretrain_16gpu_h100_bf16_config()

    # The measured H100 layout avoids TP communication while using PP to fit
    # the dense layers and EP to shard the experts across all 64 GPUs.
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.overlap_p2p_comm = False
    cfg.model.batch_p2p_comm = True

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    # Match the canonical GPT-OSS 120B H100 benchmark shape. DP=1 for
    # TP1/PP2/EP32, so this yields 1,280 microbatches per optimizer step.
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 1

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_deepep_num_sms = None
    cfg.model.moe_hybridep_num_sms = None
    # Keep HybridEP inputs static without its per-layer metadata all-reduce.
    # At this shape, 1.10x capacity gives each expert 10% headroom over the
    # 5,632-token mean. The fixed expert padding keeps downstream collective
    # shapes consistent under natural routing without over-sizing activations.
    cfg.model.moe_hybridep_pad_uneven_dispatch_inputs = False
    cfg.model.moe_expert_capacity_factor = 1.10
    cfg.model.moe_pad_expert_input_to_capacity = True
    cfg.model.moe_shared_expert_overlap = False
    # Forced routing changes the training objective and remains benchmark-only.
    cfg.model.moe_router_force_load_balancing = False

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.apply_rope_fusion = True
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False
    # Keep the provider-owned vocabulary shape identical between real-data
    # convergence and mock-data performance runs.
    cfg.tokenizer.use_tokenizer_vocab_size = False
    cfg.model.recompute_granularity = "selective"
    # Fixed-capacity dispatch keeps MoE replay collectives shape-stable while
    # full MoE recompute provides the measured H100 activation headroom.
    cfg.model.recompute_modules = ["layernorm", "moe_act", "moe", "core_attn"]

    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.main_params_dtype = torch.float16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.optimizer_offload_fraction = 0.0
    cfg.optimizer.overlap_cpu_optimizer_d2h_h2d = False

    # Keep DDP collectives ordered with PP and expert communication.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    # Retain enough watchdog headroom for first-step bring-up of the 120B model.
    cfg.dist.distributed_timeout_minutes = 15

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


# =============================================================================
# SFT Config
# =============================================================================


def nemotron_3_super_sft_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron 3 Super (120B-A12B LatentMoE).

    Default parallelism: TP=8, PP=1, EP=16, SP=True

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3 Super SFT.
    """
    cfg = _sft_common()

    # Model config — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_SUPER_HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Parallelism settings
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

    # Training-specific model overrides
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "fused"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.init_method_std = 0.014
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

    # MTP Settings (HF config has num_nextn_predict_layers=1 for the shared block;
    # mtp_num_layers=2 controls forward-pass repetitions with mtp_use_repeated_layer)
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

    # Reduced-memory optimizer state.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

    # Training Configuration. Bounded 100-step cohort schedule, matching the
    # other H100 SFT verification recipes (GBS 32 / MBS 1 / 100 steps).
    cfg.train.train_iters = 100
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    # Mixed Precision. Gradients reduce in BF16 to bound the gradient buffer;
    # grad_reduce_in_fp32 is a MixedPrecisionConfig field that setup() copies
    # onto the DDP config, so it must be set here rather than on cfg.ddp alone.
    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False

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

    # DDP config. Overlap stays off so full-model checkpoint saves are ordered
    # after all DDP collectives.
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def nemotron_3_super_sft_16gpu_h100_bf16_32k_config() -> ConfigContainer:
    """Return the 16-H100 BF16 32K full-SFT configuration.

    Long-context SFT uses a distinct PP8/CP2 layout rather than the default
    SFT topology.
    """
    cfg = nemotron_3_super_sft_16gpu_h100_bf16_config()
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    # The final stage also owns the output projection and loss. Keep four
    # decoder layers there and distribute 12 layers to each earlier stage.
    cfg.model.num_layers_in_last_pipeline_stage = 4
    cfg.model.context_parallel_size = 2
    cfg.model.expert_model_parallel_size = 2
    cfg.model.seq_length = 32768
    cfg.model.cp_comm_type = "a2a"
    # CP SFT needs per-token accounting because a CP rank can receive no
    # supervised labels after masking.
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.calculate_per_token_loss = True
    cfg.dataset.seq_length = 32768
    cfg.train.global_batch_size = 2
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False, batch_p2p_comm=False)
    cfg.ddp.average_in_collective = False
    # With uneven PP stages, asynchronous DDP collectives can race the
    # cross-stage tied-embedding reduction and PP send/recv operations.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    # Keep the complete process environment visible on the recipe and PP
    # sends/receives ordered for the long-context PP8 schedule.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


# =============================================================================
# PEFT Config
# =============================================================================


def nemotron_3_super_peft_1gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Nemotron 3 Super (120B-A12B LatentMoE).

    Default parallelism: TP=1, PP=1, EP=1, SP=True

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3 Super PEFT.
    """
    cfg = _peft_common()

    # Model config — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_SUPER_HF_MODEL_ID).to_megatron_provider(load_weights=False)

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
    cfg.model.init_method_std = 0.014
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

    # MTP Settings (HF config has num_nextn_predict_layers=1 for the shared block;
    # mtp_num_layers=2 controls forward-pass repetitions with mtp_use_repeated_layer)
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
    cfg.tokenizer.tokenizer_model = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

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

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        # HybridEP topology for this recipe.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 1,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


def nemotron_3_super_peft_16gpu_h100_bf16_config(
    peft_scheme: str | PEFT = "lora",
) -> ConfigContainer:
    """Return the 16-H100 BF16 PEFT configuration."""
    cfg = nemotron_3_super_peft_1gpu_h100_bf16_config(peft_scheme)
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 16
    cfg.train.global_batch_size = 16
    # Keep checkpoint saves ordered after all DDP collectives.
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
    }
    return cfg


__all__ = [
    "nemotron_3_super_peft_1gpu_h100_bf16_config",
    "nemotron_3_super_peft_16gpu_h100_bf16_config",
    "nemotron_3_super_pretrain_16gpu_h100_bf16_config",
    "nemotron_3_super_sft_16gpu_h100_bf16_32k_config",
    "nemotron_3_super_sft_16gpu_h100_bf16_config",
]
