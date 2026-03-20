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

from megatron.bridge import AutoBridge
from megatron.bridge.models.nemotronh import Nemotron3SuperDebugProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer


NEMOTRON_3_SUPER_HF_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"


def nemotron_3_super_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Super (120B-A12B LatentMoE).

    This is a Latent MoE model with Multi-Token Prediction (MTP). Default parallelism:
    - TP=4, PP=1, EP=8, SP=True

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Super.
    """
    cfg = _pretrain_common()

    # Model Configuration (LatentMoE with MTP) — derived from HF config via AutoBridge
    cfg.model = AutoBridge.from_hf_pretrained(NEMOTRON_3_SUPER_HF_MODEL_ID).to_megatron_provider(load_weights=False)

    # Parallelism Settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
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

    # Training Configuration
    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 3072
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

    # Mixed Precision
    cfg.mixed_precision = "nemotron_3_super_bf16_with_nvfp4_mixed"

    # Optimizer hyperparameters
    cfg.optimizer.lr = 4.5e-4
    cfg.optimizer.min_lr = 4.5e-6
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.scheduler.lr_warmup_iters = 333
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.lr_decay_style = "WSD"

    # Checkpoint Configuration
    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.async_save = True

    # DDP Configuration
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    # TODO(liding): why others recipes set this to True? and can work with calc_per_token_loss?
    cfg.ddp.average_in_collective = False

    cfg.model.init_method_std = 0.014
    cfg.model.apply_rope_fusion = False
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.use_fused_weighted_squared_relu = True

    return cfg


# =============================================================================
# Finetune Config (handles both full SFT and PEFT)
# =============================================================================


def nemotron_3_super_finetune_config(
    peft: str | PEFT | None = "lora",
) -> ConfigContainer:
    """Return a finetuning config for Nemotron 3 Super (120B-A12B LatentMoE).

    Default parallelism:
    - LoRA/DoRA: TP=1, PP=1, EP=1, SP=True, LR=1e-4
    - Full SFT: TP=1, PP=1, EP=8, SP=True, LR=5e-6

    Args:
        peft: PEFT scheme - "lora", "dora", a custom PEFT instance, or None for full SFT.

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3 Super finetuning.
    """
    is_full_sft = peft is None or (isinstance(peft, str) and peft.lower() == "none")

    if is_full_sft:
        cfg = _sft_common()
    else:
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
    cfg.model.expert_model_parallel_size = 8 if is_full_sft else 1
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
    cfg.model.moe_router_force_load_balancing = True

    # CUDA Graph (TE impl + partial scopes: ~40% throughput gain over disabled)
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3

    # MTP Settings (HF config has num_nextn_predict_layers=1 for the shared block;
    # mtp_num_layers=2 controls forward-pass repetitions with mtp_use_repeated_layer)
    cfg.model.mtp_num_layers = 2
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.use_te_rng_tracker = True

    # PEFT config - Nemotron uses Mamba-specific target modules
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if is_full_sft:
        cfg.peft = None
    elif isinstance(peft, str) and peft.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft, target_modules=mamba_target_modules)
    elif isinstance(peft, PEFT):
        cfg.peft = peft
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

    # Optimizer overrides - Nemotron uses specific optimizer settings
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
    if is_full_sft:
        cfg.optimizer.lr = 5e-6
    else:
        cfg.optimizer.lr = 1e-4
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

    return cfg


def nemotron_3_super_debug_pretrain_config() -> ConfigContainer:
    """Return a pre-training config for Nemotron 3 Super Debug (small, 1-node).

    Uses ``Nemotron3SuperDebugProvider`` which is a small model that fits on a
    single node with TP=1, EP=8.

    Returns:
        ConfigContainer: Pre-training configuration for Nemotron 3 Super Debug.
    """
    cfg = nemotron_3_super_pretrain_config()

    cfg.model = Nemotron3SuperDebugProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
        seq_length=8192,
    )

    # MoE Token Dispatcher Settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    # CUDA Graph (TE impl + partial scopes: ~40% throughput gain over disabled)
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 3

    # Transformer Engine (TE)
    cfg.model.transformer_impl = "transformer_engine"

    # Kernel Selections
    cfg.model.attention_backend = "fused"
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.use_te_rng_tracker = True

    # MTP Settings
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.calculate_per_token_loss = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.mtp_use_repeated_layer = True

    return cfg


__all__ = [
    "nemotron_3_super_pretrain_config",
    "nemotron_3_super_debug_pretrain_config",
    "nemotron_3_super_finetune_config",
]
