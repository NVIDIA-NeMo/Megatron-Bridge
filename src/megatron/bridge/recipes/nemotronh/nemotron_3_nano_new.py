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

"""
Nemotron 3 Nano model recipes with parameterless API.

This module provides pre-configured SFT and PEFT configs for Nemotron 3 Nano model.
All configs follow the parameterless API pattern - importing from _sft_common()
or _peft_common() and overriding model-specific settings.

Note: Nemotron 3 Nano is an MoE (Mixture of Experts) model with 30B-A3B parameters.
It uses Nemotron3NanoProvider instead of AutoBridge.from_hf_pretrained().

Model variants:
- nemotron_3_nano: 30B-A3B parameters (MoE model)
"""

from typing import Union

import torch

from megatron.bridge.models.nemotronh import Nemotron3NanoProvider
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# SFT Config
# =============================================================================


def nemotron_3_nano_sft_config() -> ConfigContainer:
    """Return a full SFT config for Nemotron 3 Nano (30B-A3B MoE).

    Default parallelism: TP=1, PP=1, EP=8, SP=False

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3 Nano SFT.
    """
    cfg = _sft_common()

    # Model config - uses Nemotron3NanoProvider
    cfg.model = Nemotron3NanoProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
        apply_rope_fusion=False,
        async_tensor_model_parallel_allreduce=True,
        attention_backend="fused",
        gradient_accumulation_fusion=True,
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
        seq_length=2048,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 2048

    # DeePEP settings - set to True to enable DeePEP (enabled by default for Nemotron)
    enable_deepep = True
    if enable_deepep:
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = False
    else:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
        cfg.model.moe_shared_expert_overlap = True

    cfg.model.moe_hybridep_num_sms = 16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "fused"
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

    # FP8 & MXFP8 settings
    # Note: mixed_precision="bf16_mixed" is set as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.model.moe_router_padding_for_fp8 = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # Training config overrides
    cfg.train.eval_interval = 500

    # Dataset config - packed_sequence=True by default (from _sft_common), seq_length=2048
    # _sft_common already sets seq_length=2048 and packed_sequence=True
    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides - Nemotron uses specific optimizer settings
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

    # Logger config
    cfg.logger.log_interval = 10
    cfg.logger.log_timers_to_tensorboard = False

    # RNG config - Nemotron uses seed 1234
    cfg.rng.seed = 1234

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    # Communication overlap settings
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    return cfg


# =============================================================================
# PEFT Config
# =============================================================================


def nemotron_3_nano_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Nemotron 3 Nano (30B-A3B MoE).

    Default parallelism: TP=1, PP=1, EP=8, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Nemotron 3 Nano PEFT.
    """
    cfg = _peft_common()

    # Model config - PEFT uses same parallelism as SFT
    cfg.model = Nemotron3NanoProvider(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        expert_tensor_parallel_size=1,
        expert_model_parallel_size=8,
        apply_rope_fusion=False,
        async_tensor_model_parallel_allreduce=True,
        attention_backend="fused",
        gradient_accumulation_fusion=True,
        init_method_std=0.0173,
        use_fused_weighted_squared_relu=True,
        seq_length=2048,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 2048

    # DeePEP settings - set to True to enable DeePEP (enabled by default for Nemotron)
    enable_deepep = True
    if enable_deepep:
        cfg.model.moe_token_dispatcher_type = "flex"
        cfg.model.moe_flex_dispatcher_backend = "deepep"
        cfg.model.moe_shared_expert_overlap = False
    else:
        cfg.model.moe_token_dispatcher_type = "alltoall"
        cfg.model.moe_flex_dispatcher_backend = None
        cfg.model.moe_shared_expert_overlap = True

    cfg.model.moe_hybridep_num_sms = 16

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 settings
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.model.moe_router_padding_for_fp8 = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # PEFT config - Nemotron uses Mamba-specific target modules
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
    else:
        # Default to LoRA with Mamba target modules
        cfg.peft = LoRA(
            target_modules=mamba_target_modules,
            dim=32,
            alpha=32,
            dropout=0.0,
            dropout_position="pre",
            lora_A_init_method="xavier",
            lora_B_init_method="zero",
        )

    # Training config overrides
    cfg.train.eval_interval = 500

    # Dataset config - packed_sequence=True by default (from _peft_common), seq_length=2048
    # _peft_common already sets seq_length=2048 and packed_sequence=True
    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
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

    # Logger config
    cfg.logger.log_interval = 10
    cfg.logger.log_timers_to_tensorboard = False

    # RNG config - Nemotron uses seed 1234
    cfg.rng.seed = 1234

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.use_distributed_optimizer = True

    # Communication overlap settings
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=True,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    return cfg
