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
GPT-OSS model recipes with parameterless API.

This module provides pre-configured SFT and PEFT configs for GPT-OSS model variants.
All configs follow the parameterless API pattern - returning a ConfigContainer
with all settings pre-configured.

Model variants:
- gpt_oss_20b: 20B parameters (MoE model)
- gpt_oss_120b: 120B parameters (MoE model)
"""

from typing import Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# SFT Configs
# =============================================================================


def gpt_oss_20b_sft_config() -> ConfigContainer:
    """Return a full SFT config for GPT-OSS 20B.

    Default parallelism: TP=1, PP=1, EP=8

    Returns:
        ConfigContainer with all settings pre-configured for GPT-OSS 20B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "openai/gpt-oss-20b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GPT-OSS has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed string (matches old config)
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 5e-6

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
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _sft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # MoE Overlap settings
    cfg.model.moe_shared_expert_overlap = False  # GPT-OSS default

    # Checkpoint config
    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GPT-OSS only sets check_for_nan_in_grad
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def gpt_oss_120b_sft_config() -> ConfigContainer:
    """Return a full SFT config for GPT-OSS 120B.

    Default parallelism: TP=1, PP=4, EP=8

    Returns:
        ConfigContainer with all settings pre-configured for GPT-OSS 120B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "openai/gpt-oss-120b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings (MoE-specific) - 120B SFT uses PP=4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GPT-OSS has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed string (matches old config)
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 5e-6

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
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _sft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # MoE Overlap settings
    cfg.model.moe_shared_expert_overlap = False  # GPT-OSS default

    # Checkpoint config
    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GPT-OSS only sets check_for_nan_in_grad
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


# =============================================================================
# PEFT Configs
# =============================================================================


def gpt_oss_20b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for GPT-OSS 20B.

    Default parallelism: TP=1, PP=1, EP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for GPT-OSS 20B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "openai/gpt-oss-20b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings (MoE-specific) - PEFT uses EP=1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GPT-OSS has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed string (matches old config)
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config - PEFT uses higher learning rate
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config - PEFT uses higher learning rate
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

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
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # MoE Overlap settings
    cfg.model.moe_shared_expert_overlap = False  # GPT-OSS default

    # Checkpoint config
    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GPT-OSS only sets check_for_nan_in_grad
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def gpt_oss_120b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for GPT-OSS 120B.

    Default parallelism: TP=1, PP=1, EP=8

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for GPT-OSS 120B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "openai/gpt-oss-120b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings (MoE-specific) - 120B PEFT uses PP=1, EP=8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GPT-OSS has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed string (matches old config)
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config - PEFT uses higher learning rate
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config - PEFT uses higher learning rate
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

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
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8 (mixed_precision settings)
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default
    cfg.model.moe_router_padding_for_fp8 = False  # MoE FP8 setting

    # MoE Overlap settings
    cfg.model.moe_shared_expert_overlap = False  # GPT-OSS default

    # Checkpoint config
    cfg.checkpoint.save_interval = 250
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GPT-OSS only sets check_for_nan_in_grad
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg
