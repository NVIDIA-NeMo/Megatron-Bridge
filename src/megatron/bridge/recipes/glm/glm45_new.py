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
GLM-4.5 model recipes with parameterless API.

This module provides pre-configured SFT and PEFT configs for GLM-4.5 model variants.
All configs follow the parameterless API pattern - returning a ConfigContainer
with all settings pre-configured.

Note: Packed sequence is NOT supported for GLM-4.5 models.

Model variants:
- glm45_355b: 355B parameters with 32B active (MoE model with MTP support)
- glm45_air_106b: 106B parameters with 12B active (MoE model with MTP support)
"""

from typing import Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


# =============================================================================
# SFT Configs
# =============================================================================


def glm45_355b_sft_config() -> ConfigContainer:
    """Return a full SFT config for GLM-4.5 355B-A32B.

    Default parallelism: TP=2, PP=8, EP=16

    Note: Packed sequence is NOT supported for GLM-4.5.

    Returns:
        ConfigContainer with all settings pre-configured for GLM-4.5 355B SFT.
    """
    # Get base SFT config with packed_sequence=False (not supported for GLM-4.5)
    cfg = _sft_common()

    # Override dataset - GLM-4.5 does NOT support packed_sequence
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=False, pad_seq_to_mult=1)

    # Model config from HuggingFace
    hf_path = "zai-org/GLM-4.5"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Parallelism settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GLM-4.5 has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
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

    # Optimizer config - GLM uses specific adam parameters
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
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
    cfg.model.cross_entropy_fusion_impl = "native"  # GLM uses native

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
    cfg.model.moe_shared_expert_overlap = True  # Default from GLM model provider

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GLM only sets check_for_nan_in_grad
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


def glm45_air_106b_sft_config() -> ConfigContainer:
    """Return a full SFT config for GLM-4.5 Air 106B-A12B.

    Default parallelism: TP=1, PP=4, EP=8

    Note: Packed sequence is NOT supported for GLM-4.5.

    Returns:
        ConfigContainer with all settings pre-configured for GLM-4.5 Air 106B SFT.
    """
    # Get base SFT config with packed_sequence=False (not supported for GLM-4.5)
    cfg = _sft_common()

    # Override dataset - GLM-4.5 does NOT support packed_sequence
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=False, pad_seq_to_mult=1)

    # Model config from HuggingFace
    hf_path = "zai-org/GLM-4.5-Air"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Parallelism settings (MoE-specific)
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
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GLM-4.5 has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
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

    # Optimizer config - GLM uses specific adam parameters
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
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
    cfg.model.cross_entropy_fusion_impl = "native"  # GLM uses native

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
    cfg.model.moe_shared_expert_overlap = True  # Default from GLM model provider

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GLM only sets check_for_nan_in_grad
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


def glm45_355b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for GLM-4.5 355B-A32B.

    Default parallelism: TP=2, PP=4, EP=4

    Note: Packed sequence is NOT supported for GLM-4.5.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for GLM-4.5 355B PEFT.
    """
    # Get base PEFT config with packed_sequence=False (not supported for GLM-4.5)
    cfg = _peft_common()

    # Override dataset - GLM-4.5 does NOT support packed_sequence
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=False, pad_seq_to_mult=1)

    # Model config from HuggingFace
    hf_path = "zai-org/GLM-4.5"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    # Parallelism settings (MoE-specific) - PEFT uses smaller parallelism
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GLM-4.5 has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
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

    # Optimizer config - GLM uses specific adam parameters, PEFT uses higher LR
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
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
    cfg.model.cross_entropy_fusion_impl = "native"  # GLM uses native

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
    cfg.model.moe_shared_expert_overlap = True  # Default from GLM model provider

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GLM only sets check_for_nan_in_grad
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


def glm45_air_106b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for GLM-4.5 Air 106B-A12B.

    Default parallelism: TP=1, PP=2, EP=4

    Note: Packed sequence is NOT supported for GLM-4.5.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for GLM-4.5 Air 106B PEFT.
    """
    # Get base PEFT config with packed_sequence=False (not supported for GLM-4.5)
    cfg = _peft_common()

    # Override dataset - GLM-4.5 does NOT support packed_sequence
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=False, pad_seq_to_mult=1)

    # Model config from HuggingFace
    hf_path = "zai-org/GLM-4.5-Air"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    # Parallelism settings (MoE-specific) - PEFT uses smaller parallelism
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # GLM-4.5 has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
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

    # Optimizer config - GLM uses specific adam parameters, PEFT uses higher LR
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.weight_decay = 0.1
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
    cfg.model.cross_entropy_fusion_impl = "native"  # GLM uses native

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
    cfg.model.moe_shared_expert_overlap = True  # Default from GLM model provider

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - GLM only sets check_for_nan_in_grad
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
