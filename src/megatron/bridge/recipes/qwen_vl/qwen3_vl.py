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

"""Qwen3-VL finetuning recipes with parameterless API.

This module provides SFT and PEFT configurations for Qwen3-VL MoE models (8B, 30B-A3B, 235B-A22B).
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend


# =============================================================================
# Qwen3-VL 8B SFT Configuration
# =============================================================================
def qwen3_vl_8b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-VL 8B (dense model).

    Default configuration: 1 node, 8 GPUs
    - TP=2, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "Qwen/Qwen3-VL-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (not MoE for 8B)
    cfg.model.moe_token_dispatcher_type = None
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16

    # Apply flex dispatcher backend (will be no-op for non-MoE model)
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections (not applicable for dense 8B model)
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_grouped_gemm = False

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap (not applicable for dense model)
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance (not applicable for dense model)
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding (not applicable for dense model)
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer - lower LR for full SFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=0.00005,
        min_lr=0.000005,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# Qwen3-VL 30B-A3B SFT Configuration
# =============================================================================
def qwen3_vl_30b_a3b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-VL 30B-A3B (MoE model).

    Default configuration: 4 nodes, 32 GPUs
    - TP=1, PP=1, EP=8
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "Qwen/Qwen3-VL-30B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (MoE)
    cfg.model.moe_token_dispatcher_type = None
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16

    # Apply flex dispatcher backend (dynamically sets dispatcher based on GPU arch)
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 10

    # Optimizer - lower LR for full SFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=0.00005,
        min_lr=0.000005,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# Qwen3-VL 235B-A22B SFT Configuration
# =============================================================================
def qwen3_vl_235b_a22b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-VL 235B-A22B (MoE model).

    Default configuration: 64 nodes, 512 GPUs
    - TP=4, PP=1, EP=32
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "Qwen/Qwen3-VL-235B-A22B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (MoE)
    cfg.model.moe_token_dispatcher_type = None
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16

    # Apply flex dispatcher backend (dynamically sets dispatcher based on GPU arch)
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer - lower LR for full SFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# Qwen3-VL 8B PEFT Configuration
# =============================================================================
def qwen3_vl_8b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen3-VL 8B (dense model).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
    """
    cfg = _peft_common_vlm()

    # PEFT scheme
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    # Model configuration
    hf_path = "Qwen/Qwen3-VL-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings - lower TP for PEFT
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (not MoE for 8B)
    cfg.model.moe_token_dispatcher_type = None
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16

    # Apply flex dispatcher backend (will be no-op for non-MoE model)
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections (not applicable for dense 8B model)
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_grouped_gemm = False

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap (not applicable for dense model)
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance (not applicable for dense model)
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding (not applicable for dense model)
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer - higher LR for PEFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# Qwen3-VL 30B-A3B PEFT Configuration
# =============================================================================
def qwen3_vl_30b_a3b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen3-VL 30B-A3B (MoE model).

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1, EP=4
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
    """
    cfg = _peft_common_vlm()

    # PEFT scheme
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    # Model configuration
    hf_path = "Qwen/Qwen3-VL-30B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings - lower EP for PEFT
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (MoE)
    cfg.model.moe_token_dispatcher_type = None
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16

    # Apply flex dispatcher backend (dynamically sets dispatcher based on GPU arch)
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer - higher LR for PEFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg


# =============================================================================
# Qwen3-VL 235B-A22B PEFT Configuration
# =============================================================================
def qwen3_vl_235b_a22b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen3-VL 235B-A22B (MoE model).

    Default configuration: 8 nodes, 64 GPUs
    - TP=1, PP=1, EP=16
    - LR=1e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
    """
    cfg = _peft_common_vlm()

    # PEFT scheme
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    # Model configuration
    hf_path = "Qwen/Qwen3-VL-235B-A22B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings - lower EP for PEFT
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # Token dispatcher settings (MoE)
    cfg.model.moe_token_dispatcher_type = None
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16

    # Apply flex dispatcher backend (dynamically sets dispatcher based on GPU arch)
    apply_flex_dispatcher_backend(cfg.model, moe_flex_dispatcher_backend=None)

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE kernel selections
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # MoE overlap
    cfg.model.moe_shared_expert_overlap = False

    # MoE force balance
    cfg.model.moe_router_force_load_balancing = False

    # MoE FP8 padding
    cfg.model.moe_router_padding_for_fp8 = False

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer - higher LR for PEFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Optimizer precision settings (disabled by default for full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP settings
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Comm overlap settings (MoE)
    cfg.comm_overlap = None
    # cfg.comm_overlap.delay_wgrad_compute = False
    # cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    # FP8 and MXFP8 settings (disabled by default)
    cfg.mixed_precision = "bf16_mixed"
    # cfg.mixed_precision.fp8_recipe = None
    # cfg.mixed_precision.fp8 = False
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"
    # Uncomment below to use a pretrained checkpoint
    # cfg.checkpoint.pretrained_checkpoint = "/path/to/checkpoint"

    return cfg
