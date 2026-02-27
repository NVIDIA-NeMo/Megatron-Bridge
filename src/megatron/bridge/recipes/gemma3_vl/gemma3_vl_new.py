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

"""Gemma3-VL finetuning recipes with parameterless API.

This module provides SFT and PEFT configurations for Gemma3-VL models (4B, 12B, 27B).
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Gemma3-VL 4B SFT Configuration
# =============================================================================


def gemma3_vl_4b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Gemma3-VL 4B Instruct.

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "google/gemma-3-4b-it"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
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
    cfg.model.cp_comm_type = "a2a"

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

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
        min_lr=5e-6 * 0.1,
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

    # DDP settings - VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

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
# Gemma3-VL 12B SFT Configuration
# =============================================================================


def gemma3_vl_12b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Gemma3-VL 12B Instruct.

    Default configuration: 1 node, 8 GPUs
    - TP=4, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "google/gemma-3-12b-it"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.cp_comm_type = "a2a"

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

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
        min_lr=5e-6 * 0.1,
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

    # DDP settings - VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

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
# Gemma3-VL 27B SFT Configuration
# =============================================================================


def gemma3_vl_27b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Gemma3-VL 27B Instruct.

    Default configuration: 2 nodes, 16 GPUs total
    - TP=8, PP=2
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = "google/gemma-3-27b-it"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.cp_comm_type = "a2a"

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

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
        min_lr=5e-6 * 0.1,
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

    # DDP settings - VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

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
# Gemma3-VL 4B PEFT Configuration
# =============================================================================


def gemma3_vl_4b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Gemma3-VL 4B Instruct.

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
    hf_path = "google/gemma-3-4b-it"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings
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
    cfg.model.cp_comm_type = "a2a"

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

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

    # DDP settings - VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

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
# Gemma3-VL 12B PEFT Configuration
# =============================================================================


def gemma3_vl_12b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Gemma3-VL 12B Instruct.

    Default configuration: 1 node, 8 GPUs
    - TP=1, PP=1 (lower than SFT for PEFT)
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
    hf_path = "google/gemma-3-12b-it"
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
    cfg.model.cp_comm_type = "a2a"

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

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

    # DDP settings - VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

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
# Gemma3-VL 27B PEFT Configuration
# =============================================================================


def gemma3_vl_27b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Gemma3-VL 27B Instruct.

    Default configuration: 1 node, 8 GPUs
    - TP=4, PP=1 (lower than SFT for PEFT)
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
    hf_path = "google/gemma-3-27b-it"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings - lower TP and PP for PEFT
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.cp_comm_type = "a2a"

    # TE / Transformer implementation
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph settings
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (disabled by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

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

    # DDP settings - VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

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
