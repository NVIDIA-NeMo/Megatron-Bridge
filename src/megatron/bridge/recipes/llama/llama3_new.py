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
Llama3 model recipes with parameterless API.

This module provides pre-configured SFT and PEFT configs for Llama3 model variants.
All configs follow the parameterless API pattern - returning a ConfigContainer
with all settings pre-configured.

Model variants:
- llama32_1b: Llama 3.2 1B (dense model)
- llama32_3b: Llama 3.2 3B (dense model)
- llama3_8b: Llama 3 8B (dense model)
- llama31_8b: Llama 3.1 8B (dense model)
- llama3_70b: Llama 3 70B (dense model)
- llama31_70b: Llama 3.1 70B (dense model)
- llama31_405b: Llama 3.1 405B (dense model)
"""

from typing import Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


# =============================================================================
# SFT Configs
# =============================================================================


def llama32_1b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3.2 1B.

    Default parallelism: TP=1, PP=1

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.2 1B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Llama-3.2-1B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision - use bf16_mixed string (matches old config)
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8  # packed_sequence=True, else 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # Kernel selections
    cfg.model.attention_backend = None
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

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama32_3b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3.2 3B.

    Default parallelism: TP=1, PP=1

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.2 3B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Llama-3.2-3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # FP8 & MXFP8
    # cfg.mixed_precision.fp8_recipe = "tensorwise"
    # cfg.mixed_precision.fp8 = None
    # cfg.mixed_precision.fp8_param_gather = False
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama3_8b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3 8B.

    Default parallelism: TP=2, PP=1

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3 8B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - SFT uses TP=2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama31_8b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3.1 8B.

    Default parallelism: TP=2, PP=1

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.1 8B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3.1-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - SFT uses TP=2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama3_70b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3 70B.

    Default parallelism: TP=8, PP=4

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3 70B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3-70B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 70B SFT uses TP=8, PP=4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama31_70b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3.1 70B.

    Default parallelism: TP=8, PP=4

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.1 70B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3.1-70B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 70B SFT uses TP=8, PP=4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama31_405b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Llama 3.1 405B.

    Default parallelism: TP=8, PP=16, SP=True
    Total: 128 GPUs (16 nodes)

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.1 405B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3.1-405B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length - 405B uses 2048 always
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 2048

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 405B SFT uses TP=8, PP=16, SP=True
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Pipeline split settings - required for 405B
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    # Mixed precision - convert string to config object for 405B
    cfg.mixed_precision = get_mixed_precision_config("bf16_mixed")
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # Training config - 405B uses different batch size
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 16  # 405B SFT uses 16
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

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

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - 405B SFT uses specific DDP settings
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True

    # CommOverlap config - 405B SFT uses TP communication overlap
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


# =============================================================================
# PEFT Configs
# =============================================================================


def llama32_1b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3.2 1B.

    Default parallelism: TP=1, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.2 1B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Llama-3.2-1B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config - PEFT uses higher LR
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False  # PEFT disables this

    # Scheduler config - PEFT uses higher LR
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections - PEFT disables cross_entropy_loss_fusion
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama32_3b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3.2 3B.

    Default parallelism: TP=1, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.2 3B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Llama-3.2-3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama3_8b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3 8B.

    Default parallelism: TP=1, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3 8B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # PEFT config - 8B uses dim=8, alpha=16
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - PEFT uses TP=1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama31_8b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3.1 8B.

    Default parallelism: TP=1, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.1 8B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3.1-8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # PEFT config - 8B uses dim=8, alpha=16
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - PEFT uses TP=1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama3_70b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3 70B.

    Default parallelism: TP=8, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3 70B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3-70B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # PEFT config - 70B uses dim=16, alpha=32
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 16
        peft_cfg.alpha = 32

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 70B PEFT uses TP=8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama31_70b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3.1 70B.

    Default parallelism: TP=8, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.1 70B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3.1-70B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 4096

    # PEFT config - 70B uses dim=16, alpha=32
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 16
        peft_cfg.alpha = 32

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 70B PEFT uses TP=8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def llama31_405b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Llama 3.1 405B.

    Default parallelism: TP=4, PP=8, VPP=8, SP=True
    Total: 32 GPUs (4 nodes)

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Llama 3.1 405B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "meta-llama/Meta-Llama-3.1-405B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length - 405B uses 2048
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 2048

    # PEFT config - 405B uses dim=16, alpha=32, target_modules=["linear_qkv"]
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 16
        peft_cfg.alpha = 32
        peft_cfg.target_modules = ["linear_qkv"]

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 405B PEFT uses TP=4, PP=8, VPP=8, SP=True
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Pipeline split settings
    cfg.model.account_for_embedding_in_pipeline_split = True
    cfg.model.account_for_loss_in_pipeline_split = True

    # Mixed precision
    cfg.mixed_precision = get_mixed_precision_config("bf16_mixed")
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    # Training config - 405B PEFT uses GBS=32
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.98
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.max_lr = 1e-4

    # TE
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = False
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # CommOverlap config - 405B PEFT disables TP comm overlap
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    # RNG seed
    cfg.rng.seed = 5678

    return cfg