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

"""Qwen2/2.5 fine-tuning recipes using the new parameterless API.

This module provides fine-tuning configurations for Qwen2 and Qwen2.5 models:

- SFT configs (`*_sft_config`): Full parameter fine-tuning using `_sft_common()`
- PEFT configs (`*_peft_config`): Parameter-efficient fine-tuning using `_peft_common()`
  with LoRA/DoRA support

PEFT configs accept an optional `peft_scheme` argument:
- 'lora' (default): Low-Rank Adaptation
- 'dora': Weight-Decomposed Low-Rank Adaptation
- PEFT instance: Custom user-provided PEFT configuration
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


# =============================================================================
# Qwen2 SFT (Full Fine-Tuning) Configs
# =============================================================================


def qwen2_500m_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2 500M.

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-0.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-0.5B"

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences) else 4096
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen2_1p5b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2 1.5B.

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-1.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-1.5B"

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen2_7b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2 7B.

    Recommended parallelism: TP=2, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-7B"

    # Parallelism settings - higher TP for 7B
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen2_72b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2 72B.

    Recommended parallelism: TP=8, PP=4 (4 nodes, 32 GPUs total)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-72B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-72B"

    # Parallelism settings - needs more parallelism for 72B
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# Qwen2.5 SFT (Full Fine-Tuning) Configs
# =============================================================================


def qwen25_500m_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2.5 500M.

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-0.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-0.5B"

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_1p5b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2.5 1.5B.

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-1.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-1.5B"

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_7b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2.5 7B.

    Recommended parallelism: TP=2, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-7B"

    # Parallelism settings - higher TP for 7B
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_14b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2.5 14B.

    Recommended parallelism: TP=4, PP=1 (1 node, 8 GPUs)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-14B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-14B"

    # Parallelism settings - higher TP for 14B
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_32b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2.5 32B.

    Recommended parallelism: TP=8, PP=2 (2 nodes, 16 GPUs total)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-32B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-32B"

    # Parallelism settings - needs more parallelism for 32B
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def qwen25_72b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen2.5 72B.

    Recommended parallelism: TP=8, PP=4 (4 nodes, 32 GPUs total)
    """
    cfg = _sft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-72B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-72B"

    # Parallelism settings - needs more parallelism for 72B
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _sft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 SFT uses different DDP settings
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# Qwen2 PEFT (Parameter-Efficient Fine-Tuning) Configs
# =============================================================================


def qwen2_500m_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2 500M.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-0.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-0.5B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen2_1p5b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2 1.5B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-1.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-1.5B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen2_7b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2 7B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-7B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen2_72b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2 72B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=8, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2-72B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2-72B"

    # Parallelism settings - TP=8 for 72B PEFT
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


# =============================================================================
# Qwen2.5 PEFT (Parameter-Efficient Fine-Tuning) Configs
# =============================================================================


def qwen25_500m_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2.5 500M.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-0.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-0.5B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen25_1p5b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2.5 1.5B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-1.5B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-1.5B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen25_7b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2.5 7B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-7B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-7B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen25_14b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2.5 14B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=1, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-14B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-14B"

    # Parallelism settings - TP=1 for PEFT (only training adapters)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen25_32b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2.5 32B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=8, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-32B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-32B"

    # Parallelism settings - TP=8 for 32B PEFT
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg


def qwen25_72b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for Qwen2.5 72B.

    Args:
        peft_scheme: PEFT scheme - 'lora', 'dora', or a PEFT instance. Default: 'lora'

    Recommended parallelism: TP=8, PP=1 (1 node, 8 GPUs)
    """
    cfg = _peft_common()

    # Model config
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen2.5-72B").to_megatron_provider(load_weights=False)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen2.5-72B"

    # Parallelism settings - TP=8 for 72B PEFT
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Sequence length (2048 for packed sequences)
    cfg.model.seq_length = 2048

    # PEFT config - use user-provided scheme or default to LoRA
    cfg.peft = default_peft_config(peft_scheme)

    # Set pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    # Optimizer and scheduler overrides for Qwen2
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100  # Same as train_iters
    cfg.optimizer.adam_beta2 = 0.95

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
    # Note: mixed_precision="bf16_mixed" is set in _peft_common as default
    # These are defaults for FP8, enable them if using FP8 - FP8 is not enabled by default
    # cfg.mixed_precision.fp8_recipe = "tensorwise"  # default, uncomment to enable
    # cfg.mixed_precision.fp8 = None  # not enabled by default
    # cfg.mixed_precision.fp8_param_gather = False  # default
    # cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = False  # default

    # Optimizer precision settings
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    # cfg.checkpoint.save and cfg.checkpoint.load are set in _peft_common. To override:
    # cfg.checkpoint.save = "path/to/save"
    # cfg.checkpoint.load = "path/to/load"

    # DDP config - Qwen2 PEFT DDP settings
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    return cfg
