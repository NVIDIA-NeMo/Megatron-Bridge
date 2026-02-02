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
Gemma2 model recipes with parameterless API.

This module provides pre-configured SFT and PEFT configs for Gemma2 model variants.
All configs follow the parameterless API pattern - returning a ConfigContainer
with all settings pre-configured.

Model variants:
- gemma2_2b: 2B parameters (dense model)
- gemma2_9b: 9B parameters (dense model)
- gemma2_27b: 27B parameters (dense model)
"""

from typing import Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


def _adjust_gemma2_vocab_size(model_cfg, hf_path: str):
    """Adjust vocab size for Gemma2 (model vocab < tokenizer vocab).
    
    Note: This requires HuggingFace authentication for Gemma2 models.
    If the tokenizer cannot be loaded, the vocab size adjustment is skipped.
    """
    if hasattr(model_cfg, "vocab_size") and hf_path:
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
            if len(tokenizer) > model_cfg.vocab_size:
                model_cfg.vocab_size = len(tokenizer)
        except Exception:
            # Skip vocab size adjustment if tokenizer cannot be loaded
            # (e.g., due to missing HuggingFace authentication)
            pass


# =============================================================================
# SFT Configs
# =============================================================================


def gemma2_2b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Gemma2 2B.

    Default parallelism: TP=1, PP=1

    Returns:
        ConfigContainer with all settings pre-configured for Gemma2 2B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "google/gemma-2-2b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    _adjust_gemma2_vocab_size(cfg.model, hf_path)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length - Gemma2 uses 4096 for non-packed, 2048 for packed
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

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

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False  # Gemma2 default
    cfg.train.manual_gc_interval = 0  # Gemma2 default

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
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
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma2 uses native

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
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def gemma2_9b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Gemma2 9B.

    Default parallelism: TP=4, PP=1

    Returns:
        ConfigContainer with all settings pre-configured for Gemma2 9B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "google/gemma-2-9b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    _adjust_gemma2_vocab_size(cfg.model, hf_path)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length - Gemma2 uses 4096 for non-packed, 2048 for packed
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 9B SFT uses TP=4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False  # Gemma2 default
    cfg.train.manual_gc_interval = 0  # Gemma2 default

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
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
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma2 uses native

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
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def gemma2_27b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Gemma2 27B.

    Default parallelism: TP=8, PP=2

    Returns:
        ConfigContainer with all settings pre-configured for Gemma2 27B SFT.
    """
    cfg = _sft_common()

    # Model config from HuggingFace
    hf_path = "google/gemma-2-27b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    _adjust_gemma2_vocab_size(cfg.model, hf_path)

    # Tokenizer
    cfg.tokenizer.tokenizer_model = hf_path

    # Sequence length - Gemma2 uses 4096 for non-packed, 2048 for packed
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Packed sequence settings
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Parallelism settings - 27B SFT uses TP=8, PP=2
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = None  # Will be set by PP > 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False  # Gemma2 default
    cfg.train.manual_gc_interval = 0  # Gemma2 default

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Scheduler config
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
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
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma2 uses native

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
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = True

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


# =============================================================================
# PEFT Configs
# =============================================================================


def gemma2_2b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Gemma2 2B.

    Default parallelism: TP=1, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Gemma2 2B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "google/gemma-2-2b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    _adjust_gemma2_vocab_size(cfg.model, hf_path)

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

    # Parallelism settings - PEFT uses TP=1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False  # Gemma2 default
    cfg.train.manual_gc_interval = 0  # Gemma2 default

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config - PEFT uses higher learning rate
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False  # PEFT disables distributed optimizer

    # Scheduler config - PEFT uses higher learning rate
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.max_lr = 1e-4

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma2 uses native

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

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - PEFT uses different settings (no distributed optimizer)
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def gemma2_9b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Gemma2 9B.

    Default parallelism: TP=1, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Gemma2 9B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "google/gemma-2-9b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    _adjust_gemma2_vocab_size(cfg.model, hf_path)

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

    # Parallelism settings - 9B PEFT uses TP=1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False  # Gemma2 default
    cfg.train.manual_gc_interval = 0  # Gemma2 default

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config - PEFT uses higher learning rate
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False  # PEFT disables distributed optimizer

    # Scheduler config - PEFT uses higher learning rate
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.max_lr = 1e-4

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma2 uses native

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

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - PEFT uses different settings (no distributed optimizer)
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def gemma2_27b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Gemma2 27B.

    Default parallelism: TP=4, PP=1

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for Gemma2 27B PEFT.
    """
    cfg = _peft_common()

    # Model config from HuggingFace
    hf_path = "google/gemma-2-27b"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    _adjust_gemma2_vocab_size(cfg.model, hf_path)

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

    # Parallelism settings - 27B PEFT uses TP=4
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 100
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = False  # Gemma2 default
    cfg.train.manual_gc_interval = 0  # Gemma2 default

    # Logger config
    cfg.logger.log_interval = 1

    # Optimizer config - PEFT uses higher learning rate
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.use_distributed_optimizer = False  # PEFT disables distributed optimizer

    # Scheduler config - PEFT uses higher learning rate
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.scheduler.max_lr = 1e-4

    # TE (Transformer Engine)
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernel selections
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"  # Gemma2 uses native

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

    # Checkpoint config
    cfg.checkpoint.save_interval = 100
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config - PEFT uses different settings (no distributed optimizer)
    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg
