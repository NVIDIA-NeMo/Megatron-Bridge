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

"""Qwen3.5-VL recipes.

This module provides pretrain, SFT, and PEFT configurations for Qwen3.5-VL models:

- **Dense**: 800M, 2B, 4B, 9B, 27B
- **MoE**: 35B-A3B, 122B-A10B, 397B-A17B
"""

from __future__ import annotations

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.data.builders import MockVLMSFTDatasetConfig
from megatron.bridge.recipes.common import _peft_common_vlm, _pretrain_common, _sft_common_vlm
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Qwen3.5-VL Pretrain Configurations (mock dataset)
# =============================================================================
def qwen35_vl_9b_pretrain_4gpu_h100_bf16_mock_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 9B (dense)."""
    cfg = _pretrain_common()

    hf_path = "Qwen/Qwen3.5-9B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.seq_length = 4096

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg.dataset = MockVLMSFTDatasetConfig(
        seq_length=4096,
        hf_processor_path=hf_path,
        prompt="Describe this image.",
        num_workers=1,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pad_to_max_length=True,
    )
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.train.eval_interval = 500
    cfg.train.eval_iters = 32
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 27B (dense)."""
    cfg = _pretrain_common()

    hf_path = "Qwen/Qwen3.5-27B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.seq_length = 4096

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg.dataset = MockVLMSFTDatasetConfig(
        seq_length=4096,
        hf_processor_path=hf_path,
        prompt="Describe this image.",
        num_workers=1,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pad_to_max_length=True,
    )
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.train.eval_interval = 500
    cfg.train.eval_iters = 32
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 35B-A3B (MoE)."""
    cfg = _pretrain_common()

    hf_path = "Qwen/Qwen3.5-35B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.seq_length = 4096
    cfg.model.expert_model_parallel_size = 4

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg.dataset = MockVLMSFTDatasetConfig(
        seq_length=4096,
        hf_processor_path=hf_path,
        prompt="Describe this image.",
        num_workers=1,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pad_to_max_length=True,
    )
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.train.eval_interval = 500
    cfg.train.eval_iters = 32
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_mock_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 122B-A10B (MoE)."""
    cfg = _pretrain_common()

    hf_path = "Qwen/Qwen3.5-122B-A10B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.calculate_per_token_loss = True
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.seq_length = 4096
    cfg.model.expert_model_parallel_size = 8

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg.dataset = MockVLMSFTDatasetConfig(
        seq_length=4096,
        hf_processor_path=hf_path,
        prompt="Describe this image.",
        num_workers=1,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pad_to_max_length=True,
    )
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.train.eval_interval = 500
    cfg.train.eval_iters = 32
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config() -> ConfigContainer:
    """Return a pre-training config for Qwen3.5-VL 397B-A17B (MoE)."""
    cfg = _pretrain_common()

    hf_path = "Qwen/Qwen3.5-397B-A17B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 2
    cfg.model.calculate_per_token_loss = True
    cfg.model.sequence_parallel = True
    cfg.model.freeze_language_model = True
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = False
    cfg.model.seq_length = 4096
    cfg.model.expert_model_parallel_size = 16

    cfg.optimizer, cfg.scheduler = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=3e-4,
        min_lr=3e-5,
    )

    cfg.dataset = MockVLMSFTDatasetConfig(
        seq_length=4096,
        hf_processor_path=hf_path,
        prompt="Describe this image.",
        num_workers=1,
        dataloader_type="single",
        data_sharding=True,
        pin_memory=True,
        persistent_workers=False,
        pad_to_max_length=True,
    )
    cfg.tokenizer.tokenizer_model = hf_path
    cfg.train.eval_interval = 500
    cfg.train.eval_iters = 32
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.average_in_collective = False

    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# =============================================================================
# Qwen3.5-VL Dense SFT Configurations (800M, 2B, 4B, 9B, 27B)
# =============================================================================


def qwen35_vl_800m_sft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 800M (dense).

    Default configuration: 1 GPU
    - TP=1, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=2, so max TP=2.
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-0.8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_2b_sft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 2B (dense).

    Default configuration: 1 GPU
    - TP=1, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=2, so max TP=2.
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-2B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_4b_sft_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 4B (dense).

    Default configuration: 2 GPUs
    - TP=2, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=4, so max TP=4.
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-4B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_9b_sft_4gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 9B (dense).

    Default configuration: 4 GPUs
    - TP=4, PP=1
    - LR=5e-6 (full SFT)
    - Sequence length: 4096

    Note: num_kv_heads=4, so max TP=4.
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-9B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_27b_sft_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 27B (dense).

    Default configuration: 16 GPUs
    - TP=4, PP=4
    - LR=5e-6 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-27B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=5e-7,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# =============================================================================
# Qwen3.5-VL MoE SFT Configurations (35B-A3B, 122B-A10B, 397B-A17B)
# =============================================================================


def qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 35B-A3B (MoE).

    Default configuration: 16 GPUs
    - TP=2, PP=1, EP=16
    - LR=2e-5 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-35B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-5,
        min_lr=2e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_35b_a3b_sft_2gpu_h100_bf16_fsdp_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 35B-A3B (MoE) with Megatron FSDP.

    Uses Megatron FSDP for memory-efficient training with AG/RS overlap.
    Requires fsdp_dtensor checkpoint format (convert offline with
    checkpoint_inspector.py convert-torch-dist-to-fsdp-dtensor).

    Default configuration: 2 GPUs
    - TP=1, PP=1, EP=2
    - Megatron FSDP with double buffering
    - NCCL UB disabled (heterogeneous FSDP units cause hangs)
    - LR=2e-5 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-35B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 2
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-5,
        min_lr=2e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # Megatron FSDP settings
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.nccl_ub = False
    cfg.ddp.fsdp_db_use_persist_buf_on_alloc_fail = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.num_distributed_optimizer_instances = 1

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_122b_a10b_sft_48gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 122B-A10B (MoE).

    Default configuration: 48 GPUs
    - TP=2, PP=6, EP=8
    - LR=2e-5 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-122B-A10B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 6
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 36
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-5,
        min_lr=2e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_397b_a17b_sft_128gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3.5-VL 397B-A17B (MoE).

    Default configuration: 128 GPUs
    - TP=2, PP=4, EP=32
    - LR=2e-5 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    # Model config
    hf_path = "Qwen/Qwen3.5-397B-A17B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-5,
        min_lr=2e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# =============================================================================
# Qwen3.5-VL Dense PEFT Configurations (800M, 2B, 4B, 9B, 27B)
# =============================================================================


def qwen35_vl_800m_peft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 800M (dense).

    Default configuration: 1 GPU
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-0.8B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_2b_peft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 2B (dense).

    Default configuration: 1 GPU
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-2B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_4b_peft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 4B (dense).

    Default configuration: 1 GPU
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-4B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_9b_peft_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 9B (dense).

    Default configuration: 1 GPU
    - TP=1, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-9B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_27b_peft_2gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 27B (dense).

    Default configuration: 2 GPUs
    - TP=2, PP=1
    - LR=1e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-27B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
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

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


# =============================================================================
# Qwen3.5-VL MoE PEFT Configurations (35B-A3B, 122B-A10B, 397B-A17B)
# =============================================================================


def qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 35B-A3B (MoE).

    Default configuration: 4 GPUs
    - TP=2, PP=1, EP=4
    - LR=2e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-35B-A3B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_122b_a10b_peft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 122B-A10B (MoE).

    Default configuration: 8 GPUs
    - TP=2, PP=1, EP=8
    - LR=2e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-122B-A10B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 36
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


def qwen35_vl_397b_a17b_peft_32gpu_h100_bf16_config() -> ConfigContainer:
    """Return a PEFT config for Qwen3.5-VL 397B-A17B (MoE).

    Default configuration: 32 GPUs
    - TP=2, PP=1, EP=32
    - LR=2e-4 (PEFT)
    - Sequence length: 4096
    """
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config("lora")

    # Model config
    hf_path = "Qwen/Qwen3.5-397B-A17B"
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallelism settings
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True

    # VLM-specific settings
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    # MTP
    cfg.model.mtp_num_layers = 1
    cfg.model.mtp_loss_scaling_factor = 0.1

    # TE and kernels
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # MoE settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = None
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    # Memory saving
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 300000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 4
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 32

    # Optimizer
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=200,
        lr_decay_iters=300000,
        max_lr=2e-4,
        min_lr=3e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset config
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    cfg.dataset.enable_in_batch_packing = False
    cfg.dataset.defer_in_batch_packing_to_step = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    cfg.comm_overlap = None
    cfg.mixed_precision = "bf16_mixed"
    # Keep the complete process environment visible on the recipe.
    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
    }
    return cfg


__all__ = [
    "qwen35_vl_122b_a10b_peft_8gpu_h100_bf16_config",
    "qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_mock_config",
    "qwen35_vl_122b_a10b_sft_48gpu_h100_bf16_config",
    "qwen35_vl_27b_peft_2gpu_h100_bf16_config",
    "qwen35_vl_27b_pretrain_16gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen35_vl_27b_sft_16gpu_h100_bf16_config",
    "qwen35_vl_2b_peft_1gpu_h100_bf16_config",
    "qwen35_vl_2b_sft_1gpu_h100_bf16_config",
    "qwen35_vl_35b_a3b_sft_2gpu_h100_bf16_fsdp_config",
    "qwen35_vl_35b_a3b_peft_4gpu_h100_bf16_config",
    "qwen35_vl_35b_a3b_pretrain_8gpu_h100_bf16_mock_config",
    "qwen35_vl_35b_a3b_sft_16gpu_h100_bf16_config",
    "qwen35_vl_397b_a17b_peft_32gpu_h100_bf16_config",
    "qwen35_vl_397b_a17b_pretrain_512gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen35_vl_397b_a17b_sft_128gpu_h100_bf16_config",
    "qwen35_vl_4b_peft_1gpu_h100_bf16_config",
    "qwen35_vl_4b_sft_2gpu_h100_bf16_config",
    "qwen35_vl_800m_peft_1gpu_h100_bf16_config",
    "qwen35_vl_800m_sft_1gpu_h100_bf16_config",
    "qwen35_vl_9b_peft_1gpu_h100_bf16_config",
    "qwen35_vl_9b_pretrain_4gpu_h100_bf16_mock_config",  # pragma: allowlist secret
    "qwen35_vl_9b_sft_4gpu_h100_bf16_config",
]
