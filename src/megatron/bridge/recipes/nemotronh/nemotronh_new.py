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
NemotronH model recipes with parameterless API.

This module provides pre-configured SFT and PEFT configs for NemotronH models.
All configs follow the parameterless API pattern - importing from _sft_common()
or _peft_common() and overriding model-specific settings.

Note: NemotronH are dense models (not MoE).
They use NemotronHModelProvider4B, 8B, 47B, and 56B.

Model variants:
- nemotronh_4b: 4B parameters
- nemotronh_8b: 8B parameters
- nemotronh_47b: 47B parameters
- nemotronh_56b: 56B parameters
"""

from typing import Union

import torch

from megatron.bridge.models.nemotronh import (
    NemotronHModelProvider4B,
    NemotronHModelProvider8B,
    NemotronHModelProvider47B,
    NemotronHModelProvider56B,
)
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.recipes.common import _peft_common, _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# SFT Configs
# =============================================================================


def nemotronh_4b_sft_config() -> ConfigContainer:
    """Return a full SFT config for NemotronH 4B.

    Default parallelism: TP=1, PP=1, SP=False

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 4B SFT.
    """
    cfg = _sft_common()

    # Model config - uses NemotronHModelProvider4B
    cfg.model = NemotronHModelProvider4B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config - packed_sequence=True by default (from _sft_common)
    # Override seq_length for NemotronH (uses 8192)
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    # Adjust pad_seq_to_mult for context parallelism
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides - NemotronH uses specific optimizer settings
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-6
    cfg.scheduler.min_lr = 1e-6

    # Tokenizer - HuggingFace tokenizer with special eos token
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-4B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def nemotronh_8b_sft_config() -> ConfigContainer:
    """Return a full SFT config for NemotronH 8B.

    Default parallelism: TP=2, PP=1, SP=True

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 8B SFT.
    """
    cfg = _sft_common()

    # Model config - uses NemotronHModelProvider8B
    cfg.model = NemotronHModelProvider8B(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-6
    cfg.scheduler.min_lr = 1e-6

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-8B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def nemotronh_47b_sft_config() -> ConfigContainer:
    """Return a full SFT config for NemotronH 47B.

    Default parallelism: TP=8, PP=2, SP=True

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 47B SFT.
    """
    cfg = _sft_common()

    # Model config - uses NemotronHModelProvider47B
    cfg.model = NemotronHModelProvider47B(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-6
    cfg.scheduler.min_lr = 1e-6

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-47B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def nemotronh_56b_sft_config() -> ConfigContainer:
    """Return a full SFT config for NemotronH 56B.

    Default parallelism: TP=8, PP=1, SP=True

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 56B SFT.
    """
    cfg = _sft_common()

    # Model config - uses NemotronHModelProvider56B
    cfg.model = NemotronHModelProvider56B(
        tensor_model_parallel_size=8,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=True,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-6
    cfg.scheduler.min_lr = 1e-6

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-56B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


# =============================================================================
# PEFT Configs
# =============================================================================


def nemotronh_4b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for NemotronH 4B.

    Default parallelism: TP=1, PP=1, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 4B PEFT.
    """
    cfg = _peft_common()

    # Model config - PEFT uses same parallelism as SFT for 4B
    cfg.model = NemotronHModelProvider4B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # PEFT config - NemotronH uses Mamba-specific target modules
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-5
    cfg.scheduler.min_lr = 1e-5

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-4B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def nemotronh_8b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for NemotronH 8B.

    Default parallelism: TP=1, PP=1, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 8B PEFT.
    """
    cfg = _peft_common()

    # Model config - PEFT uses TP=1, SP=False
    cfg.model = NemotronHModelProvider8B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-5
    cfg.scheduler.min_lr = 1e-5

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-8B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def nemotronh_47b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for NemotronH 47B.

    Default parallelism: TP=4, PP=1, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 47B PEFT.
    """
    cfg = _peft_common()

    # Model config - PEFT uses TP=4, PP=1
    cfg.model = NemotronHModelProvider47B(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-5
    cfg.scheduler.min_lr = 1e-5

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-47B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg


def nemotronh_56b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for NemotronH 56B.

    Default parallelism: TP=4, PP=1, SP=False

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Returns:
        ConfigContainer with all settings pre-configured for NemotronH 56B PEFT.
    """
    cfg = _peft_common()

    # Model config - PEFT uses TP=4, PP=1
    cfg.model = NemotronHModelProvider56B(
        tensor_model_parallel_size=4,
        pipeline_model_parallel_size=1,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=1,
        sequence_parallel=False,
        seq_length=8192,
    )

    # Parallelism settings
    cfg.model.pipeline_model_parallel_layout = None

    # Sequence length
    cfg.model.seq_length = 8192

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

    # PEFT config
    mamba_target_modules = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2", "in_proj", "out_proj"]
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme, target_modules=mamba_target_modules)
    elif isinstance(peft_scheme, PEFT):
        cfg.peft = peft_scheme
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

    # Training config overrides
    cfg.train.eval_interval = 50
    cfg.train.eval_iters = 10

    # Dataset config
    cfg.dataset.seq_length = 8192
    cfg.dataset.packed_sequence_specs.packed_sequence_size = 8192
    if cfg.model.context_parallel_size > 1:
        cfg.dataset.packed_sequence_specs.pad_seq_to_mult = cfg.model.context_parallel_size * 2

    # Optimizer overrides
    cfg.optimizer.adam_beta2 = 0.95
    cfg.optimizer.min_lr = 1e-5
    cfg.scheduler.min_lr = 1e-5

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "nvidia/Nemotron-H-56B-Base-8K"
    cfg.tokenizer.hf_tokenizer_kwargs = {"eos_token": "<SPECIAL_11>"}

    # Checkpoint config overrides
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    # DDP config
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.use_distributed_optimizer = True

    return cfg
