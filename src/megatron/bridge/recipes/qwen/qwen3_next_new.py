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
Qwen3-Next model recipes with parameterless API.

This module provides pre-configured SFT configs for Qwen3-Next model variants.
All configs follow the parameterless API pattern - returning a ConfigContainer
with all settings pre-configured.

Note: PEFT and packed_sequence are NOT currently supported for Qwen3-Next models.
Only full SFT is available.

Model variants:
- qwen3_next_80b_a3b: 80B parameters with 3B active (MoE model with MTP support)
"""

from typing import Union

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config, default_squad_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen3_next_80b_a3b_sft_config() -> ConfigContainer:
    """Return a full SFT config for Qwen3-Next 80B-A3B.

    Recommended parallelism: TP=1, PP=2, EP=8
    Note: Packed sequence is NOT supported for Qwen3-Next.
    Note: Qwen3-Next uses no_weight_decay_cond_type = "qwen3_next" for scheduler.

    Returns:
        ConfigContainer with all settings pre-configured for Qwen3-Next 80B-A3B SFT.
    """
    # Get base SFT config
    cfg = _sft_common()

    # Override dataset - Qwen3-Next does NOT support packed_sequence
    cfg.dataset = default_squad_config(seq_length=2048, packed_sequence=False, pad_seq_to_mult=1)

    # Model config from HuggingFace
    cfg.model = AutoBridge.from_hf_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct").to_megatron_provider(
        load_weights=False
    )

    # Tokenizer
    cfg.tokenizer.tokenizer_model = "Qwen/Qwen3-Next-80B-A3B-Instruct"

    # Sequence length
    cfg.model.seq_length = 2048
    cfg.dataset.seq_length = 2048

    # Parallelism settings (MoE-specific)
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False

    # MoE Token Dispatcher settings
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_flex_dispatcher_backend = "deepep"  # qwen3_next has moe_flex_dispatcher_backend = "deepep" when loaded via AutoBridge.from_hf_pretrained
    cfg.model.moe_hybridep_num_sms = 16

    # Mixed precision - use bf16_mixed config object
    cfg.mixed_precision = bf16_mixed()

    # Training config
    cfg.train.train_iters = 1000
    cfg.train.eval_interval = 30
    cfg.train.eval_iters = 32
    cfg.train.global_batch_size = 64  # packed_sequence=False, so use 64
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

    # Scheduler config - Qwen3-Next specific
    cfg.scheduler.lr_warmup_iters = 50
    cfg.scheduler.lr_decay_iters = None  # Will use train_iters
    cfg.scheduler.max_lr = 5e-6
    cfg.scheduler.no_weight_decay_cond_type = "qwen3_next"

    # Optimizer min_lr - Qwen3-Next uses same value as max_lr
    cfg.optimizer.min_lr = 5e-6

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

    # Memory saving (recompute & offloading) - Qwen3-Next uses selective recompute
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["layernorm", "moe", "moe_act"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
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
    cfg.model.moe_shared_expert_overlap = False

    # Checkpoint config
    cfg.checkpoint.save_interval = 50
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    # DDP config
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.use_megatron_fsdp = False

    # MoE Force Load Balancing
    cfg.model.moe_router_force_load_balancing = False

    # RNG seed
    cfg.rng.seed = 5678

    return cfg


def qwen3_next_80b_a3b_peft_config(
    peft_scheme: Union[str, PEFT] = "lora",
) -> ConfigContainer:
    """Return a PEFT config for Qwen3-Next 80B-A3B.

    Note: PEFT is NOT currently supported for Qwen3-Next models.
    This function raises NotImplementedError.

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.

    Raises:
        NotImplementedError: PEFT is not supported for Qwen3-Next models.
    """
    raise NotImplementedError(
        "PEFT is not currently supported for Qwen3-Next models. "
        "Only full SFT is available via qwen3_next_80b_a3b_sft_config()."
    )
