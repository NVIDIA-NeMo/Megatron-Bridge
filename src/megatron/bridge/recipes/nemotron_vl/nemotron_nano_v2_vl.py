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

"""Nemotron Nano V2 VL finetuning recipes with parameterless defaults.

This module provides SFT and PEFT configurations for Nemotron Nano V2 VL 12B.
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.peft.dora import DoRA
from megatron.bridge.peft.lora import VLMLoRA
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


_DEFAULT_HF_MODEL_PATH = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"
_ALL_COMPONENT_LORA_TARGET_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]
_LANGUAGE_LORA_TARGET_MODULES = [
    "*language_model*.linear_qkv",
    "*language_model*.linear_proj",
    "*language_model*.linear_fc1",
    "*language_model*.linear_fc2",
]
_VISION_LORA_TARGET_MODULES = [
    "*vision_model*.linear_qkv",
    "*vision_model*.linear_proj",
    "*vision_model*.linear_fc1",
    "*vision_model*.linear_fc2",
    "*vision_projection*.linear_fc1",
    "*vision_projection*.linear_fc2",
]


def _nemotron_vl_target_modules(
    *,
    lora_on_language_model: bool = True,
    lora_on_vision_model: bool = True,
) -> list[str]:
    """Return adapter target modules for the selected Nemotron VL components."""
    if not lora_on_language_model and not lora_on_vision_model:
        raise ValueError("At least one of lora_on_language_model or lora_on_vision_model must be True.")

    if lora_on_language_model and lora_on_vision_model:
        return _ALL_COMPONENT_LORA_TARGET_MODULES.copy()

    if lora_on_language_model:
        return _LANGUAGE_LORA_TARGET_MODULES.copy()

    return _VISION_LORA_TARGET_MODULES.copy()


def _nemotron_vl_lora_config(
    *,
    lora_on_language_model: bool = True,
    lora_on_vision_model: bool = True,
) -> VLMLoRA:
    """Build a Nemotron VL LoRA config that respects component selection flags."""
    target_modules = _nemotron_vl_target_modules(
        lora_on_language_model=lora_on_language_model,
        lora_on_vision_model=lora_on_vision_model,
    )

    if lora_on_language_model and lora_on_vision_model:
        return VLMLoRA(
            target_modules=target_modules,
            dim=16,
            alpha=32,
        )

    if lora_on_language_model:
        return VLMLoRA(
            target_modules=target_modules,
            dim=16,
            alpha=32,
            freeze_vision_model=False,
            freeze_vision_projection=False,
        )

    return VLMLoRA(
        target_modules=target_modules,
        dim=16,
        alpha=32,
        freeze_language_model=False,
    )


# =============================================================================
# Nemotron Nano V2 VL 12B SFT Configuration
# =============================================================================
def nemotron_nano_v2_vl_12b_sft_config(
    hf_model_path: str = _DEFAULT_HF_MODEL_PATH,
    pretrained_checkpoint: str | None = None,
) -> ConfigContainer:
    """Return a full SFT config for Nemotron Nano V2 VL 12B.

    Default configuration: 1 node, 8 GPUs
    - TP=4, PP=1
    - LR=1e-5 (finetune default)
    - Sequence length: 4096

    Args:
        hf_model_path: Hugging Face model path used for provider and processor setup.
        pretrained_checkpoint: Optional checkpoint path to load before finetuning.
    """
    cfg = _sft_common_vlm()

    # Model configuration
    hf_path = hf_model_path
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)
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
    cfg.train.train_iters = 2000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 0

    # Optimizer - finetune defaults
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=5,
        lr_decay_iters=None,
        max_lr=2e-5,
        min_lr=2e-6,
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

    # DDP settings - Nemotron uses average_in_collective=False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Checkpoint config - override save_interval from common
    cfg.checkpoint.save_interval = 200
    if pretrained_checkpoint is not None:
        cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint

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
# Nemotron Nano V2 VL 12B PEFT Configuration
# =============================================================================
def nemotron_nano_v2_vl_12b_peft_config(
    peft_scheme: str | PEFT = "lora",
    *,
    hf_model_path: str = _DEFAULT_HF_MODEL_PATH,
    pretrained_checkpoint: str | None = None,
    lora_on_language_model: bool = True,
    lora_on_vision_model: bool = True,
) -> ConfigContainer:
    """Return a PEFT config for Nemotron Nano V2 VL 12B.

    Default configuration: 1 node, 8 GPUs
    - TP=2, PP=1
    - LR=5e-5 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme - "lora", "dora", or a custom PEFT instance.
            Note: Default uses VLMLoRA targeting all model components.
        hf_model_path: Hugging Face model path used for provider and processor setup.
        pretrained_checkpoint: Optional checkpoint path to load before finetuning.
        lora_on_language_model: Whether LoRA targets the language model when PEFT is "lora" or "dora".
        lora_on_vision_model: Whether LoRA targets the vision model when PEFT is "lora" or "dora".
    """
    cfg = _peft_common_vlm()

    # PEFT scheme - Nemotron uses VLMLoRA by default
    if isinstance(peft_scheme, str) and peft_scheme.lower() == "lora":
        cfg.peft = _nemotron_vl_lora_config(
            lora_on_language_model=lora_on_language_model,
            lora_on_vision_model=lora_on_vision_model,
        )
    elif isinstance(peft_scheme, str) and peft_scheme.lower() == "dora":
        cfg.peft = DoRA(
            target_modules=_nemotron_vl_target_modules(
                lora_on_language_model=lora_on_language_model,
                lora_on_vision_model=lora_on_vision_model,
            ),
            dim=16,
            alpha=32,
        )
    else:
        cfg.peft = peft_scheme

    # Model configuration
    hf_path = hf_model_path
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # Parallel settings - lower TP for PEFT
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
    cfg.train.train_iters = 2000
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 500
    cfg.validation.eval_iters = 0

    # Optimizer - PEFT LR settings
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=5,
        lr_decay_iters=None,
        max_lr=2e-5,
        min_lr=2e-6,
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

    # DDP settings - Nemotron uses average_in_collective=False
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"

    # Checkpoint config - override save_interval from common
    cfg.checkpoint.save_interval = 200
    if pretrained_checkpoint is not None:
        cfg.checkpoint.pretrained_checkpoint = pretrained_checkpoint

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
