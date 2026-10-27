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

"""GB200 training recipes for Nemotron 3 Super.

For BF16 finetuning through ``scripts/training/run_recipe.py``, select
``--recipe nemotron_3_super_sft_64gpu_gb200_bf16_config --mode sft`` on 64 GPUs
or ``--recipe nemotron_3_super_peft_16gpu_gb200_bf16_config --mode lora`` on
16 GPUs, together with a pretrained checkpoint and an SFT dataset. These
recipes include the HybridEP NVL72 environment; no topology environment
variable overrides are needed when all ranks occupy one NVL72 domain.
"""

import torch

from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.nemotronh.h100.nemotron_3_super import (
    nemotron_3_super_peft_16gpu_h100_bf16_config,
    nemotron_3_super_pretrain_16gpu_h100_bf16_config,
    nemotron_3_super_sft_16gpu_h100_bf16_config,
)
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_mixed
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def nemotron_3_super_pretrain_64gpu_gb200_bf16_config(
    *,
    hf_path: str | None = None,
    text_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Return the Nemotron 3 Super BF16 pretraining config for 64 GB200 GPUs.

    This is the convergence-oriented counterpart of the canonical 64-GPU
    performance recipe. It uses the same GB200 parallel layout, HybridEP
    dispatcher, CUDA graph scopes, and process environment while retaining
    natural expert routing, runtime validation, and checkpointing.

    Returns:
        GB200 BF16 pretraining configuration.
    """
    cfg = nemotron_3_super_pretrain_16gpu_h100_bf16_config(
        hf_path=hf_path, text_only=text_only, revision=revision, trust_remote_code=trust_remote_code
    )

    cfg.mixed_precision = bf16_mixed()
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    # The H100 base is memory-bounded for 16-GPU support runs. GB200 has the
    # capacity for overlapped collectives and full-precision optimizer state.
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    # The H100 base carries the bounded 100-step verification schedule. This
    # GB200 recipe is the convergence counterpart of the 64-GPU performance
    # recipe and owns its own full-length schedule.
    cfg.train.train_iters = 39735
    cfg.scheduler.lr_warmup_iters = 333
    cfg.scheduler.lr_decay_iters = None

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    # Forced routing is benchmark-only and would change the training objective.
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = 32

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.apply_rope_fusion = True
    cfg.model.recompute_granularity = None

    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.optimizer_offload_fraction = 0.0

    cfg.env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_super_sft_64gpu_gb200_bf16_config(
    *,
    hf_path: str | None = None,
    text_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Return the BF16 SFT config for 64 GB200 GPUs in one NVL72 domain.

    Retains the H100 recipe's training objective, batch size, optimizer and
    checkpoint policy while selecting TP8/EP64 and multi-node NVLink HybridEP.
    The allocation must place all expert-parallel ranks in one NVLink domain.

    Args:
        hf_path: Hugging Face model identifier or local source directory.
        text_only: Use only the language model of a multimodal source.
        revision: Immutable Hugging Face source revision.
        trust_remote_code: Allow custom code from the selected source.

    Returns:
        GB200 SFT configuration with its HybridEP environment included.
    """
    cfg = nemotron_3_super_sft_16gpu_h100_bf16_config(
        hf_path=hf_path, text_only=text_only, revision=revision, trust_remote_code=trust_remote_code
    )
    cfg.model.expert_model_parallel_size = 64
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.env_vars.update(
        NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=64,
        NVLINK_DOMAIN_SIZE=72,
        USE_MNNVL=1,
    )
    return cfg


def nemotron_3_super_peft_16gpu_gb200_bf16_config(
    peft_scheme: str | PEFT = "lora",
    *,
    hf_path: str | None = None,
    text_only: bool = False,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Return the BF16 PEFT config for 16 GB200 GPUs in one NVL72 domain.

    Retains the H100 recipe's adapters, training objective, optimizer and
    checkpoint policy while selecting multi-node NVLink HybridEP for TP8/EP16.
    The allocation must place all expert-parallel ranks in one NVLink domain.

    Args:
        peft_scheme: LoRA, DoRA, or a custom PEFT configuration.
        hf_path: Hugging Face model identifier or local source directory.
        text_only: Use only the language model of a multimodal source.
        revision: Immutable Hugging Face source revision.
        trust_remote_code: Allow custom code from the selected source.

    Returns:
        GB200 PEFT configuration with its HybridEP environment included.
    """
    cfg = nemotron_3_super_peft_16gpu_h100_bf16_config(
        peft_scheme, hf_path=hf_path, text_only=text_only, revision=revision, trust_remote_code=trust_remote_code
    )
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.env_vars.update(
        NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16,
        NVLINK_DOMAIN_SIZE=72,
        USE_MNNVL=1,
    )
    return cfg


__all__ = [
    "nemotron_3_super_pretrain_64gpu_gb200_bf16_config",
    "nemotron_3_super_sft_64gpu_gb200_bf16_config",
    "nemotron_3_super_peft_16gpu_gb200_bf16_config",
]
