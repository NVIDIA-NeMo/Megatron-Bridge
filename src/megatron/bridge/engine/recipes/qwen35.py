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
"""Qwen3.5 TrainingEngine recipes.

Model, optimizer, DDP and precision settings are copied from the SFT recipe
``megatron.bridge.recipes.qwen.gb200.qwen35._qwen35_text_35b_a3b_long_context_common`` and
``qwen35_text_35b_a3b_sft_long_context_16gpu_gb200_bf16_config`` (Megatron-Bridge dec652474), minus
the dataset, learning-rate schedule, logging and checkpoint-path sections that the application owns.
"""

from __future__ import annotations

from copy import deepcopy

import torch
from transformers import AutoConfig

from megatron.bridge import AutoBridge
from megatron.bridge.engine.recipes.base import BRIDGE_LORA_TARGET_MODULES, TrainingRecipe
from megatron.bridge.recipes.utils.environment_utils import COMMON_RECIPE_ENV_VARS
from megatron.bridge.training.config import DistributedDataParallelConfig, OptimizerConfig, RNGConfig
from megatron.bridge.training.mixed_precision import bf16_mixed


def qwen35_text_35b_a3b_rl_16gpu_gb200_bf16(
    *,
    hf_path: str,
    hf_revision: str | None = None,
    seq_length: int = 131072,
    pad_to_max_length: bool = True,
    mtp_detach_heads: bool = False,
) -> TrainingRecipe:
    """Qwen3.5-35B-A3B text model, BF16, TP1/CP8/EP16 with one MTP layer, for 16 GB200/GB300 GPUs.

    Args:
        hf_path: HF model ID or local directory holding the text-only causal-LM config; also the layout
            ``export`` writes (config, tokenizer files, weight names).
        hf_revision: Matching HF revision, or ``None`` for a local directory.
        seq_length: Packed token budget per microbatch; a positive multiple of 16 for CP8.
        pad_to_max_length: Pad every pack to ``seq_length`` (one kernel shape) instead of to its content.
        mtp_detach_heads: Train the MTP head without letting its loss reach the main model's gradients
            (RL use); ``False`` is the coupled SFT baseline. MTP depth itself comes from the HF config.

    Returns:
        The engine recipe. Pretrained weights are supplied to the engine through ``checkpoint_path``.
    """
    if seq_length <= 0 or seq_length % 16:
        raise ValueError("seq_length must be positive and divisible by 16 for CP8.")
    hf_config = AutoConfig.from_pretrained(hf_path, revision=hf_revision)
    text_config = deepcopy(getattr(hf_config, "text_config", hf_config))
    if text_config.model_type != "qwen3_5_moe_text":
        raise ValueError("Expected a Qwen3.5 MoE text configuration.")
    if seq_length > text_config.max_position_embeddings:
        raise ValueError("seq_length exceeds the text model's max_position_embeddings.")
    if hasattr(hf_config, "text_config"):
        text_config.tie_word_embeddings = hf_config.tie_word_embeddings
    text_config.architectures = ["Qwen3_5MoeForCausalLM"]

    model = AutoBridge.from_hf_config(text_config).to_megatron_provider(load_weights=False)
    model.gdn_kernel_backend = "transformer_engine"
    model.gdn_pre_gated_delta_rule_fusion = True
    model.seq_length = seq_length
    model.pipeline_dtype = torch.bfloat16
    model.context_parallel_size = 8
    model.expert_model_parallel_size = 16
    model.expert_tensor_parallel_size = 1
    # MTP depth (mtp_num_hidden_layers) and its 0.1 loss weight come from the HF config via the bridge.
    model.mtp_detach_heads = mtp_detach_heads
    model.bias_activation_fusion = True
    model.cross_entropy_fusion_impl = "te"
    model.calculate_per_token_loss = True
    model.recompute_granularity = "selective"
    model.recompute_modules = ["gdn_norm_out", "moe"]
    model.moe_token_dispatcher_type = "flex"
    model.moe_flex_dispatcher_backend = "hybridep"
    model.moe_flex_dispatcher_num_sms = 32
    model.moe_hybridep_pad_uneven_dispatch_inputs = True
    model.moe_router_fusion = True

    # SFT recipe: distributed_fused_adam_with_cosine_annealing(lr_warmup_iters=200, lr_decay_iters=300000,
    # max_lr=2e-5, min_lr=2e-6). The schedule is the caller's (AdamParams per step); defaults kept here.
    optimizer = OptimizerConfig(
        optimizer="adam",
        lr=2e-5,
        min_lr=2e-6,
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        overlap_param_gather=False,
    )
    ddp = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        use_distributed_optimizer=True,
        data_parallel_sharding_strategy="optim_grads_params",
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        average_in_collective=False,
    )
    env_vars = {
        **COMMON_RECIPE_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 16,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
    }
    return TrainingRecipe(
        model=model,
        optimizer=optimizer,
        ddp=ddp,
        mixed_precision=bf16_mixed(),
        rng=RNGConfig(seed=1234),
        env_vars=env_vars,
        pad_to_max_length=pad_to_max_length,
        hf_path=hf_path,
        hf_revision=hf_revision,
        # Bridge's defaults cover the full-attention layers and the MoE experts; the GatedDeltaNet
        # linear-attention layers project through in_proj and out_proj instead. The router is never a target.
        lora_target_modules=[*BRIDGE_LORA_TARGET_MODULES, "in_proj", "out_proj"],
    )
