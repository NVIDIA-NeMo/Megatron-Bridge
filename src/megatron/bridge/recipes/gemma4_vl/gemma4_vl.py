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

"""Gemma 4 VL finetuning recipes.

This module provides SFT and PEFT configurations for Gemma 4 VL 26B-A4B (MoE VLM).
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer

_HF_PATH = "google/gemma-4-26B-A4B-it"


def _apply_gemma4_vl_common(cfg: ConfigContainer, hf_path: str) -> None:
    """Apply settings common to all Gemma 4 VL 26B-A4B recipes."""
    cfg.model = AutoBridge.from_hf_pretrained(hf_path).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    # MoE efficiency kernels
    cfg.model.moe_token_dispatcher_type = "alltoall"
    cfg.model.moe_grouped_gemm = True
    # EP=1 by default for single-node (8 GPU): DP=world/(TP×PP) must be ≥ EP.
    # For multi-node (32+ GPUs, TP=4, PP=2, DP≥4), override: model.expert_model_parallel_size=4
    cfg.model.expert_model_parallel_size = 1
    cfg.model.sequence_parallel = True  # Required: Megatron MoE + TP mandates sequence_parallel

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

    # Memory saving (disabled by default; enable recompute for larger batches)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training config
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation config
    cfg.validation.eval_interval = 5
    cfg.validation.eval_iters = 10

    # Optimizer precision settings (full precision)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    # Dataset configuration
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path
    # Packing requires micro_batch_size > 1; disable for MBS=1 default
    cfg.dataset.pack_sequences_in_batch = False

    # DDP settings — VLMs require no overlap
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.mixed_precision = "bf16_mixed"


# =============================================================================
# Gemma 4 VL 26B-A4B SFT Configuration
# =============================================================================
def gemma4_vl_26b_sft_config(hf_path: str = _HF_PATH) -> ConfigContainer:
    """Return a full SFT config for Gemma 4 VL 26B-A4B (MoE VLM).

    Default configuration: 2 nodes, 16 GPUs
    - TP=2, PP=1, EP=8 (max EP with 16 GPUs at TP=2,PP=1; DP=8, EP divides DP)
    - No activation recompute — EP=8 shards 87.5% of expert params per GPU
    - LR=5e-5 (full SFT)
    - Sequence length: 4096
    """
    cfg = _sft_common_vlm()

    _apply_gemma4_vl_common(cfg, hf_path)

    # Parallel settings — TP=2, PP=1, EP=8 on 2×8 GPUs
    # DP = 16/(TP*PP) = 8; EP=8 shards experts across all DP ranks (1 expert replica)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8  # override common EP=1

    # Reduce overhead to fit within 30-min wall time.
    # 40 iters × ~15s + 5 min init + 4 evals × 35s = ~20 min → 10 min for checkpoint save.
    cfg.train.train_iters = 40         # override common (was 50)
    cfg.validation.eval_interval = 10  # override common (was 5)
    cfg.validation.eval_iters = 5      # override common (was 10)

    # Optimizer — lower LR for full SFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=0.00005,
        min_lr=0.000005,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Re-apply optimizer precision (overwritten by optimizer factory)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    return cfg


# =============================================================================
# Gemma 4 VL 26B-A4B PEFT Configuration
# =============================================================================
def gemma4_vl_26b_peft_config(
    peft_scheme: str | PEFT = "lora",
    hf_path: str = _HF_PATH,
) -> ConfigContainer:
    """Return a PEFT (LoRA/DoRA) config for Gemma 4 VL 26B-A4B (MoE VLM).

    Default configuration: 1 node, 8 GPUs
    - TP=4, PP=1, EP=4 (PEFT needs less memory, drop PP)
    - LR=2e-4 (PEFT)
    - Sequence length: 4096

    Args:
        peft_scheme: PEFT scheme — "lora", "dora", or a custom PEFT instance.
        hf_path: HuggingFace model ID or local path.
    """
    cfg = _peft_common_vlm()

    # PEFT scheme
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    _apply_gemma4_vl_common(cfg, hf_path)

    # Parallel settings — TP=2, PP=1, EP=4 (splits MoE experts across EP ranks,
    # avoiding duplicate LoRA adapter _extra_state shard keys during checkpoint save)
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 4  # override common EP=1

    # Optimizer — higher LR for PEFT
    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=0.0002,
        min_lr=0.00002,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg

    # Re-apply optimizer precision (overwritten by optimizer factory)
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    return cfg
