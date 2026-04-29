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

"""Step3-VL SFT and PEFT recipe configurations.

Supported models:
    stepfun-ai/Step3-VL-10B   (~10B params, Qwen3-8B LLM + custom 1.8B ViT)
"""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.finetune_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer

_HF_PATH_10B = "stepfun-ai/Step3-VL-10B"


def _step3_vl_apply_common(cfg: ConfigContainer, hf_path: str) -> None:
    """Apply common settings for all Step3-VL-10B recipes."""
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.model.seq_length = 4096

    # VLM-specific
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.cp_comm_type = "a2a"

    # Transformer Engine
    cfg.model.transformer_impl = "transformer_engine"

    # CUDA Graph (disabled by default for VLMs)
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3

    # Kernels
    cfg.model.attention_backend = "flash"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"

    # Memory saving (off by default)
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    # Training
    cfg.train.train_iters = 50
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1
    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100

    # Validation
    cfg.validation.eval_interval = 5
    cfg.validation.eval_iters = 10

    # Dataset
    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = hf_path

    # DDP (VLMs require no overlap)
    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    # Mixed precision
    cfg.mixed_precision = "bf16_mixed"


# =============================================================================
# Step3-VL-10B  SFT
# =============================================================================
def step3_vl_10b_sft_config() -> ConfigContainer:
    """Return a full SFT config for stepfun-ai/Step3-VL-10B.

    Default: 1 node, 8 GPUs — TP=2, PP=1, seq_len=4096.
    """
    cfg = _sft_common_vlm()
    _step3_vl_apply_common(cfg, _HF_PATH_10B)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=5e-5,
        min_lr=5e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    return cfg


# =============================================================================
# Step3-VL-10B  PEFT
# =============================================================================
def step3_vl_10b_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT (LoRA/DoRA) config for stepfun-ai/Step3-VL-10B.

    Default: 1 node, 8 GPUs — TP=1, PP=1, seq_len=4096.

    Args:
        peft_scheme: ``"lora"``, ``"dora"``, or a custom PEFT instance.
    """
    cfg = _peft_common_vlm()

    if isinstance(peft_scheme, str) and peft_scheme.lower() in ("lora", "dora"):
        cfg.peft = default_peft_config(peft_scheme)
    else:
        cfg.peft = peft_scheme

    _step3_vl_apply_common(cfg, _HF_PATH_10B)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=10,
        lr_decay_iters=50,
        max_lr=2e-4,
        min_lr=2e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32

    return cfg
