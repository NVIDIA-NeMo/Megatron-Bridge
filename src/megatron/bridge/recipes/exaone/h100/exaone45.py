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
"""EXAONE 4.5 VL H100 training recipes."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common_vlm, _sft_common_vlm
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.bridge.training.config import ConfigContainer


_HF_PATH = "LGAI-EXAONE/EXAONE-4.5-33B"


def _set_optimizer_precision(cfg: ConfigContainer) -> None:
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32


def _apply_exaone45_common(cfg: ConfigContainer) -> None:
    cfg.model = AutoBridge.from_hf_pretrained(_HF_PATH).to_megatron_provider(load_weights=False)
    cfg.model.seq_length = 4096

    cfg.model.pipeline_dtype = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False
    cfg.model.freeze_mtp_model = False

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = "auto"
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.dataset.seq_length = 4096
    cfg.dataset.hf_processor_path = _HF_PATH

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.mixed_precision = "bf16_mixed"

    _set_optimizer_precision(cfg)


def exaone45_vl_33b_sft_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for EXAONE 4.5 VL 33B."""
    cfg = _sft_common_vlm()
    _apply_exaone45_common(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 32

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=5e-6,
        min_lr=1e-6,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    _set_optimizer_precision(cfg)

    return cfg


def exaone45_vl_33b_peft_4gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for EXAONE 4.5 VL 33B."""
    cfg = _peft_common_vlm()
    cfg.peft = default_peft_config(peft_scheme)
    _apply_exaone45_common(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 32

    opt_cfg, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
        lr_warmup_iters=500,
        lr_decay_iters=300000,
        max_lr=1e-4,
        min_lr=1e-5,
    )
    cfg.optimizer = opt_cfg
    cfg.scheduler = scheduler_cfg
    _set_optimizer_precision(cfg)

    return cfg


__all__ = [
    "exaone45_vl_33b_peft_4gpu_h100_bf16_config",
    "exaone45_vl_33b_sft_16gpu_h100_bf16_config",
]
