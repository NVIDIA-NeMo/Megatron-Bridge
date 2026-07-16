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
"""K-EXAONE MoE H100 training recipes."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.peft.base import PEFT
from megatron.bridge.recipes.common import _peft_common, _pretrain_common, _sft_common
from megatron.bridge.recipes.utils.dataset_utils import default_peft_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.flex_dispatcher_backend import apply_flex_dispatcher_backend
from megatron.bridge.training.mixed_precision import bf16_mixed


_HF_PATH = "LGAI-EXAONE/K-EXAONE-236B-A23B"


def _set_optimizer_precision(cfg: ConfigContainer) -> None:
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32


def _apply_exaone_moe_common(cfg: ConfigContainer) -> None:
    cfg.model = AutoBridge.from_hf_pretrained(_HF_PATH).to_megatron_provider(load_weights=False)
    cfg.tokenizer.tokenizer_model = _HF_PATH

    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.seq_length = 4096

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "deepep"
    cfg.model.moe_hybridep_num_sms = 16
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_padding_for_fp8 = False

    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = "full"
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.attention_backend = None
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None

    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.average_in_collective = True
    cfg.ddp.check_for_nan_in_grad = True
    cfg.ddp.use_distributed_optimizer = True

    cfg.train.manual_gc = True
    cfg.train.manual_gc_interval = 100
    cfg.train.manual_gc_eval = 100
    cfg.mixed_precision = bf16_mixed()

    _set_optimizer_precision(cfg)


def exaone_moe_236b_a23b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return a pre-training config for K-EXAONE 236B-A23B MoE."""
    cfg = _pretrain_common()
    _apply_exaone_moe_common(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 8
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.train.micro_batch_size = 1
    cfg.dataset.num_workers = 8

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)
    return cfg


def exaone_moe_236b_a23b_sft_64gpu_h100_bf16_config() -> ConfigContainer:
    """Return a full SFT config for K-EXAONE 236B-A23B MoE."""
    cfg = _sft_common()
    _apply_exaone_moe_common(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 8
    cfg.model.seq_length = 2048
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 32
    cfg.train.train_iters = 100
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.adam_beta2 = 0.95
    cfg.checkpoint.save_interval = 100

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)
    return cfg


def exaone_moe_236b_a23b_peft_16gpu_h100_bf16_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """Return a PEFT config for K-EXAONE 236B-A23B MoE."""
    cfg = _peft_common()
    _apply_exaone_moe_common(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 4
    cfg.model.pipeline_dtype = None
    cfg.model.seq_length = 2048

    peft_cfg = default_peft_config(peft_scheme)
    if isinstance(peft_scheme, str) and peft_scheme.lower() in ["lora", "dora"]:
        peft_cfg.dim = 8
        peft_cfg.alpha = 16
        peft_cfg.target_modules = ["linear_qkv", "linear_proj"]
    cfg.peft = peft_cfg

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 32
    cfg.train.train_iters = 100
    cfg.validation.eval_interval = 50
    cfg.validation.eval_iters = 10
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 100
    cfg.optimizer.adam_beta2 = 0.95
    cfg.checkpoint.save_interval = 100

    apply_flex_dispatcher_backend(cfg.model, cfg.model.moe_flex_dispatcher_backend)
    return cfg


__all__ = [
    "exaone_moe_236b_a23b_peft_16gpu_h100_bf16_config",
    "exaone_moe_236b_a23b_pretrain_64gpu_h100_bf16_config",
    "exaone_moe_236b_a23b_sft_64gpu_h100_bf16_config",
]
