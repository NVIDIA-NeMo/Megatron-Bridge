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

"""GB200 pretraining recipe for Nemotron 3 Nano."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


_NEMOTRON_3_NANO_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def nemotron_3_nano_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Return the Nemotron 3 Nano BF16 pretraining config for eight GB200 GPUs.

    The recipe retains the established optimizer, scheduler, routing, and BF16
    contracts. It applies the validated GB200 TP1/EP8 HybridEP topology and
    uses a 4,096-token sequence length for the paired NeMo-CI convergence
    workload.

    Returns:
        GB200 BF16 pretraining configuration.
    """
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(_NEMOTRON_3_NANO_MODEL_ID).to_megatron_provider(load_weights=False)
    # Pretraining may use a tokenizer other than the HF checkpoint tokenizer.
    # Defer the model vocabulary size to the runtime tokenizer, matching the
    # pre-migration MambaModelProvider recipe behavior.
    cfg.model.vocab_size = None
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_NANO_MODEL_ID

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.num_workers = 8
    cfg.dataset.mmap_bin_files = False

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_force_load_balancing = False

    cfg.train.train_iters = 39735
    cfg.train.global_batch_size = 3072
    cfg.train.micro_batch_size = 2
    cfg.train.manual_gc = False
    cfg.train.manual_gc_interval = 0

    cfg.model.transformer_impl = "transformer_engine"

    # Match the validated GB200 performance recipe's TE-scoped graph set.
    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.attention_backend = "fused"
    cfg.model.moe_router_fusion = False
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.apply_rope_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.rerun_state_machine.check_for_nan_in_loss = False

    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.optimizer.lr = 1.6e-3
    cfg.optimizer.weight_decay = 0.1
    cfg.optimizer.min_lr = 1.6e-5
    cfg.scheduler.lr_warmup_iters = 333

    # Keep BF16 compute while reducing gradients in BF16 instead of FP32.
    cfg.mixed_precision = get_mixed_precision_config(cfg.mixed_precision)
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_bootstrap_backend="nccl",
        tp_comm_overlap=False,
    )
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False

    cfg.checkpoint.save_interval = 200
    cfg.checkpoint.ckpt_assume_constant_structure = True
    cfg.checkpoint.dist_ckpt_strictness = "log_all"

    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = False
    cfg.ddp.use_distributed_optimizer = True
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.init_method_std = 0.0173
    cfg.model.use_fused_weighted_squared_relu = True

    return cfg


# NeMo-CI appends ``_pretrain_config`` to MODEL_RECIPE_NAME. This explicit
# alias lets the GB200 release case select the hardware recipe without changing
# the legacy ``nemotron_3_nano_pretrain_config`` default.
nemotron_3_nano_gb200_pretrain_config = nemotron_3_nano_pretrain_8gpu_gb200_bf16_config


__all__ = [
    "nemotron_3_nano_gb200_pretrain_config",
    "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config",
]
