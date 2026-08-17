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
"""GB200 performance recipes for DeepSeek V4."""

import torch

from megatron.bridge.perf_recipes._common import _benchmark_common
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.recipes.deepseek.gb200.deepseek_v4 import (
    deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.utils.cuda_graph import set_full_iteration_cuda_graph


def deepseek_v4_flash_pretrain_128gpu_gb200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Flash pretrain: 128× GB200, MXFP8, full-iteration CUDA graph."""
    cfg = deepseek_v4_flash_pretrain_64gpu_gb200_fp8mx_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    # The Sheet lists VPP=4, but the measured PP=1 W&B run resolves VPP to None.
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_model_parallel_layout = None
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.attention_backend = "auto"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_num_sms_preprocessing = 108
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_load_balancing_type = "seq_aux_loss"
    cfg.model.moe_aux_loss_coeff = 1.0e-4

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj"]
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []
    cfg.model.fine_grained_offloading_max_inflight_offloads = None

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None

    set_full_iteration_cuda_graph(cfg.model)
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True
    cfg.train.manual_gc_interval = 10

    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_page_size = 64
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.use_transformer_engine_op_fuser = True

    cfg.model.csa_compress_rotary_base = 40_000
    cfg.model.rotary_scaling_factor = 4
    cfg.model.apply_dsa_kernel_fusion = True
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.quant_recipe = None
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.model.moe_router_padding_for_quantization = True

    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.optimizer_offload_fraction = 1.0
    cfg.optimizer.offload_optimizer_states = False

    cfg.ddp.overlap_param_gather = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.ddp.average_in_collective = False
    cfg.comm_overlap.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.comm_overlap.delay_wgrad_compute = False

    cfg.checkpoint.load_optim = False
    cfg.checkpoint.load_rng = False
    cfg.checkpoint.save_optim = False

    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg
