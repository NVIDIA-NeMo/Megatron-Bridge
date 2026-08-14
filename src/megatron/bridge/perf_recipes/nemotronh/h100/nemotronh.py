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
"""H100 performance recipes for NemotronH and Nemotron 3."""

import torch

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.nemotronh.common import (
    ConfigContainer,
    _apply_nemotron_3_nano_perf_defaults,
    _benchmark_common,
    _perf_precision,
    nemotron_3_nano_pretrain_config,
    nemotronh_56b_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_pretrain_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


# Public Nemotron 3.5 Lightning checkpoint used by the Lightning recipe API.
_NEMOTRON_3_5_LIGHTNING_MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_NEMOTRON_3_5_LIGHTNING_MODEL_REVISION = "b3caaabed0263651a17dc1f2d4ce97e794f76c44"  # pragma: allowlist secret


def nemotronh_56b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× H100, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba"]

    _benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× H100, BF16."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    # The library recipe owns the measured execution layout. Performance mode
    # adds only benchmark policy and forced routing on top of that shared base.
    cfg.model.moe_router_force_load_balancing = True
    _benchmark_common(cfg)
    # `_benchmark_common` enables parameter prefetch globally. Restore the
    # shared natural-routing-safe execution mapping owned by the library
    # recipe; otherwise performance and convergence silently diverge here.
    cfg.ddp.overlap_param_gather = False
    cfg.model.moe_hybridep_num_sms = None
    cfg.checkpoint.async_save = False

    # Keep the process settings aligned with the convergence recipe. In
    # particular, do not silently change CUDA stream scheduling between the
    # natural-routing and forced-routing measurements.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, BF16, recompute MoE+layernorm."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba"])

    cfg.model.recompute_modules = ["moe", "layernorm"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, FP8 current-scaling, recompute."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 16
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba"])

    cfg.model.recompute_modules = ["layernorm"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 16
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_5_lightning_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3.5 Lightning benchmark pretrain: 16× H100, BF16."""
    cfg = nemotron_3_nano_pretrain_16gpu_h100_bf16_config()
    # Keep the benchmark workload aligned with the GB200 BF16 recipe. The
    # hardware recipes may tune execution-only knobs such as microbatch size,
    # recompute, and CUDA graph coverage independently.
    cfg.train.global_batch_size = 512
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.model.hf_model_revision = _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}

    # This mock-data, force-balanced benchmark uses lower-precision Adam moments
    # to make the measured selective-recompute and activation-offload stack fit.
    # It is not convergence-equivalent to recipes with FP32 optimizer states.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    cfg.model.recompute_modules = ["moe_act", "layernorm"]
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["expert_fc1"]
    cfg.model.activation_offload_fraction = 1.0
    cfg.model.delay_offload_until_cuda_graph = True

    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion_into_hybridep = True
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 10,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 10,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_5_lightning_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Nemotron 3.5 Lightning pretrain: 16× H100, FP8 current-scaling."""
    cfg = nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config()
    cfg.model.expert_model_parallel_size = 8
    cfg.model.mtp_num_layers = 2
    cfg.model.mtp_hybrid_override_pattern = "*E"
    cfg.model.mtp_use_repeated_layer = True
    cfg.model.keep_mtp_spec_in_bf16 = True
    cfg.model.mtp_loss_scaling_factor = 0.3
    cfg.model.hf_model_id = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.model.hf_model_revision = _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION
    cfg.tokenizer.tokenizer_model = _NEMOTRON_3_5_LIGHTNING_MODEL_ID
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _NEMOTRON_3_5_LIGHTNING_MODEL_REVISION}
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg
