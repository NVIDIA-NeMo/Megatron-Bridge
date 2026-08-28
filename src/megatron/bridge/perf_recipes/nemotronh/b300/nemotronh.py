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
"""B300 performance recipes for NemotronH and Nemotron 3."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.nemotronh.common import (
    _TE_QUANT_CFG_PATH,
    ConfigContainer,
    _apply_nemotron_3_nano_perf_defaults,
    _apply_nemotron_3_super_perf_defaults,
    _apply_nemotron_3_ultra_fsdp_hsdp,
    _apply_nemotron_3_ultra_perf_defaults,
    _benchmark_common,
    _enable_ncclep_mxfp8,
    _nemotron_3_super_nvfp4_precision,
    _perf_precision,
    load_quantization_recipe,
    nemotron_3_nano_pretrain_config,
    nemotron_3_super_pretrain_config,
    nemotron_3_ultra_pretrain_config,
    nemotronh_56b_pretrain_config,
)
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


def nemotronh_56b_pretrain_64gpu_b300_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× B300, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba", "attn"]

    _benchmark_common(cfg)
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
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B300, BF16."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.recompute_modules = ["moe_act", "layernorm"]
    cfg.model.recompute_granularity = "selective"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    _apply_nemotron_3_super_perf_defaults(cfg)
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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B300, MXFP8."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    _apply_nemotron_3_super_perf_defaults(cfg)
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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_b300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B300, NVFP4."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.recompute_modules = ["moe_act", "layernorm"]
    cfg.model.recompute_granularity = "selective"
    cfg.model.quant_recipe = load_quantization_recipe(str(_TE_QUANT_CFG_PATH))

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    _apply_nemotron_3_super_perf_defaults(cfg)
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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def nemotron_3_ultra_pretrain_288gpu_b300_fp8mx_ncclep_config() -> ConfigContainer:
    """Nemotron 3 Ultra pretrain: 288× B300, MXFP8, NCCL EP=8, Megatron-FSDP.

    Uses TP1 / PP9 / CP1 / EP8 / ETP1 / DP32 / EDP4, GBS 256 / MBS 1, and
    sequence length 8192. PP9 evenly partitions Ultra's 108 layers and
    compensates for limiting expert parallelism to a single NVL8 domain.
    """
    num_gpus = 288

    cfg = nemotron_3_ultra_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _apply_nemotron_3_ultra_perf_defaults(cfg)
    _apply_nemotron_3_ultra_fsdp_hsdp(cfg, num_gpus=num_gpus)
    # Four optimizer instances keep each dense FSDP shard within one NVL8 node.
    cfg.ddp.num_distributed_optimizer_instances = 4
    cfg.ddp.outer_dp_sharding_strategy = "optim"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 9
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]

    _enable_ncclep_mxfp8(cfg)

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
        # Transformer Engine settings for static-shape NCCL EP.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_8gpu_b300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B300, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B300, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B300, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    _apply_nemotron_3_nano_perf_defaults(cfg)
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    set_cuda_graph_modules(cfg.model, ["attn", "mamba", "moe_router", "moe_preprocess"])

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg
