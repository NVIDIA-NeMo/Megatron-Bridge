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
"""VRNVL72 performance recipes for DeepSeek V3."""

from megatron.bridge.perf_recipes._common import _enable_ncclep
from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _benchmark_common,
    _deepseek_v3_common,
    _enable_deepseek_full_iteration,
    _perf_precision,
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.perf_recipes.deepseek.gb300.deepseek_v3 import (
    _build_deepseek_v3_gb300_bf16,
    _build_deepseek_v3_gb300_fp8mx,
    _build_deepseek_v3_gb300_nvfp4,
    deepseek_v3_pretrain_256gpu_gb300_fp8cs_config,
)
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS


def deepseek_v3_pretrain_128gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VRNVL72, BF16, NCCL EP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
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
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_128gpu_vrnvl72_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VRNVL72, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

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
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_128gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VRNVL72, MXFP8, NCCL EP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_deepseek_full_iteration(cfg)
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    cfg.model.fp8_output_proj = False
    cfg.mixed_precision.fp8_dot_product_attention = False
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_128gpu_vrnvl72_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× VRNVL72, NVFP4, NCCL EP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 2048
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["mlp"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    _benchmark_common(cfg)
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
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
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VRNVL72, BF16, NCCL EP (alias of GB300)."""
    cfg = _build_deepseek_v3_gb300_bf16()
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    cfg.model.cuda_graph_scope = ["attn"]

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
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_vrnvl72_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VRNVL72, FP8-CS (alias of GB300)."""
    cfg = deepseek_v3_pretrain_256gpu_gb300_fp8cs_config()
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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VRNVL72, FP8-MX, NCCL EP (alias of GB300)."""
    cfg = _build_deepseek_v3_gb300_fp8mx()
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_64gpu_vrnvl72_fp8mx_proxy_config() -> ConfigContainer:
    """DeepSeek V3 MXFP8 debugging proxy: 13 decoder layers on 64 VRNVL72 GPUs.

    Keep GBS=4096, per-layer shapes, EP=32 and one MTP layer. PP=1 without
    virtual pipelining gives dense DP=64, expert DP=2 and 64 microbatches.
    Retain full-iteration graphs and the non-pipelined combined-1F1B MoE
    overlap path. This reduced-depth model does not reproduce at-scale
    pipeline traffic, convergence or memory use; GPU validation is required.
    """
    cfg = deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config()
    cfg.model.num_layers = 13
    # Retain DeepSeek V3's three leading dense layers; shorten its MoE pattern.
    cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Clear the inherited interleaved layout; embedding, decoder and MTP are colocated.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # Keep this explicit environment identical to the 256-GPU parent.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× VRNVL72, NVFP4 with full-iteration CUDA graph, NCCL EP."""
    cfg = _build_deepseek_v3_gb300_nvfp4()
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
        "NVTE_DPA_FP8_FORMAT": "E4M3",
    }
    return cfg


def deepseek_v3_pretrain_64gpu_vrnvl72_nvfp4_proxy_config() -> ConfigContainer:
    """DeepSeek V3 NVFP4 debugging proxy: 13 decoder layers on 64 VRNVL72 GPUs.

    Keep GBS=4096, per-layer shapes, EP=32 and one MTP layer. PP=1 without
    virtual pipelining gives dense DP=64, expert DP=2 and 64 microbatches.
    Retain full-iteration graphs and the non-pipelined combined-1F1B MoE
    overlap path. Use this reduced-depth model for feature and kernel
    performance experiments, not convergence or at-scale acceptance.
    """
    cfg = deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config()
    cfg.model.num_layers = 13
    cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Clear the inherited interleaved layout; embedding, decoder and MTP are colocated.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # Keep this explicit environment identical to the 256-GPU parent.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NVTE_USE_FAST_MATH": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
        "NVTE_DPA_FP8_FORMAT": "E4M3",
    }
    return cfg


def _apply_deepseek_v3_4gpu_vrnvl72_proxy(cfg: ConfigContainer) -> None:
    """Shrink a 256-GPU VRNVL72 DeepSeek V3 recipe onto one 4-GPU VRNVL72 node.

    Both 4-GPU proxies go through this helper so that every model, parallelism and batch
    setting matches and only the quantization recipe differs between their profiles.
    """
    # Three leading dense layers and ten MoE layers plus one MTP layer, as in the 64-GPU proxies.
    cfg.model.num_layers = 13
    cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
    # EP=4 over 32 experts keeps the parent's per-rank expert work (256 experts over EP=32):
    # 8 local experts, each receiving 4096 tokens per microbatch under forced load balancing.
    cfg.model.num_moe_experts = 32
    cfg.model.expert_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Clear the inherited interleaved layout; embedding, decoder and MTP are colocated.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)
    # Dense DP=4 and expert DP=1; 32 microbatches per rank, as in the 256-GPU parent.
    cfg.train.global_batch_size = 128

    # The parents differ in these two knobs; pin both so the proxies share one graph structure.
    # MXFP8 is the only recipe with a quantized LM head, so keep the LM head in BF16 for both.
    cfg.model.fp8_output_proj = False
    # Fused input LayerNorm and q/kv down-projection works with either recipe (NVFP4 parent setting).
    cfg.model.mla_down_proj_fusion = True


def deepseek_v3_pretrain_4gpu_vrnvl72_fp8mx_proxy_config() -> ConfigContainer:
    """DeepSeek V3 MXFP8 profiling proxy: 13 decoder layers and 32 experts on one 4-GPU VRNVL72 node.

    Matches ``deepseek_v3_pretrain_4gpu_vrnvl72_nvfp4_proxy_config`` in everything except the
    quantization recipe, so the two profiles can be compared directly. Per-layer shapes and
    per-rank expert work follow the 256-GPU parent: TP=1, PP=1, EP=4, MBS=1, GBS=128, one MTP
    layer, full-iteration CUDA graphs, NCCL EP and combined-1F1B MoE overlap. Attention and all
    GEMMs run in MXFP8; the LM head stays in BF16. This proxy does not reproduce at-scale
    communication, convergence or memory use.
    """
    cfg = deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config()
    _apply_deepseek_v3_4gpu_vrnvl72_proxy(cfg)

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    # Identical to the NVFP4 proxy apart from the NVFP4-only entries at the end of that recipe.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm in both 4-GPU proxies.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Keep DeepSeek kernel selection aligned with the 256-GPU parent recipes.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg


def deepseek_v3_pretrain_4gpu_vrnvl72_nvfp4_proxy_config() -> ConfigContainer:
    """DeepSeek V3 NVFP4 profiling proxy: 13 decoder layers and 32 experts on one 4-GPU VRNVL72 node.

    Matches ``deepseek_v3_pretrain_4gpu_vrnvl72_fp8mx_proxy_config`` in everything except the
    quantization recipe, so the two profiles can be compared directly. Per-layer shapes and
    per-rank expert work follow the 256-GPU parent: TP=1, PP=1, EP=4, MBS=1, GBS=128, one MTP
    layer, full-iteration CUDA graphs, NCCL EP and combined-1F1B MoE overlap. GEMMs run in
    NVFP4, attention in MXFP8 E4M3 as in the MXFP8 proxy, and the LM head in BF16. This proxy
    does not reproduce at-scale communication, convergence or memory use.
    """
    cfg = deepseek_v3_pretrain_256gpu_vrnvl72_nvfp4_config()
    _apply_deepseek_v3_4gpu_vrnvl72_proxy(cfg)

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    # Identical to the MXFP8 proxy apart from the NVFP4-only entries at the end.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm in both 4-GPU proxies.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Keep DeepSeek kernel selection aligned with the 256-GPU parent recipes.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
        # Run attention in MXFP8 E4M3, as the MXFP8 proxy does natively.
        "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
        "NVTE_DPA_FP8_FORMAT": "E4M3",
    }
    return cfg
