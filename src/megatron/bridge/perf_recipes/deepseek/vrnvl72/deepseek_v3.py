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
    """DeepSeek V3 pretrain: 256× VRNVL72, MXFP8, NCCL EP, with tuned Rubin execution settings.

    Starts from the shared GB300 MXFP8 base and keeps its training setup: mock data, NullTokenizer,
    GBS=4096, the LR schedule, forced load balancing and the auxiliary-loss coefficient. Uses
    PP=2 x VP=8, BF16 routing and MXFP8 forward dispatch/backward combine with the Rubin benchmark's
    receive and stash buffer sizing. Fusions, CUDA graphs, runtime and process environment retain the
    MLPerf DeepSeek-V3 execution settings.

    MXFP8 dispatch and combine require Megatron Core support for quantized communication with
    1F1B EP overlap and paged stashing. Megatron Bridge does not expose MLPerf's DataLoader prefetch factor.
    """
    cfg = _build_deepseek_v3_gb300_fp8mx()

    # Keep the inherited 16-stage layout "Et*4|(t*4|)*14tmL" over PP=2 x VP=8:
    # dense DP=128, expert DP=4 and 32 microbatches per DP rank at GBS=4096.
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8

    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    # MLPerf passes 64 SMs in the config; the NCCL_EP_* SM counts below are its per-phase settings.
    # Match the receive capacity and stash buffer sizing of the Rubin benchmark.
    cfg.model.moe_flex_dispatcher_num_sms = 64
    cfg.model.moe_dispatch_fwd_dtype = "mxfp8"
    cfg.model.moe_combine_bwd_dtype = "mxfp8"
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_router_dtype = "bf16"
    cfg.model.moe_router_padding_for_quantization = True

    # Fusions: residual add with RMSNorm, input norm with the MLA down projections, and the MLA
    # key/value RoPE with MXFP8 quantization (Megatron-LM #8144).
    cfg.model.fused_residual_rmsnorm = True
    cfg.model.mla_down_proj_fusion = True
    cfg.model.use_fused_mla_kv_rope_quant = True
    # Run the shared expert through the same CuTe DSL grouped SwiGLU path as the routed experts.
    cfg.model.use_grouped_gemm_for_shared_expert = True
    cfg.model.moe_shared_expert_glu_interleave_size = 32

    # CUDA graphs and runtime.
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.optimizer.optimizer_cuda_graph = True
    cfg.dist.use_gloo_process_groups = False
    cfg.train.manual_gc_interval = 500
    cfg.train.manual_gc_eval = False
    cfg.train.check_optimizer_step_success = False
    # The attention backend builds the causal mask, so the dataloader skips building one.
    cfg.dataset.create_attention_mask = False

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        # MLPerf inherits 1 from the Megatron Bridge library recipe environment.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # Keep per-group NCCL communicators instead of world-wide splits for every new group.
        "TORCH_DIST_USE_NCCL2": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # NCCL EP SM budgets per phase and fused dispatch preparation.
        "NCCL_EP_COMBINE_SMS": 24,
        "NCCL_EP_DISPATCH_SMS": 32,
        "NCCL_EP_HT_UNFUSED_SYNC": 1,
        "NCCL_EP_PREPROCESS_NUM_SMS": 80,
        "NCCL_EP_PROLOG_EPILOG_SMS": 148,
        "NCCL_EP_SHUFFLE_SMS": 128,
        "NVTE_EP_FUSED_PREPARE_DISPATCH": 1,
        # Transformer Engine overlap settings; MLPerf reserves no SMs for LayerNorm kernels.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_EXT_MARGIN_SM": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        # Use cuDNN LayerNorm and fuse the MXFP8 quantization of its output.
        "NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE": 1,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_MXFP8_PER_TILE_SWIZZLE": 1,
        # MLPerf allows nondeterministic Transformer Engine kernels.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
    }
    return cfg


def deepseek_v3_pretrain_64gpu_vrnvl72_fp8mx_proxy_config() -> ConfigContainer:
    """DeepSeek V3 MXFP8 debugging proxy: 13 decoder layers on 64 VRNVL72 GPUs.

    Shrinks ``deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config`` for bring-up on fewer GPUs. Keeps
    GBS=4096, per-layer shapes, EP=32, one MTP layer and the parent's fusions, CUDA graphs,
    runtime and environment. Uses MXFP8 forward dispatch and the parent's receive-capacity
    factor 1.5 and CUDA stash-buffer factor 1.2. PP=1 without virtual pipelining
    gives dense DP=64, expert DP=2 and 64 microbatches. This reduced-depth model does
    not reproduce the parent's PP=2 x VP=8 schedule,
    at-scale pipeline traffic, convergence or memory use; GPU validation is required.
    """
    cfg = deepseek_v3_pretrain_256gpu_vrnvl72_fp8mx_config()
    cfg.model.num_layers = 13
    # Retain DeepSeek V3's three leading dense layers; shorten its MoE pattern.
    cfg.model.moe_layer_freq = [0] * 3 + [1] * (cfg.model.num_layers - 3)
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Clear the inherited interleaved layout; embedding, decoder and MTP are colocated.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # Use MXFP8 forward dispatch and match the parent's buffer sizing.
    cfg.model.moe_dispatch_fwd_dtype = "mxfp8"
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2

    # Keep this explicit environment identical to the 256-GPU parent.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        # MLPerf inherits 1 from the Megatron Bridge library recipe environment.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # Keep per-group NCCL communicators instead of world-wide splits for every new group.
        "TORCH_DIST_USE_NCCL2": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # NCCL EP SM budgets per phase and fused dispatch preparation.
        "NCCL_EP_COMBINE_SMS": 24,
        "NCCL_EP_DISPATCH_SMS": 32,
        "NCCL_EP_HT_UNFUSED_SYNC": 1,
        "NCCL_EP_PREPROCESS_NUM_SMS": 80,
        "NCCL_EP_PROLOG_EPILOG_SMS": 148,
        "NCCL_EP_SHUFFLE_SMS": 128,
        "NVTE_EP_FUSED_PREPARE_DISPATCH": 1,
        # Transformer Engine overlap settings; MLPerf reserves no SMs for LayerNorm kernels.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 0,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_EXT_MARGIN_SM": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 0,
        # Use cuDNN LayerNorm and fuse the MXFP8 quantization of its output.
        "NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE": 1,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_MXFP8_PER_TILE_SWIZZLE": 1,
        # MLPerf allows nondeterministic Transformer Engine kernels.
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
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
