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
"""GB300 performance recipes for DeepSeek V3."""

from megatron.bridge.perf_recipes._common import _enable_ncclep
from megatron.bridge.perf_recipes.deepseek.common import (
    ConfigContainer,
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs,
    _benchmark_common,
    _deepseek_v3_common,
    _enable_deepseek_full_iteration,
    _enable_deepseek_precision_aware_optimizer,
    _perf_precision,
    deepseek_v3_pretrain_config,
    set_deepseek_v3_pipeline_model_parallel_layout,
)
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS


def _build_deepseek_v3_gb300_bf16() -> ConfigContainer:
    """Shared HybridEP BF16 base for the DeepSeek V3 256-GPU GB300 recipe and its VR200 alias."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = ["moe_act"]

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, BF16, NCCL EP."""
    cfg = _build_deepseek_v3_gb300_bf16()
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
        # PyTorch 2.14 prints a C++ warning on every deprecated CUDAGraph.register_generator_state() call.
        # TE layer-graph capture with PP>1 makes thousands per rank, and the stderr flood can stall ranks
        # past the NCCL timeout, so keep only C++ errors.
        "TORCH_CPP_LOG_LEVEL": "ERROR",
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, FP8 current-scaling."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 2

    cfg.model.recompute_modules = ["mla_up_proj"]

    cfg.model.cuda_graph_scope = []
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def _build_deepseek_v3_gb300_fp8mx() -> ConfigContainer:
    """Shared HybridEP MXFP8 base for the DeepSeek V3 256-GPU GB300 recipe and its VR200 alias."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_modules = []

    cfg.model.cuda_graph_scope = []
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(t*4|)*14tmL")

    _benchmark_common(cfg)
    _enable_deepseek_full_iteration(cfg)
    cfg.model.fp8_output_proj = True
    cfg.mixed_precision.fp8_dot_product_attention = True
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8, NCCL EP."""
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
    }
    return cfg


def _build_deepseek_v3_gb300_nvfp4() -> ConfigContainer:
    """Shared HybridEP NVFP4 base for the DeepSeek V3 256-GPU GB300 recipe and its VR200 alias."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    _deepseek_v3_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 32
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_scope = []
    cfg.ddp.overlap_grad_reduce = True
    cfg.comm_overlap.overlap_grad_reduce = True

    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*5|(t*4|)*14mL")

    _benchmark_common(cfg)
    _enable_deepseek_full_iteration(cfg)
    cfg.model.fp8_output_proj = False
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.model.mla_down_proj_fusion = True

    cfg.model.recompute_modules = []

    return cfg


def deepseek_v3_pretrain_256gpu_gb300_nvfp4_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, NVFP4, NCCL EP."""
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
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_DPA_FP8_RECIPE": "MXFP8BlockScaling",
        "NVTE_DPA_FP8_FORMAT": "E4M3",
    }
    return cfg


def deepseek_v3_pretrain_64gpu_gb300_bf16_fsdp_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× GB300, BF16, Megatron FSDP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
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
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× GB300, MXFP8, Megatron FSDP."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.fp8_output_proj = True
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.model.fp8_param_gather = True
    cfg.model.fp8_param = True
    cfg.model.moe_router_dtype = "bf16"
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
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def _apply_deepseek_v3_64gpu_gb300_fp8mx_perf_overrides(cfg: ConfigContainer) -> None:
    """Port the 256-GPU GB300 HSDP perf tuning onto the 64-GPU GB300 MXFP8 FSDP layout.

    Shared by the Megatron-FSDP v1 and v2 variants so the only difference between them is
    ddp.megatron_fsdp_version. Settings that are specific to the 256-GPU layout are rescaled
    here rather than copied; each such deviation is commented with the reason.
    """
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.fp8_output_proj = True
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)

    cfg.model.gradient_accumulation_fusion = False

    # Set explicitly rather than relying on _validate_and_apply_megatron_fsdp_configs() to force
    # it: the helper only sets ddp.use_megatron_fsdp, and train.py reads dist.use_megatron_fsdp
    # when picking the parameter-norm path. Leaving it False there silently reports the wrong
    # params norm whenever that validator does not run.
    cfg.dist.use_megatron_fsdp = True

    # 64 GPUs leaves no outer data-parallel axis: with a single optimizer instance MFSDP
    # rejects any outer_dp_sharding_strategy other than "no_shard" (it requires
    # num_distributed_optimizer_instances > 1). The 256-GPU recipe's "optim" outer sharding
    # across 4 instances therefore has no equivalent here.
    cfg.ddp.num_distributed_optimizer_instances = 1
    cfg.ddp.outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.expert_outer_dp_sharding_strategy = "no_shard"

    # GBS 256 over DP=64 at mbs=1 reproduces the 4 gradient-accumulation steps the 256-GPU
    # recipe ran at GBS 1024 / DP 256, so per-step comm amortization is comparable.
    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 256

    # Held at the 256-GPU values rather than batch-scaled: these are 50-iteration throughput
    # runs on mock data, and 3e-7 is far below the rate that diverged in earlier runs.
    cfg.optimizer.lr = 3e-7
    cfg.optimizer.min_lr = 1e-7

    cfg.model.fp8_param_gather = True
    cfg.model.fp8_param = True
    cfg.model.moe_router_dtype = "bf16"
    cfg.model.average_in_collective = True
    cfg.ddp.average_in_collective = True

    # Full-iteration CUDA graph with dropless MoE padding + paged stashing.
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.overlap_dispatch_backward_with_experts_wgrad = False
    cfg.ddp.megatron_fsdp_cuda_graph_mode = True
    cfg.ddp.fsdp_all_gather_in_start_param_sync = False

    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1

    # Recompute stays off, matching the 256-GPU recipe. core_attn offloading is deliberately
    # kept, where the 256-GPU recipe clears offload_modules entirely: the expert data-parallel
    # group is world/EP, which is 4 at 256 GPUs but 1 at 64, so expert optimizer state is not
    # sharded at all here and the headroom that let the larger run drop offloading is absent.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = []
    cfg.model.offload_modules = ["core_attn"]

    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.fused_residual_rmsnorm = True
    cfg.model.moe_hybridep_num_sms_preprocessing = 32

    # Owned by CommOverlapConfig, not the model or DDP config: _apply_cfgs() copies the
    # comm_overlap values onto those after the recipe runs, so setting them elsewhere is
    # silently overwritten by the comm_overlap defaults.
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.align_param_gather = True

    # CuTeDSL fused grouped MLP (moe_a2a_overlap disabled).
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    cfg.model.mla_down_proj_fusion = True
    cfg.mixed_precision.fp8_dot_product_attention = False
    cfg.model.moe_router_force_load_balancing = True

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE": 0,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system. All 64 ranks fit in one NVLink domain.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Keep DeepSeek kernel selection aligned with the measured baseline.
        # "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }


def deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdpv1_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× GB300, MXFP8, Megatron-FSDP v1."""
    cfg = deepseek_v3_pretrain_config()
    _apply_deepseek_v3_64gpu_gb300_fp8mx_perf_overrides(cfg)
    cfg.ddp.megatron_fsdp_version = 1
    return cfg


def deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdpv2_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 64× GB300, MXFP8, Megatron-FSDP v2.

    Deliberately built on the established 64-GPU FSDP recipe -- selective recompute, activation
    offloading and ZeRO-3 -- rather than the ported 256-GPU perf tuning that the v1 variant uses.
    That configuration is the memory-conservative one, and its ZeRO-3 dense sharding is what
    MFSDP v2 requires anyway.
    """
    cfg = deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config()

    # cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.recompute_modules = []
    cfg.model.offload_modules = []

    # cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    # cfg.ddp.expert_data_parallel_sharding_strategy = None
    # cfg.model.recompute_modules = ["layernorm", "mla_up_proj", "moe_act"]
    # cfg.model.fine_grained_activation_offloading = True
    # cfg.model.fine_grained_offloading_max_inflight_offloads = 1
    # cfg.model.offload_modules = ["core_attn", "attn_proj"]

    # cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_impl = "full_iteration"
    # cfg.model.cuda_graph_warmup_steps = 10

    cfg.model.fine_grained_activation_offloading = False
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1


    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
    cfg.ddp.megatron_fsdp_version = 2
    return cfg


def deepseek_v3_pretrain_4gpu_gb300_fp8mx_fsdpv2_config() -> ConfigContainer:
    """DeepSeek V3 Megatron-FSDP v2 debugging proxy: 8 decoder layers on 4 GB300 GPUs.

    Inherits the 64-GPU fsdpv2 recipe so the v2 code path, MXFP8 settings and FSDP options stay
    identical; only depth and the parallel layout shrink. Three leading dense layers are kept,
    the remaining five are MoE, and the MTP layer is retained.

    This is for reproducing setup-time and single-step failures quickly, not for performance or
    convergence. Per-rank memory is *higher* than the 64-GPU recipe, not lower: expert
    parallelism drops 64 -> 4 (16x fewer shards) while MoE layers drop 58 -> 5 (11.6x fewer), so
    each rank holds 64 experts per layer instead of 4. Lower num_moe_experts if it does not fit.
    """
    cfg = deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdpv2_config()

    cfg.model.num_layers = 4
    # Retain DeepSeek V3's three leading dense layers; shorten its MoE pattern. The list length
    # must equal num_layers.
    cfg.model.moe_layer_freq = [0] * 2 + [1] * (cfg.model.num_layers - 2)
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    # Clear the inherited layout; embedding, decoder and MTP are colocated on the single stage.
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model)

    # Expert parallelism cannot exceed the world size, so EP spans all 4 ranks.
    cfg.model.expert_model_parallel_size = 4

    # Tell HybridEP the EP group is 4 ranks wide, not the inherited 64. The whole job fits in one
    # NVLink domain, so NVLINK_DOMAIN_SIZE stays at the hardware value. Mutating the inherited
    # dict is safe: each recipe call builds a fresh env_vars from COMMON_PERF_ENV_VARS.
    cfg.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = 4

    # DP is 4 at TP=PP=CP=1. GBS 16 over DP 4 at mbs 2 keeps the two gradient-accumulation steps
    # the 64-GPU recipe runs (256 / (64 * 2)).
    cfg.train.micro_batch_size = 2
    cfg.train.global_batch_size = 16

    # _enable_ncclep(cfg)

    return cfg


def deepseek_v3_pretrain_128gpu_gb300_fp8mx_hsdpv2_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 128× GB300, MXFP8, Megatron Hybrid FSDP (HSDP) v2.

    The 64-GPU fsdpv2 recipe scaled to 128 GPUs. Everything except the settings below is
    inherited unchanged, so the two stay comparable as the v2 configuration evolves.
    """
    cfg = deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdpv2_config()

    # Two optimizer instances give an outer data-parallel axis, which is what makes this HSDP
    # rather than plain FSDP. "optim" shards optimizer state across that axis; MFSDP only
    # permits a non-"no_shard" outer strategy when instances > 1, so the pair moves together.
    # expert_outer_dp_sharding_strategy stays "no_shard", matching the 256-GPU HSDP recipe.
    cfg.ddp.num_distributed_optimizer_instances = 2
    cfg.ddp.outer_dp_sharding_strategy = "optim"

    # Doubling GPUs doubles the data-parallel size, so double the global batch to hold
    # micro_batch_size at 2 and keep the same 2 gradient-accumulation steps per iteration:
    # 512 / (DP 128 * mbs 2) == 256 / (DP 64 * mbs 2).
    cfg.train.global_batch_size = 512

    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_hsdp_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8, Megatron Hybrid FSDP (HSDP)."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.fp8_output_proj = True
    _apply_deepseek_v3_64gpu_gb300_fsdp_configs(cfg)

    cfg.ddp.data_parallel_sharding_strategy = "optim_grads"

    cfg.model.gradient_accumulation_fusion = False

    cfg.model.expert_model_parallel_size = 64
    cfg.train.micro_batch_size = 2
    cfg.train.global_batch_size = 1024

    cfg.ddp.outer_dp_sharding_strategy = "optim"
    cfg.ddp.expert_outer_dp_sharding_strategy = "no_shard"
    cfg.ddp.num_distributed_optimizer_instances = 4

    cfg.optimizer.lr = 3e-7
    cfg.optimizer.min_lr = 1e-7

    cfg.model.fp8_param_gather = True
    cfg.model.fp8_param = True
    cfg.model.moe_router_dtype = "bf16"
    cfg.model.average_in_collective = cfg.ddp.average_in_collective = True

    # Full-iteration CUDA graph with dropless MoE padding + paged stashing.
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.overlap_dispatch_backward_with_experts_wgrad = False
    cfg.ddp.megatron_fsdp_cuda_graph_mode = True
    cfg.ddp.fsdp_all_gather_in_start_param_sync = False

    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 5
    # cfg.model.moe_paged_stash_buffer_size_factor_cuda = 2.0
    # cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.2
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1

    # Offload the attention activations rather than recomputing them. recompute_granularity and
    # recompute_modules are set explicitly rather than left out, because deepseek/common.py sets
    # "selective" recompute and omitting them would silently re-enable it; offload_modules is
    # likewise stated here rather than inherited.
    #
    # fine_grained_activation_offloading is a different mechanism from cpu_offloading, which
    # matters here: moe_paged_stash rejects cpu_offloading outright, and separately rejects
    # offload_modules containing expert_fc1/moe_act/fused_group_mlp. core_attn avoids both.
    # The offloading buys the headroom that moe_expert_rank_capacity_factor=5 above needs.
    cfg.model.recompute_granularity = None
    cfg.model.recompute_modules = None
    cfg.model.offload_modules = None
    #cfg.model.offload_modules = ["core_attn"]
    cfg.model.fine_grained_activation_offloading = True
    # cfg.model.cpu_offloading_num_layers = 95
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.fused_residual_rmsnorm = True
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    # These three are owned by comm_overlap, not model/ddp: _apply_cfgs() copies the
    # comm_overlap values onto the model and DDP configs after the recipe runs, so setting
    # them anywhere else is silently overwritten by the defaults at comm_overlap.py:438.
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.align_param_gather = True

    # CuTeDSL fused grouped MLP (moe_a2a_overlap disabled).
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32
    # The fused grouped MLP (ScaledSwiGLU) does not support moe_act recomputation; keep the other
    # selective-recompute modules inherited from the FSDP base.
    cfg.model.recompute_modules = ["layernorm", "mla_up_proj"]

    cfg.model.mla_down_proj_fusion = True

    cfg.mixed_precision.fp8_dot_product_attention = False

    cfg.model.moe_router_force_load_balancing = True
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NVTE_CUDNN_MXFP8_NORM_OUTPUT_IN_INPUT_DTYPE": 0,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8, large-scale proxy (PP=4, VP=4, EP=64)."""
    cfg = deepseek_v3_pretrain_256gpu_gb300_fp8mx_config()
    cfg.train.global_batch_size = 256
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 64
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
        # NCCL EP dispatcher mode and one GPU per rank (inherited from the 256-GPU MXFP8 recipe).
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        # Transformer Engine overlap settings for this model.
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def deepseek_v3_pretrain_256gpu_gb300_fp8mx_partial_cg_dev_config() -> ConfigContainer:
    """DeepSeek V3 pretrain: 256× GB300, MXFP8 and scoped CUDA graphs."""
    cfg = deepseek_v3_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    _benchmark_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 2
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.kv_channels = 128
    cfg.model.make_vocab_size_divisible_by = 1280
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.moe_router_fusion = True
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.cuda_graph_warmup_steps = 1
    cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
    cfg.ddp.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.ddp.check_for_nan_in_grad = False

    cfg.train.micro_batch_size = 1
    cfg.train.global_batch_size = 8192
    cfg.train.exit_duration_in_mins = 220
    cfg.train.manual_gc_interval = 10

    cfg.rng.te_rng_tracker = True
    set_deepseek_v3_pipeline_model_parallel_layout(cfg.model, "Et*4|(tttt|)*14tmL")
    # _enable_deepseek_precision_aware_optimizer(cfg)
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
    }
    return cfg
