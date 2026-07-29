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
"""GB200 performance recipes for NemotronH and Nemotron 3."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.nemotronh.common import (
    _TE_QUANT_CFG_PATH,
    ConfigContainer,
    _apply_nemotron_3_super_perf_defaults,
    _apply_nemotron_3_ultra_fsdp_hsdp,
    _apply_nemotron_3_ultra_perf_defaults,
    _benchmark_common,
    _nemotron_3_super_nvfp4_precision,
    _perf_precision,
    load_quantization_recipe,
    nemotron_3_nano_pretrain_config,
    nemotron_3_super_pretrain_config,
    nemotron_3_ultra_pretrain_config,
    nemotronh_56b_pretrain_config,
)


def nemotronh_56b_pretrain_64gpu_gb200_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× GB200, FP8 current-scaling."""
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
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB200, BF16."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False

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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB200, MXFP8."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB200, NVFP4."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _nemotron_3_super_nvfp4_precision()

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_router_padding_for_quantization = True
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
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def _nemotron_3_ultra_gb200_fp8mx_config(
    *,
    num_gpus: int,
    expert_model_parallel_size: int,
    global_batch_size: int,
    hybrid_ep_ranks_per_nvlink_domain: int,
) -> ConfigContainer:
    """Shared builder for Nemotron 3 Ultra GB200 MXFP8 Megatron-FSDP perf recipes.

    Same Megatron-FSDP (HSDP) layout as the GB300 recipe, but with TP2 + sequence
    parallelism instead of TP1. GB200 carries less HBM per GPU, and TP alone would not
    help much: it shards the internal GEMM dimensions but leaves the residual stream and
    the norm inputs/outputs replicated. Sequence parallelism is what shards those, so the
    two are enabled together.
    """
    cfg = nemotron_3_ultra_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    _apply_nemotron_3_ultra_perf_defaults(cfg)

    # Apply HSDP / FSDP dtype overrides last so they win over the generic defaults.
    _apply_nemotron_3_ultra_fsdp_hsdp(cfg, num_gpus=num_gpus)

    # Parallelism
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.seq_length = 8192

    # Only tensors larger than 500MB are offloaded, which
    # approximates offloading the moe_act input for seq 8192 / MBS 1.
    cfg.model.min_offloaded_tensor_size = 500_000_000

    # MXFP8 requires router padding for quantization.
    cfg.model.moe_router_padding_for_quantization = True

    # GPU-count specific overrides of the canonical (256-GPU / EP64) defaults.
    cfg.model.expert_model_parallel_size = expert_model_parallel_size
    cfg.train.global_batch_size = global_batch_size

    # Fine-grained activation offloading. Requires NVTE_CPU_OFFLOAD_V1=1 in the
    # launch environment (set in this recipe's env_vars).
    # NOTE: also requires setting the min_offloaded_tensor_size to avoid CPU OOM issues
    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["fused_group_mlp"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 1

    # Selective recompute of the MoE activation
    # recomputes the activation output of the MoE expert MLP, while FC1 output (activation input) is saved and offloaded to cpu
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["moe_act"]

    # full iteration cuda graph
    cfg.model.cuda_graph_impl = "full_iteration"  # "transformer_engine"
    cfg.model.cuda_graph_scope = []  # ["attn", "mamba", "moe_router", "moe_preprocess"]
    cfg.ddp.megatron_fsdp_cuda_graph_mode = True
    cfg.ddp.fsdp_all_gather_in_start_param_sync = False
    cfg.model.moe_expert_rank_capacity_factor = 1.2
    # cfg.model.moe_paged_stash = True
    # cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    # cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2

    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        # graph_capture_record_stream_reuse is required here, not an
        # optimization: fine_grained_activation_offloading calls record_stream()
        # on every D2H staging tensor, and cudaEventQuery is illegal during graph
        # capture, so without this the allocator cannot recycle any of them until
        # capture ends. Offloading then frees nothing on the capture step and
        # reserved memory overshoots the eager peak (OOM at 256 GPUs / EP64). The
        # option uses the captured DAG topology instead of events, which makes
        # the one-time capture slower. TORCH_NCCL_AVOID_RECORD_STREAMS below
        # covers only the NCCL buffers, not the offload path.
        "PYTORCH_CUDA_ALLOC_CONF": ("expandable_segments:True,graph_capture_record_stream_reuse:True"),
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # HybridEP topology for the target system. The per-domain rank count has to track
        # EP: HybridEP splits the all-to-all into an intra-domain and an inter-domain leg
        # using this value, so leaving it at the 256-GPU 64 on a smaller job would
        # describe a domain larger than the job itself.
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": hybrid_ep_ranks_per_nvlink_domain,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Required by fine_grained_activation_offloading (TE >= 2.10.0) to avoid
        # offloading weights;
        "NVTE_CPU_OFFLOAD_V1": 1,
        # Enable TE's CuteDSL fused grouped MLP kernel (sm100+). Required by the
        # op fuser + fused weighted squared-ReLU with moe_act activation recompute
        # (ScaledSReLU(activation_recompute_in_mlp=True) only runs on this path).
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
    }
    return cfg


def nemotron_3_ultra_pretrain_256gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Ultra (550B-A55B LatentMoE) pretrain: 256× GB200, MXFP8, Megatron-FSDP (HSDP).

    TP2 + SP / PP1 / CP1 / EP64 / ETP1, GBS 256 / MBS 1, seq 8192, BF16 + MXFP8 mixed
    precision, HybridEP flex dispatcher, CuteDSL fused grouped MLP, selective recompute +
    fine-grained activation offload of the expert MLP, MTP=2.

    This is the GB300 recipe's layout with TP raised from 1 to 2 and sequence parallelism
    turned on, to fit GB200's smaller HBM. Megatron-FSDP shards params, grads and
    optimizer state within each 64-GPU NVLink domain and replicates optimizer-sharded
    across the four domains.
    """
    return _nemotron_3_ultra_gb200_fp8mx_config(
        num_gpus=256,
        expert_model_parallel_size=64,
        global_batch_size=256,
        hybrid_ep_ranks_per_nvlink_domain=64,
    )


def nemotron_3_ultra_pretrain_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Ultra pipeclean: 8× GB200, MXFP8, Megatron-FSDP.

    Shrunk-grid counterpart of the 256-GPU recipe, for validating that the full feature
    stack launches and trains before committing a full-scale job. EP drops 64 -> 4 so the
    expert grid (``ETP1 x EP4`` = 4) divides an 8-GPU world, and GBS drops 256 -> 8. With
    TP2 the batch splits over DP = 8 / 2 = 4, so GBS 8 / MBS 1 runs 2 microbatches.

    Below one NVLink domain ``_apply_nemotron_3_ultra_fsdp_hsdp`` resolves
    ``num_distributed_optimizer_instances`` to 1, which turns HSDP off and correspondingly
    sets ``outer_dp_sharding_strategy`` to ``no_shard``.

    This recipe only rescales parallelism: the model is still full-size Nemotron 3 Ultra,
    which does not fit on 8 GPUs. Pair it with model-shrinking overrides (see
    ``perf_bash_scripts/nt3_ultra_gb200/toy_run_perf_test_nemotron_3_ultra_gb200_fp8mx.sh``).
    """
    return _nemotron_3_ultra_gb200_fp8mx_config(
        num_gpus=8,
        expert_model_parallel_size=4,
        global_batch_size=8,
        hybrid_ep_ranks_per_nvlink_domain=8,
    )


def nemotron_3_nano_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB200, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

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
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB200, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

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
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB200, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

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
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg
