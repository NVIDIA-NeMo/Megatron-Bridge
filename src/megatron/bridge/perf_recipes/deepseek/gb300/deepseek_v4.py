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
"""GB300 performance recipes for DeepSeek V4."""

import torch

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    _dsv4_hybrid_csa_compress_ratios,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.models.deepseek.deepseek_v4_data import FixedTHDMockDatasetConfig
from megatron.bridge.models.deepseek.deepseek_v4_hybrid_provider import DeepSeekV4HybridModelProvider
from megatron.bridge.models.hybrid.hybrid_provider import transformer_engine_hybrid_stack_spec
from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.perf_recipes.deepseek.gb200.deepseek_v4 import (
    _build_deepseek_v4_flash_gb200_fp8mx,
)
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config,
)
from megatron.bridge.recipes.utils.optimizer_utils import distributed_muon_with_cosine_annealing
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import bf16_with_mxfp8_mixed
from megatron.bridge.utils.cuda_graph import clear_cuda_graph_modules


# 61 logical blocks: one attention layer (H: HCA, C: CSA) followed by one MoE layer (E).
_DSV4_PRO_MAIN_PATTERN = "HEHECE" + "HECE" * 29
_DSV4_PRO_MTP_PATTERN = "WE"


def deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Flash pretrain: 128× GB300, MXFP8."""
    cfg = _build_deepseek_v4_flash_gb200_fp8mx()

    cfg.model.expert_model_parallel_size = 32
    cfg.train.micro_batch_size = 2

    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 32,
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


def deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Pro pretrain: 256× GB300, MXFP8, 61 blocks + MTP, 65K sequence.

    Follows the MCore reference run's model, CP16/EP64/PP4/VPP4 layout, GBS256/MBS1,
    layer-wise Muon and MoK settings. Forced load balancing and fixed synthetic THD packs
    make it a performance benchmark, not a convergence recipe. No Hugging Face download
    is needed.

    Runs eagerly on MCore main. The reference run also uses attention CUDA graphs and
    overlaps the parameter gather; both need MCore #7337, so step time is not comparable
    with it until this recipe adopts them.
    """
    cfg = _pretrain_common()
    cfg.model = DeepSeekV4HybridModelProvider(
        num_layers=len(_DSV4_PRO_MAIN_PATTERN),
        hidden_size=7168,
        num_attention_heads=128,
        hybrid_layer_pattern=_DSV4_PRO_MAIN_PATTERN,
        mtp_num_layers=1,
        mtp_hybrid_override_pattern=_DSV4_PRO_MTP_PATTERN,
        vocab_size=129280,
    )
    # Match MCore's argument processing for initialization and hash-layer counting.
    cfg.model.is_hybrid_model = True
    cfg.model.hybrid_stack_spec = transformer_engine_hybrid_stack_spec
    cfg.model.csa_compress_ratios = _dsv4_hybrid_csa_compress_ratios(
        _DSV4_PRO_MAIN_PATTERN, _DSV4_PRO_MTP_PATTERN, cfg.model.mtp_num_layers
    )
    cfg.model.attention_cp_layout = "contiguous"
    cfg.model.linear_cp_layout = "contiguous"
    cfg.model.cuda_graph_thd_max_packed_sequences = 8
    cfg.model.enable_mhc_connections = True
    cfg.model.mhc_num_residual_streams = 4
    cfg.model.output_projection_groups = 16
    cfg.model.output_projection_lora_rank = 1024
    cfg.model.moe_num_hash_layers = 3
    cfg.model.hash_moe_vocab_size = cfg.model.vocab_size

    cfg.model.experimental_attention_variant = "dsv4_hybrid"
    cfg.model.multi_latent_attention = True
    cfg.model.q_lora_rank = 1536
    cfg.model.v_head_dim = 512
    cfg.model.qk_pos_emb_head_dim = 64
    cfg.model.qk_head_dim = cfg.model.kv_lora_rank = 448
    cfg.model.qk_layernorm = True
    cfg.model.layernorm_epsilon = 1e-6
    cfg.model.position_embedding_type = "rope"
    cfg.model.rope_type = "yarn"
    cfg.model.rotary_base = 10000
    cfg.model.csa_compress_rotary_base = 40000
    cfg.model.rotary_scaling_factor = 4.0
    cfg.model.original_max_position_embeddings = 65536
    cfg.model.mscale = cfg.model.mscale_all_dim = 1.0
    cfg.model.csa_window_size = 128
    cfg.model.dsa_indexer_n_heads = 64
    cfg.model.dsa_indexer_head_dim = 128
    cfg.model.dsa_indexer_topk = 1024
    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.attention_backend = None  # Resolve to MCore's auto backend.
    cfg.model.transformer_impl = "transformer_engine"
    cfg.model.activation_func = torch.nn.functional.silu
    cfg.model.gated_linear_unit = True
    cfg.model.activation_func_clamp_value = 10.0
    cfg.model.bias_activation_fusion = True
    cfg.model.gradient_accumulation_fusion = True
    cfg.model.num_moe_experts = 384
    cfg.model.ffn_hidden_size = 19072
    cfg.model.moe_ffn_hidden_size = 3072
    cfg.model.moe_shared_expert_intermediate_size = 3072
    cfg.model.moe_router_topk = 6
    cfg.model.moe_router_load_balancing_type = "seq_aux_loss"
    cfg.model.moe_aux_loss_coeff = 1e-4
    cfg.model.moe_router_score_function = "sqrtsoftplus"
    cfg.model.moe_router_topk_scaling_factor = 2.5
    cfg.model.moe_router_enable_expert_bias = True
    cfg.model.moe_router_bias_update_rate = 1e-3
    cfg.model.moe_router_dtype = "fp32"
    cfg.model.moe_router_fusion = True
    cfg.model.moe_permute_fusion = True
    cfg.model.moe_grouped_gemm = True
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.model.mtp_loss_scaling_factor = 0.1
    cfg.model.use_fused_mhc = True
    cfg.model.mhc_sinkhorn_iterations = 20
    cfg.model.seq_length = 65536
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.pipeline_dtype = torch.bfloat16
    cfg.model.context_parallel_size = 16
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.calculate_per_token_loss = True
    cfg.model.sequence_packing_scheduler = "dp_balanced"
    cfg.model.max_seqlen_per_dp_cp_rank = 4096
    cfg.model.pad_packed_seq_alignment = "max"
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_hybridep_routing_map_mode = "bool"
    cfg.model.moe_megakernel_backend = "mok"
    cfg.model.moe_megakernel_backend_config = {
        "fwd_num_comm_sms": 28,
        "bwd_num_comm_sms": 28,
        "minibatch_size": 4096,
        "macrobatch_size": 32768,
        "schedule_capacity_multiplier": 0.0625,
        "all_gather_top_experts_chunk_bytes": 2048,
    }
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model, logical_layers_per_stage=[3] + [4] * 14 + [2])
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj", "mhc"]
    cfg.model.mhc_recompute_layer_num = 2
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = None
    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_warmup_steps = 0
    cfg.model.use_transformer_engine_op_fuser = False

    cfg.mixed_precision = bf16_with_mxfp8_mixed()
    _benchmark_common(cfg, cross_entropy_impl="native")
    clear_cuda_graph_modules(cfg.model)
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.mixed_precision.grad_reduce_in_fp32 = True
    cfg.optimizer, cfg.scheduler = distributed_muon_with_cosine_annealing(
        muon_use_nesterov=False,
        muon_num_ns_steps=10,
        lr_warmup_iters=0,
        lr_decay_iters=50,
        max_lr=2e-4,
        min_lr=2e-5,
        weight_decay=0.1,
        clip_grad=1.0,
    )
    # Use MCore's layerwise Muon path with the logged NS coefficients.
    cfg.optimizer.optimizer = "muon"
    cfg.optimizer.use_distributed_optimizer = False
    cfg.optimizer.use_layer_wise_distributed_optimizer = True
    cfg.optimizer.use_layer_wise_param_layout = False
    cfg.optimizer.muon_coefficient_type = "deepseekv4"
    cfg.optimizer.muon_split_qkv = True
    cfg.optimizer.adam_beta1 = 0.9
    cfg.optimizer.adam_beta2 = 0.999
    cfg.optimizer.adam_eps = 1e-8
    cfg.optimizer.use_precision_aware_optimizer = False
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.float32
    cfg.optimizer.exp_avg_sq_dtype = torch.float32
    cfg.scheduler.lr_warmup_init = 2e-5
    cfg.scheduler.start_weight_decay = cfg.scheduler.end_weight_decay = 0.1
    # On MCore main, mHC reads its projection weight without waiting for an overlapped
    # parameter gather, so a rank could compute with the previous step's weight. Gather
    # synchronously until MCore #7337 adds that wait.
    cfg.ddp.overlap_param_gather = cfg.optimizer.overlap_param_gather = False
    cfg.optimizer.overlap_param_gather_with_optimizer_step = False
    cfg.ddp.align_param_gather = False
    cfg.ddp.grad_reduce_in_fp32 = True
    cfg.ddp.average_in_collective = False
    cfg.ddp.bucket_size = 64000000
    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=False,
        overlap_grad_reduce=True,
        overlap_param_gather=False,
        overlap_param_gather_with_optimizer_step=False,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
        align_param_gather=False,
        bucket_size=64000000,
        overlap_moe_expert_parallel_comm=False,
        delay_wgrad_compute=False,
    )
    cfg.train.manual_gc_interval = 10
    cfg.validation.eval_iters = 32
    cfg.validation.eval_interval = 200
    cfg.rng.seed = 1234
    cfg.dist.enable_megatron_core_experimental = True
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.vocab_size = cfg.model.vocab_size
    cfg.tokenizer.make_vocab_size_divisible_by = 128
    cfg.tokenizer.tensor_model_parallel_size = 1
    cfg.tokenizer.rank = 0
    cfg.dataset = FixedTHDMockDatasetConfig(seq_length=65536, vocab_size=129280, seed=1234, metadata_capacity=8)
    # The NCCL, allocator, dispatch/combine and NVTE kernel settings are the ones the reference
    # launch scripts export. Those scripts leave the shared settings, the connection count, the
    # NVLink-domain names and the LayerNorm SM margins unset; they carry this repository's
    # DeepSeek defaults.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_IN_FLIGHT_S2G_DISPATCH_API": 8,
        "NUM_OF_STAGES_DISPATCH_API": 10,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FUSED_ATTN": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
        # Not from the reference scripts. PyTorch 26.11 makes nccl2 the default "nccl" backend,
        # and that snapshot deadlocks MoK's async barrier (pytorch#202783) and fails the LayerWise
        # parameter gather when a bucket's first two ranks own nothing (pytorch#200585). Select
        # the legacy backend until the image carries both upstream fixes.
        "TORCH_DIST_USE_NCCL2": 0,
    }
    return cfg


def deepseek_v4_pro_pretrain_64gpu_gb300_fp8mx_proxy_config() -> ConfigContainer:
    """DeepSeek V4 Pro proxy: the first 16 blocks + MTP on 64× GB300, PP1.

    Keeps the full recipe's widths, experts, 65K sequence, CP16/EP64 and batch sizes on a
    quarter of the GPUs. It does not exercise the full recipe's PP/VPP schedule.
    """
    cfg = deepseek_v4_pro_pretrain_256gpu_gb300_fp8mx_config()
    proxy_pattern = _DSV4_PRO_MAIN_PATTERN[:32]
    cfg.model.num_layers = len(proxy_pattern)
    cfg.model.hybrid_layer_pattern = proxy_pattern
    cfg.model.csa_compress_ratios = _dsv4_hybrid_csa_compress_ratios(
        proxy_pattern, _DSV4_PRO_MTP_PATTERN, cfg.model.mtp_num_layers
    )
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)
    cfg.comm_overlap.overlap_p2p_comm = False

    # Keep the proxy launch environment explicit instead of inheriting it from the full recipe.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 64,
        "NUM_OF_IN_FLIGHT_S2G_DISPATCH_API": 8,
        "NUM_OF_STAGES_DISPATCH_API": 10,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CPU_OFFLOAD_V1": 1,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FUSED_ATTN": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 1,
        # Legacy NCCL backend; see the full recipe.
        "TORCH_DIST_USE_NCCL2": 0,
    }
    return cfg


def _deepseek_v4_pro_pretrain_256gpu_gb300_common() -> ConfigContainer:
    """Build the full Pro benchmark with the original MXFP8 execution settings."""
    cfg = deepseek_v4_pro_pretrain_32gpu_gb300_fp8mx_config()

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 4096
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False
    # Preserve the measured logical-block split, reserving room for MTP/loss.
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model, logical_layers_per_stage=[4] * 15 + [1])
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["mla_up_proj", "mhc"]

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.model.moe_flex_dispatcher_num_sms = 32
    cfg.model.moe_hybridep_num_sms = None

    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 3
    cfg.model.use_te_rng_tracker = True
    cfg.rng.te_rng_tracker = True

    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 0.0

    cfg.model.moe_router_force_load_balancing = True
    cfg.model.dsa_kernel_backend = "cudnn"
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.cross_entropy_loss_fusion = True
    cfg.model.cross_entropy_fusion_impl = "native"
    cfg.model.moe_mlp_glu_interleave_size = 32

    cfg.model.quant_recipe = None
    cfg.mixed_precision.fp8_param_gather = True
    cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False

    cfg.model.dsa_indexer_loss_coeff = 0.01
    cfg.model.dsa_indexer_use_sparse_loss = True

    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.dist.enable_megatron_core_experimental = True
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.fine_grained_activation_offloading = True
    cfg.model.offload_modules = ["core_attn", "attn_proj"]
    cfg.model.fine_grained_offloading_max_inflight_offloads = 2
    cfg.comm_overlap.overlap_grad_reduce = True
    return cfg


def deepseek_v4_pro_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """DeepSeek V4 Pro pretrain: 256× GB300, BF16, eager PP4/VPP4/EP64.

    Preserves the MXFP8 benchmark's full model, batch, forced routing and Adam
    precision: FP32 master weights with BF16 main gradients and Adam moments.
    Uses selective MLA/mHC/MoE-block recompute without offload or CUDA graphs,
    whose combination is unsupported by the native HybridModel runtime.
    This synthetic-data performance benchmark is not a convergence recipe.
    """
    cfg = _deepseek_v4_pro_pretrain_256gpu_gb300_common()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.quant_recipe = None
    cfg.model.moe_router_padding_for_fp8 = False
    cfg.model.moe_router_padding_for_quantization = False
    # Leave room for initialized optimizer state during steady-state training.
    cfg.model.recompute_modules = ["mla_up_proj", "mhc", "moe"]

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = []
    cfg.model.use_te_rng_tracker = False
    cfg.rng.te_rng_tracker = False
    cfg.model.moe_pad_experts_for_cuda_graph_inference = False
    cfg.model.moe_paged_stash = False
    cfg.model.fine_grained_activation_offloading = False
    cfg.model.offload_modules = []
    cfg.model.fine_grained_offloading_max_inflight_offloads = None
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
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
