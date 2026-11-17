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

"""gpt_oss benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config, gpt_oss_120b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import (
    _benchmark_common,
    _enable_ncclep,
    _perf_precision,
)
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS


BENCHMARK_ENVIRONMENTS = {
    "gpt_oss_120b_pretrain_64gpu_b200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "gpt_oss_20b_pretrain_64gpu_b300_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    },
    "gpt_oss_20b_pretrain_64gpu_b300_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "gpt_oss_120b_pretrain_64gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 0,
        "NCCL_NVLS_ENABLE": 0,
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    },
    "gpt_oss_20b_pretrain_512gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    },
    "gpt_oss_20b_pretrain_72gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 4,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "gpt_oss_20b_pretrain_8gpu_gb200_fp8mx": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 1,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
    "gpt_oss_20b_pretrain_8gpu_gb200_nvfp4": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 1,
        "NCCL_CTA_POLICY": 1,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 2,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 72,
        "USE_MNNVL": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_USE_FAST_MATH": 1,
    },
}


def _apply_gpt_oss_120b_full_iter_fp8mx_configs(cfg: ConfigContainer) -> None:
    """Apply legacy GPT-OSS 120B FP8-MX full-iteration CUDA graph settings."""
    cfg.model.cuda_graph_impl = "full_iteration"
    cfg.model.cuda_graph_scope = []
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.model.fp8_output_proj = True
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_expert_rank_capacity_factor = 1.5
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 32
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_mlp_glu_interleave_size = 32
    cfg.model.moe_pad_experts_for_cuda_graph_inference = True
    cfg.model.moe_paged_stash = True
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.2
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.use_te_rng_tracker = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.offload_modules = []
    cfg.mixed_precision.fp8_dot_product_attention = True
    cfg.rng.te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.delay_wgrad_compute = True
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = True


def _gpt_oss_20b_fp8mx_precision():
    """Return legacy GPT-OSS 20B MXFP8 perf precision settings."""
    precision_config = _perf_precision("fp8_mx")
    precision_config.fp4_param = False
    precision_config.fp4_param_gather = False
    precision_config.fp8_param = False
    precision_config.fp8_param_gather = False
    precision_config.reuse_grad_buf_for_mxfp8_param_ag = False
    precision_config.first_last_layers_bf16 = False
    precision_config.num_layers_at_start_in_bf16 = 0
    return precision_config


def _gpt_oss_20b_nvfp4_precision():
    """Return legacy GPT-OSS 20B NVFP4 perf precision settings."""
    precision_config = _perf_precision("nvfp4")
    precision_config.fp4_param = False
    precision_config.fp4_param_gather = False
    precision_config.fp8_param = False
    precision_config.fp8_param_gather = False
    precision_config.reuse_grad_buf_for_mxfp8_param_ag = False
    precision_config.first_last_layers_bf16 = True
    precision_config.num_layers_at_start_in_bf16 = 0
    precision_config.num_layers_at_end_in_bf16 = 4
    return precision_config


def _apply_gpt_oss_20b_common_configs(cfg: ConfigContainer) -> None:
    """Apply legacy GPT-OSS 20B perf defaults."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.apply_rope_fusion = False
    cfg.model.attention_backend = "auto"
    cfg.model.calculate_per_token_loss = False
    cfg.model.cpu_offloading_num_layers = 95
    cfg.model.cuda_graph_warmup_steps = 2
    cfg.model.fused_single_qkv_rope = True
    cfg.model.moe_aux_loss_coeff = 0.0
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_hybridep_num_sms = 128
    cfg.model.moe_permute_fusion = False
    cfg.model.moe_router_force_load_balancing = False
    cfg.model.moe_router_fusion = False
    cfg.model.moe_router_padding_for_quantization = True
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.position_embedding_type = "rope"
    cfg.model.seq_length = 8192
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.use_te_rng_tracker = True
    cfg.model.tp_only_amax_red = True
    cfg.model.vocab_size = 128256
    cfg.train.check_optimizer_step_success = False
    cfg.train.skip_sync_grad_norm_across_mp = False
    cfg.checkpoint.dist_ckpt_strictness = "log_all"
    cfg.checkpoint.fully_parallel_load = True
    cfg.checkpoint.load_optim = False
    cfg.tokenizer.hf_tokenizer_kwargs = {"use_fast": True}
    cfg.tokenizer.vocab_size = 128256
    cfg.optimizer.adam_eps = 1e-05
    cfg.dataset.create_attention_mask = False
    cfg.dataset.defer_npy_index_mmap = True
    cfg.dataset.fast_cache_load = True
    cfg.ddp.bucket_size = 768000000
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.nccl_ub = True
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.overlap_param_gather = False
    cfg.rng.te_rng_tracker = True
    cfg.scheduler.start_weight_decay = 0.1
    cfg.scheduler.end_weight_decay = 0.1
    cfg.scheduler.override_opt_param_scheduler = False


def _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg: ConfigContainer) -> None:
    """Apply GPT-OSS 20B Transformer Engine graph capture defaults."""
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]


def _apply_gpt_oss_20b_local_graph_configs(cfg: ConfigContainer) -> None:
    """Apply GPT-OSS 20B local full-iteration graph capture defaults."""
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_modules = "full_iteration"
    cfg.model.cuda_graph_scope = None
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32


def _apply_gpt_oss_120b_gb300_fp8mx(cfg: ConfigContainer) -> ConfigContainer:
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.moe_router_fusion = True
    cfg.model.moe_router_force_load_balancing = True
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1280
    cfg.train.micro_batch_size = 4
    cfg.model.recompute_modules = []
    _benchmark_common(cfg)
    _apply_gpt_oss_120b_full_iter_fp8mx_configs(cfg)
    return cfg


LIBRARY_RECIPES = {
    "gpt_oss_120b_pretrain_64gpu_b200_bf16_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_b200_fp8mx_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_b300_bf16_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_b300_fp8mx_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_20b_pretrain_64gpu_b300_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_64gpu_b300_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_b300_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_b300_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_gb200_bf16_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_72gpu_gb200_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_gb200_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_gb300_bf16_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_72gpu_gb300_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_gb300_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_h100_bf16_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_h100_fp8cs_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_vrnvl72_bf16_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_120b_pretrain_64gpu_vrnvl72_fp8mx_config": gpt_oss_120b_pretrain_config,
    "gpt_oss_20b_pretrain_64gpu_vrnvl72_nvfp4_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_vrnvl72_fp8mx_config": gpt_oss_20b_pretrain_config,
    "gpt_oss_20b_pretrain_8gpu_vrnvl72_nvfp4_config": gpt_oss_20b_pretrain_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "gpt_oss_120b_pretrain_64gpu_b200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.recompute_modules = ["layernorm", "moe_act"]
            cfg.model.recompute_granularity = "selective"
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = False
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
            }
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_b200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.recompute_modules = ["layernorm", "moe_act"]
            cfg.model.recompute_granularity = "selective"
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_b300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            _benchmark_common(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NCCL_EP_HT_EM_PULL_PUSH": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NCCL_IGNORE_CPU_AFFINITY": 1,
            }
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_b300_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.moe_token_dispatcher_type = "flex"
            cfg.model.moe_flex_dispatcher_backend = "hybridep"
            cfg.model.moe_hybridep_num_sms = 32
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            _benchmark_common(cfg)
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
                "NCCL_IGNORE_CPU_AFFINITY": 1,
            }
            return cfg
        case "gpt_oss_20b_pretrain_64gpu_b300_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            _apply_gpt_oss_20b_local_graph_configs(cfg)
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_64gpu_b300_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_64gpu_b300_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 64
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_64gpu_b300_nvfp4"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_b300_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            _apply_gpt_oss_20b_local_graph_configs(cfg)
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 0.0004
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 512
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_64gpu_b300_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_b300_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 4
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 64
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_64gpu_b300_nvfp4"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.recompute_modules = ["layernorm", "moe_act"]
            cfg.model.recompute_granularity = "selective"
            cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
            cfg.comm_overlap.delay_wgrad_compute = True
            cfg.comm_overlap.overlap_moe_expert_parallel_comm = True
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _perf_precision("fp8_mx")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.recompute_modules = ["layernorm", "moe_act"]
            cfg.model.recompute_granularity = "selective"
            _benchmark_common(cfg)
            _apply_gpt_oss_120b_full_iter_fp8mx_configs(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.model.recompute_granularity = "selective"
            cfg.model.recompute_modules = []
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.context_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_local_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 7
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.00052
            cfg.optimizer.min_lr = 5.2e-05
            cfg.validation.eval_interval = 192
            cfg.validation.eval_iters = 16
            cfg.scheduler.lr_warmup_iters = 32
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_512gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_72gpu_gb200_fp8mx_config" | "gpt_oss_20b_pretrain_72gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 36
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 5
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 4e-05
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 256
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_72gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 36
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 4e-05
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 64
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_72gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_gb200_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 1
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 16
            cfg.train.micro_batch_size = 2
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 1.2
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.model.cuda_graph_warmup_steps = 5
            cfg.ddp.average_in_collective = True
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 4e-05
            cfg.validation.eval_interval = 768
            cfg.validation.eval_iters = 64
            cfg.scheduler.lr_warmup_iters = 128
            cfg.mixed_precision.fp8_param_gather = True
            cfg.mixed_precision.reuse_grad_buf_for_mxfp8_param_ag = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_8gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 2
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 16
            cfg.train.micro_batch_size = 2
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 5.9999999999999995e-05
            cfg.validation.eval_interval = 768
            cfg.validation.eval_iters = 64
            cfg.scheduler.lr_warmup_iters = 128
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.expert_model_parallel_size = 64
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 4
            cfg.model.recompute_modules = []
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config" | "gpt_oss_120b_pretrain_64gpu_vrnvl72_fp8mx_config":
            _apply_gpt_oss_120b_gb300_fp8mx(cfg)
            _enable_ncclep(cfg)
            cfg.model.moe_use_grouped_tensor = True
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.context_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_local_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 7
            cfg.model.sequence_parallel = True
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.00052
            cfg.optimizer.min_lr = 5.2e-05
            cfg.validation.eval_interval = 192
            cfg.validation.eval_iters = 16
            cfg.scheduler.lr_warmup_iters = 32
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_512gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 36
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 5.9999999999999995e-05
            cfg.validation.eval_interval = 341
            cfg.validation.eval_iters = 29
            cfg.scheduler.lr_warmup_iters = 64
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_72gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_gb300_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 1
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 24
            cfg.train.micro_batch_size = 3
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "local"
            cfg.model.cuda_graph_modules = "full_iteration"
            cfg.model.cuda_graph_scope = None
            cfg.model.use_transformer_engine_op_fuser = True
            cfg.model.moe_expert_rank_capacity_factor = 1.5
            cfg.model.moe_mlp_glu_interleave_size = 32
            cfg.optimizer.lr = 0.0005
            cfg.optimizer.min_lr = 5e-05
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 256
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_8gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 2
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 24
            cfg.train.micro_batch_size = 3
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
            cfg.optimizer.lr = 0.0004
            cfg.optimizer.min_lr = 4e-05
            cfg.validation.eval_interval = 512
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 192
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_8gpu_gb200_nvfp4"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["layernorm", "moe_act"]
            cfg.model.recompute_granularity = "selective"
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_h100_fp8cs_config":
            cfg.mixed_precision = _perf_precision("fp8_cs")
            cfg.model.moe_router_fusion = True
            cfg.model.moe_router_force_load_balancing = True
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 4
            cfg.model.expert_model_parallel_size = 8
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 1280
            cfg.train.micro_batch_size = 1
            cfg.model.recompute_modules = ["layernorm", "moe_act"]
            cfg.model.recompute_granularity = "selective"
            _benchmark_common(cfg)
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "gpt_oss_120b_pretrain_64gpu_vrnvl72_bf16_config":
            apply_overrides(cfg, "gpt_oss_120b_pretrain_64gpu_gb300_bf16_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_120b_pretrain_64gpu_b200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_64gpu_vrnvl72_nvfp4_config":
            cfg.mixed_precision = _gpt_oss_20b_nvfp4_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.expert_model_parallel_size = 4
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 32
            cfg.train.micro_batch_size = 1
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            cfg.model.cuda_graph_warmup_steps = 1
            cfg.optimizer.lr = 0.0006
            cfg.optimizer.min_lr = 0.0006
            cfg.validation.eval_interval = 384
            cfg.validation.eval_iters = 43
            cfg.scheduler.lr_warmup_iters = 64
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_72gpu_gb200_fp8mx"].copy()
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_vrnvl72_fp8mx_config":
            cfg.mixed_precision = _gpt_oss_20b_fp8mx_precision()
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.context_parallel_size = 1
            cfg.model.expert_model_parallel_size = 1
            cfg.model.expert_tensor_parallel_size = 1
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 24
            cfg.train.micro_batch_size = 3
            _benchmark_common(cfg)
            _apply_gpt_oss_20b_common_configs(cfg)
            _apply_gpt_oss_20b_transformer_engine_graph_configs(cfg)
            cfg.model.cuda_graph_warmup_steps = 1
            cfg.optimizer.lr = 0.0003
            cfg.optimizer.min_lr = 3e-05
            cfg.validation.eval_interval = 2000
            cfg.validation.eval_iters = 32
            cfg.scheduler.lr_warmup_iters = 10
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 32,
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 1,
                "NCCL_CTA_POLICY": 1,
                "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 1,
                "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
                "NVLINK_DOMAIN_SIZE": 72,
                "USE_MNNVL": 1,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
            }
            return cfg
        case "gpt_oss_20b_pretrain_8gpu_vrnvl72_nvfp4_config":
            apply_overrides(cfg, "gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config")
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["gpt_oss_20b_pretrain_72gpu_gb200_fp8mx"].copy()
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
