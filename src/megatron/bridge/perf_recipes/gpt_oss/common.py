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
# ruff: noqa: F401
"""Common helpers for gpt_oss performance recipes."""

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.gpt_oss.gpt_oss import gpt_oss_20b_pretrain_config, gpt_oss_120b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


_HYBRID_EP_ENV_NAMES = {
    "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN",
    "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API",
    "NVLINK_DOMAIN_SIZE",
    "USE_MNNVL",
}


def _enable_ncclep(cfg: ConfigContainer, *, mxfp8: bool, moe_a2a_overlap: bool) -> None:
    """Swap the recipe's flex dispatcher for static-shape NCCL EP.

    Only the dispatch stack changes. Parallelism, batch sizes, precision, recompute and the CUDA
    graph mode are left exactly as the parent recipe set them, so each variant stays comparable
    to the HybridEP/DeepEP recipe it derives from.
    """
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "ncclep"
    cfg.model.moe_shared_expert_overlap = False
    cfg.model.high_priority_a2a_comm_stream = True
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_flex_dispatcher_num_sms = None
    cfg.model.moe_ncclep_zero_copy = False

    # Static shapes hand the experts the whole receive buffer, so the grouped GEMM has to consume
    # the ragged per-expert counts on device and never read the slack tail. GPT-OSS's clamped
    # quick-GeLU GLU is covered by the op fuser through TE's ScaledClampedQGeGLU.
    cfg.model.moe_grouped_gemm = True
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = moe_a2a_overlap
    cfg.comm_overlap.delay_wgrad_compute = False

    cfg.model.offload_modules = []
    # The receive buffer is sized at capacity_factor x the ideal token count and every MoE
    # activation saved for backward inherits that padding. These recipes force-balance the router,
    # so a few percent of headroom is enough; PagedStashRunner replays the step dropless and grows
    # the budget if a routing ever exceeds the static budget.
    cfg.model.moe_expert_rank_capacity_factor = 1.05
    # Paged stashing only captures TE's quantized grouped tensors, so it is a no-op outside MXFP8.
    cfg.model.moe_paged_stash = mxfp8
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.0
    cfg.model.moe_paged_stash_buffer_size_factor_cpu = 1.0

    cfg.env_vars = {name: value for name, value in cfg.env_vars.items() if name not in _HYBRID_EP_ENV_NAMES}
    cfg.env_vars["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = 1
    if mxfp8:
        cfg.model.moe_router_padding_for_quantization = True


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
