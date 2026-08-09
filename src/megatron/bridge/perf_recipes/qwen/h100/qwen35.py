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
"""H100 performance recipes for text-only Qwen3.5 MoE."""

import os

import torch

from megatron.bridge.models.qwen.modeling_qwen35 import qwen35_h100_transformer_block_spec
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen.common import (
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
)
from megatron.bridge.recipes.qwen import qwen35_text_35b_a3b_pretrain_config
from megatron.bridge.utils.cuda_graph import set_cuda_graph_modules


_QWEN35_35B_A3B = "Qwen/Qwen3.5-35B-A3B"
_QWEN35_35B_A3B_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"  # pragma: allowlist secret


def _qwen35_rank_local_kernel_cache_env() -> dict[str, str]:
    """Return portable, isolated compile-cache paths for the current rank."""
    cache_root = os.environ.get("MBRIDGE_KERNEL_CACHE_DIR")
    if cache_root is None:
        cache_base = os.environ.get("NEMO_HOME") or os.environ.get("XDG_CACHE_HOME") or "/tmp"
        cache_root = os.path.join(cache_base, "megatron-bridge", "qwen35-h100")
    rank = os.environ.get("SLURM_PROCID") or os.environ.get("RANK") or "0"
    rank_cache = os.path.join(cache_root, f"rank-{rank}")
    return {
        # HybridEP treats this value as a cache root and appends its own
        # .deepep/hybrid_ep/jit hierarchy.
        "HYBRID_EP_CACHE_DIR": rank_cache,
        "TILELANG_CACHE_DIR": os.path.join(rank_cache, "tilelang"),
        "TORCHINDUCTOR_CACHE_DIR": os.path.join(rank_cache, "torchinductor"),
        "TRITON_CACHE_DIR": os.path.join(rank_cache, "triton"),
        "TORCH_EXTENSIONS_DIR": os.path.join(cache_root, "torch-extensions"),
    }


def qwen35_text_35b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5 text 35B-A3B pretrain: 16× H100, BF16, EP=8."""
    cfg = qwen35_text_35b_a3b_pretrain_config()
    cfg.tokenizer.tokenizer_model = _QWEN35_35B_A3B
    cfg.tokenizer.hf_tokenizer_kwargs = {"revision": _QWEN35_35B_A3B_REVISION}
    cfg.mixed_precision = _perf_precision("bf16")
    # FP16 main parameters avoid the BF16 remainder buffer so the no-recompute
    # path fits on 80 GB H100s; compute, gradients, and moments remain BF16.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_grads_dtype = torch.bfloat16
    cfg.optimizer.main_params_dtype = torch.float16
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16
    cfg.optimizer.store_param_remainders = False

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_layout = None
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.recompute_modules = []

    # Freeze the measured GDN path in the recipe so the model-card command
    # does not depend on hidden command-line overrides.
    cfg.model.gated_delta_rule_backend = "flash_qla"
    cfg.model.gdn_pre_gated_delta_rule_fusion = True

    # Static dispatcher metadata removes the synchronization wall that the
    # scoped graphs targeted. The matched final-stack A/B favored eager.
    cfg.model.cuda_graph_impl = "none"
    set_cuda_graph_modules(cfg.model, [])
    cfg.model.transformer_layer_spec = qwen35_h100_transformer_block_spec

    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_flex_dispatcher_num_sms = 16
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.moe_hybridep_num_sms_preprocessing = 108
    cfg.model.moe_router_force_load_balancing = True
    # Keep auxiliary balancing local to each microbatch. The global variant
    # all-reduces expert counts across DP ranks for every MoE layer and microbatch.
    cfg.model.moe_router_load_balancing_type = "aux_loss"
    cfg.model.moe_shared_expert_overlap = True
    cfg.model.moe_expert_rank_capacity_factor = 1.05
    cfg.model.moe_permute_fusion_into_hybridep = True
    # This bounded mock-pretrain benchmark has the same fixed MBS×sequence
    # shape on every rank. Avoid HybridEP's per-layer max-token all-reduce;
    # the model-specific runtime fails closed outside this contract.
    cfg.model.moe_hybridep_assume_equal_dispatch_inputs = True
    # MCore uses this flag to validate and align static HybridEP buffers. The
    # layer spec above replaces only the SM100-only fused expert implementation
    # with the measured Hopper torch.grouped_mm implementation.
    cfg.model.use_transformer_engine_op_fuser = True

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False
    cfg.comm_overlap.overlap_grad_reduce = False
    cfg.comm_overlap.overlap_param_gather = False
    # A finite first GDN optimizer step is not sufficient validation on the
    # current H100 stack; later steps can exhaust memory or stop progressing.
    # Keep EP overlap off for the conservative public benchmark default.
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False
    cfg.comm_overlap.delay_wgrad_compute = False

    _benchmark_common(cfg)
    # Restore the final-stack HybridEP settings after the shared benchmark
    # defaults normalize legacy dispatcher fields and disable the op fuser.
    cfg.model.moe_hybridep_num_sms = None
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        **_qwen35_rank_local_kernel_cache_env(),
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 64,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 64,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg
