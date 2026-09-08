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
"""B200 performance recipes for DeepSeek V4."""

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.perf_recipes.deepseek.b300.deepseek_v4 import (
    deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config,
)
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def _apply_deepseek_v4_b200_overrides(cfg: ConfigContainer) -> None:
    """Apply the lower-memory B200 topology and communication settings."""
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.train.micro_batch_size = 1
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)
    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
    cfg.model.moe_pad_experts_for_cuda_graph_inference = False
    cfg.model.moe_paged_stash = False

    recompute_modules = list(cfg.model.recompute_modules or [])
    if "mlp" not in recompute_modules:
        recompute_modules.append("mlp")
    cfg.model.recompute_modules = recompute_modules

    cfg.comm_overlap.overlap_grad_reduce = True
    # Hash-routed MoE layers do not currently support EP A2A overlap.
    cfg.comm_overlap.delay_wgrad_compute = False
    cfg.dist.distributed_timeout_minutes = 30
    cfg.comm_overlap.overlap_moe_expert_parallel_comm = False


def deepseek_v4_flash_pretrain_128gpu_b200_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Flash pretrain: 128× B200, MXFP8."""
    cfg = deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config()
    _apply_deepseek_v4_b200_overrides(cfg)
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
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
    }
    return cfg
