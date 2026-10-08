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
"""B300 performance recipes for DeepSeek V4."""

from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.perf_recipes.deepseek.gb300.deepseek_v4 import (
    deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config,
)
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def _apply_deepseek_v4_b300_overrides(cfg: ConfigContainer) -> None:
    """Apply a B300 topology scaled down from the larger DeepSeek V3 model."""
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.micro_batch_size = 2
    cfg.model.moe_hybridep_num_sms_preprocessing = 32
    cfg.model.moe_paged_stash_buffer_size_factor_cuda = 1.5
    set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)


def deepseek_v4_flash_pretrain_128gpu_b300_fp8mx_config() -> ConfigContainer:
    """DeepSeek V4 Flash pretrain: 128× B300, MXFP8."""
    cfg = deepseek_v4_flash_pretrain_128gpu_gb300_fp8mx_config()
    _apply_deepseek_v4_b300_overrides(cfg)
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,graph_capture_record_stream_reuse:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN": 8,
        "NUM_OF_TOKENS_PER_CHUNK_COMBINE_API": 128,
        "NVLINK_DOMAIN_SIZE": 8,
        "USE_MNNVL": 0,
        "CUDNNFE_CLUSTER_OVERLAP_MARGIN": 8,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": 0,
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg
