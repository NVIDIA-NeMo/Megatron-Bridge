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
"""VRNVL72 performance recipes for Qwen3 MoE."""

from megatron.bridge.perf_recipes._common import _enable_ncclep
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen.common import ConfigContainer
from megatron.bridge.perf_recipes.qwen.gb300.qwen3_moe import (
    _build_qwen3_30b_a3b_gb300_bf16,
    _build_qwen3_30b_a3b_gb300_fp8mx,
    _build_qwen3_235b_a22b_gb300_bf16,
    _build_qwen3_235b_a22b_gb300_fp8mx,
    _build_qwen3_235b_a22b_gb300_nvfp4,
    qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config,
)


def qwen3_235b_a22b_pretrain_256gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VRNVL72, BF16, NCCL EP (alias of GB300)."""
    cfg = _build_qwen3_235b_a22b_gb300_bf16()
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

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
    }
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VRNVL72, FP8-MX, NCCL EP (alias of GB300)."""
    cfg = _build_qwen3_235b_a22b_gb300_fp8mx()
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
    }
    return cfg


def qwen3_235b_a22b_pretrain_256gpu_vrnvl72_nvfp4_config() -> ConfigContainer:
    """Qwen3 235B A22B pretrain: 256× VRNVL72, NVFP4, NCCL EP, full-iteration CG and A2A overlap."""
    cfg = _build_qwen3_235b_a22b_gb300_nvfp4()
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
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_vrnvl72_bf16_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VRNVL72, BF16, NCCL EP (alias of GB300)."""
    cfg = _build_qwen3_30b_a3b_gb300_bf16()
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
    }
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_vrnvl72_fp8mx_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VRNVL72, FP8-MX, NCCL EP (alias of GB300)."""
    cfg = _build_qwen3_30b_a3b_gb300_fp8mx()
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True

    cfg.train.micro_batch_size = 4

    from megatron.core.transformer.enums import AttnBackend

    cfg.model.attention_backend = AttnBackend.auto

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
    }
    return cfg


def qwen3_30b_a3b_pretrain_8gpu_vrnvl72_nvfp4_config() -> ConfigContainer:
    """Qwen3 30B-A3B pretrain: 8× VRNVL72, NVFP4, NCCL EP (same layout as GB300).

    NVFP4's fp4_param_gather path is incompatible with TP comm overlap, so it
    is disabled here.
    """
    cfg = qwen3_30b_a3b_pretrain_8gpu_gb300_nvfp4_config()
    _enable_ncclep(cfg)
    # Device-side expert token counts: the legacy grouped MLP path syncs tokens_per_expert to the
    # host every layer, which serializes the CPU behind the GPU when dispatch is fast.
    cfg.model.moe_use_grouped_tensor = True
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        # NCCL EP dispatcher mode and one GPU per rank.
        "NCCL_EP_HT_EM_PULL_PUSH": 1,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg
