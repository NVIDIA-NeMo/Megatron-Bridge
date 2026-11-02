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
"""GB300 performance recipes for Qwen3.5-VL."""

import torch

from megatron.bridge.perf_recipes._common import _enable_ncclep
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.qwen_vl.common import (
    CommOverlapConfig,
    ConfigContainer,
    _benchmark_common,
    _enable_partial_cuda_graphs,
    _perf_precision,
    _qwen35_vl_common,
    _qwen35_vl_post,
    _select_gdn_kernel_backend,
    qwen35_vl_35b_a3b_pretrain_mock_config,
    qwen35_vl_122b_a10b_pretrain_mock_config,
    qwen35_vl_397b_a17b_pretrain_mock_config,
)


def _build_qwen35_vl_35b_a3b_gb300_bf16() -> ConfigContainer:
    """Shared HybridEP BF16 base for the Qwen3.5-VL 35B-A3B GB300 recipes."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post(cfg)
    _enable_partial_cuda_graphs(cfg)
    _select_gdn_kernel_backend(cfg, "transformer_engine")
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB300, BF16, EP=8, NCCL EP."""
    cfg = _build_qwen35_vl_35b_a3b_gb300_bf16()
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


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB300, FP8 current-scaling."""
    cfg = _build_qwen35_vl_35b_a3b_gb300_bf16()
    cfg.mixed_precision = _perf_precision("fp8_cs")
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
    }
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB300, MXFP8 with the CuTe DSL grouped MLP."""
    cfg = _build_qwen35_vl_35b_a3b_gb300_bf16()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    # Route the MoE experts through Transformer Engine's CuTe DSL fused grouped
    # MLP (ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8), which fuses FC1 + SwiGLU +
    # FC2 for the MXFP8 recipe on SM100+.
    #
    # All three settings below are required together. Enabling the op fuser
    # without the 32-wide GLU interleaving does NOT fail: Transformer Engine
    # silently declines to match the joint fusion and falls back to the cuBLASLt
    # grouped GEMM, so the recipe would look enabled while running unfused.
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

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
        # Gate for the CuTe DSL fused grouped MLP. Transformer Engine checks this
        # env var when it registers the joint fusion, so it must be set for
        # use_transformer_engine_op_fuser above to take effect.
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB300, BF16, EP=32."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post(cfg)
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


def qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB300, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
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


def qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB300, MXFP8."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
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


def qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB300, BF16, EP=64."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
    )

    _benchmark_common(cfg)
    _qwen35_vl_post(cfg)
    _enable_partial_cuda_graphs(cfg)
    _select_gdn_kernel_backend(cfg, "transformer_engine")
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


def qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB300, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
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


def _build_qwen35_vl_397b_a17b_gb300_fp8mx() -> ConfigContainer:
    """Shared MXFP8 base for the Qwen3.5-VL 397B-A17B GB300 and VRNVL72 recipes."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB300, MXFP8.

    Matches the tuned 64× GB300 configuration: on top of the MXFP8 base it adds the CuTe DSL fused grouped MLP,
    vision-encoder CUDA graphs, the precision-aware optimizer with BF16 Adam moments, selective recompute, 128-token
    HybridEP dispatch and preprocessing chunks, and a LayerNorm SM margin of 16.
    """
    cfg = _build_qwen35_vl_397b_a17b_gb300_fp8mx()

    # CuTe DSL fused grouped MLP for the experts, as in the 35B MXFP8 recipe above. The op fuser, the 32-wide
    # GLU interleaving and NVTE_CUTEDSL_FUSED_GROUPED_MLP below are required together; with the op fuser,
    # Megatron-Core also runs the experts on grouped tensors.
    cfg.model.use_transformer_engine_op_fuser = True
    cfg.model.moe_mlp_glu_interleave_size = 32

    # Vision-encoder CUDA graph. The cap (vision patches per microbatch) is the largest real-data vision sequence
    # measured at MBS 1 plus 5%; the 784-patch mock images fit and are padded to it on every replay. Raise it with
    # MBS or images per sample. As in the VRNVL72 recipes, the graphed encoder runs without the 2D rotary embedding,
    # and images that share a microbatch attend to each other and to the padding.
    cfg.model.vision_cuda_graph_impl = "transformer_engine"
    cfg.model.vision_cuda_graph_scope = ["attn", "mlp"]
    cfg.model.max_vision_cuda_graph_seq_length = 1029

    # FP32 main parameters and gradients, BF16 Adam moments.
    cfg.optimizer.use_precision_aware_optimizer = True
    cfg.optimizer.main_params_dtype = torch.float32
    cfg.optimizer.main_grads_dtype = torch.float32
    cfg.optimizer.exp_avg_dtype = torch.bfloat16
    cfg.optimizer.exp_avg_sq_dtype = torch.bfloat16

    # Selective recompute, which _qwen35_vl_common turns off for the shared perf settings.
    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn", "gdn_norm_out"]

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
        # HybridEP dispatch and preprocessing chunks (HybridEP's default is 64).
        "NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API": 128,
        "NUM_OF_TOKENS_PER_CHUNK_PREPROCESSING_API": 128,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
        # Gate for the CuTe DSL fused grouped MLP enabled above. Transformer Engine checks it when it
        # registers the fusion; without it the op fuser runs the unfused path.
        "NVTE_CUTEDSL_FUSED_GROUPED_MLP": 1,
    }
    return cfg
