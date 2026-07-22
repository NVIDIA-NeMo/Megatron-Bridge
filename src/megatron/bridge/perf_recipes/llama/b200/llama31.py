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
"""B200 performance recipes for Llama 3.1."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.llama.common import (
    ConfigContainer,
    _enable_overlap_param_gather_with_optimizer_step,
    _llama_benchmark_common,
    _perf_precision,
    llama31_405b_pretrain_config,
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
)
from megatron.bridge.perf_recipes.llama.gb200.llama31 import (
    llama31_405b_pretrain_256gpu_gb200_bf16_config,
    llama31_405b_pretrain_256gpu_gb200_fp8cs_config,
    llama31_405b_pretrain_256gpu_gb200_fp8mx_config,
    llama31_405b_pretrain_256gpu_gb200_nvfp4_config,
)


def llama31_405b_pretrain_128gpu_b200_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B200, BF16, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _llama_benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
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
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    }
    return cfg


def llama31_405b_pretrain_128gpu_b200_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B200, FP8 current-scaling, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _llama_benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
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
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    }
    return cfg


def llama31_405b_pretrain_128gpu_b200_fp8mx_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B200, MXFP8, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _llama_benchmark_common(cfg)
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
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    }
    return cfg


def llama31_405b_pretrain_128gpu_b200_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B200, NVFP4, TP=4 PP=16."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 16
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 768
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _llama_benchmark_common(cfg)
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
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


llama31_405b_pretrain_256gpu_b200_bf16_config = llama31_405b_pretrain_256gpu_gb200_bf16_config

llama31_405b_pretrain_256gpu_b200_fp8cs_config = llama31_405b_pretrain_256gpu_gb200_fp8cs_config

llama31_405b_pretrain_256gpu_b200_fp8mx_config = llama31_405b_pretrain_256gpu_gb200_fp8mx_config


def llama31_405b_pretrain_256gpu_b200_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 256x B200, NVFP4."""
    cfg = llama31_405b_pretrain_256gpu_gb200_nvfp4_config()
    cfg.ddp.overlap_param_gather = True
    cfg.optimizer.overlap_param_gather = False
    cfg.comm_overlap.overlap_param_gather = None
    cfg.comm_overlap.align_param_gather = None
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
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg
