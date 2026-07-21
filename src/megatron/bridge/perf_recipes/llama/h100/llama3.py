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
"""H100 performance recipes for Llama 3."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.llama.common import (
    CommOverlapConfig,
    ConfigContainer,
    _enable_overlap_param_gather_with_optimizer_step,
    _llama_benchmark_common,
    _perf_precision,
    _with_global_batch_size,
    llama3_8b_pretrain_config,
    llama3_8b_sft_config,
    llama3_70b_peft_config,
    llama3_70b_pretrain_config,
    llama3_70b_sft_config,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)


def llama3_8b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× H100, BF16, CP=2."""
    cfg = llama3_8b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _llama_benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
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


def llama3_8b_pretrain_8gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× H100, FP8 current-scaling, recompute."""
    cfg = llama3_8b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 5

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.moe_token_dispatcher_type = "alltoall"

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
        "NCCL_CTA_POLICY": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def llama3_70b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× H100, BF16, TP=4 PP=4 CP=2, GBS=256."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _llama_benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
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


def llama3_70b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× H100, FP8 current-scaling, TP=4 PP=8, GBS=256."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _llama_benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
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


def llama3_8b_sft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 8B SFT: 8× H100, BF16, seq_length=4096."""
    cfg = llama3_8b_sft_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

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
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def llama3_8b_sft_8gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 8B SFT: 8× H100, FP8 current-scaling, seq_length=4096."""
    cfg = llama3_8b_sft_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["mlp"]

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
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg


def llama3_70b_sft_32gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 70B SFT: 32× H100, BF16, TP=4 PP=4 VP=5."""
    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap = True
    cfg.comm_overlap.defer_embedding_wgrad_compute = True
    cfg.comm_overlap.wgrad_deferral_limit = 22

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _llama_benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    }
    return cfg


def llama3_70b_sft_32gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 70B SFT: 32× H100, FP8 current-scaling, TP=4 PP=4 VP=5."""
    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap = CommOverlapConfig(
        tp_comm_overlap=True,
        defer_embedding_wgrad_compute=True,
        wgrad_deferral_limit=22,
    )

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _llama_benchmark_common(cfg)
    _enable_overlap_param_gather_with_optimizer_step(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        # Pipeline communication tuning for this layout.
        "NCCL_P2P_NET_CHUNKSIZE": 2097152,
    }
    return cfg


def llama3_70b_peft_8gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× H100, BF16, PP=4 VP=20, recompute."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.peft.target_modules = ["linear_qkv"]
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 20
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 1

    cfg.comm_overlap.tp_comm_overlap = False

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

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
    }
    return cfg


def llama3_70b_peft_8gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× H100, FP8 current-scaling, TP=2 PP=4 VP=20."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True

    cfg.peft.target_modules = ["linear_qkv"]
    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 20
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = False
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _llama_benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
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


def llama3_8b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 64× H100, BF16, legacy-scaled GBS."""
    cfg = _with_global_batch_size(llama3_8b_pretrain_8gpu_h100_bf16_config(), 1024)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
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


def llama3_8b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 64× H100, FP8 current-scaling, legacy-scaled GBS."""
    cfg = _with_global_batch_size(llama3_8b_pretrain_8gpu_h100_fp8cs_config(), 1024)
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
        "NCCL_CTA_POLICY": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # Use cuDNN LayerNorm for this measured baseline.
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
    }
    return cfg
