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
"""B300 performance recipes for Llama 3."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.llama.common import (
    CommOverlapConfig,
    ConfigContainer,
    _llama_benchmark_common,
    _perf_precision,
    llama3_8b_pretrain_config,
    llama3_70b_peft_config,
    llama3_70b_pretrain_config,
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
)


def llama3_8b_pretrain_8gpu_b300_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B300, BF16, CUDA graph local."""
    cfg = llama3_8b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 4

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _llama_benchmark_common(cfg)
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
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_8b_pretrain_8gpu_b300_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B300, FP8 current-scaling, CUDA graph local."""
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
    cfg.train.micro_batch_size = 4

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _llama_benchmark_common(cfg)
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
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_8b_pretrain_8gpu_b300_fp8mx_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B300, MXFP8, CUDA graph local."""
    cfg = llama3_8b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 4

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _llama_benchmark_common(cfg)
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
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_8b_pretrain_8gpu_b300_nvfp4_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B300, NVFP4."""
    cfg = llama3_8b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 4

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def llama3_70b_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, BF16, FSDP, GBS=256."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.suggested_communication_unit_size = 800000000
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    _llama_benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_70b_pretrain_64gpu_b300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, FP8 current-scaling, FSDP, GBS=256."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.suggested_communication_unit_size = 800000000
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _llama_benchmark_common(cfg)
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        # CUDA graph and allocator behavior for this recipe.
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 0,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_70b_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, MXFP8, PP=4, GBS=256."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_70b_pretrain_64gpu_b300_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, NVFP4, PP=4, GBS=256."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 256
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
        # NVFP4 fast-math path.
        "NVTE_USE_FAST_MATH": 1,
    }
    return cfg


def llama3_70b_peft_8gpu_b300_bf16_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× B300, BF16."""
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
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_70b_peft_8gpu_b300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× B300, FP8 current-scaling, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_cs")
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
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg


def llama3_70b_peft_8gpu_b300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× B300, MXFP8, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_mx")
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
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False

    cfg.dataset.offline_packing_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

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
        # B300 CPU-affinity behavior.
        "NCCL_IGNORE_CPU_AFFINITY": 1,
    }
    return cfg
