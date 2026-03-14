# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Flat performance benchmark recipes for Llama3 8B and 70B.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.  No dispatch, no layers.

Naming convention::

    {model}_{size}_{task}_{num_gpus}gpu_{gpu}_{precision}_config

Precision short-names:
    bf16   = BF16 mixed precision
    fp8cs  = FP8 per-tensor current-scaling (first/last BF16 layers disabled)
    fp8mx  = MXFP8
    nvfp4  = NVFP4
"""

from megatron.bridge.recipes.common import _benchmark_common
from megatron.bridge.recipes.llama.llama3 import (
    llama3_8b_pretrain_config,
    llama3_8b_sft_config,
    llama3_70b_peft_config,
    llama3_70b_pretrain_config,
    llama3_70b_sft_config,
)
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import (
    bf16_mixed,
    bf16_with_fp8_current_scaling_mixed,
    bf16_with_mxfp8_mixed,
    bf16_with_nvfp4_mixed,
)


def _perf_precision(compute_dtype: str):
    """Return mixed-precision config tuned for perf benchmarks.

    Identical to ``scripts/performance/utils/precision.get_precision_config``
    but importable from the library side.
    """
    if compute_dtype == "bf16":
        return bf16_mixed()
    if compute_dtype == "fp8_cs":
        cfg = bf16_with_fp8_current_scaling_mixed()
        cfg.first_last_layers_bf16 = False
        return cfg
    if compute_dtype == "fp8_mx":
        return bf16_with_mxfp8_mixed()
    if compute_dtype == "nvfp4":
        return bf16_with_nvfp4_mixed()
    raise ValueError(f"Unknown compute_dtype: {compute_dtype}")


# =============================================================================
# Llama3 70B pretrain — 64 GPU, GB300
# =============================================================================


def llama3_70b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB300, BF16, FSDP + NCCL UB."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.suggested_communication_unit_size = 800000000
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None
    cfg.ddp.nccl_ub = True
    cfg.ddp.fsdp_manual_registration = True

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 30

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB300, FP8 current-scaling, FSDP."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.suggested_communication_unit_size = 800000000
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 20

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB300, MXFP8, PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB300, NVFP4, PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B pretrain — 64 GPU, GB200
# =============================================================================


def llama3_70b_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB200, BF16, FSDP."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
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

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 20

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB200, FP8 current-scaling, FSDP."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.suggested_communication_unit_size = 800000000
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 40

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB200, MXFP8, TP=2 PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× GB200, NVFP4, TP=2 PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B pretrain — 64 GPU, B300
# =============================================================================


def llama3_70b_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, BF16, FSDP."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
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

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_b300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, FP8 current-scaling, FSDP."""
    cfg = llama3_70b_pretrain_config()
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

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, MXFP8, PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_b300_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B300, NVFP4, PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B pretrain — 64 GPU, B200
# =============================================================================


def llama3_70b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B200, BF16, TP=2 PP=4 CP=2, CUDA graph local."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_b200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B200, FP8 current-scaling, FSDP."""
    cfg = llama3_70b_pretrain_config()
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

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.ddp.suggested_communication_unit_size = 800000000
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 5

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B200, MXFP8, TP=2 PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_b200_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× B200, NVFP4, TP=2 PP=4."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B pretrain — 64 GPU, H100
# =============================================================================


def llama3_70b_pretrain_64gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× H100, BF16, TP=4 PP=4 CP=2."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama3_70b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain: 64× H100, FP8 current-scaling, TP=4 PP=8."""
    cfg = llama3_70b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B pretrain — 8 GPU, R100
# =============================================================================


def llama3_8b_pretrain_8gpu_r100_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× R100, BF16."""
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
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_r100_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× R100, FP8 current-scaling."""
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

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_r100_fp8mx_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× R100, MXFP8."""
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
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_r100_nvfp4_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× R100, NVFP4."""
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
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B pretrain — 8 GPU, GB300
# =============================================================================


def llama3_8b_pretrain_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB300, BF16, CUDA graph local."""
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

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB300, FP8 current-scaling, CUDA graph local."""
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

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB300, MXFP8, CUDA graph local."""
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

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB300, NVFP4, CUDA graph local."""
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

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B pretrain — 8 GPU, GB200
# =============================================================================


def llama3_8b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB200, BF16, CUDA graph local."""
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
    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB200, FP8 current-scaling, CUDA graph local."""
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
    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB200, MXFP8, CUDA graph local."""
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
    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× GB200, NVFP4."""
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

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B pretrain — 8 GPU, B300
# =============================================================================


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

    _benchmark_common(cfg)
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

    _benchmark_common(cfg)
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

    _benchmark_common(cfg)
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

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B pretrain — 8 GPU, B200
# =============================================================================


def llama3_8b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B200, BF16, CUDA graph local."""
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
    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_b200_fp8cs_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B200, FP8 current-scaling, CUDA graph local."""
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
    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B200, MXFP8, CUDA graph local."""
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
    cfg.train.micro_batch_size = 2

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]
    cfg.rng.te_rng_tracker = cfg.model.use_te_rng_tracker = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def llama3_8b_pretrain_8gpu_b200_nvfp4_config() -> ConfigContainer:
    """Llama3 8B pretrain: 8× B200, NVFP4."""
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

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B pretrain — 8 GPU, H100
# =============================================================================


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

    _benchmark_common(cfg)
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

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B pretrain V2 (GBS=256) — 64 GPU, GB300
# =============================================================================


def llama3_70b_pretrain_v2_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB300, BF16, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb300_bf16_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB300, FP8 current-scaling, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb300_fp8cs_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB300, MXFP8, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb300_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB300, NVFP4, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb300_nvfp4_config()
    cfg.train.global_batch_size = 256
    return cfg


# =============================================================================
# Llama3 70B pretrain V2 (GBS=256) — 64 GPU, GB200
# =============================================================================


def llama3_70b_pretrain_v2_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB200, BF16, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb200_bf16_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB200, FP8 current-scaling, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb200_fp8cs_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB200, MXFP8, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb200_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× GB200, NVFP4, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_gb200_nvfp4_config()
    cfg.train.global_batch_size = 256
    return cfg


# =============================================================================
# Llama3 70B pretrain V2 (GBS=256) — 64 GPU, B300
# =============================================================================


def llama3_70b_pretrain_v2_64gpu_b300_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B300, BF16, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b300_bf16_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_b300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B300, FP8 current-scaling, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b300_fp8cs_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B300, MXFP8, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b300_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_b300_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B300, NVFP4, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b300_nvfp4_config()
    cfg.train.global_batch_size = 256
    return cfg


# =============================================================================
# Llama3 70B pretrain V2 (GBS=256) — 64 GPU, B200
# =============================================================================


def llama3_70b_pretrain_v2_64gpu_b200_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B200, BF16, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b200_bf16_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_b200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B200, FP8 current-scaling, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b200_fp8cs_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B200, MXFP8, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b200_fp8mx_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_b200_nvfp4_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× B200, NVFP4, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_b200_nvfp4_config()
    cfg.train.global_batch_size = 256
    return cfg


# =============================================================================
# Llama3 70B pretrain V2 (GBS=256) — 64 GPU, H100
# =============================================================================


def llama3_70b_pretrain_v2_64gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× H100, BF16, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_h100_bf16_config()
    cfg.train.global_batch_size = 256
    return cfg


def llama3_70b_pretrain_v2_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3 70B pretrain V2: 64× H100, FP8 current-scaling, GBS=256."""
    cfg = llama3_70b_pretrain_64gpu_h100_fp8cs_config()
    cfg.train.global_batch_size = 256
    return cfg


# =============================================================================
# Llama3 8B SFT — 8 GPU, GB200
# =============================================================================


def llama3_8b_sft_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3 8B SFT: 8× GB200, BF16, seq_length=16384."""
    cfg = llama3_8b_sft_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.seq_length = 16384
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["mlp"]

    _benchmark_common(cfg)
    return cfg


def llama3_8b_sft_8gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3 8B SFT: 8× GB200, FP8 current-scaling, seq_length=16384."""
    cfg = llama3_8b_sft_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True

    cfg.model.seq_length = 16384
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["mlp"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 8B SFT — 8 GPU, H100
# =============================================================================


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

    _benchmark_common(cfg)
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

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B SFT — 32 GPU, GB300
# =============================================================================


def llama3_70b_sft_32gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3 70B SFT: 32× GB300, BF16, PP=2 VP=20."""
    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 20
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.defer_embedding_wgrad_compute = True
    cfg.comm_overlap.wgrad_deferral_limit = 22

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_sft_32gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B SFT: 32× GB300, FP8 current-scaling, PP=2 VP=20."""
    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 20
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.defer_embedding_wgrad_compute = True
    cfg.comm_overlap.wgrad_deferral_limit = 22

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B SFT — 32 GPU, GB200
# =============================================================================


def llama3_70b_sft_32gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3 70B SFT: 32× GB200, BF16, PP=8 VP=10."""
    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = 10
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.defer_embedding_wgrad_compute = True
    cfg.comm_overlap.wgrad_deferral_limit = 22

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_sft_32gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B SFT: 32× GB200, FP8 current-scaling, PP=8 VP=10."""
    cfg = llama3_70b_sft_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.model.seq_length = 4096
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.virtual_pipeline_model_parallel_size = 10
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 32
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False
    cfg.comm_overlap.defer_embedding_wgrad_compute = True
    cfg.comm_overlap.wgrad_deferral_limit = 22

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B SFT — 32 GPU, H100
# =============================================================================


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
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B LoRA — 8 GPU, GB300
# =============================================================================


def llama3_70b_lora_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× GB300, BF16."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× GB300, FP8 current-scaling."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× GB300, MXFP8, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B LoRA — 8 GPU, GB200
# =============================================================================


def llama3_70b_lora_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× GB200, BF16, GBS=64, seq_length=2048."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    cfg.peft.target_modules = ["linear_qkv"]
    cfg.model.seq_length = 2048
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mlp"]

    cfg.comm_overlap.tp_comm_overlap = False

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× GB200, FP8 current-scaling, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× GB200, MXFP8, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B LoRA — 8 GPU, B300
# =============================================================================


def llama3_70b_lora_8gpu_b300_bf16_config() -> ConfigContainer:
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
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_b300_fp8cs_config() -> ConfigContainer:
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
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_b300_fp8mx_config() -> ConfigContainer:
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
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B LoRA — 8 GPU, B200
# =============================================================================


def llama3_70b_lora_8gpu_b200_bf16_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× B200, BF16, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_b200_fp8cs_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× B200, FP8 current-scaling, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Llama3 70B LoRA: 8× B200, MXFP8, PP=2."""
    cfg = llama3_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False
    cfg.model.disable_parameter_transpose_cache = True
    cfg.ddp.use_distributed_optimizer = True
    cfg.optimizer.use_distributed_optimizer = True
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3 70B LoRA — 8 GPU, H100
# =============================================================================


def llama3_70b_lora_8gpu_h100_bf16_config() -> ConfigContainer:
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
    if cfg.comm_overlap is None:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg


def llama3_70b_lora_8gpu_h100_fp8cs_config() -> ConfigContainer:
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

    cfg.dataset.packed_sequence_specs.pad_cu_seqlens = True
    cfg.dataset.dataset_kwargs = {"pad_to_max_length": True}

    _benchmark_common(cfg)
    return cfg
