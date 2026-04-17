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

"""Flat performance benchmark recipes for Llama 3.1 405B.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.  No dispatch, no layers.

Naming convention::

    {model}_{size}_{task}_{num_gpus}gpu_{gpu}_{precision}_config
"""

from megatron.bridge.recipes.common import _benchmark_common
from megatron.bridge.recipes.llama.llama3 import llama31_405b_pretrain_config
from megatron.bridge.recipes.llama.llama3_perf import _perf_precision
from megatron.bridge.training.comm_overlap import (
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Llama3.1 405B pretrain — 128 GPU, GB300
# =============================================================================


def llama31_405b_pretrain_128gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB300, BF16, FSDP."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 40

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB300, FP8 current-scaling, FSDP."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 10

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB300, MXFP8, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB300, NVFP4, TP=4 PP=8."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = ["full_iteration"]

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain — 128 GPU, GB200
# =============================================================================


def llama31_405b_pretrain_128gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB200, BF16, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB200, FP8 current-scaling, FSDP."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None
    cfg.ddp.num_distributed_optimizer_instances = 2

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 92

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB200, MXFP8, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× GB200, NVFP4, TP=4 PP=16."""
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
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain — 128 GPU, B300
# =============================================================================


def llama31_405b_pretrain_128gpu_b300_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B300, BF16, TP=2 PP=8."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_b300_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B300, FP8 current-scaling, TP=2 PP=8."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_b300_fp8mx_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B300, MXFP8, TP=2 PP=8."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_128gpu_b300_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B300, NVFP4, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain — 128 GPU, B200
# =============================================================================


def llama31_405b_pretrain_128gpu_b200_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 128× B200, BF16, TP=4 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
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
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
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
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
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
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"

    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "block"
    cfg.model.recompute_num_layers = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain — 1024 GPU, H100
# =============================================================================


def llama31_405b_pretrain_1024gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 1024× H100, BF16, TP=8 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_1024gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain: 1024× H100, FP8 current-scaling, TP=8 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192

    cfg.model.moe_token_dispatcher_type = "alltoall"

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain V2 — 256 GPU, GB300
# =============================================================================


def llama31_405b_pretrain_v2_256gpu_gb300_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB300, BF16, FSDP."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.ddp.use_megatron_fsdp = True
    cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
    cfg.ddp.keep_fp8_transpose_cache = False
    cfg.ddp.average_in_collective = False
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.init_model_with_meta_device = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.checkpoint.load = None

    cfg.model.cpu_offloading = True
    cfg.model.cpu_offloading_weights = False
    cfg.model.cpu_offloading_num_layers = 40

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB300, FP8 current-scaling, TP=4 PP=8."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_256gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB300, MXFP8, TP=2 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_256gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB300, NVFP4, TP=4 PP=8."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain V2 — 256 GPU, GB200
# =============================================================================


def llama31_405b_pretrain_v2_256gpu_gb200_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB200, BF16, TP=4 PP=16."""
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
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_256gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB200, FP8 current-scaling, TP=4 PP=16."""
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
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_256gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB200, MXFP8, TP=4 PP=16."""
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
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_256gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 256× GB200, NVFP4, TP=4 PP=16."""
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
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "none"
    cfg.model.cuda_graph_scope = ["full_iteration"]

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192
    cfg.comm_overlap.tp_comm_overlap = False

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain V2 — 1024 GPU, H100
# =============================================================================


def llama31_405b_pretrain_v2_1024gpu_h100_bf16_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 1024× H100, BF16, TP=8 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h16384_tp8_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


def llama31_405b_pretrain_v2_1024gpu_h100_fp8cs_config() -> ConfigContainer:
    """Llama3.1 405B pretrain V2: 1024× H100, FP8 current-scaling, TP=8 PP=8 CP=2."""
    cfg = llama31_405b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True
    cfg.model.seq_length = 8192
    cfg.dataset.seq_length = 8192

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1536
    cfg.train.micro_batch_size = 1

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h16384_tp8_cp2_mbs1_seqlen8192

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Llama3.1 405B pretrain — GPU-count aliases (same config, different scale)
# =============================================================================

llama31_405b_pretrain_512gpu_h100_bf16_config = llama31_405b_pretrain_1024gpu_h100_bf16_config
llama31_405b_pretrain_512gpu_h100_fp8cs_config = llama31_405b_pretrain_1024gpu_h100_fp8cs_config
llama31_405b_pretrain_256gpu_b200_bf16_config = llama31_405b_pretrain_128gpu_b200_bf16_config
llama31_405b_pretrain_256gpu_b200_fp8cs_config = llama31_405b_pretrain_128gpu_b200_fp8cs_config
llama31_405b_pretrain_256gpu_gb200_bf16_config = llama31_405b_pretrain_128gpu_gb200_bf16_config
llama31_405b_pretrain_256gpu_gb200_fp8cs_config = llama31_405b_pretrain_128gpu_gb200_fp8cs_config
llama31_405b_pretrain_256gpu_gb300_bf16_config = llama31_405b_pretrain_128gpu_gb300_bf16_config
llama31_405b_pretrain_256gpu_gb300_fp8cs_config = llama31_405b_pretrain_128gpu_gb300_fp8cs_config
