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
"""B300 performance recipes for Llama 3.1."""

from megatron.bridge.perf_recipes.llama.common import (
    ConfigContainer,
    _benchmark_common,
    _perf_precision,
    llama31_405b_pretrain_config,
    userbuffers_bf16_b200_h16384_tp4_cp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h16384_tp4_cp2_mbs1_seqlen8192,
)
from megatron.bridge.perf_recipes.llama.gb300.llama31 import (
    llama31_405b_pretrain_256gpu_gb300_bf16_config,
    llama31_405b_pretrain_256gpu_gb300_fp8cs_config,
    llama31_405b_pretrain_256gpu_gb300_fp8mx_config,
    llama31_405b_pretrain_256gpu_gb300_nvfp4_config,
)


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


llama31_405b_pretrain_256gpu_b300_bf16_config = llama31_405b_pretrain_256gpu_gb300_bf16_config

llama31_405b_pretrain_256gpu_b300_fp8cs_config = llama31_405b_pretrain_256gpu_gb300_fp8cs_config

llama31_405b_pretrain_256gpu_b300_fp8mx_config = llama31_405b_pretrain_256gpu_gb300_fp8mx_config

llama31_405b_pretrain_256gpu_b300_nvfp4_config = llama31_405b_pretrain_256gpu_gb300_nvfp4_config
