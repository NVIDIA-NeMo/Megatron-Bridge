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

"""Flat performance benchmark recipes for Qwen3.5-VL MoE models.

Each function is self-contained: call base recipe, apply VLM + parallelism
overrides, call ``_benchmark_common()``, return.

Naming convention::

    qwen35_vl_{size}_{task}_{num_gpus}gpu_{gpu}_{precision}_config

Models:

- **35B-A3B**: 8 experts, 3B active parameters
- **122B-A10B**: 8 experts, 10B active parameters
- **397B-A17B**: 8 experts, 17B active parameters
"""

from megatron.bridge.recipes.common import _benchmark_common
from megatron.bridge.recipes.llama.llama3_perf import _perf_precision
from megatron.bridge.recipes.qwen_vl.qwen35_vl import (
    qwen35_vl_35b_a3b_pretrain_mock_config,
    qwen35_vl_122b_a10b_pretrain_mock_config,
    qwen35_vl_397b_a17b_pretrain_mock_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


# =============================================================================
# Helper — shared VLM perf overrides for all Qwen3.5-VL configs
# =============================================================================


def _qwen35_vl_perf_common(cfg: ConfigContainer) -> None:
    """Apply VLM-specific performance benchmark settings for Qwen3.5-VL.

    Must be called before ``_benchmark_common`` and after setting precision.
    """
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True

    cfg.model.seq_length = 4096
    cfg.dataset.seq_length = 4096

    cfg.model.moe_router_force_load_balancing = True

    cfg.ddp.overlap_grad_reduce = False
    cfg.ddp.overlap_param_gather = False

    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False


# =============================================================================
# Qwen3.5-VL 35B-A3B pretrain — 8 GPU, GB300
# =============================================================================


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB300, BF16, EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 8

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB300, FP8 current-scaling."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB300, MXFP8."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 35B-A3B pretrain — 8 GPU, B300
# =============================================================================


def qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× B300, BF16, EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 8

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× B300, FP8 current-scaling."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× B300, MXFP8."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 35B-A3B pretrain — 8 GPU, GB200
# =============================================================================


def qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB200, BF16, EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

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

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB200, FP8 current-scaling."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× GB200, MXFP8 (no attn CUDA graph)."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]
    return cfg


# =============================================================================
# Qwen3.5-VL 35B-A3B pretrain — 8 GPU, B200
# =============================================================================


def qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× B200, BF16, EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× B200, FP8 current-scaling."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 8× B200, MXFP8."""
    cfg = qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 35B-A3B pretrain — 16 GPU, H100
# =============================================================================


def qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 16× H100, BF16, PP=2 VP=12 EP=8."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 35B-A3B pretrain: 16× H100, FP8 current-scaling, PP=2 VP=12."""
    cfg = qwen35_vl_35b_a3b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Qwen3.5-VL 122B-A10B pretrain — 32 GPU, GB300
# =============================================================================


def qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB300, BF16, EP=32."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

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

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB300, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB300, MXFP8."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 122B-A10B pretrain — 32 GPU, B300
# =============================================================================


def qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× B300, BF16, EP=32."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

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

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× B300, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× B300, MXFP8."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 122B-A10B pretrain — 32 GPU, GB200
# =============================================================================


def qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB200, BF16, PP=4 EP=8."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB200, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× GB200, MXFP8."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 122B-A10B pretrain — 32 GPU, B200
# =============================================================================


def qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× B200, BF16, PP=4 VP=4 EP=8."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× B200, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 32× B200, MXFP8."""
    cfg = qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 122B-A10B pretrain — 128 GPU, H100
# =============================================================================


def qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 128× H100, BF16, TP=2 PP=8 VP=4 EP=16."""
    cfg = qwen35_vl_122b_a10b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 16
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 122B-A10B pretrain: 128× H100, FP8 current-scaling."""
    cfg = qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


# =============================================================================
# Qwen3.5-VL 397B-A17B pretrain — 64 GPU, GB300
# =============================================================================


def qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB300, BF16, EP=64."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB300, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB300, MXFP8."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 397B-A17B pretrain — 64 GPU, B300
# =============================================================================


def qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× B300, BF16, EP=64."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 64
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× B300, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× B300, MXFP8."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 397B-A17B pretrain — 64 GPU, GB200
# =============================================================================


def qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB200, BF16, PP=8 EP=8."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB200, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× GB200, MXFP8."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 397B-A17B pretrain — 64 GPU, B200
# =============================================================================


def qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× B200, BF16, PP=8 VP=4 EP=8."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=True)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× B200, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg


def qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 64× B200, MXFP8."""
    cfg = qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")
    return cfg


# =============================================================================
# Qwen3.5-VL 397B-A17B pretrain — 256 GPU, H100
# =============================================================================


def qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 256× H100, BF16, TP=2 PP=8 VP=4 EP=32."""
    cfg = qwen35_vl_397b_a17b_pretrain_mock_config()
    cfg.mixed_precision = _perf_precision("bf16")
    _qwen35_vl_perf_common(cfg)

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 32
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_a2a_overlap = True

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)

    _benchmark_common(cfg)
    return cfg


def qwen35_vl_397b_a17b_pretrain_256gpu_h100_fp8cs_config() -> ConfigContainer:
    """Qwen3.5-VL 397B-A17B pretrain: 256× H100, FP8 current-scaling."""
    cfg = qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    return cfg
