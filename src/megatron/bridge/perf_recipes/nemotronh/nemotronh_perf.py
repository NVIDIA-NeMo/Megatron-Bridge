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

"""Flat performance benchmark recipes for NemotronH and Nemotron 3 models.

Each function is self-contained: call library recipe, override fields, call
``_benchmark_common()``, return.  No dispatch, no layers.

Naming convention::

    {model}_{size}_{task}_{num_gpus}gpu_{gpu}_{precision}_config

Precision short-names:
    bf16   = BF16 mixed precision
    fp8cs  = FP8 per-tensor current-scaling
    fp8mx  = MXFP8
    nvfp4  = NVFP4
"""

from pathlib import Path

from megatron.core.quantization.utils import load_quantization_recipe

from megatron.bridge.perf_recipes._common import _benchmark_common, _perf_precision
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import nemotron_3_nano_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotron_3_super import nemotron_3_super_pretrain_config
from megatron.bridge.recipes.nemotronh.nemotronh import nemotronh_56b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, nemotron_3_super_bf16_with_nvfp4_mixed


def _with_global_batch_size(cfg: ConfigContainer, global_batch_size: int) -> ConfigContainer:
    cfg.train.global_batch_size = global_batch_size
    return cfg


def _nemotron_3_super_precision(compute_dtype: str) -> MixedPrecisionConfig:
    """Return the precision config used by Nemotron 3 Super perf recipes."""
    if compute_dtype == "nvfp4":
        cfg = nemotron_3_super_bf16_with_nvfp4_mixed()
        # Disabled until MCore PR 4358 lands.
        cfg.fp4_param_gather = False
        return cfg
    return _perf_precision(compute_dtype)


def _finalize_nemotron_3_super_perf(cfg: ConfigContainer, compute_dtype: str) -> None:
    """Apply Nemotron 3 Super perf defaults after recipe-specific overrides."""
    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    cfg.model.moe_router_force_load_balancing = True
    cfg.checkpoint.async_save = False

    if compute_dtype in {"fp8_mx", "nvfp4"}:
        cfg.model.moe_router_padding_for_quantization = True
    if compute_dtype == "nvfp4":
        cfg.model.quant_recipe = load_quantization_recipe(str(Path(__file__).with_name("te_quant.cfg")))

    _benchmark_common(cfg)


def _nemotron_3_super_pretrain_64gpu_config(compute_dtype: str, gpu: str) -> ConfigContainer:
    """Build a Nemotron 3 Super pretrain config for 64 GPUs."""
    cfg = nemotron_3_super_pretrain_config()
    cfg.mixed_precision = _nemotron_3_super_precision(compute_dtype)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.expert_model_parallel_size = 64
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_token_dispatcher_type = "flex"
    cfg.model.moe_shared_expert_overlap = False

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    if gpu == "gb200":
        cfg.model.tensor_model_parallel_size = 2
    elif gpu == "b300":
        cfg.model.expert_model_parallel_size = 8
        if compute_dtype in {"bf16", "nvfp4"}:
            cfg.model.recompute_modules = ["moe_act", "layernorm"]
    elif gpu == "b200":
        cfg.model.cuda_graph_impl = "none"
        cfg.model.recompute_modules = ["moe_act", "layernorm"]
        if compute_dtype == "bf16":
            cfg.model.recompute_modules = ["moe_act", "moe", "layernorm", "core_attn"]
        elif compute_dtype == "nvfp4":
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.cuda_graph_impl = "transformer_engine"
            cfg.model.cuda_graph_scope = ["mamba", "attn", "moe_router", "moe_preprocess"]
            cfg.model.recompute_modules = None
    elif gpu != "gb300":
        raise ValueError(f"Unsupported Nemotron 3 Super 64-GPU target: {gpu}")

    _finalize_nemotron_3_super_perf(cfg, compute_dtype)
    return cfg


# =============================================================================
# NemotronH 56B pretrain — 64 GPU, GB300
# =============================================================================


def nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× GB300, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba", "attn"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# NemotronH 56B pretrain — 64 GPU, GB200
# =============================================================================


def nemotronh_56b_pretrain_64gpu_gb200_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× GB200, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba", "attn"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# NemotronH 56B pretrain — 64 GPU, B300
# =============================================================================


def nemotronh_56b_pretrain_64gpu_b300_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× B300, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba", "attn"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# NemotronH 56B pretrain — 64 GPU, B200
# =============================================================================


def nemotronh_56b_pretrain_64gpu_b200_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× B200, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba", "attn"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# NemotronH 56B pretrain — 64 GPU, H100
# =============================================================================


def nemotronh_56b_pretrain_64gpu_h100_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 64× H100, FP8 current-scaling."""
    cfg = nemotronh_56b_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = True
    cfg.train.global_batch_size = 192
    cfg.train.micro_batch_size = 1

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba"]

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Nemotron 3 Super pretrain — 64 GPU, GB300
# =============================================================================


def nemotron_3_super_pretrain_64gpu_gb300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB300, BF16."""
    return _nemotron_3_super_pretrain_64gpu_config("bf16", "gb300")


def nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB300, MXFP8."""
    return _nemotron_3_super_pretrain_64gpu_config("fp8_mx", "gb300")


def nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB300, NVFP4."""
    return _nemotron_3_super_pretrain_64gpu_config("nvfp4", "gb300")


# =============================================================================
# Nemotron 3 Super pretrain — 64 GPU, GB200
# =============================================================================


def nemotron_3_super_pretrain_64gpu_gb200_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB200, BF16."""
    return _nemotron_3_super_pretrain_64gpu_config("bf16", "gb200")


def nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB200, MXFP8."""
    return _nemotron_3_super_pretrain_64gpu_config("fp8_mx", "gb200")


def nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× GB200, NVFP4."""
    return _nemotron_3_super_pretrain_64gpu_config("nvfp4", "gb200")


# =============================================================================
# Nemotron 3 Super pretrain — 64 GPU, B300
# =============================================================================


def nemotron_3_super_pretrain_64gpu_b300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B300, BF16."""
    return _nemotron_3_super_pretrain_64gpu_config("bf16", "b300")


def nemotron_3_super_pretrain_64gpu_b300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B300, MXFP8."""
    return _nemotron_3_super_pretrain_64gpu_config("fp8_mx", "b300")


def nemotron_3_super_pretrain_64gpu_b300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B300, NVFP4."""
    return _nemotron_3_super_pretrain_64gpu_config("nvfp4", "b300")


# =============================================================================
# Nemotron 3 Super pretrain — 64 GPU, B200
# =============================================================================


def nemotron_3_super_pretrain_64gpu_b200_bf16_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B200, BF16."""
    return _nemotron_3_super_pretrain_64gpu_config("bf16", "b200")


def nemotron_3_super_pretrain_64gpu_b200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B200, MXFP8."""
    return _nemotron_3_super_pretrain_64gpu_config("fp8_mx", "b200")


def nemotron_3_super_pretrain_64gpu_b200_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Super pretrain: 64× B200, NVFP4."""
    return _nemotron_3_super_pretrain_64gpu_config("nvfp4", "b200")


# =============================================================================
# Nemotron 3 Nano pretrain — 8 GPU, GB300
# =============================================================================


def nemotron_3_nano_pretrain_8gpu_gb300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB300, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB300, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB300, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Nemotron 3 Nano pretrain — 8 GPU, GB200
# =============================================================================


def nemotron_3_nano_pretrain_8gpu_gb200_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB200, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB200, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× GB200, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Nemotron 3 Nano pretrain — 8 GPU, B300
# =============================================================================


def nemotron_3_nano_pretrain_8gpu_b300_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B300, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B300, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B300, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Nemotron 3 Nano pretrain — 8 GPU, B200
# =============================================================================


def nemotron_3_nano_pretrain_8gpu_b200_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B200, BF16."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B200, MXFP8."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_mx")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 8× B200, NVFP4."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("nvfp4")

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2

    cfg.model.moe_flex_dispatcher_backend = "hybridep"
    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba", "moe_router", "moe_preprocess"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# Nemotron 3 Nano pretrain — 16 GPU, H100
# =============================================================================


def nemotron_3_nano_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, BF16, recompute MoE+layernorm."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("bf16")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["attn", "mamba"]

    cfg.model.recompute_modules = ["moe", "layernorm"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


def nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config() -> ConfigContainer:
    """Nemotron 3 Nano pretrain: 16× H100, FP8 current-scaling, recompute."""
    cfg = nemotron_3_nano_pretrain_config()
    cfg.mixed_precision = _perf_precision("fp8_cs")
    cfg.model.recompute_granularity = "selective"

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.sequence_parallel = False
    cfg.model.expert_model_parallel_size = 8
    cfg.train.global_batch_size = 1024
    cfg.train.micro_batch_size = 1

    cfg.model.moe_router_force_load_balancing = True

    cfg.model.cuda_graph_impl = "transformer_engine"
    cfg.model.cuda_graph_scope = ["mamba"]

    cfg.model.recompute_modules = ["moe", "layernorm", "core_attn", "moe_act"]

    cfg.comm_overlap.tp_comm_overlap = True

    _benchmark_common(cfg)
    return cfg


# =============================================================================
# NemotronH 56B pretrain — 256 GPU aliases + BF16 variants
# =============================================================================


def nemotronh_56b_pretrain_256gpu_b200_bf16_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 256× B200, BF16 (same layout as FP8-CS)."""
    cfg = nemotronh_56b_pretrain_64gpu_b200_fp8cs_config()
    cfg.mixed_precision = _perf_precision("bf16")
    return cfg


def nemotronh_56b_pretrain_256gpu_b200_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 256× B200, FP8 current-scaling, legacy-scaled GBS."""
    return _with_global_batch_size(nemotronh_56b_pretrain_64gpu_b200_fp8cs_config(), 768)


def nemotronh_56b_pretrain_256gpu_gb300_bf16_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 256× GB300, BF16 (same layout as FP8-CS)."""
    cfg = nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config()
    cfg.mixed_precision = _perf_precision("bf16")
    return cfg


def nemotronh_56b_pretrain_256gpu_gb300_fp8cs_config() -> ConfigContainer:
    """NemotronH 56B pretrain: 256× GB300, FP8 current-scaling, legacy-scaled GBS."""
    return _with_global_batch_size(nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config(), 768)
