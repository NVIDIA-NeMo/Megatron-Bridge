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
"""GB300 performance recipes for NeMoDiag V0."""

from megatron.bridge.perf_recipes.nemodiag.common import (
    _benchmark_common,
    _enable_nemodiag_full_iteration_mxfp8,
    _enable_overlap_param_gather_with_optimizer_step,
    _nemodiag_v0_common,
    _perf_precision,
    nemodiag_v0_pretrain_config,
)
from megatron.bridge.training.config import ConfigContainer


def _nemodiag_v0_gb300_config(*, precision: str, global_batch_size: int) -> ConfigContainer:
    cfg = nemodiag_v0_pretrain_config()
    cfg.mixed_precision = _perf_precision(precision)
    cfg.train.global_batch_size = global_batch_size
    _nemodiag_v0_common(cfg)

    if precision == "bf16":
        cfg.train.micro_batch_size = 1
        cfg.model.recompute_modules = ["moe_act"]
        cfg.model.cuda_graph_impl = "transformer_engine"
        cfg.model.cuda_graph_scope = ["attn", "moe_router", "moe_preprocess"]
        _benchmark_common(cfg)
        _enable_overlap_param_gather_with_optimizer_step(cfg)
    elif precision == "fp8_mx":
        cfg.train.micro_batch_size = 1
        cfg.model.recompute_modules = []
        _benchmark_common(cfg)
        _enable_nemodiag_full_iteration_mxfp8(
            cfg,
            fp8_dot_product_attention=True,
            fp8_output_proj=True,
        )
    elif precision == "nvfp4":
        cfg.train.micro_batch_size = 2
        cfg.model.recompute_modules = ["mla_up_proj"]
        cfg.model.cuda_graph_scope = []
        _benchmark_common(cfg)
    else:
        raise ValueError(f"Unsupported NeMoDiag V0 precision: {precision}")

    return cfg


def nemodiag_v0_pretrain_72gpu_gb300_bf16_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 72 GB300 GPUs, BF16."""
    return _nemodiag_v0_gb300_config(precision="bf16", global_batch_size=1152)


def nemodiag_v0_pretrain_144gpu_gb300_bf16_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 144 GB300 GPUs, BF16."""
    return _nemodiag_v0_gb300_config(precision="bf16", global_batch_size=2304)


def nemodiag_v0_pretrain_288gpu_gb300_bf16_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 288 GB300 GPUs, BF16."""
    return _nemodiag_v0_gb300_config(precision="bf16", global_batch_size=4608)


def nemodiag_v0_pretrain_72gpu_gb300_fp8mx_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 72 GB300 GPUs, MXFP8."""
    return _nemodiag_v0_gb300_config(precision="fp8_mx", global_batch_size=1152)


def nemodiag_v0_pretrain_144gpu_gb300_fp8mx_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 144 GB300 GPUs, MXFP8."""
    return _nemodiag_v0_gb300_config(precision="fp8_mx", global_batch_size=2304)


def nemodiag_v0_pretrain_288gpu_gb300_fp8mx_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 288 GB300 GPUs, MXFP8."""
    return _nemodiag_v0_gb300_config(precision="fp8_mx", global_batch_size=4608)


def nemodiag_v0_pretrain_72gpu_gb300_nvfp4_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 72 GB300 GPUs, NVFP4."""
    return _nemodiag_v0_gb300_config(precision="nvfp4", global_batch_size=1152)


def nemodiag_v0_pretrain_144gpu_gb300_nvfp4_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 144 GB300 GPUs, NVFP4."""
    return _nemodiag_v0_gb300_config(precision="nvfp4", global_batch_size=2304)


def nemodiag_v0_pretrain_288gpu_gb300_nvfp4_perf72_e144_config() -> ConfigContainer:
    """NeMoDiag V0: 288 GB300 GPUs, NVFP4."""
    return _nemodiag_v0_gb300_config(precision="nvfp4", global_batch_size=4608)


__all__ = [
    "nemodiag_v0_pretrain_72gpu_gb300_bf16_perf72_e144_config",
    "nemodiag_v0_pretrain_72gpu_gb300_fp8mx_perf72_e144_config",
    "nemodiag_v0_pretrain_72gpu_gb300_nvfp4_perf72_e144_config",
    "nemodiag_v0_pretrain_144gpu_gb300_bf16_perf72_e144_config",
    "nemodiag_v0_pretrain_144gpu_gb300_fp8mx_perf72_e144_config",
    "nemodiag_v0_pretrain_144gpu_gb300_nvfp4_perf72_e144_config",
    "nemodiag_v0_pretrain_288gpu_gb300_bf16_perf72_e144_config",
    "nemodiag_v0_pretrain_288gpu_gb300_fp8mx_perf72_e144_config",
    "nemodiag_v0_pretrain_288gpu_gb300_nvfp4_perf72_e144_config",
]
