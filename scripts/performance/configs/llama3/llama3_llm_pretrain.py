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

import logging

from utils.helpers import (
    apply_parallelism_and_batch_config,
    get_precision_config,
    set_cuda_graph_overrides,
    set_megatron_fsdp_overrides,
    set_recompute_overrides,
)

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config, llama3_70b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    CommOverlapConfig,
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer

from . import parallelism_configs as parallelism_cfg


logger = logging.getLogger(__name__)


def set_llama3_common_configs(cfg: ConfigContainer) -> None:
    """Set common performance configurations for all Llama3 configs."""
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False


def llama3_70b_gb300_64gpus_bf16_config() -> ConfigContainer:
    """GB300, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB300_64GPUS_BF16_PARALLEL_CONFIG)

    set_megatron_fsdp_overrides(cfg)
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.ddp.suggested_communication_unit_size = 800000000
    set_recompute_overrides(cfg, cpu_offloading_num_layers=30)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_gb300_64gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB300, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    # use mx parallelism config by default
    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB300_64GPUS_FP8_MX_PARALLEL_CONFIG)

    if fp8_recipe == "cs":
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB300_64GPUS_FP8_CS_PARALLEL_CONFIG)
        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, cpu_offloading_num_layers=20)
    else:
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB300_64GPUS_FP8_MX_PARALLEL_CONFIG)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_gb200_64gpus_bf16_config() -> ConfigContainer:
    """GB200, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB200_64GPUS_BF16_PARALLEL_CONFIG)

    set_megatron_fsdp_overrides(cfg)
    cfg.ddp.fsdp_double_buffer = True
    cfg.model.gradient_accumulation_fusion = False
    cfg.ddp.suggested_communication_unit_size = 800000000
    set_recompute_overrides(cfg, cpu_offloading_num_layers=20)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_gb200_64gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    # use mx parallelism config by default
    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB200_64GPUS_FP8_MX_PARALLEL_CONFIG)

    if fp8_recipe == "cs":
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB200_64GPUS_FP8_CS_PARALLEL_CONFIG)
        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, cpu_offloading_num_layers=40)
    else:
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_GB200_64GPUS_FP8_MX_PARALLEL_CONFIG)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_b200_64gpus_bf16_config() -> ConfigContainer:
    """B200, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_B200_64GPUS_BF16_PARALLEL_CONFIG)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_b200_64gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 64xGPU, FP8 cs preset."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    # use mx parallelism config by default
    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_B200_64GPUS_FP8_MX_PARALLEL_CONFIG)

    if fp8_recipe == "cs":
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_B200_64GPUS_FP8_CS_PARALLEL_CONFIG)
        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, recompute_num_layers=5)
    else:
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_B200_64GPUS_FP8_MX_PARALLEL_CONFIG)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_h100_64gpus_bf16_config() -> ConfigContainer:
    """H100, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_H100_64GPUS_BF16_PARALLEL_CONFIG)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192

    return cfg


def llama3_70b_h100_64gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_70B_H100_64GPUS_FP8_CS_PARALLEL_CONFIG)

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192

    return cfg


def llama3_8b_gb300_8gpus_bf16_config() -> ConfigContainer:
    """GB300, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_GB300_8GPUS_BF16_PARALLEL_CONFIG)

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_gb300_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB300, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    if fp8_recipe == "cs":
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_GB300_8GPUS_FP8_CS_PARALLEL_CONFIG)
    else:
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_GB300_8GPUS_FP8_MX_PARALLEL_CONFIG)

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_gb200_8gpus_bf16_config() -> ConfigContainer:
    """GB200, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_GB200_8GPUS_BF16_PARALLEL_CONFIG)

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_gb200_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    if fp8_recipe == "cs":
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_GB200_8GPUS_FP8_CS_PARALLEL_CONFIG)
    else:
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_GB200_8GPUS_FP8_MX_PARALLEL_CONFIG)

    cg_impl_map = {"cs": "none", "mx": "local"}
    cuda_graph_impl = cg_impl_map.get(fp8_recipe, "none")
    set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_b200_8gpus_bf16_config() -> ConfigContainer:
    """B200, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_B200_8GPUS_BF16_PARALLEL_CONFIG)

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_b200_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    if fp8_recipe == "cs":
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_B200_8GPUS_FP8_CS_PARALLEL_CONFIG)
    else:
        apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_B200_8GPUS_FP8_MX_PARALLEL_CONFIG)

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_h100_8gpus_bf16_config() -> ConfigContainer:
    """H100, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_H100_8GPUS_BF16_PARALLEL_CONFIG)

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg


def llama3_8b_h100_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))
    set_llama3_common_configs(cfg)

    apply_parallelism_and_batch_config(cfg, parallelism_cfg.LLAMA3_8B_H100_8GPUS_FP8_CS_PARALLEL_CONFIG)

    if fp8_recipe == "cs":
        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))

    return cfg
