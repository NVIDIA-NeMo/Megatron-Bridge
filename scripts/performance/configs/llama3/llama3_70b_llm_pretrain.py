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
    get_precision_config,
    set_cuda_graph_overrides,
    set_megatron_fsdp_overrides,
    set_recompute_overrides,
)

from megatron.bridge.recipes.llama import llama3_70b_pretrain_config
from megatron.bridge.training.comm_overlap import (
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
)
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def llama3_70b_gb200_64gpus_bf16_config() -> ConfigContainer:
    """GB200, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

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

    if fp8_recipe == "cs":
        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.context_parallel_size = 1
        cfg.model.virtual_pipeline_model_parallel_size = None

        cfg.train.micro_batch_size = 2

        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, cpu_offloading_num_layers=40)

    if fp8_recipe == "mx":
        cfg.model.tensor_model_parallel_size = 2
        cfg.model.pipeline_model_parallel_size = 4
        cfg.model.context_parallel_size = 1
        cfg.model.virtual_pipeline_model_parallel_size = 5

        cfg.train.micro_batch_size = 1

    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_b200_64gpus_bf16_config() -> ConfigContainer:
    """B200, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_b200_64gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 64xGPU, FP8 cs preset."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    if fp8_recipe == "cs":
        cfg.model.tensor_model_parallel_size = 1
        cfg.model.pipeline_model_parallel_size = 1
        cfg.model.context_parallel_size = 1
        cfg.model.virtual_pipeline_model_parallel_size = None

        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.fsdp_double_buffer = True
        cfg.model.gradient_accumulation_fusion = False
        cfg.ddp.suggested_communication_unit_size = 800000000
        set_recompute_overrides(cfg, recompute_num_layers=5)

    if fp8_recipe == "mx":
        cfg.model.tensor_model_parallel_size = 2
        cfg.model.pipeline_model_parallel_size = 4
        cfg.model.context_parallel_size = 1
        cfg.model.virtual_pipeline_model_parallel_size = 5

    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192

    return cfg


def llama3_70b_h100_64gpus_bf16_config() -> ConfigContainer:
    """H100, 64xGPU, BF16 baseline config."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192

    return cfg


def llama3_70b_h100_64gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 64xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_70b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 5
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.comm_overlap.tp_comm_overlap_cfg = userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192

    return cfg
