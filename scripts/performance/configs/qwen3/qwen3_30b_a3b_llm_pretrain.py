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

from megatron.bridge.recipes.qwen3.qwen3_moe import qwen3_30b_a3b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.utils.moe_token_drop import apply_moe_token_drop

try:
    from utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides, set_cuda_graph_overrides, set_recompute_overrides
except (ImportError, ModuleNotFoundError):
    from ..utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides, set_cuda_graph_overrides, set_recompute_overrides

logger = logging.getLogger(__name__)


def set_qwen3_specific_overrides(cfg: ConfigContainer) -> ConfigContainer:
    cfg.model.cross_entropy_fusion_impl = "te"
    cfg.model.bias_activation_fusion = True
    cfg.model.recompute_granularity = None
    cfg.model.recompute_method = None
    cfg.model.recompute_num_layers = None
    cfg.model.moe_router_fusion = True

    cfg.model = apply_moe_token_drop(cfg.model)


def qwen3_30b_a3b_h100_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """H100, 8xGPU, BF16 baseline config."""
    cfg = qwen3_30b_a3b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("bf16"),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_specific_overrides(cfg)

    return cfg


def qwen3_30b_a3b_h100_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = qwen3_30b_a3b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("fp8", fp8_recipe),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 2
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = 12
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_qwen3_specific_overrides(cfg)

    return cfg

def qwen3_30b_a3b_b200_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """B200, 8xGPU, BF16 baseline config."""
    cfg = qwen3_30b_a3b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("bf16"),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})
    set_qwen3_specific_overrides(cfg)

    return cfg

def qwen3_30b_a3b_b200_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 8xGPU, FP8 cs preset."""
    cfg = qwen3_30b_a3b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("fp8", fp8_recipe),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})
    set_qwen3_specific_overrides(cfg)

    return cfg

def qwen3_30b_a3b_gb200_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """GB200, 4xGPU, BF16 baseline config."""
    cfg = qwen3_30b_a3b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("bf16"),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})
    set_qwen3_specific_overrides(cfg)

    return cfg

def qwen3_30b_a3b_gb200_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 4xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = qwen3_30b_a3b_pretrain_config(
      mock=True, 
      precision_config=get_precision_config("fp8", fp8_recipe),
      comm_overlap_config=CommOverlapConfig(tp_comm_overlap=True),
    )

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 4
    cfg.model.seq_length = 4096
    cfg.dataset.sequence_length = 4096

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    set_cuda_graph_overrides(cfg, perf_overrides={"cuda_graphs": True})
    set_qwen3_specific_overrides(cfg)

    return cfg
