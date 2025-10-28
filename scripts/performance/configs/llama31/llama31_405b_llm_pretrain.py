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

from megatron.bridge.recipes.llama import llama31_405b_pretrain_config
from megatron.bridge.training.config import ConfigContainer

try:
    from utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides, set_cuda_graph_overrides, set_recompute_overrides
except (ImportError, ModuleNotFoundError):
    from ..utils.helpers import get_precision_config, set_megatron_fsdp_overrides, set_basic_perf_overrides, set_cuda_graph_overrides, set_recompute_overrides

logger = logging.getLogger(__name__)


def llama31_405b_h100_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """H100, 8xGPU, BF16 baseline config."""
    cfg = llama31_405b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg


def llama31_405b_h100_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama31_405b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 512
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg

def llama31_405b_b200_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """B200, 8xGPU, BF16 baseline config."""
    cfg = llama31_405b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg

def llama31_405b_b200_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 8xGPU, FP8 cs preset."""
    cfg = llama31_405b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg

def llama31_405b_gb200_bf16_config(fp8_recipe = None) -> ConfigContainer:
    """GB200, 4xGPU, BF16 baseline config."""
    cfg = llama31_405b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    set_basic_perf_overrides(cfg)

    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 8
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = 8
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg

def llama31_405b_gb200_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 4xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama31_405b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    set_basic_perf_overrides(cfg)

    if fp8_recipe == "cs":
      cfg.model.tensor_model_parallel_size = 2
      cfg.model.pipeline_model_parallel_size = 1
      cfg.model.context_parallel_size = 1
      cfg.model.virtual_pipeline_model_parallel_size = None

      set_megatron_fsdp_overrides(cfg, perf_overrides={"use_megatron_fsdp": True})
      cfg.ddp.fsdp_double_buffer = True
      set_recompute_overrides(cfg, perf_overrides={"cpu_offloading_num_layers": 95})

    if fp8_recipe == "mx":
      cfg.model.tensor_model_parallel_size = 4
      cfg.model.pipeline_model_parallel_size = 8
      cfg.model.context_parallel_size = 2
      cfg.model.virtual_pipeline_model_parallel_size = 8
 
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.mixed_precision.grad_reduce_in_fp32 = False
    cfg.ddp.grad_reduce_in_fp32 = False

    return cfg
