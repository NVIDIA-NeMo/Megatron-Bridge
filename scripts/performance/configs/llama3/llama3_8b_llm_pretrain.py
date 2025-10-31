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
)

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer


logger = logging.getLogger(__name__)


def llama3_8b_gb200_8gpus_bf16_config() -> ConfigContainer:
    """GB200, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_gb200_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """GB200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cg_impl_map = {"cs": "none", "mx": "local"}
    cuda_graph_impl = cg_impl_map.get(fp8_recipe, "none")
    set_cuda_graph_overrides(cfg, cuda_graph_impl=cuda_graph_impl, cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_b200_8gpus_bf16_config(**kwargs) -> ConfigContainer:
    """B200, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_b200_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """B200, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 2
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    set_cuda_graph_overrides(cfg, cuda_graph_impl="local", cuda_graph_scope="full_iteration")

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_h100_8gpus_bf16_config(**kwargs) -> ConfigContainer:
    """H100, 8xGPU, BF16 baseline config."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("bf16"))

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 2
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = None
    cfg.model.sequence_parallel = bool(cfg.model.tensor_model_parallel_size > 1)

    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.model.seq_length = 8192
    cfg.dataset.sequence_length = 8192

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg


def llama3_8b_h100_8gpus_fp8_config(fp8_recipe: str = "cs") -> ConfigContainer:
    """H100, 8xGPU, FP8 preset with selectable recipe (ds/cs/mx/ss)."""
    cfg = llama3_8b_pretrain_config(mock=True, precision_config=get_precision_config("fp8", fp8_recipe))

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

    if fp8_recipe == "cs":
        set_megatron_fsdp_overrides(cfg)
        cfg.ddp.nccl_ub = True
        cfg.model.gradient_accumulation_fusion = False

    cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=bool(cfg.model.tensor_model_parallel_size > 1))
    cfg.tokenizer.vocab_size = 128256
    cfg.model.should_pad_vocab = True

    return cfg
