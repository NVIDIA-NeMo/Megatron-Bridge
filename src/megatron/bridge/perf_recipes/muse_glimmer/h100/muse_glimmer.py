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

"""H100 performance recipes for Muse Glimmer."""

from megatron.bridge.perf_recipes._common import _benchmark_common
from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.recipes.muse_glimmer.h100 import (
    muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config,
)
from megatron.bridge.training.comm_overlap import CommOverlapConfig
from megatron.bridge.training.config import ConfigContainer, MockGPTDatasetConfig


def muse_glimmer_30b_pretrain_32gpu_h100_bf16_config() -> ConfigContainer:
    """Muse Glimmer dense decoder pretrain: 32× H100, BF16, TP=4 PP=4 CP=2."""
    cfg = muse_glimmer_30b_pretrain_32gpu_h100_bf16_multimodal_config()
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.context_parallel_size = 2
    cfg.model.cp_comm_type = "all_gather"
    cfg.model.hybrid_layer_pattern = "|".join(["*" * 13] * 4)
    cfg.model.freeze_vision_model = True
    cfg.model.freeze_vision_projection = True
    cfg.model.recompute_vision_layers = False
    cfg.dataset = MockGPTDatasetConfig(
        seq_length=4096,
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        num_dataset_builder_threads=1,
        split="9999,8,2",
        data_sharding=True,
        dataloader_type="single",
        skip_getting_attention_mask_from_dataset=True,
    )
    cfg.train.global_batch_size = 96
    cfg.train.micro_batch_size = 3
    cfg.ddp.overlap_grad_reduce = True
    cfg.ddp.overlap_param_gather = True
    cfg.comm_overlap = CommOverlapConfig(tp_comm_bootstrap_backend="nccl", tp_comm_overlap=True)

    _benchmark_common(cfg, cross_entropy_impl="native")
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # Transformer Engine requires this stream ordering for TP overlap on H100.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA allocator and graph-registration settings for the measured workload.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # This recipe does not use NCCL user buffers.
        "NCCL_NVLS_ENABLE": 0,
    }
    return cfg


__all__ = ["muse_glimmer_30b_pretrain_32gpu_h100_bf16_config"]
