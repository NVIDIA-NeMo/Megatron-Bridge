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

"""wan benchmark overrides applied to library recipes.

The literal names preserve existing CLI selectors. Common and model overrides
retain their ordering because graph, RNG and precision policies are dependent.
"""

from megatron.bridge.diffusion.recipes.wan.wan import wan_14b_pretrain_config
from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.perf_benchmark_overrides._common import _benchmark_common, _perf_precision
from megatron.bridge.training.perf_benchmark_overrides.environment import COMMON_PERF_ENV_VARS


BENCHMARK_ENVIRONMENTS = {
    "wan_14b_pretrain_16gpu_gb200_bf16": {
        **COMMON_PERF_ENV_VARS,
        "CUDA_DEVICE_MAX_CONNECTIONS": 32,
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        "NCCL_NVLS_ENABLE": 0,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
        "NVTE_NORM_FWD_USE_CUDNN": 1,
        "NVTE_NORM_BWD_USE_CUDNN": 1,
        "NVTE_FUSED_ATTN": 1,
    }
}
LIBRARY_RECIPES = {
    "wan_14b_pretrain_16gpu_gb200_bf16_config": wan_14b_pretrain_config,
    "wan_14b_pretrain_16gpu_gb300_bf16_config": wan_14b_pretrain_config,
    "wan_14b_pretrain_32gpu_h100_bf16_config": wan_14b_pretrain_config,
}
ALLOCATION_AWARE = ()


def apply_overrides(cfg: ConfigContainer, name: str) -> ConfigContainer:
    """Apply model-specific benchmark overrides to an existing library config."""
    match name:
        case "wan_14b_pretrain_16gpu_gb200_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.dataset.global_batch_size = 64
            cfg.dataset.micro_batch_size = 1
            cfg.model.recompute_granularity = "full"
            cfg.model.recompute_method = "block"
            cfg.model.recompute_num_layers = 1
            cfg.ddp.use_megatron_fsdp = True
            cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
            cfg.ddp.keep_fp8_transpose_cache = False
            cfg.ddp.average_in_collective = False
            cfg.model.init_model_with_meta_device = True
            cfg.model.gradient_accumulation_fusion = True
            cfg.checkpoint.ckpt_format = "fsdp_dtensor"
            cfg.optimizer.lr = 5e-06
            cfg.optimizer.min_lr = 5e-06
            _benchmark_common(cfg)
            cfg.logger.log_throughput = True
            cfg.optimizer.use_precision_aware_optimizer = False
            cfg.scheduler.lr_decay_style = "constant"
            cfg.scheduler.lr_warmup_iters = 0
            cfg.model.cuda_graph_scope = []
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["wan_14b_pretrain_16gpu_gb200_bf16"].copy()
            return cfg
        case "wan_14b_pretrain_16gpu_gb300_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 1
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 2
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = False
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.dataset.global_batch_size = 64
            cfg.dataset.micro_batch_size = 1
            cfg.model.recompute_granularity = None
            cfg.model.recompute_method = None
            cfg.model.recompute_num_layers = None
            cfg.optimizer.lr = 5e-06
            cfg.optimizer.min_lr = 5e-06
            _benchmark_common(cfg)
            cfg.logger.log_throughput = True
            cfg.optimizer.use_precision_aware_optimizer = False
            cfg.scheduler.lr_decay_style = "constant"
            cfg.scheduler.lr_warmup_iters = 0
            cfg.model.cuda_graph_scope = []
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.env_vars = BENCHMARK_ENVIRONMENTS["wan_14b_pretrain_16gpu_gb200_bf16"].copy()
            return cfg
        case "wan_14b_pretrain_32gpu_h100_bf16_config":
            cfg.mixed_precision = _perf_precision("bf16")
            cfg.model.tensor_model_parallel_size = 2
            cfg.model.pipeline_model_parallel_size = 1
            cfg.model.context_parallel_size = 4
            cfg.model.virtual_pipeline_model_parallel_size = None
            cfg.model.sequence_parallel = True
            cfg.train.global_batch_size = 64
            cfg.train.micro_batch_size = 1
            cfg.dataset.global_batch_size = 64
            cfg.dataset.micro_batch_size = 1
            cfg.train.empty_unused_memory_level = 0
            cfg.model.recompute_granularity = "full"
            cfg.model.recompute_method = "block"
            cfg.model.recompute_num_layers = 8
            cfg.optimizer.lr = 5e-06
            cfg.optimizer.min_lr = 5e-06
            _benchmark_common(cfg)
            cfg.logger.log_throughput = True
            cfg.optimizer.use_precision_aware_optimizer = False
            cfg.scheduler.lr_decay_style = "constant"
            cfg.scheduler.lr_warmup_iters = 0
            cfg.model.cuda_graph_scope = []
            cfg.model.moe_token_dispatcher_type = "alltoall"
            cfg.env_vars = {
                **COMMON_PERF_ENV_VARS,
                "CUDA_DEVICE_MAX_CONNECTIONS": 1,
                "NCCL_GRAPH_REGISTER": 0,
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
                "NCCL_NVLS_ENABLE": 0,
                "NVTE_BWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_FWD_LAYERNORM_SM_MARGIN": 16,
                "NVTE_NORM_FWD_USE_CUDNN": 1,
                "NVTE_NORM_BWD_USE_CUDNN": 1,
                "NVTE_FUSED_ATTN": 1,
            }
            return cfg
        case _:
            raise ValueError(f"Unknown benchmark overrides: {name}")
