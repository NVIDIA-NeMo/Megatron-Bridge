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
"""GB300 performance recipes for Llama 2."""

from megatron.bridge.perf_recipes.environment import COMMON_PERF_ENV_VARS
from megatron.bridge.perf_recipes.llama.common import (
    CommOverlapConfig,
    ConfigContainer,
    _llama2_benchmark_common,
    _llama2_70b_precision_config,
    llama2_70b_peft_config,
)


def llama2_70b_peft_4gpu_gb300_fp8ds_config() -> ConfigContainer:
    """Llama 2 70B LoRA: 4× GB300, FP8 DS, GBS=8, seq_length=2048."""
    cfg = llama2_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
    _llama2_benchmark_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.cuda_graph_warmup_steps = 1
    cfg.validation.eval_global_batch_size = 4
    cfg.validation.eval_interval = 48
    cfg.validation.eval_iters = 44
    cfg.validation.start_at_eval_iter = 192
    cfg.scheduler.lr_decay_iters = 800
    cfg.scheduler.lr_decay_steps = 6400
    cfg.scheduler.wd_incr_steps = 6400
    cfg.dataset.max_train_samples = 6432
    cfg.dataset.num_workers = 4
    cfg.dataset.seed = 10710
    cfg.rng.seed = 10710
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def llama2_70b_peft_8gpu_gb300_fp8ds_config() -> ConfigContainer:
    """Llama 2 70B LoRA: 8× GB300, FP8 DS, GBS=8, seq_length=2048."""
    cfg = llama2_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
    _llama2_benchmark_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 1
    cfg.train.global_batch_size = 8
    cfg.train.micro_batch_size = 1
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.model.cpu_offloading_num_layers = 11
    cfg.model.cpu_offloading = True
    cfg.validation.eval_global_batch_size = 8
    cfg.validation.eval_interval = 48
    cfg.validation.eval_iters = 22
    cfg.validation.start_at_eval_iter = 192
    cfg.scheduler.lr_decay_iters = 800
    cfg.scheduler.lr_decay_steps = 6400
    cfg.scheduler.wd_incr_steps = 6400
    cfg.dataset.max_train_samples = 6432
    cfg.dataset.num_workers = 4
    cfg.dataset.seed = 27208
    cfg.rng.seed = 27208
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def llama2_70b_peft_72gpu_gb300_fp8ds_config() -> ConfigContainer:
    """Llama 2 70B LoRA: 72× GB300, FP8 DS, GBS=8, seq_length=2048."""
    cfg = llama2_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
    _llama2_benchmark_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 8
    cfg.train.global_batch_size = 9
    cfg.train.micro_batch_size = 1
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.validation.eval_global_batch_size = 36
    cfg.validation.eval_interval = 43
    cfg.validation.eval_iters = 5
    cfg.validation.start_at_eval_iter = 172
    cfg.scheduler.lr_decay_iters = 800
    cfg.scheduler.lr_decay_steps = 7200
    cfg.scheduler.wd_incr_steps = 7200
    cfg.dataset.max_train_samples = 7236
    cfg.dataset.num_workers = 2
    cfg.dataset.seed = 14954
    cfg.rng.seed = 14954
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg


def llama2_70b_peft_512gpu_gb300_fp8ds_config() -> ConfigContainer:
    """Llama 2 70B LoRA: 512× GB300, FP8 DS, GBS=8, seq_length=2048."""
    cfg = llama2_70b_peft_config(peft_scheme="lora")
    cfg.mixed_precision = _llama2_70b_precision_config("fp8_ds")
    _llama2_benchmark_common(cfg)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.context_parallel_size = 8
    cfg.train.global_batch_size = 64
    cfg.train.micro_batch_size = 1
    cfg.model.cuda_graph_impl = "local"
    cfg.model.cuda_graph_scope = "full_iteration"
    cfg.optimizer.lr = 0.0006
    cfg.validation.eval_global_batch_size = 64
    cfg.validation.eval_interval = 6
    cfg.validation.eval_iters = 3
    cfg.validation.start_at_eval_iter = 66
    cfg.scheduler.lr_decay_iters = 600
    cfg.scheduler.lr_decay_steps = 38400
    cfg.scheduler.wd_incr_steps = 38400
    cfg.dataset.max_train_samples = 38592
    cfg.dataset.num_workers = 2
    cfg.dataset.seed = 8353
    cfg.rng.seed = 8353
    # Keep process settings next to the recipe so users can see the exact benchmark environment.
    cfg.env_vars = {
        **COMMON_PERF_ENV_VARS,
        # CUDA stream scheduling for this model and parallel layout.
        "CUDA_DEVICE_MAX_CONNECTIONS": 1,
        # CUDA graph and allocator behavior for this recipe.
        "NCCL_GRAPH_REGISTER": 0,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": 1,
        # NCCL user-buffer and launch settings.
        "NCCL_NVLS_ENABLE": 1,
        # Transformer Engine overlap settings for this model.
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
    }
    return cfg