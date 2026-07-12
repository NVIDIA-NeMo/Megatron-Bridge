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

"""Additional text-only Gemma H100 pretrain recipes."""

import os

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_LIBRARY_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def gemma_2b_pretrain_1gpu_h100_bf16_config() -> ConfigContainer:
    """Return the original Gemma 2B H100 pretrain config."""
    cfg = _pretrain_common()
    cfg.model = AutoBridge.from_hf_pretrained(
        "google/gemma-2b",
        revision="9cf48e52b224239de00d483ec8eb84fb8d0f3a3a",  # pragma: allowlist secret
    ).to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = False
    cfg.model.pipeline_dtype = None
    cfg.model.seq_length = 4096

    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.num_workers = 1

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 100
    cfg.validation.eval_iters = 1
    cfg.logger.log_interval = 1
    cfg.checkpoint.save_interval = 100
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 1000

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]
    cfg.model.cuda_graph_impl = "none"

    cfg.env_vars = {
        **COMMON_LIBRARY_ENV_VARS,
    }
    return cfg


def gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the text-only Gemma 4 26B-A4B MoE H100 pretrain config."""
    cfg = _pretrain_common()

    previous_mode = os.environ.get("GEMMA4_CONVERSION_MODE")
    os.environ["GEMMA4_CONVERSION_MODE"] = "text"
    try:
        cfg.model = AutoBridge.from_hf_pretrained(
            "google/gemma-4-26B-A4B",
            revision="6b556d30bb65a6ee0bdaec99bab0afc7bf1494fb",  # pragma: allowlist secret
        ).to_megatron_provider(load_weights=False)
    finally:
        if previous_mode is None:
            os.environ.pop("GEMMA4_CONVERSION_MODE", None)
        else:
            os.environ["GEMMA4_CONVERSION_MODE"] = previous_mode

    # TP=2 leaves headroom for the custom fp32 RMSNorm temporaries.
    cfg.model.tensor_model_parallel_size = 2
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.pipeline_dtype = None
    cfg.model.seq_length = 4096

    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.num_workers = 1

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 100
    cfg.validation.eval_iters = 1
    cfg.logger.log_interval = 1
    cfg.checkpoint.save_interval = 100
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 1000

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]
    cfg.model.cuda_graph_impl = "none"

    cfg.env_vars = {
        **COMMON_LIBRARY_ENV_VARS,
    }
    return cfg


def gemma4_31b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the text-only Gemma 4 31B dense H100 pretrain config."""
    cfg = _pretrain_common()

    previous_mode = os.environ.get("GEMMA4_CONVERSION_MODE")
    os.environ["GEMMA4_CONVERSION_MODE"] = "text"
    try:
        cfg.model = AutoBridge.from_hf_pretrained(
            "google/gemma-4-31B",
            revision="d77cb0be8ad40327cc1c6b70eff4b3f0be35bee3",  # pragma: allowlist secret
        ).to_megatron_provider(load_weights=False)
    finally:
        if previous_mode is None:
            os.environ.pop("GEMMA4_CONVERSION_MODE", None)
        else:
            os.environ["GEMMA4_CONVERSION_MODE"] = previous_mode

    # Gemma4DenseProvider does not support pipeline parallelism.
    cfg.model.tensor_model_parallel_size = 8
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 1
    cfg.model.expert_tensor_parallel_size = 1
    cfg.model.sequence_parallel = True
    cfg.model.pipeline_dtype = None
    cfg.model.seq_length = 4096

    cfg.dataset.seq_length = 4096
    cfg.dataset.blend = None
    cfg.dataset.blend_per_split = None
    cfg.dataset.num_workers = 1

    cfg.train.train_iters = 1000
    cfg.train.global_batch_size = 128
    cfg.train.micro_batch_size = 1
    cfg.validation.eval_interval = 100
    cfg.validation.eval_iters = 1
    cfg.logger.log_interval = 1
    cfg.checkpoint.save_interval = 100
    cfg.scheduler.lr_warmup_iters = 10
    cfg.scheduler.lr_decay_iters = 1000

    cfg.model.recompute_granularity = "selective"
    cfg.model.recompute_modules = ["core_attn"]
    cfg.model.cuda_graph_impl = "none"

    cfg.env_vars = {
        **COMMON_LIBRARY_ENV_VARS,
    }
    return cfg


__all__ = [
    "gemma4_26b_a4b_pretrain_8gpu_h100_bf16_config",
    "gemma4_31b_pretrain_8gpu_h100_bf16_config",
    "gemma_2b_pretrain_1gpu_h100_bf16_config",
]
