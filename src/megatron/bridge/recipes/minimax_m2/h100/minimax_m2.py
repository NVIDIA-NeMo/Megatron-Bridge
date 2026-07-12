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

"""H100 pretrain recipes for MiniMax M2 models."""

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_LIBRARY_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def minimax_m2_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiniMax-M2 H100 pretrain config."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(
        "MiniMaxAI/MiniMax-M2",
        revision="757303d492a50514c312788b5247a4f696a4c6a3",  # pragma: allowlist secret
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
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


def minimax_m2_5_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiniMax-M2.5 H100 pretrain config."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(
        "MiniMaxAI/MiniMax-M2.5",
        revision="f710177d938eff80b684d42c5aa84b382612f21f",  # pragma: allowlist secret
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
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


def minimax_m2_7_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiniMax-M2.7 H100 pretrain config."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(
        "MiniMaxAI/MiniMax-M2.7",
        revision="d494266a4affc0d2995ba1fa35c8481cbd84294b",  # pragma: allowlist secret
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 16
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


__all__ = [
    "minimax_m2_5_pretrain_16gpu_h100_bf16_config",
    "minimax_m2_7_pretrain_16gpu_h100_bf16_config",
    "minimax_m2_pretrain_16gpu_h100_bf16_config",
]
