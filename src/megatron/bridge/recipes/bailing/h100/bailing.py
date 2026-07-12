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

"""H100 pretrain recipes for text-only Bailing MoE V2 models."""

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_LIBRARY_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def ling_mini_16b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Ling Mini 16B H100 pretrain config."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(
        "inclusionAI/Ling-mini-2.0",
        revision="920c3fd9916e3d5e543fc4f609e827cad8a32983",  # pragma: allowlist secret
        trust_remote_code=True,
    ).to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = 8
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


def ling_flash_100b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Ling Flash 100B H100 pretrain config."""
    cfg = _pretrain_common()

    cfg.model = AutoBridge.from_hf_pretrained(
        "inclusionAI/Ling-flash-2.0",
        revision="18ca64a019b553be57bab50af3207fb2f3675edc",  # pragma: allowlist secret
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
    "ling_flash_100b_pretrain_16gpu_h100_bf16_config",
    "ling_mini_16b_pretrain_8gpu_h100_bf16_config",
]
