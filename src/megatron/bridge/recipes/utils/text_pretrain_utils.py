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

"""Shared construction for config-derived text-only H100 pretrain recipes."""

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.environment_utils import COMMON_LIBRARY_ENV_VARS
from megatron.bridge.training.config import ConfigContainer


def build_text_pretrain_config(
    *,
    hf_model_id: str,
    revision: str,
    tensor_parallelism: int,
    pipeline_parallelism: int,
    expert_parallelism: int = 1,
    expert_tensor_parallelism: int = 1,
    sequence_length: int = 4096,
    trust_remote_code: bool = False,
) -> ConfigContainer:
    """Build a BF16 H100 pretrain config from an immutable Hugging Face config.

    The Hugging Face wrapper is lazy: recipe construction materializes the
    configuration but never downloads or loads checkpoint weights. This keeps
    from-scratch pretraining recipes practical even for very large models.

    Args:
        hf_model_id: Hugging Face repository containing the model config.
        revision: Immutable Hugging Face commit SHA.
        tensor_parallelism: Tensor model parallel degree.
        pipeline_parallelism: Pipeline model parallel degree.
        expert_parallelism: Expert model parallel degree.
        expert_tensor_parallelism: Expert tensor parallel degree.
        sequence_length: Training sequence length.
        trust_remote_code: Whether the pinned repository's config code may run.

    Returns:
        Configured text-only pretraining container.
    """
    cfg = _pretrain_common()
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )
    cfg.model = bridge.to_megatron_provider(load_weights=False)

    cfg.model.tensor_model_parallel_size = tensor_parallelism
    cfg.model.pipeline_model_parallel_size = pipeline_parallelism
    cfg.model.virtual_pipeline_model_parallel_size = None
    cfg.model.context_parallel_size = 1
    cfg.model.expert_model_parallel_size = expert_parallelism
    cfg.model.expert_tensor_parallel_size = expert_tensor_parallelism
    cfg.model.sequence_parallel = tensor_parallelism > 1
    cfg.model.pipeline_dtype = torch.bfloat16 if pipeline_parallelism > 1 else None
    cfg.model.seq_length = sequence_length

    cfg.dataset.seq_length = sequence_length
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


__all__ = ["build_text_pretrain_config"]
