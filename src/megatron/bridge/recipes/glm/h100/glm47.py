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

"""H100 pretrain recipes for text-only GLM-4.7 models."""

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def glm47_flash_31b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the GLM-4.7-Flash 31B H100 pretrain config."""
    cfg = build_text_pretrain_config(
        hf_model_id="zai-org/GLM-4.7-Flash",
        revision="7dd20894a642a0aa287e9827cb1a1f7f91386b67",  # pragma: allowlist secret
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=8,
        trust_remote_code=True,
    )
    # Without full recompute the first step leaves insufficient allocator
    # headroom even for the MoE metric all-reduce's sub-KiB NCCL buffers.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1
    cfg.model.recompute_modules = None
    return cfg


def glm47_355b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the GLM-4.7 355B H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="zai-org/GLM-4.7",
        revision="602d01efcdd332c5238ca4bcede555defbe83eb7",  # pragma: allowlist secret
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=16,
        trust_remote_code=True,
    )


__all__ = [
    "glm47_355b_pretrain_16gpu_h100_bf16_config",
    "glm47_flash_31b_pretrain_8gpu_h100_bf16_config",
]
