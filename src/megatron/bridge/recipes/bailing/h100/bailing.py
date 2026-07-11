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

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def ling_mini_16b_pretrain_8gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Ling Mini 16B H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="inclusionAI/Ling-mini-2.0",
        revision="920c3fd9916e3d5e543fc4f609e827cad8a32983",  # pragma: allowlist secret
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=8,
        trust_remote_code=True,
    )


def ling_flash_100b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the Ling Flash 100B H100 pretrain config."""
    return build_text_pretrain_config(
        hf_model_id="inclusionAI/Ling-flash-2.0",
        revision="18ca64a019b553be57bab50af3207fb2f3675edc",  # pragma: allowlist secret
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=16,
        trust_remote_code=True,
    )


__all__ = [
    "ling_flash_100b_pretrain_16gpu_h100_bf16_config",
    "ling_mini_16b_pretrain_8gpu_h100_bf16_config",
]
