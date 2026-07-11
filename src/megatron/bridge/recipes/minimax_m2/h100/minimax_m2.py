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

from megatron.bridge.recipes.utils.text_pretrain_utils import build_text_pretrain_config
from megatron.bridge.training.config import ConfigContainer


def _minimax_m2_config(hf_model_id: str, revision: str) -> ConfigContainer:
    return build_text_pretrain_config(
        hf_model_id=hf_model_id,
        revision=revision,
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=16,
        trust_remote_code=True,
    )


def minimax_m2_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiniMax-M2 H100 pretrain config."""
    return _minimax_m2_config("MiniMaxAI/MiniMax-M2", "757303d492a50514c312788b5247a4f696a4c6a3")


def minimax_m2_5_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiniMax-M2.5 H100 pretrain config."""
    return _minimax_m2_config("MiniMaxAI/MiniMax-M2.5", "f710177d938eff80b684d42c5aa84b382612f21f")


def minimax_m2_7_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiniMax-M2.7 H100 pretrain config."""
    return _minimax_m2_config("MiniMaxAI/MiniMax-M2.7", "d494266a4affc0d2995ba1fa35c8481cbd84294b")


__all__ = [
    "minimax_m2_5_pretrain_16gpu_h100_bf16_config",
    "minimax_m2_7_pretrain_16gpu_h100_bf16_config",
    "minimax_m2_pretrain_16gpu_h100_bf16_config",
]
