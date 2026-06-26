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
"""GB200 performance recipes for GLM-5.1 and GLM-5.2 SFT."""

from megatron.bridge.perf_recipes.glm_moe_dsa.common import (
    ConfigContainer,
    _glm5_cudnn_sft_base,
)


_GLM5_GB200_CP = 32


def _glm5_gb200_cudnn_sft_config(model_id: str) -> ConfigContainer:
    """Return the 48-node GB200 GLM5 cuDNN SFT benchmark shape."""
    return _glm5_cudnn_sft_base(
        model_id,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=6,
        context_parallel_size=_GLM5_GB200_CP,
        expert_model_parallel_size=32,
        global_batch_size=56,
        sequence_parallel=False,
        num_layers_in_first_pipeline_stage=14,
        num_layers_in_last_pipeline_stage=16,
    )


def glm51_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.1 SFT: 192× GB200, BF16, 128K packed THD, CP=32, cuDNN DSA."""
    return _glm5_gb200_cudnn_sft_config("zai-org/GLM-5.1")


def glm52_sft_192gpu_gb200_bf16_config() -> ConfigContainer:
    """GLM-5.2 SFT: 192× GB200, BF16, 128K packed THD, CP=32, cuDNN DSA."""
    return _glm5_gb200_cudnn_sft_config("zai-org/GLM-5.2")
