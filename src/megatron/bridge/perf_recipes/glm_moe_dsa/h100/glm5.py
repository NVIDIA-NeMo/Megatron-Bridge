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
"""H100 performance recipes for GLM-5.1 and GLM-5.2 SFT."""

from megatron.bridge.perf_recipes.glm_moe_dsa.common import (
    ConfigContainer,
    _glm5_cudnn_sft_base,
)


_GLM5_H100_TP = 4
_GLM5_H100_PP = 26
_GLM5_H100_CP = 4
_GLM5_H100_EP = 8
_GLM5_H100_GBS = 520


def glm51_sft_416gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.1 SFT: 416x H100, BF16, 128K packed THD, CP=4, cuDNN DSA."""
    return _glm5_cudnn_sft_base(
        "zai-org/GLM-5.1",
        tensor_model_parallel_size=_GLM5_H100_TP,
        pipeline_model_parallel_size=_GLM5_H100_PP,
        context_parallel_size=_GLM5_H100_CP,
        expert_model_parallel_size=_GLM5_H100_EP,
        global_batch_size=_GLM5_H100_GBS,
        sequence_parallel=True,
    )


def glm52_sft_416gpu_h100_bf16_config() -> ConfigContainer:
    """GLM-5.2 SFT: 416x H100, BF16, 128K packed THD, CP=4, cuDNN DSA."""
    return _glm5_cudnn_sft_base(
        "zai-org/GLM-5.2",
        tensor_model_parallel_size=_GLM5_H100_TP,
        pipeline_model_parallel_size=_GLM5_H100_PP,
        context_parallel_size=_GLM5_H100_CP,
        expert_model_parallel_size=_GLM5_H100_EP,
        global_batch_size=_GLM5_H100_GBS,
        sequence_parallel=True,
    )
