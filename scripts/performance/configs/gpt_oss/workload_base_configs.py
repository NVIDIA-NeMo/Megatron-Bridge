# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Parallelism presets for GPT performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_GPT_OSS_120B_CONFIG = WorkloadBaseConfig(
    num_gpus=64,
    expert_model_parallel_size=8,
    expert_tensor_parallel_size=1,
    micro_batch_size=8,
    global_batch_size=512,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)


# GPT-OSS 120B presets ---------------------------------------------------------

GPT_OSS_120B_GB300_BF16_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_GB300_FP8_MX_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_GB200_BF16_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_GB200_FP8_MX_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_B200_BF16_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_B200_FP8_MX_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_H100_BF16_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


GPT_OSS_120B_H100_FP8_CS_BASE_CONFIG = replace(
    BASE_GPT_OSS_120B_CONFIG,
)


__all__ = [
    "GPT_OSS_120B_GB300_BF16_BASE_CONFIG",
    "GPT_OSS_120B_GB300_FP8_MX_BASE_CONFIG",
    "GPT_OSS_120B_GB200_BF16_BASE_CONFIG",
    "GPT_OSS_120B_GB200_FP8_MX_BASE_CONFIG",
    "GPT_OSS_120B_B200_BF16_BASE_CONFIG",
    "GPT_OSS_120B_B200_FP8_MX_BASE_CONFIG",
    "GPT_OSS_120B_H100_BF16_BASE_CONFIG",
    "GPT_OSS_120B_H100_FP8_CS_BASE_CONFIG",
]
