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

"""Parallelism presets for Nemotron performance configs."""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_NEMOTRONH_56B_CONFIG = WorkloadBaseConfig(
    num_gpus=64,
    global_batch_size=192,
    cuda_graph_impl="transformer_engine",
)


# NemotronH 56B presets ---------------------------------------------------------

NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=2,
    cuda_graph_scope=["mamba", "attn"],
)


NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS = replace(
    BASE_NEMOTRONH_56B_CONFIG,
    tensor_model_parallel_size=8,
    cuda_graph_scope=["mamba"],
)

__all__ = [
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB300_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_GB200_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B300_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_B200_FP8_CS",
    "NEMOTRONH_56B_PRETRAIN_CONFIG_H100_FP8_CS",
]
