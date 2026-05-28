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

"""Parallelism presets for Llama 2 70B LoRA (MLPerf v6.0 workload).

Naming: {MODEL}_{SIZE}_{TASK}_CONFIG_{GPU}_{PRECISION}_{VERSION}; MLPERF_V{1,2,3} variants bake in the canonical reference shape per scale bucket.
"""

from dataclasses import replace

from utils.utils import WorkloadBaseConfig


BASE_LLAMA2_70B_LORA_CONFIG = WorkloadBaseConfig(
    num_gpus=8,
    global_batch_size=8,
    micro_batch_size=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
)


# Llama2-70B LoRA presets — V1 baseline (MLPerf v6.0 reference: GovReport packed .npy, seq_length=8192).

LLAMA2_70B_LORA_CONFIG_GB200_BF16_V1 = replace(BASE_LLAMA2_70B_LORA_CONFIG)
LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_V1 = replace(BASE_LLAMA2_70B_LORA_CONFIG)
LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_V1 = replace(
    BASE_LLAMA2_70B_LORA_CONFIG, context_parallel_size=2, micro_batch_size=2
)

LLAMA2_70B_LORA_CONFIG_GB300_BF16_V1 = replace(BASE_LLAMA2_70B_LORA_CONFIG)
LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_V1 = replace(BASE_LLAMA2_70B_LORA_CONFIG)
LLAMA2_70B_LORA_CONFIG_GB300_NVFP4_V1 = replace(
    BASE_LLAMA2_70B_LORA_CONFIG, context_parallel_size=2, micro_batch_size=2
)


# Llama2-70B LoRA presets — MLPERF (MLPerf v6.0 reference parity, GB200/GB300 only; scale buckets V1=8/V2=72/V3=512 GPU).
# All variants pin cuda_graph_impl=local/full_iteration; recipe-level parity knobs applied by set_llama2_mlperf_parity_overrides() in llama2_llm_finetune.py.

# ---- MLPERF_V1: 8 GPU ----
LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V1 = replace(
    BASE_LLAMA2_70B_LORA_CONFIG,
    num_gpus=8,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=1,
    micro_batch_size=1,
    global_batch_size=8,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)
LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_MLPERF_V1 = LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V1

LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_MLPERF_V1 = replace(
    BASE_LLAMA2_70B_LORA_CONFIG,
    num_gpus=8,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=2,
    micro_batch_size=2,
    global_batch_size=8,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)
LLAMA2_70B_LORA_CONFIG_GB300_NVFP4_MLPERF_V1 = LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_MLPERF_V1


# ---- MLPERF_V2: 72 GPU (FP8 only in MLPerf v6.0 reference at this scale) ----
LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V2 = replace(
    BASE_LLAMA2_70B_LORA_CONFIG,
    num_gpus=72,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=8,
    micro_batch_size=1,
    global_batch_size=9,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)
LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_MLPERF_V2 = LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V2


# ---- MLPERF_V3: 512 GPU (FP8 only in MLPerf v6.0 reference at this scale) ----
LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V3 = replace(
    BASE_LLAMA2_70B_LORA_CONFIG,
    num_gpus=512,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    context_parallel_size=8,
    micro_batch_size=1,
    global_batch_size=64,
    cuda_graph_impl="local",
    cuda_graph_scope="full_iteration",
)
LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_MLPERF_V3 = LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V3


__all__ = [
    # V1 baseline
    "LLAMA2_70B_LORA_CONFIG_GB200_BF16_V1",
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_V1",
    "LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_V1",
    "LLAMA2_70B_LORA_CONFIG_GB300_BF16_V1",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_V1",
    "LLAMA2_70B_LORA_CONFIG_GB300_NVFP4_V1",
    # MLPERF V1 (8 GPU)
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V1",
    "LLAMA2_70B_LORA_CONFIG_GB200_NVFP4_MLPERF_V1",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_MLPERF_V1",
    "LLAMA2_70B_LORA_CONFIG_GB300_NVFP4_MLPERF_V1",
    # MLPERF V2 (72 GPU, FP8 only)
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V2",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_MLPERF_V2",
    # MLPERF V3 (512 GPU, FP8 only)
    "LLAMA2_70B_LORA_CONFIG_GB200_FP8_CS_MLPERF_V3",
    "LLAMA2_70B_LORA_CONFIG_GB300_FP8_CS_MLPERF_V3",
]
