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

"""Import-free workload metadata needed by the performance launcher.

The training container still resolves and instantiates the flat recipes themselves.
This table only mirrors fields required to construct launcher-side experiment
names and display configuration variants without importing the training stack.
"""

from __future__ import annotations


# Recipes with identical launcher fields share one mapping to keep this checked-in
# sidecar compact. The login-node coverage test enforces one entry per flat recipe.
_WORKLOAD_CONFIG_GROUPS = (
    (
        {
            "num_gpus": 1024,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 16384,
        },
        (
            "deepseek_v3_pretrain_1024gpu_h100_bf16_config",
            "deepseek_v3_pretrain_1024gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 1024,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 16384,
        },
        ("deepseek_v3_pretrain_1024gpu_h100_fp8sc_config",),
    ),
    (
        {
            "num_gpus": 1024,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        ("deepseek_v3_pretrain_1024gpu_h100_fp8sc_large_scale_config",),
    ),
    (
        {
            "num_gpus": 128,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        ("deepseek_v3_pretrain_128gpu_vr200_bf16_config",),
    ),
    (
        {
            "num_gpus": 128,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
            "micro_batch_size": 2,
        },
        (
            "deepseek_v3_pretrain_128gpu_vr200_fp8cs_config",
            "deepseek_v3_pretrain_128gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 128,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        ("deepseek_v3_pretrain_128gpu_vr200_fp8mx_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 16,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
        },
        (
            "deepseek_v3_pretrain_256gpu_b200_bf16_config",
            "deepseek_v3_pretrain_256gpu_b200_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
        },
        (
            "deepseek_v3_pretrain_256gpu_b200_fp8mx_config",
            "deepseek_v3_pretrain_256gpu_b200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 16,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 256,
        },
        ("deepseek_v3_pretrain_256gpu_b200_fp8mx_large_scale_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
        },
        (
            "deepseek_v3_pretrain_256gpu_b300_bf16_config",
            "deepseek_v3_pretrain_256gpu_b300_fp8cs_config",
            "deepseek_v3_pretrain_256gpu_b300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 256,
        },
        ("deepseek_v3_pretrain_256gpu_b300_fp8mx_large_scale_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
            "micro_batch_size": 2,
        },
        ("deepseek_v3_pretrain_256gpu_b300_nvfp4_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
        },
        (
            "deepseek_v3_pretrain_256gpu_gb200_bf16_config",
            "deepseek_v3_pretrain_256gpu_gb200_fp8cs_config",
            "deepseek_v3_pretrain_256gpu_gb200_fp8mx_config",
            "deepseek_v3_pretrain_256gpu_gb200_nvfp4_config",
            "deepseek_v3_pretrain_256gpu_gb300_bf16_config",
            "deepseek_v3_pretrain_256gpu_vr200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 256,
        },
        (
            "deepseek_v3_pretrain_256gpu_gb200_fp8mx_large_scale_config",
            "deepseek_v3_pretrain_256gpu_gb300_fp8mx_large_scale_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
            "micro_batch_size": 2,
        },
        (
            "deepseek_v3_pretrain_256gpu_gb300_fp8cs_config",
            "deepseek_v3_pretrain_256gpu_gb300_nvfp4_config",
            "deepseek_v3_pretrain_256gpu_vr200_fp8cs_config",
            "deepseek_v3_pretrain_256gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
        },
        (
            "deepseek_v3_pretrain_256gpu_gb300_fp8mx_config",
            "deepseek_v3_pretrain_256gpu_vr200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 256,
            "micro_batch_size": 2,
        },
        (
            "deepseek_v3_pretrain_64gpu_gb300_bf16_fsdp_config",
            "deepseek_v3_pretrain_64gpu_gb300_fp8mx_fsdp_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "deepseek_v3_pretrain_64gpu_h100_bf16_config",
            "deepseek_v3_pretrain_64gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 192,
            "pipeline_model_parallel_size": 6,
            "context_parallel_size": 32,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 56,
        },
        (
            "glm51_sft_192gpu_gb200_bf16_config",
            "glm52_sft_192gpu_gb200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 416,
            "pipeline_model_parallel_size": 13,
            "context_parallel_size": 32,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 520,
        },
        (
            "glm51_sft_416gpu_h100_bf16_config",
            "glm52_sft_416gpu_h100_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1280,
            "micro_batch_size": 4,
        },
        (
            "gpt_oss_120b_pretrain_64gpu_b200_bf16_config",
            "gpt_oss_120b_pretrain_64gpu_b200_fp8mx_config",
            "gpt_oss_120b_pretrain_64gpu_b300_bf16_config",
            "gpt_oss_120b_pretrain_64gpu_b300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1280,
            "micro_batch_size": 4,
        },
        (
            "gpt_oss_120b_pretrain_64gpu_gb200_bf16_config",
            "gpt_oss_120b_pretrain_64gpu_gb200_fp8mx_config",
            "gpt_oss_120b_pretrain_64gpu_gb300_bf16_config",
            "gpt_oss_120b_pretrain_64gpu_vr200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1280,
            "micro_batch_size": 4,
        },
        (
            "gpt_oss_120b_pretrain_64gpu_gb300_fp8mx_config",
            "gpt_oss_120b_pretrain_64gpu_vr200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1280,
        },
        (
            "gpt_oss_120b_pretrain_64gpu_h100_bf16_config",
            "gpt_oss_120b_pretrain_64gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 512,
            "context_parallel_size": 2,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 256,
        },
        (
            "gpt_oss_20b_pretrain_512gpu_gb200_fp8mx_config",
            "gpt_oss_20b_pretrain_512gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "context_parallel_size": 2,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 32,
        },
        (
            "gpt_oss_20b_pretrain_64gpu_b300_fp8mx_config",
            "gpt_oss_20b_pretrain_64gpu_b300_nvfp4_config",
            "gpt_oss_20b_pretrain_64gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 72,
            "context_parallel_size": 2,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 36,
        },
        (
            "gpt_oss_20b_pretrain_72gpu_gb200_nvfp4_config",
            "gpt_oss_20b_pretrain_72gpu_gb300_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "context_parallel_size": 2,
            "expert_model_parallel_size": 4,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4,
        },
        (
            "gpt_oss_20b_pretrain_8gpu_b300_fp8mx_config",
            "gpt_oss_20b_pretrain_8gpu_b300_nvfp4_config",
            "gpt_oss_20b_pretrain_8gpu_gb200_nvfp4_config",
            "gpt_oss_20b_pretrain_8gpu_gb300_nvfp4_config",
            "gpt_oss_20b_pretrain_8gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 8, "expert_tensor_parallel_size": 1, "global_batch_size": 24, "micro_batch_size": 3},
        ("gpt_oss_20b_pretrain_8gpu_vr200_fp8mx_config",),
    ),
    (
        {
            "num_gpus": 1024,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 16,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
        },
        (
            "kimi_k2_pretrain_1024gpu_h100_bf16_config",
            "kimi_k2_pretrain_1024gpu_h100_fp8cs_config",
            "kimi_k2_pretrain_1024gpu_h100_fp8sc_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 16,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        (
            "kimi_k2_pretrain_256gpu_b200_bf16_config",
            "kimi_k2_pretrain_256gpu_b200_fp8cs_config",
            "kimi_k2_pretrain_256gpu_b200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 16,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
            "micro_batch_size": 2,
        },
        (
            "kimi_k2_pretrain_256gpu_b300_bf16_config",
            "kimi_k2_pretrain_256gpu_b300_fp8cs_config",
            "kimi_k2_pretrain_256gpu_b300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        (
            "kimi_k2_pretrain_256gpu_gb200_bf16_config",
            "kimi_k2_pretrain_256gpu_gb200_fp8cs_config",
            "kimi_k2_pretrain_256gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
            "micro_batch_size": 2,
        },
        (
            "kimi_k2_pretrain_256gpu_gb300_bf16_config",
            "kimi_k2_pretrain_256gpu_gb300_fp8cs_config",
            "kimi_k2_pretrain_256gpu_gb300_fp8mx_config",
            "kimi_k2_pretrain_256gpu_gb300_nvfp4_config",
            "kimi_k2_pretrain_256gpu_vr200_bf16_config",
            "kimi_k2_pretrain_256gpu_vr200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 1024,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 8,
            "context_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 8,
            "global_batch_size": 1536,
        },
        (
            "llama31_405b_pretrain_1024gpu_h100_bf16_config",
            "llama31_405b_pretrain_1024gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 128,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 16,
            "virtual_pipeline_model_parallel_size": 8,
            "global_batch_size": 768,
        },
        (
            "llama31_405b_pretrain_128gpu_b200_bf16_config",
            "llama31_405b_pretrain_128gpu_b200_fp8mx_config",
            "llama31_405b_pretrain_128gpu_b200_nvfp4_config",
            "llama31_405b_pretrain_128gpu_gb200_bf16_config",
            "llama31_405b_pretrain_128gpu_gb200_fp8mx_config",
            "llama31_405b_pretrain_128gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 128,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 16,
            "virtual_pipeline_model_parallel_size": 4,
            "global_batch_size": 768,
        },
        (
            "llama31_405b_pretrain_128gpu_b200_fp8cs_config",
            "llama31_405b_pretrain_128gpu_gb200_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 128, "tensor_model_parallel_size": 2, "global_batch_size": 768},
        (
            "llama31_405b_pretrain_128gpu_b300_bf16_config",
            "llama31_405b_pretrain_128gpu_gb300_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 128,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "global_batch_size": 768,
        },
        (
            "llama31_405b_pretrain_128gpu_b300_fp8cs_config",
            "llama31_405b_pretrain_128gpu_b300_nvfp4_config",
            "llama31_405b_pretrain_128gpu_gb300_fp8cs_config",
            "llama31_405b_pretrain_128gpu_gb300_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 128,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "context_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "global_batch_size": 768,
        },
        (
            "llama31_405b_pretrain_128gpu_b300_fp8mx_config",
            "llama31_405b_pretrain_128gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 16,
            "virtual_pipeline_model_parallel_size": 8,
            "global_batch_size": 1536,
        },
        (
            "llama31_405b_pretrain_256gpu_b200_nvfp4_config",
            "llama31_405b_pretrain_256gpu_gb200_bf16_config",
            "llama31_405b_pretrain_256gpu_gb200_fp8mx_config",
            "llama31_405b_pretrain_256gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 256, "tensor_model_parallel_size": 2, "global_batch_size": 1536},
        (
            "llama31_405b_pretrain_256gpu_b300_bf16_config",
            "llama31_405b_pretrain_256gpu_gb300_bf16_config",
            "llama31_405b_pretrain_256gpu_vr200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 16,
            "virtual_pipeline_model_parallel_size": 4,
            "global_batch_size": 1536,
        },
        ("llama31_405b_pretrain_256gpu_gb200_fp8cs_config",),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "global_batch_size": 1536,
        },
        (
            "llama31_405b_pretrain_256gpu_gb300_fp8cs_config",
            "llama31_405b_pretrain_256gpu_gb300_nvfp4_config",
            "llama31_405b_pretrain_256gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "context_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "global_batch_size": 1536,
        },
        (
            "llama31_405b_pretrain_256gpu_gb300_fp8mx_config",
            "llama31_405b_pretrain_256gpu_vr200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 512,
            "tensor_model_parallel_size": 8,
            "pipeline_model_parallel_size": 8,
            "context_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 8,
            "global_batch_size": 768,
        },
        (
            "llama31_405b_pretrain_512gpu_h100_bf16_config",
            "llama31_405b_pretrain_512gpu_h100_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 8, "pipeline_model_parallel_size": 2, "global_batch_size": 32},
        (
            "llama3_70b_peft_8gpu_b200_bf16_config",
            "llama3_70b_peft_8gpu_b200_fp8mx_config",
            "llama3_70b_peft_8gpu_b300_fp8cs_config",
            "llama3_70b_peft_8gpu_b300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 20,
            "global_batch_size": 32,
        },
        (
            "llama3_70b_peft_8gpu_b200_fp8cs_config",
            "llama3_70b_peft_8gpu_gb200_fp8cs_config",
            "llama3_70b_peft_8gpu_gb300_fp8cs_config",
            "llama3_70b_peft_8gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {"num_gpus": 8, "global_batch_size": 32},
        (
            "llama3_70b_peft_8gpu_b300_bf16_config",
            "llama3_70b_peft_8gpu_gb300_bf16_config",
            "llama3_8b_sft_8gpu_h100_bf16_config",
            "llama3_8b_sft_8gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 20,
            "global_batch_size": 64,
        },
        ("llama3_70b_peft_8gpu_gb200_bf16_config",),
    ),
    (
        {
            "num_gpus": 8,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 20,
            "global_batch_size": 32,
        },
        (
            "llama3_70b_peft_8gpu_gb200_fp8mx_config",
            "llama3_70b_peft_8gpu_h100_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 20,
            "global_batch_size": 32,
        },
        ("llama3_70b_peft_8gpu_h100_fp8cs_config",),
    ),
    (
        {"num_gpus": 32, "global_batch_size": 128},
        ("llama3_70b_pretrain_32gpu_gb200_bf16_config",),
    ),
    (
        {"num_gpus": 32, "global_batch_size": 128, "micro_batch_size": 2},
        (
            "llama3_70b_pretrain_32gpu_gb200_fp8cs_config",
            "llama3_70b_pretrain_32gpu_gb300_bf16_config",
            "llama3_70b_pretrain_32gpu_gb300_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 4,
            "context_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 5,
            "global_batch_size": 256,
        },
        ("llama3_70b_pretrain_64gpu_b200_bf16_config",),
    ),
    (
        {"num_gpus": 64, "global_batch_size": 256},
        (
            "llama3_70b_pretrain_64gpu_b200_fp8cs_config",
            "llama3_70b_pretrain_64gpu_b300_bf16_config",
            "llama3_70b_pretrain_64gpu_b300_fp8cs_config",
            "llama3_70b_pretrain_64gpu_gb200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 5,
            "global_batch_size": 256,
        },
        (
            "llama3_70b_pretrain_64gpu_b200_fp8mx_config",
            "llama3_70b_pretrain_64gpu_b200_nvfp4_config",
            "llama3_70b_pretrain_64gpu_gb200_fp8mx_config",
            "llama3_70b_pretrain_64gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 5,
            "global_batch_size": 256,
        },
        (
            "llama3_70b_pretrain_64gpu_b300_fp8mx_config",
            "llama3_70b_pretrain_64gpu_b300_nvfp4_config",
            "llama3_70b_pretrain_64gpu_gb300_fp8mx_config",
            "llama3_70b_pretrain_64gpu_gb300_nvfp4_config",
            "llama3_70b_pretrain_64gpu_vr200_fp8mx_config",
            "llama3_70b_pretrain_64gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 64, "global_batch_size": 256, "micro_batch_size": 2},
        (
            "llama3_70b_pretrain_64gpu_gb200_fp8cs_config",
            "llama3_70b_pretrain_64gpu_gb300_bf16_config",
            "llama3_70b_pretrain_64gpu_gb300_fp8cs_config",
            "llama3_70b_pretrain_64gpu_vr200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 4,
            "context_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 5,
            "global_batch_size": 256,
        },
        ("llama3_70b_pretrain_64gpu_h100_bf16_config",),
    ),
    (
        {
            "num_gpus": 64,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 5,
            "global_batch_size": 256,
        },
        ("llama3_70b_pretrain_64gpu_h100_fp8cs_config",),
    ),
    (
        {
            "num_gpus": 32,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 10,
            "global_batch_size": 32,
        },
        (
            "llama3_70b_sft_32gpu_gb200_bf16_config",
            "llama3_70b_sft_32gpu_gb200_fp8cs_config",
            "llama3_70b_sft_32gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 20,
            "global_batch_size": 32,
        },
        (
            "llama3_70b_sft_32gpu_gb300_bf16_config",
            "llama3_70b_sft_32gpu_gb300_fp8cs_config",
            "llama3_70b_sft_32gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "tensor_model_parallel_size": 4,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 5,
            "global_batch_size": 32,
        },
        (
            "llama3_70b_sft_32gpu_h100_bf16_config",
            "llama3_70b_sft_32gpu_h100_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 32, "global_batch_size": 512, "micro_batch_size": 2},
        (
            "llama3_8b_pretrain_32gpu_gb200_bf16_config",
            "llama3_8b_pretrain_32gpu_gb200_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 32, "global_batch_size": 512, "micro_batch_size": 4},
        (
            "llama3_8b_pretrain_32gpu_gb300_bf16_config",
            "llama3_8b_pretrain_32gpu_gb300_fp8cs_config",
            "llama3_8b_pretrain_32gpu_gb300_fp8mx_config",
            "llama3_8b_pretrain_32gpu_gb300_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 64, "global_batch_size": 1024, "micro_batch_size": 2},
        (
            "llama3_8b_pretrain_64gpu_b200_bf16_config",
            "llama3_8b_pretrain_64gpu_b200_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 64, "context_parallel_size": 2, "global_batch_size": 1024},
        ("llama3_8b_pretrain_64gpu_h100_bf16_config",),
    ),
    (
        {"num_gpus": 64, "global_batch_size": 1024},
        ("llama3_8b_pretrain_64gpu_h100_fp8cs_config",),
    ),
    (
        {"num_gpus": 8, "global_batch_size": 128, "micro_batch_size": 2},
        (
            "llama3_8b_pretrain_8gpu_b200_bf16_config",
            "llama3_8b_pretrain_8gpu_b200_fp8cs_config",
            "llama3_8b_pretrain_8gpu_b200_fp8mx_config",
            "llama3_8b_pretrain_8gpu_gb200_bf16_config",
            "llama3_8b_pretrain_8gpu_gb200_fp8cs_config",
            "llama3_8b_pretrain_8gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {"num_gpus": 8, "global_batch_size": 128, "micro_batch_size": 4},
        (
            "llama3_8b_pretrain_8gpu_b200_nvfp4_config",
            "llama3_8b_pretrain_8gpu_b300_bf16_config",
            "llama3_8b_pretrain_8gpu_b300_fp8cs_config",
            "llama3_8b_pretrain_8gpu_b300_fp8mx_config",
            "llama3_8b_pretrain_8gpu_b300_nvfp4_config",
            "llama3_8b_pretrain_8gpu_gb200_nvfp4_config",
            "llama3_8b_pretrain_8gpu_gb300_bf16_config",
            "llama3_8b_pretrain_8gpu_gb300_fp8cs_config",
            "llama3_8b_pretrain_8gpu_gb300_fp8mx_config",
            "llama3_8b_pretrain_8gpu_gb300_nvfp4_config",
            "llama3_8b_pretrain_8gpu_vr200_bf16_config",
            "llama3_8b_pretrain_8gpu_vr200_fp8cs_config",
            "llama3_8b_pretrain_8gpu_vr200_fp8mx_config",
            "llama3_8b_pretrain_8gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 8, "context_parallel_size": 2, "global_batch_size": 128},
        ("llama3_8b_pretrain_8gpu_h100_bf16_config",),
    ),
    (
        {"num_gpus": 8, "global_batch_size": 128},
        (
            "llama3_8b_pretrain_8gpu_h100_fp8cs_config",
            "llama3_8b_pretrain_8gpu_r100_bf16_config",
            "llama3_8b_pretrain_8gpu_r100_fp8cs_config",
            "llama3_8b_pretrain_8gpu_r100_fp8mx_config",
            "llama3_8b_pretrain_8gpu_r100_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 8, "global_batch_size": 8},
        (
            "llama3_8b_sft_8gpu_gb200_bf16_config",
            "llama3_8b_sft_8gpu_gb200_fp8cs_config",
            "llama3_8b_sft_8gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 144,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 36,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2304,
        },
        (
            "nemodiag_v0_pretrain_144gpu_gb300_bf16_perf72_e144_config",
            "nemodiag_v0_pretrain_144gpu_gb300_fp8mx_perf72_e144_config",
        ),
    ),
    (
        {
            "num_gpus": 144,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 36,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2304,
            "micro_batch_size": 2,
        },
        ("nemodiag_v0_pretrain_144gpu_gb300_nvfp4_perf72_e144_config",),
    ),
    (
        {
            "num_gpus": 288,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 36,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4608,
        },
        (
            "nemodiag_v0_pretrain_288gpu_gb300_bf16_perf72_e144_config",
            "nemodiag_v0_pretrain_288gpu_gb300_fp8mx_perf72_e144_config",
        ),
    ),
    (
        {
            "num_gpus": 288,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 36,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4608,
            "micro_batch_size": 2,
        },
        ("nemodiag_v0_pretrain_288gpu_gb300_nvfp4_perf72_e144_config",),
    ),
    (
        {
            "num_gpus": 72,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 36,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1152,
        },
        (
            "nemodiag_v0_pretrain_72gpu_gb300_bf16_perf72_e144_config",
            "nemodiag_v0_pretrain_72gpu_gb300_fp8mx_perf72_e144_config",
        ),
    ),
    (
        {
            "num_gpus": 72,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 36,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1152,
            "micro_batch_size": 2,
        },
        ("nemodiag_v0_pretrain_72gpu_gb300_nvfp4_perf72_e144_config",),
    ),
    (
        {"num_gpus": 16, "expert_model_parallel_size": 8, "expert_tensor_parallel_size": 1, "global_batch_size": 1024},
        (
            "nemotron_3_nano_pretrain_16gpu_h100_bf16_config",
            "nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
            "micro_batch_size": 2,
        },
        (
            "nemotron_3_nano_pretrain_8gpu_b200_bf16_config",
            "nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config",
            "nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config",
            "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config",
            "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config",
            "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
            "micro_batch_size": 4,
        },
        (
            "nemotron_3_nano_pretrain_8gpu_b300_bf16_config",
            "nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config",
            "nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config",
            "nemotron_3_nano_pretrain_8gpu_gb300_bf16_config",
            "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config",
            "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config",
            "nemotron_3_nano_pretrain_8gpu_vr200_bf16_config",
            "nemotron_3_nano_pretrain_8gpu_vr200_fp8mx_config",
            "nemotron_3_nano_pretrain_8gpu_vr200_nvfp4_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_bf16_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8cs_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_gb200_fp8mx_config",
            "qwen3_30b_a3b_pretrain_8gpu_b200_bf16_config",
            "qwen3_30b_a3b_pretrain_8gpu_b200_fp8cs_config",
            "qwen3_30b_a3b_pretrain_8gpu_b200_fp8mx_config",
            "qwen3_30b_a3b_pretrain_8gpu_gb200_bf16_config",
            "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8cs_config",
            "qwen3_30b_a3b_pretrain_8gpu_gb200_fp8mx_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_bf16_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_fp8cs_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {"num_gpus": 64, "expert_model_parallel_size": 64, "expert_tensor_parallel_size": 1, "global_batch_size": 512},
        (
            "nemotron_3_super_pretrain_64gpu_b200_bf16_config",
            "nemotron_3_super_pretrain_64gpu_b200_fp8mx_config",
            "nemotron_3_super_pretrain_64gpu_gb300_bf16_config",
            "nemotron_3_super_pretrain_64gpu_gb300_fp8mx_config",
            "nemotron_3_super_pretrain_64gpu_gb300_nvfp4_config",
            "nemotron_3_super_pretrain_64gpu_vr200_bf16_config",
            "nemotron_3_super_pretrain_64gpu_vr200_fp8mx_config",
            "nemotron_3_super_pretrain_64gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "tensor_model_parallel_size": 2,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
        },
        (
            "nemotron_3_super_pretrain_64gpu_b200_nvfp4_config",
            "nemotron_3_super_pretrain_64gpu_gb200_bf16_config",
            "nemotron_3_super_pretrain_64gpu_gb200_fp8mx_config",
            "nemotron_3_super_pretrain_64gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 64, "expert_model_parallel_size": 8, "expert_tensor_parallel_size": 1, "global_batch_size": 512},
        (
            "nemotron_3_super_pretrain_64gpu_b300_bf16_config",
            "nemotron_3_super_pretrain_64gpu_b300_fp8mx_config",
            "nemotron_3_super_pretrain_64gpu_b300_nvfp4_config",
        ),
    ),
    (
        {"num_gpus": 256, "tensor_model_parallel_size": 2, "global_batch_size": 768},
        (
            "nemotronh_56b_pretrain_256gpu_b200_bf16_config",
            "nemotronh_56b_pretrain_256gpu_b200_fp8cs_config",
            "nemotronh_56b_pretrain_256gpu_gb300_bf16_config",
            "nemotronh_56b_pretrain_256gpu_gb300_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 64, "tensor_model_parallel_size": 2, "global_batch_size": 192},
        (
            "nemotronh_56b_pretrain_64gpu_b200_fp8cs_config",
            "nemotronh_56b_pretrain_64gpu_b300_fp8cs_config",
            "nemotronh_56b_pretrain_64gpu_gb200_fp8cs_config",
            "nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 64, "tensor_model_parallel_size": 8, "global_batch_size": 192},
        ("nemotronh_56b_pretrain_64gpu_h100_fp8cs_config",),
    ),
    (
        {
            "num_gpus": 128,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 2,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_122b_a10b_pretrain_128gpu_h100_bf16_config",
            "qwen35_vl_122b_a10b_pretrain_128gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_122b_a10b_pretrain_32gpu_b200_bf16_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8cs_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_b200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
            "micro_batch_size": 2,
        },
        (
            "qwen35_vl_122b_a10b_pretrain_32gpu_b300_bf16_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8cs_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_b300_fp8mx_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_bf16_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8cs_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_gb300_fp8mx_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_vr200_bf16_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_vr200_fp8cs_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_vr200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_bf16_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8cs_config",
            "qwen35_vl_122b_a10b_pretrain_32gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 16,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 12,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
        },
        (
            "qwen35_vl_35b_a3b_pretrain_16gpu_h100_bf16_config",
            "qwen35_vl_35b_a3b_pretrain_16gpu_h100_fp8cs_config",
            "qwen3_vl_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "qwen3_vl_30b_a3b_pretrain_16gpu_h100_fp8cs_config",
        ),
    ),
    (
        {"num_gpus": 8, "expert_model_parallel_size": 8, "expert_tensor_parallel_size": 1, "global_batch_size": 512},
        (
            "qwen35_vl_35b_a3b_pretrain_8gpu_b200_bf16_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8cs_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_b200_fp8mx_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_b200_bf16_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8cs_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_b200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
            "micro_batch_size": 8,
        },
        (
            "qwen35_vl_35b_a3b_pretrain_8gpu_b300_bf16_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8cs_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_b300_fp8mx_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_bf16_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8cs_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_gb300_fp8mx_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_vr200_bf16_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8cs_config",
            "qwen35_vl_35b_a3b_pretrain_8gpu_vr200_fp8mx_config",
            "qwen3_30b_a3b_pretrain_8gpu_b300_bf16_config",
            "qwen3_30b_a3b_pretrain_8gpu_b300_fp8cs_config",
            "qwen3_30b_a3b_pretrain_8gpu_b300_fp8mx_config",
            "qwen3_30b_a3b_pretrain_8gpu_gb300_bf16_config",
            "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8cs_config",
            "qwen3_30b_a3b_pretrain_8gpu_gb300_fp8mx_config",
            "qwen3_30b_a3b_pretrain_8gpu_vr200_bf16_config",
            "qwen3_30b_a3b_pretrain_8gpu_vr200_fp8mx_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_bf16_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_fp8cs_config",
            "qwen3_vl_30b_a3b_pretrain_8gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_397b_a17b_pretrain_256gpu_h100_bf16_config",
            "qwen35_vl_397b_a17b_pretrain_256gpu_h100_fp8cs_config",
            "qwen3_vl_235b_a22b_pretrain_256gpu_h100_bf16_config",
            "qwen3_vl_235b_a22b_pretrain_256gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_397b_a17b_pretrain_64gpu_b200_bf16_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8cs_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_b200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_397b_a17b_pretrain_64gpu_b300_bf16_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8cs_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_b300_fp8mx_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_bf16_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8cs_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_gb300_fp8mx_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_vr200_bf16_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8cs_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_vr200_fp8mx_config",
            "qwen3_next_80b_a3b_pretrain_64gpu_b200_bf16_config",
            "qwen3_next_80b_a3b_pretrain_64gpu_b200_fp8mx_config",
            "qwen3_next_80b_a3b_pretrain_64gpu_b300_bf16_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_bf16_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_fp8cs_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_bf16_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8cs_config",
            "qwen35_vl_397b_a17b_pretrain_64gpu_gb200_fp8mx_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_b200_bf16_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_b200_fp8cs_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_b200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_b200_bf16_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
        },
        (
            "qwen3_235b_a22b_pretrain_256gpu_b200_fp8cs_config",
            "qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_config",
            "qwen3_235b_a22b_pretrain_256gpu_b200_nvfp4_config",
            "qwen3_235b_a22b_pretrain_256gpu_gb200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_b200_fp8mx_large_scale_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
            "micro_batch_size": 2,
        },
        (
            "qwen3_235b_a22b_pretrain_256gpu_b300_bf16_config",
            "qwen3_235b_a22b_pretrain_256gpu_b300_fp8cs_config",
            "qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_config",
            "qwen3_235b_a22b_pretrain_256gpu_b300_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
            "micro_batch_size": 2,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_b300_fp8mx_large_scale_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
        },
        (
            "qwen3_235b_a22b_pretrain_256gpu_gb200_fp8cs_config",
            "qwen3_235b_a22b_pretrain_256gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 3,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 3,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_gb200_fp8mx_large_scale_config",),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
            "micro_batch_size": 2,
        },
        (
            "qwen3_235b_a22b_pretrain_256gpu_gb300_bf16_config",
            "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8cs_config",
            "qwen3_235b_a22b_pretrain_256gpu_gb300_nvfp4_config",
            "qwen3_235b_a22b_pretrain_256gpu_vr200_bf16_config",
            "qwen3_235b_a22b_pretrain_256gpu_vr200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 12,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
            "micro_batch_size": 2,
        },
        (
            "qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_config",
            "qwen3_235b_a22b_pretrain_256gpu_vr200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 12,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
            "micro_batch_size": 2,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_gb300_fp8mx_large_scale_config",),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 8192,
        },
        (
            "qwen3_235b_a22b_pretrain_256gpu_h100_bf16_config",
            "qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 256,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 512,
        },
        ("qwen3_235b_a22b_pretrain_256gpu_h100_fp8cs_large_scale_config",),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        ("qwen3_235b_a22b_pretrain_64gpu_b200_bf16_config",),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        (
            "qwen3_235b_a22b_pretrain_64gpu_b200_fp8cs_config",
            "qwen3_235b_a22b_pretrain_64gpu_b200_fp8mx_config",
            "qwen3_235b_a22b_pretrain_64gpu_b200_nvfp4_config",
            "qwen3_235b_a22b_pretrain_64gpu_gb200_bf16_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
            "micro_batch_size": 2,
        },
        (
            "qwen3_235b_a22b_pretrain_64gpu_b300_bf16_config",
            "qwen3_235b_a22b_pretrain_64gpu_b300_fp8cs_config",
            "qwen3_235b_a22b_pretrain_64gpu_b300_fp8mx_config",
            "qwen3_235b_a22b_pretrain_64gpu_b300_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        (
            "qwen3_235b_a22b_pretrain_64gpu_gb200_fp8cs_config",
            "qwen3_235b_a22b_pretrain_64gpu_gb200_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "virtual_pipeline_model_parallel_size": 3,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
        },
        ("qwen3_235b_a22b_pretrain_64gpu_gb200_fp8mx_config",),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
            "micro_batch_size": 2,
        },
        (
            "qwen3_235b_a22b_pretrain_64gpu_gb300_bf16_config",
            "qwen3_235b_a22b_pretrain_64gpu_gb300_fp8cs_config",
            "qwen3_235b_a22b_pretrain_64gpu_gb300_nvfp4_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 4,
            "virtual_pipeline_model_parallel_size": 12,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
            "micro_batch_size": 2,
        },
        ("qwen3_235b_a22b_pretrain_64gpu_gb300_fp8mx_config",),
    ),
    (
        {
            "num_gpus": 16,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen3_30b_a3b_pretrain_16gpu_h100_bf16_config",
            "qwen3_30b_a3b_pretrain_16gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
            "micro_batch_size": 4,
        },
        (
            "qwen3_30b_a3b_pretrain_32gpu_gb200_bf16_config",
            "qwen3_30b_a3b_pretrain_32gpu_gb200_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 32,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 2048,
            "micro_batch_size": 8,
        },
        (
            "qwen3_30b_a3b_pretrain_32gpu_gb300_bf16_config",
            "qwen3_30b_a3b_pretrain_32gpu_gb300_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
            "micro_batch_size": 4,
        },
        (
            "qwen3_30b_a3b_pretrain_64gpu_b200_bf16_config",
            "qwen3_30b_a3b_pretrain_64gpu_b200_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 16,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 4096,
        },
        (
            "qwen3_30b_a3b_pretrain_64gpu_h100_bf16_config",
            "qwen3_30b_a3b_pretrain_64gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 128,
            "expert_model_parallel_size": 128,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
        },
        (
            "qwen3_next_80b_a3b_pretrain_128gpu_h100_bf16_config",
            "qwen3_next_80b_a3b_pretrain_128gpu_h100_fp8cs_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "expert_model_parallel_size": 64,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
            "micro_batch_size": 2,
        },
        ("qwen3_next_80b_a3b_pretrain_64gpu_b300_fp8mx_config",),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
            "micro_batch_size": 2,
        },
        (
            "qwen3_next_80b_a3b_pretrain_64gpu_gb200_bf16_config",
            "qwen3_next_80b_a3b_pretrain_64gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 2,
            "virtual_pipeline_model_parallel_size": 4,
            "expert_model_parallel_size": 32,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
            "micro_batch_size": 4,
        },
        (
            "qwen3_next_80b_a3b_pretrain_64gpu_gb300_bf16_config",
            "qwen3_next_80b_a3b_pretrain_64gpu_gb300_fp8mx_config",
        ),
    ),
    (
        {
            "num_gpus": 64,
            "pipeline_model_parallel_size": 8,
            "expert_model_parallel_size": 8,
            "expert_tensor_parallel_size": 1,
            "global_batch_size": 1024,
            "micro_batch_size": 2,
        },
        (
            "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_bf16_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_fp8cs_config",
            "qwen3_vl_235b_a22b_pretrain_64gpu_gb200_fp8mx_config",
        ),
    ),
    (
        {"num_gpus": 16, "context_parallel_size": 4, "global_batch_size": 64},
        ("wan_14b_pretrain_16gpu_gb200_bf16_config",),
    ),
    (
        {"num_gpus": 32, "tensor_model_parallel_size": 2, "context_parallel_size": 4, "global_batch_size": 64},
        ("wan_14b_pretrain_32gpu_h100_bf16_config",),
    ),
)


WORKLOAD_BASE_CONFIGS = {
    name: dict(config) for config, recipe_names in _WORKLOAD_CONFIG_GROUPS for name in recipe_names
}


__all__ = ["WORKLOAD_BASE_CONFIGS"]
