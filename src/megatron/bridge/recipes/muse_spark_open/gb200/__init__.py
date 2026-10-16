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

"""GB200 recipes for Muse Spark Open."""

from megatron.bridge.recipes.muse_spark_open.gb200.muse_spark_open import (
    muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config,
    muse_spark_open_peft_64gpu_gb200_bf16_config,
    muse_spark_open_pretrain_192gpu_gb200_bf16_config,
    muse_spark_open_pretrain_256gpu_gb200_bf16_config,
    muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config,
    muse_spark_open_sft_192gpu_gb200_bf16_config,
    muse_spark_open_sft_192gpu_gb200_bf16_long_context_config,
    muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config,
    muse_spark_open_sft_256gpu_gb200_bf16_config,
)


__all__ = [
    "muse_spark_open_peft_16gpu_gb200_bf16_multimodal_config",
    "muse_spark_open_peft_64gpu_gb200_bf16_config",
    "muse_spark_open_pretrain_192gpu_gb200_bf16_config",
    "muse_spark_open_pretrain_256gpu_gb200_bf16_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_coderforge_64k_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_long_context_config",
    "muse_spark_open_sft_192gpu_gb200_bf16_multimodal_config",
    "muse_spark_open_sft_256gpu_gb200_bf16_config",
]
