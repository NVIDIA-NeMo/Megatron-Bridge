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

# Nemotron Nano v2 models
# Nemotron 3 Nano models
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_peft_config,
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_sft_config,
)

# Nemotron 3 Super models
from megatron.bridge.recipes.nemotronh.nemotron_3_super import (
    nemotron_3_super_peft_config,
    nemotron_3_super_pretrain_config,
    nemotron_3_super_sft_config,
)
from megatron.bridge.recipes.nemotronh.nemotron_nano_v2 import (
    nemotron_nano_9b_v2_peft_config,
    nemotron_nano_9b_v2_pretrain_config,
    nemotron_nano_9b_v2_sft_config,
    nemotron_nano_12b_v2_peft_config,
    nemotron_nano_12b_v2_pretrain_config,
    nemotron_nano_12b_v2_sft_config,
)

# NemotronH models
from megatron.bridge.recipes.nemotronh.nemotronh import (
    nemotronh_4b_peft_config,
    nemotronh_4b_pretrain_config,
    nemotronh_4b_sft_config,
    nemotronh_8b_peft_config,
    nemotronh_8b_pretrain_config,
    nemotronh_8b_sft_config,
    nemotronh_47b_peft_config,
    nemotronh_47b_pretrain_config,
    nemotronh_47b_sft_config,
    nemotronh_56b_peft_config,
    nemotronh_56b_pretrain_config,
    nemotronh_56b_sft_config,
)

# NemotronH perf recipes
from megatron.bridge.recipes.nemotronh.nemotronh_perf import (
    # Nemotron 3 Nano — B200
    nemotron_3_nano_pretrain_8gpu_b200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config,
    # Nemotron 3 Nano — B300
    nemotron_3_nano_pretrain_8gpu_b300_bf16_config,
    nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config,
    # Nemotron 3 Nano — GB200
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config,
    # Nemotron 3 Nano — GB300
    nemotron_3_nano_pretrain_8gpu_gb300_bf16_config,
    nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config,
    nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config,
    # Nemotron 3 Nano — H100
    nemotron_3_nano_pretrain_16gpu_h100_bf16_config,
    nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config,
    nemotronh_56b_pretrain_64gpu_b200_fp8cs_config,
    nemotronh_56b_pretrain_64gpu_b300_fp8cs_config,
    nemotronh_56b_pretrain_64gpu_gb200_fp8cs_config,
    # NemotronH 56B
    nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config,
    nemotronh_56b_pretrain_64gpu_h100_fp8cs_config,
)


__all__ = [
    # NemotronH models
    "nemotronh_4b_pretrain_config",
    "nemotronh_8b_pretrain_config",
    "nemotronh_47b_pretrain_config",
    "nemotronh_56b_pretrain_config",
    "nemotronh_4b_sft_config",
    "nemotronh_8b_sft_config",
    "nemotronh_47b_sft_config",
    "nemotronh_56b_sft_config",
    "nemotronh_4b_peft_config",
    "nemotronh_8b_peft_config",
    "nemotronh_47b_peft_config",
    "nemotronh_56b_peft_config",
    # Nemotron Nano v2 models
    "nemotron_nano_9b_v2_pretrain_config",
    "nemotron_nano_12b_v2_pretrain_config",
    "nemotron_nano_9b_v2_sft_config",
    "nemotron_nano_12b_v2_sft_config",
    "nemotron_nano_9b_v2_peft_config",
    "nemotron_nano_12b_v2_peft_config",
    # Nemotron 3 Nano models
    "nemotron_3_nano_pretrain_config",
    "nemotron_3_nano_sft_config",
    "nemotron_3_nano_peft_config",
    # Nemotron 3 Super models
    "nemotron_3_super_pretrain_config",
    "nemotron_3_super_sft_config",
    "nemotron_3_super_peft_config",
    # NemotronH perf recipes — 56B
    "nemotronh_56b_pretrain_64gpu_gb300_fp8cs_config",
    "nemotronh_56b_pretrain_64gpu_gb200_fp8cs_config",
    "nemotronh_56b_pretrain_64gpu_b300_fp8cs_config",
    "nemotronh_56b_pretrain_64gpu_b200_fp8cs_config",
    "nemotronh_56b_pretrain_64gpu_h100_fp8cs_config",
    # NemotronH perf recipes — Nemotron 3 Nano
    "nemotron_3_nano_pretrain_8gpu_gb300_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_gb300_fp8mx_config",
    "nemotron_3_nano_pretrain_8gpu_gb300_nvfp4_config",
    "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_gb200_fp8mx_config",
    "nemotron_3_nano_pretrain_8gpu_gb200_nvfp4_config",
    "nemotron_3_nano_pretrain_8gpu_b300_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_b300_fp8mx_config",
    "nemotron_3_nano_pretrain_8gpu_b300_nvfp4_config",
    "nemotron_3_nano_pretrain_8gpu_b200_bf16_config",
    "nemotron_3_nano_pretrain_8gpu_b200_fp8mx_config",
    "nemotron_3_nano_pretrain_8gpu_b200_nvfp4_config",
    "nemotron_3_nano_pretrain_16gpu_h100_bf16_config",
    "nemotron_3_nano_pretrain_16gpu_h100_fp8cs_config",
]
