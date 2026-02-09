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

try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .kimi_llm_pretrain import (
        kimi_k2_pretrain_config_b200,
        kimi_k2_pretrain_config_gb200,
        kimi_k2_pretrain_config_gb300,
        kimi_k2_pretrain_config_h100,
    )

from .kimi_workload_base_configs import (
    KIMI_K2_PRETRAIN_CONFIG_B200_BF16,
    KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX,
    KIMI_K2_PRETRAIN_CONFIG_GB200_BF16,
    KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX,
    KIMI_K2_PRETRAIN_CONFIG_GB300_BF16,
    KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX,
    KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4,
    KIMI_K2_PRETRAIN_CONFIG_H100_BF16,
    KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS,
    KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC,
)


__all__ = [
    "KIMI_K2_PRETRAIN_CONFIG_B200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_B200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB200_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_FP8_MX",
    "KIMI_K2_PRETRAIN_CONFIG_GB300_NVFP4",
    "KIMI_K2_PRETRAIN_CONFIG_H100_BF16",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_CS",
    "KIMI_K2_PRETRAIN_CONFIG_H100_FP8_SC",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "kimi_k2_pretrain_config_gb300",
            "kimi_k2_pretrain_config_gb200",
            "kimi_k2_pretrain_config_b200",
            "kimi_k2_pretrain_config_h100",
        ]
    )
