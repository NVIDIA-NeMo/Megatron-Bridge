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

try:
    import megatron.bridge  # noqa: F401

    HAVE_MEGATRON_BRIDGE = True
except ModuleNotFoundError:
    HAVE_MEGATRON_BRIDGE = False

if HAVE_MEGATRON_BRIDGE:
    from .wan_diffusion_pretrain import (
        wan_14b_pretrain_config_gb200,
        wan_14b_pretrain_config_h100,
    )

from .wan_workload_base_configs import (
    WAN_14B_PRETRAIN_CONFIG_GB200_BF16_V1,
    WAN_14B_PRETRAIN_CONFIG_H100_BF16_V1,
)


__all__ = [
    "WAN_14B_PRETRAIN_CONFIG_GB200_BF16_V1",
    "WAN_14B_PRETRAIN_CONFIG_H100_BF16_V1",
]

if HAVE_MEGATRON_BRIDGE:
    __all__.extend(
        [
            "wan_14b_pretrain_config_gb200",
            "wan_14b_pretrain_config_h100",
        ]
    )
