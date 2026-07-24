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

from megatron.bridge.recipes.nemotronh.gb200.nemotron_3_nano import (
    nemotron_3_nano_gb200_pretrain_config,
    nemotron_3_nano_pretrain_8gpu_gb200_bf16_config,
)


__all__ = [
    "nemotron_3_nano_gb200_pretrain_config",
    "nemotron_3_nano_pretrain_8gpu_gb200_bf16_config",
]
