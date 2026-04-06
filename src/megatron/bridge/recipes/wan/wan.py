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

"""Wan diffusion model recipes exposed under megatron.bridge.recipes.wan."""

from megatron.bridge.diffusion.recipes.wan.wan import (
    wan_1_3b_pretrain_config,
    wan_14b_pretrain_config,
)

__all__ = [
    "wan_1_3b_pretrain_config",
    "wan_14b_pretrain_config",
]
