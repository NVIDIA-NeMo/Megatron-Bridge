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

# Llama2 models
from .llama2 import (
    llama2_7b_pretrain_config,
)

# Llama3 models
from .llama3 import (
    llama3_8b_16k_pretrain_config,
    llama3_8b_64k_pretrain_config,
    llama3_8b_128k_pretrain_config,
    llama3_8b_pretrain_config,
    llama3_70b_16k_pretrain_config,
    llama3_70b_64k_pretrain_config,
    llama3_70b_pretrain_config,
    # Llama3.1 models
    llama31_8b_pretrain_config,
    llama31_70b_pretrain_config,
    llama31_405b_pretrain_config,
    # Llama3.2 models
    llama32_1b_pretrain_config,
    llama32_3b_pretrain_config,
)


__all__ = [
    # Llama2 models
    "llama2_7b_pretrain_config",
    # Llama3 models
    "llama3_8b_pretrain_config",
    "llama3_8b_16k_pretrain_config",
    "llama3_8b_64k_pretrain_config",
    "llama3_8b_128k_pretrain_config",
    "llama3_70b_pretrain_config",
    "llama3_70b_16k_pretrain_config",
    "llama3_70b_64k_pretrain_config",
    # Llama3.1 models
    "llama31_8b_pretrain_config",
    "llama31_70b_pretrain_config",
    "llama31_405b_pretrain_config",
    # Llama3.2 models
    "llama32_1b_pretrain_config",
    "llama32_3b_pretrain_config",
]
