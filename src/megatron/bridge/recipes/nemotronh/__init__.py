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
from megatron.bridge.recipes.nemotronh.nemotron_nano_v2 import (
    nemotron_nano_9b_v2_finetune_config,
    nemotron_nano_9b_v2_pretrain_config,
    nemotron_nano_12b_v2_finetune_config,
    nemotron_nano_12b_v2_pretrain_config,
)

# NemotronH models
from megatron.bridge.recipes.nemotronh.nemotronh import (
    nemotronh_4b_finetune_config,
    nemotronh_4b_pretrain_config,
    nemotronh_8b_finetune_config,
    nemotronh_8b_pretrain_config,
    nemotronh_47b_finetune_config,
    nemotronh_47b_pretrain_config,
    nemotronh_56b_finetune_config,
    nemotronh_56b_pretrain_config,
)

# Nemotron 3 Nano models
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_pretrain_config,
    nemotron_3_nano_finetune_config,
)

__all__ = [
    # NemotronH models
    "nemotronh_4b_pretrain_config",
    "nemotronh_8b_pretrain_config",
    "nemotronh_47b_pretrain_config",
    "nemotronh_56b_pretrain_config",
    "nemotronh_4b_finetune_config",
    "nemotronh_8b_finetune_config",
    "nemotronh_47b_finetune_config",
    "nemotronh_56b_finetune_config",
    # Nemotron Nano v2 models
    "nemotron_nano_9b_v2_pretrain_config",
    "nemotron_nano_12b_v2_pretrain_config",
    "nemotron_nano_9b_v2_finetune_config",
    "nemotron_nano_12b_v2_finetune_config",
    # Nemotron 3 Nano models
    "nemotron_3_nano_pretrain_config",
    "nemotron_3_nano_finetune_config",
]
