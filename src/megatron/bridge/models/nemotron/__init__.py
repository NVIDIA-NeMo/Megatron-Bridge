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

from megatron.bridge.models.nemotron.nemotron_bridge import NemotronBridge
from megatron.bridge.models.nemotron.nemotron_provider import (
    Nemotron3ModelProvider4B,
    Nemotron3ModelProvider8B,
    Nemotron3ModelProvider22B,
    Nemotron4ModelProvider15B,
    Nemotron4ModelProvider340B,
    NemotronModelProvider,
)


__all__ = [
    "NemotronBridge",
    "NemotronModelProvider",
    "Nemotron3ModelProvider4B",
    "Nemotron3ModelProvider8B",
    "Nemotron3ModelProvider22B",
    "Nemotron4ModelProvider15B",
    "Nemotron4ModelProvider340B",
]
