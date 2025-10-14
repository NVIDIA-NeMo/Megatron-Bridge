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

from megatron.bridge.models.llama_nemotron.llama_nemotron_bridge import LlamaNemotronBridge
from megatron.bridge.models.llama_nemotron.llama_nemotron_provider import (
    Llama31Nemotron70BProvider,
    Llama31NemotronNano8BProvider,
    Llama31NemotronUltra253BProvider,
    Llama33NemotronSuper49BProvider,
    LlamaNemotronHeterogeneousProvider,
)


__all__ = [
    "LlamaNemotronBridge",
    "Llama31NemotronNano8BProvider",
    "Llama31Nemotron70BProvider",
    "Llama33NemotronSuper49BProvider",
    "Llama31NemotronUltra253BProvider",
    "LlamaNemotronHeterogeneousProvider",
]
