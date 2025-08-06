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

from megatron.bridge.models.llama.llama_bridge import LlamaBridge  # noqa: F401
from megatron.bridge.models.llama.llama_provider import (
    CodeLlamaModelProvider7B,
    CodeLlamaModelProvider13B,
    CodeLlamaModelProvider34B,
    CodeLlamaModelProvider70B,
    Llama2ModelProvider7B,
    Llama2ModelProvider13B,
    Llama2ModelProvider70B,
    Llama3ModelProvider,
    Llama3ModelProvider8B,
    Llama3ModelProvider70B,
    Llama4Experts16ModelProvider,
    Llama4Experts128ModelProvider,
    Llama4ModelProvider,
    Llama31ModelProvider,
    Llama31ModelProvider8B,
    Llama31ModelProvider70B,
    Llama31ModelProvider405B,
    Llama32ModelProvider1B,
    Llama32ModelProvider3B,
    LlamaModelProvider,
)


__all__ = [
    "LlamaModelProvider",
    "Llama2ModelProvider7B",
    "Llama2ModelProvider13B",
    "Llama2ModelProvider70B",
    "Llama3ModelProvider",
    "Llama3ModelProvider8B",
    "Llama3ModelProvider70B",
    "Llama31ModelProvider",
    "Llama31ModelProvider8B",
    "Llama31ModelProvider70B",
    "Llama31ModelProvider405B",
    "Llama32ModelProvider1B",
    "Llama32ModelProvider3B",
    "CodeLlamaModelProvider7B",
    "CodeLlamaModelProvider13B",
    "CodeLlamaModelProvider34B",
    "CodeLlamaModelProvider70B",
    "Llama4ModelProvider",
    "Llama4Experts16ModelProvider",
    "Llama4Experts128ModelProvider",
]
