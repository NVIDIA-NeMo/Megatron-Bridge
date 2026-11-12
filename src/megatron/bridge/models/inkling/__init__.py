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

from megatron.bridge.models.inkling.configuration_inkling2 import Inkling2Config, Inkling2TextConfig

# Importing the processor registers it with AutoProcessor for Inkling2Config.
from megatron.bridge.models.inkling.data import Inkling2ImageProcessor, Inkling2Processor
from megatron.bridge.models.inkling.inkling2_bridge import Inkling2Bridge
from megatron.bridge.models.inkling.inkling2_builder import Inkling2ModelBuilder
from megatron.bridge.models.inkling.inkling2_config import Inkling2ModelConfig, Inkling2TransformerConfig
from megatron.bridge.models.inkling.modeling_inkling2.model import Inkling2Model


__all__ = [
    "Inkling2Bridge",
    "Inkling2Config",
    "Inkling2ImageProcessor",
    "Inkling2Model",
    "Inkling2ModelBuilder",
    "Inkling2ModelConfig",
    "Inkling2Processor",
    "Inkling2TextConfig",
    "Inkling2TransformerConfig",
]
