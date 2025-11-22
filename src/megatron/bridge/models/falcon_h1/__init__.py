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

from megatron.bridge.models.falcon_h1.falconh1_bridge import FalconH1Bridge
from megatron.bridge.models.falcon_h1.falconh1_provider import (
    FalconH1ModelProvider,
    FalconH1ModelProvider500M,
    FalconH1ModelProvider1P5BDeep,
    FalconH1ModelProvider7B,
    FalconH1ModelProvider34B,
)
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_model import FalconH1Model, FalconH1Config
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_block import FalconH1Stack, FalconH1StackSubmodules
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer import FalconH1Layer, FalconH1Submodules
from megatron.bridge.models.falcon_h1.modeling_falconh1.falconh1_layer_specs import falconh1_stack_spec


__all__ = [
    # Providers
    "FalconH1ModelProvider",
    "FalconH1ModelProvider500M",
    "FalconH1ModelProvider1P5BDeep",
    "FalconH1ModelProvider7B",
    "FalconH1ModelProvider34B",
    # Bridge
    "FalconH1Bridge",
    # Model and Config
    "FalconH1Model",
    "FalconH1Config",
    # Stack and Layer
    "FalconH1Stack",
    "FalconH1StackSubmodules",
    "FalconH1Layer",
    "FalconH1Submodules",
    # Specs
    "falconh1_stack_spec",
]