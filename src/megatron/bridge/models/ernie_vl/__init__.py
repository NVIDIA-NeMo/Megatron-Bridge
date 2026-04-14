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

from megatron.bridge.models.ernie_vl.modeling_ernie45_vl import Ernie45VLModel
from megatron.bridge.models.ernie_vl.ernie45_vl_bridge import Ernie45VLBridge
from megatron.bridge.models.ernie_vl.ernie45_vl_provider import Ernie45VLModelProvider
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_moe_layer import ErnieMultiTypeMoE, MultiTypeMoeSubmodules
from megatron.bridge.models.ernie_vl.modeling_ernie45_vl.ernie_decoder_layer_spec import get_ernie45_vl_decoder_block_spec


__all__ = [
    "Ernie45VLBridge",
    "Ernie45VLModel",
    "Ernie45VLModelProvider",
    "ErnieMultiTypeMoE",
    "MultiTypeMoeSubmodules",
    "get_ernie45_vl_decoder_block_spec",
]
