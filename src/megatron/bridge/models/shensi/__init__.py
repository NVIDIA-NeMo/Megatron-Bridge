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

"""Shensi: a hybrid sparse-attention MoE model with multi-stream hyper-connections."""

from megatron.bridge.models.shensi.configuration_shensi import ShensiConfig
from megatron.bridge.models.shensi.modeling_shensi import ShensiModel
from megatron.bridge.models.shensi.shensi_bridge import ShensiBridge
from megatron.bridge.models.shensi.shensi_builder import ShensiModelBuilder, ShensiModelConfig
from megatron.bridge.models.shensi.shensi_hybrid import (
    ShensiDecoderStack,
    get_shensi_hybrid_stack_spec,
    shensi_hybrid_layer_pattern,
)
from megatron.bridge.models.shensi.shensi_provider import ShensiModelProvider


__all__ = [
    "ShensiBridge",
    "ShensiConfig",
    "ShensiDecoderStack",
    "ShensiModel",
    "ShensiModelBuilder",
    "ShensiModelConfig",
    "ShensiModelProvider",
    "get_shensi_hybrid_stack_spec",
    "shensi_hybrid_layer_pattern",
]
