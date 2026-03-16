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

"""
Flow matching components for FLUX model.

This module contains the flow matching specific code, separated from the
model architecture to maintain consistency with other flow matching models
like WAN.
"""

from megatron.bridge.diffusion.models.flux.flow_matching.flux_adapter import MegatronFluxAdapter
from megatron.bridge.diffusion.models.flux.flow_matching.flux_inference_pipeline import (
    ClipConfig,
    FlowMatchEulerDiscreteScheduler,
    FluxInferencePipeline,
    T5Config,
)


__all__ = [
    "MegatronFluxAdapter",
    "FluxInferencePipeline",
    "FlowMatchEulerDiscreteScheduler",
    "T5Config",
    "ClipConfig",
]
