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

"""GLM MoE mapping helpers for fused expert weights.

These are thin aliases around the shared FusedExpertMapping / FusedGatedExpertMapping
classes in param_mapping.py.  Kept for backwards compatibility with existing imports.
"""

from typing import Optional, Tuple

from megatron.bridge.models.conversion.param_mapping import (
    FusedExpertMapping,
)
from megatron.bridge.models.conversion.param_mapping import (
    FusedGatedExpertMapping as GLMExpertGateUpProjMapping,  # noqa: F401
)


class GLMExpertDownProjMapping(FusedExpertMapping):
    """FusedExpertMapping for GLM down-projection expert weights.

    GLM down-projection weights are stored transposed relative to Megatron's layout,
    so ``transpose_on_export`` is always enabled.
    """

    def __init__(
        self,
        megatron_param: str,
        hf_param: str,
        permute_dims: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__(megatron_param, hf_param, permute_dims, transpose_on_export=True)
