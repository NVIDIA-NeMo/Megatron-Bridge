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

"""Layer specifications shared by GLM text and multimodal bridges."""

from typing import Any

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer import ModuleSpec


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


def glm_layer_spec(config: Any, vp_stage: int | None = None) -> ModuleSpec:
    """Build GLM's mixed dense/MoE block with the available backend."""
    return get_gpt_decoder_block_spec(config, use_transformer_engine=HAVE_TE, vp_stage=vp_stage)


__all__ = ["glm_layer_spec"]
