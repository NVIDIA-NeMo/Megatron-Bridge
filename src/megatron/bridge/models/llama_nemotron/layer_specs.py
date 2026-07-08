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

"""Layer specifications shared by Llama-Nemotron config paths."""

from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import get_gpt_heterogeneous_layer_spec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def llama_nemotron_layer_spec(
    config: BridgeGPTModelConfig, vp_stage: int | None = None
) -> TransformerBlockSubmodules:
    """Build the Transformer Engine heterogeneous layer specification."""
    return get_gpt_heterogeneous_layer_spec(config, use_te=True, vp_stage=vp_stage)


__all__ = ["llama_nemotron_layer_spec"]
