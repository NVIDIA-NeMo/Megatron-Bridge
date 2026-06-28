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

"""Provider-neutral ERNIE 4.5 model configuration."""

from dataclasses import dataclass, field
from typing import Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


def ernie45_decoder_block_spec(config: BridgeGPTModelConfig, vp_stage: int | None = None) -> ModuleSpec:
    """Build the ERNIE 4.5 mixed dense/MoE decoder block."""
    return get_gpt_decoder_block_spec(
        config=config,
        use_transformer_engine=True,
        vp_stage=vp_stage,
    )


@dataclass(kw_only=True)
class Ernie45ModelConfig(BridgeGPTModelConfig):
    """Builder-backed ERNIE 4.5 model config."""

    transformer_layer_spec: Callable[..., ModuleSpec] = field(default_factory=lambda: ernie45_decoder_block_spec)


__all__ = ["Ernie45ModelConfig", "ernie45_decoder_block_spec"]
