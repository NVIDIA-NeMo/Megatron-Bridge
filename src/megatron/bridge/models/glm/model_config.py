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

"""Provider-neutral GLM 4.5/4.7 model configurations."""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


glm_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)


@dataclass(kw_only=True)
class GLM45ModelConfig(BridgeGPTModelConfig):
    """Builder-backed GLM-4.5 config with its family layer spec."""

    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(default_factory=lambda: glm_layer_spec)


@dataclass(kw_only=True)
class GLM47FlashModelConfig(BridgeGPTModelConfig):
    """Builder-backed GLM-4.7 Flash config with MLA attention."""

    transformer_layer_spec: ModuleSpec | Callable[[BridgeGPTModelConfig], ModuleSpec] | None = field(
        default_factory=lambda: glm_layer_spec
    )


__all__ = ["GLM45ModelConfig", "GLM47FlashModelConfig", "glm_layer_spec"]
