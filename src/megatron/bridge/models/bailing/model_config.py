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

"""Provider-neutral Bailing MoE V2 model configuration."""

from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig


try:
    import transformer_engine  # type: ignore  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


bailing_moe2_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)


@dataclass(kw_only=True)
class BailingMoeV2ModelConfig(BridgeGPTModelConfig):
    """Builder-backed Bailing MoE V2 config with its family layer spec."""

    transformer_layer_spec: Callable[..., TransformerBlockSubmodules] = field(
        default_factory=lambda: bailing_moe2_layer_spec
    )


__all__ = ["BailingMoeV2ModelConfig", "bailing_moe2_layer_spec"]
