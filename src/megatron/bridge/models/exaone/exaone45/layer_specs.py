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

"""Provider-neutral EXAONE 4.5 text and MTP layer specifications."""

from typing import Any

from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlockSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules

from megatron.bridge.models.exaone.layer_specs import exaone4_layer_spec


def exaone_45_transformer_layer_spec(config: Any) -> ModuleSpec:
    """Create an EXAONE 4.5 layer spec backed by the post-LN layer pattern."""
    return exaone4_layer_spec(config)


def exaone_45_mtp_block_spec(
    config: Any,
    vp_stage: int | None = None,
) -> MultiTokenPredictionBlockSubmodules | None:
    """Create an MTP block spec that preserves the EXAONE transformer layer."""
    if not config.mtp_num_layers:
        return None

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

    layer_spec = exaone_45_transformer_layer_spec(config)
    block_spec = TransformerBlockSubmodules(layer_specs=[layer_spec])
    return get_gpt_mtp_block_spec(config, block_spec, use_transformer_engine=True, vp_stage=vp_stage)


__all__ = ["exaone_45_mtp_block_spec", "exaone_45_transformer_layer_spec"]
