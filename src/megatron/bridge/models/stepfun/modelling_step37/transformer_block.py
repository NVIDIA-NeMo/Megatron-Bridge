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

"""Step3.7 transformer-block spec builder for the text decoder.

Mirrors ``qwen_vl/modelling_qwen3_vl/transformer_block.py``: returns the
per-layer ``TransformerLayerSubmodules`` spec consumed by Megatron's
``GPTModel``. Step3.7 reuses Step-3.5's hybrid full/sliding decoder layer
(``build_step35_layer_spec``) verbatim — the function lives in
``step35_bridge`` to keep all Step-3.5 spec-construction logic in one place.
"""

from megatron.bridge.models.stepfun.step35_bridge import build_step35_layer_spec


def get_step37_text_layer_spec(*args, **kwargs):
    """Return the Step-3.5 hybrid layer spec used as Step3.7's text decoder."""
    return build_step35_layer_spec(*args, **kwargs)


__all__ = ["get_step37_text_layer_spec"]
