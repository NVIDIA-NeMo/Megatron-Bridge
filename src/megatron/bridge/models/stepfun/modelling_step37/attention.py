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

"""Step3.7 text-side attention re-exports.

Step3.7's text decoder reuses Step-3.5's hybrid full/sliding attention layer
(``Step35DecoderLayer``) verbatim, so this module is a structural alias of
``qwen_vl/modelling_qwen3_vl/attention.py`` — it surfaces the Step-3.5
decoder layer under the ``modelling_step37`` namespace without redefining any
attention math.
"""

from megatron.bridge.models.stepfun.step35_provider import (
    Step35DecoderLayer as Step37DecoderLayer,
)
from megatron.bridge.models.stepfun.step35_provider import (
    Step35SharedExpertMLP as Step37SharedExpertMLP,
)


__all__ = ["Step37DecoderLayer", "Step37SharedExpertMLP"]
