# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Per-model-family MegatronMIMO conversion adapters.

Importing this package side-effect registers every adapter module below it,
so that :func:`get_mimo_adapter` can dispatch on a source bridge class. New
families add one module here and one ``@register_mimo_conversion`` decorator.
"""

# Side-effect imports: each module registers its adapter at import time.
from megatron.bridge.models.megatron_mimo.conversion.adapters import qwen35_vl  # noqa: F401
