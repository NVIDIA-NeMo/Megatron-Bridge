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

"""Namespaced compatibility APIs for legacy MBridge callers.

The facade is the only convenience export. The compiler and migrated proof
remain explicitly discoverable without broadening that facade::

    from megatron.bridge.legacy.mapping_compiler import compile_legacy_mapping_registry
    from megatron.bridge.legacy.qwen3_moe import Qwen3MoELegacyMapping
"""

from megatron.bridge.legacy.mbridge import AutoBridge


__all__ = ["AutoBridge"]
