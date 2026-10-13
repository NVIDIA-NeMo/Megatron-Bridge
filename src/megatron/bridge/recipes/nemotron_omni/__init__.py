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

"""Compatibility imports for recipes now owned by ``nemotronh_multimodal``."""

import sys
from importlib import import_module


# Alias modules, not just functions: old serialized targets and code that
# patches recipe globals must resolve to the same implementation as new imports.
for _suffix in (
    "",
    ".h100",
    ".gb200",
    ".nemotron_omni",
    ".nemotron_35_super_vl",
    ".h100.nemotron_omni",
    ".h100.nemotron_35_super_vl",
    ".gb200.nemotron_35_super_vl",
):
    sys.modules[__name__ + _suffix] = import_module("megatron.bridge.recipes.nemotronh_multimodal" + _suffix)
