# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
HuggingFace dataset processors for use with HFDatasetBuilder.

This module contains processing functions that conform to the ProcessExampleFn protocol
and are designed to work with the HFDatasetConfig and HFDatasetBuilder classes.
"""

from .squad import process_squad_example


__all__ = [
    "process_squad_example",
]
