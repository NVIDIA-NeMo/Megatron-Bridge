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
Determinism verification utilities for Megatron-Bridge.

This module provides tools for detecting non-deterministic behavior during training
by comparing activations, gradients, and parameter updates across repeated runs.
"""

from megatron.bridge.training.utils.determinism.plugin import (
    DeterminismConfig,
    DeterminismDebugPlugin,
    analyze_results,
)

__all__ = [
    "DeterminismConfig",
    "DeterminismDebugPlugin",
    "analyze_results",
]
