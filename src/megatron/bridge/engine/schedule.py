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
"""forward_step_func builder and forward_only / forward_backward drivers around the MCore schedule (phase B1/B2)."""

from __future__ import annotations

from typing import Any


def run_forward_only(engine: Any, microbatches: list[Any]) -> Any:
    """Drive ``get_forward_backward_func`` in forward-only mode over prepared microbatches."""
    raise NotImplementedError("run_forward_only is implemented in phase B1.")


def run_forward_backward(engine: Any, microbatches: list[Any], loss_method: Any) -> Any:
    """Drive the schedule with gradient finalization suppressed (accumulation across calls)."""
    raise NotImplementedError("run_forward_backward is implemented in phase B2.")
