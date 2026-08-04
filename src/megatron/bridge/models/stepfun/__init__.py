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

from typing import TYPE_CHECKING, Any

from megatron.bridge.models.stepfun.modelling_step37.model import Step37Model
from megatron.bridge.models.stepfun.step35_bridge import Step35Bridge
from megatron.bridge.models.stepfun.step37_bridge import Step37Bridge


if TYPE_CHECKING:
    from megatron.bridge.models.stepfun.step37_provider import Step37ModelProvider


def __getattr__(name: str) -> Any:
    """Lazily preserve the legacy Step3.7 provider export."""
    if name == "Step37ModelProvider":
        from megatron.bridge.models.stepfun.step37_provider import Step37ModelProvider

        return Step37ModelProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Step35Bridge",
    "Step37Bridge",
    "Step37Model",
    "Step37ModelProvider",
]
