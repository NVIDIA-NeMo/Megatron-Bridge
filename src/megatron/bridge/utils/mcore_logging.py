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

"""Compatibility for MCore versions with fixed rank-zero logging."""

from megatron.core import utils as mcore_utils


def set_default_log_ranks(ranks: set[int]) -> None:
    """Select logging ranks when MCore supports configurable defaults.

    Older MCore versions always default to rank zero. Retain that behavior for
    ordinary training, but reject requests that need configurable logging ranks.

    Args:
        ranks: Global ranks that should emit default single-rank log messages.

    Raises:
        RuntimeError: If MCore lacks the setter and ranks differs from ``{0}``.
    """
    setter = getattr(mcore_utils, "set_default_log_ranks", None)
    if setter is not None:
        setter(ranks)
    elif ranks != {0}:
        raise RuntimeError(
            "This Megatron-Core version supports only rank-zero default logging. "
            "Selecting other logging ranks, including MIMO language-rank offsets, "
            "requires Megatron-Core with set_default_log_ranks support."
        )
