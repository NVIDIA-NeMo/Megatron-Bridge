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
"""save_state / load_state over Bridge checkpointing plus a per-rank user_state sidecar (phase B3)."""

from __future__ import annotations

from typing import Any

from megatron.bridge.engine.datum import CheckpointHandle


def save_engine_state(engine: Any, path: str, *, user_state: Any, async_save: bool) -> CheckpointHandle:
    """Save model, optimizer, progress and user state as one checkpoint version."""
    raise NotImplementedError("save_engine_state is implemented in phase B3.")


def load_engine_state(engine: Any, path: str) -> Any | None:
    """Restore a checkpoint saved by ``save_engine_state`` and return the local rank's user state."""
    raise NotImplementedError("load_engine_state is implemented in phase B3.")
