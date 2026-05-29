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

"""Compatibility helpers for Megatron-Core inference API differences."""

try:
    from megatron.core.inference.utils import InferenceMode
except ImportError as exc:
    if "InferenceMode" not in str(exc):
        raise

    # TODO(mcore-dev): remove this guard when Megatron-Core dev exposes InferenceMode from
    # megatron.core.inference.utils.
    class InferenceMode:
        """No-op compatibility shim for MCore commits without InferenceMode."""

        @classmethod
        def is_active(cls) -> bool:
            """Return whether MCore's process-wide inference mode is active."""
            return False

        @classmethod
        def set_active(cls) -> None:
            """Mark inference as active when the backing MCore API exists."""

        @classmethod
        def unset_active(cls) -> None:
            """Mark inference as inactive when the backing MCore API exists."""
