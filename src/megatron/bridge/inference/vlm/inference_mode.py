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

import contextlib


try:
    from megatron.core.inference.utils import InferenceMode  # type: ignore[attr-defined]
except ImportError:

    class InferenceMode:
        """Compatibility flag for MCore revisions without ``InferenceMode``."""

        _is_active: bool = False

        @classmethod
        def is_active(cls) -> bool:
            return cls._is_active

        @classmethod
        def set_active(cls) -> None:
            cls._is_active = True

        @classmethod
        def unset_active(cls) -> None:
            cls._is_active = False

        @classmethod
        @contextlib.contextmanager
        def active(cls):
            cls.set_active()
            try:
                yield
            finally:
                cls.unset_active()
