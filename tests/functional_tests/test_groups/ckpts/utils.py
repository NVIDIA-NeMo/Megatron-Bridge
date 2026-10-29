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

"""Utilities for checkpoint functional tests."""

from pathlib import Path


def ensure_mcore_checkpoint_dir(checkpoint_dir: str) -> None:
    """Create the MCore checkpoint root before launching Megatron-LM.

    Megatron-LM opens ``progress.txt`` during startup when ``--log-progress``
    is enabled, before the first checkpoint save creates the directory.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
