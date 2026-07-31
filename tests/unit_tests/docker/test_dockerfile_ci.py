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

from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


def test_dockerfile_ci_keeps_public_launchers_executable():
    dockerfile = Path(__file__).resolve().parents[3] / "docker" / "Dockerfile.ci"
    contents = dockerfile.read_text()
    repository_copy = contents.index("COPY --chmod=644 . /opt/Megatron-Bridge")

    for launcher in ("scripts/conversion/convert.sh", "scripts/training/train.sh"):
        executable_copy = f"COPY --chmod=755 {launcher} /opt/Megatron-Bridge/{launcher}"
        assert contents.index(executable_copy) > repository_copy
