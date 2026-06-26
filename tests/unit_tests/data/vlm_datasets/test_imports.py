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

import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


def test_direct_qwen_vl_collate_import_has_no_vlm_dataset_cycle():
    result = subprocess.run(
        [sys.executable, "-c", "import megatron.bridge.models.qwen_vl.data.collate_fn"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_vlm_datasets_package_import_does_not_load_collate_registry():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import megatron.bridge.data.vlm_datasets; "
                "assert 'megatron.bridge.data.vlm_datasets.collate' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_vlm_datasets_collate_registry_remains_available_from_explicit_module():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS, qwen2_5_collate_fn; "
                "assert COLLATE_FNS['Qwen2_5_VLProcessor'] is qwen2_5_collate_fn"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
