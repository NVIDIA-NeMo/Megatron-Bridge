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


COLLATE_REGISTRY_MODULE = "megatron.bridge.data.vlm_datasets.collate"
LIGHTWEIGHT_COLLATE_MODULES = [
    "megatron.bridge.data.vlm_datasets.collate_utils",
    "megatron.bridge.models.gemma_vl.data.collate_fn",
    "megatron.bridge.models.glm_vl.data.collate_fn",
    "megatron.bridge.models.kimi_vl.data.collate_fn",
    "megatron.bridge.models.ministral3.data.collate_fn",
    "megatron.bridge.models.nemotron_omni.data.collate_fn",
    "megatron.bridge.models.nemotron_vl.data.collate_fn",
    "megatron.bridge.models.qwen_audio.data.collate_fn",
    "megatron.bridge.models.qwen_vl.data.collate_fn",
]


def _assert_subprocess_succeeds(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.parametrize("module_name", LIGHTWEIGHT_COLLATE_MODULES)
def test_direct_collate_module_import_does_not_load_registry(module_name):
    _assert_subprocess_succeeds(
        "import importlib; "
        "import sys; "
        f"importlib.import_module({module_name!r}); "
        f"assert {COLLATE_REGISTRY_MODULE!r} not in sys.modules"
    )


def test_vlm_datasets_package_import_does_not_load_collate_registry():
    _assert_subprocess_succeeds(
        "import sys; "
        "import megatron.bridge.data.vlm_datasets; "
        "assert 'megatron.bridge.data.vlm_datasets.collate' not in sys.modules"
    )


def test_vlm_datasets_collate_registry_remains_available_from_explicit_module():
    _assert_subprocess_succeeds(
        "from megatron.bridge.data.vlm_datasets.collate import COLLATE_FNS, qwen2_5_collate_fn; "
        "assert COLLATE_FNS['Qwen2_5_VLProcessor'] is qwen2_5_collate_fn"
    )
