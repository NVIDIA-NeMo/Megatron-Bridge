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
from concurrent.futures import ThreadPoolExecutor

import pytest


pytestmark = pytest.mark.unit


COLLATE_REGISTRY_MODULE = "megatron.bridge.data.collators.registry"
MODEL_COLLATE_MODULES = [
    "megatron.bridge.models.gemma_vl.data.collate_fn",
    "megatron.bridge.models.glm_vl.data.collate_fn",
    "megatron.bridge.models.kimi_vl.data.collate_fn",
    "megatron.bridge.models.ministral3.data.collate_fn",
    "megatron.bridge.models.nemotron_omni.data.collate_fn",
    "megatron.bridge.models.nemotron_vl.data.collate_fn",
    "megatron.bridge.models.qwen_audio.data.collate_fn",
    "megatron.bridge.models.qwen_omni.data.collate_fn",
    "megatron.bridge.models.qwen_vl.data.collate_fn",
]


def _run_import_check(name: str, code: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(
            f"{name} timed out after {error.timeout} seconds\nstdout: {error.stdout!r}\nstderr: {error.stderr!r}"
        ) from error


def test_collator_import_boundaries_preserve_lazy_loading():
    modules = ["megatron.bridge.data.collators.visual", *MODEL_COLLATE_MODULES]
    module_assertions = "; ".join(f"assert {module!r} not in sys.modules" for module in MODEL_COLLATE_MODULES)
    checks = {
        "direct collator imports": (
            "import importlib\n"
            "import sys\n"
            f"for module_name in {modules!r}:\n"
            "    importlib.import_module(module_name)\n"
            f"    assert {COLLATE_REGISTRY_MODULE!r} not in sys.modules\n"
        ),
        "lazy registry resolution": (
            "import sys; "
            "from megatron.bridge.data.collators.registry import resolve_model_collate; "
            f"{module_assertions}; "
            "collate = resolve_model_collate('Qwen2_5_VLProcessor'); "
            "assert collate.__name__ == 'qwen2_5_collate_fn'; "
            "assert 'megatron.bridge.models.qwen_vl.data.collate_fn' in sys.modules; "
            "assert 'megatron.bridge.models.gemma_vl.data.collate_fn' not in sys.modules"
        ),
    }

    with ThreadPoolExecutor(max_workers=len(checks)) as executor:
        futures = {name: executor.submit(_run_import_check, name, code) for name, code in checks.items()}
        results = {name: future.result() for name, future in futures.items()}

    for name, result in results.items():
        assert result.returncode == 0, f"{name} failed\n{result.stderr or result.stdout}"
