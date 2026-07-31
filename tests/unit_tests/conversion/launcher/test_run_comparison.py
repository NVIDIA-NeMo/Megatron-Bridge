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

import base64
import importlib.util
import json
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_module():
    script = REPO_ROOT / "scripts" / "conversion" / "run_comparison.py"
    spec = importlib.util.spec_from_file_location("test_run_comparison_module", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_decode_arguments_roundtrips_strings():
    module = _load_module()
    expected = ["--prompt", "hello; still one argument", "--image_path", "/image path/example.png"]
    encoded = base64.urlsafe_b64encode(json.dumps(expected).encode()).decode()

    assert module._decode_arguments(encoded) == expected


def test_main_runs_repository_comparison_with_decoded_arguments(monkeypatch):
    module = _load_module()
    expected = ["--hf_model_path", "hf/model", "--prompt", "Hello world"]
    encoded = base64.urlsafe_b64encode(json.dumps(expected).encode()).decode()
    calls = []
    monkeypatch.setattr(sys, "argv", ["run_comparison.py", "--arguments-b64", encoded])
    monkeypatch.setattr(module.runpy, "run_path", lambda *args, **kwargs: calls.append((args, kwargs)))

    module.main()

    compare_script = REPO_ROOT / "examples" / "conversion" / "compare_hf_and_megatron" / "compare.py"
    assert sys.argv == [str(compare_script), *expected]
    assert calls == [((str(compare_script),), {"run_name": "__main__"})]


@pytest.mark.parametrize(
    "value",
    [
        base64.urlsafe_b64encode(json.dumps({"prompt": "hello"}).encode()).decode(),
        base64.urlsafe_b64encode(json.dumps(["hello", 1]).encode()).decode(),
        "not-valid-base64",
    ],
)
def test_decode_arguments_rejects_invalid_payloads(value):
    module = _load_module()

    with pytest.raises(ValueError, match="comparison arguments"):
        module._decode_arguments(value)
