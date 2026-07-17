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

"""Tests for the multi-GPU Hugging Face/Megatron round-trip CLI."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


pytestmark = pytest.mark.unit

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "examples" / "conversion" / "hf_megatron_roundtrip_multi_gpu.py"


@pytest.fixture(scope="module")
def cli():
    """Load the round-trip example as a module under a stable test name."""
    spec = importlib.util.spec_from_file_location("hf_megatron_roundtrip_multi_gpu_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


@pytest.mark.parametrize(("extra_args", "expected_strict"), [([], True), (["--not-strict"], False)])
def test_cli_forwards_export_strictness(cli, monkeypatch, extra_args, expected_strict):
    """The CLI is strict by default and loosens export only when requested."""
    calls = []
    monkeypatch.setattr(cli, "main", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(cli.torch.distributed, "is_initialized", lambda: False)

    cli._run_cli(["--hf-model-id", "org/model", *extra_args])

    assert len(calls) == 1
    assert calls[0][1]["strict"] is expected_strict
