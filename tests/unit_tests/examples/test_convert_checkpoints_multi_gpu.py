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

"""Unit tests for examples/conversion/convert_checkpoints_multi_gpu.py."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_PATH = _REPO_ROOT / "examples" / "conversion" / "convert_checkpoints_multi_gpu.py"


@pytest.fixture(scope="module")
def cli():
    """Load the conversion script as a module under a stable test name."""
    spec = importlib.util.spec_from_file_location("convert_checkpoints_multi_gpu_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


class _FakeProvider:
    def __init__(self, calls):
        self.calls = calls
        self.pipeline_model_parallel_layout = None

    def finalize(self):
        self.calls.append(("finalize", (), {}))

    def initialize_model_parallel(self, *args, **kwargs):
        self.calls.append(("initialize_model_parallel", args, kwargs))

    def provide_distributed_model(self, *args, **kwargs):
        self.calls.append(("provide_distributed_model", args, kwargs))
        return ["megatron-model"]


class _FakeModelBridge:
    def get_hf_tokenizer_kwargs(self):
        return {"padding_side": "left"}


class _FakeHfPretrained:
    config = type("Config", (), {"num_hidden_layers": 1, "num_nextn_predict_layers": 0})()


class TestImportHfToMegatron:
    def test_import_saves_megatron_checkpoint_with_tokenizer_metadata(self, cli, monkeypatch):
        calls = []

        class FakeBridge:
            _model_bridge = _FakeModelBridge()
            hf_pretrained = _FakeHfPretrained()

            def to_megatron_provider(self, *args, **kwargs):
                calls.append(("to_megatron_provider", args, kwargs))
                return _FakeProvider(calls)

            def save_megatron_model(self, *args, **kwargs):
                calls.append(("save_megatron_model", args, kwargs))

        def fake_from_hf_pretrained(*args, **kwargs):
            calls.append(("from_hf_pretrained", args, kwargs))
            return FakeBridge()

        monkeypatch.setattr(cli, "_ensure_distributed_initialized", lambda timeout_minutes: None)
        monkeypatch.setattr(cli, "is_safe_repo", lambda *, trust_remote_code, hf_path: trust_remote_code)
        monkeypatch.setattr(cli.AutoBridge, "from_hf_pretrained", fake_from_hf_pretrained)

        cli.import_hf_to_megatron.__wrapped__(
            hf_model="hf",
            megatron_path="/ckpt",
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            torch_dtype="bfloat16",
            trust_remote_code=True,
        )

        save_call = next(call for call in calls if call[0] == "save_megatron_model")
        assert save_call[1] == (["megatron-model"], "/ckpt")
        assert "low_memory_save" not in save_call[2]
        assert save_call[2]["hf_tokenizer_path"] == "hf"
        assert save_call[2]["hf_tokenizer_kwargs"] == {"padding_side": "left", "trust_remote_code": True}


class TestExportMegatronToHf:
    def test_export_does_not_move_loaded_model_to_cuda(self, cli, monkeypatch):
        calls = []

        class FakeModelShard:
            def cuda(self):
                raise AssertionError("export should not force loaded checkpoint shards to CUDA")

        fake_model = [FakeModelShard()]

        class FakeBridge:
            _model_bridge = object()
            hf_pretrained = _FakeHfPretrained()

            def to_megatron_provider(self, *args, **kwargs):
                calls.append(("to_megatron_provider", args, kwargs))
                return _FakeProvider(calls)

            def load_megatron_model(self, *args, **kwargs):
                calls.append(("load_megatron_model", args, kwargs))
                return fake_model

            def save_hf_pretrained(self, *args, **kwargs):
                calls.append(("save_hf_pretrained", args, kwargs))

        def fake_from_hf_pretrained(*args, **kwargs):
            calls.append(("from_hf_pretrained", args, kwargs))
            return FakeBridge()

        monkeypatch.setattr(cli, "_ensure_distributed_initialized", lambda timeout_minutes: None)
        monkeypatch.setattr(cli, "is_safe_repo", lambda *, trust_remote_code, hf_path: trust_remote_code)
        monkeypatch.setattr(cli.AutoBridge, "from_hf_pretrained", fake_from_hf_pretrained)

        cli.export_megatron_to_hf.__wrapped__(
            hf_model="hf",
            megatron_path="/ckpt/iter_0000000",
            hf_path="/hf-export",
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            distributed_save=True,
        )

        load_call = next(call for call in calls if call[0] == "load_megatron_model")
        assert load_call[1] == ("/ckpt/iter_0000000",)
        assert load_call[2]["mp_overrides"]["expert_model_parallel_size"] == 2

        save_call = next(call for call in calls if call[0] == "save_hf_pretrained")
        assert save_call[1] == (fake_model, "/hf-export")
        assert save_call[2]["distributed_save"] is True
