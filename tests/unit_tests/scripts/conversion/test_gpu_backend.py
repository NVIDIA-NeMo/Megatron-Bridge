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

"""Unit tests for the distributed GPU conversion backend."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "conversion"
_CLI_PATH = _SCRIPT_DIR / "gpu_backend.py"


@pytest.fixture(scope="module")
def cli():
    """Load the conversion script as a module under a stable test name."""
    spec = importlib.util.spec_from_file_location("gpu_backend_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    previous_utils = sys.modules.pop("utils", None)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(_SCRIPT_DIR))
        sys.modules.pop(spec.name, None)
        sys.modules.pop("utils", None)
        if previous_utils is not None:
            sys.modules["utils"] = previous_utils


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
        prepared_outputs = []

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
        monkeypatch.setattr(
            cli,
            "_prepare_distributed_output",
            lambda *args, **kwargs: prepared_outputs.append((args, kwargs)),
        )
        monkeypatch.setattr(cli, "is_safe_repo", lambda *, trust_remote_code, hf_path: trust_remote_code)
        monkeypatch.setattr(cli.AutoBridge, "from_hf_pretrained", fake_from_hf_pretrained)

        cli.import_checkpoint.__wrapped__(
            hf_model="hf",
            megatron_path="/ckpt",
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            distributed_timeout_minutes=None,
            overwrite=False,
        )

        save_call = next(call for call in calls if call[0] == "save_megatron_model")
        assert save_call[1] == (["megatron-model"], "/ckpt")
        assert "low_memory_save" not in save_call[2]
        assert save_call[2]["hf_tokenizer_path"] == "hf"
        assert save_call[2]["hf_tokenizer_kwargs"] == {"padding_side": "left", "trust_remote_code": True}
        assert prepared_outputs == [(("/ckpt",), {"overwrite": False, "source_paths": ["hf"]})]


class TestExportMegatronToHf:
    def test_export_does_not_move_loaded_model_to_cuda(self, cli, monkeypatch, tmp_path):
        calls = []
        prepared_outputs = []

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
        monkeypatch.setattr(
            cli,
            "_prepare_distributed_output",
            lambda *args, **kwargs: prepared_outputs.append((args, kwargs)),
        )
        monkeypatch.setattr(cli, "is_safe_repo", lambda *, trust_remote_code, hf_path: trust_remote_code)
        monkeypatch.setattr(cli.AutoBridge, "from_hf_pretrained", fake_from_hf_pretrained)

        checkpoint_path = tmp_path / "iter_0000000"
        checkpoint_path.mkdir()
        cli.export_checkpoint.__wrapped__(
            hf_model="hf",
            megatron_path=str(checkpoint_path),
            hf_path="/hf-export",
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            torch_dtype="bfloat16",
            trust_remote_code=True,
            strict=True,
            show_progress=True,
            distributed_save=True,
            save_every_n_ranks=1,
            distributed_timeout_minutes=None,
            export_weight_dtype=None,
            overwrite=False,
        )

        load_call = next(call for call in calls if call[0] == "load_megatron_model")
        assert load_call[1] == (str(checkpoint_path),)
        assert load_call[2]["mp_overrides"]["expert_model_parallel_size"] == 2

        save_call = next(call for call in calls if call[0] == "save_hf_pretrained")
        assert save_call[1] == (fake_model, "/hf-export")
        assert save_call[2]["distributed_save"] is True
        assert prepared_outputs == [
            (("/hf-export",), {"overwrite": False, "source_paths": [str(checkpoint_path), "hf"]})
        ]
