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

import functools
import importlib.util
import pathlib
import sys
import types

import pytest
import torch


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_SCRIPT_DIR = _REPO_ROOT / "scripts" / "conversion"
_CLI_PATH = _SCRIPT_DIR / "gpu_backend.py"


def _fake_megatron_modules():
    class AutoBridge:
        @staticmethod
        def from_hf_pretrained(*args, **kwargs):
            raise NotImplementedError

    def torchrun_main(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)

        return wrapper

    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
        "megatron.bridge.models.decorators": types.ModuleType("megatron.bridge.models.decorators"),
        "megatron.bridge.models.gpt_provider": types.ModuleType("megatron.bridge.models.gpt_provider"),
        "megatron.bridge.models.hf_pretrained": types.ModuleType("megatron.bridge.models.hf_pretrained"),
        "megatron.bridge.models.hf_pretrained.utils": types.ModuleType("megatron.bridge.models.hf_pretrained.utils"),
        "megatron.bridge.utils": types.ModuleType("megatron.bridge.utils"),
        "megatron.bridge.utils.common_utils": types.ModuleType("megatron.bridge.utils.common_utils"),
        "megatron.bridge.utils.slurm_utils": types.ModuleType("megatron.bridge.utils.slurm_utils"),
    }
    modules["megatron.bridge"].AutoBridge = AutoBridge
    modules["megatron.bridge.models.decorators"].torchrun_main = torchrun_main
    modules["megatron.bridge.models.gpt_provider"].GPTModelProvider = _FakeProvider
    modules["megatron.bridge.models.hf_pretrained.utils"].is_safe_repo = lambda **kwargs: False
    modules["megatron.bridge.utils.common_utils"].print_rank_0 = lambda message: None
    modules["megatron.bridge.utils.slurm_utils"].resolve_slurm_master_addr = lambda: None
    modules["megatron.bridge.utils.slurm_utils"].resolve_slurm_master_port = lambda: None
    return modules


@pytest.fixture(scope="module")
def cli():
    """Load the conversion script as a module under a stable test name."""
    spec = importlib.util.spec_from_file_location("gpu_backend_under_test", _CLI_PATH)
    module = importlib.util.module_from_spec(spec)
    fake_modules = _fake_megatron_modules()
    previous_modules = {name: sys.modules.get(name) for name in fake_modules}
    previous_utils = sys.modules.pop("utils", None)
    sys.modules.update(fake_modules)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path.remove(str(_SCRIPT_DIR))
        sys.modules.pop(spec.name, None)
        sys.modules.pop("utils", None)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
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


class TestPipelineLayout:
    def test_restore_uses_latest_checkpoint_iteration(self, cli, tmp_path):
        for iteration, marker in ((1, "old"), (2, "latest")):
            iteration_path = tmp_path / f"iter_{iteration:07d}"
            iteration_path.mkdir()
            (iteration_path / "run_config.yaml").write_text(
                f"model:\n  pipeline_model_parallel_layout:\n    - [{marker}]\n    - [{marker}]\n"
            )
        provider = _FakeProvider([])
        bridge = types.SimpleNamespace(_model_bridge=object(), hf_pretrained=_FakeHfPretrained())

        cli._maybe_restore_pipeline_layout(bridge, provider, str(tmp_path), pp=2)

        assert provider.pipeline_model_parallel_layout == [["latest"], ["latest"]]

    def test_restore_falls_back_to_root_config_when_latest_iteration_has_none(self, cli, tmp_path):
        (tmp_path / "iter_0000001").mkdir()
        (tmp_path / "run_config.yaml").write_text(
            "model:\n"
            "  pipeline_model_parallel_size: 2\n"
            "  pipeline_model_parallel_layout:\n"
            "    - [root-first]\n"
            "    - [root-second]\n"
        )
        provider = _FakeProvider([])
        bridge = types.SimpleNamespace(_model_bridge=object(), hf_pretrained=_FakeHfPretrained())

        cli._maybe_restore_pipeline_layout(bridge, provider, str(tmp_path), pp=2)

        assert provider.pipeline_model_parallel_layout == [["root-first"], ["root-second"]]

    def test_restore_preserves_virtual_pipeline_layout(self, cli, tmp_path):
        (tmp_path / "run_config.yaml").write_text(
            "model:\n"
            "  pipeline_model_parallel_size: 2\n"
            "  pipeline_model_parallel_layout:\n"
            "    - [embedding]\n"
            "    - [decoder]\n"
            "    - [decoder]\n"
            "    - [loss]\n"
        )
        provider = _FakeProvider([])
        bridge = types.SimpleNamespace(_model_bridge=object(), hf_pretrained=_FakeHfPretrained())

        cli._maybe_restore_pipeline_layout(bridge, provider, str(tmp_path), pp=2)

        assert provider.pipeline_model_parallel_layout == [
            ["embedding"],
            ["decoder"],
            ["decoder"],
            ["loss"],
        ]

    def test_restore_rebalances_pipeline_layout_to_pp_one(self, cli, tmp_path):
        (tmp_path / "run_config.yaml").write_text(
            "model:\n"
            "  pipeline_model_parallel_size: 2\n"
            "  pipeline_model_parallel_layout:\n"
            "    - [embedding, decoder]\n"
            "    - [decoder, loss]\n"
        )
        provider = _FakeProvider([])
        bridge = types.SimpleNamespace(_model_bridge=object(), hf_pretrained=_FakeHfPretrained())

        cli._maybe_restore_pipeline_layout(bridge, provider, str(tmp_path), pp=1)

        assert provider.pipeline_model_parallel_layout == [["embedding", "decoder", "decoder", "loss"]]

    def test_restore_regenerates_layout_for_different_pp(self, cli, tmp_path):
        checkpoint_path = tmp_path / "iter_0000001"
        checkpoint_path.mkdir()
        (checkpoint_path / "run_config.yaml").write_text(
            "model:\n  pipeline_model_parallel_layout:\n    - [old]\n    - [old]\n"
        )

        class ModelBridge:
            def generate_pipeline_layout(self, num_layers, pp, mtp_layers):
                return [[f"stage-{index}"] for index in range(pp)]

        provider = _FakeProvider([])
        bridge = types.SimpleNamespace(_model_bridge=ModelBridge(), hf_pretrained=_FakeHfPretrained())

        cli._maybe_restore_pipeline_layout(bridge, provider, str(checkpoint_path), pp=4)

        assert provider.pipeline_model_parallel_layout == [
            ["stage-0"],
            ["stage-1"],
            ["stage-2"],
            ["stage-3"],
        ]


class TestRoundtrip:
    def test_direct_roundtrip_verifies_without_saving(self, cli, monkeypatch):
        calls = []
        verified_models = []
        initialized_timeouts = []

        class FakeBridge:
            _model_bridge = object()
            hf_pretrained = _FakeHfPretrained()

            def to_megatron_provider(self, *args, **kwargs):
                calls.append(("to_megatron_provider", args, kwargs))
                return _FakeProvider(calls)

            def save_hf_pretrained(self, *args, **kwargs):
                raise AssertionError("--skip-save must not write a Hugging Face checkpoint")

            def save_megatron_model(self, *args, **kwargs):
                raise AssertionError("--skip-save must not write a Megatron checkpoint")

        monkeypatch.setattr(cli, "_ensure_distributed_initialized", initialized_timeouts.append)
        monkeypatch.setattr(cli, "is_safe_repo", lambda *, trust_remote_code, hf_path: trust_remote_code)
        monkeypatch.setattr(cli.AutoBridge, "from_hf_pretrained", lambda *args, **kwargs: FakeBridge())
        monkeypatch.setattr(cli, "_verify_roundtrip_weights", lambda bridge, model: verified_models.append(model))

        cli.roundtrip_checkpoint.__wrapped__(
            hf_model="hf/model",
            megatron_load_path=None,
            megatron_save_path="/unused-megatron",
            output_dir="/unused-hf",
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            trust_remote_code=True,
            strict=True,
            skip_save=True,
            overwrite=False,
            distributed_timeout_minutes=45,
        )

        provider_call = next(call for call in calls if call[0] == "to_megatron_provider")
        assert provider_call[2] == {"load_weights": True}
        assert verified_models == [["megatron-model"]]
        assert initialized_timeouts == [45]

    def test_loaded_roundtrip_verifies_and_saves_both_formats(self, cli, monkeypatch, tmp_path):
        calls = []
        verified_models = []

        class FakeModelShard:
            def cuda(self):
                raise AssertionError("roundtrip should not force loaded checkpoint shards to CUDA")

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

            def save_megatron_model(self, *args, **kwargs):
                calls.append(("save_megatron_model", args, kwargs))

        checkpoint_path = tmp_path / "megatron"
        checkpoint_path.mkdir()
        output_dir = tmp_path / "hf-output"
        megatron_save_path = tmp_path / "megatron-output"
        prepared_outputs = []
        monkeypatch.setattr(cli, "_ensure_distributed_initialized", lambda timeout_minutes: None)
        monkeypatch.setattr(
            cli,
            "_prepare_roundtrip_outputs",
            lambda *args, **kwargs: prepared_outputs.append((args, kwargs)),
        )
        monkeypatch.setattr(cli, "is_safe_repo", lambda *, trust_remote_code, hf_path: trust_remote_code)
        monkeypatch.setattr(cli.AutoBridge, "from_hf_pretrained", lambda *args, **kwargs: FakeBridge())
        monkeypatch.setattr(cli, "_verify_roundtrip_weights", lambda bridge, model: verified_models.append(model))

        cli.roundtrip_checkpoint.__wrapped__(
            hf_model="org/model",
            megatron_load_path=str(checkpoint_path),
            megatron_save_path=str(megatron_save_path),
            output_dir=str(output_dir),
            tp=1,
            pp=1,
            ep=2,
            etp=1,
            trust_remote_code=False,
            strict=False,
            skip_save=False,
            overwrite=True,
            distributed_timeout_minutes=None,
        )

        provider_call = next(call for call in calls if call[0] == "to_megatron_provider")
        assert provider_call[2] == {"load_weights": False}
        load_call = next(call for call in calls if call[0] == "load_megatron_model")
        assert load_call[2]["mp_overrides"]["expert_model_parallel_size"] == 2
        assert verified_models == [fake_model]
        assert prepared_outputs == [
            (
                (str(output_dir / "model"), str(megatron_save_path)),
                {
                    "overwrite": True,
                    "source_paths": ["org/model", str(checkpoint_path)],
                },
            )
        ]
        assert ("save_hf_pretrained", (fake_model, str(output_dir / "model")), {"strict": False}) in calls
        assert (
            "save_megatron_model",
            (fake_model, str(megatron_save_path)),
            {"hf_tokenizer_path": "org/model", "hf_tokenizer_kwargs": {}},
        ) in calls

    def test_weight_comparison_casts_known_precision_parameters(self, cli):
        original = torch.tensor([1.0], dtype=torch.bfloat16)
        exported = torch.tensor([1.05], dtype=torch.float32)

        match, skipped_fp8 = cli._roundtrip_weights_match("model.mlp.gate.weight", exported, original)

        assert match is True
        assert skipped_fp8 is False

    def test_verification_raises_on_weight_mismatch(self, cli, monkeypatch):
        class FakeBridge:
            hf_pretrained = types.SimpleNamespace(state={"weight": torch.tensor([2.0])})

            def export_hf_weights(self, model, *, show_progress):
                yield "weight", torch.tensor([1.0])

        monkeypatch.setattr(cli.torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(cli.torch.distributed, "broadcast", lambda tensor, src: None)
        monkeypatch.setattr(cli.torch.cuda, "current_device", lambda: "cpu")
        monkeypatch.setattr(cli._CONSOLE, "print", lambda *args, **kwargs: None)

        with pytest.raises(ValueError, match="Weight mismatch detected"):
            cli._verify_roundtrip_weights(FakeBridge(), ["megatron-model"])

    def test_output_destinations_must_not_overlap(self, cli, tmp_path):
        with pytest.raises(ValueError, match="Round-trip destinations overlap"):
            cli._prepare_roundtrip_outputs(
                str(tmp_path / "output" / "hf"),
                str(tmp_path / "output"),
                overwrite=False,
                source_paths=[],
            )
