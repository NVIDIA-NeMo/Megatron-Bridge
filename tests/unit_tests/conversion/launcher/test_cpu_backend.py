# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the single-process CPU conversion backend."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts" / "conversion"


def _load_cpu_backend():
    calls = []

    class Bridge:
        def __init__(self, config):
            self.hf_pretrained = types.SimpleNamespace(config=config)

        def export_ckpt(self, **kwargs):
            calls.append(("export_ckpt", self.hf_pretrained.config, kwargs))

    class AutoBridge:
        @staticmethod
        def import_ckpt(**kwargs):
            calls.append(("import_ckpt", kwargs))

        @staticmethod
        def from_hf_pretrained(*args, **kwargs):
            calls.append(("from_hf_pretrained", args, kwargs))
            return Bridge("reference-config")

        @staticmethod
        def from_auto_config(*args, **kwargs):
            calls.append(("from_auto_config", args, kwargs))
            return Bridge("checkpoint-config")

    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
        "megatron.bridge.models.hf_pretrained": types.ModuleType("megatron.bridge.models.hf_pretrained"),
        "megatron.bridge.models.hf_pretrained.state": types.ModuleType("megatron.bridge.models.hf_pretrained.state"),
        "megatron.bridge.models.hf_pretrained.utils": types.ModuleType("megatron.bridge.models.hf_pretrained.utils"),
        "utils": types.ModuleType("utils"),
    }
    modules["megatron.bridge"].AutoBridge = AutoBridge
    modules["megatron.bridge.models.hf_pretrained.state"].SafeTensorsStateSource = object
    modules["megatron.bridge.models.hf_pretrained.utils"].is_safe_repo = lambda **kwargs: kwargs["trust_remote_code"]
    modules["utils"].parse_dtype = lambda value: f"dtype:{value}"
    modules["utils"].prepare_output_directory = lambda *args, **kwargs: calls.append(
        ("prepare_output_directory", args, kwargs)
    )

    previous_modules = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        spec = importlib.util.spec_from_file_location("cpu_backend_under_test", SCRIPT_DIR / "cpu_backend.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, calls


def test_import_preserves_model_id_and_forwards_revision():
    module, calls = _load_cpu_backend()

    module.import_checkpoint(
        hf_model="hf/model",
        hf_revision="0123456789abcdef",  # pragma: allowlist secret
        megatron_path="/checkpoint",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        overwrite=False,
    )

    assert calls == [
        ("prepare_output_directory", ("/checkpoint",), {"overwrite": False, "source_paths": ["hf/model"]}),
        (
            "import_ckpt",
            {
                "hf_model_id": "hf/model",
                "megatron_path": "/checkpoint",
                "torch_dtype": "dtype:bfloat16",
                "device_map": "cpu",
                "trust_remote_code": True,
                "revision": "0123456789abcdef",  # pragma: allowlist secret
            },
        ),
    ]


def test_export_preserves_reference_state_layout_with_checkpoint_config(tmp_path):
    module, calls = _load_cpu_backend()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "run_config.yaml").touch()

    module.export_checkpoint(
        hf_model="hf/model",
        megatron_path=str(checkpoint),
        hf_path="/hf-export",
        show_progress=False,
        strict=True,
        trust_remote_code=False,
        overwrite=False,
    )

    assert calls == [
        (
            "prepare_output_directory",
            ("/hf-export",),
            {"overwrite": False, "source_paths": [str(checkpoint), "hf/model"]},
        ),
        ("from_hf_pretrained", ("hf/model",), {"trust_remote_code": False}),
        ("from_auto_config", (str(checkpoint), "hf/model"), {"trust_remote_code": False}),
        (
            "export_ckpt",
            "checkpoint-config",
            {
                "megatron_path": str(checkpoint),
                "hf_path": "/hf-export",
                "show_progress": False,
                "strict": True,
            },
        ),
    ]


def test_compare_hf_checkpoints_reports_bitwise_match(caplog):
    module, _ = _load_cpu_backend()
    checkpoints = {
        "reference": {"a": torch.tensor([1, 2]), "b": torch.tensor([3], dtype=torch.bfloat16)},
        "candidate": {"a": torch.tensor([1, 2]), "b": torch.tensor([3], dtype=torch.bfloat16)},
    }

    class StateSource:
        def __init__(self, path):
            self.tensors = checkpoints[path]

        def get_all_keys(self):
            return list(self.tensors)

        def load_tensors(self, keys):
            return {key: self.tensors[key] for key in keys}

    module.SafeTensorsStateSource = StateSource
    with caplog.at_level("INFO"):
        module.compare_hf_checkpoints(reference_hf_path="reference", candidate_hf_path="candidate")

    assert "tensors=2, serialized_tensor_bytes=18" in caplog.text


def test_compare_hf_checkpoints_rejects_inventory_and_value_mismatches():
    module, _ = _load_cpu_backend()
    checkpoints = {
        "reference": {"a": torch.tensor([1, 2]), "missing": torch.tensor([3])},
        "candidate": {"a": torch.tensor([1, 9]), "unexpected": torch.tensor([3])},
    }

    class StateSource:
        def __init__(self, path):
            self.tensors = checkpoints[path]

        def get_all_keys(self):
            return list(self.tensors)

        def load_tensors(self, keys):
            return {key: self.tensors[key] for key in keys}

    module.SafeTensorsStateSource = StateSource
    with pytest.raises(
        ValueError,
        match=r"missing=1, unexpected=1, shape=0, dtype=0, value=1",
    ):
        module.compare_hf_checkpoints(reference_hf_path="reference", candidate_hf_path="candidate")
