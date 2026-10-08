# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for CPU checkpoint conversion."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts" / "conversion"


def _load_cpu_backend():
    calls = []
    utils_spec = importlib.util.spec_from_file_location("utils", SCRIPT_DIR / "utils.py")
    conversion_utils = importlib.util.module_from_spec(utils_spec)
    utils_spec.loader.exec_module(conversion_utils)

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
            return types.SimpleNamespace(hf_pretrained="checkpoint-config")

    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.bridge": types.ModuleType("megatron.bridge"),
        "megatron.bridge.models": types.ModuleType("megatron.bridge.models"),
        "megatron.bridge.models.decorators": types.ModuleType("megatron.bridge.models.decorators"),
        "megatron.bridge.models.hf_pretrained": types.ModuleType("megatron.bridge.models.hf_pretrained"),
        "megatron.bridge.models.hf_pretrained.utils": types.ModuleType("megatron.bridge.models.hf_pretrained.utils"),
        "megatron.bridge.utils": types.ModuleType("megatron.bridge.utils"),
        "megatron.bridge.utils.slurm_utils": types.ModuleType("megatron.bridge.utils.slurm_utils"),
        "utils": conversion_utils,
    }
    modules["megatron.bridge"].AutoBridge = AutoBridge
    modules["megatron.bridge.models.decorators"].torchrun_main = lambda function: function
    modules["megatron.bridge.models.hf_pretrained.utils"].is_safe_repo = lambda **kwargs: kwargs["trust_remote_code"]
    modules["megatron.bridge.utils.slurm_utils"].resolve_slurm_master_addr = lambda: "master-host"
    modules["megatron.bridge.utils.slurm_utils"].resolve_slurm_master_port = lambda: 29501
    modules["utils"].parse_dtype = lambda value: f"dtype:{value}"
    modules["utils"].prepare_output_directory = lambda *args, **kwargs: calls.append(
        ("prepare_output_directory", args, kwargs)
    )
    modules["utils"].resolve_hf_model_revision = lambda model, revision: f"{model}@{revision}" if revision else model
    modules["utils"].validate_output_path = lambda *args, **kwargs: None

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


def test_distributed_import_initializes_gloo_without_selecting_cuda(monkeypatch):
    module, _ = _load_cpu_backend()
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_PROCID", "7")
    monkeypatch.setenv("SLURM_LOCALID", "7")
    initialize = Mock()
    monkeypatch.setattr(module.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(module.torch.distributed, "init_process_group", initialize)
    monkeypatch.setattr(module.torch.cuda, "set_device", lambda *_: pytest.fail("CPU ranks must not select GPUs"))

    module._ensure_distributed_initialized(45)

    initialize.assert_called_once_with(backend="gloo", timeout=module.datetime.timedelta(minutes=45))
    assert module.os.environ["RANK"] == module.os.environ["LOCAL_RANK"] == "7"
    assert module.os.environ["WORLD_SIZE"] == "8"
    assert module.os.environ["MASTER_ADDR"] == "master-host"
    assert module.os.environ["MASTER_PORT"] == "29501"


@pytest.mark.parametrize("use_builder", [False, True], ids=["provider", "builder"])
def test_distributed_import_builds_cpu_shards_and_preserves_metadata(monkeypatch, use_builder):
    module, _ = _load_cpu_backend()
    model_shards = [object()]

    def build_model(*args, **kwargs):
        assert transformer.use_cpu_initialization is True
        assert transformer.tensor_model_parallel_size == 2
        assert transformer.pipeline_model_parallel_size == 2
        assert transformer.expert_model_parallel_size == 2
        assert transformer.expert_tensor_parallel_size == 1
        return model_shards

    transformer = types.SimpleNamespace(
        finalize=Mock(),
        initialize_model_parallel=Mock(),
        provide_distributed_model=Mock(side_effect=build_model),
    )
    config = types.SimpleNamespace(transformer=transformer)
    bridge = types.SimpleNamespace(
        _model_bridge=types.SimpleNamespace(USE_MODEL_CONFIG_FOR_CONVERSION=use_builder),
        hf_model_revision="resolved-revision",
        to_megatron_provider=Mock(return_value=transformer),
        get_model_config=Mock(return_value=config),
        get_model=Mock(side_effect=build_model),
        save_megatron_model=Mock(),
    )
    monkeypatch.setattr(module, "_ensure_distributed_initialized", lambda timeout: None)
    monkeypatch.setattr(module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(module.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: pytest.fail("CPU import must not query CUDA"))
    monkeypatch.setattr(module.AutoBridge, "from_hf_pretrained", Mock(return_value=bridge))

    module.import_checkpoint(
        hf_model="hf/model",
        hf_revision="release-tag",
        megatron_path="/checkpoint",
        torch_dtype="bfloat16",
        trust_remote_code=True,
        overwrite=False,
        use_distributed=True,
        tp=2,
        pp=2,
        ep=2,
    )

    if use_builder:
        bridge.get_model.assert_called_once_with(config, wrap_with_ddp=False, mixed_precision_wrapper=None)
        bridge.to_megatron_provider.assert_not_called()
    else:
        transformer.provide_distributed_model.assert_called_once_with(wrap_with_ddp=False)
        bridge.get_model.assert_not_called()
    bridge.save_megatron_model.assert_called_once_with(
        model_shards,
        "/checkpoint",
        hf_tokenizer_path="hf/model",
        hf_tokenizer_kwargs={"trust_remote_code": True, "revision": "resolved-revision"},
        low_memory_save=False,
    )


def test_text_only_export_resolves_only_config_snapshot(tmp_path, monkeypatch):
    module, calls = _load_cpu_backend()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "run_config.yaml").touch()
    resolved = []
    monkeypatch.setattr(
        module,
        "resolve_hf_model_revision",
        lambda *args, **kwargs: resolved.append((args, kwargs)) or "config-snapshot",
    )
    module.export_checkpoint(
        hf_model="hf/model",
        hf_revision="pinned",
        megatron_path=str(checkpoint),
        hf_path="/hf-export",
        show_progress=False,
        strict=True,
        trust_remote_code=True,
        overwrite=False,
        text_only=True,
    )
    assert resolved == [(("hf/model", "pinned"), {"config_only": True})]
    loading = next(call for call in calls if call[0] == "from_hf_pretrained")
    assert loading[2]["text_only"] is True


def test_export_preserves_reference_state_layout_with_checkpoint_config(tmp_path):
    module, calls = _load_cpu_backend()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "run_config.yaml").touch()

    module.export_checkpoint(
        hf_model="hf/model",
        hf_revision="0123456789abcdef",  # pragma: allowlist secret
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
        (
            "from_hf_pretrained",
            ("hf/model",),
            {"trust_remote_code": False, "revision": "0123456789abcdef"},  # pragma: allowlist secret
        ),
        (
            "from_auto_config",
            (str(checkpoint), "hf/model@0123456789abcdef"),  # pragma: allowlist secret
            {"trust_remote_code": False},
        ),
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
