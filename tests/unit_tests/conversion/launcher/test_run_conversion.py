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

import importlib.util
import io
import logging
import sys
import types
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = REPO_ROOT / "scripts" / "conversion"


def _load_run_conversion_module():
    cpu_backend = types.ModuleType("cpu_backend")
    gpu_backend = types.ModuleType("gpu_backend")
    previous_modules = {name: sys.modules.get(name) for name in ("cpu_backend", "gpu_backend", "arguments")}
    sys.modules["cpu_backend"] = cpu_backend
    sys.modules["gpu_backend"] = gpu_backend
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        spec = importlib.util.spec_from_file_location(
            "test_conversion_run_conversion", SCRIPT_DIR / "run_conversion.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPT_DIR))
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module, cpu_backend, gpu_backend


def test_cpu_import_dispatches_to_cpu_backend():
    module, cpu_backend, _ = _load_run_conversion_module()
    calls = []
    cpu_backend.import_checkpoint = lambda **kwargs: calls.append(kwargs)

    module.main(
        [
            "import",
            "--device",
            "cpu",
            "--hf-model",
            "hf/model",
            "--megatron-path",
            "/checkpoint",
        ]
    )

    assert calls == [
        {
            "hf_model": "hf/model",
            "megatron_path": "/checkpoint",
            "torch_dtype": "bfloat16",
            "trust_remote_code": False,
            "overwrite": False,
        }
    ]


def test_gpu_export_enables_distributed_save_by_default():
    module, _, gpu_backend = _load_run_conversion_module()
    calls = []
    gpu_backend.export_checkpoint = lambda **kwargs: calls.append(kwargs)

    module.main(
        [
            "export",
            "--device",
            "gpu",
            "--hf-model",
            "hf/model",
            "--megatron-path",
            "/megatron",
            "--hf-path",
            "/hf",
            "--ep",
            "2",
        ]
    )

    assert calls[0]["distributed_save"] is True
    assert calls[0]["ep"] == 2
    assert calls[0]["hf_path"] == "/hf"


def test_gpu_roundtrip_dispatches_to_gpu_backend():
    module, _, gpu_backend = _load_run_conversion_module()
    calls = []
    gpu_backend.roundtrip_checkpoint = lambda **kwargs: calls.append(kwargs)

    module.main(
        [
            "roundtrip",
            "--device",
            "gpu",
            "--hf-model",
            "hf/model",
            "--ep",
            "2",
        ]
    )

    assert calls == [
        {
            "hf_model": "hf/model",
            "tp": 1,
            "pp": 1,
            "ep": 2,
            "etp": 1,
            "trust_remote_code": False,
            "distributed_timeout_minutes": None,
        }
    ]


def test_cpu_worker_rejects_parallelism():
    module, _, _ = _load_run_conversion_module()

    with pytest.raises(ValueError, match="TP=PP=EP=ETP=1"):
        module.main(
            [
                "import",
                "--device",
                "cpu",
                "--hf-model",
                "hf/model",
                "--megatron-path",
                "/checkpoint",
                "--tp",
                "2",
            ]
        )


def test_cpu_worker_rejects_export_weight_dtype():
    module, _, _ = _load_run_conversion_module()

    with pytest.raises(ValueError, match="only supported by the GPU backend"):
        module.main(
            [
                "export",
                "--device",
                "cpu",
                "--hf-model",
                "hf/model",
                "--megatron-path",
                "/checkpoint",
                "--hf-path",
                "/hf-export",
                "--export-weight-dtype",
                "float32",
            ]
        )


def test_worker_logging_replaces_preexisting_configuration(monkeypatch):
    module, _, _ = _load_run_conversion_module()
    stream = io.StringIO()
    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    root_logger.handlers = [logging.NullHandler()]
    root_logger.setLevel(logging.WARNING)
    monkeypatch.setattr(sys, "stderr", stream)

    try:
        module._configure_logging()
        module.logger.info("CPU import complete: /checkpoint")
    finally:
        for handler in root_logger.handlers:
            handler.close()
        root_logger.handlers = previous_handlers
        root_logger.setLevel(previous_level)

    assert stream.getvalue() == "CPU import complete: /checkpoint\n"
