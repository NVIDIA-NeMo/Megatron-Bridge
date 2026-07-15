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
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _load_arguments_module():
    script = REPO_ROOT / "scripts" / "conversion" / "arguments.py"
    spec = importlib.util.spec_from_file_location("test_conversion_arguments", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_local_import_defaults():
    module = _load_arguments_module()

    args = module.build_parser(include_execution=True).parse_args(
        ["import", "--hf-model", "hf/model", "--megatron-path", "/checkpoint"]
    )

    assert args.executor == "local"
    assert args.device == "cpu"
    assert (args.tp, args.pp, args.ep, args.etp) == (1, 1, 1, 1)


def test_parallelism_aliases_and_export_defaults():
    module = _load_arguments_module()

    args = module.build_parser(include_execution=True).parse_args(
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
            "--tensor-parallel-size",
            "2",
            "--pp",
            "2",
            "-ep",
            "4",
            "-etp",
            "2",
        ]
    )

    assert (args.tp, args.pp, args.ep, args.etp) == (2, 2, 4, 2)
    assert args.distributed_save is None
    assert args.save_every_n_ranks == 1


def test_worker_args_enable_distributed_save_by_default_for_gpu_export():
    module = _load_arguments_module()
    args = module.build_parser(include_execution=True).parse_args(
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
        ]
    )

    worker_args = module.conversion_worker_args(args)

    assert "--distributed-save" in worker_args
    assert "--no-distributed-save" not in worker_args
    assert worker_args[:3] == ["export", "--device", "gpu"]


def test_worker_args_disable_distributed_save_for_cpu_export():
    module = _load_arguments_module()
    args = module.build_parser(include_execution=True).parse_args(
        [
            "export",
            "--hf-model",
            "hf/model",
            "--megatron-path",
            "/megatron",
            "--hf-path",
            "/hf",
        ]
    )

    worker_args = module.conversion_worker_args(args)

    assert "--no-distributed-save" in worker_args
