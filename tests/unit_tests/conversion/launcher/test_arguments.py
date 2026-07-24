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
    assert args.srun_args == []
    assert (args.tp, args.pp, args.ep, args.etp) == (1, 1, 1, 1)


def test_srun_args_are_repeatable():
    module = _load_arguments_module()

    args = module.build_parser(include_execution=True).parse_args(
        [
            "import",
            "--executor",
            "slurm",
            "--srun-arg=--mpi=pmix",
            "--srun-arg=--cpus-per-task=8",
            "--hf-model",
            "hf/model",
            "--megatron-path",
            "/checkpoint",
        ]
    )

    assert args.srun_args == ["--mpi=pmix", "--cpus-per-task=8"]


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


def test_roundtrip_alias_and_worker_args():
    module = _load_arguments_module()
    args = module.build_parser(include_execution=True).parse_args(
        [
            "roundtrip",
            "--gpus-per-node",
            "8",
            "--hf-model-id",
            "hf/model",
            "--hf-revision",
            "0123456789abcdef",
            "--tp",
            "2",
            "--pp",
            "1",
            "--ep",
            "4",
            "--etp",
            "2",
            "--trust-remote-code",
            "--distributed-timeout-minutes",
            "30",
        ]
    )

    assert args.device == "gpu"
    assert args.hf_model == "hf/model"
    assert module.conversion_worker_args(args) == [
        "roundtrip",
        "--device",
        "gpu",
        "--hf-model",
        "hf/model",
        "--tp",
        "2",
        "--pp",
        "1",
        "--ep",
        "4",
        "--etp",
        "2",
        "--hf-revision",
        "0123456789abcdef",
        "--trust-remote-code",
        "--distributed-timeout-minutes",
        "30",
    ]


def test_import_worker_args_forward_hf_revision():
    module = _load_arguments_module()
    args = module.build_parser(include_execution=True).parse_args(
        [
            "import",
            "--hf-model",
            "hf/model",
            "--hf-revision",
            "0123456789abcdef",
            "--megatron-path",
            "/checkpoint",
        ]
    )

    worker_args = module.conversion_worker_args(args)

    assert worker_args[worker_args.index("--hf-revision") + 1] == "0123456789abcdef"


def test_roundtrip_worker_parser_accepts_serialized_args():
    module = _load_arguments_module()

    args = module.build_parser(include_execution=False).parse_args(
        [
            "roundtrip",
            "--device",
            "gpu",
            "--hf-model",
            "hf/model",
            "--tp",
            "2",
            "--ep",
            "4",
        ]
    )

    assert args.command == "roundtrip"
    assert args.hf_model == "hf/model"
    assert (args.tp, args.pp, args.ep, args.etp) == (2, 1, 4, 1)
    assert (
        not {
            "megatron_load_path",
            "megatron_save_path",
            "output_dir",
            "not_strict",
            "skip_save",
            "overwrite",
        }
        & vars(args).keys()
    )


def test_compare_hf_worker_args_forward_reference_revision():
    module = _load_arguments_module()
    args = module.build_parser(include_execution=True).parse_args(
        [
            "compare-hf",
            "--hf-model",
            "hf/model",
            "--hf-revision",
            "0123456789abcdef",
            "--hf-path",
            "/candidate",
        ]
    )

    assert module.conversion_worker_args(args) == [
        "compare-hf",
        "--device",
        "cpu",
        "--hf-model",
        "hf/model",
        "--hf-path",
        "/candidate",
        "--hf-revision",
        "0123456789abcdef",
    ]
