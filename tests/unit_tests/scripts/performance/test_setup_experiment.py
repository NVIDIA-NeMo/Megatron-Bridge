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

"""Tests for lightweight performance submission from a Slurm login node."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PERF_SCRIPTS_DIR = _REPO_ROOT / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

import setup_experiment


def test_default_experiment_name_uses_only_cli_metadata() -> None:
    assert (
        setup_experiment._default_experiment_name(
            use_recipes=False,
            model_recipe_name="deepseek_v3",
            task="pretrain",
            compute_dtype="fp8_mx",
            num_gpus=256,
            gpu="vr200",
            config_variant="large_scale",
        )
        == "pretrain_deepseek_v3_fp8_mx_gpus256_vr200_large_scale"
    )
    assert (
        setup_experiment._default_experiment_name(
            use_recipes=True,
            model_recipe_name="llama3_8b",
            task="pretrain",
            compute_dtype="bf16",
            num_gpus=8,
            gpu="h100",
            config_variant=None,
        )
        == "llama3_8b_pretrain_8gpu_h100"
    )


def test_perf_recipe_source_overlay_is_scoped_to_recipe_package() -> None:
    source = _REPO_ROOT / "src" / "megatron" / "bridge" / "perf_recipes"

    assert setup_experiment.PERF_RECIPE_SOURCE_DIR == source
    assert setup_experiment._perf_recipe_source_mount("/opt/Megatron-Bridge/scripts/performance") == (
        f"{source}:/opt/Megatron-Bridge/src/megatron/bridge/perf_recipes"
    )


def test_recipe_arguments_are_forwarded_unchanged() -> None:
    recipe_args = [
        "--tensor_model_parallel_size",
        "2",
        "--global_batch_size",
        "64",
        "++model.num_layers=2",
        "++env_vars.TEST_SENTINEL=1",
    ]
    argv = ["--wandb_experiment_name", "wandb-name", *recipe_args]

    assert setup_experiment._filter_run_script_args(argv) == [
        "--wandb_experiment_name",
        "wandb-name",
        *recipe_args,
    ]


def test_submission_dry_run_does_not_import_bridge_or_mcore(tmp_path: Path) -> None:
    blocker_dir = tmp_path / "login_node"
    blocker_dir.mkdir()
    (blocker_dir / "sitecustomize.py").write_text(
        """
import importlib.abc
import sys

class _RejectMegatron(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "megatron" or fullname.startswith("megatron."):
            raise RuntimeError(f"login-node import forbidden: {fullname}")
        return None

sys.meta_path.insert(0, _RejectMegatron())
"""
    )
    environment = os.environ.copy()
    environment["NEMORUN_HOME"] = str(tmp_path / "nemorun")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(blocker_dir), environment.get("PYTHONPATH")) if value
    )
    command = [
        sys.executable,
        str(_PERF_SCRIPTS_DIR / "setup_experiment.py"),
        "--model_family_name",
        "llama",
        "--model_recipe_name",
        "llama3_8b",
        "--num_gpus",
        "8",
        "--gpus_per_node",
        "8",
        "--gpu",
        "h100",
        "--account",
        "test-account",
        "--partition",
        "batch",
        "--container_image",
        "example.invalid/nemo:test",
        "--packager",
        "none",
        "--dryrun",
    ]

    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "bootstrap.py" in output
    assert "pretrain_llama3_8b_bf16_gpus8_h100" in output
    assert "login-node import forbidden" not in output
