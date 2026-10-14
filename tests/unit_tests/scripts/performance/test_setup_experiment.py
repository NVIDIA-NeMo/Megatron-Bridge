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

import json
import os
import runpy
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


try:
    import nemo_run
    from nemo_run.core.execution.nvcre import NvcreExecutor  # noqa: F401

    HAS_NVCRE = True
except (ImportError, ModuleNotFoundError):
    HAS_NVCRE = False


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


def test_bridge_source_overlay_keeps_scripts_and_package_on_the_same_checkout() -> None:
    source = _REPO_ROOT / "src" / "megatron" / "bridge"

    assert setup_experiment.BRIDGE_SOURCE_DIR == source
    assert setup_experiment._bridge_source_mount("/opt/Megatron-Bridge/scripts/performance") == (
        f"{source}:/opt/Megatron-Bridge/src/megatron/bridge"
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


@pytest.mark.parametrize("is_long_convergence_run", [False, True])
def test_submission_dry_run_does_not_import_bridge_or_mcore(tmp_path: Path, is_long_convergence_run: bool) -> None:
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

# Exercise the W&B 0.29 API surface even when testing with an older SDK.
import wandb
if hasattr(wandb.util, "generate_id"):
    del wandb.util.generate_id
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
    if is_long_convergence_run:
        command.append("--is_long_convergence_run")

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
    assert "WANDB_RUN_ID" in output
    assert "WANDB_RESUME" in output
    assert "login-node import forbidden" not in output


def test_dryrun_short_flag_is_filtered_from_recipe_args() -> None:
    """-d (short form of --dryrun) must be stripped from args forwarded to the recipe script."""
    argv = ["-d", "--tensor_model_parallel_size", "2"]
    assert "-d" not in setup_experiment._filter_run_script_args(argv)


def test_nvcre_flags_are_filtered_from_recipe_args() -> None:
    """All --nvcre_* flags and their values must be stripped before forwarding to the recipe."""
    argv = [
        "--nvcre_namespace",
        "nemo-perf",
        "--nvcre_image_pull_secret=ngc-secret",
        "--tensor_model_parallel_size",
        "2",
    ]
    filtered = setup_experiment._filter_run_script_args(argv)
    assert filtered == ["--tensor_model_parallel_size", "2"]


def test_nvcre_use_mbridge_commit_flag_does_not_swallow_the_next_arg() -> None:
    """--nvcre_use_mbridge_commit takes no value, so the arg after it must still reach the recipe."""
    argv = ["--nvcre_use_mbridge_commit", "--tensor_model_parallel_size", "2"]
    assert setup_experiment._filter_run_script_args(argv) == ["--tensor_model_parallel_size", "2"]


def test_use_mbridge_commit_without_nvcre_namespace_is_an_error() -> None:
    with pytest.raises(ValueError, match="--nvcre_namespace"):
        setup_experiment._resolve_local_bridge(True, None)


def test_resolve_local_bridge_is_none_when_the_flag_is_off() -> None:
    assert setup_experiment._resolve_local_bridge(False, "ns") is None
    assert setup_experiment._resolve_local_bridge(False, None) is None


def test_resolve_local_bridge_reads_this_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    source = setup_experiment.LocalBridge("https://github.com/o/r.git", "a" * 40)
    monkeypatch.setattr(setup_experiment, "local_bridge_source", lambda root: source)

    assert setup_experiment._resolve_local_bridge(True, "ns") is source


@pytest.mark.skipif(not HAS_NVCRE, reason="nemo_run nvcre not available")
def test_main_wires_the_pod_checkout_into_the_workloadrun(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--nvcre_use_mbridge_commit runs the scripts from the pod clone, via an init container and its volume."""
    from utils import nvcre_bridge

    source = nvcre_bridge.LocalBridge("https://github.com/o/r.git", "a" * 40)
    monkeypatch.setattr(nvcre_bridge, "local_bridge_source", lambda root: source)
    user_volume = {"name": "data", "persistentVolumeClaim": {"claimName": "pvc"}}
    user_mount = {"name": "data", "mountPath": "/data"}
    submitted = {}

    def fake_run(script, **kwargs):
        submitted.update(script=script, **kwargs)

    monkeypatch.setattr(nemo_run, "run", fake_run)
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path / "nemorun"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "setup_experiment.py",
            *("--model_family_name", "llama", "--model_recipe_name", "llama3_8b"),
            *("--num_gpus", "8", "--gpus_per_node", "4", "--gpu", "gb300"),
            *("--container_image", "example.invalid/nemo:test", "--dryrun"),
            *("--nvcre_namespace", "ns", "--nvcre_use_mbridge_commit"),
            *("--nvcre_volumes_json", json.dumps([user_volume])),
            *("--nvcre_volume_mounts_json", json.dumps([user_mount])),
        ],
    )

    runpy.run_path(str(_PERF_SCRIPTS_DIR / "setup_experiment.py"), run_name="__main__")

    executor = submitted["executor"]
    executor.job_name = "job1"
    executor.experiment_id = "exp_123456789"
    command = submitted["script"].to_command()
    spec = executor.build_workloadrun_yaml(command)["spec"]
    pod_scripts = f"{nvcre_bridge.BRIDGE_POD_ROOT}/scripts/performance/"

    assert any(pod_scripts in part for part in command)
    assert [c["name"] for c in spec["initContainers"]] == ["megatron-bridge-clone"]
    assert f"git fetch -q --depth 1 origin {'a' * 40}" in spec["initContainers"][0]["args"][0]
    assert spec["volumes"] == [user_volume, nvcre_bridge.WORKSPACE_VOLUME]
    assert spec["volumeMounts"] == [user_mount, nvcre_bridge.WORKSPACE_MOUNT]
    assert submitted["script"].env["PYTHONPATH"].startswith(f"{pod_scripts.rstrip('/')}:")


def test_nemorun_script_pythonpath_uses_the_pod_checkout_without_its_megatron_lm() -> None:
    kwargs = dict(
        script_path="/mnt/workspace/megatron-bridge/scripts/performance/run.py",
        script_dir="/mnt/workspace/megatron-bridge/scripts/performance",
        args=[],
        kubeflow_namespace=None,
        custom_env_vars={},
    )
    root = "/mnt/workspace/megatron-bridge"
    script = setup_experiment._build_nemorun_script(**kwargs)
    assert script.env["PYTHONPATH"] == f"{root}/scripts/performance:{root}/src:$PYTHONPATH"


# ── _nvcre_env_vars ───────────────────────────────────────────────────────────


def test_nvcre_env_vars_without_hf_token_secret() -> None:
    """Custom env vars must be forwarded unchanged when no HF token secret is configured."""
    custom = {"NCCL_TIMEOUT": "1800", "MY_VAR": "1"}
    result = setup_experiment._nvcre_env_vars(custom, hf_token_secret_name=None)
    assert result == custom
    assert result is not custom  # must be an independent copy


def test_nvcre_env_vars_with_hf_token_secret_enables_online_mode() -> None:
    """When an HF secret is configured, offline-mode flags must be overridden to allow pod access."""
    result = setup_experiment._nvcre_env_vars({}, hf_token_secret_name="hf-token-secret")
    assert result["HF_HUB_OFFLINE"] == "0"
    assert result["TRANSFORMERS_OFFLINE"] == "0"
    assert "HF_TOKEN" not in result  # token is injected via secret_env_vars, not here


def test_nvcre_env_vars_hf_token_secret_forces_online_mode() -> None:
    """HF secret must override any caller-supplied offline flag so the pod can reach HuggingFace."""
    custom = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    result = setup_experiment._nvcre_env_vars(custom, hf_token_secret_name="hf-token-secret")
    assert result["HF_HUB_OFFLINE"] == "0"
    assert result["TRANSFORMERS_OFFLINE"] == "0"
    assert "HF_TOKEN" not in result


def test_nvcre_env_vars_strips_plain_hf_token_when_secret_configured() -> None:
    """Plain HF_TOKEN from cluster_config environment: or --env flags must be stripped when a secret is set."""
    custom = {"HF_TOKEN": "hf_leaked_token", "NCCL_TIMEOUT": "1800"}
    result = setup_experiment._nvcre_env_vars(custom, hf_token_secret_name="hf-token-secret")
    assert "HF_TOKEN" not in result
    assert result["NCCL_TIMEOUT"] == "1800"
