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

"""Wiring test: parsed ``--csp runai`` + ``--runai_*`` args -> RunAIPlugin.

Guards against the class of bug where the launcher parses the ``--runai_*`` CLI
args but never threads them into the plugin, so they silently fall back to
defaults and ``--csp runai`` only gets default plugin behavior. We drive the
*real* argument parser (so the arg dest names are exactly what the launcher
forwards) and assert the resulting RunAIPlugin carries the parsed
resources / annotations / PVC / shm / env values.
"""

import sys
from pathlib import Path

import pytest


# scripts/performance uses runtime-style absolute imports (e.g.
# ``from utils.csp_plugins import ...``); put it on sys.path so the launcher
# modules import the same way they do when invoked as a script.
_PERF_DIR = Path(__file__).resolve().parents[3] / "scripts" / "performance"
if str(_PERF_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_DIR))

# The launcher imports nemo_run transitively; skip cleanly if it's unavailable.
pytest.importorskip("nemo_run")

from argument_parser import parse_cli_args  # noqa: E402
from setup_experiment import build_csp_plugin  # noqa: E402
from utils.csp_plugins import EKSEnvPlugin, GKEEnvPlugin, RunAIPlugin  # noqa: E402


# Minimal required args for the perf parser (model_family, model_recipe, num_gpus, gpu).
_REQUIRED_ARGV = [
    "--model_family_name",
    "llama",
    "--model_recipe_name",
    "llama31_70b",
    "--num_gpus",
    "64",
    "--gpu",
    "gb200",
]

_EXTENDED_RESOURCES = '{"nvidia.com/r0-p0": "1", "nvidia.com/r1-p0": "1"}'
_ANNOTATIONS = '{"k8s.v1.cni.cncf.io/networks": "rail0,rail1"}'
_ENV = '{"TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/nemo-workspace/hf"}'


def _parse(extra_argv):
    return parse_cli_args().parse_args(_REQUIRED_ARGV + extra_argv)


def test_runai_args_flow_into_plugin():
    """``--runai_*`` values must reach the RunAIPlugin, not be dropped to defaults."""
    args = _parse(
        [
            "--csp",
            "runai",
            "--runai_extended_resources_json",
            _EXTENDED_RESOURCES,
            "--runai_annotations_json",
            _ANNOTATIONS,
            "--runai_pvc_claim_name",
            "nemo-workspace",
            "--runai_pvc_mount_path",
            "/cm/shared/nemo-workspace",
            "--runai_large_shm",
            "false",
            "--runai_env_json",
            _ENV,
            "--runai_scheduler_name",
            "runai-scheduler",
            "--runai_labels_json",
            '{"project": "bench"}',
        ]
    )

    # Build the plugin exactly as the launcher does (same arg dest names).
    plugin = build_csp_plugin(
        args.csp,
        runai_extended_resources_json=args.runai_extended_resources_json,
        runai_annotations_json=args.runai_annotations_json,
        runai_pvc_claim_name=args.runai_pvc_claim_name,
        runai_pvc_mount_path=args.runai_pvc_mount_path,
        runai_large_shm=args.runai_large_shm,
        runai_env_json=args.runai_env_json,
        runai_scheduler_name=args.runai_scheduler_name,
        runai_labels_json=args.runai_labels_json,
    )

    assert isinstance(plugin, RunAIPlugin)
    assert plugin.extended_resources == {"nvidia.com/r0-p0": "1", "nvidia.com/r1-p0": "1"}
    assert plugin.annotations == {"k8s.v1.cni.cncf.io/networks": "rail0,rail1"}
    assert plugin.pvc_claim_name == "nemo-workspace"
    assert plugin.pvc_mount_path == "/cm/shared/nemo-workspace"
    assert plugin.large_shm is False
    assert plugin.env_vars == {"TRANSFORMERS_OFFLINE": "1", "HF_HOME": "/nemo-workspace/hf"}
    assert plugin.scheduler_name == "runai-scheduler"
    assert plugin.labels == {"project": "bench"}


def test_runai_defaults_when_args_omitted():
    """``--csp runai`` with no ``--runai_*`` args yields an empty-but-valid plugin."""
    args = _parse(["--csp", "runai"])
    plugin = build_csp_plugin(
        args.csp,
        runai_extended_resources_json=args.runai_extended_resources_json,
        runai_annotations_json=args.runai_annotations_json,
        runai_pvc_claim_name=args.runai_pvc_claim_name,
        runai_pvc_mount_path=args.runai_pvc_mount_path,
        runai_large_shm=args.runai_large_shm,
        runai_env_json=args.runai_env_json,
    )
    assert isinstance(plugin, RunAIPlugin)
    assert plugin.extended_resources == {}
    assert plugin.annotations == {}
    assert plugin.env_vars == {}
    assert plugin.pvc_claim_name is None


def test_other_csps_and_none():
    assert build_csp_plugin(None) is None
    assert isinstance(build_csp_plugin("aws"), EKSEnvPlugin)
    assert isinstance(build_csp_plugin("gcp"), GKEEnvPlugin)
