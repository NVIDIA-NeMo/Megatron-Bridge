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

"""Login-node import tests for scripts/performance/setup_experiment.py."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"


@pytest.mark.skipif(importlib.util.find_spec("nemo_run") is None, reason="nemo_run not installed")
def test_setup_experiment_import_and_metadata_lookup_do_not_require_training_stack() -> None:
    """The submit-side launcher must not import Megatron Bridge, MCore, or Transformers."""
    script = textwrap.dedent(
        f"""
        import builtins
        import sys
        from types import SimpleNamespace

        sys.path.insert(0, {str(_PERF_SCRIPTS_DIR)!r})

        blocked_roots = {{"megatron", "transformers"}}
        original_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.split(".", 1)[0] in blocked_roots:
                raise AssertionError(f"blocked training-stack import: {{name}}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import

        import setup_experiment  # noqa: F401
        from utils.utils import get_exp_name_config

        args = SimpleNamespace(
            num_gpus=8,
            tensor_model_parallel_size=None,
            pipeline_model_parallel_size=None,
            context_parallel_size=None,
            virtual_pipeline_model_parallel_size=-1,
            expert_model_parallel_size=None,
            expert_tensor_parallel_size=None,
            micro_batch_size=None,
            global_batch_size=None,
        )
        value = get_exp_name_config(args, "llama", "llama3_8b", "h100", "bf16", "pretrain", "v2")
        assert value == "gpus8_tp1_pp1_cp2_vpNone_ep1_etpNone_mbs1_gbs128"
        sys.stdout.write("ok")
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PERF_SCRIPTS_DIR)
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"
