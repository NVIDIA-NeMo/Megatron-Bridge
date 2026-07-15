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

"""Login-node coverage for import-free performance recipe metadata."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"


def _run_on_isolated_login_node(source: str) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        sys.path.insert(0, {str(_PERF_SCRIPTS_DIR)!r})

        class BlockTrainingStack(importlib.abc.MetaPathFinder):
            blocked_roots = {{"megatron", "torch", "transformer_engine", "transformers"}}

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".", 1)[0] in self.blocked_roots:
                    raise AssertionError(f"blocked training-stack import: {{fullname}}")
                return None

        sys.meta_path.insert(0, BlockTrainingStack())

        {textwrap.indent(textwrap.dedent(source), "        ").lstrip()}
        """
    )
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", script],
        check=False,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_experiment_name_lookup_does_not_import_training_stack() -> None:
    result = _run_on_isolated_login_node(
        """
        from types import SimpleNamespace
        from utils.utils import get_exp_name_config

        args = SimpleNamespace(
            num_gpus=None,
            tensor_model_parallel_size=None,
            pipeline_model_parallel_size=None,
            context_parallel_size=None,
            virtual_pipeline_model_parallel_size=-1,
            expert_model_parallel_size=None,
            expert_tensor_parallel_size=None,
            micro_batch_size=None,
            global_batch_size=None,
        )
        llama_name = get_exp_name_config(
            args, "llama", "llama3_8b", "h100", "bf16", "pretrain"
        )
        assert llama_name == "gpus8_tp1_pp1_cp2_vpNone_ep1_etpNone_mbs1_gbs128"

        nemodiag_name = get_exp_name_config(
            args,
            "nemodiag",
            "nemodiag_v0",
            "gb300",
            "fp8_mx",
            "pretrain",
            "perf72_e144",
        )
        assert nemodiag_name == "gpus288_tp1_pp2_cp1_vp4_ep36_etp1_mbs1_gbs4608"
        print("ok", end="")
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"


def test_variant_display_does_not_import_training_stack() -> None:
    result = _run_on_isolated_login_node(
        """
        import contextlib
        import io
        import utils.utils as perf_utils

        perf_utils._get_user_selection_with_timeout = lambda *_: 2
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            selected = perf_utils.select_config_variant_interactive(
                model_family_name="deepseek",
                model_recipe_name="deepseek_v3",
                gpu="h100",
                compute_dtype="fp8_sc",
                task="pretrain",
                timeout=0,
                force_interactive=True,
            )

        rendered = output.getvalue()
        assert selected == "large_scale"
        assert "global_batch_size: 16384" in rendered
        assert "global_batch_size: 1024" in rendered
        assert "(config not found)" not in rendered
        print("ok", end="")
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"


def test_lightweight_metadata_covers_all_flat_perf_recipes() -> None:
    result = _run_on_isolated_login_node(
        """
        from utils.utils import flat_perf_recipe_names
        from utils.workload_metadata import WORKLOAD_BASE_CONFIGS

        numeric_gpu_recipes = {
            name
            for name in flat_perf_recipe_names()
            if any(
                part.removesuffix("gpu").isdigit()
                for part in name.split("_")
                if part.endswith("gpu")
            )
        }
        assert set(WORKLOAD_BASE_CONFIGS) == numeric_gpu_recipes
        print("ok", end="")
        """
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"
