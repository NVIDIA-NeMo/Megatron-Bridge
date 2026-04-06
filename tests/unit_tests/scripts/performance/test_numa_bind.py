# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

# Adjust import path: the module lives under scripts/performance/utils/
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "scripts" / "performance"))

from utils.numa_bind import (
    _normalize_bdf,
    build_numactl_args,
    detect_all_numa_nodes,
    detect_numa_node,
    eval_binding_expr,
    load_override_file,
    parse_args,
    resolve_override_binding,
    validate_override_file,
)


# ---------------------------------------------------------------------------
# Expression evaluator tests
# ---------------------------------------------------------------------------


class TestEvalBindingExpr:
    def test_plain_integer(self):
        assert eval_binding_expr("0", rank=5) == "0"

    def test_plain_string(self):
        assert eval_binding_expr("3", rank=0) == "3"

    def test_simple_rank(self):
        assert eval_binding_expr("{rank}", rank=3) == "3"

    def test_floor_division(self):
        assert eval_binding_expr("{rank // 2}", rank=5) == "2"

    def test_multiplication(self):
        assert eval_binding_expr("{rank * 16}", rank=2) == "32"

    def test_addition(self):
        assert eval_binding_expr("{rank * 16 + 1}", rank=2) == "33"

    def test_modulo(self):
        assert eval_binding_expr("{rank % 4}", rank=7) == "3"

    def test_subtraction(self):
        assert eval_binding_expr("{rank - 1}", rank=3) == "2"

    def test_multiple_templates(self):
        result = eval_binding_expr("{rank * 16}, {rank * 16 + 1}", rank=2)
        assert result == "32, 33"

    def test_negative_unary(self):
        assert eval_binding_expr("{-rank}", rank=3) == "-3"

    def test_complex_expression(self):
        assert eval_binding_expr("{(rank + 1) * 4 // 2}", rank=3) == "8"

    def test_rejects_function_call(self):
        with pytest.raises(ValueError, match="Unsupported"):
            eval_binding_expr("{len('abc')}", rank=0)

    def test_rejects_attribute_access(self):
        with pytest.raises(ValueError, match="Unsupported"):
            eval_binding_expr("{rank.__class__}", rank=0)

    def test_rejects_import(self):
        with pytest.raises((ValueError, SyntaxError)):
            eval_binding_expr("{__import__('os')}", rank=0)

    def test_rejects_unknown_name(self):
        with pytest.raises(ValueError, match="Unsupported"):
            eval_binding_expr("{foo + 1}", rank=0)

    def test_rejects_string_literal(self):
        with pytest.raises(ValueError, match="Unsupported"):
            eval_binding_expr("{'hello'}", rank=0)

    def test_rejects_power_operator(self):
        with pytest.raises(ValueError, match="Unsupported operator"):
            eval_binding_expr("{rank ** 2}", rank=3)


# ---------------------------------------------------------------------------
# YAML override file tests
# ---------------------------------------------------------------------------


def _write_yaml(tmp_path, data):
    """Helper to write a YAML override file and return its path."""
    path = tmp_path / "override.yaml"
    path.write_text(yaml.dump(data))
    return str(path)


class TestResolveOverrideBinding:
    def test_template_only(self):
        data = {"binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}"}}
        result = resolve_override_binding(data, rank=3)
        assert result == {"cpunodebind": "1", "membind": "1"}

    def test_per_rank_override(self):
        data = {
            "binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}"},
            "ranks": {3: {"cpunodebind": "0", "membind": "0"}},
        }
        result = resolve_override_binding(data, rank=3)
        assert result == {"cpunodebind": "0", "membind": "0"}

    def test_per_rank_string_key(self):
        """YAML may parse integer keys as strings."""
        data = {
            "binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}"},
            "ranks": {"3": {"cpunodebind": "0", "membind": "0"}},
        }
        result = resolve_override_binding(data, rank=3)
        assert result == {"cpunodebind": "0", "membind": "0"}

    def test_per_rank_partial_override(self):
        data = {
            "binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}"},
            "ranks": {3: {"physcpubind": "0,1"}},
        }
        result = resolve_override_binding(data, rank=3)
        assert result == {"cpunodebind": "1", "membind": "1", "physcpubind": "0,1"}

    def test_no_binding_section(self):
        data = {"ranks": {0: {"cpunodebind": "0", "membind": "0"}}}
        result = resolve_override_binding(data, rank=0)
        assert result == {"cpunodebind": "0", "membind": "0"}

    def test_rank_not_in_overrides_uses_template(self):
        data = {
            "binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}"},
            "ranks": {0: {"cpunodebind": "99"}},
        }
        result = resolve_override_binding(data, rank=5)
        assert result == {"cpunodebind": "2", "membind": "2"}

    def test_plain_values(self):
        data = {"binding": {"cpunodebind": "0", "membind": "0"}}
        result = resolve_override_binding(data, rank=7)
        assert result == {"cpunodebind": "0", "membind": "0"}

    def test_physcpubind_template(self):
        data = {"binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}", "physcpubind": "{rank * 16}, {rank * 16 + 1}"}}
        result = resolve_override_binding(data, rank=2)
        assert result == {"cpunodebind": "1", "membind": "1", "physcpubind": "32, 33"}

    def test_invalid_key_in_binding(self):
        data = {"binding": {"invalid_key": "0"}}
        with pytest.raises(ValueError, match="Unknown binding key"):
            resolve_override_binding(data, rank=0)

    def test_invalid_key_in_ranks(self):
        data = {"ranks": {0: {"invalid_key": "0"}}}
        with pytest.raises(ValueError, match="Unknown binding key"):
            resolve_override_binding(data, rank=0)


class TestValidateOverrideFile:
    def test_valid_file(self, tmp_path):
        path = _write_yaml(tmp_path, {
            "binding": {"cpunodebind": "{rank // 2}", "membind": "{rank // 2}"},
            "ranks": {0: {"cpunodebind": "0"}},
        })
        validate_override_file(path)  # should not raise

    def test_invalid_expression(self, tmp_path):
        path = _write_yaml(tmp_path, {"binding": {"cpunodebind": "{rank ** 2}"}})
        with pytest.raises(ValueError, match="Invalid expression"):
            validate_override_file(path)

    def test_invalid_binding_key(self, tmp_path):
        path = _write_yaml(tmp_path, {"binding": {"bad_key": "0"}})
        with pytest.raises(ValueError, match="Unknown binding key"):
            validate_override_file(path)

    def test_unknown_top_level_key(self, tmp_path):
        path = _write_yaml(tmp_path, {"binding": {}, "extra_stuff": {}})
        with pytest.raises(ValueError, match="Unknown top-level keys"):
            validate_override_file(path)

    def test_not_a_mapping(self, tmp_path):
        path = tmp_path / "override.yaml"
        path.write_text("- just a list\n")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            validate_override_file(str(path))

    def test_binding_not_a_mapping(self, tmp_path):
        path = _write_yaml(tmp_path, {"binding": "not a dict"})
        with pytest.raises(ValueError, match="'binding' must be a mapping"):
            validate_override_file(path)

    def test_ranks_not_a_mapping(self, tmp_path):
        path = _write_yaml(tmp_path, {"ranks": "not a dict"})
        with pytest.raises(ValueError, match="'ranks' must be a mapping"):
            validate_override_file(path)


# ---------------------------------------------------------------------------
# Auto-detection tests
# ---------------------------------------------------------------------------


class TestNormalizeBdf:
    def test_4_digit_domain_unchanged(self):
        assert _normalize_bdf("0000:07:00.0") == "0000:07:00.0"

    def test_8_digit_domain_trimmed(self):
        assert _normalize_bdf("00000000:19:00.0") == "0000:19:00.0"

    def test_uppercase_lowered(self):
        assert _normalize_bdf("00000000:AE:00.0") == "0000:ae:00.0"

    def test_whitespace_stripped(self):
        assert _normalize_bdf("  0000:07:00.0\n") == "0000:07:00.0"


class TestDetectNumaNode:
    def test_success(self):
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "00000000:07:00.0\n"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            with mock.patch("utils.numa_bind.Path.exists", return_value=True):
                with mock.patch("utils.numa_bind.Path.read_text", return_value="1\n"):
                    node = detect_numa_node(0)
        assert node == 1

    def test_nvidia_smi_failure(self):
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            node = detect_numa_node(0)
        assert node is None

    def test_sysfs_not_found(self):
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0000:07:00.0\n"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            with mock.patch("utils.numa_bind.Path.exists", return_value=False):
                node = detect_numa_node(0)
        assert node is None

    def test_numa_node_negative_one(self):
        """sysfs returns -1 when NUMA info is unavailable (e.g., virtualized envs)."""
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0000:07:00.0\n"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            with mock.patch("utils.numa_bind.Path.exists", return_value=True):
                with mock.patch("utils.numa_bind.Path.read_text", return_value="-1\n"):
                    node = detect_numa_node(0)
        assert node is None

    def test_timeout(self):
        with mock.patch(
            "utils.numa_bind.subprocess.run",
            side_effect=subprocess.TimeoutExpired("nvidia-smi", 10),
        ):
            node = detect_numa_node(0)
        assert node is None

    def test_empty_bdf(self):
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "\n"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            node = detect_numa_node(0)
        assert node is None


class TestDetectAllNumaNodes:
    def test_multiple_gpus(self):
        mock_result = mock.MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "00000000:19:00.0\n00000000:2D:00.0\n00000000:3F:00.0\n00000000:66:00.0\n"

        def fake_exists(self):
            return True

        def fake_read_text(self):
            # Map BDFs to NUMA nodes: first two -> node 0, last two -> node 1
            bdf = str(self).split("/")[-2]
            node_map = {"0000:19:00.0": "0", "0000:2d:00.0": "0", "0000:3f:00.0": "1", "0000:66:00.0": "1"}
            return node_map.get(bdf, "-1") + "\n"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            with mock.patch("utils.numa_bind.Path.exists", fake_exists):
                with mock.patch("utils.numa_bind.Path.read_text", fake_read_text):
                    nodes = detect_all_numa_nodes()
        assert nodes == [0, 0, 1, 1]

    def test_nvidia_smi_failure(self):
        mock_result = mock.MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "error"

        with mock.patch("utils.numa_bind.subprocess.run", return_value=mock_result):
            nodes = detect_all_numa_nodes()
        assert nodes is None


# ---------------------------------------------------------------------------
# build_numactl_args tests
# ---------------------------------------------------------------------------


class TestBuildNumactlArgs:
    def test_cpunodebind_and_membind(self):
        args = build_numactl_args({"cpunodebind": "1", "membind": "1"})
        assert args == ["--cpunodebind", "1", "--membind", "1"]

    def test_with_physcpubind(self):
        args = build_numactl_args({"cpunodebind": "0", "membind": "0", "physcpubind": "0,1"})
        assert args == ["--cpunodebind", "0", "--membind", "0", "--physcpubind", "0,1"]

    def test_empty_binding(self):
        args = build_numactl_args({})
        assert args == []

    def test_physcpubind_only(self):
        args = build_numactl_args({"physcpubind": "0,1"})
        assert args == ["--physcpubind", "0,1"]


# ---------------------------------------------------------------------------
# CLI arg parsing tests
# ---------------------------------------------------------------------------


class TestParseArgs:
    def test_auto_mode(self):
        args = parse_args(["--mode", "auto", "--", "torchrun", "train.py"])
        assert args.mode == "auto"
        assert args.command == ["torchrun", "train.py"]
        assert args.override_file is None
        assert args.hard_fail is False

    def test_override_mode(self):
        args = parse_args(["--mode", "override", "--override-file", "/tmp/t.yaml", "--", "cmd"])
        assert args.mode == "override"
        assert args.override_file == "/tmp/t.yaml"
        assert args.command == ["cmd"]

    def test_off_mode(self):
        args = parse_args(["--mode", "off", "--", "cmd"])
        assert args.mode == "off"

    def test_hard_fail(self):
        args = parse_args(["--mode", "auto", "--hard-fail", "--", "cmd"])
        assert args.hard_fail is True

    def test_no_command(self):
        args = parse_args(["--mode", "auto"])
        assert args.command == []


# ---------------------------------------------------------------------------
# Integration: executor NUMA wrapper generation
# ---------------------------------------------------------------------------


class TestExecutorNumaIntegration:
    """Verify that slurm_executor generates the correct numa_bind.py wrapper command."""

    def _get_pre_cmds(self, **kwargs):
        """Extract the pre_cmds from the launcher template vars."""
        # Import here to avoid import errors when nemo_run isn't available
        from utils.executors import slurm_executor

        with mock.patch("utils.executors.set_nemorun_home"):
            with mock.patch("utils.executors.get_nemorun_home", return_value="/tmp/test"):
                executor = slurm_executor(
                    gpu="h100",
                    account="test",
                    partition="test",
                    log_dir="/tmp/test",
                    nodes=1,
                    num_gpus_per_node=8,
                    **kwargs,
                )
        return executor.launcher.template_vars["pre_cmds"]

    def test_auto_mode_includes_wrapper(self):
        pre_cmds = self._get_pre_cmds(numa_mode="auto")
        assert "numa_bind.py" in pre_cmds
        assert "--mode auto" in pre_cmds

    def test_off_mode_no_wrapper(self):
        pre_cmds = self._get_pre_cmds(numa_mode="off")
        assert "numa_bind.py" not in pre_cmds
        assert "numactl" not in pre_cmds

    def test_override_mode_includes_file(self):
        pre_cmds = self._get_pre_cmds(
            numa_mode="override", numa_override_file="/path/to/override.yaml"
        )
        assert "numa_bind.py" in pre_cmds
        assert "--mode override" in pre_cmds
        assert "--override-file /path/to/override.yaml" in pre_cmds

    def test_no_hardcoded_numa_divisor(self):
        """Ensure the old hardcoded NUMA divisor logic is gone."""
        for gpu in ["h100", "gb200", "gb300", "b200", "b300"]:
            pre_cmds = self._get_pre_cmds(numa_mode="auto")
            assert "SLURM_LOCALID/" not in pre_cmds
