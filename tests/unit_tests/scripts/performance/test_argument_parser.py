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

"""Tests for the performance launcher's GPU argument."""

import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.unit

_ARGUMENT_PARSER = Path(__file__).resolve().parents[4] / "scripts" / "performance" / "argument_parser.py"
_REQUIRED_ARGS = ["--model_family_name", "llama", "--model_recipe_name", "llama3_8b", "--num_gpus", "8"]


def _load_argument_parser():
    """Load argument_parser.py with a stub nemo_run, which it uses only for a help-text default."""
    nemo_run = types.ModuleType("nemo_run")
    nemo_run_config = types.ModuleType("nemo_run.config")
    nemo_run_config.get_nemorun_home = lambda: "/nemo_run"
    nemo_run.config = nemo_run_config
    spec = importlib.util.spec_from_file_location("test_performance_argument_parser", _ARGUMENT_PARSER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"nemo_run": nemo_run, "nemo_run.config": nemo_run_config}):
        spec.loader.exec_module(module)
    return module


def test_deprecated_gpu_name_maps_to_vrnvl72_with_warning(caplog, monkeypatch):
    argument_parser = _load_argument_parser()
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)

    with caplog.at_level(logging.WARNING):
        args = argument_parser.parse_cli_args().parse_args([*_REQUIRED_ARGS, "--gpu", "vr200"])

    assert args.gpu == "vrnvl72"
    assert "--gpu vr200 is deprecated; use --gpu vrnvl72." in caplog.text


@pytest.mark.parametrize("rank_variable", ["RANK", "SLURM_PROCID"])
def test_deprecated_gpu_name_maps_without_warning_on_nonzero_ranks(caplog, monkeypatch, rank_variable):
    argument_parser = _load_argument_parser()
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("SLURM_PROCID", raising=False)
    monkeypatch.setenv(rank_variable, "3")

    with caplog.at_level(logging.WARNING):
        args = argument_parser.parse_cli_args().parse_args([*_REQUIRED_ARGS, "--gpu", "vr200"])

    assert args.gpu == "vrnvl72"
    assert "deprecated" not in caplog.text


def test_current_gpu_name_is_unchanged_without_warning(caplog):
    argument_parser = _load_argument_parser()

    with caplog.at_level(logging.WARNING):
        args = argument_parser.parse_cli_args().parse_args([*_REQUIRED_ARGS, "--gpu", "vrnvl72"])

    assert args.gpu == "vrnvl72"
    assert "deprecated" not in caplog.text
    assert argument_parser.NUM_GPUS_PER_NODE_MAP[args.gpu] == 4


def test_only_current_gpu_names_are_choices():
    argument_parser = _load_argument_parser()

    assert "vr200" not in argument_parser.NUM_GPUS_PER_NODE_MAP
    with pytest.raises(SystemExit):
        argument_parser.parse_cli_args().parse_args([*_REQUIRED_ARGS, "--gpu", "vr2000"])
