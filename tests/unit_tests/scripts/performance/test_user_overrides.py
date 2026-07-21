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

"""Regression tests for performance-script user overrides."""

import sys
from pathlib import Path


_PERF_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "performance"
if str(_PERF_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_PERF_SCRIPTS_DIR))

from argument_parser import parse_cli_args
from utils.overrides import set_user_overrides

from megatron.bridge.recipes.gpt.h100.vanilla_gpt import vanilla_gpt_pretrain_1gpu_h100_bf16_config


def _parse_args(tmp_path: Path, *extra_args: str):
    parser = parse_cli_args()
    args, unknown = parser.parse_known_args(
        [
            "--model_family_name",
            "gpt",
            "--model_recipe_name",
            "vanilla_gpt",
            "--num_gpus",
            "1",
            "--gpu",
            "h100",
            "--save_config_filepath",
            str(tmp_path / "ConfigContainer.yaml"),
            *extra_args,
        ]
    )
    assert unknown == []
    return args


def test_seq_length_updates_model_and_mock_dataset(tmp_path):
    recipe = vanilla_gpt_pretrain_1gpu_h100_bf16_config()

    updated = set_user_overrides(recipe, _parse_args(tmp_path, "--seq_length", "128"))

    assert updated.model.seq_length == 128
    assert updated.dataset.seq_length == 128
