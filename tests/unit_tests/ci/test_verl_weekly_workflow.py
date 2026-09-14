# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path


WORKFLOW = Path(__file__).parents[3] / ".github" / "workflows" / "verl-e2e-weekly.yml"
RUN_SCRIPT = 'bash "$SCRIPT_PATH"'
UV_INSTALL = "python3 -m pip install uv"
UV_LAUNCH = "uv run --frozen --all-packages"


def test_verl_weekly_installs_uv_before_running_scripts():
    """Keep the external runner dependency available before executing mutable verl scripts."""
    workflow = WORKFLOW.read_text()
    start = workflow.index("git clone https://github.com/verl-project/verl.git")
    end = workflow.index(RUN_SCRIPT, start) + len(RUN_SCRIPT)
    run_block = workflow[start:end]

    assert run_block.count(UV_INSTALL) == 1
    assert run_block.index(UV_INSTALL) < run_block.index(RUN_SCRIPT)


def test_recorded_verl_runner_requires_uv():
    """Retain the failure mechanism as a fixture for the workflow setup contract."""
    assert UV_LAUNCH in (Path(__file__).with_name("fixtures") / "verl_ppo_launcher.sh").read_text()
