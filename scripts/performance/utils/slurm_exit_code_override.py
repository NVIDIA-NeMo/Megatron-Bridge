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

"""
Slurm exit code override patch for handling false-positive srun failures.

TEMPORARY WORKAROUND (Remove in next release):
Some Slurm jobs report non-zero exit codes even when training completes successfully.
This module patches SlurmBatchRequest to inject log-based success validation into
the generated sbatch script.

How it works:
1. Monkey-patches SlurmBatchRequest.materialize() to use a custom template
2. Custom template checks logs for success patterns if exit code is non-zero
3. If success pattern found (iteration X/X where X==total iterations), overrides exit code to 0

Usage:
    Import this module before creating any SlurmExecutor:
    
    from .utils.slurm_exit_code_override import *  # Apply patch
    
    executor = slurm_executor(...)  # Will use patched behavior

Removal Plan:
    To remove this workaround in the next release:
    1. Delete this file (scripts/performance/utils/slurm_exit_code_override.py)
    2. Delete the template (scripts/performance/templates/slurm_log_success_check.sh.j2)
    3. Remove the import line from setup_experiment.py
"""

import os

import nemo_run.core.execution.slurm as slurm_module
from nemo_run.core.execution.slurm import SlurmBatchRequest

# Store original implementations for restoration
_ORIGINAL_MATERIALIZE = SlurmBatchRequest.materialize
_ORIGINAL_FILL_TEMPLATE = slurm_module.fill_template


def _custom_materialize(self) -> str:
    """
    Patched materialize that injects log validation variables and custom template.

    Returns:
        str: Generated sbatch script content with log validation logic
    """

    def _custom_fill_template(template_name, variables, template_dir=None):
        """Intercept slurm.sh.j2 template requests and redirect to our custom template."""
        if template_name == "slurm.sh.j2":
            # Inject variables needed for log validation
            variables["log_dir"] = self.executor.job_details.folder or self.executor.job_dir
            variables["job_name"] = self.executor.job_details.job_name or self.executor.job_name

            # Use custom template from this script's parent directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            custom_template_dir = os.path.join(parent_dir, "templates")
            return _ORIGINAL_FILL_TEMPLATE("slurm_log_success_check.sh.j2", variables, custom_template_dir)

        # Pass through all other template requests unchanged
        return _ORIGINAL_FILL_TEMPLATE(template_name, variables, template_dir)

    # Temporarily replace fill_template during materialize
    slurm_module.fill_template = _custom_fill_template
    try:
        result = _ORIGINAL_MATERIALIZE(self)
    finally:
        # Always restore original, even if materialize fails
        slurm_module.fill_template = _ORIGINAL_FILL_TEMPLATE

    return result


# Apply the monkey-patch at module import time
SlurmBatchRequest.materialize = _custom_materialize

print("[SLURM PATCH] Exit code override enabled - will validate logs for false-positive failures")

