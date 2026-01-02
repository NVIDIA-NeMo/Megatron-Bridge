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
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import nemo_run as run
from nemo_run import Plugin


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class FaultTolerancePluginScriptArgs:
    """Arguments for FaultTolerancePlugin to pass to run.Script."""

    enable_ft_package: bool
    calc_ft_timeouts: bool


def _default_fault_tolerance_converter(args: FaultTolerancePluginScriptArgs) -> List[str]:
    """Default converter for FaultTolerancePlugin that generates hydra-style overrides."""
    return [
        f"ft.enable_ft_package={str(args.enable_ft_package).lower()}",
        f"ft.calc_ft_timeouts={str(args.calc_ft_timeouts).lower()}",
    ]


@dataclass(kw_only=True)
class FaultTolerancePlugin(Plugin):
    """
    A plugin for setting up fault tolerance configuration.
    This plugin enables workload hang detection, automatic calculation of timeouts used for hang detection,
    detection of rank(s) terminated due to an error and workload respawning in case of a failure.


    Args:
        enable_ft_package (bool): Enable the fault tolerance package. Default is True.
        calc_ft_timeouts (bool): Automatically compute timeouts. Default is True.
        num_in_job_restarts (int): Max number of restarts on failure, within the same job. Default is 3.
        num_job_retries_on_failure (int): Max number of new job restarts on failure. Default is 2.
        initial_rank_heartbeat_timeout (int): Timeouts are time intervals used by a rank monitor to detect
            that a rank is not alive. This is the max timeout for the initial heartbeat. Default is 1800.
        rank_heartbeat_timeout (int): This is the timeout for subsequent hearbeats after the initial heartbeat.
            Default is 300.
        script_args_converter_fn (Optional[Callable]): A function that takes FaultTolerancePluginScriptArgs
                                                        and returns a list of CLI arguments. If not provided,
                                                        uses the default hydra-style converter.

    Note:
        This plugin is incompatible with NsysPlugin. Nsys profiling cannot be used when fault tolerance
        is enabled.
    """

    enable_ft_package: bool = True
    calc_ft_timeouts: bool = True
    num_in_job_restarts: int = 3
    num_job_retries_on_failure: int = 2
    initial_rank_heartbeat_timeout: int = 1800
    rank_heartbeat_timeout: int = 300
    script_args_converter_fn: Optional[Callable[[FaultTolerancePluginScriptArgs], List[str]]] = None

    def setup(self, task: Union["run.Partial", "run.Script"], executor: "run.Executor"):
        """Set up the fault tolerance plugin."""
        # Set up fault tolerance launcher for both task types
        executor.launcher = run.FaultTolerance(
            max_restarts=self.num_in_job_restarts,
            initial_rank_heartbeat_timeout=self.initial_rank_heartbeat_timeout,
            rank_heartbeat_timeout=self.rank_heartbeat_timeout,
        )
        executor.retries = self.num_job_retries_on_failure

        if isinstance(task, run.Script):
            # For run.Script, append CLI overrides to the script arguments
            # Create args dataclass
            script_args = FaultTolerancePluginScriptArgs(
                enable_ft_package=self.enable_ft_package,
                calc_ft_timeouts=self.calc_ft_timeouts,
            )

            # Use custom converter or default
            converter = self.script_args_converter_fn or _default_fault_tolerance_converter
            cli_overrides = converter(script_args)

            task.args.extend(cli_overrides)
            logger.info(f"{self.__class__.__name__} added CLI overrides: {', '.join(cli_overrides)}")
        else:
            raise NotImplementedError("FaultTolerancePlugin is only supported for run.Script tasks")
