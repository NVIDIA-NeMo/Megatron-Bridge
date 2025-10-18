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
import time
from typing import Callable

import torch

from megatron.bridge.training.config import NVRxStragglerDetectionConfig
from megatron.bridge.utils.import_utils import MISSING_NVRX_MSG


try:
    # Try the newer version first (nvidia-resiliency-ext >= 0.4)
    import nvidia_resiliency_ext.attribution.straggler as straggler

    HAVE_NVRX = True
except (ImportError, ModuleNotFoundError):
    try:
        # Fall back to the older version (nvidia-resiliency-ext < 0.4)
        import nvidia_resiliency_ext.straggler as straggler

        HAVE_NVRX = True
    except (ImportError, ModuleNotFoundError):
        HAVE_NVRX = False


class NVRxStragglerDetectionManager:
    """Manager for NVIDIA Resiliency Extension straggler detection."""

    def __init__(self, config: NVRxStragglerDetectionConfig):
        """
        Initialize the NVRx straggler detection manager.

        Args:
            config: Configuration for NVRx straggler detection.

        Raises:
            ImportError: If nvidia-resiliency-ext is not available.
            ValueError: If invalid configuration is provided.
        """
        if not HAVE_NVRX:
            raise ImportError(MISSING_NVRX_MSG)
        self.config = config
        self.logger = logging.getLogger(config.logger_name)
        self.initialized = False
        self.wrapped_function = None
        self.scores_to_compute = []

        if config.calc_relative_gpu_perf:
            self.scores_to_compute.append("relative_perf_scores")
        if config.calc_individual_gpu_perf:
            self.scores_to_compute.append("individual_perf_scores")

    def initialize(self) -> None:
        """
        Initialize the straggler detector.

        Raises:
            RuntimeError: If already initialized.
        """
        if self.initialized:
            raise RuntimeError("NVRxStragglerDetectionManager is already initialized.")

        if not self.config.enabled:
            self.logger.debug("NVRx straggler detection is disabled.")
            return

        self.logger.debug("Initializing NVRx straggler detection...")

        straggler.Detector.initialize(
            scores_to_compute=self.scores_to_compute,
            gather_on_rank0=True,
            profiling_interval=self.config.profiling_interval,
            report_time_interval=self.config.report_time_interval,
        )

        self.initialized = True
        self.logger.debug("NVRx straggler detection initialized successfully.")

    def wrap_train_step_function(self, train_step_func: Callable) -> Callable:
        """
        Wrap the training step function with straggler detection monitoring.

        Args:
            train_step_func: The actual training step function to wrap for monitoring.

        Returns:
            The wrapped training step function.
        """

        if not self.initialized or not self.config.enabled:
            return train_step_func

        if self.wrapped_function is not None:
            self.logger.warning("Train step function already wrapped. Skipping.")
            return train_step_func

        try:
            # Create a wrapper object with train_step method for nvidia-resiliency-ext
            # TODO: See if NVRx can support functions directly without needing them attached to a class
            class TrainStepWrapper:
                def __init__(self, func):
                    self.train_step = func
                    self._original_func = func

                def __call__(self, *args, **kwargs):
                    return self._original_func(*args, **kwargs)

            wrapper_obj = TrainStepWrapper(train_step_func)

            # Create a callable ID for the training step function
            callable_id = straggler.CallableId(wrapper_obj, "train_step")
            straggler.Detector.wrap_callables(callable_ids=[callable_id])

            self.wrapped_function = train_step_func
            self.logger.debug("Train step function wrapped for NVRx straggler detection.")

            # Return the original function since the wrapper is just for nvidia-resiliency-ext
            return train_step_func

        except Exception as e:
            self.logger.warning(f"Failed to wrap train step function with NVRx: {e}. Continuing without wrapping.")
            return train_step_func

    def check_stragglers(self, global_rank: int) -> bool:
        """
        Check for stragglers and handle reporting.

        Args:
            global_rank: The global rank of the current process.

        Returns:
            True if stragglers were detected and stop_if_detected is True, False otherwise.
        """

        if not self.initialized or not self.config.enabled:
            return False

        time_started = time.monotonic()
        report = straggler.Detector.generate_report_if_interval_elapsed()
        stragglers_found = False

        if global_rank == 0 and report:
            # Only rank 0 processes the report since gather_on_rank0=True
            stragglers_found = self._handle_straggler_report(report)

        # Check if the report was generated and broadcast straggler detection result
        if straggler.Detector.is_interval_elapsed():
            if self.config.stop_if_detected:
                stragglers_found = self._gather_flag_from_rank0(stragglers_found)
                if stragglers_found:
                    self.logger.error("Detected stragglers. Training should be stopped.")
                    return True

            # Log reporting time
            elapsed = time.monotonic() - time_started
            self.logger.debug(f"Straggler report processing time: {elapsed:.3f} sec.")

        return False

    def _handle_straggler_report(self, report) -> bool:
        """
        Handle the straggler report from the detector.

        Args:
            report: The straggler detection report.

        Returns:
            True if stragglers were found, False otherwise.
        """
        stragglers = report.identify_stragglers(
            gpu_rel_threshold=self.config.gpu_relative_perf_threshold,
            gpu_indiv_threshold=self.config.gpu_individual_perf_threshold,
        )

        stragglers_found = stragglers["straggler_gpus_relative"] or stragglers["straggler_gpus_individual"]

        if stragglers_found:
            self._print_stragglers(stragglers)

        if self.config.num_gpu_perf_scores_to_print > 0:
            self._print_gpu_scores(report)

        if self.config.enable_logging:
            self._log_gpu_scores(report)

        return stragglers_found

    def _print_stragglers(self, stragglers) -> None:
        """Print straggler detection warnings."""
        if rel_stragglers := stragglers["straggler_gpus_relative"]:
            self.logger.warning(
                f"STRAGGLER DETECTION WARNING: Some GPUs have worse relative performance. "
                f"Affected ranks: {rel_stragglers}"
            )
        if indiv_stragglers := stragglers["straggler_gpus_individual"]:
            self.logger.warning(
                f"STRAGGLER DETECTION WARNING: Some GPUs performance dropped. Affected ranks: {indiv_stragglers}"
            )

    @staticmethod
    def _format_gpu_scores(rank_to_score, rank_to_node, num_best=3, num_worst=3) -> str:
        """Format GPU performance scores for logging."""
        num_ranks = len(rank_to_score)
        scores_and_ranks = [(s, r) for r, s in rank_to_score.items()]
        scores_and_ranks.sort(reverse=True)
        res = ""

        if num_ranks > (num_best + num_worst):
            res += f" Worst performing {num_worst}/{num_ranks} ranks:\n"
            for s, r in reversed(scores_and_ranks[-num_worst:]):
                res += f"  Rank={r} Node={rank_to_node[r]} Score={s:.2f}\n"
            res += f" Best performing {num_best}/{num_ranks} ranks:\n"
            for s, r in scores_and_ranks[:num_best]:
                res += f"  Rank={r} Node={rank_to_node[r]} Score={s:.2f}\n"
        else:
            # If the number of ranks is small enough, print them all
            for s, r in reversed(scores_and_ranks):
                res += f"  Rank={r} Node={rank_to_node[r]} Score={s:.2f}\n"

        return res

    def _print_gpu_scores(self, report) -> None:
        """Print GPU performance scores."""
        if self.config.calc_relative_gpu_perf:
            rel_perf_str = self._format_gpu_scores(
                report.gpu_relative_perf_scores,
                report.rank_to_node,
                num_best=self.config.num_gpu_perf_scores_to_print,
                num_worst=self.config.num_gpu_perf_scores_to_print,
            )
            self.logger.info(f"\nGPU relative performance:\n{rel_perf_str}")

        if self.config.calc_individual_gpu_perf:
            indiv_perf_str = self._format_gpu_scores(
                report.gpu_individual_perf_scores,
                report.rank_to_node,
                num_best=self.config.num_gpu_perf_scores_to_print,
                num_worst=self.config.num_gpu_perf_scores_to_print,
            )
            self.logger.info(f"\nGPU individual performance:\n{indiv_perf_str}")

    def _log_gpu_scores(self, report) -> None:
        """Log GPU performance scores as structured data."""
        if self.config.calc_relative_gpu_perf:
            self._log_gpu_perf_scores(
                rank_to_score=report.gpu_relative_perf_scores,
                rank_to_node=report.rank_to_node,
                score_prefix="gpu_relative_perf",
            )

        if self.config.calc_individual_gpu_perf:
            self._log_gpu_perf_scores(
                rank_to_score=report.gpu_individual_perf_scores,
                rank_to_node=report.rank_to_node,
                score_prefix="gpu_individual_perf",
            )

    def _log_gpu_perf_scores(self, rank_to_score, rank_to_node, score_prefix) -> None:
        """Log GPU performance scores with statistics."""
        scores_log = {}
        min_val = float("nan")
        med_val = float("nan")
        max_val = float("nan")

        scores = list(rank_to_score.values())
        if scores:
            scores = torch.tensor(scores, dtype=torch.float32)
            min_val = torch.min(scores).item()
            med_val = torch.median(scores).item()
            max_val = torch.max(scores).item()

        scores_log[f"{score_prefix}/min"] = min_val
        scores_log[f"{score_prefix}/median"] = med_val
        scores_log[f"{score_prefix}/max"] = max_val

        # Log the structured data
        self.logger.info(f"{score_prefix} scores: {scores_log}")

    def _gather_flag_from_rank0(self, flag: bool) -> bool:
        """Broadcast a boolean flag from rank 0 to all ranks."""
        flag_tensor = torch.tensor([1.0 if flag else 0.0], device=torch.cuda.current_device(), dtype=torch.float32)
        torch.distributed.broadcast(flag_tensor, 0)
        return bool(flag_tensor.item() > 0.0)

    def shutdown(self) -> None:
        """Shutdown the straggler detector."""
        if self.initialized and self.config.enabled:
            self.logger.info("Shutting down NVRx straggler detection...")
            straggler.Detector.shutdown()
            self.initialized = False
            self.wrapped_function = None
            self.logger.info("NVRx straggler detection shutdown complete.")


def check_nvrx_straggler_detection(nvrx_straggler_manager: "NVRxStragglerDetectionManager" | None) -> bool:
    """
    Check NVRx straggler detection and determine if training should exit.

    Args:
        nvrx_straggler_manager: The NVRx straggler detection manager, or None if disabled.

    Returns:
        bool: True if stragglers were detected and training should exit, False otherwise.
    """
    if nvrx_straggler_manager is None:
        return False
    global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    should_exit = nvrx_straggler_manager.check_stragglers(global_rank)
    return should_exit


def safe_shutdown_nvrx_straggler_manager(
    manager: "NVRxStragglerDetectionManager" | None, logger_name: str = "nvrx_straggler"
) -> None:
    """
    Safely shutdown the NVRx straggler detection manager with error handling.

    Args:
        manager: The NVRx straggler detection manager to shutdown, can be None.
        logger_name: Logger name for error reporting.
    """
    if manager is not None:
        logger = logging.getLogger(logger_name)
        try:
            manager.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down NVRx straggler detection: {e}")
