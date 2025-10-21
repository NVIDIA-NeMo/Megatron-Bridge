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
import time
import types
from dataclasses import dataclass
from typing import Any

import torch
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core.energy_monitor import EnergyMonitor
from megatron.core.timers import Timers
from megatron.core.utils import StragglerDetector
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.tensorboard.writer import SummaryWriter

from megatron.bridge.training.config import ConfigContainer
from megatron.bridge.training.nvrx_straggler import NVRxStragglerDetectionManager
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe


@dataclass
class TrainState(Stateful):
    """Dataclass to hold the state of the training process.

    Inherits from Stateful for distributed checkpointing compatibility.
    Tracks iteration count, consumed samples, flags for train/valid/test phases,
    and floating-point operations.
    """

    step: int = 0
    consumed_train_samples: int = 0
    skipped_train_samples: int = 0
    consumed_valid_samples: int = 0
    floating_point_operations_so_far: int = 0
    do_train: bool = False
    do_valid: bool = False
    do_test: bool = False

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serializes the training state into a dictionary of tensors.

        Conforms to the Stateful interface for distributed checkpointing.

        Returns:
            A dictionary where keys are state variable names and values are
            their corresponding tensor representations.
        """
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "consumed_train_samples": torch.tensor(self.consumed_train_samples, dtype=torch.int32),
            "skipped_train_samples": torch.tensor(self.skipped_train_samples, dtype=torch.int32),
            "consumed_valid_samples": torch.tensor(self.consumed_valid_samples, dtype=torch.int32),
            "floating_point_operations_so_far": torch.tensor(
                self.floating_point_operations_so_far, dtype=torch.float64
            ),
            "do_train": torch.tensor(self.do_train, dtype=torch.bool),
            "do_valid": torch.tensor(self.do_valid, dtype=torch.bool),
            "do_test": torch.tensor(self.do_test, dtype=torch.bool),
        }

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load the training state from a state dictionary.

        Args:
            state_dict: A dictionary containing the state variables as tensors.
        """
        self.step = state_dict["step"].item()
        self.consumed_train_samples = state_dict["consumed_train_samples"].item()
        self.skipped_train_samples = state_dict["skipped_train_samples"].item()
        self.consumed_valid_samples = state_dict["consumed_valid_samples"].item()
        self.floating_point_operations_so_far = state_dict["floating_point_operations_so_far"].item()
        self.do_train = state_dict["do_train"].item()
        self.do_valid = state_dict["do_valid"].item()
        self.do_test = state_dict["do_test"].item()


@dataclass
class FaultToleranceState:
    """Dataclass to hold state specific to fault tolerance mechanisms."""

    ft_state_path: str | None = None
    is_persistent_chkpt_loaded: bool = False
    is_async_chkpt_enabled: bool = False
    is_calculating_timeouts: bool = False
    is_setup_section_open: bool = False
    seen_checkpoints_cnt: int = 0
    seen_tr_iters_cnt: int = 0
    curr_eval_iter_idx: int = 0


# replacement for Megatron's global variables, except mbs calc and parallel state
class GlobalState:
    """Manages the global state of the training process.

    Provides access to configuration, tokenizer, loggers, timers,
    training state, fault tolerance state, signal handler, and straggler detector
    through properties with lazy initialization.
    """

    def __init__(self) -> None:
        """Initializes the GlobalState object."""
        # Prevent reinitialization in subsequent instantiations.
        if getattr(self, "_initialized", False):
            return
        self._initialized = True

        self._cfg: ConfigContainer | None = None
        self._tokenizer: Any | None = None
        self._tensorboard_logger: SummaryWriter | None = None
        self._wandb_logger: Any | None = None
        self._timers: Timers | None = None
        self._train_state: TrainState | None = None
        self.rank_monitor_client: Any | None = None
        self._signal_handler: DistributedSignalHandler | None = None
        self.start_time: float = time.time()
        self._ft_state: FaultToleranceState | None = None
        self._straggler_timer: StragglerDetector | None = None
        self._async_calls_queue: AsyncCallsQueue | None = None
        self._nvrx_straggler_manager: NVRxStragglerDetectionManager | None = None
        self._nvrx_straggler_created: bool = False
        self._energy_monitor: EnergyMonitor | None = None
        self._energy_monitor_created: bool = False

    @property
    def cfg(self) -> ConfigContainer | None:
        """The main configuration container object."""
        return self._cfg

    @cfg.setter
    def cfg(self, value: ConfigContainer | None) -> None:
        """Sets the configuration container and initializes the signal handler.

        Args:
            value: The ConfigContainer instance to set.
        """
        self._cfg = value

        # This lazily initializes the signal handler when the config is set
        # in order to read the exit signal from the config.
        # This assumes the global state is first initialized and that the
        # config is immediately set on the global state after initialization.
        if value is not None:
            self._set_signal_handler()

    @property
    def tokenizer(self) -> Any:
        """The tokenizer instance, lazily built based on the config."""
        if self._tokenizer is None:
            self._tokenizer = build_tokenizer(self.cfg.tokenizer)
        return self._tokenizer

    @property
    def tensorboard_logger(self) -> SummaryWriter | None:
        """The TensorBoard SummaryWriter instance, lazily initialized for rank N-1."""
        if self._tensorboard_logger is None:
            if self.cfg.logger.tensorboard_dir and get_rank_safe() == (get_world_size_safe() - 1):
                from torch.utils.tensorboard.writer import SummaryWriter

                print("> setting tensorboard ...")
                self._tensorboard_logger = SummaryWriter(
                    log_dir=self.cfg.logger.tensorboard_dir,
                    max_queue=self.cfg.logger.tensorboard_queue_size,
                )
            else:
                self._tensorboard_logger = None
        return self._tensorboard_logger

    @property
    def wandb_logger(self) -> Any | None:
        """The Weights & Biases logger instance, lazily initialized for rank N-1."""
        if self._wandb_logger is None:
            if self.cfg.logger.wandb_project and get_rank_safe() == (get_world_size_safe() - 1):
                if self.cfg.logger.wandb_exp_name == "":
                    raise ValueError("Please specify the wandb experiment name!")

                import wandb

                save_dir = self.cfg.logger.wandb_save_dir or os.path.join(self.cfg.checkpoint.save, "wandb")
                wandb_kwargs = {
                    "dir": save_dir,
                    "name": self.cfg.logger.wandb_exp_name,
                    "project": self.cfg.logger.wandb_project,
                    "config": self.cfg.to_dict(),
                    "entity": self.cfg.logger.wandb_entity,
                }
                wandb.init(**wandb_kwargs)

                self._wandb_logger = wandb
            else:
                self._wandb_logger = None
        return self._wandb_logger

    @property
    def timers(self) -> Timers:
        """The Megatron Timers instance used for tracking execution times."""
        if self._timers is None:
            self._timers = Timers(self.cfg.logger.timing_log_level, self.cfg.logger.timing_log_option)
            self._timers.write_to_wandb = types.MethodType(_timers_write_to_wandb, self._timers)
        return self._timers

    @property
    def train_state(self) -> TrainState:
        """The TrainState object holding training progress information."""
        if self._train_state is None:
            self._train_state = TrainState()
        return self._train_state

    @train_state.setter
    def train_state(self, value: TrainState) -> None:
        """Sets the training state object.

        Args:
            value: The TrainState instance to set.
        """
        self._train_state = value

    @property
    def fault_tolerance_state(self) -> FaultToleranceState:
        """The FaultToleranceState object holding FT-specific information."""
        if self._ft_state is None:
            self._ft_state = FaultToleranceState()
        return self._ft_state

    @fault_tolerance_state.setter
    def fault_tolerance_state(self, value: FaultToleranceState) -> None:
        """Sets the fault tolerance state object.

        Args:
            value: The FaultToleranceState instance to set.
        """
        self._ft_state = value

    @property
    def signal_handler(self) -> DistributedSignalHandler:
        """The DistributedSignalHandler instance for graceful shutdown."""
        if self._signal_handler is None:
            self._set_signal_handler()
        return self._signal_handler

    @property
    def straggler_timer(self) -> StragglerDetector:
        """The StragglerDetector instance for tracking slow GPUs."""
        if self._straggler_timer is None:
            self._straggler_timer = StragglerDetector()
        return self._straggler_timer

    def initialize_async_checkpoint_worker(self) -> None:
        """Initializes the async checkpoint worker."""
        if (
            self._async_calls_queue is None
            and self.cfg
            and self.cfg.checkpoint.save is not None
            and self.cfg.checkpoint.async_save
        ):
            self._async_calls_queue = AsyncCallsQueue(persistent=self.cfg.checkpoint.use_persistent_ckpt_worker)

    @property
    def async_calls_queue(self) -> AsyncCallsQueue | None:
        """The AsyncCallsQueue instance for handling asynchronous checkpoint saves."""
        return self._async_calls_queue

    @property
    def nvrx_straggler_manager(self) -> NVRxStragglerDetectionManager | None:
        """The NVRx straggler detection manager, if enabled."""
        if (
            not self._nvrx_straggler_created
            and self._nvrx_straggler_manager is None
            and self.cfg is not None
            and self.cfg.nvrx_straggler is not None
        ):
            self._nvrx_straggler_manager = NVRxStragglerDetectionManager(self.cfg.nvrx_straggler)
            self._nvrx_straggler_created = True
        return self._nvrx_straggler_manager

    @property
    def energy_monitor(self) -> EnergyMonitor | None:
        """The EnergyMonitor instance for tracking energy consumption."""
        if (
            not self._energy_monitor_created
            and self._energy_monitor is None
            and self.cfg is not None
            and self.cfg.logger.log_energy
        ):
            self._energy_monitor = EnergyMonitor()
            self._energy_monitor_created = True
        return self._energy_monitor

    def _set_signal_handler(self) -> None:
        """Initializes the distributed signal handler based on the configuration."""
        if self.cfg.train is not None:
            self._signal_handler = DistributedSignalHandler(self.cfg.train.exit_signal)

    def reset_for_restart(self) -> None:
        """Reset GlobalState components for in-process restart.

        This cleans up all stateful components that need to be reinitialized between restart iterations.
        The async calls queue for checkpointing is handled separately in aborting in order to clean up persistent workers.
        """
        self._timers = None
        self._train_state = None
        self._tensorboard_logger = None
        self._wandb_logger = None
        self._energy_monitor = None
        self._energy_monitor_created = False
        self._signal_handler = None
        self._straggler_timer = None
        self._nvrx_straggler_manager = None
        self._nvrx_straggler_created = False


def _timers_write_to_wandb(
    self: Timers,
    names: list[str],
    writer: Any,
    iteration: int,
    normalizer: float = 1.0,
    reset: bool = True,
    barrier: bool = False,
) -> None:
    """Patch to write timers to wandb for Megatron Core Timers."""
    # currently when using add_scalars,
    # torch.utils.add_scalars makes each timer its own run, which
    # polutes the runs list, so we just add each as a scalar
    assert normalizer > 0.0
    name_to_min_max_time = self._get_global_min_max_time(names, reset, barrier, normalizer)
    if writer is not None:
        for name in name_to_min_max_time:
            _, max_time = name_to_min_max_time[name]
            writer.log({name + "-time": max_time}, iteration)
