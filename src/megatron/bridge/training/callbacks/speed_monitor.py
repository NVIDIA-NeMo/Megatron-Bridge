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

import time
from collections import deque
from typing import Any

import torch.distributed as dist

from megatron.bridge.training.callbacks.abstract_callback import AbstractCallback


class SpeedMonitor(AbstractCallback):
    """
    Logs the training throughput and utilization.
    The training throughput is logged on the event once we have reached the `window_size` threshold.
    Example:
        from megatron.bridge.training.callbacks import SpeedMonitor
        config.logger.callbacks.append(SpeedMonitor(window_size=100,))
    The training throughput is logged by the PTL's logger to the following keys as described below.
    +-------------------------------------+-----------------------------------------------------------+
    | Key                                 | Logged data                                               |
    +=====================================+===========================================================+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/batches_per_sec`        | batches) of the number of batches processed per second.   |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/samples_per_sec`        | batches) of the number of samples processed per second.   |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | Rolling average (over `window_size` most recent           |
    | `throughput/tokens_per_sec`         | batches) of the number of tokens processed per second.    |
    |                                     | Only logged if dataspec returns tokens per batch.         |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/batches_per_sec` | `throughput/batches_per_sec` divided by world size.       |
    +-------------------------------------+-----------------------------------------------------------+
    | `throughput/device/samples_per_sec` | `throughput/samples_per_sec` divided by world size.       |
    +-------------------------------------+-----------------------------------------------------------+
    |                                     | `throughput/tokens_per_sec` divided by world size. Only   |
    | `throughput/device/tokens_per_sec`  | logged if dataspec returns tokens per batch.              |
    |                                     |                                                           |
    +-------------------------------------+-----------------------------------------------------------+
    Args:
        window_size (int, optional): Number of batches to use for a rolling average of throughput.
            Defaults to 100.
    """

    def __init__(
        self,
        window_size: int = 100,
    ):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.history_samples: deque[int] = deque(maxlen=window_size + 1)
        self.history_tokens: deque[int] = deque(maxlen=window_size + 1)
        self.history_wct: deque[float] = deque(maxlen=window_size + 1)
        self.history_flops: deque[float] = deque(maxlen=window_size + 1)

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.consumed_samples = 0
        self.consumed_tokens = 0

    def track(
        self,
        iteration: int,
        writer,
        wandb_writer,
        start_time: float,
        train_config,
        seq_length: int,
        **kwargs,
    ) -> None:
        """ """
        # Add the new element
        batch_size = train_config.global_batch_size
        self.consumed_samples += batch_size
        self.history_samples.append(self.consumed_samples)

        self.consumed_tokens += seq_length * (batch_size)
        self.history_tokens.append(self.consumed_tokens)

        elapsed_time = time.time() - start_time
        self.history_wct.append(elapsed_time)

        # Log the throughput
        metrics = {}
        if len(self.history_wct) == self.history_wct.maxlen:
            world_size = dist.get_world_size()
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = int(self.history_samples[-1]) - int(self.history_samples[0])
            elapsed_tokens = int(self.history_tokens[-1]) - int(self.history_tokens[0])
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            batches_per_sec = elapsed_batches / elapsed_wct
            samples_per_sec = elapsed_samples / elapsed_wct
            dev_batches_per_sec = batches_per_sec / world_size
            dev_samples_per_sec = samples_per_sec / world_size
            metrics = {
                'throughput/batches_per_sec': batches_per_sec,
                'throughput/samples_per_sec': samples_per_sec,
                'throughput/device/batches_per_sec': dev_batches_per_sec,
                'throughput/device/samples_per_sec': dev_samples_per_sec,
                'throughput/micro_batch_size': train_config.micro_batch_size,
                'throughput/global_batch_size': batch_size,
            }
            if elapsed_tokens > 0:
                tokens_per_sec = elapsed_tokens / elapsed_wct
                dev_tokens_per_sec = tokens_per_sec / world_size
                metrics.update({'throughput/tokens_per_sec': tokens_per_sec})
                metrics.update({'throughput/device/tokens_per_sec': dev_tokens_per_sec})

        for metric, value in metrics.items():
            writer.add_scalar(metric, value, iteration)
        
        if wandb_writer:
            wandb_writer.log(metrics, iteration)
