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
from typing import Any, Optional

from megatron.bridge.training.callbacks.abstract_callback import AbstractCallback


class RuntimeEstimator(AbstractCallback):
    """
    Estimates total training time.
    The training time is computed by taking the time elapsed for the current duration and multiplying
    out to the full extended length of the training run.
    This callback provides a best attempt estimate. This estimate may be inaccurate if throughput
    changes through training or other significant changes are made to the model or dataloader.
    Example:
        from megatron.bridge.training.callbacks import RuntimeEstimator
        config.logger.callbacks.append(RuntimeEstimator(time_unit="hours"))
    +-----------------------------+-------------------------------+
    | Key                         | Logged data                   |
    +=============================+===============================+
    | `time/remaining_estimate`   | Estimated time to completion  |
    +-----------------------------+-------------------------------+
    | `time/tokens`               | Number of consumed tokens     |
    +-----------------------------+-------------------------------+
    | `time/samples`              | Number of consumed samples    |
    +-----------------------------+-------------------------------+
    | `time/batches`              | Number of consumed batches    |
    +-----------------------------+-------------------------------+
    | `time/total`                | Total training time           |
    +-----------------------------+-------------------------------+
    Args:
        time_unit (str, optional): Time unit to use for `time` logging. Can be one of
            'seconds', 'minutes', 'hours', or 'days'. Defaults to 'hours'.
    """

    def __init__(self, time_unit: str = 'hours') -> None:
        self._enabled = True
        self.start_time = None
        self.start_dur = None

        self.time_unit = time_unit
        self.divider = 1
        if time_unit == 'seconds':
            self.divider = 1
        elif time_unit == 'minutes':
            self.divider = 60
        elif time_unit == 'hours':
            self.divider = 60 * 60
        elif time_unit == 'days':
            self.divider = 60 * 60 * 24
        else:
            raise ValueError(
                f'Invalid time_unit: {time_unit}. Must be one of "seconds", "minutes", "hours", or "days".',
            )

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0
        self.eval_wct_per_label: dict[str, list[float]] = {}
        # How often eval is called as fraction of total training time
        self.eval_frequency_per_label: dict[str, float] = {}
        self.num_tokens: int = 0
        self.num_samples: int = 0
        self.num_batches: int = 0

    def _get_elapsed_duration(self, iteraion: int, train_iters: int) -> Optional[float]:
        """Get the elapsed duration.
        This method computes fractional progress in an epoch
        provided at least 1 epoch has passed by recording how many batches were in each epoch.
        """
        if iteraion is not None:
            return iteraion / train_iters

        return None

    def track_start(self, iteraion: int, train_iters: int) -> None:
        """ """
        if self._enabled and self.start_time is None:
            self.start_time = time.time()
            self.start_dur = self._get_elapsed_duration(iteraion, train_iters)
            if self.start_dur is None:
                self._enabled = False

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
        if not self._enabled:
            return

        elapsed_dur = self._get_elapsed_duration(iteration, train_config.train_iters)
        assert elapsed_dur is not None, 'elapsed_dur should be not None. Please, make sure that training has started.'

        assert self.start_dur is not None
        assert self.start_time is not None

        time_metrics = {}
        if elapsed_dur > self.start_dur:
            elapsed_time = time.time() - self.start_time
            elapsed_time -= self.total_eval_wct  # Subtract time spent evaluating
            rate = elapsed_time / (elapsed_dur - self.start_dur)
            remaining_time = rate * (1 - elapsed_dur)

            # Add remaining time from each evaluator using known frequencies. We explicitly compute
            # frequency instead of using time interpolation to avoid saw tooth pattern in estimates
            for dataloader_label, eval_wcts in self.eval_wct_per_label.items():
                # Discard first eval_wct if possible as it is often slower due to dataset downloading
                eval_wct_avg = None
                num_evals_finished = len(eval_wcts)
                if num_evals_finished > 1:
                    eval_wct_avg = sum(eval_wcts[1:]) / (num_evals_finished - 1)
                else:
                    eval_wct_avg = sum(eval_wcts) / num_evals_finished
                eval_rate = self.eval_frequency_per_label[dataloader_label]
                num_total_evals = 1 / eval_rate * (1 - self.start_dur)
                remaining_calls = num_total_evals - num_evals_finished
                remaining_time += eval_wct_avg * remaining_calls

            time_metrics['time/remaining_estimate'] = remaining_time / self.divider

        batch_size = train_config.global_batch_size
        self.num_tokens += seq_length * (batch_size)
        time_metrics['time/tokens'] = self.num_tokens
        self.num_samples += batch_size
        time_metrics['time/samples'] = self.num_samples
        self.num_batches += 1
        time_metrics['time/batches'] = self.num_batches
        time_metrics['time/total'] = (time.time() - start_time) / self.divider

        for metric, value in time_metrics.items():
            writer.add_scalar(metric, value, iteration)

        if wandb_writer:
            wandb_writer.log(time_metrics, iteration)
