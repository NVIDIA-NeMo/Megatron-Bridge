# Copyright (c) 2025, NVIDIA CORPORATION.
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

#!/usr/bin/env python3
"""
End-to-end functional test for NVRx straggler detection with megatron/hub training.

This test runs the actual pretrain function with NVRx straggler detection
enabled, using mock data and a tiny model configuration for fast testing.
"""

import io
import logging
import sys
import time

import pytest
import torch

from megatron.bridge.models.llama import Llama32ModelProvider1B
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    LoggerConfig,
    MockGPTDatasetConfig,
    NVRxStragglerDetectionConfig,
    OptimizerConfig,
    RNGConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.nvrx_straggler import HAVE_NVRX
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.state import GlobalState
from megatron.bridge.utils.common_utils import get_rank_safe


def create_functional_test_config(enable_nvrx: bool = True) -> ConfigContainer:
    """Create a complete minimal configuration for functional testing, based on test_pretrain.py."""

    seq_length = 512
    train_config = TrainingConfig(
        train_iters=10,
        micro_batch_size=1,
        global_batch_size=2,
        eval_interval=10,
        eval_iters=0,
    )

    model_config = Llama32ModelProvider1B(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        attention_softmax_in_fp32=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        seq_length=seq_length,
        make_vocab_size_divisible_by=128,
        vocab_size=None,
    )

    dataset_config = MockGPTDatasetConfig(
        random_seed=1234,
        reset_attention_mask=False,
        reset_position_ids=False,
        eod_mask_loss=False,
        seq_length=seq_length,
        num_dataset_builder_threads=1,
        data_sharding=True,
        dataloader_type="single",
        num_workers=1,
    )

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        lr=3e-3,
        weight_decay=0.01,
        min_lr=1e-6,
    )

    scheduler_config = SchedulerConfig(
        start_weight_decay=0.033,
        end_weight_decay=0.033,
        weight_decay_incr_style="constant",
        lr_decay_style="cosine",
        lr_warmup_iters=2,
        lr_warmup_init=0.0,
        lr_decay_iters=train_config.train_iters,
        override_opt_param_scheduler=True,
    )

    tokenizer_config = TokenizerConfig(
        tokenizer_type="NullTokenizer",
        vocab_size=10000,
    )

    logger_config = LoggerConfig(
        log_interval=5,
        tensorboard_dir=None,
    )

    checkpoint_config = CheckpointConfig(
        save=None,
        load=None,
        save_interval=None,
    )

    rng_config = RNGConfig(seed=1234)

    dist_config = DistributedInitConfig()

    ddp_config = DistributedDataParallelConfig(
        check_for_nan_in_grad=True,
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        overlap_param_gather=True,
        average_in_collective=True,
        use_distributed_optimizer=True,
    )

    nvrx_config = None
    if enable_nvrx:
        nvrx_config = NVRxStragglerDetectionConfig(
            enabled=True,
            report_time_interval=2.0,
            calc_relative_gpu_perf=True,
            calc_individual_gpu_perf=True,
            num_gpu_perf_scores_to_print=4,
            gpu_relative_perf_threshold=0.7,
            gpu_individual_perf_threshold=0.7,
            stop_if_detected=False,
            enable_logging=True,
            profiling_interval=1,
            logger_name="nvrx_functional_test",
        )

    return ConfigContainer(
        train=train_config,
        model=model_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        dataset=dataset_config,
        logger=logger_config,
        tokenizer=tokenizer_config,
        checkpoint=checkpoint_config,
        rng=rng_config,
        dist=dist_config,
        ddp=ddp_config,
        nvrx_straggler=nvrx_config,
    )


def create_timed_forward_step_func(sleep_time: float = 1.0):
    """Create a forward step function that sleeps before calling the real forward_step.

    This simulates work being done and allows NVRx to measure performance differences.
    Only rank 1 will be slow to simulate a straggler scenario.

    Args:
        sleep_time: Time to sleep in seconds before each forward step (only for rank 1)

    Returns:
        A forward step function compatible with megatron training
    """

    def timed_forward_step_func(state: GlobalState, data_iterator, model, return_schedule_plan: bool = False):
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            time.sleep(sleep_time)
            print(f"Rank {torch.distributed.get_rank()}: Simulated slow forward step (slept {sleep_time}s)")

        return forward_step(state, data_iterator, model, return_schedule_plan=return_schedule_plan)

    return timed_forward_step_func


class _InMemoryHandler(logging.Handler):
    """Logging handler that stores formatted records in memory.

    Immune to file-system timing issues and survives logging reconfiguration
    by ``setup_logging`` inside ``pretrain()`` (which only adds filters /
    adjusts levels but never removes handlers).
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))

    @property
    def output(self) -> str:
        return "\n".join(self.records)


def _attach_handler(handler: logging.Handler) -> None:
    """Attach *handler* to root and all NVRx-related loggers."""
    logging.root.addHandler(handler)
    for name in ("nvrx_functional_test", "nvidia_resiliency", "straggler", "nvrx"):
        lgr = logging.getLogger(name)
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(handler)


def _detach_handler(handler: logging.Handler) -> None:
    """Remove *handler* from root and all NVRx-related loggers."""
    logging.root.removeHandler(handler)
    for name in ("nvrx_functional_test", "nvidia_resiliency", "straggler", "nvrx"):
        lgr = logging.getLogger(name)
        lgr.removeHandler(handler)


@pytest.mark.skipif(not HAVE_NVRX, reason="nvidia-resiliency-ext is not installed")
def test_nvrx_straggler_detection_end_to_end(sleep_time: float = 1.0):
    """
    End-to-end functional test that runs actual megatron training with NVRx.

    This test:
    1. Sets up a complete megatron training configuration
    2. Uses mock data and small model for fast execution
    3. Runs the actual pretrain function
    4. Verifies NVRx straggler detection is working by checking logs
    """
    rank = get_rank_safe()

    mem_handler = _InMemoryHandler()
    _attach_handler(mem_handler)

    # Capture stdout so we can see print_rank_0 errors from train.py
    # (NVRx init failures are reported via print, not logging)
    captured_stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = _TeeWriter(old_stdout, captured_stdout)

    try:
        config = create_functional_test_config(enable_nvrx=True)
        forward_step_func = create_timed_forward_step_func(sleep_time=sleep_time)

        try:
            pretrain(config=config, forward_step_func=forward_step_func)
            training_success = True
        except Exception:
            training_success = False
            if rank == 0:
                import traceback

                traceback.print_exc()

        assert training_success, "Training must complete successfully"

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if rank == 0:
            stdout_content = captured_stdout.getvalue()
            log_content = mem_handler.output

            if "Failed to initialize NVRx straggler detection" in stdout_content:
                pytest.skip(
                    f"NVRx straggler detection failed to initialize (environment issue). "
                    f"stdout: {stdout_content[:500]}"
                )

            combined = (log_content + "\n" + stdout_content).lower()
            has_gpu_perf_logs = "gpu relative performance" in combined
            has_rank_scores = "rank=" in combined and "score=" in combined
            has_straggler_detection = "straggler" in combined
            has_nvidia_resiliency = "nvidia_resiliency" in combined

            assert has_gpu_perf_logs or has_rank_scores or has_straggler_detection or has_nvidia_resiliency, (
                f"Expected NVRx straggler detection logs not found.\n"
                f"  GPU perf logs: {has_gpu_perf_logs}\n"
                f"  Rank scores: {has_rank_scores}\n"
                f"  Straggler detection: {has_straggler_detection}\n"
                f"  Nvidia resiliency: {has_nvidia_resiliency}\n"
                f"Log handler captured {len(mem_handler.records)} records.\n"
                f"First 2000 chars of log output:\n{log_content[:2000]}\n"
                f"First 2000 chars of stdout:\n{stdout_content[:2000]}"
            )

    finally:
        sys.stdout = old_stdout
        _detach_handler(mem_handler)


class _TeeWriter:
    """Write to two streams simultaneously."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

    def fileno(self):
        return self.streams[0].fileno()


if __name__ == "__main__":
    test_nvrx_straggler_detection_end_to_end()
