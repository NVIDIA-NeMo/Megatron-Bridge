#!/usr/bin/env python3
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
Validation script for profiler context exit behavior.

This script trains Llama 3.2 1B for 100 steps with nsys profiling enabled
for steps 10-12. It validates that step times return to normal after
profiling ends.

Usage:
    torchrun --nproc_per_node=2 examples/validate_profiling_exit.py

    Or with nsys (recommended):
    nsys profile -s none -t nvtx,cuda -o profiling_validation --force-overwrite true \
        --capture-range=cudaProfilerApi --capture-range-end=stop \
        torchrun --nproc_per_node=2 examples/validate_profiling_exit.py
"""

import torch

from megatron.bridge.recipes.llama.llama32_1b import pretrain_config
from megatron.bridge.training.config import ProfilingConfig
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain


def main():
    """Main training function with profiling validation."""

    # Create config for Llama 3.2 1B with mock data
    config = pretrain_config(
        # Output directory
        dir="/tmp/profiling_validation",
        name="llama32_1b_profiling_test",
        # Use mock data for quick testing
        mock=True,
        # Training settings - 50 steps total
        train_iters=100,
        global_batch_size=32,
        micro_batch_size=1,
        seq_length=512,
        # Evaluation settings
        # Model parallelism (adjust based on available GPUs)
        tensor_parallelism=1,
        pipeline_parallelism=1,
        context_parallelism=1,
        sequence_parallelism=False,
        # Learning rate settings
        lr=3e-4,
        min_lr=3e-5,
        lr_warmup_iters=10,
        lr_decay_iters=50,
        # Precision
        precision_config="bf16_mixed",
    )

    # Configure nsys profiling for steps 5-6
    # This validates that profiling starts cleanly and exits without leaving NVTX overhead
    config.profiling = ProfilingConfig(
        use_nsys_profiler=True,
        profile_step_start=5,
        profile_step_end=6,
        profile_ranks=[0],  # Only profile rank 0
    )

    # Adjust evaluation and logging
    config.train.eval_interval = 200  # No eval during profiling test
    config.train.eval_iters = 0
    config.model.seq_length = 512
    config.checkpoint.save = None
    config.logger.log_interval = 10
    config.dataset.num_workers = 0

    # Run pretraining
    pretrain(
        config=config,
        forward_step_func=forward_step,
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    print("\n" + "=" * 80)
    print("Profiling validation complete!")
    print("=" * 80)
    print("\nTo validate NVTX context was properly closed:")
    print("1. Check the training logs for step times")
    print("2. Steps 1-4: Warmup/compilation (may be slower)")
    print("3. Steps 5-6: PROFILING ACTIVE (nsys capturing)")
    print("4. Step 7: One-time spike (nsys writing report file - expected)")
    print("5. Steps 8-50: POST-PROFILING (should return to baseline)")
    print("\n✓ Success criteria:")
    print("  - Steps 8+ have similar times to steps 1-4 baseline")
    print("  - No persistent slowdown after profiling window")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
