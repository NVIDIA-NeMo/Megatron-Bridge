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

"""Keep Bridge's FP32 load-memory contract when allocation is owned by Core."""

import os
from datetime import timedelta

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.transformer import TransformerConfig

from megatron.bridge.training.config import OptimizerConfig, SchedulerConfig
from megatron.bridge.training.optim import setup_optimizer


@pytest.fixture(scope="module")
def distributed_group():
    if not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) != 2:
        pytest.skip("requires a two-GPU distributed launch")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group("nccl", timeout=timedelta(seconds=60))
    parallel_state.initialize_model_parallel()
    try:
        yield
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("initialized", [False, True])
def test_bridge_optimizer_loading_has_no_duplicate_fp32_moments(initialized, distributed_group):
    raw = torch.nn.Module()
    raw.weight = torch.nn.Parameter(torch.ones(2048, 2048, device="cuda", dtype=torch.bfloat16))
    raw.config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True)
    model = DistributedDataParallel(raw.config, DistributedDataParallelConfig(use_distributed_optimizer=True), raw)
    config = OptimizerConfig(
        lr=0.001,
        min_lr=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=False,
    )
    schedule = SchedulerConfig(start_weight_decay=0.0, end_weight_decay=0.0)
    schedule.lr_warmup_steps = 0
    schedule.lr_decay_steps = schedule.wd_incr_steps = 10
    optimizer, _ = setup_optimizer(config, schedule, [model])
    leaf = optimizer
    while isinstance(leaf, ChainedOptimizer):
        assert len(leaf.chained_optimizers) == 1
        leaf = leaf.chained_optimizers[0]

    def moment_pointers():
        return [
            state[name].data_ptr() for state in leaf.optimizer.state.values() for name in ("exp_avg", "exp_avg_sq")
        ]

    if initialized:
        optimizer.load_state_dict(optimizer.state_dict())
        pointers = moment_pointers()
        assert pointers
    saved = optimizer.state_dict()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    optimizer.load_state_dict(saved)
    torch.cuda.synchronize()
    live_growth = torch.cuda.memory_allocated() - before
    peak_growth = torch.cuda.max_memory_allocated() - before
    # One rank's two moments occupy 16 MiB. Allow 1 MiB for bookkeeping,
    # but never a second copy of those moments during either load phase.
    assert peak_growth <= live_growth + 1024**2, (peak_growth, live_growth)
    assert moment_pointers()
    if initialized:
        assert moment_pointers() == pointers
