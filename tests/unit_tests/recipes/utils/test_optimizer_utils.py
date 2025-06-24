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

"""Tests for optinizer_utils module."""

from megatron.core.optimizer import OptimizerConfig

from megatron.hub.recipes.utils.optimizer_utils import distributed_fused_adam_with_cosine_annealing
from megatron.hub.training.config import SchedulerConfig


class TestOptimizerUtils:
    """Test optimizer and scheduler configs."""

    def test_optimizer_config(self):
        """Test optimizer config."""

        optim_cfg, _ = distributed_fused_adam_with_cosine_annealing(
            adam_beta2=0.98,
            adam_eps=1e-5,
            weight_decay=0.01,
            max_lr=3e-4,
            min_lr=3e-5,
        )

        assert isinstance(optim_cfg, OptimizerConfig)
        assert optim_cfg.lr == 3e-4
        assert optim_cfg.weight_decay == 0.01
        assert optim_cfg.adam_beta2 == 0.98
        assert optim_cfg.bf16 == True

    def test_scheduler_config(self):
        """Test scheduler config."""

        _, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=1999,
            lr_decay_iters=12345,
        )

        assert isinstance(scheduler_cfg, SchedulerConfig)
        assert scheduler_cfg.lr_warmup_iters == 1999
        assert scheduler_cfg.lr_decay_iters == 12345
