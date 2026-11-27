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

import pytest
from megatron.core.optimizer import OptimizerConfig

from megatron.bridge.recipes.utils.optimizer_utils import (
    distributed_fused_adam_with_cosine_annealing,
    distributed_fused_adam_with_cosine_annealing_samples,
    distributed_muon_with_cosine_annealing,
)
from megatron.bridge.training.config import SchedulerConfig


@pytest.mark.parametrize("effective", [0.0, 0.033, 0.1])
def test_constant_recipe_config_preserves_effective_decay(effective):
    from types import SimpleNamespace

    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.training.config.training_config import SchedulerConfig as MLMSchedulerConfig

    optimizer, scheduler = distributed_fused_adam_with_cosine_annealing(
        weight_decay=0.1,
        start_weight_decay=effective,
        end_weight_decay=effective,
    )
    assert optimizer.weight_decay == 0.1
    scheduler.normalize_weight_decay(optimizer)
    assert optimizer.weight_decay == effective
    MLMSchedulerConfig(start_weight_decay=effective, end_weight_decay=effective).validate_run(
        train_iters=10,
        train_samples=None,
        skip_train=False,
        weight_decay=optimizer.weight_decay,
    )
    groups = SimpleNamespace(param_groups=[{"wd_mult": mult, "lr_mult": 1.0} for mult in (0.0, 0.5, 1.0)])
    schedule = OptimizerParamScheduler(
        groups,
        init_lr=0.0,
        max_lr=0.001,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=10,
        lr_decay_style="linear",
        start_wd=scheduler.start_weight_decay,
        end_wd=scheduler.end_weight_decay,
        wd_incr_steps=10,
        wd_incr_style="constant",
    )
    for step in (0, 5, 10):
        schedule.step(step - schedule.num_steps)
        assert [group["weight_decay"] for group in groups.param_groups] == pytest.approx(
            [0.0, effective / 2, effective]
        )


def test_late_recipe_override_is_normalized_by_validation(monkeypatch):
    from megatron.bridge.recipes import vanilla_gpt_pretrain_config
    from megatron.bridge.training.config import runtime_config_update

    monkeypatch.setenv("WORLD_SIZE", "1")
    config = vanilla_gpt_pretrain_config()
    config.model.num_layers = 2
    config.model.hidden_size = 128
    config.model.num_attention_heads = 4
    config.scheduler.start_weight_decay = config.scheduler.end_weight_decay = 0.02
    with pytest.warns(UserWarning, match=r"optimizer.weight_decay=.*Applying weight_decay=0.02"):
        runtime_config_update(config)
    config.validate()
    assert config.optimizer.weight_decay == 0.02


def test_unset_and_nonconstant_weight_decay_bounds():
    optimizer = OptimizerConfig(weight_decay=0.07)
    scheduler = SchedulerConfig(start_weight_decay=None, end_weight_decay=None)
    scheduler.normalize_weight_decay(optimizer)
    assert scheduler.start_weight_decay == scheduler.end_weight_decay == 0.07
    scheduler = SchedulerConfig(start_weight_decay=0.01, end_weight_decay=0.1, weight_decay_incr_style="linear")
    scheduler.normalize_weight_decay(optimizer)
    assert optimizer.weight_decay == 0.07


class TestOptimizerUtils:
    """Test optimizer and scheduler configs."""

    def test_optimizer_config(self):
        """Test optimizer config."""

        optim_cfg, _ = distributed_fused_adam_with_cosine_annealing(
            adam_beta2=0.98,
            adam_eps=1e-8,
            weight_decay=0.01,
            max_lr=3e-4,
            min_lr=3e-5,
        )

        assert isinstance(optim_cfg, OptimizerConfig)
        assert optim_cfg.lr == 3e-4
        assert optim_cfg.weight_decay == 0.01
        assert optim_cfg.adam_eps == 1e-8
        assert optim_cfg.adam_beta2 == 0.98
        assert optim_cfg.bf16 is True

    def test_muon_optimizer_config(self):
        """Test Muon optimizer config."""

        optim_cfg, _ = distributed_muon_with_cosine_annealing(
            muon_momentum=0.98,
            muon_use_nesterov=False,
            muon_extra_scale_factor=1.01,
            weight_decay=0.01,
            max_lr=3e-4,
            min_lr=3e-5,
        )

        assert isinstance(optim_cfg, OptimizerConfig)
        assert optim_cfg.lr == 3e-4
        assert optim_cfg.weight_decay == 0.01
        assert optim_cfg.muon_extra_scale_factor == 1.01
        assert optim_cfg.muon_nesterov is False
        assert optim_cfg.muon_momentum == 0.98
        assert optim_cfg.bf16 is True

    def test_muon_lion_optimizer_config(self):
        """Test Muon+Lion optimizer config."""

        optim_cfg, _ = distributed_muon_with_cosine_annealing(
            muon_momentum=0.98,
            muon_use_nesterov=False,
            muon_extra_scale_factor=1.01,
            muon_scalar_optimizer="lion",
            lion_beta2=0.95,
            weight_decay=0.01,
            max_lr=3e-4,
            min_lr=3e-5,
        )

        assert isinstance(optim_cfg, OptimizerConfig)
        assert optim_cfg.lr == 3e-4
        assert optim_cfg.weight_decay == 0.01
        assert optim_cfg.lion_beta2 == 0.95
        assert optim_cfg.muon_scalar_optimizer == "lion"
        assert optim_cfg.muon_extra_scale_factor == 1.01
        assert optim_cfg.muon_nesterov is False
        assert optim_cfg.muon_momentum == 0.98
        assert optim_cfg.bf16 is True

    def test_scheduler_config(self):
        """Test scheduler config."""

        _, scheduler_cfg = distributed_fused_adam_with_cosine_annealing(
            lr_warmup_iters=1999,
            lr_decay_iters=12345,
        )

        assert isinstance(scheduler_cfg, SchedulerConfig)
        assert scheduler_cfg.lr_warmup_iters == 1999
        assert scheduler_cfg.lr_decay_iters == 12345

    def test_muon_scheduler_config(self):
        """Test Muon scheduler config."""

        _, scheduler_cfg = distributed_muon_with_cosine_annealing(
            lr_warmup_iters=1999,
            lr_decay_iters=12345,
        )

        assert isinstance(scheduler_cfg, SchedulerConfig)
        assert scheduler_cfg.lr_warmup_iters == 1999
        assert scheduler_cfg.lr_decay_iters == 12345

    def test_muon_lion_scheduler_config(self):
        """Test Muon+Lion scheduler config."""

        _, scheduler_cfg = distributed_muon_with_cosine_annealing(
            muon_scalar_optimizer="lion",
            lr_warmup_iters=1999,
            lr_decay_iters=12345,
        )

        assert isinstance(scheduler_cfg, SchedulerConfig)
        assert scheduler_cfg.lr_warmup_iters == 1999
        assert scheduler_cfg.lr_decay_iters == 12345

    def test_sample_based_optimizer_config(self):
        """Test sample-based optimizer config."""

        optim_cfg, _ = distributed_fused_adam_with_cosine_annealing_samples(
            precision="bf16-mixed",
            adam_beta2=0.95,
            adam_eps=1e-8,
            weight_decay=0.1,
            max_lr=1e-4,
            min_lr=1e-5,
        )

        assert isinstance(optim_cfg, OptimizerConfig)
        assert optim_cfg.lr == 1e-4
        assert optim_cfg.min_lr == 1e-5
        assert optim_cfg.weight_decay == 0.1
        assert optim_cfg.adam_beta2 == 0.95
        assert optim_cfg.bf16 is True
        assert optim_cfg.use_distributed_optimizer is True

    def test_sample_based_scheduler_config(self):
        """Test sample-based scheduler config."""

        _, scheduler_cfg = distributed_fused_adam_with_cosine_annealing_samples(
            lr_warmup_samples=1000,
            lr_decay_samples=8000,
        )

        assert isinstance(scheduler_cfg, SchedulerConfig)
        assert scheduler_cfg.lr_warmup_samples == 1000
        assert scheduler_cfg.lr_decay_samples == 8000
        assert scheduler_cfg.lr_warmup_iters == 0  # Should be 0 for sample-based
        assert scheduler_cfg.lr_decay_iters is None  # Should be None for sample-based
        assert scheduler_cfg.lr_decay_style == "cosine"

    def test_sample_based_scheduler_config_with_none_defaults(self):
        """Test sample-based scheduler config with None defaults (auto from train_samples)."""

        _, scheduler_cfg = distributed_fused_adam_with_cosine_annealing_samples(
            lr_warmup_samples=None,  # Should default to None for auto calculation
            lr_decay_samples=None,  # Should default to None for auto calculation
        )

        assert isinstance(scheduler_cfg, SchedulerConfig)
        assert scheduler_cfg.lr_warmup_samples is None  # Will auto-calculate from train_samples
        assert scheduler_cfg.lr_decay_samples is None  # Will auto-calculate from train_samples
        assert scheduler_cfg.lr_warmup_iters == 0
        assert scheduler_cfg.lr_decay_iters is None
