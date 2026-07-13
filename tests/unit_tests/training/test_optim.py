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

"""Tests for setup_optimizer in optim.py."""

import builtins
from unittest.mock import MagicMock, patch

import torch
from megatron.core.optimizer import OptimizerConfig, ParamGroupOverride, ParamKey

from megatron.bridge.training.config import SchedulerConfig
from megatron.bridge.training.optim import sync_hybrid_device_optimizer_fp32_master_copies


class TestSetupOptimizerMuP:
    """Tests for μP optimizer scaling in setup_optimizer."""

    def _make_optimizer_config(self, lr=1e-3, min_lr=1e-5, optimizer="adam"):
        return OptimizerConfig(optimizer=optimizer, lr=lr, min_lr=min_lr, bf16=True)

    def _make_scheduler_config(self):
        cfg = SchedulerConfig(lr_decay_iters=1000, lr_decay_style="cosine")
        cfg.lr_warmup_steps = 0
        cfg.lr_decay_steps = 1000
        cfg.wsd_decay_steps = None
        return cfg

    def _make_model_mock(self, use_mup=False, mup_width_mult=1.0):
        model = MagicMock()
        model_config = MagicMock()
        model_config.use_mup = use_mup
        model_config.mup_width_mult = mup_width_mult
        return model, model_config

    def _make_param_key(self):
        """Create a simple ParamKey instance for use in fake overrides."""
        return ParamKey(name="*.weight")

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_disabled_skips_overrides(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """When use_mup=False, get_mup_config_overrides is not called."""
        from megatron.bridge.training.optim import setup_optimizer

        model, model_config = self._make_model_mock(use_mup=False)
        mock_get_model_config.return_value = model_config
        mock_get_optimizer.return_value = MagicMock()

        with patch("megatron.bridge.training.optim.get_mup_config_overrides") as mock_mup:
            setup_optimizer(
                optimizer_config=self._make_optimizer_config(),
                scheduler_config=self._make_scheduler_config(),
                model=model,
            )
            mock_mup.assert_not_called()

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_enabled_calls_overrides(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """When use_mup=True, get_mup_config_overrides is called with correct args."""
        from megatron.bridge.training.optim import setup_optimizer

        model, model_config = self._make_model_mock(use_mup=True, mup_width_mult=2.0)
        mock_get_model_config.return_value = model_config
        mock_get_optimizer.return_value = MagicMock()

        fake_overrides = {self._make_param_key(): ParamGroupOverride(lr_mult=0.5)}

        with patch("megatron.bridge.training.optim.get_mup_config_overrides", return_value=fake_overrides) as mock_mup:
            optimizer_config = self._make_optimizer_config(lr=1e-3, optimizer="adam")
            setup_optimizer(
                optimizer_config=optimizer_config,
                scheduler_config=self._make_scheduler_config(),
                model=model,
            )
            mock_mup.assert_called_once_with(
                config=optimizer_config,
                mup_width_mult=2.0,
                optimizer_type="adam",
            )

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_overrides_merged_with_existing(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """μP overrides are merged with existing config_overrides."""
        from megatron.bridge.training.optim import setup_optimizer

        model, model_config = self._make_model_mock(use_mup=True, mup_width_mult=4.0)
        mock_get_model_config.return_value = model_config

        mup_key = ParamKey(name="*.weight")
        existing_key = ParamKey(name="*.bias")
        mup_overrides = {mup_key: ParamGroupOverride(lr_mult=0.25)}
        existing_overrides = {existing_key: ParamGroupOverride(wd_mult=0.0)}

        captured_overrides = {}

        def capture_optimizer_call(**kwargs):
            captured_overrides.update(kwargs.get("config_overrides") or {})
            return MagicMock()

        mock_get_optimizer.side_effect = capture_optimizer_call

        with patch("megatron.bridge.training.optim.get_mup_config_overrides", return_value=mup_overrides):
            with patch(
                "megatron.bridge.training.optim.OptimizerConfigOverrideProvider.build_config_overrides",
                return_value=existing_overrides,
            ):
                setup_optimizer(
                    optimizer_config=self._make_optimizer_config(),
                    scheduler_config=self._make_scheduler_config(),
                    model=model,
                )

        assert mup_key in captured_overrides
        assert existing_key in captured_overrides

    @patch("megatron.bridge.training.optim._get_scheduler")
    @patch("megatron.bridge.training.optim.get_megatron_optimizer")
    @patch("megatron.bridge.training.optim.get_model_config")
    def test_mup_model_list_uses_first_chunk(self, mock_get_model_config, mock_get_optimizer, _mock_get_scheduler):
        """When model is a list, get_model_config is called on the first chunk."""
        from megatron.bridge.training.optim import setup_optimizer

        model1, model_config = self._make_model_mock(use_mup=False)
        model2 = MagicMock()
        mock_get_model_config.return_value = model_config
        mock_get_optimizer.return_value = MagicMock()

        setup_optimizer(
            optimizer_config=self._make_optimizer_config(),
            scheduler_config=self._make_scheduler_config(),
            model=[model1, model2],
        )

        mock_get_model_config.assert_called_once_with(model1)


class _FakeHDO:
    """Stand-in for HybridDeviceOptimizer used to satisfy the isinstance check."""


class _FakeParamRange:
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end


class _FakeDistribOpt:
    """Stand-in for DistributedOptimizer wrapping an HDO-like inner optimizer."""

    def __init__(self, *, model_param: torch.Tensor, shard_main_param: torch.Tensor | None, inner: object):
        self.optimizer = inner
        self.model_float16_groups = [[model_param]]
        self.shard_fp32_from_float16_groups = [[shard_main_param]]
        self._numel = model_param.numel()

    def _get_model_param_range_map(self, _param: torch.Tensor) -> dict:
        return {"param": _FakeParamRange(0, self._numel)}


class _PlainDistribOpt:
    """Stand-in for a DistributedOptimizer that does not wrap an HDO."""

    def __init__(self) -> None:
        self.optimizer = object()


class _ChainedOpt:
    """Stand-in for a ChainedOptimizer exposing the ``chained_optimizers`` attribute."""

    def __init__(self, sub_opts: list[object]) -> None:
        self.chained_optimizers = sub_opts


class TestSyncHybridDeviceOptimizerFp32MasterCopies:
    """Tests for the post-load FP32 master sync workaround helper."""

    def test_none_optimizer_is_noop(self):
        """A ``None`` optimizer is a no-op and returns ``False``."""
        assert sync_hybrid_device_optimizer_fp32_master_copies(None) is False

    def test_walks_all_three_fp32_levels(self):
        """The helper refreshes level-1 shard, level-2 CPU clone, and level-3 working copy."""
        model_param = torch.full((4,), 1.0, dtype=torch.bfloat16)
        shard_main_param = torch.zeros(4, dtype=torch.float32)
        cpu_clone = torch.zeros(4, dtype=torch.float32)
        fp32_working = torch.zeros(4, dtype=torch.float32)

        inner = _FakeHDO()
        inner.gpu_params_map_cpu_copy = {model_param: cpu_clone}

        update_calls: list[bool] = []

        def _fake_update_fp32() -> None:
            update_calls.append(True)
            fp32_working.data.copy_(model_param.data)

        inner.update_fp32_param_by_new_param = _fake_update_fp32

        distrib_opt = _FakeDistribOpt(
            model_param=model_param,
            shard_main_param=shard_main_param,
            inner=inner,
        )

        with patch(
            "megatron.core.optimizer.cpu_offloading.hybrid_optimizer.HybridDeviceOptimizer",
            _FakeHDO,
        ):
            synced = sync_hybrid_device_optimizer_fp32_master_copies(distrib_opt)

        ones = torch.ones(4, dtype=torch.float32)
        assert synced is True
        assert torch.allclose(shard_main_param, ones)
        assert torch.allclose(cpu_clone, ones)
        assert update_calls == [True]
        assert torch.allclose(fp32_working, ones)

    def test_no_op_when_inner_is_not_hdo(self):
        """A DistributedOptimizer that does not wrap an HDO is left untouched."""
        with patch(
            "megatron.core.optimizer.cpu_offloading.hybrid_optimizer.HybridDeviceOptimizer",
            _FakeHDO,
        ):
            assert sync_hybrid_device_optimizer_fp32_master_copies(_PlainDistribOpt()) is False

    def test_import_error_is_noop(self):
        """Missing HybridDeviceOptimizer support is a no-op."""
        original_import = builtins.__import__

        def _raise_for_hdo(
            name: str,
            globals_: dict[str, object] | None = None,
            locals_: dict[str, object] | None = None,
            fromlist: tuple[str, ...] = (),
            level: int = 0,
        ) -> object:
            if name == "megatron.core.optimizer.cpu_offloading.hybrid_optimizer":
                raise ImportError("HybridDeviceOptimizer unavailable")
            return original_import(name, globals_, locals_, fromlist, level)

        with patch("builtins.__import__", side_effect=_raise_for_hdo):
            assert sync_hybrid_device_optimizer_fp32_master_copies(_PlainDistribOpt()) is False

    def test_chained_optimizer_walks_each_sub_opt(self):
        """A ChainedOptimizer dispatches to every sub-optimizer, syncing HDO ones."""
        model_param = torch.full((2,), 7.0, dtype=torch.bfloat16)
        shard_main_param = torch.zeros(2, dtype=torch.float32)

        # No level-2/level-3 attrs: helper should still sync level 1 and return True.
        hdo_distrib_opt = _FakeDistribOpt(
            model_param=model_param,
            shard_main_param=shard_main_param,
            inner=_FakeHDO(),
        )
        chained = _ChainedOpt([_PlainDistribOpt(), hdo_distrib_opt])

        with patch(
            "megatron.core.optimizer.cpu_offloading.hybrid_optimizer.HybridDeviceOptimizer",
            _FakeHDO,
        ):
            synced = sync_hybrid_device_optimizer_fp32_master_copies(chained)

        assert synced is True
        assert torch.allclose(shard_main_param, torch.full((2,), 7.0, dtype=torch.float32))

    def test_skips_none_shard_main_param(self):
        """Level-1 entries with a ``None`` shard_main_param are skipped without raising."""
        model_param = torch.full((4,), 3.0, dtype=torch.bfloat16)
        distrib_opt = _FakeDistribOpt(
            model_param=model_param,
            shard_main_param=None,
            inner=_FakeHDO(),
        )

        with patch(
            "megatron.core.optimizer.cpu_offloading.hybrid_optimizer.HybridDeviceOptimizer",
            _FakeHDO,
        ):
            synced = sync_hybrid_device_optimizer_fp32_master_copies(distrib_opt)

        assert synced is True
