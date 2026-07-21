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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.optimizer import OptimizerConfig, ParamGroupOverride, ParamKey

from megatron.bridge.training.config import SchedulerConfig
from megatron.bridge.training.optim import (
    memory_efficient_fp32_optimizer_state_loading,
    sync_hybrid_device_optimizer_fp32_master_copies,
)


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


class _FakeFusedAdam(torch.optim.Optimizer):
    """CPU stand-in for TE FusedAdam with an observable override loader."""

    def __init__(
        self,
        param: torch.Tensor,
        *,
        master_weights: bool = False,
        exp_avg_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__([param], {"lr": 1e-3})
        self.master_weights = master_weights
        self.store_param_remainders = False
        self.name_to_dtype_map = {"exp_avg": exp_avg_dtype, "exp_avg_sq": exp_avg_dtype}
        self.override_load_calls = 0

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.override_load_calls += 1
        super().load_state_dict(state_dict)


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
        self.is_stub_optimizer = False
        self.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
        self.config = SimpleNamespace(use_precision_aware_optimizer=False, optimizer_cpu_offload=False)

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


class _FakeLayerWiseChildOpt:
    """Stand-in for a LayerWiseDistributedOptimizer's wrapped child optimizer."""

    def __init__(self, inner: torch.optim.Optimizer) -> None:
        self.optimizer = inner


class TestMemoryEfficientFp32OptimizerStateLoading:
    """Tests for the scoped TE FusedAdam checkpoint-load fast path."""

    @staticmethod
    def _distributed_optimizer(
        *,
        param_dtype: torch.dtype = torch.float32,
        master_weights: bool = False,
        state_dtype: torch.dtype = torch.float32,
    ) -> tuple[_FakeDistribOpt, _FakeFusedAdam, torch.Tensor]:
        param = torch.zeros(4, dtype=param_dtype)
        inner = _FakeFusedAdam(param, master_weights=master_weights, exp_avg_dtype=state_dtype)
        distributed = _FakeDistribOpt(
            model_param=torch.zeros(4, dtype=torch.bfloat16),
            shard_main_param=param,
            inner=inner,
        )
        return distributed, inner, param

    @staticmethod
    def _state_dict(
        inner: _FakeFusedAdam,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[dict[str, object], torch.Tensor]:
        state_dict = inner.state_dict()
        exp_avg = torch.ones(4, dtype=dtype)
        state_dict["state"] = {
            0: {
                "exp_avg": exp_avg,
                "exp_avg_sq": torch.full((4,), 2.0, dtype=dtype),
            }
        }
        return state_dict, exp_avg

    def test_uses_base_loader_without_reallocating_fp32_state(self):
        """FP32 distributed shards adopt supplied state tensors directly."""
        distributed, inner, param = self._distributed_optimizer()
        state_dict, exp_avg = self._state_dict(inner)

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            patch("megatron.bridge.training.optim.torch.cuda.empty_cache") as mock_empty_cache,
        ):
            with memory_efficient_fp32_optimizer_state_loading(distributed) as patched:
                inner.load_state_dict(state_dict)
                mock_empty_cache.assert_not_called()

            assert patched == 1
            assert inner.override_load_calls == 0
            assert inner.state[param]["exp_avg"] is exp_avg
            mock_empty_cache.assert_called_once_with()

            inner.load_state_dict(state_dict)

        assert inner.override_load_calls == 1

    def test_falls_back_for_non_fp32_incoming_state(self):
        """A non-FP32 state dict retains Transformer Engine's conversion path."""
        distributed, inner, _ = self._distributed_optimizer()
        state_dict, _ = self._state_dict(inner, dtype=torch.bfloat16)

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            patch("megatron.bridge.training.optim.torch.cuda.empty_cache"),
            memory_efficient_fp32_optimizer_state_loading(distributed) as patched,
        ):
            inner.load_state_dict(state_dict)

        assert patched == 1
        assert inner.override_load_calls == 1

    @pytest.mark.parametrize(
        ("param_dtype", "master_weights", "state_dtype"),
        [
            (torch.bfloat16, False, torch.float32),
            (torch.float32, True, torch.float32),
            (torch.float32, False, torch.bfloat16),
        ],
    )
    def test_does_not_patch_incompatible_fused_adam(
        self,
        param_dtype: torch.dtype,
        master_weights: bool,
        state_dtype: torch.dtype,
    ):
        """Mixed parameters, master weights, and compressed state stay on TE's path."""
        distributed, inner, _ = self._distributed_optimizer(
            param_dtype=param_dtype,
            master_weights=master_weights,
            state_dtype=state_dtype,
        )

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_fp32_optimizer_state_loading(distributed) as patched:
                pass

        assert patched == 0
        assert "load_state_dict" not in inner.__dict__

    @pytest.mark.parametrize("incompatibility", ["precision_aware", "cpu_offload", "fsdp", "stub"])
    def test_does_not_patch_incompatible_distributed_optimizer(self, incompatibility: str):
        """Special distributed optimizer modes retain their existing loader."""
        distributed, inner, _ = self._distributed_optimizer()
        if incompatibility == "precision_aware":
            distributed.config.use_precision_aware_optimizer = True
        elif incompatibility == "cpu_offload":
            distributed.config.optimizer_cpu_offload = True
        elif incompatibility == "fsdp":
            distributed.ddp_config.use_megatron_fsdp = True
        else:
            distributed.is_stub_optimizer = True

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_fp32_optimizer_state_loading(distributed) as patched:
                pass

        assert patched == 0
        assert "load_state_dict" not in inner.__dict__

    def test_patches_all_eligible_chained_optimizers(self):
        """Dense and expert DistributedOptimizers both use the scoped loader."""
        distributed_optimizers = [self._distributed_optimizer()[0] for _ in range(2)]

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            patch("megatron.bridge.training.optim.torch.cuda.empty_cache"),
        ):
            with memory_efficient_fp32_optimizer_state_loading(_ChainedOpt(distributed_optimizers)) as patched:
                assert all("load_state_dict" in opt.optimizer.__dict__ for opt in distributed_optimizers)

        assert patched == 2
        assert all("load_state_dict" not in opt.optimizer.__dict__ for opt in distributed_optimizers)

    def test_restores_methods_when_later_optimizer_setup_raises(self):
        """A partial chained-optimizer setup is rolled back when inspection fails."""
        first_distributed, first_inner, _ = self._distributed_optimizer()
        second_distributed, second_inner, _ = self._distributed_optimizer()
        second_inner.param_groups = [{}]

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            patch("megatron.bridge.training.optim.torch.cuda.empty_cache") as mock_empty_cache,
            pytest.raises(KeyError, match="params"),
        ):
            with memory_efficient_fp32_optimizer_state_loading(_ChainedOpt([first_distributed, second_distributed])):
                pass

        assert "load_state_dict" not in first_inner.__dict__
        mock_empty_cache.assert_called_once_with()

    def test_te_unavailable_is_noop(self):
        """An environment without Transformer Engine retains the existing loader."""
        distributed, inner, _ = self._distributed_optimizer()

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=None),
            patch("megatron.bridge.training.optim.torch.cuda.empty_cache") as mock_empty_cache,
            memory_efficient_fp32_optimizer_state_loading(distributed) as patched,
        ):
            pass

        assert patched == 0
        assert "load_state_dict" not in inner.__dict__
        mock_empty_cache.assert_not_called()

    def test_does_not_patch_layerwise_optimizer_children(self):
        """LayerWise optimizer children lack distributed FP32 shards and remain unchanged."""
        _, inner, _ = self._distributed_optimizer()
        layerwise = _ChainedOpt([_FakeLayerWiseChildOpt(inner)])

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_fp32_optimizer_state_loading(layerwise) as patched:
                pass

        assert patched == 0
        assert "load_state_dict" not in inner.__dict__

    def test_restores_methods_when_loading_raises(self):
        """The scoped replacement is removed when checkpoint loading fails."""
        distributed, inner, _ = self._distributed_optimizer()

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            patch("megatron.bridge.training.optim.torch.cuda.empty_cache"),
            pytest.raises(RuntimeError, match="load failed"),
        ):
            with memory_efficient_fp32_optimizer_state_loading(distributed):
                raise RuntimeError("load failed")

        assert "load_state_dict" not in inner.__dict__

    def test_none_optimizer_is_noop(self):
        """A missing optimizer is a no-op."""
        with memory_efficient_fp32_optimizer_state_loading(None) as patched:
            assert patched == 0


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
