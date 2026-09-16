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

from megatron.bridge.peft.lora import get_lora_plus_config_overrides
from megatron.bridge.training.config import SchedulerConfig
from megatron.bridge.training.optim import (
    memory_efficient_fp32_optimizer_state_loading,
    memory_efficient_precision_aware_optimizer_state_checkpointing,
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
        master_weight_dtype: torch.dtype = torch.float32,
        exp_avg_dtype: torch.dtype = torch.float32,
        exp_avg_sq_dtype: torch.dtype | None = None,
        store_param_remainders: bool = False,
    ) -> None:
        super().__init__([param], {"lr": 1e-3})
        self.master_weights = master_weights
        self.master_weight_dtype = master_weight_dtype
        self.exp_avg_dtype = exp_avg_dtype
        self.exp_avg_sq_dtype = exp_avg_dtype if exp_avg_sq_dtype is None else exp_avg_sq_dtype
        self.store_param_remainders = store_param_remainders
        self.name_to_dtype_map = {"exp_avg": exp_avg_dtype, "exp_avg_sq": exp_avg_dtype}
        self.override_load_calls = 0
        self.get_unscaled_state_calls = 0

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self.override_load_calls += 1
        super().load_state_dict(state_dict)

    def get_unscaled_state(
        self,
        param: torch.nn.Parameter,
        state_name: str,
        skip_unscale: bool = False,
        *,
        multiplier: float = 1.0,
    ) -> torch.Tensor:
        del skip_unscale
        self.get_unscaled_state_calls += 1
        return self.state[param][state_name].float() * multiplier


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


class TestMemoryEfficientPrecisionAwareOptimizerStateCheckpointing:
    """Tests for CPU staging of portable precision-aware Adam checkpoint state."""

    @staticmethod
    def _distributed_optimizer(
        *,
        state_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[_FakeDistribOpt, _FakeFusedAdam, torch.Tensor]:
        param = torch.zeros(4, dtype=torch.bfloat16)
        inner = _FakeFusedAdam(param, master_weights=True, exp_avg_dtype=state_dtype)
        inner.state[param] = {
            "exp_avg": torch.ones(4, dtype=state_dtype),
            "exp_avg_sq": torch.full((4,), 2.0, dtype=state_dtype),
        }
        distributed = _FakeDistribOpt(model_param=param, shard_main_param=param, inner=inner)
        distributed.config.use_precision_aware_optimizer = True
        return distributed, inner, param

    def test_stages_unscaled_state_on_cpu_and_restores_method(self):
        distributed, inner, param = self._distributed_optimizer()

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True) as patched:
                state = inner.get_unscaled_state(param, "exp_avg")
                assert patched == 1
                assert state.device.type == "cpu"

            assert "get_unscaled_state" not in inner.__dict__
            inner.get_unscaled_state(param, "exp_avg_sq")

        assert inner.get_unscaled_state_calls == 2

    def test_forwards_positional_and_keyword_arguments(self):
        """The wrapper stays compatible when TE extends its accessor signature."""
        distributed, inner, param = self._distributed_optimizer()

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True):
                state = inner.get_unscaled_state(param, "exp_avg", True, multiplier=3.0)

        torch.testing.assert_close(state, torch.full((4,), 3.0))

    def test_rejects_non_tensor_state_and_restores_instance_method(self):
        """TE return-contract drift fails clearly without leaking the patch."""
        distributed, inner, param = self._distributed_optimizer()
        original_instance_method = MagicMock(return_value="not a tensor")
        inner.get_unscaled_state = original_instance_method

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            pytest.raises(TypeError, match="must return a torch.Tensor"),
        ):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True):
                inner.get_unscaled_state(param, "exp_avg")

        assert inner.__dict__["get_unscaled_state"] is original_instance_method

    def test_rejects_missing_te_state_accessor(self):
        """An incompatible TE API fails before checkpoint construction begins."""
        distributed, inner, _ = self._distributed_optimizer()
        inner.get_unscaled_state = None

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            pytest.raises(RuntimeError, match=r"FusedAdam\.get_unscaled_state\(\).*callable"),
        ):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True):
                pass

    @pytest.mark.parametrize("incompatibility", ["fp32", "cpu_offload", "fsdp", "stub"])
    def test_does_not_patch_incompatible_optimizer(self, incompatibility: str):
        state_dtype = torch.float32 if incompatibility == "fp32" else torch.bfloat16
        distributed, inner, _ = self._distributed_optimizer(state_dtype=state_dtype)
        if incompatibility == "cpu_offload":
            distributed.config.optimizer_cpu_offload = True
        elif incompatibility == "fsdp":
            distributed.ddp_config.use_megatron_fsdp = True
        elif incompatibility == "stub":
            distributed.is_stub_optimizer = True

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True) as patched:
                assert patched == 0

        assert "get_unscaled_state" not in inner.__dict__

    def test_patches_all_eligible_chained_optimizers(self):
        distributed_optimizers = [self._distributed_optimizer()[0] for _ in range(2)]

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(
                _ChainedOpt(distributed_optimizers), enabled=True
            ) as patched:
                assert patched == 2
                assert all("get_unscaled_state" in opt.optimizer.__dict__ for opt in distributed_optimizers)

        assert all("get_unscaled_state" not in opt.optimizer.__dict__ for opt in distributed_optimizers)

    def test_te_unavailable_and_none_optimizer_are_noops(self):
        distributed, inner, _ = self._distributed_optimizer()

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=None):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True) as patched:
                assert patched == 0
        with memory_efficient_precision_aware_optimizer_state_checkpointing(None, enabled=True) as patched:
            assert patched == 0

        assert "get_unscaled_state" not in inner.__dict__

    def test_disabled_is_noop(self):
        distributed, inner, _ = self._distributed_optimizer()

        with patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=False) as patched:
                assert patched == 0

        assert "get_unscaled_state" not in inner.__dict__

    def test_restores_method_when_checkpointing_raises(self):
        distributed, inner, _ = self._distributed_optimizer()

        with (
            patch("megatron.bridge.training.optim._get_te_fused_adam_class", return_value=_FakeFusedAdam),
            pytest.raises(RuntimeError, match="save failed"),
        ):
            with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True):
                raise RuntimeError("save failed")

        assert "get_unscaled_state" not in inner.__dict__

    @pytest.mark.run_only_on("gpu")
    def test_real_te_fused_adam_stages_state_and_restores_method(self):
        """The pinned TE precision-aware optimizer returns CPU checkpoint state."""
        te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
        fused_adam_class = te_optimizers.FusedAdam
        param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16, device="cuda"))
        inner = fused_adam_class(
            [param],
            master_weights=True,
            master_weight_dtype=torch.float16,
            exp_avg_dtype=torch.bfloat16,
            exp_avg_sq_dtype=torch.bfloat16,
            use_decoupled_grad=True,
        )
        inner.initialize_state(param, store_param_remainders=False)
        distributed = _FakeDistribOpt(model_param=param, shard_main_param=param, inner=inner)
        distributed.config.use_precision_aware_optimizer = True

        assert "get_unscaled_state" not in inner.__dict__
        with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True) as patched:
            portable_state = next(iter(inner.state_dict()["state"].values()))
            assert patched == 1
            assert portable_state
            assert all(state.device.type == "cpu" for state in portable_state.values())

        assert "get_unscaled_state" not in inner.__dict__
        assert inner.get_unscaled_state(param, "exp_avg").device.type == "cuda"

    @pytest.mark.run_only_on("gpu")
    def test_real_te_fused_adam_all_fp32_state_is_not_patched(self):
        """The real TE optimizer keeps its native path when no expansion is needed."""
        te_optimizers = pytest.importorskip("transformer_engine.pytorch.optimizers")
        fused_adam_class = te_optimizers.FusedAdam
        param = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16, device="cuda"))
        inner = fused_adam_class(
            [param],
            master_weights=True,
            master_weight_dtype=torch.float32,
            exp_avg_dtype=torch.float32,
            exp_avg_sq_dtype=torch.float32,
            use_decoupled_grad=True,
        )
        inner.initialize_state(param, store_param_remainders=False)
        distributed = _FakeDistribOpt(model_param=param, shard_main_param=param, inner=inner)
        distributed.config.use_precision_aware_optimizer = True

        with memory_efficient_precision_aware_optimizer_state_checkpointing(distributed, enabled=True) as patched:
            native_state = next(iter(inner.state_dict()["state"].values()))
            assert patched == 0
            assert all(state.device.type == "cuda" for state in native_state.values())

        assert "get_unscaled_state" not in inner.__dict__


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


class TestGetLoraPlusConfigOverrides:
    """Tests for ``get_lora_plus_config_overrides`` (LoRA+ A/B LR split)."""

    def _make_config(self, lr=3e-5, min_lr=0.0, **kwargs):
        return OptimizerConfig(optimizer="adam", lr=lr, min_lr=min_lr, bf16=True, **kwargs)

    def _get_override(self, overrides, name):
        """Find a ParamKey by its ``name`` glob and return its ParamGroupOverride."""
        for key, val in overrides.items():
            if (
                getattr(key, "name", None) == name
                and not getattr(key, "predicate", None)
                and not getattr(key, "with_name_predicate", None)
                and not getattr(key, "attr", None)
            ):
                return val
        return None

    @pytest.mark.parametrize("ratio", [0, -1, -0.5])
    def test_non_positive_ratio_raises(self, ratio):
        with pytest.raises(ValueError, match="lora_plus_ratio must be > 0"):
            get_lora_plus_config_overrides(self._make_config(), ratio)

    def test_a_keeps_base_lr(self):
        overrides = get_lora_plus_config_overrides(self._make_config(lr=3e-5, min_lr=1e-6), 16.0)
        a = self._get_override(overrides, "*.linear_in.weight")
        assert a is not None
        assert a["max_lr"] == 3e-5
        assert a["min_lr"] == 1e-6

    def test_b_is_ratio_times_base_lr(self):
        overrides = get_lora_plus_config_overrides(self._make_config(lr=3e-5, min_lr=1e-6), 16.0)
        b = self._get_override(overrides, "*.linear_out.weight")
        assert b is not None
        assert b["max_lr"] == pytest.approx(3e-5 * 16)
        assert b["min_lr"] == pytest.approx(1e-6 * 16)

    def test_none_min_lr_defaults_to_zero(self):
        cfg = OptimizerConfig(optimizer="adam", lr=3e-5, min_lr=None, bf16=True)
        overrides = get_lora_plus_config_overrides(cfg, 16.0)
        a = self._get_override(overrides, "*.linear_in.weight")
        b = self._get_override(overrides, "*.linear_out.weight")
        assert a["min_lr"] == 0.0
        assert b["min_lr"] == 0.0
        assert b["max_lr"] == pytest.approx(3e-5 * 16)

    def test_b_schedule_is_ratio_scaled_copy_of_a(self):
        """B's max_lr/min_lr are both exactly ratio x A's — the whole schedule scales."""
        ratio = 8.0
        overrides = get_lora_plus_config_overrides(self._make_config(lr=5e-5, min_lr=2e-6), ratio)
        a = self._get_override(overrides, "*.linear_in.weight")
        b = self._get_override(overrides, "*.linear_out.weight")
        assert b["max_lr"] / a["max_lr"] == pytest.approx(ratio)
        assert b["min_lr"] / a["min_lr"] == pytest.approx(ratio)

    def test_ratio_one_keeps_equal_lr(self):
        """ratio == 1.0 still produces valid overrides with A == B (no-op split)."""
        overrides = get_lora_plus_config_overrides(self._make_config(lr=3e-5, min_lr=0.0), 1.0)
        a = self._get_override(overrides, "*.linear_in.weight")
        b = self._get_override(overrides, "*.linear_out.weight")
        assert a["max_lr"] == b["max_lr"] == 3e-5
        assert a["min_lr"] == b["min_lr"] == 0.0

    def test_preserves_standard_bias_wd_override(self):
        """The standard bias/1-D wd_mult=0.0 override is preserved alongside LoRA+."""
        overrides = get_lora_plus_config_overrides(self._make_config(), 16.0)
        wd_skips = [v for v in overrides.values() if v.get("wd_mult") == 0.0]
        assert wd_skips, "standard wd_mult=0.0 override should be preserved"

    def test_preserves_decoupled_lr_override(self):
        """A configured decoupled_lr is preserved by get_standard_config_overrides."""
        cfg = self._make_config(decoupled_lr=1e-4, decoupled_min_lr=1e-5)
        overrides = get_lora_plus_config_overrides(cfg, 16.0)
        # The decoupled entry is keyed by attr, not name.
        decoupled = [v for k, v in overrides.items() if getattr(k, "attr", None)]
        assert decoupled, "decoupled_lr override should be preserved"
        assert decoupled[0]["max_lr"] == 1e-4

    def test_returns_param_key_keyed_mapping(self):
        """Result is keyed by ParamKey; values are dict-like (ParamGroupOverride is a TypedDict)."""
        overrides = get_lora_plus_config_overrides(self._make_config(), 16.0)
        assert all(isinstance(k, ParamKey) for k in overrides)
        assert all(isinstance(v, dict) for v in overrides.values())
        # Only the LoRA A/B entries carry the LR fields; standard entries may not.
        assert self._get_override(overrides, "*.linear_in.weight")["max_lr"] is not None
        assert self._get_override(overrides, "*.linear_out.weight")["max_lr"] is not None

    def test_only_two_lora_name_globs_added(self):
        """Exactly two name-glob entries are added: linear_in and linear_out."""
        overrides = get_lora_plus_config_overrides(self._make_config(), 16.0)
        name_keys = [k for k in overrides if getattr(k, "name", None) and not getattr(k, "attr", None)]
        names = {k.name for k in name_keys}
        assert {"*.linear_in.weight", "*.linear_out.weight"}.issubset(names)
