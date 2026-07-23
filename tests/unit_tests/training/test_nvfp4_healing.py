# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the NVFP4 FP8-healing callback."""

import contextlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.bridge.training.callbacks import CallbackManager
from megatron.bridge.training.nvfp4_healing import (
    VALID_HEALING_RECIPES,
    NVFP4HealingCallback,
    NVFP4HealingConfig,
    _unwrap_model_chunk,
)


def make_config(**overrides):
    defaults = {"healing_iter": 3, "healing_recipe": "delayed"}
    defaults.update(overrides)
    return NVFP4HealingConfig(**defaults)


class TestNVFP4HealingConfig:
    def test_valid_config_defaults(self):
        cfg = make_config()
        assert cfg.healing_iter == 3
        assert cfg.healing_recipe == "delayed"
        assert cfg.pre_quantize_base_weights is False
        assert cfg.store_quantized_params_on_gpu is False
        assert cfg.fp8_amax_history_len == 1024
        assert cfg.fp8_amax_compute_algo == "max"
        assert cfg.reduce_amax is True
        assert cfg.reset_cuda_graph_warmup is False

    def test_mxfp8_recipe_accepted(self):
        assert make_config(healing_recipe="mxfp8").healing_recipe == "mxfp8"

    def test_valid_recipes_constant(self):
        assert VALID_HEALING_RECIPES == ("delayed", "mxfp8")

    def test_invalid_recipe_raises(self):
        with pytest.raises(ValueError, match="healing_recipe"):
            make_config(healing_recipe="FP8_DS")

    def test_invalid_healing_iter_raises(self):
        with pytest.raises(ValueError, match="healing_iter"):
            make_config(healing_iter=0)


def make_context(step):
    ctx = Mock()
    ctx.state.train_state.step = step
    return ctx


class TestHealingTrigger:
    def test_fires_exactly_at_healing_iter(self):
        cb = NVFP4HealingCallback(make_config(healing_iter=3))
        with patch.object(cb, "_apply_healing") as apply:
            cb.on_train_step_end(make_context(step=1))  # step+1 == 2 != 3
            apply.assert_not_called()
            assert cb.healed is False
            cb.on_train_step_end(make_context(step=2))  # step+1 == 3
            apply.assert_called_once()
            assert cb.healed is True

    def test_fires_only_once(self):
        cb = NVFP4HealingCallback(make_config(healing_iter=1))
        with patch.object(cb, "_apply_healing") as apply:
            cb.on_train_step_end(make_context(step=0))
            cb.on_train_step_end(make_context(step=0))
            assert apply.call_count == 1

    def test_does_not_fire_past_healing_iter(self):
        cb = NVFP4HealingCallback(make_config(healing_iter=2))
        with patch.object(cb, "_apply_healing") as apply:
            cb.on_train_step_end(make_context(step=5))
            apply.assert_not_called()
            assert cb.healed is False


def _has_te_with_nvfp4():
    """True when a real Transformer Engine >= 2.7.0.dev0 is importable (same gate as production)."""
    try:
        import transformer_engine  # noqa: F401
        from megatron.core.utils import is_te_min_version

        return is_te_min_version("2.7.0.dev0")
    except Exception:
        return False


requires_te = pytest.mark.skipif(not _has_te_with_nvfp4(), reason="requires Transformer Engine >= 2.7.0.dev0")


class TestRecipePatchAndRestore:
    def test_patch_replaces_and_on_train_end_restores(self):
        import megatron.core.fp4_utils as fp4_utils

        original = fp4_utils.get_fp4_recipe
        cb = NVFP4HealingCallback(make_config())
        sentinel = object()
        try:
            cb._patch_fp4_recipe(sentinel)
            assert fp4_utils.get_fp4_recipe(Mock()) is sentinel
            cb.on_train_end(Mock())
            assert fp4_utils.get_fp4_recipe is original
        finally:
            fp4_utils.get_fp4_recipe = original

    def test_double_patch_keeps_true_original(self):
        import megatron.core.fp4_utils as fp4_utils

        original = fp4_utils.get_fp4_recipe
        cb = NVFP4HealingCallback(make_config())
        try:
            cb._patch_fp4_recipe(object())
            cb._patch_fp4_recipe(object())
            cb._restore_fp4_recipe()
            assert fp4_utils.get_fp4_recipe is original
        finally:
            fp4_utils.get_fp4_recipe = original

    def test_restore_without_patch_is_noop(self):
        import megatron.core.fp4_utils as fp4_utils

        original = fp4_utils.get_fp4_recipe
        cb = NVFP4HealingCallback(make_config())
        cb.on_train_end(Mock())
        assert fp4_utils.get_fp4_recipe is original


class TestHealingRecipeConstruction:
    @requires_te
    def test_delayed_recipe_fields(self):
        from transformer_engine.common.recipe import DelayedScaling

        cb = NVFP4HealingCallback(
            make_config(healing_recipe="delayed", fp8_amax_history_len=16, fp8_amax_compute_algo="max")
        )
        model_config = SimpleNamespace(fp8_dot_product_attention=False)
        recipe = cb._build_healing_recipe(model_config)
        assert isinstance(recipe, DelayedScaling)
        assert recipe.amax_history_len == 16
        assert recipe.amax_compute_algo == "max"

    @requires_te
    def test_mxfp8_recipe(self):
        from transformer_engine.common.recipe import MXFP8BlockScaling

        cb = NVFP4HealingCallback(make_config(healing_recipe="mxfp8"))
        recipe = cb._build_healing_recipe(SimpleNamespace(fp8_dot_product_attention=False))
        assert isinstance(recipe, MXFP8BlockScaling)


class TestCudaGraphReset:
    def test_resets_graph_and_result_state(self, monkeypatch):
        from megatron.core.full_cuda_graph import FullCudaGraphWrapper

        monkeypatch.setattr(FullCudaGraphWrapper, "cuda_graph", {"training": "g", "validation": "g"}, raising=False)
        monkeypatch.setattr(FullCudaGraphWrapper, "result", {"training": "r", "validation": "r"}, raising=False)
        monkeypatch.setattr(FullCudaGraphWrapper, "curr_iteration", {"training": 5, "validation": 4}, raising=False)

        cb = NVFP4HealingCallback(make_config())
        cb._reset_cuda_graphs()

        assert FullCudaGraphWrapper.cuda_graph == {"training": None, "validation": None}
        assert FullCudaGraphWrapper.result == {"training": None, "validation": None}
        assert FullCudaGraphWrapper.curr_iteration == {"training": 5, "validation": 4}

    def test_reset_warmup_counters_when_configured(self, monkeypatch):
        from megatron.core.full_cuda_graph import FullCudaGraphWrapper

        monkeypatch.setattr(FullCudaGraphWrapper, "cuda_graph", {"training": "g", "validation": "g"}, raising=False)
        monkeypatch.setattr(FullCudaGraphWrapper, "result", {"training": "r", "validation": "r"}, raising=False)
        monkeypatch.setattr(FullCudaGraphWrapper, "curr_iteration", {"training": 5, "validation": 4}, raising=False)

        cb = NVFP4HealingCallback(make_config(reset_cuda_graph_warmup=True))
        cb._reset_cuda_graphs()

        assert FullCudaGraphWrapper.curr_iteration == {"training": 0, "validation": 0}


class FakeLayer(torch.nn.Module):
    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.linear = torch.nn.Linear(4, 4)


def make_fake_chunk(layer_numbers, num_layers=None, first_last_bf16=False, start_bf16=0, end_bf16=0, op_fuser=False):
    layers = torch.nn.ModuleList([FakeLayer(n) for n in layer_numbers])
    config = SimpleNamespace(
        num_layers=num_layers if num_layers is not None else len(layer_numbers),
        first_last_layers_bf16=first_last_bf16,
        num_layers_at_start_in_bf16=start_bf16,
        num_layers_at_end_in_bf16=end_bf16,
        use_transformer_engine_op_fuser=op_fuser,
        fp8_dot_product_attention=False,
    )
    return SimpleNamespace(decoder=SimpleNamespace(layers=layers), config=config)


def linear_is_target(module):
    return isinstance(module, torch.nn.Linear)


def patch_target_module():
    return patch.object(NVFP4HealingCallback, "_is_target_module", staticmethod(linear_is_target))


class TestUnwrapModelChunk:
    def test_returns_object_without_module_attr(self):
        chunk = make_fake_chunk([1])
        assert _unwrap_model_chunk(chunk) is chunk

    def test_follows_nested_module_chain(self):
        inner = make_fake_chunk([1])
        wrapped = SimpleNamespace(module=SimpleNamespace(module=inner))
        assert _unwrap_model_chunk(wrapped) is inner


class TestLayerIteration:
    def test_yields_all_target_modules(self):
        model = [make_fake_chunk([1, 2, 3])]
        cb = NVFP4HealingCallback(make_config())
        with patch_target_module():
            found = list(cb._iter_quantizable_modules(model))
        assert len(found) == 3
        assert all(isinstance(m, torch.nn.Linear) for _, m in found)

    def test_skips_first_and_last_bf16_layers(self):
        model = [make_fake_chunk([1, 2, 3, 4], first_last_bf16=True, start_bf16=1, end_bf16=1)]
        cb = NVFP4HealingCallback(make_config())
        with patch_target_module():
            found = list(cb._iter_quantizable_modules(model))
        assert {layer.layer_number for layer, _ in found} == {2, 3}

    def test_bf16_skip_ignored_when_flag_off(self):
        model = [make_fake_chunk([1, 2, 3, 4], first_last_bf16=False, start_bf16=1, end_bf16=1)]
        cb = NVFP4HealingCallback(make_config())
        with patch_target_module():
            found = list(cb._iter_quantizable_modules(model))
        assert len(found) == 4

    def test_pipeline_rank_uses_global_layer_numbers(self):
        # Simulates the last PP rank holding global layers 3..4 of a 4-layer model
        # with the final layer kept in BF16.
        model = [make_fake_chunk([3, 4], num_layers=4, first_last_bf16=True, start_bf16=1, end_bf16=1)]
        cb = NVFP4HealingCallback(make_config())
        with patch_target_module():
            found = list(cb._iter_quantizable_modules(model))
        assert {layer.layer_number for layer, _ in found} == {3}

    def test_iterates_across_virtual_pipeline_chunks(self):
        chunk_a = make_fake_chunk([1, 2], num_layers=4)
        chunk_b = make_fake_chunk([3, 4], num_layers=4)
        cb = NVFP4HealingCallback(make_config())
        with patch_target_module():
            found = list(cb._iter_quantizable_modules([chunk_a, chunk_b]))
        assert [layer.layer_number for layer, _ in found] == [1, 2, 3, 4]

    def test_unsupported_model_structure_raises(self):
        cb = NVFP4HealingCallback(make_config())
        with pytest.raises(RuntimeError, match="decoder.layers"):
            list(cb._iter_quantizable_modules([SimpleNamespace(config=SimpleNamespace())]))


class TestPreQuantization:
    def test_rejects_trainable_weights(self):
        model = [make_fake_chunk([1])]
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True))
        with (
            patch_target_module(),
            patch.object(cb, "_build_quantizers", return_value=(Mock(), Mock())),
        ):
            with pytest.raises(ValueError, match="frozen"):
                cb._pre_quantize(model)

    def test_stashes_fp8_and_replaces_weights_with_nvfp4(self):
        model = [make_fake_chunk([1, 2])]
        for layer in model[0].decoder.layers:
            layer.linear.weight.requires_grad_(False)
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True, store_quantized_params_on_gpu=True))
        fake_nvfp4 = Mock(side_effect=lambda w: torch.zeros_like(w))
        fake_fp8 = Mock(side_effect=lambda w: torch.ones_like(w))
        with (
            patch_target_module(),
            patch.object(cb, "_build_quantizers", return_value=(fake_nvfp4, fake_fp8)),
            patch.object(cb, "_validate_quantized"),
        ):
            cb._pre_quantize(model)

        assert len(cb._fp8_stash) == 2
        assert all(torch.all(stashed == 1) for stashed in cb._fp8_stash)
        for layer in model[0].decoder.layers:
            weight = layer.linear.weight
            assert isinstance(weight, torch.nn.Parameter)
            assert weight.requires_grad is False
            assert torch.all(weight == 0)

    def test_on_data_init_start_noop_without_flag(self):
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=False))
        ctx = Mock()
        with patch.object(cb, "_pre_quantize") as pre_quantize:
            cb.on_data_init_start(ctx)
            pre_quantize.assert_not_called()


class TestValidateQuantized:
    def test_nvfp4_missing_storage_raises(self):
        cb = NVFP4HealingCallback(make_config())
        bad = SimpleNamespace(_rowwise_data=None, _columnwise_data=torch.zeros(1))
        with pytest.raises(RuntimeError, match="rowwise"):
            cb._validate_quantized(bad, "nvfp4")

    def test_delayed_missing_data_raises(self):
        cb = NVFP4HealingCallback(make_config())
        with pytest.raises(RuntimeError, match="data"):
            cb._validate_quantized(SimpleNamespace(_data=None), "delayed")

    def test_valid_tensors_pass(self):
        cb = NVFP4HealingCallback(make_config())
        cb._validate_quantized(SimpleNamespace(_rowwise_data=torch.zeros(1), _columnwise_data=torch.zeros(1)), "mxfp8")
        cb._validate_quantized(SimpleNamespace(_data=torch.zeros(1)), "delayed")


class TestApplyHealing:
    def test_missing_stash_raises_when_prequantize_configured(self):
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True))
        ctx = make_context(step=2)
        ctx.model = [make_fake_chunk([1])]
        with (
            patch.object(cb, "_reset_cuda_graphs"),
            patch.object(cb, "_swap_in_fp8_weights"),
            patch.object(cb, "_build_healing_recipe", return_value=object()),
            patch.object(cb, "_patch_fp4_recipe"),
        ):
            with pytest.raises(RuntimeError, match="stash"):
                cb._apply_healing(ctx)

    def test_healing_order_with_prequantized_weights(self):
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True))
        cb._fp8_stash = [object()]
        ctx = make_context(step=2)
        ctx.model = [make_fake_chunk([1])]
        call_order = []
        recipe = object()
        with (
            patch.object(cb, "_reset_cuda_graphs", side_effect=lambda: call_order.append("reset")),
            patch.object(cb, "_swap_in_fp8_weights", side_effect=lambda m: call_order.append("swap")),
            patch.object(cb, "_build_healing_recipe", return_value=recipe),
            patch.object(cb, "_patch_fp4_recipe", side_effect=lambda r: call_order.append("patch")),
        ):
            cb._apply_healing(ctx)
        assert call_order == ["reset", "swap", "patch"]

    def test_recipe_switch_only_without_prequantize(self):
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=False))
        ctx = make_context(step=2)
        ctx.model = [make_fake_chunk([1])]
        with (
            patch.object(cb, "_reset_cuda_graphs"),
            patch.object(cb, "_swap_in_fp8_weights") as swap,
            patch.object(cb, "_build_healing_recipe", return_value=object()),
            patch.object(cb, "_patch_fp4_recipe") as patch_recipe,
        ):
            cb._apply_healing(ctx)
        swap.assert_not_called()
        patch_recipe.assert_called_once()


class TestSwapInFP8Weights:
    def test_stash_count_mismatch_raises(self):
        model = [make_fake_chunk([1, 2])]
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True))
        cb._fp8_stash = [object()]  # 1 stash entry vs 2 modules
        with patch_target_module():
            with pytest.raises(RuntimeError, match="stash"):
                cb._swap_in_fp8_weights(model)


class TestHookRegistration:
    def test_registers_expected_hooks(self):
        manager = CallbackManager()
        manager.add(NVFP4HealingCallback(make_config()))
        assert manager.has_callbacks("on_data_init_start")
        assert manager.has_callbacks("on_train_step_end")
        assert manager.has_callbacks("on_train_end")
        assert not manager.has_callbacks("on_train_start")
        assert not manager.has_callbacks("on_train_step_start")


class TestRequireTE:
    def test_raises_when_te_version_too_old(self, monkeypatch):
        import megatron.core.utils as mcu

        from megatron.bridge.training.nvfp4_healing import _require_te

        monkeypatch.setattr(mcu, "is_te_min_version", lambda v: False)
        with pytest.raises(RuntimeError, match="Transformer Engine"):
            _require_te()


class TestOnDataInitStart:
    def test_prequantizes_when_enabled(self):
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True))
        ctx = Mock()
        ctx.model = [make_fake_chunk([1])]
        with patch.object(cb, "_pre_quantize") as pre_quantize, patch("torch.cuda.empty_cache"):
            cb.on_data_init_start(ctx)
            pre_quantize.assert_called_once_with(ctx.model)


class TestResetCudaGraphsFallbacks:
    def test_missing_module_is_noop(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "megatron.core.full_cuda_graph":
                raise ImportError("simulated")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        cb = NVFP4HealingCallback(make_config())
        cb._reset_cuda_graphs()  # returns cleanly

    def test_missing_attr_is_noop(self, monkeypatch):
        from megatron.core.full_cuda_graph import FullCudaGraphWrapper

        monkeypatch.delattr(FullCudaGraphWrapper, "cuda_graph", raising=False)
        cb = NVFP4HealingCallback(make_config())
        cb._reset_cuda_graphs()  # returns cleanly


def _inject_fake_te_quantizers(monkeypatch):
    """Inject fake Transformer Engine quantizer modules so _build_quantizers runs CPU-only."""
    tex = types.ModuleType("transformer_engine_torch")
    tex.DType = types.SimpleNamespace(kFloat8E4M3="e4m3")
    f8 = types.ModuleType("transformer_engine.pytorch.tensor.float8_tensor")
    f8.Float8Quantizer = lambda **kwargs: ("float8", kwargs)
    mx = types.ModuleType("transformer_engine.pytorch.tensor.mxfp8_tensor")
    mx.MXFP8Quantizer = lambda dtype: ("mxfp8", dtype)
    nv = types.ModuleType("transformer_engine.pytorch.tensor.nvfp4_tensor")
    nv.NVFP4Quantizer = lambda: "nvfp4"
    monkeypatch.setitem(sys.modules, "transformer_engine_torch", tex)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.float8_tensor", f8)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.mxfp8_tensor", mx)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.nvfp4_tensor", nv)


class TestBuildQuantizers:
    def test_delayed_builds_float8_quantizer(self, monkeypatch):
        import megatron.bridge.training.nvfp4_healing as mod

        monkeypatch.setattr(mod, "_require_te", lambda: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        _inject_fake_te_quantizers(monkeypatch)

        cb = NVFP4HealingCallback(make_config(healing_recipe="delayed"))
        nvfp4_q, fp8_q = cb._build_quantizers()
        assert nvfp4_q == "nvfp4"
        assert fp8_q[0] == "float8"
        assert fp8_q[1]["fp8_dtype"] == "e4m3"

    def test_mxfp8_builds_mxfp8_quantizer(self, monkeypatch):
        import megatron.bridge.training.nvfp4_healing as mod

        monkeypatch.setattr(mod, "_require_te", lambda: None)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        _inject_fake_te_quantizers(monkeypatch)

        cb = NVFP4HealingCallback(make_config(healing_recipe="mxfp8"))
        nvfp4_q, fp8_q = cb._build_quantizers()
        assert nvfp4_q == "nvfp4"
        assert fp8_q == ("mxfp8", "e4m3")


class FakeStorage:
    """Fake quantized storage tracking cpu()/pin_memory()/to() calls."""

    def __init__(self):
        self.pinned = False
        self.moved_to = None

    def cpu(self):
        return self

    def pin_memory(self):
        self.pinned = True
        return self

    def to(self, device, non_blocking):
        self.moved_to = (device, non_blocking)
        return self


class TestStashFP8CopyPinned:
    def test_mxfp8_pins_rowwise_and_columnwise(self):
        cb = NVFP4HealingCallback(make_config(healing_recipe="mxfp8", store_quantized_params_on_gpu=False))
        q = SimpleNamespace(_rowwise_data=FakeStorage(), _columnwise_data=FakeStorage())
        q.clone = lambda: q
        cb._stash_fp8_copy(q)
        assert cb._fp8_stash == [q]
        assert q._rowwise_data.pinned and q._columnwise_data.pinned

    def test_delayed_pins_data_and_transpose(self):
        cb = NVFP4HealingCallback(make_config(healing_recipe="delayed", store_quantized_params_on_gpu=False))
        q = SimpleNamespace(_data=FakeStorage(), _transpose=FakeStorage())
        q.clone = lambda: q
        cb._stash_fp8_copy(q)
        assert q._data.pinned and q._transpose.pinned


class TestMoveStashEntryToDevice:
    def test_mxfp8_moves_rowwise_and_columnwise(self):
        cb = NVFP4HealingCallback(make_config(healing_recipe="mxfp8"))
        w = SimpleNamespace(_rowwise_data=FakeStorage(), _columnwise_data=FakeStorage())
        cb._move_stash_entry_to_device(w, "cuda:0")
        assert w._rowwise_data.moved_to == ("cuda:0", True)
        assert w._columnwise_data.moved_to == ("cuda:0", True)

    def test_delayed_moves_data_and_transpose(self):
        cb = NVFP4HealingCallback(make_config(healing_recipe="delayed"))
        w = SimpleNamespace(_data=FakeStorage(), _transpose=FakeStorage())
        cb._move_stash_entry_to_device(w, "cuda:0")
        assert w._data.moved_to == ("cuda:0", True)
        assert w._transpose.moved_to == ("cuda:0", True)


class TestIsTargetModule:
    def test_true_for_te_linear_with_weight_false_otherwise(self, monkeypatch):
        import transformer_engine.pytorch as te

        class Linear(torch.nn.Module):
            pass

        class LayerNormLinear(torch.nn.Module):
            pass

        monkeypatch.setattr(te, "Linear", Linear, raising=False)
        monkeypatch.setattr(te, "LayerNormLinear", LayerNormLinear, raising=False)

        target = Linear()
        target.weight = torch.zeros(1)
        assert NVFP4HealingCallback._is_target_module(target) is True
        assert NVFP4HealingCallback._is_target_module(torch.nn.Linear(2, 2)) is False


class TestSwapInFP8WeightsMocked:
    def test_swaps_stashed_weights_and_clears_stash(self, monkeypatch):
        model = [make_fake_chunk([1, 2])]  # two Linear modules, op_fuser disabled
        cb = NVFP4HealingCallback(make_config(pre_quantize_base_weights=True, store_quantized_params_on_gpu=True))
        cb._fp8_stash = [torch.ones(2, 2), torch.ones(2, 2)]

        class FakeStream:
            def synchronize(self):
                self.synced = True

        monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
        monkeypatch.setattr(torch.cuda, "Stream", lambda: FakeStream())
        monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())

        with patch_target_module():
            cb._swap_in_fp8_weights(model)

        assert cb._fp8_stash == []
        for layer in model[0].decoder.layers:
            assert torch.all(layer.linear.weight == 1)
            assert layer.linear.weight.requires_grad is False
