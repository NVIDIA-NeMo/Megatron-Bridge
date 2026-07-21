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


class TestHookRegistration:
    def test_registers_expected_hooks(self):
        manager = CallbackManager()
        manager.add(NVFP4HealingCallback(make_config()))
        assert manager.has_callbacks("on_data_init_start")
        assert manager.has_callbacks("on_train_step_end")
        assert manager.has_callbacks("on_train_end")
        assert not manager.has_callbacks("on_train_start")
        assert not manager.has_callbacks("on_train_step_start")
