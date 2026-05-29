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
Unit tests for DeepSeek recipe configuration builders.

Patterned after Qwen recipe tests: import all exported helpers from
`megatron.bridge.recipes.deepseek`, monkeypatch `AutoBridge` to a lightweight
fake that returns a minimal provider, and assert a valid ConfigContainer is
built with small overrides.
"""

import importlib
from typing import Callable

import pytest
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.pipeline_parallel_layer_layout import PipelineParallelLayerLayout

from megatron.bridge.recipes.deepseek import (
    set_deepseek_v3_pipeline_model_parallel_layout,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.deepseek_v3 import _build_standalone_mtp_layout


_deepseek_module = importlib.import_module("megatron.bridge.recipes.deepseek")
_DEEPSEEK_RECIPE_NAMES = frozenset(
    {
        "deepseek_v2_pretrain_config",
        "deepseek_v2_lite_pretrain_config",
        "deepseek_v3_pretrain_config",
        "deepseek_v3_pretrain_config_32nodes",
        "deepseek_v4_flash_pretrain_mxfp8_config",
        "deepseek_v4_flash_pretrain_muon_config",
    }
)
_DEEPSEEK_EXPORTED_NAMES = set(getattr(_deepseek_module, "__all__", ()))
assert _DEEPSEEK_RECIPE_NAMES <= _DEEPSEEK_EXPORTED_NAMES
assert {"set_deepseek_v4_pipeline_model_parallel_layout"} <= _DEEPSEEK_EXPORTED_NAMES
_DEEPSEEK_RECIPE_FUNCS = [getattr(_deepseek_module, name) for name in sorted(_DEEPSEEK_RECIPE_NAMES)]
assert all(callable(recipe_func) for recipe_func in _DEEPSEEK_RECIPE_FUNCS)


class _FakeModelCfg:
    # Minimal provider to accept attribute assignments used in recipes
    def __init__(self):
        # Provide defaults for attributes that recipes might read
        self.rotary_base = 10000.0
        self.num_moe_experts = 0
        self.apply_rope_fusion = False
        self.vocab_size = 1024

    def finalize(self):
        return None


class _FakeBridge:
    def __init__(self):
        pass

    def to_megatron_provider(self, load_weights: bool = False):
        return _FakeModelCfg()

    @staticmethod
    def from_hf_pretrained(hf_path: str, **kwargs):
        return _FakeBridge()


def _assert_basic_config(cfg):
    from megatron.bridge.training.config import ConfigContainer

    assert isinstance(cfg, ConfigContainer)
    assert cfg.model is not None
    assert cfg.train is not None
    assert cfg.optimizer is not None
    assert cfg.scheduler is not None
    assert cfg.dataset is not None
    assert cfg.logger is not None
    assert cfg.tokenizer is not None
    assert cfg.checkpoint is not None
    assert cfg.rng is not None
    assert cfg.train.global_batch_size >= 1
    assert cfg.train.micro_batch_size >= 1
    assert cfg.dataset.seq_length >= 1


@pytest.mark.parametrize("recipe_func", _DEEPSEEK_RECIPE_FUNCS)
def test_each_deepseek_recipe_builds_config(recipe_func: Callable, monkeypatch: pytest.MonkeyPatch):
    # Monkeypatch AutoBridge in the specific module where the recipe function is defined
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    # DeepSeek recipes are all pretrain configs - call without parameters
    cfg = recipe_func()

    _assert_basic_config(cfg)

    # Ensure tokenizer is properly configured
    # DeepSeek pretrain recipes use either NullTokenizer or HuggingFaceTokenizer
    if cfg.tokenizer.tokenizer_type == "NullTokenizer":
        assert cfg.tokenizer.vocab_size is not None
    else:
        assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"
        assert cfg.tokenizer.tokenizer_model is not None

    # Parallelism and shaping
    assert getattr(cfg.model, "tensor_model_parallel_size", 1) >= 1
    assert getattr(cfg.model, "pipeline_model_parallel_size", 1) >= 1


def test_deepseek_v3_pipeline_layout_can_place_mtp_in_standalone_stage():
    model_cfg = _FakeModelCfg()
    model_cfg.num_layers = 61
    model_cfg.mtp_num_layers = 1
    model_cfg.pipeline_model_parallel_size = 8
    model_cfg.virtual_pipeline_model_parallel_size = 2

    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, mtp_standalone=True)

    layout = model_cfg.pipeline_model_parallel_layout
    assert layout[-2] == ["mtp"]
    assert layout[-1] == ["loss"]
    assert sum(stage.count("decoder") for stage in layout) == model_cfg.num_layers

    parsed_layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size=8)
    assert parsed_layout.validate_layer_layout(model_cfg.num_layers, model_cfg.mtp_num_layers)
    assert parsed_layout.layout[6][1] == [LayerType.mtp]
    assert parsed_layout.layout[7][1] == [LayerType.loss]


def test_deepseek_v4_pipeline_layout_distributes_decoder_layers_and_places_mtp_loss():
    model_cfg = _FakeModelCfg()
    model_cfg.num_layers = 7
    model_cfg.mtp_num_layers = 2
    model_cfg.pipeline_model_parallel_size = 3

    set_deepseek_v4_pipeline_model_parallel_layout(model_cfg)

    assert model_cfg.pipeline_model_parallel_layout == [
        ["embedding", "decoder", "decoder", "decoder"],
        ["decoder", "decoder"],
        ["decoder", "decoder", "mtp", "mtp", "loss"],
    ]


def test_deepseek_v4_pipeline_layout_disables_layout_for_single_stage():
    model_cfg = _FakeModelCfg()
    model_cfg.num_layers = 7
    model_cfg.mtp_num_layers = 2
    model_cfg.pipeline_model_parallel_size = 1

    set_deepseek_v4_pipeline_model_parallel_layout(model_cfg)

    assert model_cfg.pipeline_model_parallel_layout is None


def test_build_standalone_mtp_layout_rejects_too_few_total_stages():
    with pytest.raises(ValueError, match="at least three"):
        _build_standalone_mtp_layout(num_decoder_layers=61, total_stages=2, mtp_layers=1)


def test_build_standalone_mtp_layout_rejects_zero_mtp_layers():
    with pytest.raises(ValueError, match="mtp_num_layers > 0"):
        _build_standalone_mtp_layout(num_decoder_layers=61, total_stages=4, mtp_layers=0)


def test_deepseek_v3_pipeline_layout_can_place_multiple_mtp_layers_in_standalone_stage():
    model_cfg = _FakeModelCfg()
    model_cfg.num_layers = 61
    model_cfg.mtp_num_layers = 2
    model_cfg.pipeline_model_parallel_size = 4
    model_cfg.virtual_pipeline_model_parallel_size = None

    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, mtp_standalone=True)

    layout = model_cfg.pipeline_model_parallel_layout
    assert layout[-2] == ["mtp", "mtp"]
    assert layout[-1] == ["loss"]
    assert sum(stage.count("decoder") for stage in layout) == model_cfg.num_layers

    parsed_layout = PipelineParallelLayerLayout(layout, pipeline_model_parallel_size=4)
    assert parsed_layout.validate_layer_layout(model_cfg.num_layers, model_cfg.mtp_num_layers)


def test_deepseek_v3_pipeline_layout_prefers_explicit_layout_over_standalone_mtp():
    model_cfg = _FakeModelCfg()
    model_cfg.mtp_num_layers = 0
    explicit_layout = [["embedding", "decoder", "loss"]]

    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, explicit_layout, mtp_standalone=True)

    assert model_cfg.pipeline_model_parallel_layout is explicit_layout


def test_deepseek_v3_pipeline_layout_requires_num_layers_for_standalone_mtp():
    model_cfg = _FakeModelCfg()
    model_cfg.mtp_num_layers = 1
    model_cfg.pipeline_model_parallel_size = 4
    model_cfg.virtual_pipeline_model_parallel_size = None

    with pytest.raises(ValueError, match="num_layers"):
        set_deepseek_v3_pipeline_model_parallel_layout(model_cfg, mtp_standalone=True)


def test_deepseek_v3_pipeline_layout_keeps_default_mtp_with_loss():
    model_cfg = _FakeModelCfg()
    model_cfg.mtp_num_layers = 1
    model_cfg.pipeline_model_parallel_size = 8
    model_cfg.virtual_pipeline_model_parallel_size = 2

    set_deepseek_v3_pipeline_model_parallel_layout(model_cfg)

    assert model_cfg.pipeline_model_parallel_layout[-1][-2:] == ["mtp", "loss"]


def _build_deepseek_v4_recipe(name: str, monkeypatch: pytest.MonkeyPatch):
    mod = importlib.import_module("megatron.bridge.recipes.deepseek.deepseek_v4")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)
    return getattr(mod, name)()


def test_deepseek_v4_adam_mxfp8_recipe_uses_validated_optimizer_defaults(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_v4_recipe("deepseek_v4_flash_pretrain_mxfp8_config", monkeypatch)

    assert cfg.optimizer.optimizer == "adam"
    assert cfg.optimizer.lr == 2.7e-4
    assert cfg.optimizer.min_lr == 2.7e-5
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1e-20
    assert cfg.scheduler.start_weight_decay == 0.1
    assert cfg.scheduler.end_weight_decay == 0.1
    assert cfg.scheduler.weight_decay_incr_style == "constant"
    assert cfg.ddp.use_distributed_optimizer is True
    assert cfg.ddp.overlap_param_gather is True
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.model.csa_backend == "cudnn_dsa"
    assert cfg.model.use_fused_mhc is True
    assert cfg.mixed_precision.fp8_recipe == "mxfp8"
    assert cfg.mixed_precision.fp8_param_gather is False
    assert cfg.model.mtp_eval_in_bf16 is True


def test_deepseek_v4_muon_bf16_recipe_uses_validated_optimizer_defaults(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_v4_recipe("deepseek_v4_flash_pretrain_muon_config", monkeypatch)

    assert cfg.optimizer.optimizer == "muon"
    assert cfg.optimizer.lr == 2.7e-4
    assert cfg.optimizer.min_lr == 2.7e-5
    assert cfg.optimizer.weight_decay == 0.1
    assert cfg.optimizer.adam_beta1 == 0.9
    assert cfg.optimizer.adam_beta2 == 0.95
    assert cfg.optimizer.adam_eps == 1e-20
    assert cfg.optimizer.muon_momentum == 0.95
    assert cfg.optimizer.muon_nesterov is True
    assert cfg.optimizer.muon_scale_mode == "unit_rms_norm"
    assert cfg.optimizer.muon_num_ns_steps == 5
    assert cfg.optimizer.muon_extra_scale_factor == 0.2
    assert cfg.ddp.use_distributed_optimizer is False
    assert cfg.ddp.overlap_grad_reduce is True
    assert cfg.ddp.grad_reduce_in_fp32 is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.model.csa_backend == "cudnn_dsa"
    assert cfg.model.use_fused_mhc is True
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp8 is None
    assert cfg.mixed_precision.fp8_param_gather is False


def test_deepseek_v4_rejects_muon_mxfp8(monkeypatch: pytest.MonkeyPatch):
    mod = importlib.import_module("megatron.bridge.recipes.deepseek.deepseek_v4")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    with pytest.raises(ValueError, match="Muon \\+ MXFP8"):
        mod._deepseek_v4_flash_pretrain_config(optimizer_type="muon", mxfp8=True)


def test_deepseek_v4_rejects_invalid_optimizer(monkeypatch: pytest.MonkeyPatch):
    mod = importlib.import_module("megatron.bridge.recipes.deepseek.deepseek_v4")
    monkeypatch.setattr(mod, "AutoBridge", _FakeBridge)

    with pytest.raises(ValueError, match="Invalid DeepSeek-V4 optimizer type"):
        mod._deepseek_v4_flash_pretrain_config(optimizer_type="sgd")
