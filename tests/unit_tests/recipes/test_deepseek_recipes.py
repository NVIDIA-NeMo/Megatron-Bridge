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

from megatron.bridge.models.model_provider import ModelProviderMixin
from megatron.bridge.recipes.deepseek import (
    set_deepseek_v3_pipeline_model_parallel_layout,
    set_deepseek_v4_pipeline_model_parallel_layout,
)
from megatron.bridge.recipes.deepseek.deepseek_v3 import _build_standalone_mtp_layout
from tests.unit_tests.recipes.recipe_test_utils import patch_recipe_module_global


_deepseek_module = importlib.import_module("megatron.bridge.recipes.deepseek")
_DEEPSEEK_RECIPE_NAMES = frozenset(
    {
        "deepseek_v2_pretrain_config",
        "deepseek_v2_lite_pretrain_config",
        "deepseek_v3_pretrain_config",
        "deepseek_v3_pretrain_config_32nodes",
        "deepseek_v4_flash_pretrain_config",
        "deepseek_v4_flash_pretrain_mxfp8_config",
        "deepseek_v4_flash_pretrain_muon_config",
        "deepseek_v4_flash_sft_config",
        "deepseek_v4_flash_no_mtp_sft_config",
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
        self.make_vocab_size_divisible_by = 128

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


class _DSv4ProviderStub(ModelProviderMixin):
    # Minimal provider to exercise the DSv4 auto-layout in apply_overrides_and_finalize
    def __init__(
        self,
        *,
        experimental_attention_variant,
        num_layers=43,
        mtp_num_layers=1,
        pipeline_model_parallel_size=1,
        pipeline_model_parallel_layout=None,
    ):
        self.experimental_attention_variant = experimental_attention_variant
        self.num_layers = num_layers
        self.mtp_num_layers = mtp_num_layers
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.pipeline_model_parallel_layout = pipeline_model_parallel_layout

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        return None

    def finalize(self):
        return None


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
    # Always patch AutoBridge in the base deepseek_v3 module (where base configs call it)
    deepseek_v3_mod = importlib.import_module("megatron.bridge.recipes.deepseek.deepseek_v3")
    patch_recipe_module_global(monkeypatch, deepseek_v3_mod, "AutoBridge", _FakeBridge)
    # Also patch in the recipe's own module if it directly imports AutoBridge
    module_name = recipe_func.__module__
    mod = importlib.import_module(module_name)
    if hasattr(mod, "AutoBridge"):
        patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)

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


def test_model_and_perf_deepseek_recipes_bake_environment_defaults(monkeypatch: pytest.MonkeyPatch):
    """Model H100 and flat GB200 recipes should own topology-correct env defaults."""
    deepseek_v3_mod = importlib.import_module("megatron.bridge.recipes.deepseek.deepseek_v3")
    patch_recipe_module_global(monkeypatch, deepseek_v3_mod, "AutoBridge", _FakeBridge)

    recipe_module = importlib.import_module("megatron.bridge.recipes.deepseek.h100.deepseek_v3")
    patch_recipe_module_global(monkeypatch, recipe_module, "AutoBridge", _FakeBridge)
    recipe_config = recipe_module.deepseek_v3_pretrain_1024gpu_h100_bf16_config()

    perf_module = importlib.import_module("megatron.bridge.perf_recipes.deepseek.gb200.deepseek_v3")
    perf_config = perf_module.deepseek_v3_pretrain_256gpu_gb200_bf16_config()

    expected_recipe = {
        "NVTE_FWD_LAYERNORM_SM_MARGIN": 20,
        "NVTE_BWD_LAYERNORM_SM_MARGIN": 20,
    }
    assert recipe_config.env_vars.items() >= expected_recipe.items()
    assert recipe_config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 8
    assert recipe_config.env_vars["NVLINK_DOMAIN_SIZE"] == 8
    assert recipe_config.env_vars["USE_MNNVL"] == 0
    assert perf_config.env_vars["NVTE_FWD_LAYERNORM_SM_MARGIN"] == 20
    assert perf_config.env_vars["NVTE_BWD_LAYERNORM_SM_MARGIN"] == 20
    assert "TORCHINDUCTOR_WORKER_START" not in recipe_config.env_vars
    assert "QUANTIZATION_TYPE_DEBUG" not in recipe_config.env_vars
    assert "TORCHINDUCTOR_WORKER_START" not in perf_config.env_vars
    assert "QUANTIZATION_TYPE_DEBUG" not in perf_config.env_vars
    assert perf_config.env_vars["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] == 64
    assert perf_config.env_vars["NVLINK_DOMAIN_SIZE"] == 72
    assert perf_config.env_vars["USE_MNNVL"] == 1


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


def test_deepseek_v4_flash_full_model_layout_fits_hash_layers_on_first_stage():
    # DSv4-Flash full model: 43 decoder layers, 1 MTP layer, PP=4. The first stage
    # must hold at least the 3 hash-routed MoE layers alongside the embedding.
    model_cfg = _FakeModelCfg()
    model_cfg.num_layers = 43
    model_cfg.mtp_num_layers = 1
    model_cfg.pipeline_model_parallel_size = 4

    set_deepseek_v4_pipeline_model_parallel_layout(model_cfg)

    layout = model_cfg.pipeline_model_parallel_layout
    assert layout[0][0] == "embedding"
    assert layout[0].count("decoder") >= 3
    assert layout[-1][-2:] == ["mtp", "loss"]
    assert sum(stage.count("decoder") for stage in layout) == 43


def test_dsv4_provider_auto_sets_pipeline_layout_for_pp_gt_1():
    # The mbridge/verl path builds the provider and calls apply_overrides_and_finalize
    # directly (no recipe), so the DSv4 layout must be auto-set there for PP > 1.
    provider = _DSv4ProviderStub(experimental_attention_variant="dsv4_hybrid")
    provider.apply_overrides_and_finalize(overrides={"pipeline_model_parallel_size": 4})

    layout = provider.pipeline_model_parallel_layout
    assert layout is not None
    assert layout[0][0] == "embedding"
    assert layout[0].count("decoder") >= 3
    assert layout[-1][-2:] == ["mtp", "loss"]
    assert sum(stage.count("decoder") for stage in layout) == 43


def test_dsv4_provider_keeps_single_stage_layout_unset():
    provider = _DSv4ProviderStub(experimental_attention_variant="dsv4_hybrid", pipeline_model_parallel_size=1)
    provider.apply_overrides_and_finalize()

    assert provider.pipeline_model_parallel_layout is None


def test_non_dsv4_provider_is_not_auto_laid_out():
    provider = _DSv4ProviderStub(experimental_attention_variant=None)
    provider.apply_overrides_and_finalize(overrides={"pipeline_model_parallel_size": 4})

    assert provider.pipeline_model_parallel_layout is None


def test_dsv4_provider_preserves_user_supplied_layout():
    user_layout = [["embedding", "decoder", "loss"]]
    provider = _DSv4ProviderStub(
        experimental_attention_variant="dsv4_hybrid",
        pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=user_layout,
    )
    provider.apply_overrides_and_finalize()

    assert provider.pipeline_model_parallel_layout is user_layout


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
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)
    patch_recipe_module_global(monkeypatch, mod, "deepseek_v4_supports_blackwell_fused_kernels", lambda: True)
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
    assert cfg.model.apply_dsa_kernel_fusion is False
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.use_fused_mhc is True
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
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
    assert cfg.model.apply_dsa_kernel_fusion is False
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.use_fused_mhc is True
    assert cfg.mixed_precision.bf16 is True
    assert cfg.mixed_precision.fp8 is None
    assert cfg.mixed_precision.fp8_param_gather is False


def test_deepseek_v4_base_recipe_uses_blackwell_defaults(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_v4_recipe("deepseek_v4_flash_pretrain_config", monkeypatch)

    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.model.context_parallel_size == 1
    assert cfg.model.apply_dsa_kernel_fusion is False
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.use_fused_mhc is True
    assert cfg.model.dsa_indexer_loss_coeff == 0.0
    assert cfg.model.dsa_indexer_use_sparse_loss is False
    assert cfg.train.global_batch_size == 128
    assert cfg.train.micro_batch_size == 1


@pytest.mark.parametrize(
    "recipe_name",
    [
        "deepseek_v4_flash_pretrain_config",
        "deepseek_v4_flash_pretrain_mxfp8_config",
        "deepseek_v4_flash_pretrain_muon_config",
        "deepseek_v4_flash_sft_config",
        "deepseek_v4_flash_no_mtp_sft_config",
    ],
)
def test_deepseek_v4_recipes_disable_blackwell_only_fusions_when_unavailable(
    recipe_name: str, monkeypatch: pytest.MonkeyPatch
):
    mod = importlib.import_module("megatron.bridge.recipes.deepseek.deepseek_v4")
    patch_recipe_module_global(monkeypatch, mod, "AutoBridge", _FakeBridge)
    patch_recipe_module_global(monkeypatch, mod, "deepseek_v4_supports_blackwell_fused_kernels", lambda: False)

    cfg = getattr(mod, recipe_name)()

    assert cfg.model.apply_dsa_kernel_fusion is False
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.use_fused_mhc is False


def test_deepseek_v4_flash_sft_recipe_uses_fused_mhc(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_v4_recipe("deepseek_v4_flash_sft_config", monkeypatch)

    # Fused mHC and fused rope are both enabled (full-model validated on GB300; the
    # historical rope NaN was the rotary_percent mapping bug fixed in #4271).
    assert cfg.model.use_fused_mhc is True
    assert cfg.model.apply_rope_fusion is True
    assert cfg.model.tensor_model_parallel_size == 1
    assert cfg.model.pipeline_model_parallel_size == 4
    assert cfg.model.expert_model_parallel_size == 8
    assert cfg.optimizer.optimizer == "adam"
    assert cfg.tokenizer.tokenizer_type == "HuggingFaceTokenizer"


def test_deepseek_v4_flash_no_mtp_sft_recipe_disables_mtp(monkeypatch: pytest.MonkeyPatch):
    cfg = _build_deepseek_v4_recipe("deepseek_v4_flash_no_mtp_sft_config", monkeypatch)

    assert cfg.model.use_fused_mhc is True
    assert cfg.model.mtp_num_layers is None
    assert cfg.model.mtp_loss_scaling_factor == 0.0
