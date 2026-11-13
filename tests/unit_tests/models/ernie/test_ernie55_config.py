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

"""CPU tests of the ERNIE 5.5 transformer config, the D-layer config rewrite and the D-layer guards."""

import copy
import dataclasses
import math

import pytest
import torch
from megatron.core.models.common.embeddings import _yarn_get_mscale
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig as MCoreMLATransformerConfig

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    context_parallel,
    ernie55_d_layer_config,
    validate_ernie55_config,
)
from megatron.bridge.models.transformer_config import MLATransformerConfig


pytestmark = pytest.mark.unit

# A small, valid D-only stack: the indexer loss is on, so the root variant must be "dsa".
BASE = dict(
    num_layers=2,
    hidden_size=64,
    num_attention_heads=4,
    q_lora_rank=32,
    kv_lora_rank=16,
    qk_head_dim=8,
    qk_pos_emb_head_dim=8,
    v_head_dim=8,
    # Bridge's MLATransformerConfig resolves normalization to "LayerNorm" (reverse-MRO field walk).
    normalization="RMSNorm",
    rope_type="rope",
    rotary_base=10000.0,
    layernorm_epsilon=1e-6,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
    # BF16 only; the parameters may stay fp32 (params_dtype), as in the CPU tests.
    bf16=True,
    experimental_attention_variant="dsa",
    dsa_kernel_backend="none",
    dsa_indexer_n_heads=4,
    dsa_indexer_head_dim=16,
    dsa_indexer_topk=6,
    dsa_indexer_loss_coeff=0.01,
    dsa_indexer_use_sparse_loss=True,
    dsa_indexer_k_norm_fp32=True,
    csa_window_size=3,
    ernie_vha_rank=2,
)

ERNIE_FIELDS = {
    "ernie_vha_rank": None,
    "ernie_vha_bf16_mix": True,
    "ernie_output_gate_fp32": False,
    "ernie_native_sink_grad": True,
    "ernie_d_num_attention_heads": None,
    "ernie_d_q_lora_rank": None,
    "ernie_d_kv_lora_rank": None,
    "ernie_d_qk_head_dim": None,
    "ernie_d_qk_pos_emb_head_dim": None,
    "ernie_d_v_head_dim": None,
    "ernie_d_rotary_base": None,
    "ernie_per_row_loss": False,
    "ernie_fused_ops": True,
}


def _config(**overrides) -> Ernie55TransformerConfig:
    config = Ernie55TransformerConfig(**{**BASE, **overrides})
    config.finalize()
    return config


def _with(config, **overrides):
    """Shallow copy of a finalized config with existing fields overridden."""
    changed = copy.copy(config)
    for name, value in overrides.items():
        if not hasattr(changed, name):
            raise ValueError(f"{type(changed).__name__} has no field {name!r}")
        setattr(changed, name, value)
    return changed


def _absorbed_mla_softmax_scale(config) -> float:
    """AbsorbedMLA's softmax scale, written out from its constructor."""
    mscale = _yarn_get_mscale(config.rotary_scaling_factor, config.mscale_all_dim)
    return mscale * mscale / math.sqrt(config.qk_head_dim + config.qk_pos_emb_head_dim)


class TestErnie55TransformerConfig:
    def test_fields_and_defaults(self):
        fields = {f.name: f.default for f in dataclasses.fields(Ernie55TransformerConfig)}
        for name, default in ERNIE_FIELDS.items():
            assert fields[name] == default, name
        assert issubclass(Ernie55TransformerConfig, MLATransformerConfig)
        assert issubclass(Ernie55TransformerConfig, MCoreMLATransformerConfig)
        assert Ernie55TransformerConfig(**BASE).multi_latent_attention is True

    def test_finalize_runs_post_init_then_validates(self):
        config = _config()
        assert config.kv_channels == BASE["hidden_size"] // BASE["num_attention_heads"]
        assert config.num_query_groups == BASE["num_attention_heads"]
        assert config.attention_latent_norm_epsilon == BASE["layernorm_epsilon"]
        assert config.dsa_kernel_backend == "none"

    def test_finalize_rejects_missing_vha_rank(self):
        with pytest.raises(ValueError, match="ernie_vha_rank"):
            _config(ernie_vha_rank=None)

    @pytest.mark.parametrize("value", ["false", 0, None])
    def test_finalize_rejects_a_fused_ops_flag_that_is_not_a_bool(self, value):
        with pytest.raises(ValueError, match="ernie_fused_ops must be a bool"):
            _config(ernie_fused_ops=value)

    def test_ernie_fields_survive_per_layer_copies(self):
        config = _config(
            ernie_vha_rank=16, ernie_d_rotary_base=8e6, ernie_output_gate_fp32=True, ernie_fused_ops=False
        )
        layer_config = DSALayerConfig.from_config(config)
        assert type(layer_config) is DSALayerConfig
        assert (layer_config.ernie_vha_rank, layer_config.ernie_d_rotary_base) == (16, 8e6)
        assert layer_config.ernie_output_gate_fp32 is True
        # The RoPE dispatcher reads the flag from the per-layer config it is handed.
        assert layer_config.ernie_fused_ops is False
        validate_ernie55_config(layer_config)


class TestDLayerConfig:
    def test_overrides_and_forced_settings(self):
        stack = _with(
            _config(
                ernie_d_num_attention_heads=8,
                ernie_d_q_lora_rank=24,
                ernie_d_kv_lora_rank=12,
                ernie_d_qk_head_dim=6,
                ernie_d_qk_pos_emb_head_dim=4,
                ernie_d_v_head_dim=10,
                ernie_d_rotary_base=8e6,
            ),
            # Settings the D copy must override, as a DSv4 or YaRN stack would carry them.
            rope_type="yarn",
            rotary_percent=0.5,
            rotary_interleaved=True,
            rotary_scaling_factor=40.0,
            mscale=0.707,
            mscale_all_dim=1.0,
            apply_rope_fusion=True,
            qk_layernorm=False,
            attention_output_gate=True,
        )
        d = ernie55_d_layer_config(stack)
        assert (d.num_attention_heads, d.q_lora_rank, d.kv_lora_rank) == (8, 24, 12)
        assert (d.qk_head_dim, d.qk_pos_emb_head_dim, d.v_head_dim, d.rotary_base) == (6, 4, 10, 8e6)
        assert d.num_query_groups == 8
        assert d.kv_channels == stack.kv_channels
        assert (d.rope_type, d.rotary_percent, d.rotary_interleaved) == ("rope", 1.0, False)
        assert (d.rotary_scaling_factor, d.mscale, d.mscale_all_dim) == (1.0, 1.0, 0.0)
        assert d.apply_rope_fusion is False
        assert d.qk_layernorm is True
        assert d.attention_output_gate is False
        assert d.experimental_attention_variant == stack.experimental_attention_variant
        assert _absorbed_mla_softmax_scale(d) == pytest.approx(1.0 / math.sqrt(6 + 4), rel=1e-12)
        # The stack-level YaRN settings would have produced a different scale.
        assert _absorbed_mla_softmax_scale(stack) != pytest.approx(_absorbed_mla_softmax_scale(d))

    def test_none_overrides_inherit_the_stack_geometry(self):
        stack = _config()
        d = ernie55_d_layer_config(stack)
        for name in ("num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_head_dim", "v_head_dim"):
            assert getattr(d, name) == getattr(stack, name), name
        assert (d.qk_pos_emb_head_dim, d.rotary_base) == (stack.qk_pos_emb_head_dim, stack.rotary_base)
        assert d.num_query_groups == stack.num_attention_heads
        assert _absorbed_mla_softmax_scale(d) == pytest.approx(1.0 / math.sqrt(8 + 8), rel=1e-12)

    def test_missing_derived_fields_are_filled(self):
        stack = _with(_config(), kv_channels=None, attention_latent_norm_epsilon=None)
        d = ernie55_d_layer_config(stack)
        assert d.kv_channels == stack.v_head_dim
        assert d.attention_latent_norm_epsilon == stack.layernorm_epsilon

    def test_does_not_mutate_the_input(self):
        stack = _with(_config(ernie_d_num_attention_heads=8), rope_type="yarn", qk_layernorm=False)
        before = {name: copy.copy(value) for name, value in vars(stack).items() if not callable(value)}
        d = ernie55_d_layer_config(stack)
        assert d is not stack
        after = {name: value for name, value in vars(stack).items() if not callable(value)}
        assert after == before

    @pytest.mark.parametrize("per_layer", [False, True])
    def test_keeps_the_config_type_and_is_idempotent(self, per_layer):
        config = _config(ernie_d_q_lora_rank=24)
        if per_layer:
            config = DSALayerConfig.from_config(config)
        d = ernie55_d_layer_config(config)
        assert type(d) is type(config)
        assert vars(ernie55_d_layer_config(d)) == vars(d)


GUARD_CASES = [
    ({"multi_latent_attention": False}, "multi_latent_attention"),
    ({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size"),
    ({"context_parallel_size": 2, "dynamic_context_parallel": True}, "dynamic"),
    ({"bf16": False}, "BF16"),
    ({"fp16": True}, "BF16"),
    ({"bf16": False, "fp16": True}, "BF16"),
    ({"fp8": "e4m3"}, "FP8"),
    ({"fp4": "e2m1"}, "FP4"),
    ({"quant_recipe": "any-recipe"}, "quant_recipe"),
    ({"use_kitchen": True}, "Kitchen"),
    ({"use_kitchen_attention": True}, "Kitchen"),
    ({"delay_wgrad_compute": True}, "delay_wgrad_compute"),
    ({"cuda_graph_impl": "local"}, "CUDA graphs"),
    ({"cuda_graph_impl": "transformer_engine"}, "CUDA graphs"),
    ({"enable_cuda_graph": True}, "CUDA graphs"),
    ({"recompute_granularity": "selective", "recompute_modules": ["core_attn"]}, "core_attn"),
    ({"recompute_granularity": "selective", "recompute_modules": ["mla_up_proj"]}, "mla_up_proj"),
    ({"recompute_granularity": "selective", "recompute_modules": None}, "core_attn"),
    ({"add_bias_linear": True}, "add_bias_linear"),
    ({"attention_dropout": 0.1}, "attention_dropout"),
    ({"normalization": "LayerNorm"}, "RMSNorm"),
    ({"attention_latent_norm_epsilon": 1e-5}, "attention_latent_norm_epsilon"),
    ({"cache_mla_latents": True}, "cache_mla_latents"),
    ({"q_lora_rank": None}, "q_lora_rank"),
    ({"ernie_vha_rank": None}, "ernie_vha_rank"),
    ({"ernie_vha_rank": 0}, "ernie_vha_rank"),
    ({"ernie_vha_rank": True}, "ernie_vha_rank"),
    ({"dsa_kernel_backend": "tilelang"}, "dsa_kernel_backend"),
    ({"dsa_kernel_backend": "min-memory-torch"}, "dsa_kernel_backend"),
    ({"dsa_kernel_backend": None}, "dsa_kernel_backend"),
    ({"dsa_kernel_backend": "cudnn", "attention_backend": AttnBackend.unfused}, "attention_backend"),
    ({"dsa_indexer_n_heads": None}, "dsa_indexer_n_heads"),
    ({"dsa_indexer_head_dim": 0}, "dsa_indexer_head_dim"),
    ({"dsa_indexer_topk": None}, "dsa_indexer_topk"),
    ({"csa_window_size": 0}, "csa_window_size"),
    ({"dsa_indexer_mode": "simplified"}, "dsa_indexer_mode"),
    ({"dsa_indexer_topk_freq": 2}, "dsa_indexer_topk_freq"),
    ({"dsa_indexer_kpool": 2}, "dsa_indexer_kpool"),
    ({"dsa_indexer_rope_interleaved": True}, "dsa_indexer_rope_interleaved"),
    ({"dsa_indexer_rotate_activation": False}, "dsa_indexer_rotate_activation"),
    ({"dsa_indexer_scoring_relu": False}, "dsa_indexer_scoring_relu"),
    ({"dsa_indexer_k_norm_fp32": False}, "dsa_indexer_k_norm_fp32"),
    ({"dsa_indexer_weights_proj_output_dtype": "fp32"}, "dsa_indexer_weights_proj_output_dtype"),
    ({"dsa_indexer_k_norm_epsilon": 1e-3}, "dsa_indexer_k_norm_epsilon"),
    ({"dsa_indexer_use_sparse_loss": False}, "dsa_indexer_use_sparse_loss"),
    ({"experimental_attention_variant": None}, "loss_scale_func"),
]


class TestValidation:
    def test_valid_config_passes(self):
        validate_ernie55_config(_config())

    @pytest.mark.parametrize("overrides,match", GUARD_CASES, ids=[str(case[0]) for case in GUARD_CASES])
    def test_guards_raise(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            validate_ernie55_config(_with(_config(), **overrides))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"recompute_granularity": "full", "recompute_method": "uniform", "recompute_num_layers": 1},
            {"recompute_granularity": "selective", "recompute_modules": ["mlp", "moe_act"]},
            {"dsa_kernel_backend": "cudnn"},
            {"dsa_kernel_backend": "cudnn", "attention_backend": AttnBackend.flash},
            {"attention_backend": AttnBackend.unfused},
            {"attention_latent_norm_epsilon": None},
            {"attention_latent_norm_epsilon": BASE["layernorm_epsilon"]},
            {"dsa_indexer_k_norm_epsilon": BASE["layernorm_epsilon"]},
            {"cuda_graph_impl": "none"},
            {"ernie_d_q_lora_rank": 24},
            {"params_dtype": torch.bfloat16, "bf16": True},
        ],
        ids=lambda overrides: str(overrides),
    )
    def test_supported_settings_pass(self, overrides):
        validate_ernie55_config(_with(_config(), **overrides))

    def test_d_layer_q_lora_rank_can_come_from_the_override(self):
        validate_ernie55_config(_with(_config(), q_lora_rank=None, ernie_d_q_lora_rank=24))


class TestContextParallelEnvelope:
    """CP > 1 needs Megatron-Core's CP partition converter (dev); without it (main) the D layer raises."""

    def test_cp_follows_the_converter(self):
        config = _with(_config(), context_parallel_size=2)
        if context_parallel.load_cp_layout_api() is None:
            with pytest.raises(RuntimeError, match="CP partition converter"):
                validate_ernie55_config(config)
        else:
            validate_ernie55_config(config)

    def test_cp_without_the_converter_raises(self, monkeypatch):
        monkeypatch.setattr(context_parallel, "load_cp_layout_api", lambda: None)
        with pytest.raises(RuntimeError, match="CP partition converter"):
            validate_ernie55_config(_with(_config(), context_parallel_size=2))
        validate_ernie55_config(_config())

    def test_cp_partition_mode_must_be_known(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        with pytest.raises(ValueError, match="cp_partition_mode"):
            validate_ernie55_config(_with(_config(), context_parallel_size=2, cp_partition_mode="striped"))


class TestPerRowLossMode:
    def test_needs_per_row_reduction(self):
        assert _config(ernie_per_row_loss=True).ernie_per_row_loss
        with pytest.raises(ValueError, match="calculate_per_token_loss=False"):
            _config(ernie_per_row_loss=True, calculate_per_token_loss=True)


class TestLossScaleHookRequirement:
    """The D layer scales its KL only through DSAIndexerLossAutoScaler, which the schedule drives."""

    @pytest.mark.parametrize("variant", ["dsa", "dsv4_hybrid"])
    def test_scaled_root_variants_pass(self, variant):
        validate_ernie55_config(_with(_config(), experimental_attention_variant=variant))

    def test_other_root_variant_without_hook_raises(self):
        with pytest.raises(ValueError, match="never scales the indexer gradients"):
            validate_ernie55_config(_with(_config(), experimental_attention_variant=None))

    @pytest.mark.parametrize("variant", [None, "dsa", "dsv4_hybrid"])
    def test_explicit_dsa_hook_passes(self, variant):
        validate_ernie55_config(
            _with(
                _config(),
                experimental_attention_variant=variant,
                experimental_attention_variant_loss_scale_func=dsa.DSAIndexerLossAutoScaler.set_loss_scale,
            )
        )

    @pytest.mark.parametrize("variant", [None, "dsa", "dsv4_hybrid"])
    def test_foreign_hook_raises(self, variant):
        """A hook that does not drive DSAIndexerLossAutoScaler (e.g. a model-specific scaler) mis-scales the KL."""
        scales = []
        with pytest.raises(ValueError, match="replaces the schedule's"):
            validate_ernie55_config(
                _with(
                    _config(),
                    experimental_attention_variant=variant,
                    experimental_attention_variant_loss_scale_func=scales.append,
                )
            )

    @pytest.mark.parametrize("coeff", [0.0, None])
    def test_no_indexer_loss_needs_no_hook(self, coeff):
        validate_ernie55_config(_with(_config(), dsa_indexer_loss_coeff=coeff, experimental_attention_variant=None))
        validate_ernie55_config(
            _with(
                _config(),
                dsa_indexer_loss_coeff=coeff,
                experimental_attention_variant=None,
                experimental_attention_variant_loss_scale_func=lambda scale: None,
            )
        )
