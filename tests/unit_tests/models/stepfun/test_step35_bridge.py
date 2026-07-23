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

"""Unit tests for Step35Bridge (Step-3.5-Flash)."""

from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVGMapping,
    merge_qkvg_weights,
    split_qkvg_weights,
)
from megatron.bridge.models.stepfun import step35_bridge as _step35_bridge_mod
from megatron.bridge.models.stepfun.step35_bridge import (
    StackedExpertAutoMapping,
    StackedExpertGatedMLPMapping,
    Step35Bridge,
    Step35DecoderLayer,
    Step35SharedExpertMLP,
    _MTPDenseLayerSpecsList,
    build_step35_layer_spec,
)


def _make_hf_config(**overrides) -> SimpleNamespace:
    """Build a minimal HF-like config namespace that satisfies Step35Bridge.provider_bridge.

    Only the fields read by the Step35-specific portion of provider_bridge / mapping_registry
    are populated. Defaults mirror a small Step-3.5-Flash variant: 4 main decoder layers
    + 2 MTP layers, half full / half sliding attention, 4 experts with `moe_layers_enum`
    enumerating only the MoE-bearing main decoder layers.
    """
    base = dict(
        num_hidden_layers=4,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_attention_groups=8,
        head_dim=32,
        vocab_size=512,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        torch_dtype="bfloat16",
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=64,
        share_expert_dim=64,
        use_head_wise_attn_gate=True,
        attention_output_gate=True,
        attention_other_setting={
            "attention_type": "sliding_attention",
            "num_attention_heads": 96,
            "num_attention_groups": 8,
            "head_dim": 128,
        },
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        rope_theta=10000.0,
        moe_layers_enum="2,3",
        num_nextn_predict_layers=2,
        # HF carries ``sliding_window`` as a scalar token count; the bridge
        # pads it into the ``[left, right]`` window form expected by
        # ``Step35DecoderLayer``.
        sliding_window=512,
        swiglu_limits=[None, None, None, None],
        swiglu_limits_shared=[None, None, None, None],
        partial_rotary_factors=[0.5, 0.5, 0.5, 0.5],
        use_qk_norm=True,
        use_moe_router_bias=True,
        moe_router_activation="softmax",
        moe_router_scaling_factor=1.0,
        need_fp32_gate=False,
        zero_centered=True,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeProvider:
    """Stand-in for a `Step35ModelProvider` returned by the parent `provider_bridge`.

    Mirrors only the fields that Step35Bridge.provider_bridge reads or writes.
    """

    def __init__(self):
        self.num_layers = 4
        self.mtp_num_layers = 2
        self.num_query_groups = None
        self.num_moe_experts = None
        self.moe_router_topk = None
        self.moe_shared_expert_intermediate_size = None
        self.head_wise_attn_gate = None
        self.attention_output_gate = None
        self.layer_types = None
        self.attention_other_setting = None
        self.sliding_attention_setting = None
        self.kv_channels = None
        self.rotary_base = None
        self.rotary_base_per_layer = None
        # Step35Bridge.provider_bridge unconditionally writes the following:
        self.normalization = None
        self.layernorm_zero_centered_gamma = None
        self.gated_linear_unit = None
        self.add_bias_linear = None
        self.add_qkv_bias = None
        self.hidden_dropout = None
        self.attention_dropout = None
        self.qk_layernorm = None
        self.autocast_dtype = None
        self.moe_grouped_gemm = None
        self.moe_router_load_balancing_type = None
        self.moe_aux_loss_coeff = None
        self.moe_router_pre_softmax = None
        self.moe_token_dispatcher_type = None
        self.moe_permute_fusion = None
        self.moe_layer_freq = None
        self.transformer_layer_spec = None
        self.rotary_percents = None
        self.swiglu_limits = None
        self.swiglu_limits_shared = None
        self.moe_router_enable_expert_bias = None
        self.moe_router_score_function = None
        self.moe_router_topk_scaling_factor = None
        # Only assigned when hf_config.need_fp32_gate is truthy.
        self.moe_router_dtype = None


class _FakeHFPretrained:
    """Stand-in for `PreTrainedCausalLM`; only `.config` is read."""

    def __init__(self, config):
        self.config = config


# ---------------------------------------------------------------------------
# Registration / class hierarchy
# ---------------------------------------------------------------------------


class TestStep35BridgeRegistration:
    def test_register_bridge_attributes_keep_hf_identifiers(self):
        """The decorator must register the bridge under the upstream HF strings.

        These cannot be renamed to ``step35`` / ``Step35ForCausalLM`` without
        breaking ``AutoConfig.from_pretrained("stepfun-ai/Step-3.5-Flash")``.
        """
        # PROVIDER_CLASS is populated by the @register_bridge decorator
        from megatron.bridge.models.stepfun.step35_provider import Step35ModelProvider

        assert Step35Bridge.PROVIDER_CLASS is Step35ModelProvider


# ---------------------------------------------------------------------------
# provider_bridge (Step35-specific assignments)
# ---------------------------------------------------------------------------


class TestStep35BridgeProviderBridge:
    """Verify that Step35Bridge.provider_bridge applies the Step-3.5-Flash overrides.

    The parent `MegatronModelBridge.provider_bridge` is patched out so the test
    does not depend on the global CONFIG_MAPPING resolution path.
    """

    def _run(self, hf_overrides=None, provider_overrides=None):
        hf_config = _make_hf_config(**(hf_overrides or {}))
        provider = _FakeProvider()
        # Simulate what `MegatronModelBridge.provider_bridge` (mocked below) would
        # have populated through `CONFIG_MAPPING` before `Step35Bridge.provider_bridge`
        # runs its own assignments.
        provider.num_query_groups = hf_config.num_attention_groups
        provider.num_moe_experts = hf_config.moe_num_experts
        provider.moe_router_topk = hf_config.moe_top_k
        provider.moe_shared_expert_intermediate_size = hf_config.share_expert_dim
        provider.head_wise_attn_gate = hf_config.use_head_wise_attn_gate
        provider.attention_output_gate = hf_config.attention_output_gate
        provider.layer_types = hf_config.layer_types
        provider.kv_channels = hf_config.head_dim
        for k, v in (provider_overrides or {}).items():
            setattr(provider, k, v)

        with patch.object(MegatronModelBridge, "provider_bridge", return_value=provider):
            result = Step35Bridge().provider_bridge(_FakeHFPretrained(hf_config))

        return hf_config, result

    def test_core_field_assignment(self):
        hf_config, p = self._run()
        assert p.num_query_groups == hf_config.num_attention_groups
        assert p.num_moe_experts == hf_config.moe_num_experts
        assert p.moe_router_topk == hf_config.moe_top_k
        assert p.moe_shared_expert_intermediate_size == hf_config.share_expert_dim
        assert p.head_wise_attn_gate is True

    def test_nonstandard_hf_fields_are_mapped_by_base_bridge_path(self):
        # ``attention_other_setting`` is intentionally NOT in CONFIG_MAPPING —
        # the bridge reads it from ``hf_config`` directly when populating
        # ``sliding_attention_setting``.
        hf_config = _make_hf_config()
        kwargs = Step35Bridge().hf_config_to_provider_kwargs(hf_config)
        assert kwargs["num_query_groups"] == hf_config.num_attention_groups
        assert kwargs["num_moe_experts"] == hf_config.moe_num_experts
        assert kwargs["moe_router_topk"] == hf_config.moe_top_k
        assert kwargs["moe_shared_expert_intermediate_size"] == hf_config.share_expert_dim
        assert kwargs["head_wise_attn_gate"] is True
        assert kwargs["layer_types"] == hf_config.layer_types

    def test_attention_output_gate_mapped_by_base_bridge_path(self):
        # HF Step-3.5-Flash can carry ``attention_output_gate`` verbatim into
        # the base provider kwargs. Step35Bridge.provider_bridge normalizes it
        # later when ``head_wise_attn_gate`` is also enabled.
        hf_config = _make_hf_config(attention_output_gate=True)
        kwargs = Step35Bridge().hf_config_to_provider_kwargs(hf_config)
        assert kwargs["attention_output_gate"] is True

    def test_attention_output_gate_entry_in_config_mapping(self):
        # Regression guard for non-head-wise-gate configs that still request
        # MCore's full output-gate layout.
        assert ("attention_output_gate", "attention_output_gate") in Step35Bridge.CONFIG_MAPPING

    def test_head_wise_attn_gate_preserved_through_step35_provider_bridge(self):
        """The Step35-specific provider_bridge must not overwrite ``head_wise_attn_gate``.

        The CONFIG_MAPPING parent path is what sets it (mapped from HF's
        ``use_head_wise_attn_gate``); Step35Bridge.provider_bridge only adds
        the Step-3.5-specific fields on top.
        """
        _, p = self._run()
        assert p.head_wise_attn_gate is True

    def test_native_head_wise_gate_disables_attention_output_gate(self, monkeypatch):
        """MCore versions with native scalar gates do not need the fallback."""
        monkeypatch.setattr(_step35_bridge_mod, "_mcore_supports_head_wise_attn_gate", lambda: True)
        _, p = self._run(
            hf_overrides={
                "attention_output_gate": True,
                "use_head_wise_attn_gate": True,
            },
        )

        assert p.head_wise_attn_gate is True
        assert p.attention_output_gate is False

    def test_attention_output_gate_preserved_as_legacy_mcore_fallback(self, monkeypatch):
        """MCore versions without native scalar gates need expanded gate rows."""
        monkeypatch.setattr(_step35_bridge_mod, "_mcore_supports_head_wise_attn_gate", lambda: False)
        _, p = self._run(
            hf_overrides={
                "attention_output_gate": True,
                "use_head_wise_attn_gate": True,
            },
        )

        assert p.head_wise_attn_gate is True
        assert p.attention_output_gate is True

    def test_head_wise_gate_without_native_support_enables_output_gate_fallback(self, monkeypatch):
        """The published Step-3.5-Flash config carries no ``attention_output_gate``."""
        monkeypatch.setattr(_step35_bridge_mod, "_mcore_supports_head_wise_attn_gate", lambda: False)
        _, p = self._run(
            hf_overrides={
                "attention_output_gate": None,
                "use_head_wise_attn_gate": True,
            },
        )

        assert p.head_wise_attn_gate is True
        assert p.attention_output_gate is True

    def test_head_wise_gate_with_native_support_keeps_output_gate_off(self, monkeypatch):
        """With native scalar gates, no fallback is enabled for the published config."""
        monkeypatch.setattr(_step35_bridge_mod, "_mcore_supports_head_wise_attn_gate", lambda: True)
        _, p = self._run(
            hf_overrides={
                "attention_output_gate": None,
                "use_head_wise_attn_gate": True,
            },
        )

        assert p.head_wise_attn_gate is True
        assert p.attention_output_gate is False

    def test_attention_output_gate_preserved_without_head_wise_gate(self):
        _, p = self._run(
            hf_overrides={
                "attention_output_gate": True,
                "use_head_wise_attn_gate": False,
            },
        )

        assert p.head_wise_attn_gate is False
        assert p.attention_output_gate is True

    def test_sliding_attention_setting_populated_from_hf_config(self):
        """Shape fields are pulled out of HF's ``attention_other_setting`` and
        renamed to the Megatron-facing keys that ``Step35DecoderLayer``
        consumes at construction time. ``window_size`` is padded from HF's
        scalar ``sliding_window`` to a ``[left, right]`` pair."""
        _, p = self._run()
        assert p.sliding_attention_setting == {
            "window_size": [512, 0],
            "num_attention_heads": 96,
            "num_query_groups": 8,
            "kv_channels": 128,
        }

    def test_sliding_attention_setting_falls_back_to_defaults(self):
        """When HF carries no ``sliding_window`` and no sliding-attention
        ``attention_other_setting``, the bridge keeps the built-in defaults."""
        _, p = self._run(
            hf_overrides={"sliding_window": None, "attention_other_setting": None},
        )
        assert p.sliding_attention_setting == {
            "window_size": [512, 0],
            "num_attention_heads": 96,
            "num_query_groups": 8,
            "kv_channels": 128,
        }

    def test_provider_restores_mtp_layer_types_after_config_validation_split(self):
        main_layer_types = [
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ]
        mtp_layer_types = ["full_attention", "sliding_attention"]

        _, p = self._run(
            hf_overrides={
                "layer_types": main_layer_types,
                "mtp_layer_types": mtp_layer_types,
            },
        )

        assert p.layer_types == main_layer_types + mtp_layer_types

    def test_rotary_percents_copied_from_partial_rotary_factors(self):
        """``rotary_percents`` is the Megatron-facing rename of HF's
        ``partial_rotary_factors`` and is indexed per layer by
        ``Step35DecoderLayer``."""
        hf_config, p = self._run()
        assert p.rotary_percents == hf_config.partial_rotary_factors

    def test_rope_theta_scalar(self):
        _, p = self._run(hf_overrides={"rope_theta": 50000.0})
        assert p.rotary_base == 50000.0
        # No per-layer overrides created from a scalar rope_theta.
        assert p.rotary_base_per_layer is None

    def test_rope_theta_per_layer_list(self):
        per_layer = [1000.0 + i for i in range(6)]
        _, p = self._run(hf_overrides={"rope_theta": per_layer})
        assert p.rotary_base == per_layer[0]
        assert p.rotary_base_per_layer == per_layer

    def test_moe_layer_freq_constructed_from_enum(self):
        _, p = self._run()
        # num_layers=4 and moe_layers_enum="2,3" => main decoder layers 2 and 3 are MoE.
        # MTP layers are kept dense by _MTPDenseLayerSpecsList rather than this list.
        assert p.moe_layer_freq == [0, 0, 1, 1]

    def test_moe_layer_freq_skipped_when_enum_none(self):
        provider_overrides = {"moe_layer_freq": "untouched"}
        _, p = self._run(
            hf_overrides={"moe_layers_enum": None},
            provider_overrides=provider_overrides,
        )
        assert p.moe_layer_freq == "untouched"

    def test_static_overrides_applied(self):
        hf_config, p = self._run()
        assert p.normalization == "RMSNorm"
        # HF Step-3.5 weights store γ-1 (HF flag ``zero_centered``); TE norm
        # applies (1+w). The bridge forwards the flag through directly so the
        # behavior follows the upstream config rather than being hard-coded.
        assert p.layernorm_zero_centered_gamma is hf_config.zero_centered
        assert p.gated_linear_unit is True
        assert p.add_bias_linear is False
        assert p.add_qkv_bias is False
        # ``qk_layernorm`` now comes straight from the HF config rather than
        # being hard-coded — older revisions of the bridge pinned it to True.
        assert p.qk_layernorm is hf_config.use_qk_norm
        # ``autocast_dtype`` is normalized to a ``torch.dtype`` regardless of
        # whether HF carried it as a string ("bfloat16") or a ``torch.dtype``.
        assert p.autocast_dtype is torch.bfloat16
        assert p.hidden_dropout == 0.0
        assert p.attention_dropout == 0.0
        assert p.moe_grouped_gemm is True
        assert p.moe_router_load_balancing_type == "aux_loss"
        assert p.moe_aux_loss_coeff == 1e-3
        assert p.moe_router_pre_softmax is False
        assert p.moe_token_dispatcher_type == "alltoall"
        assert p.moe_permute_fusion is True

    def test_autocast_dtype_normalizes_string_aliases(self):
        """HF's ``torch_dtype`` can arrive as a string alias from ``config.json``;
        the bridge normalizes the supported aliases to ``torch.dtype`` instances."""
        for alias, expected in (
            ("bfloat16", torch.bfloat16),
            ("float16", torch.float16),
            ("float32", torch.float32),
        ):
            _, p = self._run(hf_overrides={"torch_dtype": alias})
            assert p.autocast_dtype is expected, f"alias {alias!r} should map to {expected}"

    def test_autocast_dtype_passthrough_for_torch_dtype_instance(self):
        """When HF already gave a ``torch.dtype``, the bridge assigns it as-is."""
        _, p = self._run(hf_overrides={"torch_dtype": torch.float16})
        assert p.autocast_dtype is torch.float16

    def test_autocast_dtype_rejects_unknown_string(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown torch dtype"):
            self._run(hf_overrides={"torch_dtype": "fp8"})

    def test_autocast_dtype_rejects_unsupported_type(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown torch dtype"):
            self._run(hf_overrides={"torch_dtype": 16})

    def test_moe_router_fields_copied_from_hf_config(self):
        """The MoE router-bias / score-function / topk-scaling-factor fields
        are HF-driven (Step-3.5 exposes them in its config) — they should
        flow into the provider 1:1."""
        hf_config, p = self._run()
        assert p.moe_router_enable_expert_bias is hf_config.use_moe_router_bias
        assert p.moe_router_score_function == hf_config.moe_router_activation
        assert p.moe_router_topk_scaling_factor == hf_config.moe_router_scaling_factor

    def test_swiglu_limits_copied_from_hf_config(self):
        """Per-layer SwiGLU clamp values (routed and shared expert) are
        forwarded to the provider; ``Step35DecoderLayer`` indexes them by
        global layer id at construction time."""
        hf_config, p = self._run()
        assert p.swiglu_limits == hf_config.swiglu_limits
        assert p.swiglu_limits_shared == hf_config.swiglu_limits_shared

    def test_need_fp32_gate_sets_moe_router_dtype(self):
        """``need_fp32_gate=True`` opts into the FP32 router-dtype path."""
        _, p = self._run(hf_overrides={"need_fp32_gate": True})
        assert p.moe_router_dtype == "fp32"

    def test_need_fp32_gate_false_leaves_moe_router_dtype_untouched(self):
        _, p = self._run()  # fixture default: need_fp32_gate=False
        assert p.moe_router_dtype is None

    def test_transformer_layer_spec_uses_custom_builder(self):
        _, p = self._run()
        assert p.transformer_layer_spec is build_step35_layer_spec

    def test_transformer_layer_spec_target_passes_checkpoint_load_validation(self):
        """The serialized spec target must be loadable by the checkpoint instantiate path."""
        from megatron.bridge.utils.instantiate_utils import _validate_target_prefix

        _, p = self._run()
        spec = p.transformer_layer_spec
        target = f"{spec.__module__}.{spec.__qualname__}"
        # Raises InstantiationException for private path segments — the exact
        # gate that rejects a converted checkpoint's run_config on load.
        _validate_target_prefix(target=target, full_key="transformer_layer_spec")


# ---------------------------------------------------------------------------
# build_step35_layer_spec
# ---------------------------------------------------------------------------


class TestBuildStep35LayerSpec:
    """Cover the per-spec rewrite loop in ``build_step35_layer_spec``.

    Mocks out the two Megatron-Core spec builders so the test can run without
    a real backend, and feeds them a hand-rolled mix of MoE and dense layer
    specs to exercise both branches of the ``shared_experts`` rebind guard.
    """

    class _OriginalSharedExpert:
        """Sentinel class for the *pre-rebind* shared-expert builder."""

    def _moe_spec(self):
        """Layer spec shaped like an MoE main-decoder layer: mlp.submodules has
        a ``shared_experts`` partial that the bridge must replace."""
        original = partial(self._OriginalSharedExpert, clamp=2.5, gated=True)
        mlp_submods = SimpleNamespace(shared_experts=original)
        return SimpleNamespace(
            module="placeholder",
            submodules=SimpleNamespace(mlp=SimpleNamespace(submodules=mlp_submods)),
        ), original

    def _dense_spec(self):
        """Layer spec shaped like a dense main-decoder layer: mlp has no
        ``submodules`` attribute, so the ``getattr(..., None)`` guard must
        short-circuit without raising."""
        return SimpleNamespace(
            module="placeholder",
            submodules=SimpleNamespace(mlp=SimpleNamespace()),
        )

    def _build(self, layer_specs):
        fake_block = SimpleNamespace(layer_specs=layer_specs)
        fake_dense_mtp = SimpleNamespace(module="placeholder")
        cfg = SimpleNamespace(qk_layernorm=True)
        with (
            patch.object(_step35_bridge_mod, "get_gpt_decoder_block_spec", return_value=fake_block) as mock_block,
            patch.object(
                _step35_bridge_mod,
                "get_gpt_layer_with_transformer_engine_spec",
                return_value=fake_dense_mtp,
            ) as mock_dense,
        ):
            out = build_step35_layer_spec(cfg)
        return out, fake_dense_mtp, mock_block, mock_dense, cfg

    def test_moe_shared_experts_rebound_to_step35_shared_expert_mlp(self):
        moe_spec, original_partial = self._moe_spec()
        out, fake_dense_mtp, mock_block, mock_dense, cfg = self._build([moe_spec])

        # Builders were invoked with the Step35-specific kwargs.
        mock_block.assert_called_once_with(cfg, use_transformer_engine=True, normalization="RMSNorm")
        mock_dense.assert_called_once_with(num_experts=None, moe_grouped_gemm=False, qk_layernorm=True)

        # The MoE spec's ``shared_experts`` was rebound to ``Step35SharedExpertMLP``
        # while preserving the original partial's keyword arguments verbatim.
        new_shared = moe_spec.submodules.mlp.submodules.shared_experts
        assert new_shared is not original_partial
        assert isinstance(new_shared, partial)
        assert new_shared.func is Step35SharedExpertMLP
        assert new_shared.keywords == original_partial.keywords

        # Module class rewritten on both the main-decoder spec and the dense MTP spec.
        assert moe_spec.module is Step35DecoderLayer
        assert fake_dense_mtp.module is Step35DecoderLayer

        # ``layer_specs`` is wrapped so MTP layers resolve to the dense spec on -1.
        assert isinstance(out.layer_specs, _MTPDenseLayerSpecsList)
        assert out.layer_specs[-1] is fake_dense_mtp

    def test_dense_layer_short_circuits_without_mutation(self):
        """A dense layer (no ``mlp.submodules``) must skip the rebind branch
        without raising and without growing a ``submodules`` attribute."""
        dense_spec = self._dense_spec()
        out, *_ = self._build([dense_spec])

        assert dense_spec.module is Step35DecoderLayer
        assert not hasattr(dense_spec.submodules.mlp, "submodules")
        # The dense spec is still the first (and only) entry under forward iteration.
        assert list(out.layer_specs) == [dense_spec]

    def test_mixed_moe_and_dense_layers_both_handled(self):
        """Both branches of the guard exercised in a single pass — order of
        appearance in ``layer_specs`` must not affect the per-spec decision."""
        dense_spec = self._dense_spec()
        moe_spec, original_partial = self._moe_spec()
        out, *_ = self._build([dense_spec, moe_spec])

        assert dense_spec.module is Step35DecoderLayer
        assert moe_spec.module is Step35DecoderLayer
        assert not hasattr(dense_spec.submodules.mlp, "submodules")
        new_shared = moe_spec.submodules.mlp.submodules.shared_experts
        assert new_shared.func is Step35SharedExpertMLP
        assert new_shared.keywords == original_partial.keywords
        assert list(out.layer_specs) == [dense_spec, moe_spec]


# ---------------------------------------------------------------------------
# mapping_registry
# ---------------------------------------------------------------------------


class TestStep35BridgeMappingRegistry:
    def _registry(self, num_nextn_predict_layers=2, num_hidden_layers=4):
        bridge = Step35Bridge()
        bridge.hf_config = SimpleNamespace(
            num_nextn_predict_layers=num_nextn_predict_layers,
            num_hidden_layers=num_hidden_layers,
        )
        return list(bridge.mapping_registry())

    def test_main_decoder_mappings_present(self):
        params = [str(m.megatron_param) for m in self._registry()]
        assert "embedding.word_embeddings.weight" in params
        assert "output_layer.weight" in params
        assert "decoder.final_layernorm.weight" in params
        assert any("linear_qkv.weight" in p and "mtp" not in p for p in params)
        assert any("linear_proj.weight" in p and "mtp" not in p for p in params)
        assert any("router.weight" in p for p in params)
        assert any("router.expert_bias" in p for p in params)
        assert any("q_layernorm.weight" in p and "mtp" not in p for p in params)
        assert any("k_layernorm.weight" in p and "mtp" not in p for p in params)

    def test_qkvg_mapping_includes_g_proj(self):
        """The Step-3.5-Flash per-head g_proj must be fused into linear_qkv."""
        qkvg = [m for m in self._registry() if isinstance(m, QKVGMapping) and "mtp" not in m.megatron_param]
        assert qkvg, "QKVGMapping for the main decoder is missing"
        assert qkvg[0].hf_param.get("g") == "model.layers.*.self_attn.g_proj.weight"

    def test_stacked_expert_mappings_for_moe(self):
        mappings = self._registry()
        # MoE fc1 (gate + up stacked across experts)
        assert any(isinstance(m, StackedExpertGatedMLPMapping) for m in mappings)
        # MoE fc2 (down stacked across experts)
        assert any(isinstance(m, StackedExpertAutoMapping) for m in mappings)

    def test_dense_gated_mlp_mapping_present(self):
        """Dense MLP fc1 mapping covers layers 0-2 plus MTP layers."""
        gated = [m for m in self._registry() if isinstance(m, GatedMLPMapping)]
        assert any(
            "mlp.linear_fc1.weight" in str(m.megatron_param)
            and "share" not in str(m.megatron_param)
            and "experts" not in str(m.megatron_param)
            for m in gated
        )

    def test_no_mtp_mappings_when_hf_config_none(self):
        bridge = Step35Bridge()
        bridge.hf_config = None  # explicit None triggers the warning branch
        registry = list(bridge.mapping_registry())
        assert all("mtp." not in str(m.megatron_param) for m in registry)

    def test_mtp_mappings_generated_for_each_layer_and_prefix(self):
        registry = self._registry(num_nextn_predict_layers=2, num_hidden_layers=4)
        mtp_params = [str(m.megatron_param) for m in registry if str(m.megatron_param).startswith("mtp.")]
        # 2 MTP layers x both sub-layer prefixes ('mtp_model_layer' / 'transformer_layer')
        for layer in (0, 1):
            for prefix in ("mtp_model_layer", "transformer_layer"):
                assert any(p.startswith(f"mtp.layers.{layer}.{prefix}.") for p in mtp_params), (
                    f"missing mappings for mtp.layers.{layer}.{prefix}"
                )

    def test_mtp_layer_index_offset_to_hf(self):
        """MTP layer N must reference HF layer (num_hidden_layers + N)."""
        registry = self._registry(num_nextn_predict_layers=2, num_hidden_layers=4)
        # Look at one of the auto-generated AutoMapping entries.
        mtp_auto = [
            m
            for m in registry
            if isinstance(m, AutoMapping)
            and str(m.megatron_param) == "mtp.layers.1.mtp_model_layer.self_attention.linear_proj.weight"
        ]
        assert mtp_auto, "expected AutoMapping for MTP layer 1 linear_proj"
        # HF index = num_hidden_layers + mtp_layer_idx = 4 + 1 = 5
        assert mtp_auto[0].hf_param == "model.layers.5.self_attn.o_proj.weight"

    def test_mtp_specific_norm_and_proj_present(self):
        registry = self._registry(num_nextn_predict_layers=1, num_hidden_layers=4)
        params = [str(m.megatron_param) for m in registry]
        assert "mtp.layers.0.enorm.weight" in params
        assert "mtp.layers.0.hnorm.weight" in params
        assert "mtp.layers.0.eh_proj.weight" in params
        assert "mtp.layers.0.final_layernorm.weight" in params


# ---------------------------------------------------------------------------
# StackedExpertAutoMapping / StackedExpertGatedMLPMapping
# ---------------------------------------------------------------------------


class _RecordingAutoMapping(AutoMapping):
    """Captures the first positional arg passed to `super().hf_to_megatron`."""

    captured = None

    def hf_to_megatron(self, hf_weights, megatron_module):
        _RecordingAutoMapping.captured = hf_weights
        return hf_weights


class TestStackedExpertAutoMapping:
    def test_expert_idx_parsed_from_megatron_param(self):
        m = StackedExpertAutoMapping(
            megatron_param="decoder.layers.3.mlp.experts.linear_fc2.weight7",
            hf_param="model.layers.3.moe.down_proj.weight",
        )
        assert m._expert_idx() == 7

    def test_hf_to_megatron_slices_to_expert(self, monkeypatch):
        m = StackedExpertAutoMapping(
            megatron_param="decoder.layers.0.mlp.experts.linear_fc2.weight2",
            hf_param="model.layers.0.moe.down_proj.weight",
        )
        # 4-expert stacked tensor; expert 2's row should be selected.
        stacked = torch.stack([torch.full((3, 4), float(i)) for i in range(4)])

        captured = {}
        monkeypatch.setattr(
            AutoMapping,
            "hf_to_megatron",
            lambda self, w, mod: captured.setdefault("w", w),
        )
        m.hf_to_megatron(stacked, megatron_module=None)
        assert torch.equal(captured["w"], stacked[2])


class TestStackedExpertGatedMLPMapping:
    def test_hf_to_megatron_slices_both_gate_and_up(self, monkeypatch):
        m = StackedExpertGatedMLPMapping(
            megatron_param="decoder.layers.5.mlp.experts.linear_fc1.weight1",
            gate="model.layers.5.moe.gate_proj.weight",
            up="model.layers.5.moe.up_proj.weight",
        )
        gate = torch.stack([torch.full((2, 3), float(i)) for i in range(4)])
        up = torch.stack([torch.full((2, 3), float(10 + i)) for i in range(4)])

        seen = {}
        monkeypatch.setattr(
            GatedMLPMapping,
            "hf_to_megatron",
            lambda self, w, mod: seen.setdefault("w", w),
        )
        m.hf_to_megatron({"gate": gate, "up": up}, megatron_module=None)

        # Expert index 1 -> both gate/up tensors must be sliced to row 1.
        assert torch.equal(seen["w"]["gate"], gate[1])
        assert torch.equal(seen["w"]["up"], up[1])

    @patch("megatron.bridge.models.conversion.model_bridge.parallel_state.get_expert_model_parallel_world_size")
    def test_grouped_export_accumulates_gate_and_up_separately(self, mock_ep_size):
        mock_ep_size.return_value = 1
        model_config = SimpleNamespace(num_moe_experts=2)
        mapping = SimpleNamespace(is_grouped_export=True)
        buffers = {}

        task0 = SimpleNamespace(
            mapping=mapping,
            param_name="decoder.layers.5.mlp.experts.linear_fc1.weight0",
        )
        first = MegatronModelBridge._accumulate_grouped_export(
            None,
            task0,
            {
                "model.layers.5.moe.gate_proj.weight": torch.full((2, 3), 1.0),
                "model.layers.5.moe.up_proj.weight": torch.full((2, 3), 2.0),
            },
            model_config,
            buffers,
            {},
        )

        task1 = SimpleNamespace(
            mapping=mapping,
            param_name="decoder.layers.5.mlp.experts.linear_fc1.weight1",
        )
        second = MegatronModelBridge._accumulate_grouped_export(
            None,
            task1,
            {
                "model.layers.5.moe.gate_proj.weight": torch.full((2, 3), 3.0),
                "model.layers.5.moe.up_proj.weight": torch.full((2, 3), 4.0),
            },
            model_config,
            buffers,
            {},
        )

        assert first is None
        assert set(second) == {
            "model.layers.5.moe.gate_proj.weight",
            "model.layers.5.moe.up_proj.weight",
        }
        assert torch.equal(
            second["model.layers.5.moe.gate_proj.weight"],
            torch.stack([torch.ones(2, 3), torch.full((2, 3), 3.0)]),
        )
        assert torch.equal(
            second["model.layers.5.moe.up_proj.weight"],
            torch.stack([torch.full((2, 3), 2.0), torch.full((2, 3), 4.0)]),
        )


class TestQKVGMappingHelpers:
    def test_scalar_gate_expands_for_mcore_attention_output_gate_layout(self):
        provider = SimpleNamespace(
            attention_output_gate=True,
            num_attention_heads=4,
            num_query_groups=2,
            kv_channels=2,
            hidden_size=3,
        )
        q = torch.arange(4 * 2 * 3, dtype=torch.float32).reshape(8, 3)
        k = torch.arange(100, 100 + 2 * 2 * 3, dtype=torch.float32).reshape(4, 3)
        v = torch.arange(200, 200 + 2 * 2 * 3, dtype=torch.float32).reshape(4, 3)
        g = torch.arange(300, 300 + 4 * 3, dtype=torch.float32).reshape(4, 3)

        merged = merge_qkvg_weights(provider, q, k, v, g)

        assert merged.shape == (24, 3)
        q2, k2, v2, g2 = split_qkvg_weights(provider, merged)
        assert torch.equal(q2, q)
        assert torch.equal(k2, k)
        assert torch.equal(v2, v)
        assert torch.equal(g2, g)


# ---------------------------------------------------------------------------
# _MTPDenseLayerSpecsList
# ---------------------------------------------------------------------------


class TestMTPDenseLayerSpecsList:
    def test_negative_index_returns_dense_spec(self):
        sentinel = object()
        lst = _MTPDenseLayerSpecsList(["a", "b", "c"], dense_mtp_spec=sentinel)
        assert lst[-1] is sentinel
        assert lst[-2] is sentinel

    def test_positive_index_falls_through(self):
        lst = _MTPDenseLayerSpecsList(["a", "b", "c"], dense_mtp_spec=object())
        assert lst[0] == "a"
        assert lst[2] == "c"

    def test_iteration_unaffected(self):
        """`TransformerBlock` iterates via the C-level list iterator, which
        bypasses `__getitem__`. Iteration must therefore yield the real specs,
        not the dense MTP sentinel."""
        sentinel = object()
        lst = _MTPDenseLayerSpecsList(["a", "b", "c"], dense_mtp_spec=sentinel)
        assert list(lst) == ["a", "b", "c"]
