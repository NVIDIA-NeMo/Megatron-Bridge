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

"""Unit tests for Step35ModelProvider / Step35DecoderLayer / Step35Config."""

import dataclasses
from types import SimpleNamespace
from unittest.mock import patch

from megatron.core.transformer.transformer_layer import TransformerLayer

from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.models.stepfun.configuration_step35 import Step35Config
from megatron.bridge.models.stepfun.step35_provider import (
    Step35DecoderLayer,
    Step35ModelProvider,
)


# ---------------------------------------------------------------------------
# Step35Config (HF identifiers preserved)
# ---------------------------------------------------------------------------


class TestStep35Config:
    def test_class_attributes_preserve_hf_identifiers(self):
        """HF-facing strings must stay as `step3p5` / `Step3p5ForCausalLM` even
        though the Python class is `Step35Config`."""
        assert Step35Config.model_type == "step3p5"
        assert Step35Config.architectures == ["Step3p5ForCausalLM"]

    def test_defaults(self):
        cfg = Step35Config()
        # Architecture defaults that downstream code relies on.
        assert cfg.hidden_size == 4096
        assert cfg.num_attention_heads == 64
        assert cfg.num_attention_groups == 8
        assert cfg.num_hidden_layers == 45
        assert cfg.moe_num_experts == 288
        assert cfg.moe_top_k == 8
        assert cfg.head_dim == 128
        # MoE layer enumeration covers layers 3-44 (42 entries).
        assert min(cfg.moe_layers_enum) == 3
        assert max(cfg.moe_layers_enum) == 44
        assert len(cfg.moe_layers_enum) == 42

    def test_overrides_round_trip(self):
        cfg = Step35Config(hidden_size=1024, moe_num_experts=8, head_dim=64)
        assert cfg.hidden_size == 1024
        assert cfg.moe_num_experts == 8
        assert cfg.head_dim == 64


# ---------------------------------------------------------------------------
# Step35ModelProvider
# ---------------------------------------------------------------------------


class TestStep35ModelProvider:
    def test_is_gpt_provider_subclass(self):
        assert issubclass(Step35ModelProvider, GPTModelProvider)

    def test_dataclass_has_step35_specific_fields(self):
        fields = {f.name for f in dataclasses.fields(Step35ModelProvider)}
        assert "layer_types" in fields
        assert "attention_other_setting" in fields

    def test_default_values(self):
        # Build the dataclass using only the new fields; everything else falls
        # back to GPTModelProvider defaults, which we do not exercise here.
        defaults = {
            f.name: f.default
            for f in dataclasses.fields(Step35ModelProvider)
            if f.name in ("layer_types", "attention_other_setting")
        }
        assert defaults == {"layer_types": None, "attention_other_setting": None}


# ---------------------------------------------------------------------------
# Step35DecoderLayer (hybrid attention layer constructor)
# ---------------------------------------------------------------------------


def _make_config(layer_types, sliding_setting=None, *, attention_other_setting=True):
    """Build a TransformerConfig-like SimpleNamespace for Step35DecoderLayer."""
    cfg = SimpleNamespace(
        layer_types=layer_types,
        attention_other_setting=attention_other_setting,
        sliding_attention_setting=sliding_setting
        or {
            "rotary_percent": 1.0,
            "num_attention_heads": 96,
            "num_query_groups": 8,
            "head_dim": 128,
        },
        # Pre-existing values that must be overridden for sliding layers.
        rotary_percent=0.5,
        num_attention_heads=64,
        num_query_groups=8,
        kv_channels=128,
        num_layers=len(layer_types),
    )
    return cfg


class _SuperInitRecorder:
    """Captures the config that Step35DecoderLayer hands to TransformerLayer."""

    def __init__(self):
        self.captured_config = None

    def __call__(self, instance, config, **_):
        self.captured_config = config


class TestStep35DecoderLayerIsSliding:
    """The constructor must distinguish full vs sliding layers and only deep-copy /
    override the config when needed. Verified by patching out TransformerLayer.__init__
    so we can introspect what super() sees."""

    def _build(
        self,
        *,
        layer_number,
        is_mtp_layer=False,
        add_layer_offset=True,
        layer_types=None,
        attention_other_setting=True,
        offset_return=0,
        pp_rank=0,
    ):
        layer_types = (
            layer_types
            if layer_types is not None
            else [
                "full_attention",
                "sliding_attention",
            ]
        )
        config = _make_config(layer_types, attention_other_setting=attention_other_setting)
        recorder = _SuperInitRecorder()

        with (
            patch.object(TransformerLayer, "__init__", lambda self, config, **kw: recorder(self, config, **kw)),
            patch("megatron.bridge.models.stepfun.step35_provider.get_pg_rank", return_value=pp_rank),
            patch(
                "megatron.bridge.models.stepfun.step35_provider.get_transformer_layer_offset",
                return_value=offset_return,
            ),
        ):
            Step35DecoderLayer(
                config=config,
                submodules=None,
                layer_number=layer_number,
                pg_collection=SimpleNamespace(pp="dummy"),
                vp_stage=None,
                is_mtp_layer=is_mtp_layer,
                add_layer_offset=add_layer_offset,
            )

        return config, recorder.captured_config

    def test_full_attention_keeps_original_config(self):
        original, captured = self._build(layer_number=1)  # layer_idx=0 -> full_attention
        # No deep-copy happens; downstream sub-modules see the original instance.
        assert captured is original
        assert captured.num_attention_heads == 64
        assert captured.num_query_groups == 8

    def test_sliding_attention_overrides_shape(self):
        original, captured = self._build(layer_number=2)  # layer_idx=1 -> sliding
        # Must be a deep-copy so other layers' configs are not mutated.
        assert captured is not original
        assert captured.rotary_percent == 1.0
        assert captured.num_attention_heads == 96
        assert captured.num_query_groups == 8
        assert captured.kv_channels == 128
        # And the original config is untouched.
        assert original.num_attention_heads == 64
        assert original.rotary_percent == 0.5

    def test_mtp_layer_uses_global_layer_index_after_main_decoder(self):
        """For ``is_mtp_layer=True`` the layer index is offset after the main
        decoder, so MTP entries can be represented in per-layer config lists."""
        original, captured = self._build(
            layer_number=1,
            is_mtp_layer=True,
            offset_return=100,  # would have triggered out-of-range sliding lookup
            pp_rank=2,
        )
        assert captured is original  # full attention

    def test_pp_offset_applied_for_main_decoder(self):
        """With ``add_layer_offset=True`` the resolved index is
        ``layer_number + get_transformer_layer_offset(...) - 1``; the test forces
        offset=1 so layer_number=1 maps to index 1 (sliding) instead of 0 (full)."""
        original, captured = self._build(layer_number=1, offset_return=1)
        assert captured is not original
        assert captured.num_attention_heads == 96

    def test_no_attention_other_setting_disables_override(self):
        """``attention_other_setting`` acts as the truthy enable flag — when
        unset (None / falsy), even ``sliding_attention`` layers fall through to
        the global config."""
        original, captured = self._build(
            layer_number=2,
            attention_other_setting=None,
        )
        assert captured is original
        assert captured.num_attention_heads == 64

    def test_layer_idx_outside_layer_types_falls_through(self):
        original, captured = self._build(
            layer_number=10,  # idx=9, outside len(layer_types)=2
            layer_types=["full_attention", "sliding_attention"],
        )
        assert captured is original

    def test_sliding_override_does_not_leak_via_alias(self):
        """Mutating the deep-copied config must not change the original's
        ``sliding_attention_setting`` dict."""
        original, captured = self._build(layer_number=2)
        captured.sliding_attention_setting["num_attention_heads"] = 999
        assert original.sliding_attention_setting["num_attention_heads"] == 96
