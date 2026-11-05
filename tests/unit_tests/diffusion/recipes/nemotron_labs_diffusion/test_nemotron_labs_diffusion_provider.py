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

"""Unit tests for NemotronLabsDiffusionModelProvider."""

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest
from megatron.core.transformer.spec_utils import ModuleSpec

from megatron.bridge.diffusion.models.common.nemotron_labs_diffusion_attention import NemotronLabsDiffusionAttention
from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
    NemotronLabsDiffusionModelProvider,
)
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**overrides):
    """Return a NemotronLabsDiffusionModelProvider with minimal required fields set."""
    defaults = dict(
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_layers=4,
        num_attention_heads=8,
        vocab_size=32000,
    )
    defaults.update(overrides)
    return NemotronLabsDiffusionModelProvider(**defaults)


def _make_submodules_spec():
    """Build a fake ModuleSpec tree that mirrors the real Megatron layout."""
    core_attn = MagicMock()
    self_attention = MagicMock()
    self_attention.submodules = MagicMock(core_attention=core_attn)

    submodules = MagicMock()
    submodules.self_attention = self_attention

    spec = MagicMock()
    spec.submodules = submodules
    return spec


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------


class TestNemotronLabsDiffusionModelProviderDefaults:
    """Ensure all diffusion-specific fields have the expected defaults."""

    def setup_method(self):
        self.provider = _make_provider()

    def test_mask_token_id_default(self):
        assert self.provider.mask_token_id == 100

    def test_dlm_paradigm_default(self):
        assert self.provider.dlm_paradigm == "sbd_block_diff"

    def test_block_size_default(self):
        assert self.provider.block_size == 64

    def test_different_seed_per_dp_default(self):
        assert self.provider.different_seed_per_dp is True

    def test_apply_llama4_style_scaling_default(self):
        assert self.provider.apply_llama4_style_query_key_layer_scaling is False

    def test_dlm_loss_weight_default(self):
        assert self.provider.dlm_loss_weight == pytest.approx(0.3)

    def test_ar_loss_weight_default(self):
        assert self.provider.ar_loss_weight == pytest.approx(1.0)

    def test_position_embedding_type_default(self):
        assert self.provider.position_embedding_type == "none"

    def test_has_no_hf_or_vision_config(self):
        from megatron.bridge.training.utils.config_utils import _ConfigContainerBase
        from megatron.bridge.utils.instantiate_utils import instantiate

        serialized = _ConfigContainerBase._convert_value_to_dict(self.provider)
        restored = instantiate(serialized)
        names = {field.name for field in fields(self.provider)}
        for name in (
            "hf_config",
            "image_token_id",
            "freeze_language_model",
            "freeze_vision_model",
            "freeze_vision_projection",
        ):
            assert name not in names
            assert name not in serialized
            assert not hasattr(self.provider, name)
            assert not hasattr(restored, name)


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestNemotronLabsDiffusionModelProviderInheritance:
    """NemotronLabsDiffusionModelProvider must inherit from MistralModelProvider."""

    def test_is_subclass_of_mistral(self):
        assert issubclass(NemotronLabsDiffusionModelProvider, MistralModelProvider)

    def test_instance_is_mistral(self):
        provider = _make_provider()
        assert isinstance(provider, MistralModelProvider)


# ---------------------------------------------------------------------------
# Overriding defaults via constructor
# ---------------------------------------------------------------------------


class TestNemotronLabsDiffusionModelProviderOverrides:
    """Constructor keyword arguments must override defaults."""

    def test_custom_mask_token_id(self):
        provider = _make_provider(mask_token_id=999)
        assert provider.mask_token_id == 999

    def test_custom_block_size(self):
        provider = _make_provider(block_size=128)
        assert provider.block_size == 128

    def test_custom_dlm_loss_weight(self):
        provider = _make_provider(dlm_loss_weight=0.5)
        assert provider.dlm_loss_weight == pytest.approx(0.5)

    def test_custom_ar_loss_weight(self):
        provider = _make_provider(ar_loss_weight=2.0)
        assert provider.ar_loss_weight == pytest.approx(2.0)

    def test_disable_different_seed_per_dp(self):
        provider = _make_provider(different_seed_per_dp=False)
        assert provider.different_seed_per_dp is False


# ---------------------------------------------------------------------------
# provide() — core_attention replacement
# ---------------------------------------------------------------------------


class TestNemotronLabsDiffusionProvideMethod:
    """provide() must replace core_attention with NemotronLabsDiffusionAttention."""

    def _run_provide(self, provider, spec):
        """Call provider.provide() with a mocked transformer_layer_spec and parent provide()."""
        provider.transformer_layer_spec = lambda cfg: spec

        resolved_specs = []

        def build(*args):
            resolved_specs.append(provider.transformer_layer_spec)
            return MagicMock()

        with patch.object(MistralModelProvider, "provide", side_effect=build) as mock_parent:
            result = provider.provide()

        return result, mock_parent, resolved_specs[0]

    def test_core_attention_replaced_with_nemotron_labs_diffusion_attention(self):
        provider = _make_provider()
        spec = _make_submodules_spec()
        original_core = spec.submodules.self_attention.submodules.core_attention
        _, _, resolved_spec = self._run_provide(provider, spec)
        assert resolved_spec.submodules.self_attention.submodules.core_attention.func is NemotronLabsDiffusionAttention
        assert spec.submodules.self_attention.submodules.core_attention is original_core

    def test_parent_provide_called_once(self):
        provider = _make_provider()
        spec = _make_submodules_spec()
        _, mock_parent, _ = self._run_provide(provider, spec)
        mock_parent.assert_called_once()

    def test_transformer_layer_spec_restored_after_provide(self):
        """provide() temporarily resolves transformer_layer_spec for the parent call,
        then restores the original. After provide() returns, the field must be the
        original callable, not the resolved ModuleSpec."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        original = lambda cfg: spec  # noqa: E731
        provider.transformer_layer_spec = original

        with patch.object(MistralModelProvider, "provide", return_value=MagicMock()):
            provider.provide()

        assert provider.transformer_layer_spec is original

    def test_provide_with_already_resolved_modulespec(self):
        """When transformer_layer_spec is already a ModuleSpec, it must not be called again."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        real_spec = ModuleSpec(module=MagicMock, submodules=spec.submodules)
        original_core = real_spec.submodules.self_attention.submodules.core_attention
        provider.transformer_layer_spec = real_spec
        resolved_specs = []

        def build(*args):
            resolved_specs.append(provider.transformer_layer_spec)
            return MagicMock()

        with patch.object(MistralModelProvider, "provide", side_effect=build):
            provider.provide()

        assert provider.transformer_layer_spec is real_spec
        assert real_spec.submodules.self_attention.submodules.core_attention is original_core
        assert (
            resolved_specs[0].submodules.self_attention.submodules.core_attention.func
            is NemotronLabsDiffusionAttention
        )

    def test_provide_passes_pre_post_process_to_parent(self):
        """pre_process and post_process arguments must be forwarded to the parent."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        provider.transformer_layer_spec = lambda cfg: spec

        with patch.object(MistralModelProvider, "provide", return_value=MagicMock()) as mock_parent:
            provider.provide(pre_process=True, post_process=False)

        mock_parent.assert_called_once_with(True, False, None)

    def test_provide_passes_vp_stage_to_parent(self):
        provider = _make_provider()
        spec = _make_submodules_spec()
        provider.transformer_layer_spec = lambda cfg: spec

        with patch.object(MistralModelProvider, "provide", return_value=MagicMock()) as mock_parent:
            provider.provide(vp_stage=2)

        mock_parent.assert_called_once_with(None, None, 2)

    def test_provide_layer_spec_callable_receives_vp_stage_when_supported(self):
        """If the callable layer spec accepts vp_stage, it must be called with it."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        received_kwargs = {}

        def layer_spec_fn(cfg, vp_stage=None):
            received_kwargs["vp_stage"] = vp_stage
            return spec

        provider.transformer_layer_spec = layer_spec_fn

        with patch.object(MistralModelProvider, "provide", return_value=MagicMock()):
            provider.provide(vp_stage=3)

        assert received_kwargs["vp_stage"] == 3

    def test_provide_layer_spec_callable_without_vp_stage_param(self):
        """If the callable does NOT accept vp_stage, it must be called without it."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        call_count = {"n": 0}

        def layer_spec_fn(cfg):
            call_count["n"] += 1
            return spec

        provider.transformer_layer_spec = layer_spec_fn

        with patch.object(MistralModelProvider, "provide", return_value=MagicMock()):
            provider.provide(vp_stage=1)

        assert call_count["n"] == 1

    def test_provide_noop_when_no_submodules(self):
        """If the spec has no submodules attribute, provide() must not raise."""
        provider = _make_provider()
        spec_no_submodules = MagicMock(spec=[])  # spec without 'submodules'

        provider.transformer_layer_spec = lambda cfg: spec_no_submodules

        with patch.object(MistralModelProvider, "provide", return_value=MagicMock()):
            provider.provide()  # must not raise


def test_text_defaults_match_ministral3():
    from megatron.bridge.models.ministral3.ministral3_provider import Ministral3ModelProvider

    native = NemotronLabsDiffusionModelProvider(num_layers=1, hidden_size=128)
    reference = Ministral3ModelProvider(num_layers=1, hidden_size=128)
    for name in (
        "normalization",
        "activation_func",
        "add_bias_linear",
        "gated_linear_unit",
        "num_attention_heads",
        "num_query_groups",
        "kv_channels",
        "seq_length",
        "rotary_base",
        "yarn_rotary_scaling_factor",
        "yarn_original_max_position_embeddings",
        "yarn_beta_fast",
        "yarn_beta_slow",
        "yarn_correction_range_round_to_int",
        "yarn_mscale",
        "yarn_mscale_all_dim",
        "attention_dropout",
        "hidden_dropout",
        "share_embeddings_and_output_weights",
        "init_method_std",
        "layernorm_epsilon",
        "params_dtype",
        "bf16",
    ):
        assert getattr(native, name) == getattr(reference, name), name
