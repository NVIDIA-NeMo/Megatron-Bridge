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

"""Unit tests for NemotronDiffusionModelProvider."""

from unittest.mock import MagicMock, patch

import pytest

from megatron.bridge.diffusion.models.common.nemotron_diffusion_attention import NemotronDiffusionAttention
from megatron.bridge.diffusion.models.nemotron_diffusion.nemotron_diffusion_provider import (
    NemotronDiffusionModelProvider,
)
from megatron.bridge.models.ministral3.ministral3_provider import Ministral3ModelProvider


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider(**overrides):
    """Return a NemotronDiffusionModelProvider with minimal required fields set."""
    defaults = dict(
        hidden_size=1024,
        ffn_hidden_size=4096,
        num_layers=4,
        num_attention_heads=8,
        vocab_size=32000,
    )
    defaults.update(overrides)
    return NemotronDiffusionModelProvider(**defaults)


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


class TestNemotronDiffusionModelProviderDefaults:
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
        assert self.provider.apply_llama4_style_query_key_layer_scaling is True

    def test_dlm_loss_weight_default(self):
        assert self.provider.dlm_loss_weight == pytest.approx(0.3)

    def test_ar_loss_weight_default(self):
        assert self.provider.ar_loss_weight == pytest.approx(1.0)

    def test_position_embedding_type_default(self):
        assert self.provider.position_embedding_type == "none"

    def test_freeze_vision_model_default(self):
        assert self.provider.freeze_vision_model is True


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestNemotronDiffusionModelProviderInheritance:
    """NemotronDiffusionModelProvider must inherit from Ministral3ModelProvider."""

    def test_is_subclass_of_ministral3(self):
        assert issubclass(NemotronDiffusionModelProvider, Ministral3ModelProvider)

    def test_instance_is_ministral3(self):
        provider = _make_provider()
        assert isinstance(provider, Ministral3ModelProvider)


# ---------------------------------------------------------------------------
# Overriding defaults via constructor
# ---------------------------------------------------------------------------


class TestNemotronDiffusionModelProviderOverrides:
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

    def test_disable_freeze_vision(self):
        provider = _make_provider(freeze_vision_model=False)
        assert provider.freeze_vision_model is False

    def test_disable_different_seed_per_dp(self):
        provider = _make_provider(different_seed_per_dp=False)
        assert provider.different_seed_per_dp is False


# ---------------------------------------------------------------------------
# provide() — core_attention replacement
# ---------------------------------------------------------------------------


class TestNemotronDiffusionProvideMethod:
    """provide() must replace core_attention with NemotronDiffusionAttention."""

    def _run_provide(self, provider, spec):
        """Call provider.provide() with a mocked transformer_layer_spec and parent provide()."""
        provider.transformer_layer_spec = lambda cfg: spec

        with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()) as mock_parent:
            result = provider.provide()

        return result, mock_parent, spec

    def test_core_attention_replaced_with_nemotron_diffusion_attention(self):
        provider = _make_provider()
        spec = _make_submodules_spec()
        self._run_provide(provider, spec)
        assert spec.submodules.self_attention.submodules.core_attention is NemotronDiffusionAttention

    def test_parent_provide_called_once(self):
        provider = _make_provider()
        spec = _make_submodules_spec()
        _, mock_parent, _ = self._run_provide(provider, spec)
        mock_parent.assert_called_once()

    def test_transformer_layer_spec_updated_on_provider(self):
        """After provide(), self.transformer_layer_spec should be the resolved ModuleSpec."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        self._run_provide(provider, spec)
        assert provider.transformer_layer_spec is spec

    def test_provide_with_already_resolved_modulespec(self):
        """When transformer_layer_spec is already a ModuleSpec, it must not be called again."""
        from megatron.bridge.models.gpt_provider import ModuleSpec

        provider = _make_provider()
        spec = _make_submodules_spec()
        # Patch isinstance to treat the MagicMock as a ModuleSpec
        with patch(
            "megatron.bridge.diffusion.models.nemotron_diffusion.nemotron_diffusion_provider.ModuleSpec",
            ModuleSpec,
        ):
            # Make the spec an actual ModuleSpec instance
            real_spec = MagicMock(spec=ModuleSpec)
            real_spec.submodules = spec.submodules
            provider.transformer_layer_spec = real_spec

            with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()):
                provider.provide()

        assert real_spec.submodules.self_attention.submodules.core_attention is NemotronDiffusionAttention

    def test_provide_passes_pre_post_process_to_parent(self):
        """pre_process and post_process arguments must be forwarded to the parent."""
        provider = _make_provider()
        spec = _make_submodules_spec()
        provider.transformer_layer_spec = lambda cfg: spec

        with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()) as mock_parent:
            provider.provide(pre_process=True, post_process=False)

        mock_parent.assert_called_once_with(True, False, None)

    def test_provide_passes_vp_stage_to_parent(self):
        provider = _make_provider()
        spec = _make_submodules_spec()
        provider.transformer_layer_spec = lambda cfg: spec

        with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()) as mock_parent:
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

        with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()):
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

        with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()):
            provider.provide(vp_stage=1)

        assert call_count["n"] == 1

    def test_provide_noop_when_no_submodules(self):
        """If the spec has no submodules attribute, provide() must not raise."""
        provider = _make_provider()
        spec_no_submodules = MagicMock(spec=[])  # spec without 'submodules'

        provider.transformer_layer_spec = lambda cfg: spec_no_submodules

        with patch.object(Ministral3ModelProvider, "provide", return_value=MagicMock()):
            provider.provide()  # must not raise
