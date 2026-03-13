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

"""Unit tests for Flux model."""

from contextlib import nullcontext

import pytest
import torch
from megatron.core import parallel_state
from torch import nn

from megatron.bridge.diffusion.models.flux.flux_model import Flux
from megatron.bridge.diffusion.models.flux.flux_provider import FluxProvider


pytestmark = [pytest.mark.unit]


# Dummy blocks so we can build Flux without Transformer Engine (TE) or distributed init.
# They accept the same constructor/forward args as the real layers and preserve shapes.


class DummyMMDiTLayer(nn.Module):
    """Pass-through double block; same forward signature as MMDiTLayer."""

    def __init__(self, config=None, submodules=None, layer_number=0, context_pre_only=False):
        super().__init__()
        self.layer_number = layer_number

    def _get_layer_offset(self, config):
        return 0

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {}

    def forward(self, hidden_states, encoder_hidden_states, rotary_pos_emb, emb):
        return hidden_states, encoder_hidden_states


class DummyFluxSingleTransformerBlock(nn.Module):
    """Pass-through single block; same forward signature as FluxSingleTransformerBlock."""

    def __init__(self, config=None, submodules=None, layer_number=0):
        super().__init__()
        self.layer_number = layer_number

    def _get_layer_offset(self, config):
        return 0

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return {}

    def forward(self, hidden_states, rotary_pos_emb, emb):
        return hidden_states, None


def _mock_flux_layers(monkeypatch):
    """Replace TE-dependent Flux layers with dummies so Flux can be built without TE."""
    import megatron.bridge.diffusion.models.flux.flux_model as flux_model_module

    monkeypatch.setattr(flux_model_module, "MMDiTLayer", DummyMMDiTLayer, raising=False)
    monkeypatch.setattr(
        flux_model_module, "FluxSingleTransformerBlock", DummyFluxSingleTransformerBlock, raising=False
    )


def _mock_parallel_state(monkeypatch):
    """Mock parallel_state and Flux TE-dependent layers so Flux can be built without TE/distributed init."""
    _mock_flux_layers(monkeypatch)
    monkeypatch.setattr(parallel_state, "is_pipeline_first_stage", lambda: True, raising=False)
    monkeypatch.setattr(parallel_state, "is_pipeline_last_stage", lambda: True, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda *args, **kwargs: 1, raising=False)
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_group", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_data_parallel_group", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(
        parallel_state, "get_tensor_and_data_parallel_group", lambda *args, **kwargs: None, raising=False
    )
    monkeypatch.setattr(
        parallel_state, "get_pipeline_model_parallel_group", lambda *args, **kwargs: None, raising=False
    )
    monkeypatch.setattr(parallel_state, "model_parallel_is_initialized", lambda: False, raising=False)
    # Additional mocks for layer/TE code that may call with check_initialized=False
    monkeypatch.setattr(parallel_state, "get_embedding_group", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_position_embedding_group", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(parallel_state, "get_amax_reduction_group", lambda *args, **kwargs: None, raising=False)


def _minimal_flux_provider(
    num_joint_layers=1,
    num_single_layers=1,
    hidden_size=64,
    in_channels=16,
    context_dim=64,
    vec_in_dim=32,
    guidance_embed=False,
    **kwargs,
):
    """FluxProvider with minimal sizes for fast unit tests."""
    return FluxProvider(
        num_layers=1,
        num_joint_layers=num_joint_layers,
        num_single_layers=num_single_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=hidden_size * 4,
        num_attention_heads=4,
        kv_channels=hidden_size // 4,
        num_query_groups=4,
        in_channels=in_channels,
        context_dim=context_dim,
        model_channels=32,
        vec_in_dim=vec_in_dim,
        patch_size=2,
        guidance_embed=guidance_embed,
        axes_dims_rope=[8, 4, 4],
        **kwargs,
    )


class TestFluxInit:
    """Test Flux model initialization."""

    def test_flux_init_from_provider(self, monkeypatch):
        """Test Flux can be built from FluxProvider with minimal config."""
        _mock_parallel_state(monkeypatch)
        _mock_flux_layers(monkeypatch)
        provider = _minimal_flux_provider()

        model = Flux(config=provider)

        assert model.config is provider
        assert model.hidden_size == provider.hidden_size
        assert model.num_attention_heads == provider.num_attention_heads
        assert model.in_channels == provider.in_channels
        assert model.out_channels == provider.in_channels
        assert model.patch_size == provider.patch_size
        assert model.guidance_embed is provider.guidance_embed
        assert model.pre_process is True
        assert model.post_process is True
        assert len(model.double_blocks) == provider.num_joint_layers
        assert len(model.single_blocks) == provider.num_single_layers
        assert hasattr(model, "img_embed")
        assert hasattr(model, "txt_embed")
        assert hasattr(model, "timestep_embedding")
        assert hasattr(model, "vector_embedding")
        assert hasattr(model, "pos_embed")
        assert hasattr(model, "norm_out")
        assert hasattr(model, "proj_out")

    def test_flux_init_with_guidance_embed(self, monkeypatch):
        """Test Flux has guidance_embedding when guidance_embed=True."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider(guidance_embed=True)

        model = Flux(config=provider)

        assert model.guidance_embed is True
        assert hasattr(model, "guidance_embedding")

    def test_flux_init_without_guidance_embed(self, monkeypatch):
        """Test Flux has no guidance_embedding when guidance_embed=False."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider(guidance_embed=False)

        model = Flux(config=provider)

        assert model.guidance_embed is False
        assert not hasattr(model, "guidance_embedding")


class TestFluxGetFp8Context:
    """Test get_fp8_context."""

    def test_get_fp8_context_when_fp8_disabled(self, monkeypatch):
        """Test get_fp8_context returns nullcontext when fp8 is not set."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider()
        model = Flux(config=provider)
        assert getattr(provider, "fp8", None) in (None, False, "")

        ctx = model.get_fp8_context()

        assert isinstance(ctx, nullcontext)

    def test_get_fp8_context_when_fp8_false(self, monkeypatch):
        """Test get_fp8_context returns nullcontext when config.fp8 is False."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider()
        provider.fp8 = False
        model = Flux(config=provider)

        ctx = model.get_fp8_context()

        assert isinstance(ctx, nullcontext)


class TestFluxSetInputTensor:
    """Test set_input_tensor (pipeline parallelism hook)."""

    def test_set_input_tensor_no_op(self, monkeypatch):
        """Test set_input_tensor is a no-op and does not raise."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider()
        model = Flux(config=provider)

        model.set_input_tensor(torch.randn(2, 3, 4))
        model.set_input_tensor(None)


class TestFluxForward:
    """Test Flux forward pass."""

    def test_forward_output_shape(self, monkeypatch):
        """Test forward returns correct shape [S, B, out_channels] (sequence-first)."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider()
        model = Flux(config=provider)
        batch_size = 2
        txt_seq_len = 4
        img_seq_len = 8
        # Sequence-first as used by Megatron adapter
        img = torch.randn(img_seq_len, batch_size, provider.in_channels)
        txt = torch.randn(txt_seq_len, batch_size, provider.context_dim)
        y = torch.randn(batch_size, provider.vec_in_dim)
        timesteps = torch.rand(batch_size)
        img_ids = torch.zeros(batch_size, img_seq_len, 3)
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3)

        out = model.forward(
            img=img,
            txt=txt,
            y=y,
            timesteps=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
        )

        # Output is image part only, sequence-first (Flux sets out_channels = config.in_channels)
        assert out.shape == (
            img_seq_len,
            batch_size,
            provider.patch_size * provider.patch_size * provider.in_channels,
        )

    def test_forward_with_guidance(self, monkeypatch):
        """Test forward with guidance tensor when guidance_embed=True."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider(guidance_embed=True)
        model = Flux(config=provider)
        batch_size = 2
        txt_seq_len = 4
        img_seq_len = 8
        img = torch.randn(img_seq_len, batch_size, provider.in_channels)
        txt = torch.randn(txt_seq_len, batch_size, provider.context_dim)
        y = torch.randn(batch_size, provider.vec_in_dim)
        timesteps = torch.rand(batch_size)
        img_ids = torch.zeros(batch_size, img_seq_len, 3)
        txt_ids = torch.zeros(batch_size, txt_seq_len, 3)
        guidance = torch.full((batch_size,), 3.5)

        out = model.forward(
            img=img,
            txt=txt,
            y=y,
            timesteps=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
        )

        assert out.shape[0] == img_seq_len
        assert out.shape[1] == batch_size

    def test_forward_guidance_none_when_embed_disabled(self, monkeypatch):
        """Test forward accepts guidance=None when guidance_embed=False."""
        _mock_parallel_state(monkeypatch)
        provider = _minimal_flux_provider(guidance_embed=False)
        model = Flux(config=provider)
        batch_size = 2
        img = torch.randn(8, batch_size, provider.in_channels)
        txt = torch.randn(4, batch_size, provider.context_dim)
        y = torch.randn(batch_size, provider.vec_in_dim)
        timesteps = torch.rand(batch_size)
        img_ids = torch.zeros(batch_size, 8, 3)
        txt_ids = torch.zeros(batch_size, 4, 3)

        out = model.forward(
            img=img,
            txt=txt,
            y=y,
            timesteps=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=None,
        )

        assert out.dim() == 3


class TestFluxShardedStateDict:
    """Test sharded_state_dict (requires parallel_state for replica IDs)."""

    def test_sharded_state_dict_returns_dict(self, monkeypatch):
        """Test sharded_state_dict returns a dict-like structure."""
        _mock_parallel_state(monkeypatch)
        monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
        monkeypatch.setattr(parallel_state, "get_virtual_pipeline_model_parallel_rank", lambda: None, raising=False)
        monkeypatch.setattr(
            parallel_state, "get_virtual_pipeline_model_parallel_world_size", lambda: None, raising=False
        )
        monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_rank", lambda: 0, raising=False)
        monkeypatch.setattr(
            parallel_state, "get_data_parallel_rank", lambda with_context_parallel=False: 0, raising=False
        )
        provider = _minimal_flux_provider()
        model = Flux(config=provider)

        result = model.sharded_state_dict(prefix="", sharded_offsets=(), metadata=None)

        assert isinstance(result, dict)
        # Should contain keys for double_blocks, single_blocks, and other modules
        assert len(result) > 0
