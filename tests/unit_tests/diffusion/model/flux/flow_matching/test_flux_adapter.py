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

"""Unit tests for MegatronFluxAdapter."""

from unittest.mock import MagicMock

import pytest
import torch

from megatron.bridge.diffusion.common.flow_matching.adapters.base import FlowMatchingContext
from megatron.bridge.diffusion.models.flux.flow_matching.flux_adapter import MegatronFluxAdapter


pytestmark = [pytest.mark.unit]


class TestMegatronFluxAdapterInit:
    """Test MegatronFluxAdapter initialization."""

    def test_init_default(self):
        """Test initialization with default guidance_scale."""
        adapter = MegatronFluxAdapter()
        assert adapter.guidance_scale == 3.5

    def test_init_custom_guidance_scale(self):
        """Test initialization with custom guidance_scale."""
        adapter = MegatronFluxAdapter(guidance_scale=7.5)
        assert adapter.guidance_scale == 7.5


class TestMegatronFluxAdapterPackLatents:
    """Test _pack_latents."""

    def test_pack_latents_shape(self):
        """Test packed latents have correct shape [B, (H//2)*(W//2), C*4]."""
        adapter = MegatronFluxAdapter()
        batch_size, channels, height, width = 2, 16, 64, 64
        latents = torch.randn(batch_size, channels, height, width)

        packed = adapter._pack_latents(latents)

        expected_seq = (height // 2) * (width // 2)
        expected_channels = channels * 4
        assert packed.shape == (batch_size, expected_seq, expected_channels)

    def test_pack_latents_different_sizes(self):
        """Test _pack_latents with different H, W."""
        adapter = MegatronFluxAdapter()
        latents = torch.randn(1, 8, 32, 48)
        packed = adapter._pack_latents(latents)
        assert packed.shape == (1, (32 // 2) * (48 // 2), 8 * 4)


class TestMegatronFluxAdapterUnpackLatents:
    """Test _unpack_latents."""

    def test_unpack_latents_shape(self):
        """Test unpacked latents have correct shape [B, C, H, W]."""
        adapter = MegatronFluxAdapter()
        batch_size, height, width = 2, 64, 64
        num_patches = (height // 2) * (width // 2)
        channels_packed = 16 * 4
        packed = torch.randn(batch_size, num_patches, channels_packed)

        unpacked = adapter._unpack_latents(packed, height, width)

        assert unpacked.shape == (batch_size, 16, height, width)

    def test_pack_unpack_roundtrip(self):
        """Test that pack then unpack recovers original shape and values."""
        adapter = MegatronFluxAdapter()
        batch_size, channels, height, width = 2, 16, 64, 64
        original = torch.randn(batch_size, channels, height, width)

        packed = adapter._pack_latents(original)
        unpacked = adapter._unpack_latents(packed, height, width)

        assert unpacked.shape == original.shape
        assert torch.allclose(unpacked, original)


class TestMegatronFluxAdapterPrepareLatentImageIds:
    """Test _prepare_latent_image_ids."""

    def test_prepare_latent_image_ids_shape(self):
        """Test output shape [B, (H//2)*(W//2), 3] with (col0, y, x); implementation uses zeros for col0."""
        adapter = MegatronFluxAdapter()
        batch_size, height, width = 2, 64, 64
        device = torch.device("cpu")
        dtype = torch.float32

        ids = adapter._prepare_latent_image_ids(batch_size, height, width, device, dtype)

        expected_seq = (height // 2) * (width // 2)
        assert ids.shape == (batch_size, expected_seq, 3)
        assert ids.device == device
        assert ids.dtype == dtype
        # Implementation only sets columns 1 (y) and 2 (x); column 0 stays 0
        assert ids[0, 0, 0] == 0
        assert ids[1, 0, 0] == 0
        # y, x indices
        assert ids[0, 0, 1] == 0
        assert ids[0, 0, 2] == 0

    def test_prepare_latent_image_ids_device_dtype(self):
        """Test device and dtype are applied."""
        adapter = MegatronFluxAdapter()
        ids = adapter._prepare_latent_image_ids(1, 32, 32, torch.device("cpu"), torch.float64)
        assert ids.dtype == torch.float64


def _make_context(
    batch_size=2,
    height=64,
    width=64,
    channels=16,
    text_seq_len=77,
    text_dim=4096,
    device=None,
    dtype=torch.float32,
    cfg_dropout_prob=0.0,
    batch_extras=None,
):
    """Build a FlowMatchingContext for tests."""
    device = device or torch.device("cpu")
    batch = {
        "prompt_embeds": torch.randn(batch_size, text_seq_len, text_dim),
        "pooled_prompt_embeds": torch.randn(batch_size, 768),
        **(batch_extras or {}),
    }
    return FlowMatchingContext(
        noisy_latents=torch.randn(batch_size, channels, height, width),
        latents=torch.randn(batch_size, channels, height, width),
        timesteps=torch.randint(0, 1000, (batch_size,)).float(),
        sigma=torch.ones(batch_size),
        task_type="t2v",
        data_type="image",
        device=device,
        dtype=dtype,
        batch=batch,
        cfg_dropout_prob=cfg_dropout_prob,
    )


class TestMegatronFluxAdapterPrepareInputs:
    """Test prepare_inputs."""

    def test_prepare_inputs_keys_and_shapes(self):
        """Test prepare_inputs returns expected keys and sequence-first layout."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context(batch_size=2, height=64, width=64)

        inputs = adapter.prepare_inputs(ctx)

        # Sequence-first: seq dim first
        seq_len = (64 // 2) * (64 // 2)
        assert inputs["img"].shape == (seq_len, 2, 16 * 4)
        assert inputs["txt"].shape == (77, 2, 4096)
        assert inputs["y"].shape == (2, 768)
        assert inputs["timesteps"].shape == (2,)
        assert inputs["img_ids"].shape == (2, seq_len, 3)
        assert inputs["txt_ids"].shape == (2, 77, 3)
        assert inputs["_original_shape"] == (2, 16, 64, 64)
        assert (inputs["timesteps"] >= 0).all() and (inputs["timesteps"] <= 1).all()

    def test_prepare_inputs_prompt_embeds_sb_layout(self):
        """Test prepare_inputs when prompt_embeds are [S, B, D] (Megatron layout)."""
        adapter = MegatronFluxAdapter()
        batch_size, text_seq_len, text_dim = 2, 77, 4096
        # [S, B, D]
        prompt_embeds = torch.randn(text_seq_len, batch_size, text_dim)
        ctx = _make_context(
            batch_size=batch_size,
            batch_extras={"prompt_embeds": prompt_embeds},
        )

        inputs = adapter.prepare_inputs(ctx)

        # Should be transposed to [S, B, D] for model (sequence-first)
        assert inputs["txt"].shape == (text_seq_len, batch_size, text_dim)

    def test_prepare_inputs_missing_prompt_embeds_raises(self):
        """Test prepare_inputs raises when prompt_embeds not in batch."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context(batch_extras={})
        ctx.batch.pop("prompt_embeds", None)

        with pytest.raises(ValueError, match="Expected 'prompt_embeds' in batch"):
            adapter.prepare_inputs(ctx)

    def test_prepare_inputs_non_4d_latents_raises(self):
        """Test prepare_inputs raises when noisy_latents are not 4D."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context()
        ctx.noisy_latents = torch.randn(2, 16, 64)  # 3D

        with pytest.raises(ValueError, match="expects 4D latents"):
            adapter.prepare_inputs(ctx)

    def test_prepare_inputs_default_pooled_embeds(self):
        """Test prepare_inputs uses zeros when pooled_prompt_embeds missing."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context(batch_extras={})
        ctx.batch.pop("pooled_prompt_embeds", None)

        inputs = adapter.prepare_inputs(ctx)

        assert inputs["y"].shape == (2, 768)
        assert torch.all(inputs["y"] == 0)

    def test_prepare_inputs_text_ids_from_batch(self):
        """Test prepare_inputs uses text_ids from batch when present."""
        adapter = MegatronFluxAdapter()
        text_ids = torch.randn(2, 77, 3)
        ctx = _make_context(batch_extras={"text_ids": text_ids})

        inputs = adapter.prepare_inputs(ctx)

        assert "txt_ids" in inputs
        assert inputs["txt_ids"].shape == (2, 77, 3)

    def test_prepare_inputs_cfg_dropout_zero_never_drops(self):
        """Test with cfg_dropout_prob=0 text is never zeroed."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context(cfg_dropout_prob=0.0)

        inputs = adapter.prepare_inputs(ctx)

        assert not torch.all(inputs["txt"] == 0)
        assert not torch.all(inputs["y"] == 0)

    def test_prepare_inputs_cfg_dropout_one_always_drops(self):
        """Test with cfg_dropout_prob=1.0 text and pooled are zeroed."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context(cfg_dropout_prob=1.0)

        inputs = adapter.prepare_inputs(ctx)

        assert torch.all(inputs["txt"] == 0)
        assert torch.all(inputs["y"] == 0)

    def test_prepare_inputs_guidance_when_model_has_guidance_embed(self):
        """Test guidance key is set when model has guidance_embed=True."""
        adapter = MegatronFluxAdapter(guidance_scale=5.0)
        # Use a plain object for unwrapped model so hasattr(unwrapped, "module") is False
        # (MagicMock would have .module and the unwrap loop would go to None)
        class UnwrappedModel:
            guidance_embed = True

        wrapper = MagicMock()
        wrapper.module = UnwrappedModel()
        ctx = _make_context(batch_extras={"_model": wrapper})

        inputs = adapter.prepare_inputs(ctx)

        assert "guidance" in inputs
        assert inputs["guidance"].shape == (2,)
        assert (inputs["guidance"] == 5.0).all()

    def test_prepare_inputs_no_guidance_without_model(self):
        """Test guidance is not in inputs when batch has no _model."""
        adapter = MegatronFluxAdapter()
        ctx = _make_context(batch_extras={})

        inputs = adapter.prepare_inputs(ctx)

        assert "guidance" not in inputs

    def test_prepare_inputs_no_guidance_when_guidance_embed_false(self):
        """Test guidance is not in inputs when model has guidance_embed=False."""
        adapter = MegatronFluxAdapter()
        # Unwrapped model with guidance_embed=False
        class UnwrappedModel:
            guidance_embed = False

        wrapper = MagicMock()
        wrapper.module = UnwrappedModel()
        ctx = _make_context(batch_extras={"_model": wrapper})

        inputs = adapter.prepare_inputs(ctx)

        assert "guidance" not in inputs


class TestMegatronFluxAdapterForward:
    """Test forward."""

    def test_forward_returns_unpacked_shape(self):
        """Test forward returns [B, C, H, W] and uses _original_shape."""
        adapter = MegatronFluxAdapter()
        batch_size, channels, height, width = 2, 16, 64, 64
        seq_len = (height // 2) * (width // 2)
        # Simulate prepare_inputs output (without popping _original_shape)
        inputs = {
            "img": torch.randn(seq_len, batch_size, channels * 4),
            "txt": torch.randn(77, batch_size, 4096),
            "y": torch.randn(batch_size, 768),
            "timesteps": torch.rand(batch_size),
            "img_ids": torch.zeros(batch_size, seq_len, 3),
            "txt_ids": torch.zeros(batch_size, 77, 3),
            "_original_shape": (batch_size, channels, height, width),
        }
        # Model returns [S, B, D] (sequence-first)
        model = MagicMock()
        model.return_value = torch.randn(seq_len, batch_size, channels * 4)

        out = adapter.forward(model, inputs)

        assert out.shape == (batch_size, channels, height, width)
        model.assert_called_once()
        call_kw = model.call_args[1]
        assert call_kw.get("guidance") is None

    def test_forward_handles_tuple_output(self):
        """Test forward uses first element when model returns tuple."""
        adapter = MegatronFluxAdapter()
        batch_size, channels, height, width = 2, 16, 64, 64
        seq_len = (height // 2) * (width // 2)
        pred = torch.randn(seq_len, batch_size, channels * 4)
        inputs = {
            "img": torch.randn(seq_len, batch_size, channels * 4),
            "txt": torch.randn(77, batch_size, 4096),
            "y": torch.randn(batch_size, 768),
            "timesteps": torch.rand(batch_size),
            "img_ids": torch.zeros(batch_size, seq_len, 3),
            "txt_ids": torch.zeros(batch_size, 77, 3),
            "_original_shape": (batch_size, channels, height, width),
        }
        model = MagicMock()
        model.return_value = (pred, None)

        out = adapter.forward(model, inputs)

        assert out.shape == (batch_size, channels, height, width)

    def test_forward_passes_guidance_when_present(self):
        """Test forward passes guidance to model when in inputs."""
        adapter = MegatronFluxAdapter()
        batch_size, channels, height, width = 2, 16, 64, 64
        seq_len = (height // 2) * (width // 2)
        inputs = {
            "img": torch.randn(seq_len, batch_size, channels * 4),
            "txt": torch.randn(77, batch_size, 4096),
            "y": torch.randn(batch_size, 768),
            "timesteps": torch.rand(batch_size),
            "img_ids": torch.zeros(batch_size, seq_len, 3),
            "txt_ids": torch.zeros(batch_size, 77, 3),
            "guidance": torch.full((batch_size,), 3.5),
            "_original_shape": (batch_size, channels, height, width),
        }
        model = MagicMock()
        model.return_value = torch.randn(seq_len, batch_size, channels * 4)

        adapter.forward(model, inputs)

        call_kw = model.call_args[1]
        assert "guidance" in call_kw
        assert call_kw["guidance"] is not None
        assert call_kw["guidance"].shape == (batch_size,)
