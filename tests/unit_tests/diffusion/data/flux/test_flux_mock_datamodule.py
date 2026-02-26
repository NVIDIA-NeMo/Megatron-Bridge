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

import torch

from megatron.bridge.diffusion.data.flux.flux_mock_datamodule import FluxMockDataModuleConfig


def test_flux_mock_datamodule_build_and_batch_shapes():
    cfg = FluxMockDataModuleConfig(
        path="",
        seq_length=1024,
        packing_buffer_size=None,
        micro_batch_size=2,
        global_batch_size=8,
        num_workers=0,
        # Use small shapes for a light-weight test run
        image_H=256,
        image_W=256,
        vae_channels=16,
        vae_scale_factor=8,
        prompt_seq_len=128,
        context_dim=512,
        pooled_prompt_dim=256,
        image_precached=True,
        text_precached=True,
        num_train_samples=100,
    )
    train_dl, val_dl, test_dl = cfg.build_datasets(_context=None)
    assert train_dl is val_dl and val_dl is test_dl

    batch = next(iter(train_dl))
    expected_keys = {
        "latents",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "text_ids",
        "loss_mask",
    }
    assert expected_keys.issubset(set(batch.keys()))

    # Basic sanity checks on shapes/dtypes
    batch_size = cfg.micro_batch_size
    latent_h = cfg.image_H // cfg.vae_scale_factor
    latent_w = cfg.image_W // cfg.vae_scale_factor

    # Check latents shape: [B, C, H, W]
    assert batch["latents"].shape == (batch_size, cfg.vae_channels, latent_h, latent_w)
    assert batch["latents"].dtype == torch.bfloat16

    # Check prompt_embeds shape: [B, seq_len, context_dim]
    assert batch["prompt_embeds"].shape == (batch_size, cfg.prompt_seq_len, cfg.context_dim)
    assert batch["prompt_embeds"].dtype == torch.bfloat16

    # Check pooled_prompt_embeds shape: [B, pooled_dim]
    assert batch["pooled_prompt_embeds"].shape == (batch_size, cfg.pooled_prompt_dim)
    assert batch["pooled_prompt_embeds"].dtype == torch.bfloat16

    # Check text_ids shape: [B, seq_len, 3]
    assert batch["text_ids"].shape == (batch_size, cfg.prompt_seq_len, 3)
    assert batch["text_ids"].dtype == torch.bfloat16

    # Check loss_mask shape: [B, num_patches] where num_patches = (H/2) * (W/2) for FLUX
    num_patches = latent_h * latent_w
    assert batch["loss_mask"].shape == (batch_size, num_patches)
    assert batch["loss_mask"].dtype == torch.bfloat16


def test_flux_mock_datamodule_without_precaching():
    """Test the mock datamodule with non-precached data."""
    cfg = FluxMockDataModuleConfig(
        path="",
        seq_length=1024,
        micro_batch_size=1,
        global_batch_size=4,
        num_workers=0,
        image_H=128,
        image_W=128,
        image_precached=False,
        text_precached=False,
        num_train_samples=50,
    )
    train_dl, _, _ = cfg.build_datasets(_context=None)

    batch = next(iter(train_dl))

    # When not precached, should have raw images and text
    assert "images" in batch or "latents" in batch
    assert "txt" in batch or "prompt_embeds" in batch

    # If images are not precached, they should be raw RGB
    if "images" in batch:
        assert batch["images"].shape[1] == 3  # RGB channels
        assert batch["images"].dtype == torch.bfloat16


def test_flux_mock_datamodule_different_image_sizes():
    """Test the mock datamodule with different image dimensions."""
    cfg = FluxMockDataModuleConfig(
        path="",
        seq_length=2048,
        micro_batch_size=1,
        global_batch_size=2,
        num_workers=0,
        image_H=512,
        image_W=1024,
        vae_channels=16,
        vae_scale_factor=8,
        num_train_samples=20,
    )
    train_dl, _, _ = cfg.build_datasets(_context=None)

    batch = next(iter(train_dl))

    latent_h = 512 // 8  # 64
    latent_w = 1024 // 8  # 128

    assert batch["latents"].shape == (1, 16, latent_h, latent_w)
    # Loss mask should cover all latent positions
    assert batch["loss_mask"].shape == (1, latent_h * latent_w)
