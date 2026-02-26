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

from megatron.bridge.diffusion.data.wan.wan_taskencoder import WanTaskEncoder, cook, parallel_state


def test_cook_extracts_expected_fields():
    sample = {
        "__key__": "k",
        "__restore_key__": "rk",
        "__subflavors__": [],
        "json": {"meta": 1},
        "pth": torch.randn(1, 2, 2, 2),
        "pickle": torch.randn(3, 4),
        "unused": 123,
    }
    out = cook(sample)
    assert "json" in out and out["json"] is sample["json"]
    assert "pth" in out and torch.equal(out["pth"], sample["pth"])
    assert "pickle" in out and torch.equal(out["pickle"], sample["pickle"])
    # ensure basic keys from the sample are preserved by cook via basic_sample_keys()
    assert out["__key__"] == sample["__key__"]
    assert out["__restore_key__"] == sample["__restore_key__"]
    assert out["__subflavors__"] == sample["__subflavors__"]


def test_encode_sample_no_context_parallel(monkeypatch):
    # Ensure CP world size is 1 to avoid extra padding branch
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    # Ensure seeded wrapper has an active worker config
    from megatron.energon.task_encoder.base import WorkerConfig

    class _FakeWorkerCfg:
        def worker_seed(self):
            return 123

        active_worker_sample_index = 0

    monkeypatch.setattr(WorkerConfig, "active_worker_config", _FakeWorkerCfg(), raising=False)

    # Construct a minimal, consistent sample
    c = 8
    F_latents, H_latents, W_latents = 4, 8, 6
    patch_temporal, patch_spatial = 1, 2
    # video latent before patchify has shape [c, F_latents, H_latents, W_latents]
    # where grid sizes (patch counts) are (F_latents // pF, H_latents // pH, W_latents // pW)
    video_latent = torch.randn(c, F_latents, H_latents, W_latents)
    context_len, context_dim = 256, 64
    context_embeddings = torch.randn(context_len, context_dim)
    sample = {
        "__key__": "k",
        "__restore_key__": "rk",
        "__subflavors__": [],
        "json": {"meta": 1},
        "pth": video_latent,
        "pickle": context_embeddings,
    }

    enc = WanTaskEncoder(
        seq_length=1024, patch_temporal=patch_temporal, patch_spatial=patch_spatial, packing_buffer_size=None
    )
    out = enc.encode_sample(sample)

    # Grid / patches
    F_patches = F_latents // patch_temporal
    H_patches = H_latents // patch_spatial
    W_patches = W_latents // patch_spatial
    num_patches = F_patches * H_patches * W_patches
    patch_vec_dim = c * patch_temporal * patch_spatial * patch_spatial

    assert out.video.shape == (num_patches, patch_vec_dim)
    assert out.latent_shape.dtype == torch.int32
    assert torch.equal(out.latent_shape, torch.tensor([F_patches, H_patches, W_patches], dtype=torch.int32))

    # Loss mask and seq lengths
    assert out.loss_mask.dtype == torch.bfloat16
    assert out.loss_mask.shape[0] == num_patches
    assert torch.equal(out.seq_len_q, torch.tensor([num_patches], dtype=torch.int32))
    # context embeddings are padded to fixed 512 inside encode_sample
    assert torch.equal(out.seq_len_kv, torch.tensor([512], dtype=torch.int32))
    assert torch.equal(out.seq_len_q_padded, out.seq_len_q)
    assert torch.equal(out.seq_len_kv_padded, out.seq_len_kv)

    # Metadata passthrough
    assert out.video_metadata == sample["json"]
    assert out.__key__ == sample["__key__"]
    assert out.__restore_key__ == sample["__restore_key__"]
    assert out.__subflavors__ == sample["__subflavors__"]


def test_batch_with_packing_buffer_size(monkeypatch):
    # Force CP world size 1
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    # Ensure seeded wrapper has an active worker config
    from megatron.energon.task_encoder.base import WorkerConfig

    class _FakeWorkerCfg:
        def worker_seed(self):
            return 456

        active_worker_sample_index = 0

    monkeypatch.setattr(WorkerConfig, "active_worker_config", _FakeWorkerCfg(), raising=False)

    c = 4
    F_latents, H_latents, W_latents = 2, 4, 4
    patch_temporal, patch_spatial = 1, 2
    video_latent = torch.randn(c, F_latents * patch_temporal, H_latents * patch_spatial, W_latents * patch_spatial)
    sample = {
        "__key__": "k",
        "__restore_key__": "rk",
        "__subflavors__": [],
        "json": {"meta": 1},
        "pth": video_latent,
        "pickle": torch.randn(32, 128),
    }

    enc = WanTaskEncoder(
        seq_length=256, patch_temporal=patch_temporal, patch_spatial=patch_spatial, packing_buffer_size=3
    )
    diff_sample = enc.encode_sample(sample)
    batch = enc.batch([diff_sample])

    assert isinstance(batch, dict)
    for k in [
        "video_latents",
        "context_embeddings",
        "loss_mask",
        "seq_len_q",
        "seq_len_q_padded",
        "seq_len_kv",
        "seq_len_kv_padded",
        "grid_sizes",
        "video_metadata",
    ]:
        assert k in batch

    # video_latents: [S, 1, ...], where S equals sample.video length when CP world size is 1
    assert batch["video_latents"].shape[1] == 1
    assert batch["context_embeddings"].shape[1] == 1
    assert batch["loss_mask"].shape[1] == 1
