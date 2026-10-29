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

from megatron.bridge.diffusion.data.flux.flux_taskencoder import FluxTaskEncoder, cook, parallel_state


def test_cook_extracts_expected_fields():
    sample = {
        "__key__": "k",
        "__restore_key__": "rk",
        "__subflavors__": [],
        "json": {"meta": 1, "resolution": "1024x1024"},
        "pth": torch.randn(16, 128, 128),  # [C, H, W] image latents
        "pickle": {
            "prompt_embeds": torch.randn(512, 4096),  # T5 embeddings
            "pooled_prompt_embeds": torch.randn(768),  # CLIP pooled
        },
        "unused": 123,
    }
    out = cook(sample)
    assert "json" in out and out["json"] is sample["json"]
    assert "pth" in out and torch.equal(out["pth"], sample["pth"])
    assert "pickle" in out and out["pickle"] is sample["pickle"]
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
    C = 16  # latent channels
    H_latents, W_latents = 128, 128
    vae_scale_factor = 8

    # Image latent shape: [C, H, W]
    image_latent = torch.randn(C, H_latents, W_latents)

    # Text embeddings
    text_seq_len, context_dim = 256, 4096
    prompt_embeds = torch.randn(text_seq_len, context_dim)
    pooled_prompt_embeds = torch.randn(768)

    text_embeddings = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }

    sample = {
        "__key__": "k",
        "__restore_key__": "rk",
        "__subflavors__": [],
        "json": {"meta": 1},
        "pth": image_latent,
        "pickle": text_embeddings,
    }

    enc = FluxTaskEncoder(
        seq_length=1024,
        vae_scale_factor=vae_scale_factor,
        latent_channels=C,
        packing_buffer_size=None,
    )
    out = enc.encode_sample(sample)

    # Check latent shape storage
    assert out.latent_shape.dtype == torch.int32
    assert torch.equal(out.latent_shape, torch.tensor([H_latents, W_latents], dtype=torch.int32))

    # Check video (image latent) shape - should be unpacked [C, H, W]
    assert out.video.shape == (C, H_latents, W_latents)

    # Loss mask and seq lengths
    # For FLUX, seq_len_q is (H/2)*(W/2) after packing
    seq_len_q = (H_latents // 2) * (W_latents // 2)
    assert out.loss_mask.dtype == torch.bfloat16
    assert out.loss_mask.shape[0] == seq_len_q
    assert torch.equal(out.seq_len_q, torch.tensor([seq_len_q], dtype=torch.int32))

    # Context embeddings are padded to fixed 512 inside encode_sample
    assert torch.equal(out.seq_len_kv, torch.tensor([512], dtype=torch.int32))
    assert torch.equal(out.seq_len_q_padded, out.seq_len_q)
    assert torch.equal(out.seq_len_kv_padded, out.seq_len_kv)

    # Check context embeddings shape
    assert out.context_embeddings.shape[0] == 512  # padded length

    # Metadata passthrough
    assert isinstance(out.video_metadata, dict)
    assert "pooled_prompt_embeds" in out.video_metadata
    assert "text_ids" in out.video_metadata
    assert out.__key__ == sample["__key__"]
    assert out.__restore_key__ == sample["__restore_key__"]
    assert out.__subflavors__ == sample["__subflavors__"]


def test_encode_sample_with_context_parallel(monkeypatch):
    """Test encoding with context parallelism enabled to check padding."""
    # Set CP world size to 2 to trigger padding logic
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 2, raising=False)
    from megatron.energon.task_encoder.base import WorkerConfig

    class _FakeWorkerCfg:
        def worker_seed(self):
            return 456

        active_worker_sample_index = 0

    monkeypatch.setattr(WorkerConfig, "active_worker_config", _FakeWorkerCfg(), raising=False)

    C = 16
    H_latents, W_latents = 64, 64  # Smaller size

    image_latent = torch.randn(C, H_latents, W_latents)
    text_embeddings = {
        "prompt_embeds": torch.randn(200, 4096),
        "pooled_prompt_embeds": torch.randn(768),
    }

    sample = {
        "__key__": "test",
        "__restore_key__": "test_restore",
        "__subflavors__": [],
        "json": {},
        "pth": image_latent,
        "pickle": text_embeddings,
    }

    enc = FluxTaskEncoder(seq_length=2048, packing_buffer_size=None)
    out = enc.encode_sample(sample)

    # With CP world size 2, sharding factor is 2*2=4
    # seq_len_q_padded should be divisible by 4
    seq_len_q = (H_latents // 2) * (W_latents // 2)
    assert out.seq_len_q_padded.item() % 4 == 0
    assert out.seq_len_q_padded.item() >= seq_len_q

    # seq_len_kv_padded should also be divisible by 4
    assert out.seq_len_kv_padded.item() % 4 == 0
    assert out.seq_len_kv_padded.item() >= 512  # original padded length


def test_batch_without_packing(monkeypatch):
    """Test batching multiple samples without packing."""
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    from megatron.energon.task_encoder.base import WorkerConfig

    class _FakeWorkerCfg:
        def worker_seed(self):
            return 789

        active_worker_sample_index = 0

    monkeypatch.setattr(WorkerConfig, "active_worker_config", _FakeWorkerCfg(), raising=False)

    C = 16
    H, W = 64, 64

    # Create multiple samples
    samples_data = []
    for i in range(2):
        image_latent = torch.randn(C, H, W)
        text_embeddings = {
            "prompt_embeds": torch.randn(256, 4096),
            "pooled_prompt_embeds": torch.randn(768),
        }
        sample = {
            "__key__": f"sample_{i}",
            "__restore_key__": f"restore_{i}",
            "__subflavors__": [],
            "json": {"id": i},
            "pth": image_latent,
            "pickle": text_embeddings,
        }
        samples_data.append(sample)

    enc = FluxTaskEncoder(seq_length=2048, packing_buffer_size=None)
    encoded_samples = [enc.encode_sample(s) for s in samples_data]
    batch = enc.batch(encoded_samples)

    assert isinstance(batch, dict)
    expected_keys = [
        "latents",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "text_ids",
        "loss_mask",
        "seq_len_q",
        "seq_len_q_padded",
        "seq_len_kv",
        "seq_len_kv_padded",
        "latent_shape",
        "image_metadata",
    ]
    for k in expected_keys:
        assert k in batch

    # Check batch dimensions
    assert batch["latents"].shape[0] == 2  # batch size
    assert batch["latents"].shape[1:] == (C, H, W)
    assert batch["prompt_embeds"].shape[0] == 2
    assert batch["pooled_prompt_embeds"].shape[0] == 2
    assert batch["text_ids"].shape[0] == 2
    assert batch["loss_mask"].shape[0] == 2


def test_batch_with_packing_buffer_size(monkeypatch):
    """Test batching with packing buffer size."""
    # Force CP world size 1
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    from megatron.energon.task_encoder.base import WorkerConfig

    class _FakeWorkerCfg:
        def worker_seed(self):
            return 999

        active_worker_sample_index = 0

    monkeypatch.setattr(WorkerConfig, "active_worker_config", _FakeWorkerCfg(), raising=False)

    C = 16
    H, W = 64, 64

    image_latent = torch.randn(C, H, W)
    text_embeddings = {
        "prompt_embeds": torch.randn(300, 4096),
        "pooled_prompt_embeds": torch.randn(768),
    }

    sample = {
        "__key__": "packed",
        "__restore_key__": "packed_restore",
        "__subflavors__": [],
        "json": {"meta": "data"},
        "pth": image_latent,
        "pickle": text_embeddings,
    }

    enc = FluxTaskEncoder(
        seq_length=2048,
        vae_scale_factor=8,
        packing_buffer_size=3,
    )
    diff_sample = enc.encode_sample(sample)
    batch = enc.batch([diff_sample])

    assert isinstance(batch, dict)
    for k in [
        "latents",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "text_ids",
        "loss_mask",
        "seq_len_q",
        "seq_len_q_padded",
        "seq_len_kv",
        "seq_len_kv_padded",
        "latent_shape",
        "image_metadata",
    ]:
        assert k in batch

    # With packing, batch size is 1 but has batch dimension [1, ...]
    assert batch["latents"].shape[0] == 1
    assert batch["latents"].shape[1:] == (C, H, W)
    assert batch["prompt_embeds"].shape[0] == 1
    assert batch["pooled_prompt_embeds"].shape[0] == 1
    assert batch["text_ids"].shape[0] == 1
    if batch["loss_mask"] is not None:
        assert batch["loss_mask"].shape[0] == 1


def test_encode_sample_with_alternative_text_format(monkeypatch):
    """Test encoding with alternative text embedding keys (t5_embeds, clip_embeds)."""
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1, raising=False)
    from megatron.energon.task_encoder.base import WorkerConfig

    class _FakeWorkerCfg:
        def worker_seed(self):
            return 111

        active_worker_sample_index = 0

    monkeypatch.setattr(WorkerConfig, "active_worker_config", _FakeWorkerCfg(), raising=False)

    C = 16
    H, W = 128, 128

    image_latent = torch.randn(C, H, W)
    # Use alternative keys
    text_embeddings = {
        "t5_embeds": torch.randn(400, 4096),
        "clip_embeds": torch.randn(768),
    }

    sample = {
        "__key__": "alt_format",
        "__restore_key__": "alt_restore",
        "__subflavors__": [],
        "json": {},
        "pth": image_latent,
        "pickle": text_embeddings,
    }

    enc = FluxTaskEncoder(seq_length=1024, packing_buffer_size=None)
    out = enc.encode_sample(sample)

    # Should successfully encode even with alternative keys
    assert out.video.shape == (C, H, W)
    assert out.context_embeddings.shape[0] == 512  # padded
    assert "pooled_prompt_embeds" in out.video_metadata
    assert "text_ids" in out.video_metadata
