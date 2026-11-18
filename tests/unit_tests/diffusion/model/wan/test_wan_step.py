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

import pytest
import torch

from megatron.bridge.diffusion.models.wan.wan_step import WanForwardStep, wan_data_step


class _DummyIter:
    def __init__(self, batch):
        # mimic attribute used inside wan_data_step
        self.iterable = [batch]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="wan_data_step moves tensors to CUDA")
def test_wan_data_step_builds_packed_seq_params_cuda_guarded():
    # Construct minimal batch with required seq_len fields
    # S=8, B=2
    batch = {
        "seq_len_q": torch.tensor([3, 5], dtype=torch.int32),
        "seq_len_q_padded": torch.tensor([4, 6], dtype=torch.int32),
        "seq_len_kv": torch.tensor([2, 7], dtype=torch.int32),
        "seq_len_kv_padded": torch.tensor([2, 8], dtype=torch.int32),
        # include a tensor field to exercise device transfer
        # shape: [S, B, H, D]
        "video_latents": torch.randn(8, 2, 4, 16, dtype=torch.float32),
    }
    it = iter(_DummyIter(batch).iterable)
    qkv_format = "sbhd"
    out = wan_data_step(qkv_format, it)

    assert "packed_seq_params" in out
    for k in ["self_attention", "cross_attention"]:
        assert k in out["packed_seq_params"]
        p = out["packed_seq_params"][k]
        assert hasattr(p, "cu_seqlens_q")
        assert hasattr(p, "cu_seqlens_q_padded")
        assert hasattr(p, "cu_seqlens_kv")
        assert hasattr(p, "cu_seqlens_kv_padded")
    # spot-check CUDA device after move
    assert out["video_latents"].is_cuda
    # Verify transpose from (S, B, H, D) -> (B, S, H, D)
    assert out["video_latents"].shape == (2, 8, 4, 16)


def test_wan_forward_step_loss_partial_creation():
    step = WanForwardStep()
    mask = torch.ones(4, dtype=torch.float32)
    loss_fn = step._create_loss_function(mask, check_for_nan_in_loss=False, check_for_spiky_loss=False)
    # Just validate it's callable and is a functools.partial
    import functools

    assert isinstance(loss_fn, functools.partial)
    assert callable(loss_fn)


@pytest.mark.unit
@pytest.mark.parametrize(
    "lengths,padded_lengths,expected",
    [
        ([37440], [37440], True),
        ([16, 32], [16, 32], True),
        ([16, 31], [16, 32], False),  # Final sequence has tail padding.
        ([15, 32], [16, 32], False),  # Padding between sequences.
        ([16, 32], [32, 16], False),  # Equal total length is insufficient.
        ([16], [16, 0], False),
    ],
)
def test_same_cpu_sequence_lengths(lengths, padded_lengths, expected):
    from megatron.bridge.diffusion.models.wan.wan_step import _same_cpu_sequence_lengths

    assert _same_cpu_sequence_lengths(torch.tensor(lengths), torch.tensor(padded_lengths)) is expected


@pytest.mark.unit
def test_same_cpu_sequence_lengths_does_not_compare_device_tensors(monkeypatch):
    from megatron.bridge.diffusion.models.wan.wan_step import _same_cpu_sequence_lengths

    def unexpected_equal(*args):
        pytest.fail("Device tensors must not require a synchronous equality check")

    monkeypatch.setattr(torch, "equal", unexpected_equal)
    cpu_lengths = torch.tensor([16])
    device_lengths = torch.empty(1, dtype=torch.int32, device="meta")
    assert not _same_cpu_sequence_lengths(device_lengths, device_lengths)
    assert not _same_cpu_sequence_lengths(cpu_lengths, device_lengths)
    assert not _same_cpu_sequence_lengths(device_lengths, cpu_lengths)


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="wan_data_step moves tensors to CUDA")
@pytest.mark.parametrize("q_padded", [[16, 32], [16, 48], [32, 32]])
@pytest.mark.parametrize("kv_padded", [[16, 16], [16, 32]])
@pytest.mark.parametrize("input_device", ["cpu", "cuda"])
def test_wan_data_step_preserves_padding_boundaries(q_padded, kv_padded, input_device):
    q_lengths, kv_lengths = [16, 32], [16, 16]
    batch = {
        "seq_len_q": torch.tensor(q_lengths, dtype=torch.int32, device=input_device),
        "seq_len_q_padded": torch.tensor(q_padded, dtype=torch.int32, device=input_device),
        "seq_len_kv": torch.tensor(kv_lengths, dtype=torch.int32, device=input_device),
        "seq_len_kv_padded": torch.tensor(kv_padded, dtype=torch.int32, device=input_device),
        "video_latents": torch.randn(sum(q_padded), 1, 4, device=input_device),
    }
    out = wan_data_step("thd", iter([batch]))
    self_attention = out["packed_seq_params"]["self_attention"]
    cross_attention = out["packed_seq_params"]["cross_attention"]

    def check_boundaries(actual, lengths):
        expected = torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32)
        torch.testing.assert_close(actual.cpu(), expected)
        assert actual.is_cuda

    for params in (self_attention, cross_attention):
        check_boundaries(params.cu_seqlens_q, q_lengths)
        check_boundaries(params.cu_seqlens_q_padded, q_padded)
        assert (params.cu_seqlens_q is params.cu_seqlens_q_padded) == (input_device == "cpu" and q_lengths == q_padded)
    check_boundaries(cross_attention.cu_seqlens_kv, kv_lengths)
    check_boundaries(cross_attention.cu_seqlens_kv_padded, kv_padded)
    assert (cross_attention.cu_seqlens_kv is cross_attention.cu_seqlens_kv_padded) == (
        input_device == "cpu" and kv_lengths == kv_padded
    )
    assert self_attention.cu_seqlens_q is self_attention.cu_seqlens_kv
    assert self_attention.cu_seqlens_q_padded is self_attention.cu_seqlens_kv_padded
    assert self_attention.cu_seqlens_q is cross_attention.cu_seqlens_q
    assert out["video_latents"].shape == (1, sum(q_padded), 4)
