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
