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

import torch

from megatron.bridge.training.nemotron_omni_step import get_batch_from_iterator
from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


def test_packed_batch_preserves_mamba_sequence_boundaries(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, **kwargs: self)
    batch = {
        "input_ids": torch.tensor([[1, 2, 0, 0, 3, 4, 5, 0]]),
        "labels": torch.tensor([[2, -100, -100, -100, 4, 5, -100, -100]]),
        "loss_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]),
        "position_ids": torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
        "attention_mask": None,
        "cu_seqlens_q": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seqlens_kv": torch.tensor([0, 2, 5], dtype=torch.int32),
        "cu_seqlens_q_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "cu_seqlens_kv_padded": torch.tensor([0, 4, 8], dtype=torch.int32),
        "max_seqlen_q": torch.tensor(4, dtype=torch.int32),
        "max_seqlen_kv": torch.tensor(4, dtype=torch.int32),
        "total_tokens": 8,
    }

    moved = get_batch_from_iterator(
        iter([batch]),
        is_first_pp_stage=True,
        is_last_pp_stage=True,
    )
    metadata = {
        key: value
        for key, value in moved.items()
        if key.startswith("cu_seqlens") or key.startswith("max_seqlen") or key == "total_tokens"
    }
    packed_seq_params = get_packed_seq_params(metadata)

    assert moved["total_tokens"] == 8
    assert packed_seq_params.seq_idx.tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
