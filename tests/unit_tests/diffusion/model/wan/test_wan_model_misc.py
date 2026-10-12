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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import megatron.bridge.diffusion.models.wan.wan_model as wan_model_module
from megatron.bridge.diffusion.models.wan.wan_model import WanModel, sinusoidal_embedding_1d


def test_sinusoidal_embedding_1d_shape_and_dtype():
    dim = 16
    pos = torch.arange(10, dtype=torch.float32)
    emb = sinusoidal_embedding_1d(dim, pos)
    assert emb.shape == (pos.shape[0], dim)
    assert emb.dtype == torch.float32


@pytest.mark.unit
@pytest.mark.parametrize("pre_process", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_wan_decoder_receives_local_text_tokens(monkeypatch, pre_process, sequence_parallel, tp_rank):
    """Each TP rank must supply its own text shard to the KV projection's gather."""
    model = WanModel.__new__(WanModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=4, sequence_parallel=sequence_parallel)
    model.pre_process = pre_process
    model.post_process = False
    model.out_channels = 1
    model.patch_size = (1, 1, 1)
    model.num_heads = 1
    model.patch_embedding = nn.Conv3d(1, 4, kernel_size=1)
    model.timesteps_proj = nn.Identity()
    model.time_embedder = nn.Identity()
    model.time_proj_act_fn = nn.Identity()
    model.time_proj = nn.Linear(4, 24)
    model.text_embedding = nn.Linear(3, 4)
    model.rope_embeddings = lambda *args: None

    class RecordingDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_tensor = torch.zeros(4 if sequence_parallel else 8, 1, 4)
            self.context = None

        def forward(self, hidden_states, *, context, **kwargs):
            self.context = context
            return hidden_states

    model.decoder = RecordingDecoder()
    scattered = []

    def scatter(tensor):
        scattered.append(tensor)
        return tensor.chunk(2, dim=0)[tp_rank].contiguous()

    monkeypatch.setattr(wan_model_module.tensor_parallel, "scatter_to_sequence_parallel_region", scatter)
    context = torch.arange(18, dtype=torch.float32).reshape(6, 1, 3)
    embedded_context = model.text_embedding(context)
    model(
        x=torch.zeros(8, 1, 1),
        grid_sizes=[(1, 2, 4)],
        t=torch.zeros(1, 4),
        context=context,
        packed_seq_params={"self_attention": SimpleNamespace(cu_seqlens_q_padded=torch.tensor([0, 8]))},
    )

    expected = embedded_context.chunk(2, dim=0)[tp_rank] if sequence_parallel else embedded_context
    torch.testing.assert_close(model.decoder.context, expected)
    if sequence_parallel:
        # Scatter after the replicated embedding, including on intermediate PP stages.
        torch.testing.assert_close(scattered[-1], embedded_context)
        assert len(scattered) == 1 + int(pre_process)
    else:
        assert not scattered
