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

"""The per-stream token table shards like Megatron-Core's vocabulary-parallel embedding.

Two CPU/gloo ranks share one vocabulary: each rank owns half of the table, the all-reduce in
`VocabParallelEmbedding` reconstructs the full lookup, and the local weight gradient matches the
reference table's gradient slice. This is the path TP > 1 training takes; the single-rank tests
cover the replicated fallback.
"""

import tempfile

import pytest
import torch
import torch.multiprocessing as mp


def _worker(rank: int, store_dir: str) -> None:
    import torch.distributed as dist
    from megatron.core import parallel_state

    dist.init_process_group(backend="gloo", init_method=f"file://{store_dir}/store", rank=rank, world_size=2)
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=2, pipeline_model_parallel_size=1, context_parallel_size=1
    )

    from megatron.bridge.models.shensi.configuration_shensi import ShensiTransformerConfig
    from megatron.bridge.models.shensi.modeling_shensi import ShensiStreamEmbeddings

    vocab, hidden, streams = 16, 4, 2
    config = ShensiTransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        hc_mult=streams,
        hc_conv_kernels=[2],
        params_dtype=torch.float32,
        use_cpu_initialization=True,
        perform_initialization=False,
        sequence_parallel=False,
        deterministic_mode=False,
        init_method=torch.nn.init.normal_,
    )
    config.vocab_size = vocab
    config.actual_vocab_size = vocab
    module = ShensiStreamEmbeddings(config, vocab_size=vocab)
    assert module.sharded_embedding and not module.reduce_scatter

    full_table = torch.arange(vocab * (streams - 1) * hidden, dtype=torch.float32).view(vocab, (streams - 1) * hidden)
    per_rank = vocab // 2
    with torch.no_grad():
        module.token_embeddings.weight.copy_(full_table[rank * per_rank : (rank + 1) * per_rank])
        conv = module.convs[0]
        conv.weight.zero_()
        conv.weight[:, 0, -1] = 1.0
        conv.bias.zero_()

    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    embeds = torch.randn(4, 2, hidden, generator=torch.Generator().manual_seed(0))
    out = module(embeds, ids)
    tokens = torch.nn.functional.embedding(ids, full_table).transpose(0, 1)
    tokens = tokens.reshape(4, 2, streams - 1, hidden)
    torch.testing.assert_close(out[:, :, 0], embeds, rtol=0, atol=0)
    torch.testing.assert_close(out[:, :, 1], tokens[:, :, 0] + embeds, rtol=1e-6, atol=1e-6)

    out.float().sum().backward()
    counts = torch.zeros(vocab)
    counts.index_add_(0, ids.reshape(-1), torch.ones(ids.numel()))
    expected_grad = counts[:, None].expand(vocab, (streams - 1) * hidden)
    torch.testing.assert_close(
        module.token_embeddings.weight.grad,
        expected_grad[rank * per_rank : (rank + 1) * per_rank],
        rtol=0,
        atol=0,
    )

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_gloo_available(), reason="needs torch.distributed's gloo backend")
def test_stream_table_shards_across_tensor_parallel_ranks():
    mp.spawn(_worker, args=(tempfile.mkdtemp(),), nprocs=2, join=True)
