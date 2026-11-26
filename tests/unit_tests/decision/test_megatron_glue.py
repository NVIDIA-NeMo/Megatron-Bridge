# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU tests for the Megatron glue used by decision-model post-training."""

import pytest
import torch

from megatron.bridge.decision import megatron_glue as glue


def test_canonical_name_strips_wrappers():
    assert (
        glue.canonical_name("module.module.language_model.decoder.layers.0.x") == "language_model.decoder.layers.0.x"
    )
    assert glue.canonical_name("a._checkpoint_wrapped_module.b") == "a.b"


def test_gather_vocab_parallel_rows_single_rank():
    weight = torch.randn(10, 4)
    rows = glue.gather_vocab_parallel_rows(weight, [3, 0, 9], tp_group=None)
    assert torch.equal(rows, weight[[3, 0, 9]])


def test_last_token_pool_right_padding():
    hidden = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    pooled = glue.last_token_pool(hidden, torch.tensor([5, 2]))
    assert torch.equal(pooled[0], hidden[0, 4]) and torch.equal(pooled[1], hidden[1, 1])


def test_final_hidden_states_processor_transposes_without_sp():
    proc = glue.final_hidden_states_processor(tp_group=None)
    h = torch.randn(7, 2, 4)  # [sequence, batch, hidden]
    out = proc(hidden_states=h, config=None)
    assert out.shape == (2, 7, 4) and torch.equal(out[1, 3], h[3, 1])


def test_trainable_state_roundtrip(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 2))
    model[0].weight.requires_grad_(False)
    glue.save_trainable_state(model, tmp_path, tp_rank=0, write=True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    frozen = model[0].weight.clone()
    saved_bias = model[1].bias.clone() - 1.0
    assert glue.load_trainable_state(model, tmp_path, tp_rank=0) == 3
    assert torch.allclose(model[1].bias, saved_bias)
    assert torch.equal(model[0].weight, frozen)  # frozen weights are untouched


def test_load_trainable_state_reports_missing(tmp_path):
    a = torch.nn.Linear(2, 2)
    a.weight.requires_grad_(False)
    glue.save_trainable_state(a, tmp_path, tp_rank=0, write=True)
    b = torch.nn.Linear(2, 2)
    with pytest.raises(KeyError):
        glue.load_trainable_state(b, tmp_path, tp_rank=0)


def test_replicated_head_processor_requests_split_sp_backward(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace

    calls = []
    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")

    def gather(values, *, tensor_parallel_output_grad, group):
        calls.append((tensor_parallel_output_grad, group))
        return values

    tensor_parallel.gather_from_sequence_parallel_region = gather
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel", tensor_parallel)
    group = object()
    processor = glue.final_hidden_states_processor(tp_group=group)
    values = torch.randn(7, 2, 4)
    actual = processor(hidden_states=values, config=SimpleNamespace(sequence_parallel=True))
    assert calls == [(False, group)]
    assert torch.equal(actual, values.transpose(0, 1))
