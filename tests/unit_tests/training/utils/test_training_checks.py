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

"""Unit tests for the parameter-update and expert-load training checks."""

import json
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.training.utils import training_checks
from megatron.bridge.training.utils.training_checks import (
    ExpertLoadMonitor,
    ParameterUpdateCheck,
    expert_load_statistics,
    master_weight_accessors,
    merge_parameter_records,
    moe_layers,
    parameter_keys,
    tensor_fingerprint,
)


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _rank_zero(monkeypatch):
    monkeypatch.setenv("RANK", "0")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.proj = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.unused = torch.nn.Linear(4, 4)

    def forward(self, tokens):
        return self.norm(self.proj(self.embedding(tokens)))


def _context(model, optimizer, step=0):
    return SimpleNamespace(
        model=[model], optimizer=optimizer, state=SimpleNamespace(train_state=SimpleNamespace(step=step))
    )


def test_fingerprint_detects_one_changed_value_across_chunks():
    tensor = torch.arange(1000, dtype=torch.bfloat16)
    reference = tensor_fingerprint(tensor, chunk_numel=64)
    assert torch.equal(reference, tensor_fingerprint(tensor.clone(), chunk_numel=7))
    changed = tensor.clone()
    changed[999] += 8
    assert not torch.equal(reference, tensor_fingerprint(changed, chunk_numel=64))


def test_parameter_keys_strip_wrappers_and_separate_stages_chunks_and_experts():
    first, second = torch.nn.Module(), torch.nn.Module()
    first.module = torch.nn.Linear(2, 2)
    second.module = torch.nn.Linear(2, 2)
    second.module.weight.allreduce = False

    keys = [key for key, _ in parameter_keys([first, second], pp_rank=1, pp_size=2, ep_rank=3)]

    assert keys == ["pp1.vp0.weight", "pp1.vp0.bias", "pp1.vp1.ep3.weight", "pp1.vp1.bias"]
    assert [key for key, _ in parameter_keys([first])] == ["weight", "bias"]


def test_master_weight_accessors_cover_distributed_and_plain_optimizers():
    model_param = torch.nn.Parameter(torch.ones(4))
    shard = torch.zeros(2)

    class DistributedOptimizer:
        model_param_group_index_map = {model_param: (0, 0)}

        def _get_main_param_and_optimizer_states(self, param):
            assert param is model_param
            return {"param": shard, "exp_avg_sq": torch.ones(2)}

    chained = SimpleNamespace(chained_optimizers=[DistributedOptimizer()])
    assert master_weight_accessors(chained)[id(model_param)]()["param"] is shard

    plain_param = torch.nn.Parameter(torch.ones(2))
    plain = torch.optim.SGD([plain_param], lr=0.1)
    assert master_weight_accessors(plain)[id(plain_param)]()["param"] is plain_param

    master = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    inner = torch.optim.SGD([master], lr=0.1)
    float16 = SimpleNamespace(optimizer=inner, float16_groups=[[model_param]], fp32_from_float16_groups=[[master]])
    assert master_weight_accessors(float16)[id(model_param)]()["param"] is master


def test_merge_combines_ranks_with_or_and_ignores_missing_information():
    merged = merge_parameter_records(
        [
            {"w": {"trainable": True, "optimizer": False, "gradient": None, "changed": None}},
            {"w": {"trainable": True, "optimizer": True, "gradient": False, "changed": True}},
            {"w": {"trainable": True, "optimizer": True, "gradient": True, "changed": False}},
        ]
    )
    assert merged["w"] == {"trainable": True, "optimizer": True, "gradient": True, "changed": True}


def test_parameter_update_check_reports_trained_unused_and_frozen_tensors(tmp_path):
    torch.manual_seed(0)
    model = _Model()
    model.embedding.weight.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    check = ParameterUpdateCheck(tmp_path / "report.json")

    weights = torch.randn(3, 4)
    check.on_train_start(_context(model, optimizer))
    for step in range(3):
        optimizer.zero_grad()
        (model(torch.tensor([1, 2, 3])) * weights).sum().backward()
        # Megatron's gradient buffers give unused parameters a zero gradient rather than none.
        for param in model.unused.parameters():
            param.grad = torch.zeros_like(param)
        optimizer.step()
        check.on_train_step_end(_context(model, optimizer, step))
    check.on_train_end(_context(model, optimizer))

    report = json.loads((tmp_path / "report.json").read_text())
    tensors = report["tensors"]
    assert tensors["proj.weight"] == {
        "trainable": True,
        "numel": 16,
        "optimizer": True,
        "gradient": True,
        "changed": True,
    }
    assert tensors["unused.weight"]["gradient"] is False
    assert tensors["unused.weight"]["changed"] is False
    assert tensors["embedding.weight"] == {
        "trainable": False,
        "numel": 32,
        "optimizer": False,
        "gradient": None,
        "changed": False,
    }
    assert report["summary"]["trainable_without_gradient"] == 2
    assert report["summary"]["frozen_tensors"] == 1


def test_parameter_update_check_flags_a_frozen_tensor_that_changed_and_a_trainable_one_outside_the_optimizer(
    tmp_path,
):
    model = _Model()
    model.embedding.weight.requires_grad_(False)
    optimizer = torch.optim.SGD(model.proj.parameters(), lr=0.1)
    check = ParameterUpdateCheck(tmp_path / "report.json")

    check.on_train_start(_context(model, optimizer))
    check.on_train_step_end(_context(model, optimizer))
    with torch.no_grad():
        model.embedding.weight.add_(1.0)
    check.on_train_end(_context(model, optimizer))

    tensors = json.loads((tmp_path / "report.json").read_text())["tensors"]
    assert tensors["embedding.weight"]["changed"] is True
    assert tensors["norm.weight"]["optimizer"] is False


def test_master_weights_created_after_the_first_update_are_recorded_later(tmp_path):
    model_param = torch.nn.Parameter(torch.ones(4))
    master = torch.ones(4)
    states = {}

    class LazyOptimizer:
        model_param_group_index_map = {model_param: (0, 0)}

        def _get_main_param_and_optimizer_states(self, param):
            return {"param": states["master_param"], "exp_avg_sq": torch.ones(4)}

    module = torch.nn.Module()
    module.weight = model_param
    check = ParameterUpdateCheck(tmp_path / "report.json")
    check.on_train_start(_context(module, LazyOptimizer()))
    check.on_train_step_end(_context(module, LazyOptimizer()))
    states["master_param"] = master
    check.on_train_step_end(_context(module, LazyOptimizer()))
    master.add_(1.0)
    check.on_train_end(_context(module, LazyOptimizer()))

    assert json.loads((tmp_path / "report.json").read_text())["tensors"]["weight"]["changed"] is True


def test_expert_load_statistics_reports_ratios_and_idle_experts():
    statistics = expert_load_statistics(torch.tensor([[4, 2, 2, 0], [0, 0, 0, 0]]))
    assert statistics["tokens"] == [8, 0]
    assert statistics["max_load"] == [2.0, None]
    assert statistics["min_load"] == [0.0, None]
    assert statistics["idle"] == [1, 4]


class _Experts(torch.nn.Module):
    def forward(self, hidden, tokens_per_expert, probs):
        return hidden, None


class _MoELayer(torch.nn.Module):
    def __init__(self, local_expert_indices, num_experts=4):
        super().__init__()
        self.experts = _Experts()
        self.local_expert_indices = local_expert_indices
        self.config = SimpleNamespace(num_moe_experts=num_experts)


def test_moe_layers_finds_layers_by_their_expert_attributes():
    model = torch.nn.Module()
    model.layer = _MoELayer([0, 1, 2, 3])
    model.dense = torch.nn.Linear(2, 2)
    assert [(chunk, name) for chunk, name, _ in moe_layers([model])] == [(0, "layer")]


def test_expert_load_monitor_counts_tokens_per_global_expert(tmp_path, capsys):
    model = torch.nn.Module()
    model.first = _MoELayer([2, 3])
    model.second = _MoELayer([0, 1])
    monitor = ExpertLoadMonitor(tmp_path / "experts.json")
    monitor.on_train_start(_context(model, None))

    hidden = torch.zeros(1)
    for _ in range(2):  # two microbatches
        model.first.experts(hidden, torch.tensor([3, 1]), None)
        model.second.experts(hidden, [2, 2], None)
    monitor.on_train_step_end(_context(model, None, step=49))
    model.first.experts(hidden, torch.tensor([0, 4]), None)
    model.second.experts(hidden, torch.tensor([1, 3]), None)
    monitor.on_train_step_end(_context(model, None, step=50))
    monitor.on_train_end(_context(model, None))

    report = json.loads((tmp_path / "experts.json").read_text())
    assert report["layers"] == ["first", "second"]
    first_step, second_step = report["steps"]
    assert first_step["iteration"] == 50
    # Layer "first" holds experts 2 and 3 (6 and 2 tokens); layer "second" holds experts 0 and 1 (4 and 4).
    assert first_step["tokens"] == [8, 8]
    assert first_step["max_load"] == [3.0, 2.0]
    assert first_step["idle"] == [2, 2]
    assert second_step["max_load"] == [4.0, 3.0]
    assert second_step["idle"] == [3, 2]
    assert report["total_tokens"] == [[0, 0, 6, 6], [5, 7, 0, 0]]
    assert report["experts_without_tokens_in_run"] == 4
    assert "expert load | iteration       50 | worst max/mean load: 3.000 (first)" in capsys.readouterr().out
    # Hooks are removed at the end of training.
    model.first.experts(hidden, torch.tensor([5, 5]), None)
    assert monitor._counts.sum() == 0 and monitor._host_counts.sum() == 0


def test_expert_load_monitor_rejects_counts_for_the_wrong_number_of_experts(tmp_path):
    model = torch.nn.Module()
    model.layer = _MoELayer([0, 1])
    monitor = ExpertLoadMonitor(tmp_path / "experts.json")
    monitor.on_train_start(_context(model, None))
    with pytest.raises(ValueError, match="counts for 3 experts but hold 2"):
        model.layer.experts(torch.zeros(1), torch.tensor([1, 1, 1]), None)


def test_expert_load_monitor_does_nothing_for_a_dense_model(tmp_path):
    monitor = ExpertLoadMonitor(tmp_path / "experts.json")
    monitor.on_train_start(_context(torch.nn.Linear(2, 2), None))
    monitor.on_train_step_end(_context(torch.nn.Linear(2, 2), None))
    monitor.on_train_end(_context(torch.nn.Linear(2, 2), None))
    assert not (tmp_path / "experts.json").exists()
    assert training_checks.EXPERT_LOAD_REPORT == "expert_load.json"
