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

from megatron.bridge.engine.refit import (
    RefitEntry,
    RefitPlan,
    RolloutLayout,
    _assign_buckets,
    _reshard_entries,
)


BF16 = torch.bfloat16


def _report(name, local_shape, tp_dim, count, expert, index, rank):
    return (name, local_shape, BF16, tp_dim, count, expert, index, rank, f"megatron.{name}")


def _plan(layout: RolloutLayout) -> RefitPlan:
    """Trainer TP2 x DP2 on ranks 0-3 (TP innermost) with EP4 experts, plus one collective tensor."""
    reports = [[] for _ in range(4)]
    for rank in range(4):
        reports[rank].append(_report("mlp.gate_proj.weight", (8, 16), 0, 2, None, rank % 2, rank))
        reports[rank].append(_report("experts.%d.down_proj.weight" % rank, (16, 4), 1, 1, rank, 0, rank))
    collective = RefitEntry("embed.weight", (32, 16), BF16, ((0, 1, 2, 3),), None, None, None, reshard=False)
    entries = _assign_buckets(_reshard_entries(reports) + [collective], bucket_bytes=1 << 30)
    return RefitPlan(4, layout, num_experts=4, entries=entries)


def test_plan_merges_shards_into_replica_rows_and_targets_the_rollout_layout():
    plan = _plan(RolloutLayout(world_size=4, tensor_parallel_size=2, expert_parallel_size=4))
    dense, expert0 = plan.entries[0], plan.entries[1]
    # Two TP shards per replica, two replicas: the global shape is restored along the TP dim.
    assert dense.shape == (16, 16) and dense.sources == ((0, 1), (2, 3)) and dense.split_dim == 0
    assert expert0.sources == ((0,),) and expert0.split_dim is None and expert0.expert == 0
    # The rollout rebuilds the plan from JSON; it must issue exactly what the trainer planned.
    assert RefitPlan.decode(plan.encode()) == plan
    # Reshard and collective entries never share a bucket.
    assert [entry.bucket for entry in plan.entries] == [0, 0, 0, 0, 0, 1]

    atoms = plan.atoms()
    by_name = {}
    for atom in atoms[0] + atoms[1]:
        by_name.setdefault(atom.entry.name, []).append(atom)
    # Rollout ranks are 4..7: TP2 splits the dense weight along the trainer's TP dim over [4,5] and [6,7].
    (dense_atom,) = by_name["mlp.gate_proj.weight"]
    assert dense_atom.dst == ((4, 5), (6, 7)) and dense_atom.dst_dim == 0
    # EP4 places expert e whole on rollout rank e.
    assert [(a.dst, a.dst_dim) for a in by_name["experts.3.down_proj.weight"]] == [(((7,),), None)]
    # Collective tensors go whole to every rollout rank.
    (embed_atom,) = by_name["embed.weight"]
    assert embed_atom.dst == ((4, 5, 6, 7),) and embed_atom.dst_dim is None


def test_replicated_expert_targets_rotate_over_source_replicas():
    # Rollout EP2 over 4 ranks: expert 3 lives on rollout ranks 1 and 3, one atom per contiguous block.
    plan = _plan(RolloutLayout(world_size=4, expert_parallel_size=2))
    atoms = [atom for bucket in plan.atoms() for atom in bucket if atom.entry.name == "experts.3.down_proj.weight"]
    assert [atom.dst for atom in atoms] == [((5,),), ((7,),)]
    dense = [atom for bucket in plan.atoms() for atom in bucket if atom.entry.name == "mlp.gate_proj.weight"]
    assert [atom.src for atom in dense] == [((0, 1),)] and dense[0].dst == ((4, 5, 6, 7),)
