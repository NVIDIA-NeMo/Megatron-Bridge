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

"""Closed-form check of the expert COLUMN-parallel adapter (experts.linear_fc1) at ETP > 1.

Its forward is exact and its `dL/dB` is already correct, so of the quantities modeled here
only `dL/dA` can tell the fixed path from the broken one -- which is why a forward-only check
calls the broken version green. (`dL/dx` also discriminates, but it flows through the
dispatcher and is out of scope for this file.)

`A` is sharded over the low-rank dim and gathered to a full `z`; `B` is sharded over the output
and consumes that full `z`, so the true `dL/dz` sums over every rank. `explicit_expert_comm`
forces `allreduce_dgrad=False` on the adapter's `linear_out`, so without a compensating
backward all-reduce rank r sees only its own term. Production wiring lives in
test_expert_lora_etp.py; real collectives in test_expert_lora_etp_distributed.py.

The reference comes from autograd on the merged-weight formulation, independently of how the
sharded arms are assembled -- otherwise the fixed arm would be compared against its own
definition and could not fail.
"""

import torch


T, IN, DIM, OUT, ETP = 7, 12, 8, 16, 4
DIM_SHARD, OUT_SHARD = DIM // ETP, OUT // ETP
TOL = 1e-12


def _fixtures():
    x = torch.randn(T, IN, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    a = torch.randn(DIM, IN, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    b = torch.randn(OUT, DIM, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    g = [
        torch.randn(T, OUT_SHARD, dtype=torch.float64, generator=torch.Generator().manual_seed(10 + r))
        for r in range(ETP)
    ]
    return x, a, b, g


def _reference_grads(x, a0, b0, g):
    a, b = a0.clone().requires_grad_(True), b0.clone().requires_grad_(True)
    outs = [x @ (b[r * OUT_SHARD : (r + 1) * OUT_SHARD, :] @ a).t() for r in range(ETP)]
    torch.autograd.backward(outs, g)
    return a.grad.clone(), b.grad.clone()


def _da_from(per_rank_dz, x):
    """`dL/dA_r` is built from rank r's slice of whatever `dL/dz` that rank holds."""
    return torch.cat([per_rank_dz[r][:, r * DIM_SHARD : (r + 1) * DIM_SHARD].t() @ x for r in range(ETP)], dim=0)


def _local_dz(a0, b0, g):
    shards = [b0[r * OUT_SHARD : (r + 1) * OUT_SHARD, :] for r in range(ETP)]
    return [g[r] @ shards[r] for r in range(ETP)]


def test_dL_dA_matches_merged_weight_after_the_backward_all_reduce():
    x, a, b, g = _fixtures()
    ref_da, _ = _reference_grads(x, a, b, g)
    summed = sum(_local_dz(a, b, g))
    got = _da_from([summed] * ETP, x)
    assert (got - ref_da).abs().max() / ref_da.abs().max() < TOL


def test_dL_dB_was_already_correct():
    x, a, b, g = _fixtures()
    _, ref_db = _reference_grads(x, a, b, g)
    z = x @ a.t()
    got = torch.cat([g[r].t() @ z for r in range(ETP)], dim=0)
    assert (got - ref_db).abs().max() / ref_db.abs().max() < TOL


def test_without_the_all_reduce_dL_dA_loses_the_cross_rank_terms():
    """Negative control: the defect this file exists to catch.

    ``_local_dz`` is the fc1 backward ``ParallelLinearAdapter(is_expert=True)`` ships at
    ETP > 1 (``allreduce_dgrad`` suppressed under ``explicit_expert_comm``): a green forward
    with every cross-rank ``dL/dA`` term missing. It also subsumes
    ``GroupedExpertLinearAdapter``'s fc1 defect, whose severed autograd yields ``dL/dA = 0``
    -- a rel-err > 0.1 trivially.
    """
    x, a, b, g = _fixtures()
    ref_da, _ = _reference_grads(x, a, b, g)
    got = _da_from(_local_dz(a, b, g), x)
    assert (got - ref_da).abs().max() / ref_da.abs().max() > 0.1
