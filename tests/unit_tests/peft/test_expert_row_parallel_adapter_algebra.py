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

"""Closed-form check of the expert ROW-parallel adapter (experts.linear_fc2) at ETP > 1.

The adapter must compute what a merged `B @ A` folded into the base weight would, on the
forward pass and on both parameter gradients. (Both arms model the unscaled product; the
`alpha/dim` scaling applied by the shipped classes is elementwise and exercised by
test_expert_lora_etp.py.) This exercises the arithmetic the fix relies on -- per-rank
`A_r @ h_r` summed across the expert-tensor-parallel group, then a zero-embedded `B_r @ z`
that the dispatcher's cross-ETP sum reassembles exactly once -- without needing GPUs or a
process group. These are closed-form models, not the shipped class: the production wiring is
covered by test_expert_lora_etp.py, and the real collectives by
test_expert_lora_etp_distributed.py.

The token count is deliberately NOT a multiple of ETP so the algebra cannot depend on
divisibility. The pad/unpad path itself is not modeled here; its gradient flow is covered in
test_expert_lora_etp.py.
"""

import torch


T, IN, DIM, OUT, ETP = 7, 12, 4, 8, 4
IN_SHARD, OUT_SHARD = IN // ETP, OUT // ETP
TOL = 1e-10


def _inputs():
    x = torch.randn(T, IN, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    g = torch.randn(T, OUT, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    return x, g


def _params():
    a = torch.randn(DIM, IN, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    b = torch.randn(OUT, DIM, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    return a.requires_grad_(True), b.requires_grad_(True)


def _run(build):
    x, g = _inputs()
    a, b = _params()
    build(x, a, b).backward(g)
    return a.grad.clone(), b.grad.clone()


def _merged(x, a, b):
    """The reference: one dense matmul against the merged weight."""
    return x @ (b @ a).t()


def _fixed(x, a, b):
    """Per-rank shards, `A @ h` summed across ETP, `B_r @ z` zero-embedded, dispatcher sums."""
    z = sum(x[:, r * IN_SHARD : (r + 1) * IN_SHARD] @ a[:, r * IN_SHARD : (r + 1) * IN_SHARD].t() for r in range(ETP))
    parts = []
    for r in range(ETP):
        shard = b[r * OUT_SHARD : (r + 1) * OUT_SHARD, :] @ z.t()
        left, right = r * OUT_SHARD, (ETP - 1) * OUT_SHARD - r * OUT_SHARD
        parts.append(torch.nn.functional.pad(shard, (0, 0, left, right)).t())
    return sum(parts)


def _unfixed(x, a, b):
    """`A @ h` left as a per-rank partial, and a gathered delta the dispatcher counts ETP times."""
    z = [x[:, r * IN_SHARD : (r + 1) * IN_SHARD] @ a[:, r * IN_SHARD : (r + 1) * IN_SHARD].t() for r in range(ETP)]
    gathered = torch.cat([b[r * OUT_SHARD : (r + 1) * OUT_SHARD, :] @ z[r].t() for r in range(ETP)], dim=0).t()
    return ETP * gathered


def test_forward_matches_merged_weight():
    x, _ = _inputs()
    a, b = _params()
    assert torch.allclose(_fixed(x, a, b), _merged(x, a, b), atol=TOL)


def test_parameter_gradients_match_merged_weight():
    ref_da, ref_db = _run(_merged)
    got_da, got_db = _run(_fixed)
    assert torch.allclose(got_da, ref_da, atol=TOL), (got_da - ref_da).abs().max()
    assert torch.allclose(got_db, ref_db, atol=TOL), (got_db - ref_db).abs().max()


def test_the_unfixed_arithmetic_is_detected():
    """Negative control. Without it the assertions above cannot be shown to discriminate.

    ``_unfixed`` is the arithmetic ``ParallelLinearAdapter(is_expert=True)`` ships for
    experts.linear_fc2 at ETP > 1: the suppressed row-parallel reduce leaves ``A_r @ h_r``
    a per-rank partial, and the gathered full-width delta is summed once per rank by the
    dispatcher's expert-TP reduce-scatter. The same full-delta-vs-partial-term error (modulo autograd
    severance) is ``GroupedExpertLinearAdapter``'s fc2 path.
    """
    ref_da, ref_db = _run(_merged)
    bad_da, bad_db = _run(_unfixed)
    assert (bad_da - ref_da).abs().max() / ref_da.abs().max() > 0.1
    assert (bad_db - ref_db).abs().max() / ref_db.abs().max() > 0.1
