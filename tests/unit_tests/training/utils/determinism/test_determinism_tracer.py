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

"""CPU unit tests for the determinism tracer (no GPU required).

Covers the properties the HybridEP-safe rewrite relies on: staged==eager digest
equivalence, bit-sensitivity, the empty/all-zero sentinel, in-place op classification,
the deferred stage->stash->flush pipeline, and offline first-divergence detection.
"""

import json

import pytest
import torch

from megatron.bridge.training.utils.determinism import collective_trace as ct
from megatron.bridge.training.utils.determinism import op_trace
from megatron.bridge.training.utils.determinism.diff_streams import diff_rank
from megatron.bridge.training.utils.determinism.signature import (
    _EMPTY_DIGEST,
    finalize_staged,
    stage_tensor,
    tensor_signature,
)


# --------------------------------------------------------------------------- signature
@pytest.mark.parametrize(
    "t",
    [
        torch.randn(1000, dtype=torch.float32),
        torch.randn(37, dtype=torch.bfloat16),  # odd byte count -> pad path
        torch.randint(-9, 9, (64,), dtype=torch.int64),
        torch.randint(0, 256, (64,), dtype=torch.uint8),  # unsigned: hash_tensor xor has no uint kernel
        torch.tensor(3.14),  # 0-dim scalar
        torch.zeros(2, dtype=torch.float32),  # single-lane all-zero (sentinel regression)
    ],
)
def test_staged_equals_eager(t):
    """The deferred (staged + finalized) digest must equal the eager digest bit-for-bit."""
    eager = tensor_signature(t).digest
    staged = finalize_staged(stage_tensor(t))["digest"]
    assert staged == eager
    assert len(staged) == 16  # 64-bit torch.hash_tensor digest


def test_bit_sensitivity():
    """A single-element (1-ULP) change must flip the digest — the property that matters."""
    a = torch.randn(4096, dtype=torch.float32)
    b = a.clone()
    b[123] = torch.nextafter(b[123], torch.tensor(float("inf")))  # 1 ULP
    assert tensor_signature(a).digest != tensor_signature(b).digest


def test_hash_tensor_known_tradeoffs():
    """Pin torch.hash_tensor's accepted limitations so a future change is noticed.

    hash_tensor xor-reduces values upcast to their 64-bit equivalent, so by design it does
    NOT distinguish signed zero (+0.0 upcasts equal to -0.0) and is permutation-invariant
    (xor is order-independent). These are "strong screen, not a key" limitations — value
    divergence is still caught; a pure reorder or signed-zero flip is not.
    """
    assert tensor_signature(torch.zeros(8)).digest == tensor_signature(torch.full((8,), -0.0)).digest
    v = torch.randn(500)
    assert tensor_signature(v).digest == tensor_signature(v.flip(0).contiguous()).digest


def test_empty_and_sentinel():
    assert finalize_staged(stage_tensor(torch.empty(0)))["digest"] == _EMPTY_DIGEST
    # NB: torch.hash_tensor xor-reduces upcast bytes, so an all-zero tensor also hashes to
    # 0; empty (numel 0) vs all-zero (numel N) are disambiguated by shape/numel, which the
    # diff compares alongside the digest -- so they never false-match.
    assert tensor_signature(torch.zeros(2)).numel == 2
    assert stage_tensor(None) is None
    assert finalize_staged(None) is None


# --------------------------------------------------------------------------- op_trace
@pytest.mark.parametrize(
    "name,expected",
    [
        ("aten.add_.Tensor", True),
        ("aten::copy_.default", True),
        ("aten.add.out", True),
        ("aten._foreach_add_.List", True),
        ("aten.add.Tensor", False),
        ("aten.native_batch_norm.default", False),
        ("aten.mm.default", False),
    ],
)
def test_is_inplace(name, expected):
    assert op_trace._is_inplace(name) is expected


# --------------------------------------------------------------------------- deferred flow
def test_deferred_stage_stash_flush(tmp_path):
    """Staging performs no host sync; flush_pending finalizes and writes valid digests."""
    ct.enable(out_dir=str(tmp_path))
    try:
        ct.set_active(True, window=1)
        # Nothing is written until the drain: staged records sit in _S.pending.
        ct._stash_named("aten.mm.default", "aten", [stage_tensor(torch.randn(16))], [stage_tensor(torch.randn(16))])
        assert len(ct._S.pending) == 1
        stream = tmp_path / "stream_pp0_tp0_cp0_dp0.fp"
        body_before = [line for line in stream.read_text().splitlines() if '"_header"' not in line]
        assert body_before == []  # not flushed yet
        ct.set_active(False)  # drains the window
        assert ct._S.pending == []
    finally:
        ct.disable()

    records = [json.loads(line) for line in stream.read_text().splitlines() if '"_header"' not in line]
    assert len(records) == 1
    rec = records[0]
    assert rec["op"] == "aten.mm.default" and rec["group"] == "aten" and rec["window"] == 1
    # finalized signatures carry a 16-hex digest (not the raw GPU hash tensor)
    assert len(rec["output"][0]["digest"]) == 16
    assert "h_t" not in rec["output"][0]


# --------------------------------------------------------------------------- diff_streams
def _write_stream(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"_header": True, "config": {"world_size": 2}}) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")


def _rec(op, align_idx, in_digest, out_digest, group="dp"):
    def sig(d):
        return {"shape": [4], "dtype": "torch.bfloat16", "digest": d, "numel": 4}

    return {
        "seq_id": align_idx,
        "window": 1,
        "op": op,
        "group": group,
        "align_idx": align_idx,
        "scope": "layers.0",
        "caller": "x:1",
        "input": [sig(in_digest)],
        "output": [sig(out_digest)],
    }


def test_diff_no_divergence(tmp_path):
    recs = [_rec("all_reduce", 0, "i0", "o0"), _rec("reduce_scatter_tensor", 1, "i1", "o1")]
    _write_stream(tmp_path / "A" / "s.fp", recs)
    _write_stream(tmp_path / "B" / "s.fp", recs)
    assert diff_rank(str(tmp_path / "A" / "s.fp"), str(tmp_path / "B" / "s.fp")) is None


def test_diff_first_divergence_inputs_match(tmp_path):
    a = [_rec("all_reduce", 0, "i0", "o0"), _rec("reduce_scatter_tensor", 1, "i1", "oA")]
    b = [_rec("all_reduce", 0, "i0", "o0"), _rec("reduce_scatter_tensor", 1, "i1", "oB")]
    _write_stream(tmp_path / "A" / "s.fp", a)
    _write_stream(tmp_path / "B" / "s.fp", b)
    div = diff_rank(str(tmp_path / "A" / "s.fp"), str(tmp_path / "B" / "s.fp"))
    assert div is not None
    assert div["op"] == "reduce_scatter_tensor"
    assert div["align_idx"] == 1
    assert div["input_match"] is True  # inputs matched -> reduction-order/topology root cause
