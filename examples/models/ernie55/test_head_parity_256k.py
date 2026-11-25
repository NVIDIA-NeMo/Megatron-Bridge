# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from head_parity_256k import compare


def test_logits_and_hidden_gradients_can_have_different_physical_owners(tmp_path):
    old, new = tmp_path / "legacy", tmp_path / "contiguous-head"
    old.mkdir()
    new.mkdir()
    for rank in range(64):
        dp, cp = divmod(rank, 16)
        pos = np.arange(16384 * cp, 16384 * (cp + 1), dtype=np.int32)
        sample = pos[[15, 1023]]
        row = {"dp": dp, "bounds": np.array([0, 262144]), "pos": pos, "tokens": pos, "sample": sample}
        for kind in ("main", "mtp"):
            row[kind + "_labels"] = pos
            row[kind + "_ce"] = np.ones_like(pos, dtype=np.float32)
            row[kind + "_dh"] = sample[:, None].astype(np.float32)
            row[kind + "_logits"] = sample[:, None].astype(np.float32)
        name = f"rank{rank:02}.npz"
        np.savez(old / name, **row)
        # Permute whole rank ownership as well as local row order.
        row["logit_sample"] = np.array([1023, 15]) + (15 - cp) * 16384
        for kind in ("main", "mtp"):
            row[kind + "_logits"] = row["logit_sample"][:, None].astype(np.float32)
        np.savez(new / name, **row)
    result = compare(old, new)
    for stats in result.values():
        assert stats["sample_raw_logit_relative_l2"] == stats["sample_head_input_grad_relative_l2"] == 0
    path = new / "rank00.npz"
    with np.load(path) as arrays:
        bad = dict(arrays)
    bad["logit_sample"] = bad["logit_sample"] + 1
    np.savez(path, **bad)
    with pytest.raises(ValueError, match="different global sample coordinates"):
        compare(old, new)
