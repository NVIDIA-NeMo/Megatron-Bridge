#!/usr/bin/env python3
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

"""CPU checks for the distributed master-update comparison; run with pytest."""

import numpy as np
import pytest
from compare_optimizer_256k import SHAPES, coordinates, errors, read_arrays


def test_master_shards_are_joined_by_logical_coordinate(tmp_path, monkeypatch):
    name = "last_shared_fc2"
    monkeypatch.setitem(SHAPES, name, (2, 2048))
    records = []
    for rank, (start, end) in enumerate(((2048, 4096), (0, 2048))):
        records.append({"rank": rank, "observed": {name: {"primary": True, "start": start, "end": end}}})
        ix = coordinates(name)
        ix = ix[(start <= ix) & (ix < end)][::-1]
        pre = ix.astype(np.float32) + 1
        post = np.nextafter(pre, np.float32(np.inf))
        np.savez(tmp_path / f"{rank:02}-{name}.npz", index=ix, pre=pre, grad=-pre, post=post)
    data = read_arrays(tmp_path, records, name)
    expected = coordinates(name).astype(np.float32) + 1
    np.testing.assert_array_equal(data["grad"], -expected)
    np.testing.assert_array_equal(data["delta"], np.spacing(expected).astype(np.float64))
    assert data["delta"].dtype == np.float64
    with pytest.raises(ValueError, match="overlapping or missing"):
        read_arrays(tmp_path, records + records[:1], name)


def test_all_bos_channels_are_sampled_exactly_once():
    ix = coordinates("embedding")
    np.testing.assert_array_equal(ix[(ix >= 4096) & (ix < 8192)], np.arange(4096, 8192))
    assert np.all(np.diff(ix) > 0)


def test_relative_peak_and_l2_gate_each_apply():
    ref = np.ones(25, dtype=np.float32)
    one_outlier = ref.copy()
    one_outlier[12] += 0.02
    scattered_error = ref + 0.006
    first = errors(ref, one_outlier)
    second = errors(ref, scattered_error)
    assert first["relative_l2"] < 0.005 < 0.01 < first["peak_relative"] and not first["passed"]
    assert 0.005 < second["relative_l2"] and second["peak_relative"] < 0.01 and not second["passed"]
    assert errors(ref, ref)["passed"]
