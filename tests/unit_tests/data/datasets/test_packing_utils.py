#!/usr/bin/env python3
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

import numpy as np

from megatron.bridge.data.datasets.packing_utils import (
    find_first_bin_that_fits,
    create_packing_strategy,
    fill_packing_strategy,
    first_fit_decreasing,
    first_fit_shuffle,
    create_hist,
    first_fit,
)


class TestDataPackingUtils:
    def test_find_first_bin_that_fits(self):
        bins = [
            [1111, 2, 3],
            [17, 11, 0, -5],
            [100, 200],
        ]
        bin_size = 1
        s = 11
        first_bin_that_fits = find_first_bin_that_fits(bins, s, bin_size)

        assert first_bin_that_fits == -1

        bin_size = 1000
        first_bin_that_fits = find_first_bin_that_fits(bins, s, bin_size)

        assert first_bin_that_fits == 1

    def test_first_fit(self):
        bs = 128
        seqlens = [4096 for i in range(bs)]
        pack_size = 2048

        res = first_fit(seqlens, pack_size)

        assert res == [[4096] for i in range(bs)]

    def test_first_fit_decreasing(self):
        seqlens = [1111, 8192, 4096, 1000]
        pack_size = 2048

        first_fit = first_fit_decreasing(seqlens, pack_size)

        assert first_fit == [[8192], [4096], [1111], [1000]]

    def test_first_fit_shuffle(self):
        seqlens = [1111, 8192, 4096, 1000]
        pack_size = 4096

        first_fit = first_fit_shuffle(seqlens, pack_size)

        assert type(first_fit) == list

    def test_create_hist(self):
        ids = [1, 2, 3]
        dataset = [{"input_ids": ids} for i in range(6)]
        truncate_seq_len = 5

        hist, seq = create_hist(dataset, truncate_seq_len)

        assert seq == [0, 0, 6, 0, 0, 0]

    def test_create_packing_strategy(self):
        hist = [1, 77]
        pack_size = 1

        assignments, packing_metadata = create_packing_strategy(hist, pack_size)

        assert packing_metadata == {"dataset_max_seqlen": 1, "max_samples_per_bin": 2}

        sequences = {
            0: [
                {"input_ids": [19, 0, 21413, 1873], "answer_start_idx": 0}
                for i in range(128)
            ],
            1: [
                {"input_ids": [17, 35, 2, 11], "answer_start_idx": 0}
                for i in range(128)
            ],
            2: [
                {"input_ids": [111, 9999, 5, 6], "answer_start_idx": 0}
                for i in range(128)
            ],
        }

        try:
            data = fill_packing_strategy(assignments, sequences, 1, 1000)
        except AssertionError as e:
            assert e.args[0] == "Error: There are items left over from the assignment"
