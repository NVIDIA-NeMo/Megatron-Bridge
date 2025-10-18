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
import pytest

from megatron.bridge.data.datasets.packing_utils import (
    create_hist,
    create_packing_strategy,
    fill_packing_strategy,
    find_first_bin_that_fits,
    first_fit,
    first_fit_decreasing,
    first_fit_shuffle,
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
            0: [{"input_ids": [19, 0, 21413, 1873], "answer_start_idx": 0} for i in range(128)],
            1: [{"input_ids": [17, 35, 2, 11], "answer_start_idx": 0} for i in range(128)],
            2: [{"input_ids": [111, 9999, 5, 6], "answer_start_idx": 0} for i in range(128)],
        }

        try:
            fill_packing_strategy(assignments, sequences, 1, 1000)
        except AssertionError as e:
            assert e.args[0] == "Error: There are items left over from the assignment"

    def test_fill_packing_strategy_basic_with_loss_mask(self):
        """Test fill_packing_strategy with sequences that have loss_mask directly provided."""
        # Create simple assignments: two packs, first with sequences of length 2, second with length 3
        assignments = [[2, 2], [3]]

        # Create sequences with loss_mask provided directly
        # Note: sequences dict must have keys for all seq_len from 0 to pack_size
        sequences = {
            0: [],
            1: [],
            2: [
                {"input_ids": [1, 2, 3], "loss_mask": [True, False, True]},  # len 3, but seq_len = 2 (len - 1)
                {"input_ids": [4, 5, 6], "loss_mask": [False, True, True]},
            ],
            3: [
                {"input_ids": [7, 8, 9, 10], "loss_mask": [True, True, False, True]},  # len 4, but seq_len = 3
            ],
            4: [],
            5: [],
        }

        pack_size = 5
        pad_id = 0

        # Set seed for deterministic results due to randomization in function
        np.random.seed(42)
        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        # Should have 2 packed sequences
        assert len(output_data) == 2

        # First pack: two sequences of length 2 each
        first_pack = output_data[0]
        assert len(first_pack["input_ids"]) == 6  # 3 + 3
        assert len(first_pack["loss_mask"]) == 6
        assert len(first_pack["seq_start_id"]) == 2  # Two sequences in this pack
        assert first_pack["seq_start_id"] == [0, 3]  # First starts at 0, second at 3

        # Check loss_mask is correctly rolled by 1 (aligned with labels)
        # Original: [True, False, True] -> rolled: [False, True, False]
        # Original: [False, True, True] -> rolled: [True, True, False]
        expected_loss_mask_1 = [False, True, False] + [True, True, False]
        assert first_pack["loss_mask"] == expected_loss_mask_1

        # Second pack: one sequence of length 3
        second_pack = output_data[1]
        assert len(second_pack["input_ids"]) == 4  # One sequence of input length 4
        assert len(second_pack["loss_mask"]) == 4
        assert len(second_pack["seq_start_id"]) == 1  # One sequence in this pack
        assert second_pack["seq_start_id"] == [0]

        # Check loss_mask is correctly rolled
        # Original: [True, True, False, True] -> rolled: [True, False, True, False]
        expected_loss_mask_2 = [True, False, True, False]
        assert second_pack["loss_mask"] == expected_loss_mask_2

    def test_fill_packing_strategy_with_answer_start_idx(self):
        """Test fill_packing_strategy with sequences that use answer_start_idx."""
        assignments = [[2], [1, 1]]

        sequences = {
            0: [],
            1: [
                {"input_ids": [10, 20], "answer_start_idx": 0},  # seq_len = 1
                {"input_ids": [30, 40], "answer_start_idx": 1},  # seq_len = 1
            ],
            2: [
                {"input_ids": [50, 60, 70], "answer_start_idx": 1},  # seq_len = 2
            ],
            3: [],
        }

        pack_size = 3
        pad_id = 999

        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        assert len(output_data) == 2

        # First pack: one sequence of length 2
        first_pack = output_data[0]
        assert len(first_pack["input_ids"]) == 3
        assert first_pack["input_ids"] == [50, 60, 70]
        assert len(first_pack["seq_start_id"]) == 1

        # Loss mask computed from answer_start_idx = 1
        # For idx >= (answer_start_idx - 1) = 0, and token != pad_id
        # So for tokens [50, 60, 70]: [True, True, True] (all >= 0 and != 999)
        expected_loss_mask = [True, True, True]
        assert first_pack["loss_mask"] == expected_loss_mask

        # Second pack: two sequences of length 1 each
        second_pack = output_data[1]
        assert len(second_pack["input_ids"]) == 4  # 2 + 2
        assert len(second_pack["seq_start_id"]) == 2
        assert second_pack["seq_start_id"] == [0, 2]

    def test_fill_packing_strategy_with_padding_in_answer_start_idx(self):
        """Test answer_start_idx logic correctly handles padding tokens."""
        assignments = [[1]]

        sequences = {
            0: [],
            1: [
                {"input_ids": [100, 999, 200], "answer_start_idx": 1},  # seq_len = 2, pad_id in sequence
            ],
            2: [],
            3: [],
        }

        pack_size = 3
        pad_id = 999

        # Set seed for deterministic results
        np.random.seed(42)
        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        # Loss mask: for idx >= (answer_start_idx - 1) = 0, and token != pad_id
        # [100, 999, 200]: idx 0 >= 0 and 100 != 999 -> True
        #                   idx 1 >= 0 but 999 == 999 -> False
        #                   idx 2 >= 0 and 200 != 999 -> True
        expected_loss_mask = [True, False, True]
        assert output_data[0]["loss_mask"] == expected_loss_mask

    def test_fill_packing_strategy_missing_keys_error(self):
        """Test error when sequences don't have loss_mask or answer_start_idx."""
        assignments = [[1]]

        sequences = {
            0: [],
            1: [
                {"input_ids": [1, 2]},  # Missing both loss_mask and answer_start_idx
            ],
            2: [],
        }

        pack_size = 2
        pad_id = 0

        with pytest.raises(ValueError, match="Key errors loss_mask and answer_start_idx missing"):
            fill_packing_strategy(assignments, sequences, pack_size, pad_id)

    def test_fill_packing_strategy_single_sequence(self):
        """Test with a single sequence in a single pack."""
        assignments = [[3]]

        sequences = {
            0: [],
            1: [],
            2: [],
            3: [
                {"input_ids": [1, 2, 3, 4], "loss_mask": [True, True, False, True]},
            ],
            4: [],
        }

        pack_size = 4
        pad_id = 0

        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        assert len(output_data) == 1
        pack = output_data[0]
        assert pack["input_ids"] == [1, 2, 3, 4]
        assert pack["loss_mask"] == [True, False, True, False]  # Rolled by 1
        assert pack["seq_start_id"] == [0]

    def test_fill_packing_strategy_empty_sequences_for_length(self):
        """Test with sequence lengths that have no data in assignments."""
        # Only assign sequences of length 1 (which we have data for)
        assignments = [[1]]

        # Provide sequences for length 1 only, others are empty
        sequences = {
            0: [],
            1: [
                {"input_ids": [10, 20], "loss_mask": [True, False]},
            ],
            2: [],  # Empty - no sequences of this length
            3: [],
        }

        pack_size = 3
        pad_id = 0

        # This should work fine - empty sequences are handled gracefully
        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        # Should have 1 pack since we only have assignments for length 1
        assert len(output_data) == 1

        # First pack should have the length-1 sequence
        first_pack = output_data[0]
        assert first_pack["input_ids"] == [10, 20]
        assert first_pack["loss_mask"] == [False, False]  # Rolled: [True, False] -> [False, False]
        assert first_pack["seq_start_id"] == [0]

    def test_fill_packing_strategy_multiple_sequences_per_pack(self):
        """Test packing multiple sequences of different lengths in same pack."""
        assignments = [[1, 2, 1]]  # Three sequences in one pack: lengths 1, 2, 1

        sequences = {
            0: [],
            1: [
                {"input_ids": [100, 101], "loss_mask": [True, False]},  # seq_len = 1
                {"input_ids": [200, 201], "loss_mask": [False, True]},  # seq_len = 1
            ],
            2: [
                {"input_ids": [300, 301, 302], "loss_mask": [True, True, False]},  # seq_len = 2
            ],
            3: [],
            4: [],
            5: [],
            6: [],
            7: [],
        }

        pack_size = 7
        pad_id = 0

        output_data = fill_packing_strategy(assignments, sequences, pack_size, pad_id)

        assert len(output_data) == 1
        pack = output_data[0]

        # Should concatenate: [100, 101] + [300, 301, 302] + [200, 201] = 7 tokens
        assert len(pack["input_ids"]) == 7
        assert len(pack["loss_mask"]) == 7
        assert len(pack["seq_start_id"]) == 3  # Three sequences
        assert pack["seq_start_id"] == [0, 2, 5]  # Start positions: 0, 2, 5

        # Check concatenation order and loss mask rolling
        expected_input_ids = [100, 101, 300, 301, 302, 200, 201]
        assert pack["input_ids"] == expected_input_ids

        # Loss masks rolled: [True, False] -> [False, False]
        #                    [True, True, False] -> [True, False, False]
        #                    [False, True] -> [True, False]
        expected_loss_mask = [False, False] + [True, False, False] + [True, False]
        assert pack["loss_mask"] == expected_loss_mask

    def test_fill_packing_strategy_randomization_and_determinism(self):
        """Test that randomization works but results are deterministic with same seed."""
        assignments = [[2, 2]]

        # Create exactly the number of sequences needed to test permutation
        sequences = {
            0: [],
            1: [],
            2: [
                {"input_ids": [i, i + 1, i + 2], "loss_mask": [True] * 3}
                for i in range(10, 30, 10)  # [10,11,12], [20,21,22] - exactly 2 sequences
            ],
            3: [],
            4: [],
            5: [],
            6: [],
        }

        pack_size = 6
        pad_id = 0

        # Set seed for reproducibility
        np.random.seed(42)
        output_data_1 = fill_packing_strategy(assignments, sequences.copy(), pack_size, pad_id)

        # Reset sequences since they get consumed
        sequences = {
            0: [],
            1: [],
            2: [
                {"input_ids": [i, i + 1, i + 2], "loss_mask": [True] * 3}
                for i in range(10, 30, 10)  # [10,11,12], [20,21,22] - exactly 2 sequences
            ],
            3: [],
            4: [],
            5: [],
            6: [],
        }

        # Same seed should give same result
        np.random.seed(42)
        output_data_2 = fill_packing_strategy(assignments, sequences.copy(), pack_size, pad_id)

        assert output_data_1[0]["input_ids"] == output_data_2[0]["input_ids"]
        assert output_data_1[0]["loss_mask"] == output_data_2[0]["loss_mask"]
