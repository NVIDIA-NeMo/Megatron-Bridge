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


import pytest

from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size


class TestCalculatePaddedVocabSize:
    """Test cases for the calculate_padded_vocab_size function."""

    def test_no_padding_needed(self):
        """Test when vocab size is already properly padded."""
        # 1024 is divisible by 128 * 8 = 1024
        result = calculate_padded_vocab_size(
            vocab_size=1024, make_vocab_size_divisible_by=128, tensor_model_parallel_size=8
        )
        assert result == 1024

    def test_padding_needed_simple(self):
        """Test basic padding calculation."""
        # 32000 -> should pad to next multiple of 128 * 8 = 1024
        # 32000 / 1024 = 31.25, so ceiling is 32
        # 32 * 1024 = 32768
        result = calculate_padded_vocab_size(
            vocab_size=32000, make_vocab_size_divisible_by=128, tensor_model_parallel_size=8
        )
        assert result == 32768

    def test_padding_with_different_parameters(self):
        """Test padding with different divisibility and parallelism parameters."""
        # 50000 with 64 * 4 = 256 multiple
        # 50000 / 256 = 195.31, so ceiling is 196
        # 196 * 256 = 50176
        result = calculate_padded_vocab_size(
            vocab_size=50000, make_vocab_size_divisible_by=64, tensor_model_parallel_size=4
        )
        assert result == 50176

    def test_tensor_parallel_size_one(self):
        """Test with tensor parallel size of 1."""
        # 1000 with 128 * 1 = 128 multiple
        # 1000 / 128 = 7.81, so ceiling is 8
        # 8 * 128 = 1024
        result = calculate_padded_vocab_size(
            vocab_size=1000, make_vocab_size_divisible_by=128, tensor_model_parallel_size=1
        )
        assert result == 1024

    def test_vocab_size_zero_raises_error(self):
        """Test that vocab_size of 0 raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            calculate_padded_vocab_size(vocab_size=0, make_vocab_size_divisible_by=128, tensor_model_parallel_size=8)

    def test_negative_vocab_size_raises_error(self):
        """Test that negative vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            calculate_padded_vocab_size(
                vocab_size=-1000, make_vocab_size_divisible_by=128, tensor_model_parallel_size=8
            )

    def test_zero_divisible_by_raises_error(self):
        """Test that make_vocab_size_divisible_by of 0 raises ValueError."""
        with pytest.raises(ValueError, match="make_vocab_size_divisible_by must be positive"):
            calculate_padded_vocab_size(vocab_size=32000, make_vocab_size_divisible_by=0, tensor_model_parallel_size=8)

    def test_negative_divisible_by_raises_error(self):
        """Test that negative make_vocab_size_divisible_by raises ValueError."""
        with pytest.raises(ValueError, match="make_vocab_size_divisible_by must be positive"):
            calculate_padded_vocab_size(
                vocab_size=32000, make_vocab_size_divisible_by=-128, tensor_model_parallel_size=8
            )

    def test_zero_tensor_parallel_size_raises_error(self):
        """Test that tensor_model_parallel_size of 0 raises ValueError."""
        with pytest.raises(ValueError, match="tensor_model_parallel_size must be positive"):
            calculate_padded_vocab_size(
                vocab_size=32000, make_vocab_size_divisible_by=128, tensor_model_parallel_size=0
            )

    def test_negative_tensor_parallel_size_raises_error(self):
        """Test that negative tensor_model_parallel_size raises ValueError."""
        with pytest.raises(ValueError, match="tensor_model_parallel_size must be positive"):
            calculate_padded_vocab_size(
                vocab_size=32000, make_vocab_size_divisible_by=128, tensor_model_parallel_size=-8
            )
