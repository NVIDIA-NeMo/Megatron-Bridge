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

from megatron.bridge.training.setup import _validate_and_set_vocab_size


class TestValidateAndSetVocabSize:
    """Test cases for the _validate_and_set_vocab_size function."""

    def test_vocab_size_none_uses_tokenizer_padded_size(self):
        """Test that None vocab_size uses tokenizer's padded vocab size."""
        result = _validate_and_set_vocab_size(
            model_vocab_size=None,
            tokenizer_vocab_size=32004,
            tokenizer_padded_vocab_size=32768,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
        )
        assert result == 32768

    def test_vocab_size_smaller_than_tokenizer_raises_error(self):
        """Test that vocab_size smaller than tokenizer raises ValueError."""
        with pytest.raises(ValueError, match="cannot be smaller than tokenizer's vocab_size"):
            _validate_and_set_vocab_size(
                model_vocab_size=30000,
                tokenizer_vocab_size=32004,
                tokenizer_padded_vocab_size=32768,
                make_vocab_size_divisible_by=128,
                tensor_model_parallel_size=8,
            )

    def test_vocab_size_properly_padded_returns_same_value(self):
        """Test that properly padded vocab_size returns the same value."""
        # 40960 = 40 * 1024, where 1024 = 128 * 8
        result = _validate_and_set_vocab_size(
            model_vocab_size=40960,
            tokenizer_vocab_size=32004,
            tokenizer_padded_vocab_size=32768,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
        )
        assert result == 40960

    def test_vocab_size_not_properly_padded_raises_error(self):
        """Test that improperly padded vocab_size raises ValueError."""
        with pytest.raises(ValueError, match="not properly padded for tensor parallelism"):
            _validate_and_set_vocab_size(
                model_vocab_size=40000,  # Not divisible by 128*8=1024
                tokenizer_vocab_size=32004,
                tokenizer_padded_vocab_size=32768,
                make_vocab_size_divisible_by=128,
                tensor_model_parallel_size=8,
            )

    def test_vocab_size_equal_to_tokenizer_size_properly_padded(self):
        """Test vocab_size equal to tokenizer size that's properly padded."""
        # Test with a tokenizer size that happens to be properly padded
        result = _validate_and_set_vocab_size(
            model_vocab_size=32768,
            tokenizer_vocab_size=32768,
            tokenizer_padded_vocab_size=32768,
            make_vocab_size_divisible_by=128,
            tensor_model_parallel_size=8,
        )
        assert result == 32768

    def test_vocab_size_equal_to_tokenizer_size_not_properly_padded(self):
        """Test vocab_size equal to tokenizer size that's not properly padded."""
        with pytest.raises(ValueError, match="not properly padded for tensor parallelism"):
            _validate_and_set_vocab_size(
                model_vocab_size=32004,  # Same as tokenizer but not properly padded
                tokenizer_vocab_size=32004,
                tokenizer_padded_vocab_size=32768,
                make_vocab_size_divisible_by=128,
                tensor_model_parallel_size=8,
            )
