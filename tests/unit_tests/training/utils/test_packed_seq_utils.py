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

import torch

from megatron.bridge.training.utils.packed_seq_utils import get_packed_seq_params


class TestGetPackedSeqParams:
    """Test suite for get_packed_seq_params function."""

    def test_without_cu_seqlens_unpadded(self):
        """Test get_packed_seq_params when cu_seqlens_unpadded is NOT present.

        This corresponds to pad_seq_to_mult == 1 (no padding for CP).
        The function should return PackedSeqParams with only cu_seqlens_q/kv set,
        and cu_seqlens_q_padded/kv_padded should NOT be set to avoid the slower TE kernel.
        """
        # Create batch without cu_seqlens_unpadded
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, 384, -1, -1]),
            "cu_seqlens_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)

        # Verify cu_seqlens_q and cu_seqlens_kv use padded values
        expected_cu_seqlens = torch.IntTensor([0, 128, 256, 384])
        torch.testing.assert_close(result.cu_seqlens_q, expected_cu_seqlens)
        torch.testing.assert_close(result.cu_seqlens_kv, expected_cu_seqlens)

        # Verify padded variants are NOT set (None) to avoid slower kernel path
        assert result.cu_seqlens_q_padded is None
        assert result.cu_seqlens_kv_padded is None

        # Verify other params
        assert result.max_seqlen_q == 128
        assert result.max_seqlen_kv == 128
        assert result.qkv_format == "thd"

    def test_with_cu_seqlens_unpadded(self):
        """Test get_packed_seq_params when cu_seqlens_unpadded IS present.

        This corresponds to pad_seq_to_mult > 1 (actual padding for THD CP).
        The function should return PackedSeqParams with both unpadded and padded variants.
        """
        # Create batch with cu_seqlens_unpadded (for THD CP with padding)
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, 384, -1, -1]),  # Padded lengths
            "cu_seqlens_argmin": torch.tensor(4),
            "cu_seqlens_unpadded": torch.IntTensor([0, 120, 245, 370, -1, -1]),  # Actual unpadded lengths
            "cu_seqlens_unpadded_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)

        # Verify cu_seqlens_q and cu_seqlens_kv use unpadded values
        expected_unpadded = torch.IntTensor([0, 120, 245, 370])
        torch.testing.assert_close(result.cu_seqlens_q, expected_unpadded)
        torch.testing.assert_close(result.cu_seqlens_kv, expected_unpadded)

        # Verify padded variants are set for THD CP support
        expected_padded = torch.IntTensor([0, 128, 256, 384])
        torch.testing.assert_close(result.cu_seqlens_q_padded, expected_padded)
        torch.testing.assert_close(result.cu_seqlens_kv_padded, expected_padded)

        # Verify other params
        assert result.max_seqlen_q == 128
        assert result.max_seqlen_kv == 128
        assert result.qkv_format == "thd"

    def test_without_argmin_falls_back_to_torch_argmin(self):
        """Test that function falls back to torch.argmin when argmin tensors not provided."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, -1, -1]),
            "max_seqlen": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)

        # Should find argmin at index 3 (where -1 starts)
        expected_cu_seqlens = torch.IntTensor([0, 128, 256])
        torch.testing.assert_close(result.cu_seqlens_q, expected_cu_seqlens)
        torch.testing.assert_close(result.cu_seqlens_kv, expected_cu_seqlens)

    def test_with_batch_dimension(self):
        """Test that function correctly squeezes batch dimensions."""
        # Create batch with extra batch dimension
        batch = {
            "cu_seqlens": torch.IntTensor([[0, 64, 128, -1]]),  # Shape [1, 4]
            "cu_seqlens_argmin": torch.tensor([[3]]),  # Shape [1, 1]
            "max_seqlen": torch.tensor([[64]]),  # Shape [1, 1]
        }

        result = get_packed_seq_params(batch)

        expected_cu_seqlens = torch.IntTensor([0, 64, 128])
        torch.testing.assert_close(result.cu_seqlens_q, expected_cu_seqlens)
        assert result.max_seqlen_q == 64

    def test_without_max_seqlen(self):
        """Test that function handles missing max_seqlen gracefully."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 100, 200, -1]),
            "cu_seqlens_argmin": torch.tensor(3),
        }

        result = get_packed_seq_params(batch)

        assert result.max_seqlen_q is None
        assert result.max_seqlen_kv is None

    def test_unpadded_without_argmin(self):
        """Test unpadded seqlens processing when argmin is not provided."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, -1]),
            "cu_seqlens_argmin": torch.tensor(3),
            "cu_seqlens_unpadded": torch.IntTensor([0, 120, 240, -1]),
            # No cu_seqlens_unpadded_argmin - should use torch.argmin
            "max_seqlen": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)

        expected_unpadded = torch.IntTensor([0, 120, 240])
        torch.testing.assert_close(result.cu_seqlens_q, expected_unpadded)
        torch.testing.assert_close(result.cu_seqlens_kv, expected_unpadded)

    def test_single_sequence(self):
        """Test with a single sequence (common edge case)."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 512, -1]),
            "cu_seqlens_argmin": torch.tensor(2),
            "max_seqlen": torch.tensor(512),
        }

        result = get_packed_seq_params(batch)

        expected = torch.IntTensor([0, 512])
        torch.testing.assert_close(result.cu_seqlens_q, expected)
        assert result.cu_seqlens_q_padded is None  # No unpadded, so no padded variants

    def test_performance_no_unnecessary_padded_variants(self):
        """Verify that when unpadded is not provided, padded variants are None.

        This is the key performance optimization - when pad_seq_to_mult == 1,
        we don't set cu_seqlens_*_padded to avoid triggering the slower TE kernel.
        """
        batch = {
            "cu_seqlens": torch.IntTensor([0, 256, 512, 768, 1024, -1]),
            "cu_seqlens_argmin": torch.tensor(5),
            "max_seqlen": torch.tensor(256),
        }

        result = get_packed_seq_params(batch)

        # Critical: padded variants must be None to avoid perf regression
        assert result.cu_seqlens_q_padded is None, (
            "cu_seqlens_q_padded should be None when cu_seqlens_unpadded is not provided"
        )
        assert result.cu_seqlens_kv_padded is None, (
            "cu_seqlens_kv_padded should be None when cu_seqlens_unpadded is not provided"
        )

        # But cu_seqlens_q/kv should still be set
        assert result.cu_seqlens_q is not None
        assert result.cu_seqlens_kv is not None
