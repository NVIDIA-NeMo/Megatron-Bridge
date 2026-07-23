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

"""Tests for the canonical collate-time MCore THD packing implementation."""

import pytest
import torch

from megatron.bridge.data.packing.in_batch import pack_right_padded_sequence_batch_to_mcore_thd


def _pack_padded_sequence_for_test(
    tokens: torch.Tensor,
    labels: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    pad_token_id: int = 0,
    pad_to_multiple_of: int = 1,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Adapt the canonical mutable-batch API to compact assertions in these tests."""
    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    pack_right_padded_sequence_batch_to_mcore_thd(
        batch,
        pad_token_id=pad_token_id,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    cu_seqlens = batch.get("cu_seqlens_q_padded")
    if cu_seqlens is None:
        cu_seqlens = batch["cu_seqlens_q"]
    return (
        batch["tokens"],
        batch.get("labels"),
        batch.get("loss_mask"),
        batch.get("attention_mask"),
        batch["position_ids"],
        cu_seqlens,
        batch["max_seqlen_q"],
    )


class TestPackBatchSequences:
    """Tests for the _pack_padded_sequence_for_test function."""

    def test_basic_packing(self):
        """Test basic sequence packing functionality."""
        batch_size, seq_len = 2, 8
        # Tokens with padding at the end (pad_token_id=0)
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],  # length 3
                [4, 5, 6, 7, 0, 0, 0, 0],  # length 4
            ]
        )
        labels = torch.tensor(
            [
                [2, 3, -100, -100, -100, -100, -100, -100],
                [5, 6, 7, -100, -100, -100, -100, -100],
            ]
        )
        loss_mask = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = None

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        # Packed output should have shape [1, total_valid_len]
        assert packed_tokens.shape[0] == 1
        total_len = packed_tokens.shape[1]
        assert total_len == 7  # 3 + 4

        # cu_seqlens should have num_sequences + 1 elements
        assert len(cu_seqlens) == 3  # [0, 3, 7]
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 3  # first sequence length
        assert cu_seqlens[2] == 7  # total length

        # max_seqlen should be max of sequence lengths
        assert max_seqlen.item() == 4

        # Attention mask should be None for packed sequences
        assert packed_attn is None

    def test_packing_with_pad_to_multiple_of(self):
        """Test packing with padding to a multiple (for CP compatibility)."""
        batch_size = 2
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0, 0, 0],  # length 3 -> padded to 4 (mult of 2)
                [4, 5, 6, 7, 8, 0, 0, 0, 0, 0],  # length 5 -> padded to 6 (mult of 2)
            ]
        )
        labels = torch.tensor(
            [
                [2, 3, -100, -100, -100, -100, -100, -100, -100, -100],
                [5, 6, 7, 8, -100, -100, -100, -100, -100, -100],
            ]
        )
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(10).unsqueeze(0).expand(batch_size, -1)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=2,  # Pad each sequence to multiple of 2
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        # Total length should be 4 + 6 = 10 (padded lengths)
        assert packed_tokens.shape[1] == 10

        # cu_seqlens should use padded lengths
        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 4  # 3 -> 4 (padded)
        assert cu_seqlens[2] == 10  # 5 -> 6, total = 4 + 6

        # max_seqlen should be 6 (longest padded sequence)
        assert max_seqlen.item() == 6

    def test_packing_with_larger_multiple(self):
        """Test packing with larger pad_to_multiple_of (e.g., for CP=4)."""
        tokens = torch.tensor(
            [
                [1, 2, 0, 0, 0, 0, 0, 0],  # length 2 -> padded to 4
                [3, 4, 5, 0, 0, 0, 0, 0],  # length 3 -> padded to 4
            ]
        )
        labels = torch.tensor(
            [
                [2, -100, -100, -100, -100, -100, -100, -100],
                [4, 5, -100, -100, -100, -100, -100, -100],
            ]
        )
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=4,
        )

        packed_tokens, *_, cu_seqlens, max_seqlen = result

        # Both sequences padded to 4, total = 8
        assert packed_tokens.shape[1] == 8
        assert cu_seqlens.tolist() == [0, 4, 8]
        assert max_seqlen.item() == 4

    def test_packing_single_sequence(self):
        """Test packing a single sequence."""
        tokens = torch.tensor([[1, 2, 3, 4, 5, 0, 0, 0]])  # length 5
        labels = torch.tensor([[2, 3, 4, 5, -100, -100, -100, -100]])
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(8).unsqueeze(0)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, *_, cu_seqlens, max_seqlen = result

        assert packed_tokens.shape[1] == 5
        assert cu_seqlens.tolist() == [0, 5]
        assert max_seqlen.item() == 5

    def test_packing_no_padding_sequences(self):
        """Test packing sequences with no padding."""
        tokens = torch.tensor(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
            ]
        )
        labels = torch.tensor(
            [
                [2, 3, 4, -100],
                [6, 7, 8, -100],
            ]
        )
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(4).unsqueeze(0).expand(2, -1)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, *_, cu_seqlens, max_seqlen = result

        # Both sequences full length
        assert packed_tokens.shape[1] == 8
        assert cu_seqlens.tolist() == [0, 4, 8]

    def test_packing_preserves_loss_mask_zeros(self):
        """Test that loss_mask zeros are preserved during packing."""
        tokens = torch.tensor([[1, 2, 3, 0, 0]])
        labels = torch.tensor([[2, 3, -100, -100, -100]])
        loss_mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 0.0]])  # Second token masked
        position_ids = torch.arange(5).unsqueeze(0)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, *_ = result

        # Only first 3 tokens should be kept
        assert packed_loss_mask.shape[1] == 3
        assert packed_loss_mask[0, 0].item() == 1.0
        assert packed_loss_mask[0, 1].item() == 0.0  # Preserved
        assert packed_loss_mask[0, 2].item() == 1.0

    def test_packing_position_ids_reset(self):
        """Test that position_ids are correctly packed."""
        tokens = torch.tensor(
            [
                [1, 2, 0, 0],
                [3, 4, 5, 0],
            ]
        )
        labels = torch.zeros_like(tokens)
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.tensor(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 3],
            ]
        )

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        _, _, _, _, packed_pos, *_ = result

        # Position IDs should be extracted from original sequences
        assert packed_pos.shape[1] == 5  # 2 + 3
        assert packed_pos[0, 0].item() == 0  # First seq, pos 0
        assert packed_pos[0, 1].item() == 1  # First seq, pos 1
        assert packed_pos[0, 2].item() == 0  # Second seq, pos 0
        assert packed_pos[0, 3].item() == 1  # Second seq, pos 1
        assert packed_pos[0, 4].item() == 2  # Second seq, pos 2

    def test_packing_empty_batch_raises(self):
        """Test that all-padding batch fails instead of hiding invalid metadata."""
        tokens = torch.tensor([[0, 0, 0, 0]])  # All padding
        labels = torch.tensor([[-100, -100, -100, -100]])
        loss_mask = torch.zeros(1, 4)
        position_ids = torch.arange(4).unsqueeze(0)

        with pytest.raises(ValueError, match="Cannot pack a batch containing an empty sequence row"):
            _pack_padded_sequence_for_test(
                tokens=tokens,
                labels=labels,
                loss_mask=loss_mask,
                attention_mask=None,
                position_ids=position_ids,
                pad_token_id=0,
                pad_to_multiple_of=1,
            )

    def test_packing_different_dtypes(self):
        """Test packing with different tensor dtypes."""
        tokens = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
        labels = torch.tensor([[2, 3, -100, -100]], dtype=torch.long)
        loss_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)
        position_ids = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, _, packed_pos, cu_seqlens, _ = result

        # Dtypes should be preserved
        assert packed_tokens.dtype == torch.long
        assert packed_labels.dtype == torch.long
        assert packed_loss_mask.dtype == torch.float32
        assert packed_pos.dtype == torch.long
        assert cu_seqlens.dtype == torch.int32

    def test_packing_padding_extends_position_ids(self):
        """Test that padding extends position_ids correctly."""
        tokens = torch.tensor([[1, 2, 3, 0]])  # length 3
        labels = torch.zeros_like(tokens)
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.tensor([[0, 1, 2, 3]])

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=4,  # Pad to 4
        )

        _, _, _, _, packed_pos, cu_seqlens, _ = result

        # Length should be 4 (padded)
        assert packed_pos.shape[1] == 4

        # Original positions should be preserved
        assert packed_pos[0, 0].item() == 0
        assert packed_pos[0, 1].item() == 1
        assert packed_pos[0, 2].item() == 2
        # Padding position should be extended
        assert packed_pos[0, 3].item() == 3

    def test_packing_cu_seqlens_dtype(self):
        """Test that cu_seqlens is int32 as expected by attention kernels."""
        tokens = torch.tensor([[1, 2, 0]])
        labels = torch.zeros_like(tokens)
        loss_mask = torch.ones_like(tokens, dtype=torch.float)
        position_ids = torch.arange(3).unsqueeze(0)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
        )

        _, _, _, _, _, cu_seqlens, _ = result

        assert cu_seqlens.dtype == torch.int32

    def test_packing_none_labels_loss_mask(self):
        """Test packing with labels=None and loss_mask=None (non-last PP stage)."""
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],  # length 3
                [4, 5, 6, 7, 0, 0, 0, 0],  # length 4
            ]
        )
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=None,
            loss_mask=None,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=1,
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        assert packed_tokens.shape == (1, 7)
        assert torch.equal(packed_tokens, torch.tensor([[1, 2, 3, 4, 5, 6, 7]]))
        assert packed_labels is None
        assert packed_loss_mask is None
        assert packed_attn is None
        assert packed_pos.shape == (1, 7)
        assert torch.equal(packed_pos, torch.tensor([[0, 1, 2, 0, 1, 2, 3]]))
        assert cu_seqlens.tolist() == [0, 3, 7]
        assert max_seqlen.item() == 4

    def test_packing_none_labels_loss_mask_with_padding(self):
        """Test packing with None labels/loss_mask and pad_to_multiple_of > 1."""
        tokens = torch.tensor(
            [
                [1, 2, 3, 0, 0, 0, 0, 0],  # length 3 -> padded to 4
                [4, 5, 6, 7, 8, 0, 0, 0],  # length 5 -> padded to 8
            ]
        )
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=None,
            loss_mask=None,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
            pad_to_multiple_of=4,
        )

        packed_tokens, packed_labels, packed_loss_mask, packed_attn, packed_pos, cu_seqlens, max_seqlen = result

        assert packed_tokens.shape == (1, 12)
        assert packed_labels is None
        assert packed_loss_mask is None
        assert packed_attn is None
        assert cu_seqlens.tolist() == [0, 4, 12]
        assert max_seqlen.item() == 8

    def test_packing_none_labels_empty_batch(self):
        """Test empty batch with None labels/loss_mask raises like other all-padding batches."""
        tokens = torch.tensor([[0, 0, 0, 0]])
        position_ids = torch.arange(4).unsqueeze(0)

        with pytest.raises(ValueError, match="Cannot pack a batch containing an empty sequence row"):
            _pack_padded_sequence_for_test(
                tokens=tokens,
                labels=None,
                loss_mask=None,
                attention_mask=None,
                position_ids=position_ids,
                pad_token_id=0,
                pad_to_multiple_of=1,
            )

    def test_packing_gpu_tensor(self):
        """Test packing works on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        tokens = torch.tensor([[1, 2, 3, 0, 0]], device="cuda")
        labels = torch.tensor([[2, 3, -100, -100, -100]], device="cuda")
        loss_mask = torch.ones_like(tokens, dtype=torch.float, device="cuda")
        position_ids = torch.arange(5, device="cuda").unsqueeze(0)

        result = _pack_padded_sequence_for_test(
            tokens=tokens,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=None,
            position_ids=position_ids,
            pad_token_id=0,
        )

        packed_tokens, _, _, _, _, cu_seqlens, _ = result

        assert packed_tokens.device.type == "cuda"
        assert cu_seqlens.device.type == "cuda"
