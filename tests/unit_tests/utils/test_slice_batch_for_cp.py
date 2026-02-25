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

"""Tests for slice_batch_for_context_parallel function in common_utils."""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import torch

from megatron.bridge.utils.common_utils import slice_batch_for_context_parallel


@dataclass
class MockPackedSeqParams:
    """Mock PackedSeqParams for testing THD format."""

    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    cu_seqlens_q_padded: Optional[torch.Tensor] = None
    cu_seqlens_kv_padded: Optional[torch.Tensor] = None
    max_seqlen_q: Optional[torch.Tensor] = None
    max_seqlen_kv: Optional[torch.Tensor] = None
    qkv_format: str = "thd"


class MockPGCollection:
    """Mock ProcessGroupCollection for testing."""

    def __init__(self, cp_size: int = 1, cp_rank: int = 0):
        self._cp_size = cp_size
        self._cp_rank = cp_rank
        self.cp = MagicMock()
        self.cp.size.return_value = cp_size
        self.cp.rank.return_value = cp_rank


class TestSliceBatchForContextParallelCpSize1:
    """Tests for slice_batch_for_context_parallel when CP size <= 1 (no-op case)."""

    def test_cp_size_1_returns_unchanged(self):
        """Test that tensors are returned unchanged when cp_size=1."""
        batch_size, seq_len, hidden = 2, 16, 64
        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)

        pg_collection = MockPGCollection(cp_size=1, cp_rank=0)

        result = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=None,
            pg_collection=pg_collection,
        )

        out_embeds, out_labels, out_loss_mask, out_pos_ids, out_attn_mask = result

        # All tensors should be unchanged
        assert torch.equal(out_embeds, inputs_embeds)
        assert torch.equal(out_labels, labels)
        assert torch.equal(out_loss_mask, loss_mask)
        assert torch.equal(out_pos_ids, position_ids)
        assert torch.equal(out_attn_mask, attention_mask)

    def test_cp_size_0_returns_unchanged(self):
        """Test that tensors are returned unchanged when cp_size=0."""
        batch_size, seq_len, hidden = 1, 8, 32
        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 100, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        attention_mask = None

        pg_collection = MockPGCollection(cp_size=0, cp_rank=0)

        result = slice_batch_for_context_parallel(
            inputs_embeds=inputs_embeds,
            labels=labels,
            loss_mask=loss_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            packed_seq_params=None,
            pg_collection=pg_collection,
        )

        out_embeds, out_labels, out_loss_mask, out_pos_ids, out_attn_mask = result

        assert torch.equal(out_embeds, inputs_embeds)
        assert torch.equal(out_labels, labels)


class TestSliceBatchForContextParallelBSHD:
    """Tests for slice_batch_for_context_parallel with BSHD format (non-packed)."""

    def test_bshd_format_uses_get_batch_on_this_cp_rank(self):
        """Test that BSHD format triggers get_batch_on_this_cp_rank."""
        batch_size, seq_len, hidden = 2, 16, 64
        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)

        pg_collection = MockPGCollection(cp_size=2, cp_rank=0)

        # Mock get_batch_on_this_cp_rank to return sliced tensors
        def mock_get_batch(batch_dict, cp_group=None):
            # Simulate slicing by returning half of each tensor
            result = {}
            for k, v in batch_dict.items():
                if v is not None and isinstance(v, torch.Tensor):
                    if k == "decoder_input":  # (B, T, D) format after transpose
                        result[k] = v[:, : seq_len // 2, :]
                    elif v.dim() == 4:  # attention_mask
                        result[k] = v[:, :, : seq_len // 2, : seq_len // 2]
                    else:
                        result[k] = v[:, : seq_len // 2]
                else:
                    result[k] = v
            return result

        with patch(
            "megatron.core.utils.get_batch_on_this_cp_rank",
            side_effect=mock_get_batch,
        ):
            result = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                attention_mask=attention_mask,
                packed_seq_params=None,  # BSHD format
                pg_collection=pg_collection,
            )

        out_embeds, out_labels, out_loss_mask, out_pos_ids, out_attn_mask = result

        # Check that output tensors are sliced (half the sequence length)
        assert out_embeds.shape[0] == seq_len // 2  # T dimension
        assert out_embeds.shape[1] == batch_size  # B dimension
        assert out_labels.shape[1] == seq_len // 2
        assert out_loss_mask.shape[1] == seq_len // 2

    def test_bshd_format_handles_none_tensors(self):
        """Test that BSHD format handles None tensors gracefully."""
        batch_size, seq_len, hidden = 1, 8, 32
        inputs_embeds = torch.randn(seq_len, batch_size, hidden)

        pg_collection = MockPGCollection(cp_size=2, cp_rank=0)

        def mock_get_batch(batch_dict, cp_group=None):
            result = {}
            for k, v in batch_dict.items():
                if v is not None and isinstance(v, torch.Tensor):
                    if k == "decoder_input":
                        result[k] = v[:, : seq_len // 2, :]
                    else:
                        result[k] = v[:, : seq_len // 2] if v.dim() > 1 else v
                else:
                    result[k] = None
            return result

        with patch(
            "megatron.core.utils.get_batch_on_this_cp_rank",
            side_effect=mock_get_batch,
        ):
            result = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=None,
                loss_mask=None,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=None,
                pg_collection=pg_collection,
            )

        out_embeds, out_labels, out_loss_mask, out_pos_ids, out_attn_mask = result

        assert out_embeds is not None
        assert out_labels is None
        assert out_loss_mask is None
        assert out_pos_ids is None
        assert out_attn_mask is None


class TestSliceBatchForContextParallelTHD:
    """Tests for slice_batch_for_context_parallel with THD (packed) format."""

    def test_thd_format_uses_tex_partitioned_indices(self):
        """Test that THD format triggers TransformerEngine's thd_get_partitioned_indices."""
        batch_size, seq_len, hidden = 1, 16, 64
        cp_size = 2
        cp_rank = 0

        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 1000, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)
        packed_seq_params = MockPackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens,
            qkv_format="thd",
        )

        pg_collection = MockPGCollection(cp_size=cp_size, cp_rank=cp_rank)

        # Mock tex.thd_get_partitioned_indices
        mock_indices = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11])  # First half of each sequence

        with patch.dict("sys.modules", {"transformer_engine_torch": MagicMock()}):
            import sys

            mock_tex = sys.modules["transformer_engine_torch"]
            mock_tex.thd_get_partitioned_indices.return_value = mock_indices

            result = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                pg_collection=pg_collection,
            )

        out_embeds, out_labels, out_loss_mask, out_pos_ids, out_attn_mask = result

        # Verify tex.thd_get_partitioned_indices was called
        mock_tex.thd_get_partitioned_indices.assert_called_once()

        # Check output shapes match the indices
        assert out_embeds.shape[0] == len(mock_indices)  # T dimension
        assert out_embeds.shape[1] == batch_size  # B dimension

    def test_thd_format_with_padded_cu_seqlens(self):
        """Test THD format uses cu_seqlens_q_padded when available."""
        batch_size, seq_len, hidden = 1, 20, 32
        cp_size = 2
        cp_rank = 1

        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 100, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        # Padded cu_seqlens (includes padding for divisibility)
        cu_seqlens = torch.tensor([0, 6, 16], dtype=torch.int32)
        cu_seqlens_padded = torch.tensor([0, 10, 20], dtype=torch.int32)

        packed_seq_params = MockPackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded,
            qkv_format="thd",
        )

        pg_collection = MockPGCollection(cp_size=cp_size, cp_rank=cp_rank)

        mock_indices = torch.tensor([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])  # Second half

        with patch.dict("sys.modules", {"transformer_engine_torch": MagicMock()}):
            import sys

            mock_tex = sys.modules["transformer_engine_torch"]
            mock_tex.thd_get_partitioned_indices.return_value = mock_indices

            result = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                pg_collection=pg_collection,
            )

        out_embeds, out_labels, out_loss_mask, out_pos_ids, out_attn_mask = result

        # Verify padded cu_seqlens was used
        call_args = mock_tex.thd_get_partitioned_indices.call_args
        assert torch.equal(call_args[0][0], cu_seqlens_padded)

    def test_thd_format_without_padded_cu_seqlens_fallback(self):
        """Test THD format falls back to cu_seqlens_q when padded version is None."""
        batch_size, seq_len, hidden = 1, 12, 16
        cp_size = 2
        cp_rank = 0

        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 50, (batch_size, seq_len))
        loss_mask = torch.ones(batch_size, seq_len)
        position_ids = torch.arange(seq_len).unsqueeze(0)

        cu_seqlens = torch.tensor([0, 6, 12], dtype=torch.int32)

        packed_seq_params = MockPackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=None,  # No padded version
            qkv_format="thd",
        )

        pg_collection = MockPGCollection(cp_size=cp_size, cp_rank=cp_rank)

        mock_indices = torch.tensor([0, 1, 2, 6, 7, 8])

        with patch.dict("sys.modules", {"transformer_engine_torch": MagicMock()}):
            import sys

            mock_tex = sys.modules["transformer_engine_torch"]
            mock_tex.thd_get_partitioned_indices.return_value = mock_indices

            slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                pg_collection=pg_collection,
            )

        # Verify cu_seqlens_q was used as fallback
        call_args = mock_tex.thd_get_partitioned_indices.call_args
        assert torch.equal(call_args[0][0], cu_seqlens)


class TestSliceBatchForContextParallelTranspose:
    """Tests verifying correct tensor transpose operations."""

    def test_input_transposed_before_slicing(self):
        """Test that inputs_embeds is transposed from (T,B,D) to (B,T,D) for slicing."""
        batch_size, seq_len, hidden = 2, 8, 32

        # Input is (T, B, D) format
        inputs_embeds = torch.randn(seq_len, batch_size, hidden)

        pg_collection = MockPGCollection(cp_size=2, cp_rank=0)

        captured_batch = {}

        def mock_get_batch(batch_dict, cp_group=None):
            captured_batch.update(batch_dict)
            # Check that decoder_input is in (B, T, D) format
            di = batch_dict.get("decoder_input")
            if di is not None:
                assert di.shape == (batch_size, seq_len, hidden), f"Expected (B,T,D), got {di.shape}"
            return batch_dict

        with patch(
            "megatron.core.utils.get_batch_on_this_cp_rank",
            side_effect=mock_get_batch,
        ):
            slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=None,
                loss_mask=None,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=None,  # BSHD format
                pg_collection=pg_collection,
            )

        # Verify decoder_input was captured with correct shape
        assert "decoder_input" in captured_batch
        assert captured_batch["decoder_input"].shape == (batch_size, seq_len, hidden)

    def test_output_transposed_back_to_tbd(self):
        """Test that outputs are transposed back to (T,B,D) format."""
        batch_size, seq_len, hidden = 2, 16, 64
        sliced_seq_len = seq_len // 2

        inputs_embeds = torch.randn(seq_len, batch_size, hidden)

        pg_collection = MockPGCollection(cp_size=2, cp_rank=0)

        def mock_get_batch(batch_dict, cp_group=None):
            result = {}
            for k, v in batch_dict.items():
                if v is not None and isinstance(v, torch.Tensor):
                    if k == "decoder_input":
                        # Return sliced (B, T/2, D)
                        result[k] = v[:, :sliced_seq_len, :]
                    else:
                        result[k] = v
                else:
                    result[k] = v
            return result

        with patch(
            "megatron.core.utils.get_batch_on_this_cp_rank",
            side_effect=mock_get_batch,
        ):
            result = slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=None,
                loss_mask=None,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=None,
                pg_collection=pg_collection,
            )

        out_embeds, *_ = result

        # Output should be in (T, B, D) format
        assert out_embeds.shape == (sliced_seq_len, batch_size, hidden)


class TestSliceBatchForContextParallelEdgeCases:
    """Edge case tests for slice_batch_for_context_parallel."""

    def test_none_inputs_embeds_with_cp_size_gt_1(self):
        """Test handling when inputs_embeds is None but CP is enabled."""
        pg_collection = MockPGCollection(cp_size=2, cp_rank=0)

        def mock_get_batch(batch_dict, cp_group=None):
            return batch_dict

        with patch(
            "megatron.core.utils.get_batch_on_this_cp_rank",
            side_effect=mock_get_batch,
        ):
            result = slice_batch_for_context_parallel(
                inputs_embeds=None,
                labels=torch.randint(0, 100, (1, 8)),
                loss_mask=torch.ones(1, 8),
                position_ids=torch.arange(8).unsqueeze(0),
                attention_mask=None,
                packed_seq_params=None,
                pg_collection=pg_collection,
            )

        out_embeds, *_ = result
        assert out_embeds is None

    def test_non_thd_qkv_format_uses_bshd_path(self):
        """Test that non-THD qkv_format (e.g., 'sbhd') uses BSHD slicing path."""
        batch_size, seq_len, hidden = 1, 8, 16

        inputs_embeds = torch.randn(seq_len, batch_size, hidden)
        labels = torch.randint(0, 50, (batch_size, seq_len))

        # Non-THD format
        packed_seq_params = MockPackedSeqParams(
            cu_seqlens_q=torch.tensor([0, 8]),
            cu_seqlens_kv=torch.tensor([0, 8]),
            qkv_format="sbhd",  # Not THD
        )

        pg_collection = MockPGCollection(cp_size=2, cp_rank=0)

        mock_called = {"get_batch": False}

        def mock_get_batch(batch_dict, cp_group=None):
            mock_called["get_batch"] = True
            return batch_dict

        with patch(
            "megatron.core.utils.get_batch_on_this_cp_rank",
            side_effect=mock_get_batch,
        ):
            slice_batch_for_context_parallel(
                inputs_embeds=inputs_embeds,
                labels=labels,
                loss_mask=None,
                position_ids=None,
                attention_mask=None,
                packed_seq_params=packed_seq_params,
                pg_collection=pg_collection,
            )

        # Verify BSHD path was used (get_batch_on_this_cp_rank called)
        assert mock_called["get_batch"]
