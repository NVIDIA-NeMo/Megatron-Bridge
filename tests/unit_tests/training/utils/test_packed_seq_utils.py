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

import megatron.core
import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams

from megatron.bridge.training.utils.packed_seq_utils import (
    build_uniform_packed_seq_params,
    get_packed_seq_cp_partition_indices,
    get_packed_seq_params,
    get_packed_seq_q_cu_seqlens,
    get_thd_cp_partition_indices,
    repack_mcore_thd_position_ids,
    unpack_mcore_thd_tensor_for_position_ids,
)


def test_get_packed_seq_q_cu_seqlens_prefers_padded_boundaries():
    actual = torch.tensor([0, 6, 14], dtype=torch.int32)
    padded = torch.tensor([0, 8, 16], dtype=torch.int32)

    unpadded, physical = get_packed_seq_q_cu_seqlens(PackedSeqParams(cu_seqlens_q=actual, cu_seqlens_q_padded=padded))
    assert unpadded is actual
    assert physical is padded

    unpadded, physical = get_packed_seq_q_cu_seqlens(PackedSeqParams(cu_seqlens_q=actual))
    assert unpadded is actual
    assert physical is actual


def test_get_thd_cp_partition_indices_rejects_unsupported_mcore(monkeypatch):
    monkeypatch.setattr(megatron.core, "__version__", "0.17.1")

    with pytest.raises(
        RuntimeError,
        match=r"requires Megatron-Core >= 0\.18\.0.*found 0\.17\.1\. Please upgrade Megatron-Core\.",
    ):
        get_thd_cp_partition_indices(
            torch.tensor([0, 8], dtype=torch.int32),
            total_tokens=8,
            cp_group=object(),
            device=torch.device("cpu"),
        )


def test_get_thd_cp_partition_indices_rejects_pre_feature_mcore_snapshot(monkeypatch):
    def old_get_batch_on_this_cp_rank(batch, cp_group=None):
        raise AssertionError("Unsupported MCore API should not be called.")

    monkeypatch.setattr(megatron.core, "__version__", "0.18.0+oldhash")
    monkeypatch.setattr(
        "megatron.bridge.training.utils.packed_seq_utils.get_batch_on_this_cp_rank",
        old_get_batch_on_this_cp_rank,
    )

    with pytest.raises(
        RuntimeError,
        match=r"requires Megatron-Core >= 0\.18\.0.*found 0\.18\.0\+oldhash\. Please upgrade Megatron-Core\.",
    ):
        get_thd_cp_partition_indices(
            torch.tensor([0, 8], dtype=torch.int32),
            total_tokens=8,
            cp_group=object(),
            device=torch.device("cpu"),
        )


def test_get_packed_seq_cp_partition_indices_uses_padded_boundaries(monkeypatch):
    monkeypatch.setattr(megatron.core, "__version__", "0.18.0")
    actual = torch.tensor([0, 6, 14], dtype=torch.int32)
    padded = torch.tensor([0, 8, 16], dtype=torch.int32)
    seen = {}

    class FakeProcessGroup:
        @staticmethod
        def size():
            return 4

        @staticmethod
        def rank():
            return 0

    cp_group = FakeProcessGroup()

    def fake_get_batch_on_this_cp_rank(batch, *, is_hybrid_cp, cp_group):
        seen["batch"] = batch
        seen["is_hybrid_cp"] = is_hybrid_cp
        seen["cp_group"] = cp_group
        return {**batch, "tokens": torch.tensor([[0, 1, 14, 15]], dtype=torch.int64)}

    monkeypatch.setattr(
        "megatron.bridge.training.utils.packed_seq_utils.get_batch_on_this_cp_rank",
        fake_get_batch_on_this_cp_rank,
    )

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=actual,
        cu_seqlens_kv=actual,
        cu_seqlens_q_padded=padded,
        cu_seqlens_kv_padded=padded,
    )

    index = get_packed_seq_cp_partition_indices(
        packed_seq_params,
        total_tokens=16,
        cp_size=4,
        cp_rank=0,
        device=torch.device("cpu"),
        cp_group=cp_group,
    )

    assert torch.equal(seen["batch"]["cu_seqlens"], padded.unsqueeze(0))
    assert seen["batch"]["tokens"].shape == (1, 16)
    assert seen["is_hybrid_cp"] is False
    assert seen["cp_group"] is cp_group
    assert torch.equal(index, torch.tensor([0, 1, 14, 15], dtype=torch.long))


def test_get_packed_seq_cp_partition_indices_rejects_group_rank_mismatch():
    class FakeProcessGroup:
        @staticmethod
        def size():
            return 2

        @staticmethod
        def rank():
            return 1

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 8], dtype=torch.int32),
    )

    with pytest.raises(ValueError, match="rank 1 and size 2"):
        get_packed_seq_cp_partition_indices(
            packed_seq_params,
            total_tokens=8,
            cp_size=2,
            cp_rank=0,
            device=torch.device("cpu"),
            cp_group=FakeProcessGroup(),
        )


def test_unpack_and_repack_mcore_thd_position_rows():
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 5], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 4, 8], dtype=torch.int32),
    )
    packed_tokens = torch.tensor([[10, 11, 0, 0, 20, 21, 22, 0]])

    rows, attention_mask, padded_starts, lengths = unpack_mcore_thd_tensor_for_position_ids(
        packed_tokens, packed_seq_params
    )

    assert rows.tolist() == [[10, 11, 0], [20, 21, 22]]
    assert attention_mask.tolist() == [[True, True, False], [True, True, True]]
    assert padded_starts == [0, 4]
    assert lengths == [2, 3]

    row_position_ids = torch.tensor(
        [
            [[0, 1, 0], [0, 1, 2]],
            [[0, 1, 0], [10, 11, 12]],
            [[0, 1, 0], [20, 21, 22]],
        ]
    )
    packed_position_ids = repack_mcore_thd_position_ids(
        row_position_ids,
        padded_starts=padded_starts,
        lengths=lengths,
        total_length=packed_tokens.size(1),
    )

    assert packed_position_ids.tolist() == [
        [[0, 1, 0, 0, 0, 1, 2, 0]],
        [[0, 1, 0, 0, 10, 11, 12, 0]],
        [[0, 1, 0, 0, 20, 21, 22, 0]],
    ]


class TestGetPackedSeqParams:
    """Test suite for get_packed_seq_params function."""

    def test_current_mcore_metadata_fields(self):
        """Test get_packed_seq_params with current MCore metadata field names."""
        batch = {
            "cu_seqlens_q": torch.IntTensor([0, 120, 245, 370]),
            "cu_seqlens_kv": torch.IntTensor([0, 120, 245, 370]),
            "cu_seqlens_q_padded": torch.IntTensor([0, 128, 256, 384]),
            "cu_seqlens_kv_padded": torch.IntTensor([0, 128, 256, 384]),
            "max_seqlen_q": torch.tensor(128),
            "max_seqlen_kv": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)

        torch.testing.assert_close(result.cu_seqlens_q, batch["cu_seqlens_q"])
        torch.testing.assert_close(result.cu_seqlens_kv, batch["cu_seqlens_kv"])
        torch.testing.assert_close(result.cu_seqlens_q_padded, batch["cu_seqlens_q_padded"])
        torch.testing.assert_close(result.cu_seqlens_kv_padded, batch["cu_seqlens_kv_padded"])
        assert result.max_seqlen_q == 128
        assert result.max_seqlen_kv == 128
        assert result.qkv_format == "thd"

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

    def test_total_tokens_generates_seq_idx(self):
        """Test that passing total_tokens causes PackedSeqParams to generate seq_idx.

        This is critical for hybrid SSM/Mamba models in varlen (packed sequence)
        settings. Without total_tokens, seq_idx remains None and SSM state bleeds
        across sequence boundaries.
        """
        batch = {
            "cu_seqlens": torch.IntTensor([0, 5, 7, 11, -1]),
            "cu_seqlens_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(6),
            "total_tokens": 16,
        }

        result = get_packed_seq_params(batch)

        assert result.total_tokens == 16
        assert result.seq_idx is not None
        # seq_idx maps each token position to its sequence index:
        # seq 0: tokens 0-4 (len 5), seq 1: tokens 5-6 (len 2),
        # seq 2: tokens 7-10 (len 4), seq 3: tokens 11-15 (len 5)
        expected_seq_idx = torch.IntTensor([[0, 0, 0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3]])
        torch.testing.assert_close(result.seq_idx, expected_seq_idx)

    def test_without_total_tokens_seq_idx_is_none(self):
        """Test that omitting total_tokens leaves seq_idx as None (backward compat)."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, -1]),
            "cu_seqlens_argmin": torch.tensor(3),
            "max_seqlen": torch.tensor(128),
        }

        result = get_packed_seq_params(batch)

        assert result.total_tokens is None
        assert result.seq_idx is None

    def test_total_tokens_with_cu_seqlens_unpadded(self):
        """Test total_tokens flows through when cu_seqlens_unpadded is present."""
        batch = {
            "cu_seqlens": torch.IntTensor([0, 128, 256, 384, -1]),
            "cu_seqlens_argmin": torch.tensor(4),
            "cu_seqlens_unpadded": torch.IntTensor([0, 120, 245, 370, -1]),
            "cu_seqlens_unpadded_argmin": torch.tensor(4),
            "max_seqlen": torch.tensor(128),
            "total_tokens": 384,
        }

        result = get_packed_seq_params(batch)

        assert result.total_tokens == 384
        # seq_idx should be generated from cu_seqlens_q_padded (the padded variant)
        assert result.seq_idx is not None

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


def test_build_uniform_packed_seq_params_uses_padded_batch_lengths():
    result = build_uniform_packed_seq_params(batch_size=3, seq_length=8, device=torch.device("cpu"))

    expected = torch.IntTensor([0, 8, 16, 24])
    torch.testing.assert_close(result.cu_seqlens_q, expected)
    torch.testing.assert_close(result.cu_seqlens_kv, expected)
    torch.testing.assert_close(result.cu_seqlens_q_padded, expected)
    torch.testing.assert_close(result.cu_seqlens_kv_padded, expected)
    assert result.max_seqlen_q == 8
    assert result.max_seqlen_kv == 8
    assert result.qkv_format == "thd"
