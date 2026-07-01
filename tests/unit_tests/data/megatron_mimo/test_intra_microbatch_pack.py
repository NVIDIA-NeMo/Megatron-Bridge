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

import pytest
import torch

from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import assemble_packed_sequence


def _padded(rows, S, pad):
    out = torch.full((len(rows), S), pad, dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.tensor(r, dtype=torch.long)
    return out


@pytest.mark.unit
class TestAssemblePackedSequence:
    def test_cu_seqlens_and_concat(self):
        rows = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]  # lengths 3,2,4
        lengths = [3, 2, 4]
        tokens = _padded(rows, S=8, pad=0)
        out = assemble_packed_sequence([0, 1, 2], tokens, lengths, pad_token_id=0)
        assert out["cu_seqlens"].tolist() == [0, 3, 5, 9]
        assert out["cu_seqlens"].dtype == torch.int32
        assert out["input_ids"].shape == (1, 9)
        assert out["input_ids"][0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert out["max_seqlen"] == 4

    def test_roundtrip_recovers_samples(self):
        rows = [[10, 11], [20, 21, 22], [30]]
        lengths = [2, 3, 1]
        tokens = _padded(rows, S=5, pad=0)
        group = [2, 0, 1]  # arbitrary order
        out = assemble_packed_sequence(group, tokens, lengths)
        cu = out["cu_seqlens"].tolist()
        rec = [out["input_ids"][0, cu[k] : cu[k + 1]].tolist() for k in range(len(group))]
        assert rec == [rows[i] for i in group]  # order preserved, only real tokens

    def test_labels_loss_mask_position_ids(self):
        rows = [[1, 2], [3, 4, 5]]
        lengths = [2, 3]
        tokens = _padded(rows, 4, 0)
        labels = _padded([[1, 1], [2, 2, 2]], 4, -100)
        loss_mask = _padded([[1, 1], [1, 1, 0]], 4, 0)
        pos = _padded([[0, 1], [0, 1, 2]], 4, 0)
        out = assemble_packed_sequence([0, 1], tokens, lengths, labels=labels, loss_mask=loss_mask, position_ids=pos)
        assert out["labels"][0].tolist() == [1, 1, 2, 2, 2]
        assert out["loss_mask"][0].tolist() == [1, 1, 1, 1, 0]
        assert out["position_ids"][0].tolist() == [0, 1, 0, 1, 2]  # positions reset per sample

    def test_mrope_3d_position_ids_now_packed(self):
        # MRoPE [3, B, S] is now packed to [3, 1, T] (capability added; previously raised).
        tokens = _padded([[1, 2]], 2, 0)
        pos = torch.arange(3 * 1 * 2).reshape(3, 1, 2)  # [3, B=1, S=2]
        out = assemble_packed_sequence([0], tokens, [2], position_ids=pos)
        assert out["position_ids"].shape == (3, 1, 2)
        torch.testing.assert_close(out["position_ids"], pos[:, 0, :2].unsqueeze(1))

    def test_all_zero_lengths_raise(self):
        tokens = torch.zeros((2, 4), dtype=torch.long)
        with pytest.raises(ValueError, match="empty packed shard"):
            assemble_packed_sequence([0, 1], tokens, [0, 0])


@pytest.mark.unit
class TestMRoPEAndPackLanguageShard:
    def test_assemble_handles_mrope_position_ids(self):
        import torch

        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import assemble_packed_sequence

        tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # lengths 3, 2
        # MRoPE [3, B, S]: distinct per-axis positions
        pos = torch.arange(3 * 2 * 4).reshape(3, 2, 4)
        out = assemble_packed_sequence([0, 1], tokens, [3, 2], position_ids=pos)
        assert out["position_ids"].shape == (3, 1, 5)  # T = 3 + 2
        # sample0 real [3,:3] then sample1 real [3,:2] concatenated along T
        expected = torch.cat([pos[:, 0, :3], pos[:, 1, :2]], dim=1).unsqueeze(1)
        torch.testing.assert_close(out["position_ids"], expected)

    def test_assemble_rejects_bad_3d(self):
        import torch

        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import assemble_packed_sequence

        with pytest.raises(ValueError):
            assemble_packed_sequence([0], torch.tensor([[1, 2]]), [2], position_ids=torch.zeros(4, 1, 2))

    def test_pack_language_shard(self):
        import torch

        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),  # real 3, 2 -> T=5
            "labels": torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]]),
            "loss_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
            "position_ids": torch.arange(3 * 2 * 4).reshape(3, 2, 4),  # MRoPE
            "modality_inputs": {"images": "carry"},
        }
        packed, kw = pack_language_shard(batch, pad_token_id=0, lengths=torch.tensor([3, 2]))
        assert packed["input_ids"].shape == (1, 5)
        assert packed["input_ids"].tolist() == [[1, 2, 3, 4, 5]]
        assert packed["labels"].tolist() == [[1, 2, 3, 4, 5]]
        assert packed["loss_mask"].tolist() == [[1, 1, 1, 1, 1]]
        assert packed["position_ids"].shape == (3, 1, 5)
        assert packed["attention_mask"] is None
        assert packed["modality_inputs"] == {"images": "carry"}  # carried through
        # packing_kwargs: cu_seqlens block-diagonal boundaries, int handling
        assert kw["cu_seqlens_q"].tolist() == [0, 3, 5]
        assert kw["cu_seqlens_kv"].tolist() == [0, 3, 5]
        assert kw["max_seqlen_q"] == 3 and kw["max_seqlen_kv"] == 3

    def test_pack_language_shard_noop_without_input_ids(self):
        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        b = {"labels": None}
        out, kw = pack_language_shard(b, pad_token_id=0)
        assert out is b and kw is None

    def test_pack_language_shard_stage_gt0_packs_with_input_ids_nulled(self):
        # Bug #2 (packing + PP>1): on language PP stages > 0 ``input_ids`` is nulled, but the caller
        # still supplies ``lengths`` (derived from ``input_ids`` before nulling, via the batch_spec
        # that keeps input_ids on every language stage when packing). The shard must still pack
        # ``position_ids`` (MRoPE [3,B,S]) / ``labels`` / ``loss_mask`` to the SAME [1,T] so the THD
        # rotary on the receiving stage is sized to T, not the dense seq_length.
        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        batch = {
            "input_ids": None,  # nulled on the non-first PP stage
            "labels": torch.arange(2 * 4).reshape(2, 4),
            "loss_mask": torch.ones(2, 4),
            "position_ids": torch.arange(3 * 2 * 4).reshape(3, 2, 4),  # MRoPE [3, B=2, S=4]
        }
        packed, kw = pack_language_shard(batch, pad_token_id=0, lengths=torch.tensor([3, 2]))
        T = 5  # 3 + 2
        assert kw is not None
        assert kw["cu_seqlens_q"].tolist() == [0, 3, T]
        assert kw["cu_seqlens_q"][-1].item() == T  # cu_seqlens[-1] == packed length
        assert packed["position_ids"].shape == (3, 1, T)  # MRoPE packed to [3, 1, T]
        assert packed["labels"].shape == (1, T)
        assert packed["loss_mask"].shape == (1, T)
        assert packed["attention_mask"] is None  # block-diagonal attention via cu_seqlens
        assert packed["input_ids"] is None  # stays nulled (no input_ids to pack on this stage)


@pytest.mark.unit
class TestPadTokenIdAndLengthSource:
    """F4: the caller-provided lengths (priority-0) drive the tight pack regardless of pad id."""

    def test_pack_respects_nonzero_pad_token_id(self):
        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        pad = 248044
        # Two samples padded with a non-zero pad id; real lengths 3 and 2.
        input_ids = torch.tensor([[1, 2, 3, pad, pad], [4, 5, pad, pad, pad]])
        packed, kw = pack_language_shard({"input_ids": input_ids}, pad_token_id=pad, lengths=torch.tensor([3, 2]))
        # cu boundaries are the REAL lengths (3, 2), not S=5.
        assert kw["cu_seqlens_q"].tolist() == [0, 3, 5]
        # THD input_ids contain no pad id, total == sum(real lengths).
        assert pad not in packed["input_ids"].flatten().tolist()
        assert packed["input_ids"].shape == (1, 5)
        assert packed["input_ids"].tolist() == [[1, 2, 3, 4, 5]]

    def test_packing_kwargs_has_padded_cu_seqlens(self):
        # F9 / CP>1: padded cu_seqlens populated (coincide with unpadded; tight pack).
        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])  # real 3, 2
        _, kw = pack_language_shard({"input_ids": input_ids}, pad_token_id=0, lengths=torch.tensor([3, 2]))
        for k in ("cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
            assert k in kw
            assert kw[k].dtype == torch.int32
            assert kw[k].tolist() == [0, 3, 5]
            assert len(kw[k]) == 3  # len(group)+1
        torch.testing.assert_close(kw["cu_seqlens_q_padded"], kw["cu_seqlens_q"])
        torch.testing.assert_close(kw["cu_seqlens_kv_padded"], kw["cu_seqlens_kv"])


@pytest.mark.unit
class TestPPConsistentPacking:
    """F5: input_ids and labels/loss_mask pack to the same [1, T] from the caller-provided lengths."""

    def test_labels_packed_consistently_with_logits_shape(self):
        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        # The caller derives lengths from input_ids once and reuses them so input_ids (logits)
        # and labels/loss_mask pack to an identical [1, T].
        lengths = torch.tensor([3, 2])  # real 3, 2 -> T = 5

        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        first, kw_first = pack_language_shard({"input_ids": input_ids}, pad_token_id=0, lengths=lengths)
        first_T = first["input_ids"].shape[1]

        labels = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]])
        loss_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        last, kw_last = pack_language_shard(
            {"input_ids": None, "labels": labels, "loss_mask": loss_mask},
            pad_token_id=0,
            lengths=lengths,
        )
        assert last["labels"].shape == (1, first_T)
        assert last["loss_mask"].shape == (1, first_T)
        assert kw_last["cu_seqlens_q"].tolist() == kw_first["cu_seqlens_q"].tolist()
        assert last["labels"].tolist() == [[1, 2, 3, 4, 5]]

    def test_explicit_lengths_validate_position_ids_only_batch_dim(self):
        from megatron.bridge.data.megatron_mimo.intra_microbatch_pack import pack_language_shard

        pos = torch.arange(3 * 3 * 4).reshape(3, 3, 4)
        with pytest.raises(ValueError, match="expected \\[3\\]"):
            pack_language_shard({"input_ids": None, "position_ids": pos}, lengths=torch.tensor([4, 4]))
