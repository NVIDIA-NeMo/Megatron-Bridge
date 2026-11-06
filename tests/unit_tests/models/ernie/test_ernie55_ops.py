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

"""CPU tests of the ERNIE 5.5 D-layer tensor helpers: index geometry, VHA, output gate, sink gradient."""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant import csa, dsa

from megatron.bridge.models.ernie.modeling_ernie55 import ops


pytestmark = pytest.mark.unit

W, K = 3, 5
# Documents shorter than, equal to and longer than the window and the dense-regime limit W + K.
DOC_LENS = [1, W - 1, W, W + 1, W + K - 1, W + K, 3 * (W + K)]
TAIL_ROWS = 3


def _packed(doc_lens, *, padded_lens=None):
    """Packed THD metadata for ``doc_lens``; ``padded_lens`` adds ``cu_seqlens_*_padded``."""

    def cumsum(lengths):
        return torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)

    cu = cumsum(doc_lens)
    cu_padded = cumsum(padded_lens) if padded_lens is not None else None
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )


def _brute_doc_starts(boundaries, num_rows):
    """Row -> start of the document that owns it; rows at or past the last boundary join the last document."""
    starts = []
    for row in range(num_rows):
        if row >= boundaries[-1]:
            starts.append(boundaries[-2])
            continue
        owner = next(doc for doc in range(len(boundaries) - 1) if boundaries[doc] <= row < boundaries[doc + 1])
        starts.append(boundaries[owner])
    return starts


def _geometry(doc_start, window=W):
    starts, ends = ops.ernie55_selection_bounds(doc_start, window)
    return starts, ends, ops.ernie55_window_indices(doc_start, window_size=window, batch_size=1)[0]


class TestDocumentStarts:
    def test_unpadded_multidoc_with_tail_rows(self):
        psp = _packed(DOC_LENS)
        num_rows = sum(DOC_LENS) + TAIL_ROWS
        starts = ops.ernie55_document_starts(psp, num_rows, "cpu")
        assert starts.dtype == torch.int64 and starts.shape == (num_rows,)
        assert starts.tolist() == _brute_doc_starts(psp.cu_seqlens_q.tolist(), num_rows)
        last_start = psp.cu_seqlens_q[-2].item()
        assert starts[-TAIL_ROWS:].tolist() == [last_start] * TAIL_ROWS

    def test_padded_cu_seqlens_take_precedence(self):
        padded = [4 * math.ceil(length / 4) for length in DOC_LENS]
        psp = _packed(DOC_LENS, padded_lens=padded)
        num_rows = sum(padded)
        starts = ops.ernie55_document_starts(psp, num_rows, "cpu")
        assert starts.tolist() == _brute_doc_starts(psp.cu_seqlens_q_padded.tolist(), num_rows)
        assert starts.tolist() != _brute_doc_starts(psp.cu_seqlens_q.tolist(), num_rows)

    def test_empty_documents_are_skipped(self):
        cu = torch.tensor([0, 2, 2, 5, 5, 6], dtype=torch.int32)
        psp = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=3, max_seqlen_kv=3)
        assert ops.ernie55_document_starts(psp, 8, "cpu").tolist() == [0, 0, 2, 2, 2, 5, 5, 5]

    @pytest.mark.parametrize("packed", [None, "sbhd"])
    def test_without_thd_every_sample_is_one_document(self, packed):
        psp = None if packed is None else PackedSeqParams(qkv_format="sbhd")
        assert ops.ernie55_document_starts(psp, 7, "cpu").tolist() == [0] * 7


class TestSelectionGeometry:
    @pytest.mark.parametrize("padded", [False, True])
    def test_bounds_and_window_match_brute_force(self, padded):
        padded_lens = [length + 2 for length in DOC_LENS] if padded else None
        psp = _packed(DOC_LENS, padded_lens=padded_lens)
        num_rows = sum(padded_lens or DOC_LENS) + TAIL_ROWS
        doc_start = ops.ernie55_document_starts(psp, num_rows, "cpu")
        starts, ends, window = _geometry(doc_start)
        assert starts.dtype == ends.dtype == window.dtype == torch.int32
        assert window.shape == (num_rows, W)
        for t in range(num_rows):
            d = doc_start[t].item()
            candidates = list(range(d, t - W + 1))
            window_keys = list(range(max(d, t - W + 1), t + 1))
            assert starts[t].item() == d
            assert ends[t].item() == max(d, t - W + 1)
            assert list(range(starts[t].item(), ends[t].item())) == candidates
            assert window[t].tolist() == window_keys + [-1] * (W - len(window_keys))
            assert not set(candidates) & set(window_keys)
            assert sorted(candidates + window_keys) == list(range(d, t + 1))
            assert all(d <= key <= t for key in candidates + window_keys)

    def test_dense_regime_limit(self):
        # A row needs the indexer only once it has more than K remote candidates: pos > W + K - 1.
        psp = _packed([4 * (W + K)])
        doc_start = ops.ernie55_document_starts(psp, 4 * (W + K), "cpu")
        starts, ends, _ = _geometry(doc_start)
        num_candidates = (ends - starts).tolist()
        for pos, count in enumerate(num_candidates):
            assert count == max(0, pos - W + 1)
            assert (count <= K) == (pos <= W + K - 1)

    def test_window_is_shared_across_samples(self):
        doc_start = torch.zeros(6, dtype=torch.int64)
        window = ops.ernie55_window_indices(doc_start, window_size=W, batch_size=3)
        assert window.shape == (3, 6, W)
        assert torch.equal(window[0], window[1]) and torch.equal(window[0], window[2])
        assert window[0, 0].tolist() == [0, -1, -1]

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            ops.ernie55_selection_bounds(torch.zeros(3, dtype=torch.int64), 0)
        with pytest.raises(ValueError, match="window_size"):
            ops.ernie55_window_indices(torch.zeros(3, dtype=torch.int64), window_size=0, batch_size=1)


class TestNonEmptyKernelBounds:
    """Kernels never see an empty candidate row; the placeholder never reaches attention."""

    def test_kernel_ends(self):
        psp = _packed(DOC_LENS)
        num_rows = sum(DOC_LENS) + TAIL_ROWS
        doc_start = ops.ernie55_document_starts(psp, num_rows, "cpu")
        starts, ends, _ = _geometry(doc_start)
        kernel_ends = ops.ernie55_kernel_selection_ends(starts=starts, ends=ends)
        assert kernel_ends.dtype == torch.int32
        assert bool((kernel_ends > starts).all())
        nonempty = ends > starts
        assert torch.equal(kernel_ends[nonempty], ends[nonempty])
        # Empty rows get exactly one causal placeholder: their document start.
        assert torch.equal(kernel_ends[~nonempty], starts[~nonempty] + 1)
        rows = torch.arange(num_rows, dtype=torch.int32)
        assert bool((starts[~nonempty] <= rows[~nonempty]).all())

    def test_mask_topk_beyond_end(self):
        ends = torch.tensor([0, 0, 2, 5], dtype=torch.int32)
        topk = torch.tensor([[[0, -1, -1], [0, -1, -1], [1, 0, -1], [4, 2, 3]]], dtype=torch.int32)
        masked = ops.mask_topk_beyond_end(topk, ends=ends)
        assert masked.dtype == torch.int32
        assert masked.tolist() == [[[-1, -1, -1], [-1, -1, -1], [1, 0, -1], [4, 2, 3]]]
        with pytest.raises(ValueError, match="Expected topk"):
            ops.mask_topk_beyond_end(topk[0], ends=ends)

    def test_reference_selector_and_loss_on_kernel_bounds(self):
        """Megatron-Core's reference top-k/KL on the widened bounds give ERNIE's exact selection."""
        doc_lens = [2, 3 * (W + K), W + 1]
        psp = _packed(doc_lens)
        num_rows = sum(doc_lens)
        doc_start = ops.ernie55_document_starts(psp, num_rows, "cpu")
        starts, ends, _ = _geometry(doc_start)
        kernel_ends = ops.ernie55_kernel_selection_ends(starts=starts, ends=ends)

        # One indexer head with positive q.k = s + 1: scores strictly increase with the key index.
        q_indexer = torch.ones(num_rows, 1, 1, 1)
        k_indexer = torch.arange(1, num_rows + 1, dtype=torch.float32).view(num_rows, 1, 1)
        weights = torch.ones(num_rows, 1, 1)
        index_scores, topk = dsa.fused_qk_topk_naive(
            q_indexer, k_indexer, weights, K, varlen_starts=starts, varlen_ends=kernel_ends, use_relu=True
        )
        selected = ops.mask_topk_beyond_end(ops.pad_topk(topk, K), ends=ends)
        for t in range(num_rows):
            candidates = list(range(starts[t].item(), ends[t].item()))
            expected = sorted(candidates)[-K:]
            got = [i for i in selected[0, t].tolist() if i >= 0]
            assert sorted(got) == expected, t
            assert selected[0, t].tolist().count(-1) == K - len(expected)

        heads, dim = 2, 4
        query = torch.randn(num_rows, 1, heads, dim)
        key = torch.randn(num_rows, 1, heads, dim)
        # compute_dsa_indexer_loss only asks the TP group for its size.
        pg_collection = SimpleNamespace(tp=SimpleNamespace(size=lambda: 1))
        loss = dsa.compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk,
            query=query,
            key=key,
            softmax_scale=dim**-0.5,
            loss_coeff=1.0,
            sparse_loss=True,
            pg_collection=pg_collection,
            varlen_starts=starts,
            varlen_ends=kernel_ends,
            calculate_per_token_loss=True,
        )
        assert torch.isfinite(loss)
        placeholder_rows = (ends == starts).unsqueeze(0)
        placeholder_loss = dsa.compute_dsa_indexer_loss(
            index_scores=index_scores,
            topk_indices=topk,
            query=query,
            key=key,
            softmax_scale=dim**-0.5,
            loss_coeff=1.0,
            sparse_loss=True,
            pg_collection=pg_collection,
            varlen_starts=starts,
            varlen_ends=kernel_ends,
            query_valid_rows=placeholder_rows,
            calculate_per_token_loss=True,
        )
        assert placeholder_loss.item() == 0.0


class TestPadTopk:
    def test_pads_with_minus_one(self):
        topk = torch.tensor([[[3, 1], [2, -1]]], dtype=torch.int64)
        padded = ops.pad_topk(topk, 4)
        assert padded.dtype == torch.int32
        assert padded.tolist() == [[[3, 1, -1, -1], [2, -1, -1, -1]]]
        assert ops.pad_topk(topk, 2).tolist() == topk.tolist()

    def test_rejects_wider_input(self):
        with pytest.raises(ValueError, match="Cannot pad"):
            ops.pad_topk(torch.zeros(1, 1, 3, dtype=torch.int32), 2)


class TestPackEnd:
    """The host value of a pack's last boundary, noted by the batch builder or read once from the tensor."""

    def test_noted_value_is_returned_without_reading_the_tensor(self, monkeypatch):
        cu = torch.tensor([0, 3, 9], dtype=torch.int32)
        params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=6, max_seqlen_kv=6)
        ops.record_pack_end(params, 9)
        monkeypatch.setattr(torch.Tensor, "__int__", lambda self: pytest.fail("read the device tensor"))
        assert ops.pack_end(params) == 9

    def test_unnoted_or_replaced_boundaries_are_read_once(self):
        cu = torch.tensor([0, 3, 9], dtype=torch.int32)
        params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=6, max_seqlen_kv=6)
        assert ops.pack_end(params) == 9
        ops.record_pack_end(params, 9)
        replaced = torch.tensor([0, 3, 12], dtype=torch.int32)
        params.cu_seqlens_q = params.cu_seqlens_kv = replaced
        assert ops.pack_end(params) == 12


class TestDropInvalidKeyBlocks:
    """The CP-1 FlashMLA key lists: the reference's rows without their all-invalid key blocks."""

    @pytest.mark.parametrize("width, block", [(12, 4), (13, 4), (2176, 128), (2176, 64)])
    def test_keeps_every_non_empty_block_in_order(self, width, block):
        generator = torch.Generator().manual_seed(width + block)
        rows = 9
        flat = torch.full((rows, width), -1, dtype=torch.int32)
        for row in range(1, rows):  # row 0 holds no valid key
            valid = torch.rand(width, generator=generator) < (row / rows) ** 3
            flat[row, valid] = torch.randint(0, 10_000, (int(valid.sum()),), generator=generator, dtype=torch.int32)
        kept, length = ops.drop_invalid_key_blocks(flat, block=block)
        padded_width = width + (-width % block)
        assert kept.dtype == length.dtype == torch.int32 and kept.shape == (rows, padded_width)
        blocks = torch.nn.functional.pad(flat, (0, padded_width - width), value=-1).view(rows, -1, block)
        for row in range(rows):
            non_empty = [chunk for chunk in blocks[row] if bool((chunk >= 0).any())]
            expected = torch.cat(non_empty) if non_empty else torch.empty(0, dtype=torch.int32)
            assert int(length[row]) == expected.numel() == len(non_empty) * block
            assert torch.equal(kept[row, : expected.numel()], expected), row
            assert bool((kept[row, expected.numel() :] == -1).all()), row

    def test_reference_rows(self):
        """``[top-k (score order, -1 after) | window]``: the top-k's non-empty blocks, then the window's."""
        topk, window = torch.full((2, 8), -1, dtype=torch.int32), torch.full((2, 4), -1, dtype=torch.int32)
        topk[0, :5] = torch.tensor([40, 7, 33, 12, 2])
        window[0] = torch.tensor([50, 51, 52, 53])
        window[1, :2] = torch.tensor([0, 1])  # a document's second row: no remote key, a partial window
        kept, length = ops.drop_invalid_key_blocks(torch.cat([topk, window], dim=-1), block=4)
        assert length.tolist() == [12, 4]
        assert kept[0].tolist() == [40, 7, 33, 12, 2, -1, -1, -1, 50, 51, 52, 53]
        assert kept[1].tolist() == [0, 1] + [-1] * 10

    def test_rejects_bad_arguments(self):
        with pytest.raises(ValueError, match="rows, width"):
            ops.drop_invalid_key_blocks(torch.zeros(1, 2, 3, dtype=torch.int32), block=4)
        with pytest.raises(ValueError, match="positive"):
            ops.drop_invalid_key_blocks(torch.zeros(2, 3, dtype=torch.int32), block=0)


def _vha_fp64(out, u, v):
    mix = torch.eye(out.size(-2), dtype=torch.float64) + v.double() @ u.double().t()
    return torch.einsum("kh,...hd->...kd", mix, out.double())


class TestVHAPostmix:
    @pytest.mark.parametrize("bf16_mix", [True, False])
    def test_identity_when_v_is_zero(self, bf16_mix):
        out = torch.randn(5, 2, 4, 3)
        u = torch.randn(4, 2)
        assert torch.equal(ops.vha_postmix(out, u=u, v=torch.zeros(4, 2), bf16_mix=bf16_mix), out)

    @pytest.mark.parametrize("bf16_mix", [True, False])
    def test_orientation(self, bf16_mix):
        # R = 1, U = e_0, V = e_1: M = I + e_1 e_0^T, so out'_1 = out_1 + out_0 and the rest is unchanged.
        out = torch.randn(3, 1, 4, 5, dtype=torch.float64)
        u = torch.zeros(4, 1, dtype=torch.float64)
        v = torch.zeros(4, 1, dtype=torch.float64)
        u[0, 0] = 1.0
        v[1, 0] = 1.0
        mixed = ops.vha_postmix(out, u=u, v=v, bf16_mix=bf16_mix)
        assert torch.equal(mixed[..., 1, :], out[..., 1, :] + out[..., 0, :])
        for head in (0, 2, 3):
            assert torch.equal(mixed[..., head, :], out[..., head, :])

    @pytest.mark.parametrize("bf16_mix", [True, False])
    def test_matches_fp64_formula(self, bf16_mix):
        torch.manual_seed(0)
        out = torch.randn(6, 2, 8, 5, dtype=torch.float64)
        u, v = torch.randn(8, 3, dtype=torch.float64), torch.randn(8, 3, dtype=torch.float64)
        torch.testing.assert_close(ops.vha_postmix(out, u=u, v=v, bf16_mix=bf16_mix), _vha_fp64(out, u, v))

    def test_bf16_contracts_agree_within_bf16(self):
        torch.manual_seed(1)
        out = torch.randn(16, 1, 8, 32).to(torch.bfloat16)
        u, v = 0.3 * torch.randn(8, 4), 0.3 * torch.randn(8, 4)
        bf16_mix = ops.vha_postmix(out, u=u, v=v, bf16_mix=True)
        fp32_mix = ops.vha_postmix(out, u=u, v=v, bf16_mix=False)
        assert bf16_mix.dtype == fp32_mix.dtype == torch.bfloat16
        reference = _vha_fp64(out, u, v)
        scale = reference.abs().max().item()
        assert (fp32_mix.double() - reference).abs().max().item() <= 2**-8 * scale
        assert (bf16_mix.double() - reference).abs().max().item() <= 2**-6 * scale

    def test_fp32_mix_rounds_once(self):
        torch.manual_seed(2)
        out = torch.randn(4, 1, 8, 16).to(torch.bfloat16)
        u, v = torch.randn(8, 2), torch.randn(8, 2)
        expected = (out.float() + torch.einsum("kh,...hd->...kd", v @ u.t(), out.float())).to(torch.bfloat16)
        assert torch.equal(ops.vha_postmix(out, u=u, v=v, bf16_mix=False), expected)

    @pytest.mark.parametrize("bf16_mix", [True, False])
    def test_gradcheck(self, bf16_mix):
        torch.manual_seed(3)
        out = torch.randn(3, 1, 4, 2, dtype=torch.float64, requires_grad=True)
        u = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        v = torch.randn(4, 2, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda o, a, b: ops.vha_postmix(o, u=a, v=b, bf16_mix=bf16_mix), (out, u, v))

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="Expected u and v"):
            ops.vha_postmix(torch.randn(2, 4, 3), u=torch.randn(3, 2), v=torch.randn(3, 2), bf16_mix=True)


class TestSigmoidOutputGate:
    def test_matches_fp64(self):
        torch.manual_seed(4)
        out, logits = torch.randn(5, 2, 12, dtype=torch.float64), torch.randn(5, 2, 12, dtype=torch.float64)
        expected = out * (1.0 / (1.0 + torch.exp(-logits)))
        torch.testing.assert_close(ops.sigmoid_output_gate(out, gate_logits=logits, fp32=False), expected)
        torch.testing.assert_close(ops.sigmoid_output_gate(out, gate_logits=logits, fp32=True), expected)

    def test_flatten_order(self):
        # out[..., h * D + d] is head h, channel d of the [..., H, D] attention output.
        heads, dim = 3, 4
        per_head = torch.arange(heads * dim, dtype=torch.float32).view(1, 1, heads, dim) + 1.0
        out = per_head.reshape(1, 1, heads * dim)
        head, channel = 2, 1
        logits = torch.full_like(out, -1e4)
        logits[..., head * dim + channel] = 1e4
        gated = ops.sigmoid_output_gate(out, gate_logits=logits, fp32=False).view(1, 1, heads, dim)
        expected = torch.zeros_like(per_head)
        expected[..., head, channel] = per_head[..., head, channel]
        assert torch.equal(gated, expected)

    def test_fp32_mode_rounds_once(self):
        torch.manual_seed(5)
        out = torch.randn(64, 1, 32).to(torch.bfloat16)
        logits = torch.randn(64, 1, 32).to(torch.bfloat16)
        expected = (out.float() * torch.sigmoid(logits.float())).to(torch.bfloat16)
        assert torch.equal(ops.sigmoid_output_gate(out, gate_logits=logits, fp32=True), expected)
        assert ops.sigmoid_output_gate(out, gate_logits=logits, fp32=False).dtype == torch.bfloat16

    def test_gradcheck(self):
        torch.manual_seed(6)
        out = torch.randn(3, 1, 6, dtype=torch.float64, requires_grad=True)
        logits = torch.randn(3, 1, 6, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(
            lambda o, g: ops.sigmoid_output_gate(o, gate_logits=g, fp32=False), (out, logits)
        )

    def test_shape_validation(self):
        with pytest.raises(ValueError, match="Gate shape"):
            ops.sigmoid_output_gate(torch.randn(2, 1, 6), gate_logits=torch.randn(2, 1, 5), fp32=False)


def _sparse_attention_fp64(query, kv, sink, indices, scale, value_dim):
    """Dense-mask fp64 oracle: softmax over ``indices`` with the sink in the denominator only.

    Returns the output over the first ``value_dim`` kv channels and the LSE without the sink.
    """
    valid = indices >= 0
    keys = kv[indices.clamp_min(0)]
    logits = torch.einsum("rhd,rmd->rhm", query, keys) * scale
    logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - torch.logaddexp(lse, sink.view(1, -1)).unsqueeze(-1))
    return torch.einsum("rhm,rmd->rhd", probs, keys[..., :value_dim]), lse


def _random_indices(rows, num_keys, width, generator):
    """Duplicate-free per-row key indices with -1 holes; every row keeps at least one key."""
    indices = torch.full((rows, width), -1, dtype=torch.int64)
    for row in range(rows):
        count = int(torch.randint(1, width + 1, (1,), generator=generator))
        picks = torch.randperm(num_keys, generator=generator)[:count]
        slots = torch.randperm(width, generator=generator)[:count]
        indices[row, slots] = picks
    return indices


class TestAttentionSinkGrad:
    """The analytic sink gradient used by the fused fallback matches autograd."""

    def test_lse_with_sink(self):
        lse = torch.tensor([[0.0, 1.0], [-2.0, 3.0]], dtype=torch.float64)
        sink = torch.tensor([0.5, -1.0], dtype=torch.float64)
        expected = torch.log(torch.exp(lse) + torch.exp(sink))
        torch.testing.assert_close(ops.lse_with_sink(lse, attn_sink=sink), expected)
        assert ops.lse_with_sink(lse.float(), attn_sink=sink.float()).dtype == torch.float32
        assert ops.lse_with_sink(lse.to(torch.bfloat16), attn_sink=sink.float()).dtype == torch.float32

    def test_matches_autograd_fp64(self):
        generator = torch.Generator().manual_seed(7)
        rows, heads, dim, value_dim, width = 9, 4, 6, 4, 5
        query = torch.randn(rows, heads, dim, dtype=torch.float64, generator=generator)
        kv = torch.randn(rows, dim, dtype=torch.float64, generator=generator)
        sink = torch.randn(heads, dtype=torch.float64, generator=generator).requires_grad_(True)
        indices = _random_indices(rows, rows, width, generator)
        out, lse = _sparse_attention_fp64(query, kv, sink, indices, 0.4, value_dim)
        d_out = torch.randn_like(out)
        (expected,) = torch.autograd.grad(out, sink, d_out)
        analytic = ops.attention_sink_grad(attn_sink=sink.detach(), lse=lse.detach(), out=out.detach(), d_out=d_out)
        assert analytic.dtype == torch.float64
        torch.testing.assert_close(analytic, expected)
        fp32_analytic = ops.attention_sink_grad(
            attn_sink=sink.detach().float(), lse=lse.float(), out=out.float(), d_out=d_out.float()
        )
        assert fp32_analytic.dtype == torch.float32
        torch.testing.assert_close(fp32_analytic.double(), expected, rtol=1e-5, atol=1e-6)

    def test_matches_megatron_reference_attention(self):
        """Autograd through Megatron-Core's unfused sparse attention, value sliced to the latent width."""
        generator = torch.Generator().manual_seed(8)
        seqlen, heads, dim, latent = 7, 4, 10, 8
        query = torch.randn(seqlen, 1, heads, dim, generator=generator)
        kv = torch.randn(seqlen, 1, dim, generator=generator)
        sink = torch.randn(heads, generator=generator).requires_grad_(True)
        indices = _random_indices(seqlen, seqlen, 6, generator).unsqueeze(0).to(torch.int32)
        scale = dim**-0.5
        out = csa.unfused_compressed_sparse_attn(query, kv, sink, indices, scale).view(seqlen, 1, heads, dim)
        latent_out = out[..., :latent]
        d_out = torch.randn(seqlen, 1, heads, latent, generator=generator)
        (expected,) = torch.autograd.grad(latent_out, sink, d_out)

        ref_out, lse = _sparse_attention_fp64(
            query[:, 0].double(), kv[:, 0].double(), sink.detach().double(), indices[0].long(), scale, latent
        )
        torch.testing.assert_close(latent_out[:, 0].double(), ref_out, rtol=1e-5, atol=1e-6)
        analytic = ops.attention_sink_grad(
            attn_sink=sink.detach(), lse=lse.float(), out=latent_out.detach()[:, 0], d_out=d_out[:, 0]
        )
        torch.testing.assert_close(analytic, expected, rtol=1e-4, atol=1e-5)

    def test_disabled_sink_has_zero_gradient(self):
        lse = torch.randn(4, 2)
        out, d_out = torch.randn(4, 2, 3), torch.randn(4, 2, 3)
        grad = ops.attention_sink_grad(attn_sink=torch.full((2,), float("-inf")), lse=lse, out=out, d_out=d_out)
        assert torch.equal(grad, torch.zeros(2))


def _valid_slots(topk_idxs, topk_length):
    """Slots a sparse kernel reads: non-negative and, with ``topk_length``, inside each row's prefix."""
    valid = topk_idxs >= 0
    if topk_length is not None:
        positions = torch.arange(topk_idxs.size(-1)).view(1, -1)
        valid = valid & (positions < topk_length.view(-1, 1))
    return valid


def _dense_flash_mla_forward(
    q, kv, topk_idxs, softmax_scale, d_v=512, attn_sink=None, topk_length=None, indexer_topk=0
):
    """CPU stand-in for FlashMLA's sparse forward: output with the sink, LSE without it."""
    sink = attn_sink if attn_sink is not None else torch.full((q.size(1),), float("-inf"), dtype=q.dtype)
    indices = topk_idxs.long().masked_fill(~_valid_slots(topk_idxs, topk_length), -1)
    out, lse = _sparse_attention_fp64(q, kv, sink, indices, softmax_scale, d_v)
    return out.to(q.dtype), lse.to(q.dtype), None


def _dense_sparse_backward(q, kv, out, d_out, lse, attn_sink, topk_idxs, softmax_scale, topk_length, **kwargs):
    """CPU stand-in for cuDNN's sparse backward: softmax backward with ``P = exp(S - logaddexp(lse, sink))``."""
    d_v = out.size(-1)
    lse_total = torch.logaddexp(lse, attn_sink.view(1, -1).to(lse.dtype))
    valid = _valid_slots(topk_idxs, topk_length)
    keys = kv[topk_idxs.clamp_min(0).long()]
    logits = torch.einsum("rhd,rmd->rhm", q, keys) * softmax_scale
    probs = torch.exp(logits.masked_fill(~valid.unsqueeze(1), float("-inf")) - lse_total.unsqueeze(-1))
    delta = (d_out * out).sum(-1, keepdim=True)
    d_logits = probs * (torch.einsum("rhd,rmd->rhm", d_out, keys[..., :d_v]) - delta)
    d_q = torch.einsum("rhm,rmd->rhd", d_logits, keys) * softmax_scale
    d_keys = torch.einsum("rhm,rhd->rmd", d_logits, q) * softmax_scale
    d_keys[..., :d_v] += torch.einsum("rhm,rhd->rmd", probs, d_out)
    d_keys = d_keys * valid.unsqueeze(-1)
    d_kv = torch.zeros_like(kv).index_add_(0, topk_idxs.clamp_min(0).long().flatten(), d_keys.flatten(0, 1))
    d_sink = -(torch.exp(attn_sink.view(1, -1) - lse_total) * delta.squeeze(-1)).sum(0)
    return d_q, d_kv, d_sink


class TestSinkGradFallback:
    """FlashMLA forward with the sink, sink-free backward on ``lse_full`` and analytic ``d_sink``.

    Dense stand-ins replace the GPU kernels, so this checks the decomposition and the kernel
    contract the fallback mirrors from Megatron-Core DSA at ``d_qk = 576`` (compacted indices with
    ``topk_length``, a 512-aligned forward width, non-negative placeholders in the backward);
    the kernels themselves are covered by the GPU tests.
    """

    @pytest.mark.parametrize("layout", ["compact", "blocks"])
    @pytest.mark.parametrize("deterministic", [False, True])
    def test_fallback_gradients_match_autograd(self, monkeypatch, deterministic, layout):
        from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

        from megatron.bridge.models.ernie.modeling_ernie55 import latent_mqa_core

        fsa = latent_mqa_core.fused_sparse_attention
        forward_calls, backward_calls, deterministic_calls = [], [], []

        def recording_forward(*args, **kwargs):
            forward_calls.append((args, kwargs))
            return _dense_flash_mla_forward(*args, **kwargs)

        def recording_backward(*args, **kwargs):
            backward_calls.append((args, kwargs))
            return _dense_sparse_backward(*args, **kwargs)

        monkeypatch.setattr(fsa, "_csa_fwd_flash_mla", recording_forward)
        monkeypatch.setattr(fsa, "_csa_sparse_attention_backward", recording_backward)
        real_deterministic_kwargs = getattr(fsa, "_deterministic_sparse_bwd_kwargs", None)
        if real_deterministic_kwargs is not None:  # Megatron-Core dev

            def recording_deterministic_kwargs(*args, **kwargs):
                inspect.signature(real_deterministic_kwargs).bind(*args, **kwargs)
                deterministic_calls.append(kwargs)
                return {}

            monkeypatch.setattr(fsa, "_deterministic_sparse_bwd_kwargs", recording_deterministic_kwargs)

        generator = torch.Generator().manual_seed(9)
        rows, heads, dim, latent, width, scale = 8, 4, 6, 4, 5, 0.37
        query = torch.randn(rows, heads, dim, dtype=torch.float64, generator=generator)
        kv = torch.randn(rows, dim, dtype=torch.float64, generator=generator)
        sink = torch.randn(heads, dtype=torch.float64, generator=generator)
        local = _random_indices(rows, rows, width, generator).to(torch.int32)
        if layout == "compact":
            flat, topk_length = fused_sparse_attention.build_flat_topk_idxs(
                local.unsqueeze(0), batch_size=1, compact=True
            )
        else:  # the CP-1 lists: -1 holes inside the rows' lengths
            flat, topk_length = ops.drop_invalid_key_blocks(local, block=2)
            assert bool(((flat < 0) & (torch.arange(flat.size(-1)) < topk_length.view(-1, 1))).any())
        list_width = flat.size(-1)
        d_out = torch.randn(rows, heads, latent, dtype=torch.float64, generator=generator)

        leaves = [tensor.clone().requires_grad_(True) for tensor in (query, kv, sink)]
        out = latent_mqa_core._FlashMLASinkGradFallback.apply(*leaves, flat, topk_length, scale, latent, deterministic)
        out.backward(d_out)

        expected_leaves = [tensor.clone().requires_grad_(True) for tensor in (query, kv, sink)]
        expected_out, _ = _sparse_attention_fp64(*expected_leaves[:2], expected_leaves[2], local.long(), scale, latent)
        expected_out.backward(d_out)

        torch.testing.assert_close(out, expected_out)
        for name, got, expected in zip(("query", "kv", "sink"), leaves, expected_leaves):
            torch.testing.assert_close(got.grad, expected.grad, msg=name)

        # The index format Megatron-Core DSA hands the kernels at d_qk = 576.
        if layout == "compact":
            assert torch.equal(topk_length, (local >= 0).sum(-1).int())
        ((forward_args, forward_kwargs),) = forward_calls
        forward_idxs = forward_args[2]
        assert forward_idxs.size(-1) % 512 == 0 and bool((forward_idxs[:, list_width:] == -1).all())
        assert torch.equal(forward_kwargs["topk_length"], topk_length)
        ((args, kwargs),) = backward_calls
        assert torch.all(torch.isneginf(args[5])), "the cuDNN backward must run with the sink disabled"
        backward_idxs = args[6]
        prefix = torch.arange(list_width).view(1, -1) < topk_length.view(-1, 1)
        # Placeholders past each row's length, the forward's slots (holes included) inside it.
        assert backward_idxs.shape == flat.shape and bool((backward_idxs[~prefix] >= 0).all())
        assert torch.equal(backward_idxs[prefix], flat[prefix])
        assert torch.equal(kwargs["topk_length"], topk_length) and kwargs["softmax_scale"] == scale
        if real_deterministic_kwargs is not None:
            ((call,),) = [deterministic_calls]
            assert call == {"requested": deterministic}
