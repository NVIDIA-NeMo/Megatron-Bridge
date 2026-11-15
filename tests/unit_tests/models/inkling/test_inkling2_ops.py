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

"""Unit tests for the Inkling2 decoder operators (CPU)."""

import math

import pytest
import torch

from megatron.bridge.models.inkling.modeling_inkling2 import ops
from megatron.bridge.models.inkling.modeling_inkling2.attention import _eager_relative_sink_attention


pytestmark = pytest.mark.unit


def _loop_static_conv(x, weight, starts):
    """Per-element static convolution: y[t] = (x[t] + sum_lag w[:, W-1-lag] x[t-lag]) / sqrt(W+1)."""
    seq, channels = x.shape
    width = weight.shape[1]
    out = torch.empty(seq, channels)
    for t in range(seq):
        start = max(s for s in starts if s <= t)
        acc = torch.zeros(channels)
        for lag in range(width):
            if t - lag >= start:
                acc += weight[:, width - 1 - lag].float() * x[t - lag].float()
        out[t] = (acc + x[t].float()) / math.sqrt(width + 1)
    return out


def _loop_dynamic_conv(x, static, dynamic, starts):
    seq, channels = x.shape
    width, heads = static.shape[0], dynamic.shape[1]
    head_dim = channels // heads
    out = torch.empty(seq, channels)
    for t in range(seq):
        start = max(s for s in starts if s <= t)
        acc = torch.zeros(channels)
        for lag in range(width):
            if t - lag >= start:
                taps = static[lag].float() + dynamic[t, :, lag].float().repeat_interleave(head_dim)
                acc += taps * x[t - lag].float()
        out[t] = (acc + x[t].float()) / math.sqrt(2 * width + 1)
    return out


class TestShortConvolution:
    def test_static_matches_loop_and_respects_documents(self):
        torch.manual_seed(0)
        starts = [0, 5, 12]
        x = torch.randn(17, 24)
        weight = torch.randn(24, 4)
        positions = ops.document_positions(17, 1, torch.device("cpu"), torch.tensor(starts + [17]))
        actual = ops.static_short_conv(x.unsqueeze(1), weight, positions).squeeze(1)
        torch.testing.assert_close(actual, _loop_static_conv(x, weight, starts), rtol=1e-5, atol=1e-5)

    def test_dynamic_matches_loop(self):
        torch.manual_seed(1)
        starts = [0, 7]
        x = torch.randn(13, 32)
        static = torch.randn(4, 32)
        dynamic = torch.randn(13, 4, 4)
        positions = ops.document_positions(13, 1, torch.device("cpu"), torch.tensor(starts + [13]))
        actual = ops.dynamic_short_conv(x.unsqueeze(1), static, dynamic.unsqueeze(1), positions).squeeze(1)
        torch.testing.assert_close(actual, _loop_dynamic_conv(x, static, dynamic, starts), rtol=1e-5, atol=1e-5)

    def test_document_start_ignores_previous_document(self):
        x = torch.randn(8, 1, 4)
        weight = torch.randn(4, 4)
        positions = ops.document_positions(8, 1, torch.device("cpu"), torch.tensor([0, 4, 8]))
        changed = x.clone()
        changed[:4] += 100.0
        first = ops.static_short_conv(x, weight, positions)
        second = ops.static_short_conv(changed, weight, positions)
        torch.testing.assert_close(first[4:], second[4:])


class TestAttnRes:
    def test_single_source_returns_that_source(self):
        source = torch.randn(3, 2, 16)
        mix = ops.attn_res_mix([source], torch.randn(16), normalize_values=False)
        torch.testing.assert_close(mix[0], source)

    def test_weights_follow_normalized_key_scores(self):
        torch.manual_seed(2)
        sources = [torch.randn(5, 8) for _ in range(3)]
        query = torch.randn(2, 8)
        mix = ops.attn_res_mix(sources, query, normalize_values=True)
        stacked = torch.stack(sources)
        normalized = stacked * torch.rsqrt(stacked.square().mean(-1, keepdim=True) + 1e-6)
        scores = torch.einsum("ntd,qd->qtn", normalized, query / math.sqrt(8))
        expected = torch.einsum("qtn,ntd->qtd", scores.softmax(-1), normalized)
        torch.testing.assert_close(mix, expected, rtol=1e-5, atol=1e-6)

    def test_partial_addend_joins_last_source(self):
        torch.manual_seed(3)
        a, b, c = torch.randn(4, 6), torch.randn(4, 6), torch.randn(4, 6)
        query = torch.randn(6)
        with_addend = ops.attn_res_mix([a, b], query, normalize_values=False, partial_addend=c)
        summed = ops.attn_res_mix([a, b + c], query, normalize_values=False)
        torch.testing.assert_close(with_addend, summed)


class TestPointwiseOps:
    def test_l2_clip_scales_only_long_vectors(self):
        short = torch.full((1, 1, 4), 0.5)
        long = torch.full((1, 1, 4), 10.0)
        torch.testing.assert_close(ops.l2_clip(short, 2.0), short)
        clipped = ops.l2_clip(long, 2.0)
        assert torch.linalg.vector_norm(clipped).item() == pytest.approx(2.0, rel=1e-5)

    def test_l2_clip_applies_learned_scale_and_output_scale(self):
        x = torch.ones(2, 1, 3, 2)
        scale = torch.full((3, 2), 0.5)
        tau = torch.tensor([[1.0], [2.0]])
        out = ops.l2_clip(x, 100.0, scale=scale, output_scale=tau)
        torch.testing.assert_close(out[0], torch.full((1, 3, 2), 0.5))
        torch.testing.assert_close(out[1], torch.full((1, 3, 2), 1.0))

    def test_length_scaling(self):
        tau = ops.length_scaling_tau(torch.tensor([0, 8190, 8191, 16383]), 8192, 0.1)
        torch.testing.assert_close(tau, torch.tensor([1.0, 1.0, 1.0, 1.0 + 0.1 * math.log(2.0)]))

    def test_situ_uses_raw_gate_in_sigmoid(self):
        gate, up = torch.tensor([10.0]), torch.tensor([50.0])
        out = ops.situ_and_mul(torch.cat((gate, up)), 4.0, 25.0)
        expected = 4.0 * math.tanh(10.0 / 4.0) * torch.sigmoid(gate) * 25.0 * math.tanh(50.0 / 25.0)
        torch.testing.assert_close(out, expected)

    def test_document_positions(self):
        positions = ops.document_positions(7, 1, torch.device("cpu"), torch.tensor([0, 3, 7]))
        assert positions.squeeze(1).tolist() == [0, 1, 2, 0, 1, 2, 3]
        unpacked = ops.document_positions(3, 2, torch.device("cpu"))
        assert unpacked.tolist() == [[0, 0], [1, 1], [2, 2]]


class TestCompiledOps:
    """``compiled=True`` routes an operator through ``torch.compile``, stubbed here so the test runs on CPU."""

    @pytest.fixture
    def compile_calls(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
        calls = []

        def fake_compile(fn, *, dynamic=None, fullgraph=False, recompile_limit=None):
            calls.append(
                (fn.__name__, {"dynamic": dynamic, "fullgraph": fullgraph, "recompile_limit": recompile_limit})
            )
            return fn

        monkeypatch.setattr(torch, "compile", fake_compile)
        monkeypatch.setattr(ops, "_COMPILED", {})
        return calls

    @pytest.fixture
    def restore_dynamo_limits(self):
        import torch._dynamo

        entries = getattr(torch._dynamo.config, "_config", {})
        saved = {
            name: entries[name].default
            for name in ("recompile_limit", "accumulated_recompile_limit")
            if name in entries
        }
        yield
        for name, default in saved.items():
            entries[name].default = default

    def test_compiled_operators_match_eager(self, compile_calls, restore_dynamo_limits):
        torch.manual_seed(4)
        sources = [torch.randn(5, 1, 8) for _ in range(3)]
        addend = torch.randn(5, 1, 8)
        x = torch.randn(5, 1, 8)
        positions = ops.document_positions(5, 1, torch.device("cpu"), torch.tensor([0, 2, 5]))
        cases = {
            "_attn_res_mix": lambda c: ops.attn_res_mix(
                sources, torch.randn(8), normalize_values=True, partial_addend=addend, compiled=c
            ),
            "_residual_add": lambda c: ops.residual_add(x, addend, compiled=c),
            "_static_short_conv": lambda c: ops.static_short_conv(x, torch.randn(8, 4), positions, compiled=c),
            "_dynamic_short_conv": lambda c: ops.dynamic_short_conv(
                x, torch.randn(4, 8), torch.randn(5, 1, 2, 4), positions, compiled=c
            ),
            "_l2_clip": lambda c: ops.l2_clip(torch.randn(5, 1, 2, 4) * 3, 2.0, scale=torch.ones(2, 4), compiled=c),
            "_situ_and_mul": lambda c: ops.situ_and_mul(torch.randn(6, 16), 4.0, 25.0, compiled=c),
        }
        for case in cases.values():
            torch.manual_seed(5)
            eager = case(False)
            torch.manual_seed(5)
            torch.testing.assert_close(case(True), eager)

        compiled = dict(compile_calls)
        assert sorted(compiled) == sorted(cases)
        assert all(kwargs["fullgraph"] for kwargs in compiled.values())
        assert all(kwargs["recompile_limit"] == ops._RECOMPILE_LIMIT for kwargs in compiled.values())
        # The routed token count of the expert activation changes with every batch; sequence shapes do not.
        assert compiled["_situ_and_mul"]["dynamic"] is None
        assert all(kwargs["dynamic"] is False for name, kwargs in compiled.items() if name != "_situ_and_mul")

    def test_each_operator_compiles_once(self, compile_calls, restore_dynamo_limits):
        x = torch.randn(4, 1, 8)
        for _ in range(3):
            ops.residual_add(x, x, compiled=True)
        assert [name for name, _ in compile_calls] == ["_residual_add"]

    def test_eager_operators_do_not_compile(self, compile_calls):
        ops.residual_add(torch.ones(2), torch.ones(2))
        ops.situ_and_mul(torch.ones(1, 4), 4.0, 25.0)
        assert compile_calls == []

    def test_context_parallel_short_conv_stays_eager(self, compile_calls):
        positions = ops.document_positions(6, 1, torch.device("cpu"), torch.tensor([0, 6]))

        class LocalHalo:
            """A single-rank halo: no rows from other ranks."""

            def extend(self, x):
                return x

            def shift(self, extended, x, lag):
                return ops._lagged(x, lag, positions)

        x, weight = torch.randn(6, 1, 8), torch.randn(8, 4)
        with_halo = ops.static_short_conv(x, weight, positions, halo=LocalHalo(), compiled=True)
        torch.testing.assert_close(with_halo, ops.static_short_conv(x, weight, positions))
        assert compile_calls == []

    def test_recompile_limits_reach_other_threads(self, compile_calls, restore_dynamo_limits):
        import threading

        import torch._dynamo

        if not hasattr(torch._dynamo.config, "_config"):
            pytest.skip("this PyTorch release keeps no per-entry dynamo config defaults")
        ops.residual_add(torch.ones(2), torch.ones(2), compiled=True)
        seen = {}

        def read_limits():
            seen["recompile"] = torch._dynamo.config.recompile_limit
            seen["accumulated"] = torch._dynamo.config.accumulated_recompile_limit

        # Full activation recompute replays the forward pass on autograd's thread.
        thread = threading.Thread(target=read_limits)
        thread.start()
        thread.join()
        assert seen["recompile"] >= ops._RECOMPILE_LIMIT
        assert seen["accumulated"] >= ops._ACCUMULATED_RECOMPILE_LIMIT


class TestRelativeSinkAttention:
    def test_matches_gate_sink_formulation(self):
        """The in-softmax sink equals scaling the ordinary output by sigmoid(lse - sink)."""
        torch.manual_seed(4)
        seq, heads, kv_heads, dim, extent = 9, 4, 2, 8, 4
        q = torch.randn(seq, 1, heads, dim)
        k = torch.randn(seq, 1, kv_heads, dim)
        v = torch.randn(seq, 1, kv_heads, dim)
        rel = torch.randn(1, heads, seq, extent)
        sink = torch.randn(1, heads, seq)
        docs = torch.zeros(seq, dtype=torch.long)
        out = _eager_relative_sink_attention(q, k, v, rel, sink, window=None, docs=docs)

        kk = k.repeat_interleave(2, dim=2)[:, 0].transpose(0, 1)
        vv = v.repeat_interleave(2, dim=2)[:, 0].transpose(0, 1)
        scores = torch.einsum("qhd,hkd->hqk", q[:, 0], kk) / math.sqrt(dim)
        distance = torch.arange(seq)[:, None] - torch.arange(seq)[None, :]
        bias = torch.gather(rel[0], 2, distance.clamp(0, extent - 1).expand(heads, seq, seq))
        scores = scores + torch.where((distance >= 0) & (distance < extent), bias, torch.zeros_like(bias))
        scores = scores.masked_fill(distance < 0, float("-inf"))
        lse = torch.logsumexp(scores, -1)
        expected = torch.einsum("hqk,hkd->qhd", scores.softmax(-1), vv) * torch.sigmoid(lse - sink[0]).T.unsqueeze(-1)
        torch.testing.assert_close(out[:, 0], expected, rtol=1e-5, atol=1e-5)

    def test_window_and_documents_mask_keys(self):
        torch.manual_seed(5)
        seq = 8
        q, k, v = (torch.randn(seq, 1, 2, 4) for _ in range(3))
        rel = torch.zeros(1, 2, seq, 3)
        sink = torch.full((1, 2, seq), -1e4)
        docs = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        out = _eager_relative_sink_attention(q, k, v, rel, sink, window=3, docs=docs)
        changed = v.clone()
        changed[0] += 100.0
        out_changed = _eager_relative_sink_attention(q, k, changed, rel, sink, window=3, docs=docs)
        # Token 3 sees tokens 1-3 (window 3); tokens 4+ belong to another document.
        torch.testing.assert_close(out[3:], out_changed[3:])
        assert not torch.allclose(out[2], out_changed[2])
