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
"""TP1 BF16 multimax and softmax without dense corrected logits or FP32 softmax."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _terms(x, rt):
    r0, r1, r2, r3 = tl.load(rt), tl.load(rt + 1), tl.load(rt + 2), tl.load(rt + 3)
    t0, t1, t2, t3 = tl.load(rt + 4), tl.load(rt + 5), tl.load(rt + 6), tl.load(rt + 7)
    a, b = tl.maximum(r0 - x, 0.0), tl.maximum(x - r1, 0.0)
    c, d = tl.maximum(r2 - x, 0.0), tl.maximum(x - r3, 0.0)
    # The baseline materializes this corrected value in BF16 before CE.
    y = libdevice.add_rn(x, libdevice.mul_rn(t0, a))
    y = libdevice.add_rn(y, libdevice.mul_rn(t1, b))
    y = libdevice.add_rn(y, libdevice.mul_rn(t2, libdevice.mul_rn(c, c)))
    y = libdevice.add_rn(y, libdevice.mul_rn(t3, libdevice.mul_rn(d, d)))
    return y.to(tl.bfloat16).to(tl.float32), a, b, c, d, t0, t1, t2, t3


@triton.jit
def _tile(
    raw,
    rt,
    labels,
    maxima,
    sums,
    targets,
    winners,
    v: tl.constexpr,
    tiles: tl.constexpr,
    b: tl.constexpr,
    need_preds: tl.constexpr,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * b + tl.arange(0, b)
    x = tl.load(raw + row.to(tl.int64) * v + ids, ids < v, 0.0).to(tl.float32)
    y, _, _, _, _, _, _, _, _ = _terms(x, rt)
    y = tl.where(ids < v, y, -float("inf"))
    m = tl.max(y, 0)
    off = row * tiles + tile
    tl.store(maxima + off, m)
    tl.store(sums + off, tl.sum(tl.exp(y - m), 0))
    label = tl.load(labels + row)
    tl.store(targets + off, tl.sum(tl.where((ids < v) & (ids == label), y, 0.0), 0))
    if need_preds:
        tl.store(winners + off, tl.min(tl.where(y == m, ids, 2147483647), 0))


@triton.jit
def _rows(
    maxima,
    sums,
    targets,
    winners,
    labels,
    max_out,
    sum_out,
    loss_out,
    preds_out,
    v: tl.constexpr,
    tiles: tl.constexpr,
    b: tl.constexpr,
    need_preds: tl.constexpr,
):
    row = tl.program_id(0)
    ix = tl.arange(0, b)
    off = row * tiles + ix
    tile_max = tl.load(maxima + off, ix < tiles, -float("inf"))
    row_max = tl.max(tile_max, 0)
    denom = tl.sum(tl.exp(tile_max - row_max) * tl.load(sums + off, ix < tiles, 0.0), 0)
    target = tl.sum(tl.load(targets + off, ix < tiles, 0.0), 0)
    label = tl.load(labels + row)
    tl.store(max_out + row, row_max)
    tl.store(sum_out + row, denom)
    tl.store(loss_out + row, tl.log(denom) - tl.where((label >= 0) & (label < v), target - row_max, 0.0))
    if need_preds:
        win = tl.load(winners + off, ix < tiles, 2147483647)
        tl.store(preds_out + row, tl.min(tl.where(tile_max == row_max, win, 2147483647), 0))


@triton.jit
def _back(
    raw, rt, labels, maximum, denom, upstream, draw, partial, v: tl.constexpr, tiles: tl.constexpr, b: tl.constexpr
):
    row, tile = tl.program_id(0), tl.program_id(1)
    ids = tile * b + tl.arange(0, b)
    off = row.to(tl.int64) * v + ids
    x = tl.load(raw + off, ids < v, 0.0).to(tl.float32)
    y, a, bb, c, d, t0, t1, t2, t3 = _terms(x, rt)
    norm = tl.div_rn(tl.exp(y - tl.load(maximum + row)), tl.load(denom + row))
    g = (norm - (ids == tl.load(labels + row)).to(tl.float32)) * tl.load(upstream + row)
    g = tl.where(ids < v, g.to(tl.bfloat16).to(tl.float32), 0.0)
    ga, gb = tl.where(a > 0, g * t0, 0.0), tl.where(bb > 0, g * t1, 0.0)
    gc, gd = g * t2 * 2.0 * c, g * t3 * 2.0 * d
    tl.store(draw + off, g - ga + gb - gc + gd, ids < v)
    p = partial + (row * tiles + tile).to(tl.int64) * 8
    tl.store(p, tl.sum(ga, 0))
    tl.store(p + 1, -tl.sum(gb, 0))
    tl.store(p + 2, tl.sum(gc, 0))
    tl.store(p + 3, -tl.sum(gd, 0))
    tl.store(p + 4, tl.sum(g * a, 0))
    tl.store(p + 5, tl.sum(g * bb, 0))
    tl.store(p + 6, tl.sum(g * c * c, 0))
    tl.store(p + 7, tl.sum(g * d * d, 0))


def forward(raw, rt, labels, maxima, denom, loss, preds):
    """Compute per-token CE, optional argmax and FP32 row statistics in two Triton kernels."""
    rows, vocab = raw.shape
    block = 2048
    tiles = triton.cdiv(vocab, block)
    pmax, psum, ptarget = torch.empty((3, rows, tiles), device=raw.device).unbind(0)
    need = preds is not None
    winners = torch.empty((rows, tiles), device=raw.device, dtype=torch.int32) if need else labels
    _tile[(rows, tiles)](raw, rt, labels, pmax, psum, ptarget, winners, vocab, tiles, block, need, num_warps=8)
    _rows[(rows,)](
        pmax,
        psum,
        ptarget,
        winners,
        labels,
        maxima,
        denom,
        loss,
        preds if need else labels,
        vocab,
        tiles,
        triton.next_power_of_2(tiles),
        need,
        num_warps=4,
    )


def backward(raw, rt, labels, maxima, denom, upstream):
    """Return raw-logit gradients and eight FP32 multimax parameter gradients for this token block."""
    rows, vocab = raw.shape
    block = 2048
    tiles = triton.cdiv(vocab, block)
    p = torch.empty((rows * tiles, 8), dtype=torch.float32, device=raw.device)
    draw = torch.empty_like(raw)
    _back[(rows, tiles)](
        raw, rt, labels, maxima, denom, upstream.contiguous(), draw, p, vocab, tiles, block, num_warps=8
    )
    return draw, p.sum(0)
