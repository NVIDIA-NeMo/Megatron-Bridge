# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from megatron.bridge.models.ernie.modeling_ernie55.chunked_head import _StrictMtpTrainingLoss


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA training head")
@pytest.mark.parametrize("save_projection", [True, False])
def test_strict_mtp_head_matches_dense_vjp(save_projection):
    pytest.importorskip("fused_weight_gradient_mlp_cuda")
    torch.manual_seed(717)
    device = torch.device("cuda")
    rows, width, vocab = 1057, 128, 256
    x = torch.randn(rows, width, device=device, dtype=torch.bfloat16, requires_grad=True)
    weight = (torch.randn(vocab, width, device=device) / width**0.5).bfloat16().requires_grad_()
    weight.main_grad = torch.full_like(weight, 0.25, dtype=torch.float32)
    ranges = torch.tensor([-1.0, 0.75, -1.5, 1.5], device=device, requires_grad=True)
    scales = torch.tensor([0.2, -0.1, 0.03, -0.02], device=device, requires_grad=True)
    labels = torch.randint(vocab, (rows,), device=device)
    labels[::7] = -100
    upstream = torch.randn(rows, device=device).masked_fill(labels < 0, 0)

    loss, prediction = _StrictMtpTrainingLoss.apply(
        x, weight, ranges, scales, labels, 768, save_projection, True, "triton"
    )
    actual = torch.autograd.grad(loss, (x, ranges, scales), upstream)

    # Project at the forward's original chunk geometry and mirror the native BF16
    # boundary between CE and multimax with an independent eager PyTorch expression.
    projection_rows = rows if save_projection else 768
    raw = torch.cat([part @ weight.T for part in x.split(projection_rows)]).detach().requires_grad_()
    z = raw.float()
    a, b, c, d = ranges.unbind()
    p, q, r, t = scales.unbind()
    corrected = (
        z + p * (a - z).relu() + q * (z - b).relu() + r * (c - z).relu().square() + t * (z - d).relu().square()
    ).bfloat16()
    expected_loss = F.cross_entropy(corrected.float(), labels, reduction="none")
    draw, dranges, dscales = torch.autograd.grad(expected_loss, (raw, ranges, scales), upstream)
    valid = labels >= 0
    torch.testing.assert_close(loss[valid], expected_loss[valid], rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(prediction, corrected.argmax(-1))
    expected = (draw @ weight, dranges, dscales)
    for observed, reference in zip(actual, expected):
        assert torch.isfinite(observed).all()
        relative = (observed.float() - reference.float()).norm() / reference.float().norm()
        assert relative < 0.01
    assert torch.count_nonzero(actual[0][~valid]) == 0
    reference_wgrad = 0.25 + draw.float().T @ x.float()
    assert torch.isfinite(weight.main_grad).all()
    torch.testing.assert_close(weight.main_grad, reference_wgrad, atol=3e-4, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA training head")
def test_only_mtp_selects_the_strict_backward(monkeypatch):
    from types import SimpleNamespace

    from megatron.bridge.models.ernie.modeling_ernie55 import chunked_head

    x = torch.empty(3, 1, 16, device="cuda", dtype=torch.bfloat16)
    w = torch.nn.Parameter(torch.empty(32, 16, device="cuda", dtype=x.dtype))
    w.main_grad = torch.zeros_like(w, dtype=torch.float32)
    cfg = SimpleNamespace(
        ernie_chunked_head_backend="triton",
        ernie_chunked_head_rows=16,
        ernie_head_save_projection=True,
        ernie_mtp_strict_head_backward=True,
    )
    layer = SimpleNamespace(
        training=True,
        config=cfg,
        weight=w,
        output_dtype=None,
        sequence_parallel=False,
        multimax_ranges=torch.zeros(4, device="cuda"),
        multimax_ts=torch.zeros(4, device="cuda"),
    )
    calls = []

    def record(which):
        def apply(*args):
            calls.append(which)
            return x.new_zeros(3, dtype=torch.float32), x.new_empty(0, dtype=torch.long)

        return staticmethod(apply)

    monkeypatch.setattr(chunked_head._MultimaxTrainingLoss, "apply", record("production"))
    monkeypatch.setattr(chunked_head._StrictMtpTrainingLoss, "apply", record("strict"))
    labels = torch.zeros(1, 3, dtype=torch.long, device="cuda")
    for kind in ("main", "mtp"):
        chunked_head.chunked_training_loss(layer, x, labels, None, head_kind=kind)
    cfg.ernie_mtp_strict_head_backward = False
    chunked_head.chunked_training_loss(layer, x, labels, None, head_kind="mtp")
    assert calls == ["production", "strict", "production"]
