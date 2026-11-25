# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from megatron.bridge.models.ernie.modeling_ernie55 import chunked_head, head_owners


def test_reference_head_layout_is_scoped_and_does_not_change_the_pack(monkeypatch):
    group = SimpleNamespace(size=lambda: 4)
    cfg = SimpleNamespace(context_parallel_size=4, cp_partition_mode="zigzag")
    outer = SimpleNamespace(cp_partition_mode="zigzag", qkv_format="thd", local_cp_size=None)
    inner = SimpleNamespace(cp_partition_mode="contiguous")
    monkeypatch.setattr(head_owners.ContiguousBackwardOwners, "for_pack", lambda pack, group, config: pack)
    with head_owners.reference_head_pack(outer, group):
        borrowed = head_owners.reference_head_owners(cfg)
        assert borrowed is not outer and borrowed.cp_partition_mode == "zigzag"
        borrowed.cp_partition_mode = "contiguous"
        with head_owners.reference_head_pack(inner, group):
            assert head_owners.reference_head_owners(cfg) is None
        assert head_owners.reference_head_owners(cfg).cp_partition_mode == "zigzag"
    with pytest.raises(ValueError, match="forward context"):
        head_owners.reference_head_owners(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 training heads")
@pytest.mark.parametrize("save_projection", [True, False])
@pytest.mark.parametrize("strict_mtp", [True, False])
def test_reference_wgrad_uses_both_bf16_head_gradients(save_projection, strict_mtp):
    torch.manual_seed(991)
    dtype = torch.bfloat16
    rows, width, vocab = 89, 64, 128
    x = [torch.randn(rows, width, device="cuda", dtype=dtype).requires_grad_() for _ in range(2)]
    w = (torch.randn(vocab, width, device="cuda") / width**0.5).to(dtype).requires_grad_()
    w.main_grad = torch.full_like(w, 0.125, dtype=torch.float32)
    # A zero multimax correction makes the dense BF16 output projection an independent oracle.
    ranges = torch.zeros(4, device="cuda", requires_grad=True)
    scales = torch.zeros(4, device="cuda", requires_grad=True)
    expected, losses, upstream = [], [], []
    for index, hidden in enumerate(x):
        labels = torch.randint(vocab, (rows,), device="cuda")
        labels[::7] = -100
        dy = torch.randn(rows, device="cuda").masked_fill(labels < 0, 0)
        fn = chunked_head._StrictMtpTrainingLoss if index and strict_mtp else chunked_head._MultimaxTrainingLoss
        loss, _ = fn.apply(hidden, w, ranges, scales, labels, 48, save_projection, False, "triton", True)
        losses.append(loss)
        upstream.append(dy)
        raw = torch.cat([h @ w.T for h in hidden.split(rows if save_projection else 48)]).detach().requires_grad_()
        reference_loss = F.cross_entropy(raw.float(), labels, reduction="none")
        (d_raw,) = torch.autograd.grad(reference_loss, raw, dy)
        expected.append(d_raw.T @ hidden.detach())
    separate = [torch.autograd.grad(loss, w, dy, retain_graph=True)[0] for loss, dy in zip(losses, upstream)]
    (actual,) = torch.autograd.grad(losses, w, upstream)
    assert torch.equal(actual, separate[0] + separate[1])
    for gradient, reference in zip(separate, expected):
        assert gradient.dtype == dtype and torch.isfinite(gradient).all()
        relative_l2 = (gradient.float() - reference.float()).norm() / reference.float().norm()
        assert relative_l2 < 0.004
    assert torch.equal(w.main_grad, torch.full_like(w.main_grad, 0.125))
