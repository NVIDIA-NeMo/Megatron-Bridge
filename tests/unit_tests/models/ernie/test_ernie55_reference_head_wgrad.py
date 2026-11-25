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


def test_cp4_packed_reference_head_uses_contiguous_gradient_owners():
    """Run with torchrun --nproc-per-node=4 -m pytest -k cp4_packed_reference_head."""
    import os

    import torch.distributed as dist
    from megatron.core.context_parallel_layout.routes import build_thd_cp_partition_route

    if not torch.cuda.is_available() or os.environ.get("WORLD_SIZE") != "4" or "LOCAL_RANK" not in os.environ:
        pytest.skip("requires four CUDA ranks launched by torchrun")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    own_group = not dist.is_initialized()
    if own_group:
        dist.init_process_group("nccl")
    try:
        group = dist.group.WORLD
        rank, cp = group.rank(), group.size()
        for lengths in [(256,), (8, 24, 96, 128)]:
            torch.manual_seed(303)
            cu = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
            positions = []
            for lo, hi in zip(cu.tolist()[:-1], cu.tolist()[1:]):
                width = (hi - lo) // (2 * cp)
                for part in (rank, 2 * cp - rank - 1):
                    positions.extend(range(lo + part * width, lo + (part + 1) * width))
            zig = torch.tensor(positions, device="cuda")
            cont = torch.arange(rank * len(zig), (rank + 1) * len(zig), device="cuda")
            rows = sum(lengths)
            states = torch.randn(rows, 1, 64, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(128, 64, device="cuda", dtype=torch.bfloat16) * 0.1
            targets = torch.randint(128, (1, rows), device="cuda")
            targets[:, ::9] = -100
            dy = torch.randn(1, rows, device="cuda").masked_fill(targets < 0, 0)
            route = build_thd_cp_partition_route(cu, cp, rank)
            observed = []
            for layout, pos in (("zigzag", zig), ("contiguous", cont)):
                pack = SimpleNamespace(
                    qkv_format="thd",
                    cu_seqlens_q_padded=cu,
                    local_cp_size=None,
                    cp_partition_mode=layout,
                    cp_partition_route=route,
                )
                cfg = SimpleNamespace(
                    ernie_chunked_head_backend="triton",
                    ernie_chunked_head_rows=32,
                    ernie_head_save_projection=True,
                    ernie_head_reference_wgrad=True,
                    context_parallel_size=cp,
                    cp_partition_mode=layout,
                    cuda_graph_impl="none",
                )
                w = weight.detach().clone().requires_grad_()
                w.main_grad = torch.zeros_like(w, dtype=torch.float32)
                layer = SimpleNamespace(
                    training=True,
                    output_dtype=None,
                    config=cfg,
                    weight=w,
                    sequence_parallel=False,
                    multimax_ranges=torch.zeros(4, device="cuda"),
                    multimax_ts=torch.zeros(4, device="cuda"),
                )
                h = states[pos].detach().requires_grad_()
                with head_owners.reference_head_pack(pack, group):
                    loss, argmax = chunked_head.chunked_training_loss(
                        layer, h, targets[:, pos], None, return_argmax=True
                    )
                dx, dw = torch.autograd.grad(loss, (h, w), dy[:, pos])
                observed.append((loss.T.contiguous(), argmax, dx, dw))
                assert pack.cp_partition_mode == layout
            left, right = observed
            for actual, contiguous in zip(left[:3], right[:3]):
                gathered = [torch.empty_like(contiguous) for _ in range(cp)]
                dist.all_gather(gathered, contiguous, group=group)
                assert torch.equal(actual, torch.cat(gathered)[zig])
            assert torch.equal(left[3], right[3])
    finally:
        if own_group:
            dist.destroy_process_group()
