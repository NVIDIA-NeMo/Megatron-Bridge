# Copyright (c) ModelScope Contributors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for Megatron-Bridge from the ERNIE 5.5 reference implementation's tests of its Triton kernels.
# Changed from the original: the kernels come from megatron.bridge.models.ernie.modeling_ernie55.fused_ops, the
# multimax reference is the reference implementation's apply_multimax written out here, the SiTU-GLU and RoPE dispatch
# tests use Bridge's dispatchers (fused_dispatch), and the remote top-k test is not ported (Bridge's D layers select
# their top-k with Megatron-Core's cuDNN DSA plans, not with the reference's Python loop).
"""GPU tests of the ERNIE 5.5 Triton kernels against the reference implementation's eager formulas.

The assertions are the reference implementation's own: its kernels claim bit-exact SiTU-GLU forward and backward,
multimax forward and bf16 logit gradient (fp32 logit gradient within 1e-6), and THD-CP local RoPE forward and
backward. Bridge carries the tests with the kernels; it does not rely on them for parity work.

Needs a CUDA device and Triton; without them the tests skip. One GPU is enough::

    python -m pytest -rA tests/unit_tests/models/ernie/test_ernie55_fused_ops_gpu.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import triton_available
from tests.unit_tests.models.ernie.ernie55_test_utils import isolate_fused_dispatch


pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() or not triton_available(), reason="the Triton kernels need CUDA and Triton"
    ),
]


@pytest.fixture(autouse=True)
def _local_device():
    """One device per rank when ``torch.distributed.run`` starts several (the tests themselves are per-device)."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _situ_eager(x, beta, linear_beta, compute_dtype=torch.float32):
    gate, up = x.to(compute_dtype).chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


def _multimax_eager(logits, ranges, ts):
    """The reference implementation's ``apply_multimax(logits, ranges, ts, compute_in_fp32=True)``."""
    x = logits.to(torch.float32)
    r = ranges.reshape(-1).to(torch.float32)
    t = ts.reshape(-1).to(torch.float32)
    out = (
        x
        + t[0] * torch.relu(r[0] - x)
        + t[1] * torch.relu(x - r[1])
        + t[2] * torch.relu(r[2] - x).square()
        + t[3] * torch.relu(x - r[3]).square()
    )
    return out.to(logits.dtype)


@pytest.mark.parametrize("linear_beta", [25.0, None])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_situ_glu_matches_eager(linear_beta, dtype):
    from megatron.bridge.models.ernie.modeling_ernie55.fused_ops import fused_situ_glu

    torch.manual_seed(0)
    # N = 2048, the ERNIE 5.5 expert width: 257 rows, each launched as two full 1024-column blocks.
    x = (torch.randn(257, 4096, device="cuda") * 4).to(dtype)
    grad = torch.randn(257, 2048, device="cuda").to(dtype)
    x_ref = x.clone().requires_grad_(True)
    out_ref = _situ_eager(x_ref, 4.0, linear_beta)
    out_ref.backward(grad)
    x_fused = x.clone().requires_grad_(True)
    out = fused_situ_glu(x_fused, 4.0, linear_beta)
    out.backward(grad)
    assert out.dtype == dtype
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(x_fused.grad, x_ref.grad, rtol=0, atol=0)


def test_situ_dispatch_uses_fused_kernel_on_cuda(monkeypatch):
    """Bridge's SiTU-GLU dispatcher (the reference's ``ErnieSiTUActivation`` test): contiguous inputs with the
    model's scales take the kernel; a non-contiguous input, or other scales, Megatron-Core's ``situ_glu``."""
    from megatron.core import activations
    from megatron.core.transformer import mlp as mcore_mlp

    from megatron.bridge.models.ernie.modeling_ernie55 import install_ernie55_fused_ops

    dispatch = isolate_fused_dispatch(monkeypatch)
    install_ernie55_fused_ops(
        SimpleNamespace(
            ernie_fused_ops=True, activation_func_tanh_clamp_scale=4.0, activation_func_tanh_clamp_scale_linear=25.0
        )
    )
    situ_glu = mcore_mlp.situ_glu
    x = torch.randn(4, 3, 64, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(situ_glu(x, 4.0, 25.0), _situ_eager(x, 4.0, 25.0), rtol=0, atol=0)
    assert dispatch.FUSED_OP_CALLS == {"situ_glu@megatron.core.transformer.mlp": 1}
    # Non-contiguous input takes the eager path.
    x_nc = torch.randn(64, 12, device="cuda", dtype=torch.bfloat16).T
    torch.testing.assert_close(situ_glu(x_nc, 4.0, 25.0), activations.situ_glu(x_nc, 4.0, 25.0, 0.0), rtol=0, atol=0)
    # So do scales that no model built with ernie_fused_ops=True recorded (another model's SiTU-GLU).
    torch.testing.assert_close(situ_glu(x, 2.0, None), activations.situ_glu(x, 2.0, None, 0.0), rtol=0, atol=0)
    assert dispatch.FUSED_OP_CALLS == {"situ_glu@megatron.core.transformer.mlp": 1}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_multimax_matches_eager(dtype):
    from megatron.bridge.models.ernie.modeling_ernie55.fused_ops import fused_multimax

    torch.manual_seed(0)
    logits = (torch.randn(3, 2, 10001, device="cuda") * 8).to(dtype)
    # Include exact hits on the thresholds (relu' = 0 there).
    logits.view(-1)[:4] = torch.tensor([-6.0, -2.0, 3.0, 7.0], dtype=dtype)
    grad = torch.randn_like(logits)
    ranges = torch.tensor([-6.0, 7.0, -2.0, 3.0], device="cuda", dtype=torch.bfloat16)
    ts = torch.tensor([0.5, -0.25, 0.01, -0.02], device="cuda", dtype=torch.bfloat16)

    def run(fn):
        x = logits.clone().requires_grad_(True)
        r, t = ranges.clone().requires_grad_(True), ts.clone().requires_grad_(True)
        out = fn(x, r, t)
        out.backward(grad)
        return out, x.grad, r.grad, t.grad

    out_ref, gx_ref, gr_ref, gt_ref = run(_multimax_eager)
    out, gx, gr, gt = run(fused_multimax)
    assert out.dtype == dtype and gr.dtype == ranges.dtype and gt.dtype == ts.dtype
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(gx, gx_ref, rtol=0, atol=0)
    else:
        torch.testing.assert_close(gx, gx_ref, rtol=1e-6, atol=1e-6)
    # The scalar grads are bf16 sums over 60k elements; compare them in fp32 against the eager fp32 sums.
    torch.testing.assert_close(gr.float(), gr_ref.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gt.float(), gt_ref.float(), rtol=1e-2, atol=1e-2)


def _unfused_rope():
    """Megatron-Core's ``apply_thd_cp_local_rope_unfused``, without Bridge's dispatcher if one is installed."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    current = cp_utils.apply_thd_cp_local_rope_unfused
    return getattr(current, "ernie55_unfused", current)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape", [(64, 512), (1, 512), (None, 64)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_thd_cp_local_rope_matches_unfused(inverse, shape, dtype):
    from megatron.core.models.common.embeddings import RotaryEmbedding

    from megatron.bridge.models.ernie.modeling_ernie55.fused_ops import fused_thd_cp_local_rope

    torch.manual_seed(0)
    heads, dim = shape
    pos_dim, rows, global_start = 64, 300, 1900
    cu_seqlens = torch.tensor([0, 1000, 2050, 2100, 4096], device="cuda", dtype=torch.int32)
    freqs = RotaryEmbedding(pos_dim, rotary_percent=1.0, rotary_base=10000)(4096, packed_seq=True)
    # (t, h, d) query, (t, 1, d) kv, (t, d) single-head; the eager helper's input layouts.
    x_shape = (rows, dim) if heads is None else (rows, heads, dim)
    x = (torch.randn(x_shape, device="cuda") * 3).to(dtype)
    grad = torch.randn(x_shape, device="cuda").to(dtype)
    unfused = _unfused_rope()
    config = SimpleNamespace(rotary_interleaved=False)

    def run(fn):
        xi = x.clone().requires_grad_(True)
        out = fn(xi)
        out.backward(grad)
        return out, xi.grad

    out_ref, gx_ref = run(
        lambda t: unfused(t, freqs, dim - pos_dim, pos_dim, cu_seqlens, global_start, config, inverse=inverse)
    )
    out, gx = run(
        lambda t: fused_thd_cp_local_rope(t, freqs, dim - pos_dim, pos_dim, cu_seqlens, global_start, inverse=inverse)
    )
    assert out.shape == out_ref.shape and out.dtype == dtype
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(gx, gx_ref, rtol=0, atol=0)


def test_thd_cp_local_rope_dispatch(monkeypatch):
    """Bridge's RoPE dispatcher: ERNIE configs with the flag take the kernel, every other config Megatron-Core's."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_utils

    from megatron.bridge.models.ernie.modeling_ernie55 import install_ernie55_fused_ops

    dispatch = isolate_fused_dispatch(monkeypatch)
    install_ernie55_fused_ops(SimpleNamespace(ernie_fused_ops=True))
    dispatcher = cp_utils.apply_thd_cp_local_rope_unfused
    assert dispatcher.ernie55_unfused is not None
    x = torch.randn(8, 1, 128, device="cuda", dtype=torch.bfloat16)
    freqs = torch.rand(16, 1, 1, 64, device="cuda")
    cu = torch.tensor([0, 16], device="cuda", dtype=torch.int32)
    ernie = SimpleNamespace(ernie_fused_ops=True, rotary_interleaved=False)
    ref = dispatcher.ernie55_unfused(x, freqs, 64, 64, cu, 4, SimpleNamespace(rotary_interleaved=False))
    torch.testing.assert_close(dispatcher(x, freqs, 64, 64, cu, 4, ernie), ref, rtol=0, atol=0)
    assert dispatch.FUSED_OP_CALLS == {"thd_cp_local_rope@forward": 1}
    # Non-ERNIE configs keep the mcore path.
    torch.testing.assert_close(dispatcher(x, freqs, 64, 64, cu, 4, SimpleNamespace(rotary_interleaved=False)), ref)
    assert dispatch.FUSED_OP_CALLS == {"thd_cp_local_rope@forward": 1}
