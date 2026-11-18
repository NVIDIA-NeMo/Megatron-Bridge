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

"""CPU tests of the ERNIE 5.5 fused fp32 mHC kernels' wiring (``ernie_fused_mhc``) and of their backward math.

The kernels themselves need CUDA and Triton (``test_ernie55_fused_mhc_gpu.py``). Here:

* the flag's default, its validation and the operation slots it selects;
* ``fused_mhc`` imports Triton lazily, and its eligibility check rejects what the kernels do not take;
* with the flag on, CPU tensors run the eager operations, bit for bit;
* the backward algorithm the kernels implement, written out in PyTorch step by step (the Sinkhorn VJP with its
  per-step state recompute, the sigmoid, alpha, bias and norm terms), matches autograd through the eager
  formulas in float64;
* the kernels' loops avoid the Triton loop-carry pitfall that Python-semantics helpers cannot see.
"""

import ast
import dataclasses
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import fused_dispatch, fused_mhc
from megatron.bridge.models.ernie.modeling_ernie55.hyper_connection import (
    Ernie55HyperConnectionModule,
    ernie55_fused_mhc_enabled,
    ernie55_h_post_bda,
    ernie55_h_post_bda_dispatch,
    ernie55_sinkhorn,
    use_ernie55_hyper_connections,
)


pytestmark = pytest.mark.unit

HIDDEN = 32
STREAMS = 4
ITERATIONS = 20
EPS = 1e-6


@dataclasses.dataclass
class _MhcConfig(TransformerConfig):
    """Megatron-Core's config plus the one ERNIE field the hyper-connection module reads."""

    ernie_fused_mhc: bool = False


def _module(*, fused: bool, seed: int = 0, convert: bool = True) -> HyperConnectionModule:
    config = _MhcConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_sinkhorn_iterations=ITERATIONS,
        use_cpu_initialization=True,
        ernie_fused_mhc=fused,
    )
    module = HyperConnectionModule(config, layer_number=1)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        weight = module.mapping_proj.weight
        weight.copy_(torch.randn(weight.shape, generator=generator) / math.sqrt(weight.shape[1]))
        module.bias.copy_(torch.randn(module.bias.shape, generator=generator))
        for alpha in (module.alpha_pre, module.alpha_post, module.alpha_res):
            alpha.copy_(0.6 + 0.3 * torch.rand(1, generator=generator))
    if convert:
        use_ernie55_hyper_connections(module)
    return module


# ---------------------------------------------------------------------------------------------
# Config and wiring
# ---------------------------------------------------------------------------------------------


def test_config_flag_defaults_off():
    fields = {f.name: f.default for f in dataclasses.fields(Ernie55TransformerConfig)}
    assert fields["ernie_fused_mhc"] is False
    assert ernie55_fused_mhc_enabled(SimpleNamespace(ernie_fused_mhc=True))
    assert not ernie55_fused_mhc_enabled(SimpleNamespace(ernie_fused_mhc=False))
    assert not ernie55_fused_mhc_enabled(SimpleNamespace())


@pytest.mark.parametrize("fused", [False, True])
def test_operation_slots_follow_the_flag(fused):
    eager_slot = ernie55_h_post_bda_dispatch if fused else ernie55_h_post_bda
    converted = _module(fused=fused)
    assert type(converted) is Ernie55HyperConnectionModule
    assert converted._ernie_fused_mhc is fused
    assert converted._h_post_bda_op is eager_slot
    assert converted._sinkhorn_op is ernie55_sinkhorn
    assert converted._proj_rms_compute_h_op is None
    direct = Ernie55HyperConnectionModule(converted.config, layer_number=1)
    assert direct._ernie_fused_mhc is fused
    assert direct._h_post_bda_op is eager_slot


def test_fused_mhc_imports_triton_lazily():
    """``fused_mhc`` imports only PyTorch at module level; Triton is bound by ``_load_kernels`` on the first launch."""
    tree = ast.parse(inspect.getsource(fused_mhc))
    top_level = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level.add((node.module or "").split(".")[0])
    assert top_level <= {"__future__", "torch"}, top_level
    assert fused_mhc._KERNELS_LOADED or fused_mhc.triton is None


def _cuda_like(dtype=torch.float32):
    """A stand-in with the attributes the eligibility check reads, for checks past the device test on CPU."""
    return SimpleNamespace(is_cuda=True, dtype=dtype, device=torch.device("cuda", 0))


@pytest.mark.parametrize(
    "streams, n, operands, expected",
    [
        (torch.zeros(2, 8), 4, {}, "not on a CUDA device"),
        (_cuda_like(torch.float16), 4, {}, "stream dtype"),
        (_cuda_like(), 3, {}, "3 residual streams"),
        (_cuda_like(), 16, {}, "16 residual streams"),
        (_cuda_like(), 4, {"bias": torch.zeros(8)}, "layer output bias"),
        (_cuda_like(), 4, {"fp32_operands": (_cuda_like(torch.bfloat16),)}, "must be fp32"),
        (_cuda_like(), 4, {"fp32_operands": (torch.zeros(1),)}, "must be fp32"),
        (_cuda_like(), 4, {"layer_output": _cuda_like(torch.int32)}, "layer output dtype"),
    ],
    ids=["cpu", "fp16-streams", "n3", "n16", "bias", "bf16-param", "cpu-param", "int-output"],
)
def test_eligibility_rejects_what_the_kernels_do_not_take(streams, n, operands, expected):
    reason = fused_mhc.fused_mhc_unsupported_reason(streams, n, **operands)
    assert reason is not None and expected in reason
    assert not fused_mhc.fused_mhc_supports(streams, n, **operands)


def test_eligibility_needs_triton(monkeypatch):
    monkeypatch.setattr(fused_dispatch, "triton_available", lambda: False)
    assert fused_mhc.fused_mhc_unsupported_reason(_cuda_like(), 4) == "Triton does not import"
    monkeypatch.setattr(fused_dispatch, "triton_available", lambda: True)
    assert fused_mhc.fused_mhc_unsupported_reason(_cuda_like(torch.bfloat16), 4, layer_output=_cuda_like()) is None


@pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
def test_flag_on_cpu_runs_the_eager_operations_bitwise(monkeypatch, stream_dtype):
    """On CPU the flag changes nothing: outputs and every gradient are equal, and no fused launch is counted."""
    from tests.unit_tests.models.ernie.ernie55_test_utils import isolate_fused_dispatch

    dispatch = isolate_fused_dispatch(monkeypatch)
    results = []
    for fused in (False, True):
        module = _module(fused=fused, seed=3)
        generator = torch.Generator().manual_seed(4)
        x = (torch.randn(6, 2, STREAMS * HIDDEN, generator=generator) * 3).to(stream_dtype).requires_grad_(True)
        y = torch.randn(6, 2, HIDDEN, generator=generator).to(torch.bfloat16).requires_grad_(True)
        # The three operations the flag routes: the mappings, the aggregate and the residual-update slot.
        h_pre, h_post, h_res = module.compute_mappings(x)
        aggregated = module.aggregate(x, h_pre)
        out = module._h_post_bda_op(h_res, x.view(6, 2, STREAMS, HIDDEN), h_post, y, None)
        cotangent = torch.randn(out.shape, generator=generator).to(out.dtype)
        loss = (out.float() * cotangent.float()).sum() + (aggregated * 0.5).sum()
        loss.backward()
        grads = [x.grad, y.grad] + [p.grad for p in module.parameters()]
        results.append([aggregated.detach(), out.detach(), *grads])
    for eager, fused in zip(*results, strict=True):
        assert torch.equal(eager, fused)
    assert not any(key.startswith("mhc_") for key in dispatch.FUSED_OP_CALLS)


# ---------------------------------------------------------------------------------------------
# The backward algorithm of the kernels, in float64
# ---------------------------------------------------------------------------------------------


def _col(mat: torch.Tensor) -> torch.Tensor:
    return mat / (mat.sum(dim=-2, keepdim=True) + EPS)


def _row(mat: torch.Tensor) -> torch.Tensor:
    return mat / (mat.sum(dim=-1, keepdim=True) + EPS)


def _sinkhorn_vjp(logits: torch.Tensor, grad: torch.Tensor, iterations: int) -> torch.Tensor:
    """``_mhc_map_bwd``'s Sinkhorn VJP: per reverse step, the forward states are re-derived from ``M0``."""
    column_steps = max(iterations, 1)
    p = logits.softmax(dim=-1)
    m0 = p + EPS
    g = grad
    for t in range(column_steps - 1, 0, -1):
        a_prev = _col(m0)
        for _ in range(1, t):
            a_prev = _col(_row(a_prev))
        b_t = _row(a_prev)
        a_t = _col(b_t)
        g = g / (b_t.sum(dim=-2, keepdim=True) + EPS)
        g = g - (g * a_t).sum(dim=-2, keepdim=True)
        g = g / (a_prev.sum(dim=-1, keepdim=True) + EPS)
        g = g - (g * b_t).sum(dim=-1, keepdim=True)
    g = g / (m0.sum(dim=-2, keepdim=True) + EPS)
    g = g - (g * _col(m0)).sum(dim=-2, keepdim=True)
    return p * (g - (g * p).sum(dim=-1, keepdim=True))


def _triton_loop_copies(source: str) -> list[str]:
    """Plain ``name = other_name`` assignments inside the loops of the ``@triton.jit`` functions in ``source``."""
    found = set()
    for kernel in ast.walk(ast.parse(source)):
        if not isinstance(kernel, ast.FunctionDef):
            continue
        if not any(ast.unparse(decorator).startswith("triton.jit") for decorator in kernel.decorator_list):
            continue
        for loop in ast.walk(kernel):
            if not isinstance(loop, (ast.For, ast.While)):
                continue
            for node in ast.walk(loop):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            found.add(f"{kernel.name}:{node.lineno}: {target.id} = {node.value.id}")
    return sorted(found)


def test_triton_kernel_loops_never_copy_one_name_into_another():
    """Guard against Triton's loop-carry rule, which silently broke the Sinkhorn VJP for iterations > 2.

    Triton dry-runs a loop body and carries only the names whose value then differs from their live-in value. In
    a body that starts with ``a_prev = a_t`` (and ``a_prev`` equal to ``a_t`` on entry), the dry run leaves
    ``a_prev`` at its live-in value, so it is not carried and holds its pre-loop value after the loop, unlike
    Python. The kernels must not copy one name into another inside a loop; the float64 helpers above run under
    Python semantics and cannot see this.
    """
    assert _triton_loop_copies(inspect.getsource(fused_mhc)) == []
    # The guard flags the pattern the first Sinkhorn backward used.
    buggy = (
        "@triton.jit\n"
        "def kernel(m0, T: tl.constexpr):\n"
        "    for step in range(0, T - 1):\n"
        "        a_t = f(m0)\n"
        "        a_prev = a_t\n"
        "        for _ in range(0, T - 1 - step):\n"
        "            a_prev = a_t\n"
        "            a_t = f(a_prev)\n"
    )
    assert _triton_loop_copies(buggy) == ["kernel:5: a_prev = a_t", "kernel:7: a_prev = a_t"]


def _sinkhorn_vjp_with_triton_carries(logits: torch.Tensor, grad: torch.Tensor, iterations: int) -> torch.Tensor:
    """The first kernel's Sinkhorn VJP as Triton ran it: ``a_prev`` was not carried and kept ``A_1`` after the loop."""
    p = logits.softmax(dim=-1)
    m0 = p + EPS
    g = grad
    for t in range(max(iterations, 1) - 1, 0, -1):
        a_t = _col(m0)
        a_first = a_t
        for _ in range(t):
            b_t = _row(a_t)
            a_t = _col(b_t)
        a_prev = a_first
        g = g / (b_t.sum(dim=-2, keepdim=True) + EPS)
        g = g - (g * a_t).sum(dim=-2, keepdim=True)
        g = g / (a_prev.sum(dim=-1, keepdim=True) + EPS)
        g = g - (g * b_t).sum(dim=-1, keepdim=True)
    g = g / (m0.sum(dim=-2, keepdim=True) + EPS)
    g = g - (g * _col(m0)).sum(dim=-2, keepdim=True)
    return p * (g - (g * p).sum(dim=-1, keepdim=True))


def test_the_old_carry_bug_is_visible_only_beyond_two_iterations():
    """Why iterations 1 (and 2) passed on GPU: with ``t = 1`` the stale ``a_prev`` is the right state."""
    generator = torch.Generator().manual_seed(11)
    logits = torch.randn(64, STREAMS, STREAMS, generator=generator, dtype=torch.float64)
    cotangent = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
    for iterations in (1, 2):
        torch.testing.assert_close(
            _sinkhorn_vjp_with_triton_carries(logits, cotangent, iterations),
            _sinkhorn_vjp(logits, cotangent, iterations),
            rtol=1e-12,
            atol=0,
        )
    expected = _sinkhorn_vjp(logits, cotangent, ITERATIONS)
    wrong = _sinkhorn_vjp_with_triton_carries(logits, cotangent, ITERATIONS)
    assert ((wrong - expected).norm() / expected.norm()).item() > 1.0


@pytest.mark.parametrize("iterations", [0, 1, 2, 5, 20])
@pytest.mark.parametrize("scale", [0.5, 6.0])
def test_sinkhorn_vjp_matches_autograd_in_float64(iterations, scale):
    generator = torch.Generator().manual_seed(iterations)
    logits = (scale * torch.randn(7, 3, STREAMS, STREAMS, generator=generator, dtype=torch.float64)).requires_grad_()
    cotangent = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
    (ernie55_sinkhorn(logits, iterations, EPS) * cotangent).sum().backward()
    torch.testing.assert_close(_sinkhorn_vjp(logits.detach(), cotangent, iterations), logits.grad, rtol=1e-10, atol=0)


def test_mapping_backward_matches_autograd_in_float64():
    """The whole ``_mhc_map_bwd`` + host algorithm: projection, norm, alpha and bias gradients, a zero-norm row."""
    n, width, rows = STREAMS, STREAMS * 16, 9
    num_proj = n * n + 2 * n
    generator = torch.Generator().manual_seed(5)

    def draw(*shape, scale=1.0):
        return scale * torch.randn(*shape, generator=generator, dtype=torch.float64)

    x = draw(rows, width, scale=3.0)
    x[3] = 0.0  # torch's norm backward is 0 at a zero norm; the kernel's coefficient must be too
    weight = draw(num_proj, width, scale=1 / math.sqrt(width))
    alphas = [0.6 + 0.3 * torch.rand(1, generator=generator, dtype=torch.float64) for _ in range(3)]
    bias = draw(num_proj)
    g_pre, g_post, g_res = draw(rows, n), draw(rows, n), draw(rows, n, n)

    # Autograd through the eager formulas (hyper_connection.compute_mappings, in float64).
    leaves = [t.clone().requires_grad_() for t in (x, weight, *alphas, bias)]
    lx, lw, la_pre, la_post, la_res, lb = leaves
    proj = torch.matmul(lx, lw.T.contiguous())
    r = 1.0 / (lx.norm(dim=-1, keepdim=True) / math.sqrt(width) + EPS)
    alpha = torch.cat([la_pre.expand(n), la_post.expand(n), la_res.expand(n * n)])
    h = r * proj * alpha + lb
    h_pre = h[:, :n].sigmoid() + EPS
    h_post = h[:, n : 2 * n].sigmoid() * 2
    h_res = ernie55_sinkhorn(h[:, 2 * n :].reshape(rows, n, n), ITERATIONS, EPS)
    ((h_pre * g_pre).sum() + (h_post * g_post).sum() + (h_res * g_res).sum()).backward()

    # The kernel's algorithm: recompute from proj and the norm, then the per-row terms and the GEMMs.
    proj = x @ weight.T
    nrm = x.norm(dim=-1)
    inv_sqrt_width = 1.0 / math.sqrt(width)
    r = 1.0 / (nrm * inv_sqrt_width + EPS)
    alpha = torch.cat([alphas[0].expand(n), alphas[1].expand(n), alphas[2].expand(n * n)])
    rp = r[:, None] * proj
    logits = rp * alpha + bias
    s_pre, s_post = logits[:, :n].sigmoid(), logits[:, n : 2 * n].sigmoid()
    g_h = torch.cat(
        [
            g_pre * (1 - s_pre) * s_pre,
            (g_post * 2) * (1 - s_post) * s_post,
            _sinkhorn_vjp(logits[:, 2 * n :].reshape(rows, n, n), g_res, ITERATIONS).reshape(rows, n * n),
        ],
        dim=1,
    )
    g_rp = g_h * alpha
    g_proj = g_rp * r[:, None]
    g_nrm = -(g_rp * proj).sum(dim=1) * r * r * inv_sqrt_width
    coef = torch.where(nrm > 0, g_nrm / nrm, torch.zeros_like(nrm))
    g_alpha = g_h * rp
    expected = {
        "x": g_proj @ weight + coef[:, None] * x,
        "weight": g_proj.T @ x,
        "alpha_pre": g_alpha[:, :n].sum().view(1),
        "alpha_post": g_alpha[:, n : 2 * n].sum().view(1),
        "alpha_res": g_alpha[:, 2 * n :].sum().view(1),
        "bias": g_h.sum(dim=0),
    }
    for (name, value), leaf in zip(expected.items(), leaves, strict=True):
        torch.testing.assert_close(value, leaf.grad, rtol=1e-10, atol=1e-12, msg=name)
