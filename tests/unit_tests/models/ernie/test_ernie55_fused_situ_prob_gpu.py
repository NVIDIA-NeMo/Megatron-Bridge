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

"""GPU tests of the fused routed SiTU-GLU x probability kernel (``ernie_fused_situ_prob``).

* The kernel against the unfused path it replaces: Bridge's SiTU-GLU Triton kernel (``fused_ops``), then
  Megatron-Core's product with the fp32 probabilities and the cast, ``(situ_glu(x) * probs).to(x.dtype)``. Inputs at
  sigma 1 and 4, beta 1 and 4, linear beta off, 2 and 25 (ERNIE 5.5's), random probabilities with zeros (padded rows)
  and ones, and widths 2048 (ERNIE 5.5's ``moe_intermediate_size``) and 1000 (a partial column block), plus the rows
  of ``SITU_PROB_SUBNORMAL_ROWS``, whose products are fp32 subnormals. The forward output and the input gradient must
  be bitwise equal; the probability gradient, which the kernel sums in its own order, must lie within the
  summation-order bound ``2 n u sum_j |dy_j a_j|`` of each row (``u = 2^-24``).
* The subnormal rows on their own: they reach the subnormal range on the unfused path, where PyTorch's multiply keeps
  what libdevice's flush-to-zero ``mul_rn`` would zero, and the kernel matches them.
* Probability shapes ``[rows]`` and ``[rows, 1]``, a 3-D input, no rows, no probability gradient, fp32 inputs, and a
  repeated backward (the kernel is deterministic).
* The ERNIE 5.5 routed experts (Transformer Engine's grouped linears): Megatron-Core's ``TEGroupedMLP.forward``
  against :class:`Ernie55GroupedMLP` on the same weights and inputs, with and without ``moe_act`` recompute: output,
  input gradient and expert weight gradients bitwise, the probability gradient within the bound; bf16 probabilities
  fall back to Megatron-Core's operations, bitwise.
* The ERNIE 5.5 MoE layer from ``ernie55_hybrid_stack_spec`` with ``ernie_fused_situ_prob`` off and on, plain and with
  ``"moe"`` in ``recompute_modules`` (selective MoE recompute, where the kernel runs in the forward and again in the
  recompute): output, routed and shared expert gradients bitwise; the router and input gradients, which receive the
  probability gradient, within fp32 summation-order rounding. These runs use ``torch.use_deterministic_algorithms``
  (warnings only), so that Megatron-Core's unfused unpermute adds with ``index_add_`` instead of the atomic
  ``scatter_add_``, whose bf16 sums depend on arrival order; two flag-off runs must then agree bitwise, which the test
  checks before it compares the flag's runs.

Needs CUDA and Triton; the module tests also need Transformer Engine with grouped linear layers. Without them the
tests skip; with ``ERNIE55_GPU_STRICT=1`` (GPU acceptance runs) they fail instead. Runs on one GPU, or on every rank
of a ``torch.distributed.run`` launch (each rank tests its own GPU; the module tests use expert parallelism over all
ranks)::

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m torch.distributed.run --nproc-per-node 4 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_fused_situ_prob_gpu.py

(``CUBLAS_WORKSPACE_CONFIG`` makes cuBLAS reproducible for the MoE-layer tests' deterministic algorithms.)
"""

from __future__ import annotations

import logging
import os

import pytest
import torch

from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import triton_available
from tests.unit_tests.models.ernie.ernie55_test_utils import (
    SITU_PROB_SUBNORMAL_ROWS,
    isolate_fused_dispatch,
    situ_prob_subnormal_rows,
)


logger = logging.getLogger(__name__)

# ERNIE55_GPU_STRICT=1 turns every skip of this file into a failure (GPU acceptance runs).
_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        (not torch.cuda.is_available() or not triton_available()) and not _STRICT,
        reason="the Triton kernels need CUDA and Triton",
    ),
]

U = 2.0**-24
ROWS = 4099
EXPERTS_SITE = "megatron.core.transformer.moe.experts"
MLP_SITE = "megatron.core.transformer.mlp"
ROUTED_SITE = "situ_glu_prob@routed_experts"
_INT_VIEW = {torch.bfloat16: torch.int16, torch.float16: torch.int16, torch.float32: torch.int32}


def _skip(reason: str) -> None:
    """Skip the test, or fail it under ``ERNIE55_GPU_STRICT=1``, where a skip would hide a missing kernel."""
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _local_device():
    """One device per rank when ``torch.distributed.run`` starts several; under strict mode, CUDA and Triton."""
    if not torch.cuda.is_available() or not triton_available():
        _skip("the Triton kernels need CUDA and Triton")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def _bitwise(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Bit-for-bit equality (signed zeros and NaN payloads included)."""
    if actual.dtype != expected.dtype or actual.shape != expected.shape:
        return False
    view = _INT_VIEW[actual.dtype]
    return torch.equal(actual.contiguous().view(view), expected.contiguous().view(view))


def _situ_kernel(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """Bridge's SiTU-GLU Triton kernel (the reference implementation's)."""
    from megatron.bridge.models.ernie.modeling_ernie55.fused_ops import fused_situ_glu

    return fused_situ_glu(x, beta, linear_beta)


def _unfused(x: torch.Tensor, probs: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """The routed activation the kernel replaces: the SiTU-GLU kernel, then Megatron-Core's multiply and cast."""
    act = _situ_kernel(x, beta, linear_beta)
    return (act * probs).to(act.dtype)


def _fused(x: torch.Tensor, probs: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    from megatron.bridge.models.ernie.modeling_ernie55.fused_situ_prob import fused_situ_glu_prob

    return fused_situ_glu_prob(x, probs, beta, linear_beta)


def _inputs(rows: int, width: int, sigma: float, *, seed: int, dtype: torch.dtype = torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = (sigma * torch.randn(rows, 2 * width, device="cuda", generator=generator)).to(dtype)
    probs = 3 * torch.rand(rows, 1, device="cuda", generator=generator)
    probs[::13] = 0.0  # padded rows carry zero probabilities
    probs[5::29] = 1.0
    grad = torch.randn(rows, width, device="cuda", generator=generator).to(dtype)
    # Rows 1-4: products in the fp32 subnormal range.
    edge = slice(1, 1 + len(SITU_PROB_SUBNORMAL_ROWS))
    x[edge], probs[edge], grad[edge] = situ_prob_subnormal_rows(width, dtype=dtype, device="cuda")
    return x, probs, grad


def _run(function, x: torch.Tensor, probs: torch.Tensor, grad: torch.Tensor, beta: float, linear_beta: float | None):
    xi, pi = x.clone().requires_grad_(True), probs.clone().requires_grad_(True)
    out = function(xi, pi, beta, linear_beta)
    out.backward(grad)
    return out.detach(), xi.grad, pi.grad


def _row_bound(grad: torch.Tensor, act: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Bound on the difference of two fp32 sums of each row of ``grad * act`` taken in different orders.

    Plus the smallest subnormal, 2^-149, so that a row of zero products has a nonzero bound: a larger floor would
    accept any error in rows whose products are all subnormal.
    """
    return 2 * act.shape[-1] * U * (grad.float() * act.float()).abs().sum(-1).reshape(shape) + 2.0**-149


def _check_prob_grad(actual: torch.Tensor, expected: torch.Tensor, bound: torch.Tensor, name: str) -> None:
    assert actual.dtype == expected.dtype == torch.float32 and actual.shape == expected.shape, name
    diff = (actual - expected).abs()
    ratio = (diff / bound).max().item()
    share = (actual == expected).float().mean().item()
    logger.info(
        "%s: probability gradient bitwise in %.2f%% of rows, max |diff| / summation-order bound %.3g",
        name,
        100 * share,
        ratio,
    )
    assert bool((diff <= bound).all()), f"{name}: probability gradient outside the bound (max ratio {ratio:.3g})"


# ---------------------------------------------------------------------------------------------
# The kernel
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("width", [2048, 1000])
@pytest.mark.parametrize("linear_beta", [None, 2.0, 25.0], ids=["linear-beta-off", "linear-beta-2", "linear-beta-25"])
@pytest.mark.parametrize("beta", [1.0, 4.0])
@pytest.mark.parametrize("sigma", [1.0, 4.0])
def test_kernel_matches_the_unfused_path(sigma, beta, linear_beta, width):
    name = f"sigma {sigma}, beta {beta}, linear beta {linear_beta}, width {width}"
    x, probs, grad = _inputs(ROWS, width, sigma, seed=int(100 * sigma + 10 * beta + width + (linear_beta or 0)))
    out_ref, x_grad_ref, prob_grad_ref = _run(_unfused, x, probs, grad, beta, linear_beta)
    out, x_grad, prob_grad = _run(_fused, x, probs, grad, beta, linear_beta)
    assert _bitwise(out, out_ref), f"{name}: forward"
    assert _bitwise(x_grad, x_grad_ref), f"{name}: input gradient"
    with torch.no_grad():
        bound = _row_bound(grad, _situ_kernel(x, beta, linear_beta), probs.shape)
    _check_prob_grad(prob_grad, prob_grad_ref, bound, name)


@pytest.mark.parametrize("linear_beta", [None, 2.0, 25.0], ids=["linear-beta-off", "linear-beta-2", "linear-beta-25"])
@pytest.mark.parametrize("beta", [1.0, 4.0])
def test_kernel_keeps_subnormal_products(beta, linear_beta):
    """The kernel's products keep fp32 subnormals as PyTorch's multiply does (no flush to zero).

    First the rows of ``SITU_PROB_SUBNORMAL_ROWS`` are checked to reach the subnormal range on the unfused path, so a
    flush-to-zero product would change the forward (row 0), the input gradient (row 3) and the probability gradient;
    then the kernel must match the unfused path on them. Whether row 1's activation is a bf16 subnormal depends on
    Triton's plain fp32 multiply inside the SiTU-GLU kernel, so it is only logged.
    """
    name = f"subnormal rows, beta {beta}, linear beta {linear_beta}"
    x, probs, grad = situ_prob_subnormal_rows(2048, device="cuda")
    out_ref, x_grad_ref, prob_grad_ref = _run(_unfused, x, probs, grad, beta, linear_beta)
    out, x_grad, prob_grad = _run(_fused, x, probs, grad, beta, linear_beta)
    tiny = torch.finfo(torch.float32).tiny
    with torch.no_grad():
        act = _situ_kernel(x, beta, linear_beta).float()
        product = act[0] * probs[0]
        act_grad = grad[2:].float() * probs[2:]

    def subnormal(t: torch.Tensor) -> bool:
        return bool(((t.abs() < tiny) & (t != 0)).all())

    # Row 0: a normal activation whose product with the probability is subnormal; the unfused output keeps it.
    assert bool((act[0].abs() >= tiny).all()) and subnormal(product), f"{name}: row 0 misses the subnormal range"
    assert bool((out_ref[0] != 0).all()), f"{name}: row 0's unfused output flushed"
    # Rows 2-3: dy * p is subnormal and rounds to the normal 2^-126 in bf16, which reaches row 3's input gradient.
    assert subnormal(act_grad), f"{name}: rows 2-3 miss the subnormal range"
    assert bool((act_grad.to(torch.bfloat16).float().abs() == tiny).all()), f"{name}: rows 2-3 round differently"
    assert bool((x_grad_ref[3] != 0).any()), f"{name}: row 3's unfused input gradient is zero"
    # Why the products avoid libdevice, logged: the SiTU-GLU kernel's up-half gradient is libdevice.mul_rn of the
    # output gradient; for a subnormal one (2^-127) it is zero under flush to zero and 0.8 * 2^-126 otherwise.
    probe = torch.cat([torch.full((1, 128), 2.0), torch.ones(1, 128)], -1).to(device="cuda", dtype=torch.bfloat16)
    probe.requires_grad_(True)
    _situ_kernel(probe, 4.0, None).backward(torch.full((1, 128), 2.0**-127, device="cuda", dtype=torch.bfloat16))
    logger.info(
        "%s: row 1 activation subnormal %s, nonzero unfused input gradients in rows 0-3 %s; libdevice.mul_rn "
        "flushes subnormals %s",
        name,
        subnormal(act[1]),
        [int((x_grad_ref[row] != 0).sum()) for row in range(4)],
        bool((probe.grad[:, 128:] == 0).all()),
    )

    assert _bitwise(out, out_ref), f"{name}: forward"
    assert _bitwise(x_grad, x_grad_ref), f"{name}: input gradient"
    _check_prob_grad(prob_grad, prob_grad_ref, _row_bound(grad, act, probs.shape), name)


def test_kernel_fp32_inputs():
    x, probs, grad = _inputs(ROWS, 2048, 4.0, seed=3, dtype=torch.float32)
    out_ref, x_grad_ref, prob_grad_ref = _run(_unfused, x, probs, grad, 4.0, 25.0)
    out, x_grad, prob_grad = _run(_fused, x, probs, grad, 4.0, 25.0)
    assert _bitwise(out, out_ref) and _bitwise(x_grad, x_grad_ref)
    with torch.no_grad():
        bound = _row_bound(grad, _situ_kernel(x, 4.0, 25.0), probs.shape)
    _check_prob_grad(prob_grad, prob_grad_ref, bound, "fp32")


def test_kernel_shapes_and_edge_cases():
    x, probs, grad = _inputs(256, 2048, 4.0, seed=4)
    out_ref, x_grad_ref, _ = _run(_unfused, x, probs, grad, 4.0, 25.0)
    _, _, prob_grad_2d = _run(_fused, x, probs, grad, 4.0, 25.0)

    # [rows] probabilities, as Megatron-Core's dispatchers hand them before TEGroupedMLP adds the trailing axis: the
    # same kernel arithmetic, so the probability gradient is bitwise the [rows, 1] one.
    out, x_grad, prob_grad = _run(_fused, x, probs.reshape(-1), grad, 4.0, 25.0)
    assert _bitwise(out, out_ref) and _bitwise(x_grad, x_grad_ref)
    assert prob_grad.shape == (256,) and _bitwise(prob_grad, prob_grad_2d.reshape(-1))

    # A 3-D input [s, b, 2n] with one probability per (s, b) row.
    out3, x_grad3, prob_grad3 = _run(
        _fused, x.reshape(64, 4, -1), probs.reshape(64, 4), grad.reshape(64, 4, -1), 4.0, 25.0
    )
    assert _bitwise(out3.reshape(256, -1), out_ref) and _bitwise(x_grad3.reshape(256, -1), x_grad_ref)
    assert prob_grad3.shape == (64, 4) and _bitwise(prob_grad3.reshape(-1), prob_grad_2d.reshape(-1))

    # No rows: empty outputs, an empty probability gradient.
    out0, x_grad0, prob_grad0 = _run(_fused, x[:0], probs[:0], grad[:0], 4.0, 25.0)
    assert out0.shape == (0, 2048) and x_grad0.shape == (0, 4096) and prob_grad0.shape == (0, 1)

    # Probabilities without a gradient: only the input gradient, still bitwise.
    xi = x.clone().requires_grad_(True)
    _fused(xi, probs, 4.0, 25.0).backward(grad)
    assert _bitwise(xi.grad, x_grad_ref)


def test_kernel_backward_is_deterministic():
    x, probs, grad = _inputs(ROWS, 2048, 4.0, seed=5)
    first = _run(_fused, x, probs, grad, 4.0, 25.0)
    second = _run(_fused, x, probs, grad, 4.0, 25.0)
    for a, b in zip(first, second):
        assert _bitwise(a, b)


# ---------------------------------------------------------------------------------------------
# The routed experts and the MoE layer
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def expert_parallel(tmp_path_factory):
    """Expert parallelism over every rank (one rank without a launcher), and the process groups."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import EXPERTS, _missing_dependencies

    missing = _missing_dependencies()
    if missing:
        _skip("the routed-expert tests need " + "; ".join(missing))
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if not dist.is_initialized():
        if "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl")
        else:
            store = tmp_path_factory.mktemp("ernie55_fused_situ_prob") / "store"
            dist.init_process_group("nccl", init_method=f"file://{store}", rank=0, world_size=1)
    world = dist.get_world_size()
    if EXPERTS % world:
        _skip(f"{EXPERTS} experts do not divide over {world} ranks")
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, expert_model_parallel_size=world
    )
    model_parallel_cuda_manual_seed(1234)
    yield ProcessGroupCollection.use_mpu_process_groups(), world
    parallel_state.destroy_model_parallel()


@pytest.fixture
def deterministic_algorithms():
    """``torch.use_deterministic_algorithms(True, warn_only=True)`` for one test, then the previous mode.

    Uninitialized memory is not filled: the runs compare the layer as training runs it.
    """
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.utils.deterministic.fill_uninitialized_memory,
    )
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    yield
    torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
    torch.utils.deterministic.fill_uninitialized_memory = previous[2]


@pytest.fixture(autouse=True)
def _fresh_moe_metrics():
    from megatron.core.transformer.moe import moe_logging

    saved = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    yield
    moe_logging.set_moe_metrics_tracker(saved)


def _config(world: int, *, flag: bool, recompute_modules: list[str] | None = None):
    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import make_config

    recompute = {}
    if recompute_modules:
        recompute = dict(recompute_granularity="selective", recompute_modules=recompute_modules)
    return make_config(
        activation_func_tanh_clamp_scale=4.0,
        activation_func_tanh_clamp_scale_linear=25.0,
        expert_model_parallel_size=world,
        ernie_fused_ops=True,
        ernie_fused_situ_prob=flag,
        **recompute,
    )


def _build_moe(config, pg_collection):
    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import build_moe

    return build_moe(config, pg_collection, device=torch.cuda.current_device())


@pytest.mark.parametrize(
    "recompute,probs_dtype",
    [(False, torch.float32), (True, torch.float32), (False, torch.bfloat16)],
    ids=["plain", "moe-act-recompute", "bf16-probs-fallback"],
)
def test_routed_experts_match_megatron_core(expert_parallel, monkeypatch, recompute, probs_dtype):
    """Megatron-Core's ``TEGroupedMLP.forward`` against :class:`Ernie55GroupedMLP` on one module and one input."""
    import torch.distributed as dist
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55GroupedMLP, install_ernie55_fused_ops
    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import HIDDEN

    pg_collection, world = expert_parallel
    dispatch = isolate_fused_dispatch(monkeypatch)
    config = _config(world, flag=True, recompute_modules=["moe_act"] if recompute else None)
    install_ernie55_fused_ops(config)
    experts = _build_moe(config, pg_collection).experts
    assert type(experts) is Ernie55GroupedMLP and experts.activation_recompute is recompute
    local = experts.num_local_experts

    generator = torch.Generator().manual_seed(21 + dist.get_rank())
    counts = torch.randint(0, 96, (local,), generator=generator)
    counts[0] = 0  # an expert without tokens
    rows = int(counts.sum())
    hidden = torch.randn(rows, HIDDEN, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    probs = (3 * torch.rand(rows, generator=generator)).to(device="cuda", dtype=probs_dtype)
    out_grad = torch.randn(rows, HIDDEN, generator=generator).to(device="cuda", dtype=torch.bfloat16)

    def run(forward):
        experts.zero_grad(set_to_none=True)
        h, p = hidden.clone().requires_grad_(True), probs.clone().requires_grad_(True)
        output, bias = forward(experts, h, counts.clone(), p)
        assert bias is None
        output.backward(out_grad)
        grads = {name: param.grad.clone() for name, param in experts.named_parameters()}
        return output.detach(), h.grad, p.grad, grads

    passes = 2 if recompute else 1
    out_ref, h_grad_ref, p_grad_ref, grads_ref = run(TEGroupedMLP.forward)
    assert dispatch.FUSED_OP_CALLS == {f"situ_glu@{EXPERTS_SITE}": passes}
    dispatch.FUSED_OP_CALLS.clear()
    out, h_grad, p_grad, grads = run(Ernie55GroupedMLP.forward)
    if probs_dtype == torch.float32:
        assert dispatch.FUSED_OP_CALLS == {ROUTED_SITE: passes}
    else:  # bf16 probabilities: Megatron-Core's operations, with Bridge's SiTU-GLU kernel
        assert dispatch.FUSED_OP_CALLS == {f"situ_glu@{EXPERTS_SITE}": passes}

    assert _bitwise(out, out_ref), "output"
    assert _bitwise(h_grad, h_grad_ref), "input gradient"
    assert grads.keys() == grads_ref.keys()
    for name, value in grads_ref.items():
        assert _bitwise(grads[name], value), name
    if probs_dtype != torch.float32:
        assert _bitwise(p_grad, p_grad_ref), "probability gradient (fallback)"
        return
    # The summation-order bound needs the activation and its output gradient (fc2's input gradient).
    with torch.no_grad():
        fc1_output, _ = experts.linear_fc1(hidden, counts.tolist())
        act = _situ_kernel(fc1_output, 4.0, 25.0)
    fc2_input = (act * probs.unsqueeze(-1)).to(act.dtype).requires_grad_(True)
    fc2_output, _ = experts.linear_fc2(fc2_input, counts.tolist())
    (act_out_grad,) = torch.autograd.grad(fc2_output, fc2_input, out_grad)
    _check_prob_grad(p_grad, p_grad_ref, _row_bound(act_out_grad, act, probs.shape), f"experts {recompute=}")


@pytest.mark.parametrize("recompute", [False, True], ids=["plain", "moe-recompute"])
def test_moe_layer_with_and_without_the_flag(expert_parallel, monkeypatch, recompute, deterministic_algorithms):
    """The ERNIE 5.5 MoE layer, ``ernie_fused_situ_prob`` off and on, on the same parameters and input.

    With deterministic algorithms (see the module docstring), after checking that two flag-off runs agree bitwise.
    """
    import torch.distributed as dist
    from megatron.core.transformer.moe.experts import TEGroupedMLP

    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55GroupedMLP, install_ernie55_fused_ops
    from tests.unit_tests.models.ernie.test_ernie55_moe_gpu import HIDDEN, TOKENS

    pg_collection, world = expert_parallel
    dispatch = isolate_fused_dispatch(monkeypatch)
    generator = torch.Generator().manual_seed(31 + dist.get_rank())
    hidden = torch.randn(TOKENS, 1, HIDDEN, generator=generator).to(device="cuda", dtype=torch.bfloat16)
    out_grad = torch.randn(TOKENS, 1, HIDDEN, generator=generator).to(device="cuda", dtype=torch.bfloat16)

    def run(flag: bool) -> dict[str, torch.Tensor]:
        config = _config(world, flag=flag, recompute_modules=["moe"] if recompute else None)
        install_ernie55_fused_ops(config)
        moe = _build_moe(config, pg_collection)
        assert type(moe.experts) is (Ernie55GroupedMLP if flag else TEGroupedMLP)
        assert moe.moe_layer_recompute is recompute
        x = hidden.clone().requires_grad_(True)
        output, _ = moe(x)
        output.backward(out_grad)
        result = {"output": output.detach(), "d input": x.grad}
        result.update({f"d {name}": param.grad for name, param in moe.named_parameters()})
        return result

    passes = 2 if recompute else 1
    expected = run(False)
    assert dispatch.FUSED_OP_CALLS[f"situ_glu@{EXPERTS_SITE}"] == passes
    shared = dispatch.FUSED_OP_CALLS[f"situ_glu@{MLP_SITE}"]
    dispatch.FUSED_OP_CALLS.clear()
    # Precondition: the flag-off layer reproduces itself bitwise, so a difference below is the flag's.
    repeat = run(False)
    assert repeat.keys() == expected.keys()
    unreproducible = [name for name, value in expected.items() if not _bitwise(repeat[name], value)]
    assert not unreproducible, f"two flag-off runs differ in {unreproducible}"
    dispatch.FUSED_OP_CALLS.clear()
    actual = run(True)
    assert dispatch.FUSED_OP_CALLS == {ROUTED_SITE: passes, f"situ_glu@{MLP_SITE}": shared}

    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if name.startswith("d router.") or name == "d input":
            # These receive the probability gradient: fp32 summation order in the router, then bf16 rounding.
            error = ((actual[name].double() - value.double()).norm() / value.double().norm().clamp_min(1e-30)).item()
            logger.info("MoE layer %s: %s relative error %.3g", "recompute" if recompute else "plain", name, error)
            assert error <= 1e-3, f"{name}: relative error {error:.3g}"
        else:
            assert _bitwise(actual[name], value), name
