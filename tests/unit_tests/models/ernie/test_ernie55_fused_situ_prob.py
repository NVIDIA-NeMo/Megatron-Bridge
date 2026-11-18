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

"""CPU tests of the fused routed SiTU-GLU x probability path (``ernie_fused_situ_prob``).

The Triton kernels need CUDA; ``test_ernie55_fused_situ_prob_gpu.py`` runs them. Here, on CPU:

* ``fused_situ_prob`` imports without Triton, and only defining its kernels needs Triton;
* the kernels' arithmetic, written out in PyTorch, against autograd of the unfused routed activation
  ``(situ_glu(x) * probs).to(x.dtype)``: the forward and the input gradient bitwise, the probability gradient within
  the fp32 summation-order bound, at sigma 1 and 4, beta 1 and 4, linear beta off and on, random probabilities. One
  rounding instead of two would change outputs, so the comparison can fail;
* the GPU tests' rows of fp32-subnormal products: with PyTorch's product the kernels' arithmetic matches the unfused
  path on them, with a flush-to-zero product (``libdevice.mul_rn`` under Triton) it does not;
* the dispatch conditions (:func:`situ_glu_prob_unfit_reason`) and the launch counting;
* :class:`Ernie55GroupedMLP` against Megatron-Core's ``TEGroupedMLP`` forward on Transformer-Engine-free grouped
  linears, through the unfused fallback and through the fused branch (the kernel replaced by its PyTorch
  arithmetic), with and without ``moe_act`` recompute, and the ERNIE 5.5 MoE layer through a reentrant checkpoint
  (the structure of ``"moe"`` in ``recompute_modules``);
* the conversion (flag, idempotence, other experts, unknown subclasses, the forward fingerprint) and the config
  envelope of the flag.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import subprocess
import sys
import textwrap
import zlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe import experts as mcore_experts
from megatron.core.transformer.moe import moe_logging
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, SequentialMLP, TEGroupedMLP

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    Ernie55GroupedMLP,
    Ernie55MoELayer,
    ernie55_hybrid_stack_spec,
    fused_ops,
    fused_situ_prob,
    install_ernie55_fused_ops,
    use_ernie55_fused_situ_prob,
)
from megatron.bridge.models.ernie.modeling_ernie55 import moe as moe_module
from tests.unit_tests.models.ernie.ernie55_test_utils import (
    CallRecorder,
    cpu_current_device,
    isolate_fused_dispatch,
    situ_prob_subnormal_rows,
)


pytestmark = pytest.mark.unit

EXPERTS_SITE = "megatron.core.transformer.moe.experts"
MLP_SITE = "megatron.core.transformer.mlp"
LOCAL_EXPERTS = 4
HIDDEN = 16
MOE_FFN = 24
U = 2.0**-24
SITU = dict(ernie_fused_ops=True, activation_func_tanh_clamp_scale=4.0, activation_func_tanh_clamp_scale_linear=25.0)

# The D-layer envelope that Ernie55TransformerConfig.finalize validates for every ERNIE 5.5 config.
_D_LAYER_FIELDS = dict(
    num_attention_heads=4,
    q_lora_rank=32,
    kv_lora_rank=16,
    qk_head_dim=8,
    qk_pos_emb_head_dim=8,
    v_head_dim=8,
    ernie_vha_rank=2,
    experimental_attention_variant="dsa",
    dsa_kernel_backend="none",
    dsa_indexer_n_heads=4,
    dsa_indexer_head_dim=16,
    dsa_indexer_topk=6,
    dsa_indexer_k_norm_fp32=True,
    csa_window_size=3,
)


@pytest.fixture
def dispatch(monkeypatch):
    """The ``fused_dispatch`` module with a clean dispatch state (restored after the test)."""
    return isolate_fused_dispatch(monkeypatch)


@pytest.fixture
def cpu_checkpoint_without_output(monkeypatch):
    """Let Megatron-Core's ``CheckpointWithoutOutput`` (``moe_act`` recompute) run on a CPU-only PyTorch.

    Its recompute region snapshots and restores the CPU and CUDA RNG states; without CUDA only the CPU state is kept
    (the activation draws no random numbers). The rebuild points the released output's storage at the recomputed
    data with a C++ extension that it compiles on first use; here the data is copied into the storage instead, which
    gives every alias of the output the same values.
    """
    from megatron.core.tensor_parallel import random as mcore_random

    def share_storage(dst: torch.Tensor, src: torch.Tensor) -> None:
        storage = dst.untyped_storage()
        storage.resize_(src.untyped_storage().nbytes())
        storage.copy_(src.untyped_storage())

    monkeypatch.setattr(mcore_random, "_get_all_rng_states", lambda: (torch.get_rng_state(), None, {}, "live"))
    monkeypatch.setattr(
        mcore_random, "_set_all_rng_states", lambda cpu_state, *args, **kwargs: torch.set_rng_state(cpu_state)
    )
    monkeypatch.setattr(mcore_random, "_get_share_storage", lambda: share_storage)


def _config(**overrides) -> Ernie55TransformerConfig:
    """Finalized ERNIE 5.5 SiTU-GLU config with bf16 experts (as in training) and grouped GEMMs."""
    values = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        transformer_impl="local",
        bf16=True,
        params_dtype=torch.bfloat16,
        ffn_hidden_size=32,
        gated_linear_unit=True,
        activation_func=F.silu,
        num_moe_experts=LOCAL_EXPERTS,
        moe_router_topk=2,
        moe_ffn_hidden_size=MOE_FFN,
        moe_shared_expert_intermediate_size=12,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-4,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="allgather",
        moe_grouped_gemm=True,
        **SITU,
        **_D_LAYER_FIELDS,
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


# ---------------------------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------------------------


def _situ(x: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """SiTU-GLU in fp32, rounded once to the input dtype: the arithmetic of Bridge's SiTU-GLU kernel."""
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(x.dtype)


def _unfused(x: torch.Tensor, probs: torch.Tensor, beta: float, linear_beta: float | None) -> torch.Tensor:
    """Megatron-Core's routed activation (``TEGroupedMLP._unfused_forward``): SiTU-GLU, the fp32 product, the cast back."""
    act = _situ(x, beta, linear_beta)
    return (act * probs).to(act.dtype)


class _KernelArithmetic(torch.autograd.Function):
    """``fused_situ_prob``'s two kernels, statement for statement, in eager PyTorch (fp32 inside).

    ``mul`` is the kernels' fp32 product (``a * p``, ``dy * p``, ``dy * a``): PyTorch's, as ``_mul_rn_ieee`` computes
    it, unless a test passes another one.
    """

    @staticmethod
    def forward(ctx, x, probs, beta, linear_beta, mul=torch.mul):
        ctx.save_for_backward(x, probs)
        ctx.beta, ctx.linear_beta, ctx.mul = beta, linear_beta, mul
        act = _situ(x, beta, linear_beta).float()  # first rounding
        return mul(act, probs.reshape(*x.shape[:-1], 1)).to(x.dtype)  # second rounding

    @staticmethod
    def backward(ctx, out_grad):
        x, probs = ctx.saved_tensors
        p = probs.reshape(*x.shape[:-1], 1)
        g = out_grad.float()
        # The multiply's gradient, cast to the activation dtype, enters the SiTU-GLU backward.
        act_grad = ctx.mul(g, p).to(x.dtype)
        with torch.enable_grad():
            leaf = x.detach().requires_grad_(True)
            act = _situ(leaf, ctx.beta, ctx.linear_beta)
            (x_grad,) = torch.autograd.grad(act, leaf, act_grad)
        prob_grad = ctx.mul(g, act.detach().float()).sum(-1).reshape(probs.shape)
        return x_grad, prob_grad, None, None, None


def _kernel_arithmetic(x, probs, beta, linear_beta=None, mul=torch.mul):
    return _KernelArithmetic.apply(x, probs, beta, linear_beta, mul)


def _flush(t: torch.Tensor) -> torch.Tensor:
    """fp32 subnormals to zeros of their sign, as a ``.ftz`` instruction treats its inputs and its result."""
    return torch.where(t.abs() < torch.finfo(torch.float32).tiny, t * 0.0, t)


def _flushing_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``libdevice.mul_rn`` under Triton's ``nvvm-reflect-ftz``: ``mul.rn.ftz.f32``."""
    return _flush(_flush(a.float()) * _flush(b.float()))


def _bitwise(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Bit-for-bit equality, signed zeros included."""
    view = torch.int32 if actual.dtype == torch.float32 else torch.int16
    return actual.dtype == expected.dtype and torch.equal(actual.view(view), expected.view(view))


def _row_bound(out_grad: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    """Bound on the difference of two fp32 sums of a row of ``out_grad * act`` in different orders.

    Plus the smallest subnormal, so that a row of zero products has a nonzero bound and rows of subnormal products
    keep a tight one.
    """
    terms = (out_grad.float() * act.float()).abs().sum(-1)
    return 2 * act.shape[-1] * U * terms + 2.0**-149


def test_kernel_module_imports_without_triton():
    """``fused_situ_prob`` imports only PyTorch; with Triton blocked it loads, and only its kernels need Triton."""
    code = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys


        class BlockTriton(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "triton" or name.startswith("triton."):
                    raise ImportError(f"{name} is blocked")
                return None


        sys.meta_path.insert(0, BlockTriton())
        spec = importlib.util.spec_from_file_location("ernie55_fused_situ_prob_without_triton", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded = sorted(name for name in sys.modules if name == "triton" or name.startswith("triton."))
        assert not loaded, loaded
        assert module.triton is None and not module._KERNELS_LOADED
        try:
            module._load_kernels()
        except ImportError:
            pass
        else:
            raise AssertionError("defining the kernels must import Triton")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, fused_situ_prob.__file__], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]


@pytest.mark.parametrize("linear_beta", [None, 2.0, 25.0], ids=["linear-beta-off", "linear-beta-2", "linear-beta-25"])
@pytest.mark.parametrize("beta", [1.0, 4.0])
@pytest.mark.parametrize("sigma", [1.0, 4.0])
def test_kernel_arithmetic_matches_the_unfused_path(sigma, beta, linear_beta):
    """Two roundings forward; the cast activation gradient into the SiTU-GLU backward; a row sum for the probability.

    Autograd of the unfused routed activation and the kernels' decomposition agree: the output and the input gradient
    bitwise, the probability gradient within the summation-order bound of each row.
    """
    generator = torch.Generator().manual_seed(int(1000 * sigma + 10 * beta + (linear_beta or 0)))
    rows, width = 300, 96
    x = (sigma * torch.randn(rows, 2 * width, generator=generator)).to(torch.bfloat16)
    probs = 3 * torch.rand(rows, 1, generator=generator)
    probs[::17] = 0.0  # padded rows carry zero probabilities
    out_grad = torch.randn(rows, width, generator=generator).to(torch.bfloat16)

    def run(function):
        xi, pi = x.clone().requires_grad_(True), probs.clone().requires_grad_(True)
        out = function(xi, pi, beta, linear_beta)
        out.backward(out_grad)
        return out.detach(), xi.grad, pi.grad

    expected, x_grad_ref, prob_grad_ref = run(_unfused)
    actual, x_grad, prob_grad = run(_kernel_arithmetic)
    assert actual.dtype == torch.bfloat16 and prob_grad.dtype == torch.float32
    assert torch.equal(actual, expected)
    assert torch.equal(x_grad, x_grad_ref)
    bound = _row_bound(out_grad, _situ(x, beta, linear_beta))
    assert ((prob_grad - prob_grad_ref).abs().squeeze(-1) <= bound).all()

    # The second rounding matters: one rounding of the fp32 activation times the probability differs.
    gate, up = x.float().chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    assert not torch.equal((gate * up * probs).to(torch.bfloat16), expected)


@pytest.mark.parametrize("linear_beta", [None, 2.0, 25.0], ids=["linear-beta-off", "linear-beta-2", "linear-beta-25"])
@pytest.mark.parametrize("beta", [1.0, 4.0])
def test_subnormal_rows_need_an_ieee_product(beta, linear_beta):
    """The GPU tests' subnormal rows (``SITU_PROB_SUBNORMAL_ROWS``) tell PyTorch's product from a flush-to-zero one.

    The kernels' three products stand in for PyTorch's multiplies, which keep fp32 subnormals; ``libdevice.mul_rn``
    under Triton's ``nvvm-reflect-ftz`` does not. On these rows the kernels' arithmetic with PyTorch's product is the
    unfused path's (forward and input gradient bitwise, probability gradient within the row bound), and the same
    arithmetic with a flush-to-zero product differs in the forward (rows 0 and 1), the input gradient (row 3) and the
    probability gradient (row 1). Here the SiTU-GLU arithmetic keeps subnormals; on the GPU both kernels flush in it.
    """
    x, probs, out_grad = situ_prob_subnormal_rows(96)
    tiny = torch.finfo(torch.float32).tiny

    def run(mul=None):
        function = _unfused if mul is None else functools.partial(_kernel_arithmetic, mul=mul)
        xi, pi = x.clone().requires_grad_(True), probs.clone().requires_grad_(True)
        out = function(xi, pi, beta, linear_beta)
        out.backward(out_grad)
        return out.detach(), xi.grad, pi.grad

    # The rows reach the subnormal range on the unfused path.
    act = _situ(x, beta, linear_beta).float()
    product, act_grad = act * probs, out_grad[2:].float() * probs[2:]
    assert bool((act[0].abs() >= tiny).all() and (product[0].abs() < tiny).all() and (product[0] != 0).all())
    assert bool((act[1].abs() < tiny).all() and (act[1] != 0).all())
    assert bool((act_grad.abs() < tiny).all() and (act_grad.to(torch.bfloat16).float().abs() == tiny).all())

    expected, x_grad_ref, prob_grad_ref = run()
    bound = _row_bound(out_grad, act)
    actual, x_grad, prob_grad = run(torch.mul)
    assert _bitwise(actual, expected) and _bitwise(x_grad, x_grad_ref)
    assert bool(((prob_grad - prob_grad_ref).abs().squeeze(-1) <= bound).all())

    flushed, x_grad_flushed, prob_grad_flushed = run(_flushing_mul)
    assert not torch.equal(flushed[0], expected[0]) and not torch.equal(flushed[1], expected[1])
    assert bool((x_grad_ref[3] != 0).any()) and not torch.equal(x_grad_flushed[3], x_grad_ref[3])
    assert (prob_grad_flushed - prob_grad_ref).abs()[1, 0] > bound[1]


# ---------------------------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------------------------


def _force_kernel_support(monkeypatch, dispatch) -> None:
    """Let every tensor pass the device, dtype and Triton check, as a CUDA tensor would."""
    monkeypatch.setattr(dispatch, "fused_kernels_support", lambda tensor: True)


def test_unfit_reasons(dispatch, monkeypatch):
    """The fused kernel takes exactly the SiTU-GLU kernel's calls with fp32 probabilities, one per row."""
    reason = dispatch.situ_glu_prob_unfit_reason
    install_ernie55_fused_ops(SimpleNamespace(**SITU))
    x = torch.randn(6, 16).to(torch.bfloat16)
    probs = torch.rand(6, 1)
    assert "CUDA" in reason(x, probs, 4.0, 25.0)  # a CPU tensor
    _force_kernel_support(monkeypatch, dispatch)
    assert reason(x, probs, 4.0, 25.0) is None
    assert reason(x, probs.reshape(6), 4.0, 25.0) is None
    assert reason(x.reshape(2, 3, 16), probs, 4.0, 25.0) is None
    assert "scales" in reason(x, probs, 2.0, 25.0)
    assert "scales" in reason(x, probs, 4.0, None)
    assert "glu_linear_offset" in reason(x, probs, 4.0, 25.0, 0.5)
    assert "even" in reason(torch.randn(6, 15).to(torch.bfloat16), probs, 4.0, 25.0)
    assert "contiguous" in reason(torch.randn(16, 6).to(torch.bfloat16).t(), probs, 4.0, 25.0)
    assert "bfloat16" in reason(x, probs.to(torch.bfloat16), 4.0, 25.0)
    assert "not contiguous" in reason(x, torch.rand(6, 2)[:, :1], 4.0, 25.0)
    assert "one value per row" in reason(x, torch.rand(5, 1), 4.0, 25.0)
    # A model built later with these scales and ernie_fused_ops=False turns the SiTU-GLU kernel off, and this one too.
    install_ernie55_fused_ops(SimpleNamespace(**{**SITU, "ernie_fused_ops": False}))
    assert "scales" in reason(x, probs, 4.0, 25.0)


def test_launches_are_counted(dispatch, monkeypatch):
    kernel = CallRecorder(lambda x, probs, beta, linear_beta: torch.zeros(*x.shape[:-1], x.shape[-1] // 2))
    monkeypatch.setattr(fused_situ_prob, "fused_situ_glu_prob", kernel)
    x, probs = torch.randn(6, 16), torch.rand(6, 1)
    dispatch.fused_situ_glu_prob(x, probs, 4.0, 25.0)
    dispatch.fused_situ_glu_prob(x, probs, 4.0)
    assert [(args[0] is x, args[1] is probs, args[2:]) for args, _ in kernel.calls] == [
        (True, True, (4.0, 25.0)),
        (True, True, (4.0, None)),
    ]
    assert dispatch.FUSED_OP_CALLS == {"situ_glu_prob@routed_experts": 2}


def test_situ_glu_dispatcher_keeps_its_conditions(dispatch, monkeypatch):
    """The SiTU-GLU dispatcher reads the shared conditions: the kernel for the recorded scales, else Megatron-Core."""
    install_ernie55_fused_ops(SimpleNamespace(**SITU))
    dispatcher = importlib.import_module(EXPERTS_SITE).situ_glu
    kernel = CallRecorder(lambda x, beta, linear_beta: torch.full((*x.shape[:-1], x.shape[-1] // 2), 7.0))
    monkeypatch.setattr(fused_ops, "fused_situ_glu", kernel)
    x = torch.randn(6, 16).to(torch.bfloat16)
    eager = dispatcher.ernie55_eager
    assert torch.equal(dispatcher(x, 4.0, 25.0), eager(x, 4.0, 25.0, 0.0))  # CPU
    _force_kernel_support(monkeypatch, dispatch)
    assert torch.equal(dispatcher(x, 4.0, 25.0), torch.full((6, 8), 7.0))
    assert torch.equal(dispatcher(x, 4.0, 25.0, 0.5), eager(x, 4.0, 25.0, 0.5))
    assert torch.equal(dispatcher(x, 2.0, 25.0), eager(x, 2.0, 25.0, 0.0))
    assert len(kernel.calls) == 1 and dispatch.FUSED_OP_CALLS == {f"situ_glu@{EXPERTS_SITE}": 1}


# ---------------------------------------------------------------------------------------------
# Ernie55GroupedMLP against Megatron-Core's TEGroupedMLP.forward
# ---------------------------------------------------------------------------------------------


class _GroupedLinear(torch.nn.Module):
    """Transformer-Engine-free stand-in for TE's grouped linear: one weight per local expert, rows split by count."""

    def __init__(self, num_gemms: int, input_size: int, output_size: int, *, config, init_method, bias, **kwargs):
        super().__init__()
        assert not bias
        self.num_gemms = num_gemms
        for index in range(num_gemms):
            weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
            init_method(weight)
            self.register_parameter(f"weight{index}", weight)

    def forward(self, x: torch.Tensor, tokens_per_expert: list[int]) -> tuple[torch.Tensor, None]:
        assert isinstance(tokens_per_expert, list) and len(tokens_per_expert) == self.num_gemms
        chunks = torch.split(x, tokens_per_expert)
        return torch.cat([F.linear(chunk, getattr(self, f"weight{i}")) for i, chunk in enumerate(chunks)]), None


_GROUPED_EXPERTS = functools.partial(
    TEGroupedMLP, submodules=GroupedMLPSubmodules(linear_fc1=_GroupedLinear, linear_fc2=_GroupedLinear)
)


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    """O(1) values for every parameter, keyed by name, and an O(1) expert bias."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            generator = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float64)
            if name.endswith("routed_scaling_factor_param"):
                values = 1.0 + 0.3 * noise.clamp(-1.5, 1.5)
            elif "norm" in name:
                values = 1.0 + 0.2 * noise
            else:
                values = 2 * noise / param.shape[-1] ** 0.5
            param.copy_(values.to(param.dtype))


def _grouped_mlp(config: Ernie55TransformerConfig, *, seed: int) -> TEGroupedMLP:
    experts = _GROUPED_EXPERTS(LOCAL_EXPERTS, config, pg_collection=SimpleNamespace(ep=None, expt_tp=None))
    _randomize_(experts, seed)
    return experts


def _expert_inputs(seed: int, rows: int = 37):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
    tokens_per_expert = torch.tensor([9, 0, 17, rows - 26])
    probs = 3 * torch.rand(rows, generator=generator)
    out_grad = torch.randn(rows, HIDDEN, generator=generator).to(torch.bfloat16)
    return hidden, tokens_per_expert, probs, out_grad


def _run_experts(forward, experts, inputs) -> dict[str, torch.Tensor]:
    hidden, tokens_per_expert, probs, out_grad = inputs
    experts.zero_grad(set_to_none=True)
    hidden = hidden.clone().requires_grad_(True)
    probs = probs.clone().requires_grad_(True)
    output, bias = forward(experts, hidden, tokens_per_expert, probs)
    assert bias is None
    output.backward(out_grad)
    result = {"output": output.detach(), "d hidden": hidden.grad, "d probs": probs.grad}
    result.update({f"d {name}": param.grad for name, param in experts.named_parameters()})
    return result


def _assert_matches(actual: dict, expected: dict, *, probs_bound: torch.Tensor | None) -> None:
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype and actual[name].shape == value.shape, name
        if name == "d probs" and probs_bound is not None:
            assert ((actual[name] - value).abs() <= probs_bound).all(), name
        else:
            assert torch.equal(actual[name], value), name


@pytest.mark.parametrize("moe_act_recompute", [False, True], ids=["plain", "moe-act-recompute"])
def test_grouped_mlp_unfused_fallback_is_megatron_cores_forward(
    dispatch, cpu_checkpoint_without_output, moe_act_recompute
):
    """On CPU the activation falls back to Megatron-Core's operations: the copied forward is bitwise its source."""
    recompute = dict(recompute_granularity="selective", recompute_modules=["moe_act"]) if moe_act_recompute else {}
    config = _config(ernie_fused_situ_prob=True, **recompute)
    install_ernie55_fused_ops(config)
    experts = _grouped_mlp(config, seed=3)
    assert experts.activation_recompute is moe_act_recompute
    inputs = _expert_inputs(seed=4)
    expected = _run_experts(TEGroupedMLP.forward, experts, inputs)
    assert use_ernie55_fused_situ_prob(SimpleNamespace(experts=experts))
    assert type(experts) is Ernie55GroupedMLP
    _assert_matches(_run_experts(Ernie55GroupedMLP.forward, experts, inputs), expected, probs_bound=None)
    assert not dispatch.FUSED_OP_CALLS


@pytest.mark.parametrize("moe_act_recompute", [False, True], ids=["plain", "moe-act-recompute"])
def test_grouped_mlp_fused_branch_matches_megatron_cores_forward(
    dispatch, cpu_checkpoint_without_output, monkeypatch, moe_act_recompute
):
    """The fused branch (the kernel replaced by its PyTorch arithmetic) against Megatron-Core's forward with Bridge's
    SiTU-GLU arithmetic: output, input and weight gradients bitwise, probability gradient within the row bound."""
    recompute = dict(recompute_granularity="selective", recompute_modules=["moe_act"]) if moe_act_recompute else {}
    config = _config(ernie_fused_situ_prob=True, **recompute)
    install_ernie55_fused_ops(config)
    _force_kernel_support(monkeypatch, dispatch)
    monkeypatch.setattr(fused_ops, "fused_situ_glu", _situ)
    monkeypatch.setattr(fused_situ_prob, "fused_situ_glu_prob", _kernel_arithmetic)
    experts = _grouped_mlp(config, seed=5)
    inputs = _expert_inputs(seed=6)

    expected = _run_experts(TEGroupedMLP.forward, experts, inputs)
    # Megatron-Core's activation ran the SiTU-GLU dispatcher (once more under moe_act recompute).
    calls = 2 if moe_act_recompute else 1
    assert dispatch.FUSED_OP_CALLS == {f"situ_glu@{EXPERTS_SITE}": calls}
    dispatch.FUSED_OP_CALLS.clear()

    use_ernie55_fused_situ_prob(SimpleNamespace(experts=experts))
    actual = _run_experts(Ernie55GroupedMLP.forward, experts, inputs)
    assert dispatch.FUSED_OP_CALLS == {"situ_glu_prob@routed_experts": calls}
    hidden, tokens_per_expert, _, out_grad = inputs
    with torch.no_grad():
        fc1_output, _ = experts.linear_fc1(hidden, tokens_per_expert.tolist())
        act = _situ(fc1_output, 4.0, 25.0)
    out_grad_fc2_input = torch.cat(
        [
            chunk @ getattr(experts.linear_fc2, f"weight{i}")
            for i, chunk in enumerate(torch.split(out_grad, tokens_per_expert.tolist()))
        ]
    )
    _assert_matches(actual, expected, probs_bound=_row_bound(out_grad_fc2_input, act))


def test_grouped_mlp_runs_megatron_cores_code_on_its_other_paths(dispatch, monkeypatch):
    """The forward stays Megatron-Core's (op-fuser path, zero-copy buffers, padding, paged stashing), and probabilities
    on the input go to Megatron-Core's ``_unfused_forward`` unchanged."""
    assert Ernie55GroupedMLP.forward is TEGroupedMLP.forward
    config = _config(ernie_fused_situ_prob=True)
    experts = _grouped_mlp(config, seed=7)
    use_ernie55_fused_situ_prob(SimpleNamespace(experts=experts))
    sentinel = torch.zeros(1)
    calls = []

    def mcore_unfused_forward(self, *args):
        calls.append((self, args))
        return sentinel

    monkeypatch.setattr(TEGroupedMLP, "_unfused_forward", mcore_unfused_forward)
    hidden, tokens_per_expert, probs, _ = _expert_inputs(seed=8)
    monkeypatch.setattr(config, "moe_apply_probs_on_input", True)
    assert experts._unfused_forward(hidden, tokens_per_expert, probs) is sentinel
    assert len(calls) == 1
    module, args = calls[0]
    assert module is experts and args[0] is hidden and args[1] is tokens_per_expert and args[2] is probs
    assert not dispatch.FUSED_OP_CALLS


# ---------------------------------------------------------------------------------------------
# Conversion and the forward fingerprint
# ---------------------------------------------------------------------------------------------


def test_conversion(dispatch):
    config = _config(ernie_fused_situ_prob=True)
    experts = _grouped_mlp(config, seed=9)
    state = {name: value.clone() for name, value in experts.state_dict().items()}
    layer = SimpleNamespace(experts=experts)
    assert use_ernie55_fused_situ_prob(layer) and type(experts) is Ernie55GroupedMLP
    assert use_ernie55_fused_situ_prob(layer) and type(experts) is Ernie55GroupedMLP  # idempotent
    after = experts.state_dict()
    assert after.keys() == state.keys() and all(torch.equal(after[name], state[name]) for name in state)

    # Other experts keep their class and the unfused activation.
    sequential = SequentialMLP.__new__(SequentialMLP)
    torch.nn.Module.__init__(sequential)
    assert not use_ernie55_fused_situ_prob(SimpleNamespace(experts=sequential))
    assert type(sequential) is SequentialMLP

    # A TEGroupedMLP subclass this function does not know is an error.
    other = _grouped_mlp(config, seed=9)
    other.__class__ = type("OtherGroupedMLP", (TEGroupedMLP,), {})
    with pytest.raises(TypeError, match="OtherGroupedMLP"):
        use_ernie55_fused_situ_prob(SimpleNamespace(experts=other))


def test_conversion_refuses_an_unknown_megatron_core_forward(monkeypatch):
    experts = _grouped_mlp(_config(ernie_fused_situ_prob=True), seed=10)
    monkeypatch.setattr(moe_module, "te_grouped_mlp_forward_fingerprint", lambda: "0" * 64)
    with pytest.raises(RuntimeError, match="TEGroupedMLP._unfused_forward is not a version"):
        use_ernie55_fused_situ_prob(SimpleNamespace(experts=experts))
    assert type(experts) is TEGroupedMLP


def test_megatron_cores_forward_is_the_copied_one():
    """Megatron-Core's TEGroupedMLP._unfused_forward is the version Ernie55GroupedMLP._unfused_forward copies.

    If this fails after a Megatron-Core update, compare the copy in modeling_ernie55/moe.py with the new method,
    update the copy, and add the new fingerprint to TE_GROUPED_MLP_FORWARD_FINGERPRINTS.
    """
    assert moe_module.te_grouped_mlp_forward_fingerprint() in moe_module.TE_GROUPED_MLP_FORWARD_FINGERPRINTS


def test_fingerprint_ignores_layout_comments_and_docstrings_only():
    normalize = moe_module._normalized_source_text
    source = '''
        def f(x, y):
            """Docstring."""
            # A comment.
            z = x * y  # Trailing comment.
            return (z +
                    1)
    '''
    reformatted = '''
        def f(x, y):
            """Another docstring."""
            z = x * y
            return (z + 1)
    '''
    changed = '''
        def f(x, y):
            """Docstring."""
            z = x * y
            return (z + 2)
    '''
    assert normalize(source) == normalize(reformatted)
    assert normalize(source) != normalize(changed)


# ---------------------------------------------------------------------------------------------
# The MoE layer and its recompute
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_collection(tmp_path_factory):
    """Single-rank gloo process groups; the router allocates its buffers on torch.cuda.current_device()."""
    created_group = not torch.distributed.is_initialized()
    if created_group:
        store = tmp_path_factory.mktemp("ernie55_situ_prob_gloo") / "store"
        torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    created_model_parallel = not parallel_state.model_parallel_is_initialized()
    if created_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch.cuda, "current_device", cpu_current_device)
            yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if created_model_parallel:
            parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def _fresh_moe_metrics():
    saved = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    yield
    moe_logging.set_moe_metrics_tracker(saved)


def _moe_builder(config: Ernie55TransformerConfig, experts: object):
    """The ``E`` layer's MoE builder from ``ernie55_hybrid_stack_spec`` with CPU leaves and the given experts."""
    from tests.unit_tests.models.ernie.test_ernie55_moe import _localize

    builder = _localize(ernie55_hybrid_stack_spec(config).submodules.moe_layer.submodules.mlp)
    submodules = dataclasses.replace(builder.keywords["submodules"], experts=experts)
    return functools.partial(builder.func, *builder.args, **{**builder.keywords, "submodules": submodules})


def _moe(config: Ernie55TransformerConfig, pg_collection, *, experts: object = _GROUPED_EXPERTS) -> Ernie55MoELayer:
    moe = _moe_builder(config, experts)(
        config=config, pg_collection=pg_collection, is_mtp_layer=False, name=None, hash_moe_layer_threshold=None
    )
    moe.set_layer_number(1)
    _randomize_(moe, seed=11)
    return moe.train()


def test_moe_layer_converts_its_experts_with_the_flag(dispatch, pg_collection):
    assert type(_moe(_config(), pg_collection).experts) is TEGroupedMLP
    assert type(_moe(_config(ernie_fused_situ_prob=True), pg_collection).experts) is Ernie55GroupedMLP
    # Experts that are not a TEGroupedMLP keep their class (one warning per process).
    from tests.unit_tests.models.ernie.test_ernie55_moe import _LOCAL_EXPERTS

    moe = _moe(_config(ernie_fused_situ_prob=True), pg_collection, experts=_LOCAL_EXPERTS)
    assert type(moe.experts) is SequentialMLP


@pytest.mark.parametrize("checkpointed", [False, True], ids=["plain", "moe-recompute"])
def test_moe_layer_matches_with_and_without_the_flag(dispatch, monkeypatch, pg_collection, checkpointed):
    """The ERNIE 5.5 MoE layer, flag off and on, through a reentrant checkpoint as with ``"moe"`` recompute.

    The checkpoint runs the layer without grad, then replays it with grad in the backward, as Megatron-Core's
    ``tensor_parallel.checkpoint`` does for ``"moe"`` in ``recompute_modules``; the fused kernel (its PyTorch
    arithmetic here) runs in both passes. The output and every expert gradient are bitwise; the router and input
    gradients, which receive the probability gradient, match to fp32 rounding.
    """
    _force_kernel_support(monkeypatch, dispatch)
    monkeypatch.setattr(fused_ops, "fused_situ_glu", _situ)
    monkeypatch.setattr(fused_situ_prob, "fused_situ_glu_prob", _kernel_arithmetic)
    generator = torch.Generator().manual_seed(12)
    hidden = torch.randn(10, 2, HIDDEN, generator=generator).to(torch.bfloat16)
    out_grad = torch.randn(10, 2, HIDDEN, generator=generator).to(torch.bfloat16)

    def run(flag: bool) -> dict[str, torch.Tensor]:
        config = _config(ernie_fused_situ_prob=flag)
        install_ernie55_fused_ops(config)
        moe = _moe(config, pg_collection)
        assert type(moe.experts) is (Ernie55GroupedMLP if flag else TEGroupedMLP)
        x = hidden.clone().requires_grad_(True)
        if checkpointed:
            output = torch.utils.checkpoint.checkpoint(lambda h: moe(h)[0], x, use_reentrant=True)
        else:
            output = moe(x)[0]
        output.backward(out_grad)
        result = {"output": output.detach(), "d input": x.grad}
        result.update({f"d {name}": param.grad for name, param in moe.named_parameters()})
        return result

    passes = 2 if checkpointed else 1
    expected = run(False)
    # The shared expert's SiTU-GLU goes through the MLP site's dispatcher in both runs.
    assert dispatch.FUSED_OP_CALLS == {f"situ_glu@{EXPERTS_SITE}": passes, f"situ_glu@{MLP_SITE}": passes}
    dispatch.FUSED_OP_CALLS.clear()
    actual = run(True)
    assert dispatch.FUSED_OP_CALLS == {"situ_glu_prob@routed_experts": passes, f"situ_glu@{MLP_SITE}": passes}
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        assert actual[name].dtype == value.dtype, name
        if name.startswith("d router.") or name == "d input":
            torch.testing.assert_close(actual[name].float(), value.float(), rtol=1e-3, atol=1e-6, msg=name)
        else:
            assert torch.equal(actual[name], value), name


# ---------------------------------------------------------------------------------------------
# Config envelope
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides,message",
    [
        (dict(ernie_fused_ops=False), "set ernie_fused_ops=True"),
        (dict(activation_func_tanh_clamp_scale=None, activation_func_tanh_clamp_scale_linear=None), "SiTU-GLU"),
        (dict(bias_activation_fusion=True), "bias_activation_fusion must be False"),
    ],
)
def test_flag_envelope(overrides, message):
    with pytest.raises(ValueError, match=rf"^ERNIE 5\.5 MoE: ernie_fused_situ_prob=True .*{message}"):
        _config(ernie_fused_situ_prob=True, **overrides)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("use_te_activation_func", True, "use_te_activation_func"),
        ("moe_mlp_glu_interleave_size", 8, "moe_mlp_glu_interleave_size"),
        ("moe_apply_probs_on_input", True, "moe_apply_probs_on_input"),
    ],
)
def test_flag_envelope_covers_fields_changed_after_finalize(field, value, message):
    """Fields that Megatron-Core's own construction checks would reject together with SiTU-GLU first."""
    from megatron.bridge.models.ernie.modeling_ernie55 import validate_ernie55_moe_config

    config = _config(ernie_fused_situ_prob=True)
    assert hasattr(config, field)
    setattr(config, field, value)
    with pytest.raises(ValueError, match=rf"^ERNIE 5\.5 MoE: ernie_fused_situ_prob=True .*{message}"):
        validate_ernie55_moe_config(config)


def test_flag_is_off_by_default_and_must_be_a_bool():
    assert _config().ernie_fused_situ_prob is False
    with pytest.raises(ValueError, match="ernie_fused_situ_prob must be a bool"):
        _config(ernie_fused_situ_prob=1)
    # Without the flag the routed experts stay Megatron-Core's (no swap, no copied forward).
    assert mcore_experts.TEGroupedMLP is TEGroupedMLP
