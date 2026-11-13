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

"""CPU tests of the ERNIE 5.5 MoE stack against an independent fp64 re-derivation of its math.

The oracle in this file is written from the ERNIE 5.5 MoE specification, not from the Bridge
implementation. Per token ``x`` (the pre-MLP norm output), with ``E`` experts and top-``k`` routing:

* scores ``s = sigmoid(x W^T) + sigmoid(x W_1^T)``;
* selection ``I = top_k(s + b)`` with the expert bias ``b``, and unbiased gate values ``p = s[I]``;
* ``p = p / (sum(p) + 1e-20)`` when ``norm_topk_prob`` and ``k > 1``, then ``p = p * rho[I]``;
* ``y = sum_{e in I} p_e W2_e act(W1_e x) + W2_s act(W1_s x)`` with the ungated shared expert, where
  ``act([g | u])`` is SwiGLU ``silu(g) * u`` or SiTU-GLU ``beta tanh(g / beta) sigmoid(g) * u'`` with
  ``u' = linear_beta tanh(u / linear_beta)``, or ``u' = u`` when ``linear_beta`` is None;
* latent MoE: the experts act on ``x W_L1^T`` and the combined output passes the optional RMSNorm and
  ``W_L2`` before the shared output is added;
* expert bias update ``b <- b + rate * sign(mean(c) - c)`` from the exact token counts ``c``.

Everything runs on CPU with Transformer-Engine-free stand-ins: local linear layers, ``SequentialMLP``
experts and the allgather dispatcher (the alltoall dispatcher needs CUDA streams; the GPU file covers it
with grouped experts at EP4).
"""

import dataclasses
import functools
import math
import re
import zlib
from unittest import mock

import pytest
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _update_router_expert_bias, reset_model_temporary_tensors
from megatron.core.extensions import transformer_engine as te_extensions
from megatron.core.optimizer.emerging_optimizers import _is_muon_excluded
from megatron.core.optimizer.layer_wise_optimizer import is_managed_by_layer_wise_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.moe import moe_layer as mcore_moe_layer
from megatron.core.transformer.moe import moe_logging
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    Ernie55LatentProjection,
    Ernie55MoELayer,
    Ernie55SplitFeatureRouter,
    downcasting_norm,
    ernie55_hybrid_stack_spec,
    validate_ernie55_moe_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.stack_spec import _copy_spec_tree
from tests.unit_tests.models.ernie.ernie55_test_utils import cpu_current_device


pytestmark = pytest.mark.unit

HIDDEN = 16
EXPERTS = 8
TOPK = 4
MOE_FFN = 8
SHARED_FFN = 12
DENSE_FFN = 24
LATENT = 12
EPS = 1e-6
RATE = 1e-4
SEQ, BATCH = 12, 2

# fp32 module vs fp64 oracle, as a normalized max error (max |diff| / max |expected|).
TOL = 2e-5
# Routing probabilities, absolute.
PROB_TOL = 1e-6
# Selection margin below which an fp32 forward could legitimately disagree with the fp64 oracle.
SAFE_MARGIN = 1e-4

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


def _reference_dense_manifest(hidden: int, ffn: int) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Dense layer 0 of the reference proxy, relative to the layer (sizes as arguments)."""
    return {
        "pre_mlp_layernorm.weight": ((hidden,), torch.bfloat16),
        "mlp.linear_fc1.weight": ((2 * ffn, hidden), torch.bfloat16),
        "mlp.linear_fc2.weight": ((hidden, ffn), torch.bfloat16),
    }


def _reference_moe_manifest(
    *, hidden: int, experts: int, local_experts: int, ffn: int, shared_ffn: int, token_count_dtype: torch.dtype
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """MoE layer 1 of the reference proxy (parameters and buffers), relative to the layer.

    Parameter and buffer names, shapes and dtypes of the reference implementation's MoE layer (one rank
    at EP4), with the proxy sizes as arguments. Its ``local_tokens_per_expert`` buffer is bf16
    (``token_count_dtype=torch.bfloat16``); Megatron-Core's is exact int64, a documented difference
    (``modeling_ernie55.moe``).
    """
    manifest = {
        "pre_mlp_layernorm.weight": ((hidden,), torch.bfloat16),
        "mlp.router.weight": ((experts, hidden), torch.float32),
        "mlp.router.weight_1": ((experts, hidden), torch.float32),
        "mlp.router.routed_scaling_factor_param": ((experts,), torch.float32),
        "mlp.router.expert_bias": ((experts,), torch.float32),
        "mlp.router.local_tokens_per_expert": ((experts,), token_count_dtype),
        "mlp.shared_experts.linear_fc1.weight": ((2 * shared_ffn, hidden), torch.bfloat16),
        "mlp.shared_experts.linear_fc2.weight": ((hidden, shared_ffn), torch.bfloat16),
    }
    for expert in range(local_experts):
        manifest[f"mlp.experts.linear_fc1.weight{expert}"] = ((2 * ffn, hidden), torch.bfloat16)
        manifest[f"mlp.experts.linear_fc2.weight{expert}"] = ((hidden, ffn), torch.bfloat16)
    return manifest


# ---------------------------------------------------------------------------------------------
# Runtime fixtures and builders
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pg_collection(tmp_path_factory):
    """Single-rank gloo process groups; the router allocates its buffers on torch.cuda.current_device()."""
    created_group = not torch.distributed.is_initialized()
    if created_group:
        store = tmp_path_factory.mktemp("ernie55_moe_gloo") / "store"
        torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    created_model_parallel = not parallel_state.model_parallel_is_initialized()
    if created_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            # These tests run on CPU even when a GPU is present (torch's own callers still see the GPU).
            patch.setattr(torch.cuda, "current_device", cpu_current_device)
            yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if created_model_parallel:
            parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def _fresh_moe_metrics():
    """Give every test an empty Megatron-Core MoE metrics tracker."""
    saved = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    yield
    moe_logging.set_moe_metrics_tracker(saved)


def _config(**overrides) -> Ernie55TransformerConfig:
    """Finalized ERNIE 5.5 config with fp32 parameters and the small MoE geometry above."""
    values = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        transformer_impl="local",
        bf16=True,
        params_dtype=torch.float32,
        ffn_hidden_size=DENSE_FFN,
        gated_linear_unit=True,
        activation_func=F.silu,
        num_moe_experts=EXPERTS,
        moe_router_topk=TOPK,
        moe_ffn_hidden_size=MOE_FFN,
        moe_shared_expert_intermediate_size=SHARED_FFN,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=RATE,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="allgather",
        **_D_LAYER_FIELDS,
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _generator(seed: int, name: str) -> torch.Generator:
    return torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    """Draw every parameter and the expert bias at O(1) scale, keyed by name (order independent)."""
    with torch.no_grad():
        for name, param in module.named_parameters():
            noise = torch.randn(param.shape, generator=_generator(seed, name), dtype=torch.float64)
            if name.endswith("routed_scaling_factor_param"):
                values = 1.0 + 0.3 * noise.clamp(-1.5, 1.5)
            elif "norm" in name:
                values = 1.0 + 0.2 * noise
            else:
                values = noise / math.sqrt(param.shape[-1])
            param.copy_(values.to(param.dtype))
        for name, buffer in module.named_buffers():
            if name.endswith("expert_bias"):
                noise = torch.randn(buffer.shape, generator=_generator(seed, name), dtype=torch.float64)
                buffer.copy_((0.05 * noise).to(buffer.dtype))


def _router(config: Ernie55TransformerConfig, pg_collection, *, seed: int | None = None) -> Ernie55SplitFeatureRouter:
    router = Ernie55SplitFeatureRouter(config, pg_collection=pg_collection)
    router.set_layer_number(1)
    if seed is not None:
        _randomize_(router, seed)
    return router.train()


class _DuplicatedLinear(torch.nn.Module):
    """Stand-in for ``TELinear(parallel_mode="duplicated")`` that Megatron-Core builds for latent MoE."""

    def __init__(self, input_size: int, output_size: int, *, config, init_method, bias: bool, **kwargs) -> None:
        super().__init__()
        assert kwargs.get("parallel_mode") == "duplicated" and not bias
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        init_method(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None


_LOCAL_EXPERTS = functools.partial(
    SequentialMLP, submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)
)


def _localize(node: object) -> object:
    """Copy a spec tree with its Transformer Engine leaves replaced by CPU equivalents.

    ``TENorm`` becomes ``WrappedTorchNorm``, the TE column/row linears become the local ones, and the
    routed experts (``TEGroupedMLP`` with TE, ``SequentialMLP`` without) become ``SequentialMLP`` with
    local linears. Every other class, including the ERNIE modules the spec installs, is kept.
    """
    leaves = {
        id(te_extensions.TENorm): WrappedTorchNorm,
        id(te_extensions.TEColumnParallelLinear): ColumnParallelLinear,
        id(te_extensions.TERowParallelLinear): RowParallelLinear,
    }

    def walk(item: object) -> object:
        if id(item) in leaves:
            return leaves[id(item)]
        if isinstance(item, functools.partial):
            if item.func in (SequentialMLP, TEGroupedMLP):
                return _LOCAL_EXPERTS
            return functools.partial(item.func, *walk(item.args), **walk(item.keywords))
        if dataclasses.is_dataclass(item) and not isinstance(item, type):
            changes = {f.name: walk(getattr(item, f.name)) for f in dataclasses.fields(item) if f.init}
            return dataclasses.replace(item, **changes)
        if isinstance(item, dict):
            return {key: walk(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return type(item)(walk(value) for value in item)
        return item

    return walk(node)


def _moe(config: Ernie55TransformerConfig, pg_collection, *, seed: int) -> Ernie55MoELayer:
    """Build the ``E`` layer's MoE module from ``ernie55_hybrid_stack_spec`` with CPU stand-ins."""
    builder = _localize(ernie55_hybrid_stack_spec(config).submodules.moe_layer.submodules.mlp)
    moe = builder(
        config=config, pg_collection=pg_collection, is_mtp_layer=False, name=None, hash_moe_layer_threshold=None
    )
    moe.set_layer_number(1)
    _randomize_(moe, seed)
    return moe.train()


@pytest.fixture
def latent_moe_without_te(monkeypatch):
    """Let Megatron-Core build its latent-MoE modules without Transformer Engine.

    ``MoELayer`` asserts TE for latent projections and builds them as ``TELinear`` (which
    :class:`Ernie55MoELayer` then replaces) plus a ``TENorm`` for ``fc2_norm``.
    """
    monkeypatch.setattr(mcore_moe_layer, "HAVE_TE", True)
    monkeypatch.setattr(mcore_moe_layer, "TELinear", _DuplicatedLinear, raising=False)
    monkeypatch.setattr(mcore_moe_layer, "TENorm", WrappedTorchNorm, raising=False)


# ---------------------------------------------------------------------------------------------
# fp64 oracle
# ---------------------------------------------------------------------------------------------


def _swiglu(z: torch.Tensor) -> torch.Tensor:
    gate, up = z.chunk(2, dim=-1)
    return F.silu(gate) * up


def _situ_glu(z: torch.Tensor, *, beta: float, linear_beta: float | None) -> torch.Tensor:
    gate, up = z.chunk(2, dim=-1)
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return gate * up


def _oracle_activation(config: Ernie55TransformerConfig):
    beta = config.activation_func_tanh_clamp_scale
    if beta is None:
        return _swiglu
    return functools.partial(_situ_glu, beta=beta, linear_beta=config.activation_func_tanh_clamp_scale_linear)


def _oracle_scores(x: torch.Tensor, weight: torch.Tensor, weight_1: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x @ weight.T) + torch.sigmoid(x @ weight_1.T)


def _oracle_routing(
    scores: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor, *, topk: int, normalize: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """fp64 routing: probabilities ``[N, E]``, the boolean map, and each row's selection margin."""
    biased = scores + bias
    order = torch.argsort(biased, dim=-1, descending=True, stable=True)
    selected = order[:, :topk]
    gates = scores.gather(-1, selected)
    if normalize and topk > 1:
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-20)
    gates = gates * scale[selected]
    probs = torch.zeros_like(scores).scatter(-1, selected, gates)
    routing_map = torch.zeros_like(scores, dtype=torch.bool).scatter(-1, selected, True)
    ranked = biased.gather(-1, order)
    margin = ranked[:, topk - 1] - ranked[:, topk]
    return probs, routing_map, margin


def _router_tensors(router: Ernie55SplitFeatureRouter) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """fp64 leaf copies of the router parameters (by name, gradients enabled) and of the expert bias."""
    params = {name: param.detach().double().requires_grad_(True) for name, param in router.named_parameters()}
    return params, router.expert_bias.detach().double()


def _mlp_tensors(mlp: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        mlp.linear_fc1.weight.detach().double().requires_grad_(True),
        mlp.linear_fc2.weight.detach().double().requires_grad_(True),
    )


def _oracle_mlp(x: torch.Tensor, fc1: torch.Tensor, fc2: torch.Tensor, activation) -> torch.Tensor:
    return activation(x @ fc1.T) @ fc2.T


def _oracle_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, what: str) -> None:
    actual, expected = actual.detach().double(), expected.detach().double()
    assert actual.shape == expected.shape, f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    error = ((actual - expected).abs().max() / expected.abs().max().clamp_min(1e-30)).item()
    assert error <= tol, f"{what}: normalized max error {error:.3e} > {tol:.1e}"


def _planted_near_tie_scores(bias: torch.Tensor, *, rows: int, topk: int, margin: float, seed: int) -> torch.Tensor:
    """fp32 scores with a near-tie of ``+-margin`` between the k-th and (k+1)-th biased values of every row.

    Each row picks a selected set ``S`` and, as its weakest member, the expert ``w in S`` with the largest
    bias; the strongest unselected expert ``c`` has the smallest bias outside ``S``. Then
    ``s_w + b_w = s_c + b_c +- margin`` while ``s_c - s_w = b_w - b_c -+ margin > 0``: on even rows the bias
    alone puts ``w`` ahead of ``c``, on odd rows ``c`` wins by ``margin``. The other experts are at least
    0.02 away from the boundary, and every score lies in ``(0, 2)``.
    """
    experts = bias.numel()
    generator = torch.Generator().manual_seed(seed)
    bias = bias.double()
    out = torch.empty(rows, experts, dtype=torch.float64)
    for row in range(rows):
        while True:
            permutation = torch.randperm(experts, generator=generator)
            selected, rest = permutation[:topk], permutation[topk:]
            weakest = selected[torch.argmax(bias[selected])]
            challenger = rest[torch.argmin(bias[rest])]
            if bias[weakest] - bias[challenger] > 10 * margin:
                break
        base = 0.6 + 0.6 * torch.rand((), generator=generator, dtype=torch.float64)
        delta = margin if row % 2 == 0 else -margin
        biased = torch.empty(experts, dtype=torch.float64)
        biased[selected] = base + delta + 0.02 + 0.3 * torch.rand(topk, generator=generator, dtype=torch.float64)
        biased[rest] = base - 0.02 - 0.3 * torch.rand(experts - topk, generator=generator, dtype=torch.float64)
        biased[weakest] = base + delta
        biased[challenger] = base
        out[row] = biased - bias
    return out.float()


# ---------------------------------------------------------------------------------------------
# Router: routing, gating, z-loss
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("topk", [1, 4])
@pytest.mark.parametrize("normalize", [True, False])
def test_routing_matches_oracle_on_planted_near_ties(pg_collection, topk, normalize):
    """Exact selection sets at 1e-6 near-ties; the bias only selects; normalization and the scale follow."""
    experts, rows, margin = 16, 64, 1e-6
    config = _config(num_moe_experts=experts, moe_router_topk=topk, ernie_norm_topk_prob=normalize)
    router = _router(config, pg_collection, seed=3)
    with torch.no_grad():
        jitter = torch.rand(experts, generator=torch.Generator().manual_seed(topk))
        router.expert_bias.copy_(torch.linspace(-0.08, 0.08, experts) + 0.004 * jitter)
    scores = _planted_near_tie_scores(router.expert_bias, rows=rows, topk=topk, margin=margin, seed=topk)
    state, bias = _router_tensors(router)

    probs, routing_map = router.routing(scores)
    expected_probs, expected_map, row_margin = _oracle_routing(
        scores.double(), bias, state["routed_scaling_factor_param"].detach(), topk=topk, normalize=normalize
    )

    # The fixture really plants near-ties, and the bias decides the boundary in half of the rows.
    assert bool((row_margin.abs() > 0.5 * margin).all() and (row_margin.abs() < 2 * margin).all())
    unbiased_map = torch.zeros_like(expected_map).scatter(-1, scores.double().topk(topk, dim=-1).indices, True)
    assert (unbiased_map != expected_map).any(dim=-1).sum() >= rows // 4
    # A gate that used the biased scores would be off by far more than the tolerance.
    biased_gates, _, _ = _oracle_routing(
        scores.double() + bias, bias, state["routed_scaling_factor_param"].detach(), topk=topk, normalize=normalize
    )
    assert (biased_gates - expected_probs).abs().max() > 1e3 * PROB_TOL

    assert routing_map.dtype == torch.bool and probs.dtype == torch.float32
    assert torch.equal(routing_map, expected_map)
    assert (probs.double() - expected_probs).abs().max().item() <= PROB_TOL


def test_gating_is_an_fp32_split_feature_sigmoid_sum(pg_collection):
    """Two fp32 GEMMs on the bf16 router input, kept in fp32 under autocast, with fp32 gradients."""
    router = _router(_config(), pg_collection, seed=5)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(1)).bfloat16()
    hidden.requires_grad_(True)
    state, _ = _router_tensors(router)

    scores = router.gating(hidden)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_scores = router.gating(hidden.detach())
    expected = _oracle_scores(hidden.detach().double(), state["weight"], state["weight_1"])

    assert scores.dtype == torch.float32 and scores.shape == (SEQ, BATCH, EXPERTS)
    assert torch.equal(autocast_scores, scores.detach())
    assert (scores.double() - expected).abs().max().item() <= PROB_TOL
    assert bool(((scores > 0) & (scores < 2)).all())

    cotangent = torch.randn(scores.shape, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    scores.backward(cotangent.float())
    expected.backward(cotangent)
    assert router.weight.grad.dtype == torch.float32 and router.weight_1.grad.dtype == torch.float32
    _assert_close(router.weight.grad, state["weight"].grad, TOL, "d weight")
    _assert_close(router.weight_1.grad, state["weight_1"].grad, TOL, "d weight_1")
    assert hidden.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("normalize", [True, False])
def test_router_forward_and_gradients_match_oracle(pg_collection, normalize):
    """Full router forward on random input; gradients reach weight, weight_1 and the per-expert scale."""
    router = _router(_config(ernie_norm_topk_prob=normalize), pg_collection, seed=11)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    hidden_module = hidden.float().requires_grad_(True)
    hidden_oracle = hidden_module.detach().double().requires_grad_(True)
    state, bias = _router_tensors(router)

    probs, routing_map = router(hidden_module)
    scores = _oracle_scores(hidden_oracle.reshape(-1, HIDDEN), state["weight"], state["weight_1"])
    expected_probs, expected_map, row_margin = _oracle_routing(
        scores, bias, state["routed_scaling_factor_param"], topk=TOPK, normalize=normalize
    )
    assert row_margin.min().item() > SAFE_MARGIN, "the seed planted an accidental near-tie"
    assert torch.equal(routing_map, expected_map)
    assert (probs.double() - expected_probs.detach()).abs().max().item() <= PROB_TOL

    cotangent = torch.randn(probs.shape, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
    (probs * cotangent.float()).sum().backward()
    (expected_probs * cotangent).sum().backward()
    _assert_close(router.weight.grad, state["weight"].grad, TOL, "d weight")
    _assert_close(router.weight_1.grad, state["weight_1"].grad, TOL, "d weight_1")
    _assert_close(
        router.routed_scaling_factor_param.grad,
        state["routed_scaling_factor_param"].grad,
        TOL,
        "d routed_scaling_factor_param",
    )
    _assert_close(hidden_module.grad, hidden_oracle.grad, TOL, "d input")
    assert router.expert_bias.grad is None


def test_z_loss_is_taken_on_the_scores(pg_collection):
    """The tracked z-loss is mean((logsumexp s)^2) of the sigmoid-sum scores; coefficient 0 adds no gradient."""
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(6))
    router = _router(_config(moe_z_loss_coeff=1e-2), pg_collection, seed=13)
    router(hidden)
    scores = _oracle_scores(hidden.double().reshape(-1, HIDDEN), router.weight.double(), router.weight_1.double())
    expected = torch.logsumexp(scores, dim=-1).square().mean()
    tracked = moe_logging.get_moe_metrics_tracker().metrics["z_loss"].values[0]
    assert abs(tracked.item() - expected.item()) <= 1e-6 * expected.item()

    grads = {}
    for coeff in (None, 0.0):
        router = _router(_config(moe_z_loss_coeff=coeff), pg_collection, seed=13)
        probs, _ = router(hidden)
        probs.square().sum().backward()
        grads[coeff] = (router.weight.grad.clone(), router.weight_1.grad.clone())
    for without, with_zero in zip(grads[None], grads[0.0]):
        torch.testing.assert_close(with_zero, without, rtol=0, atol=0)


def test_z_loss_excludes_padding(pg_collection):
    """With a [s, b] padding mask, the tracked z-loss is the mean of (logsumexp s)^2 over valid tokens only."""
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(10))
    padding = torch.zeros(SEQ, BATCH, dtype=torch.bool)
    padding[-4:, 0] = True
    padding[-1:, 1] = True
    # Larger router inputs on the padding, so that counting it would move the mean far beyond the tolerance.
    hidden[padding] = 8.0 * hidden[padding]
    router = _router(_config(moe_z_loss_coeff=1e-2), pg_collection, seed=13)
    router(hidden, padding_mask=padding)

    scores = _oracle_scores(hidden.double().reshape(-1, HIDDEN), router.weight.double(), router.weight_1.double())
    z_values = torch.logsumexp(scores, dim=-1).square()
    expected = z_values[~padding.reshape(-1)].mean()
    assert abs(z_values.mean().item() - expected.item()) > 1e-3 * expected.item()
    tracked = moe_logging.get_moe_metrics_tracker().metrics["z_loss"].values[0]
    assert abs(tracked.item() - expected.item()) <= 1e-6 * expected.item()


# ---------------------------------------------------------------------------------------------
# Router: token counts and the expert-bias update
# ---------------------------------------------------------------------------------------------


def _oracle_counts(routing_map: torch.Tensor, padding: torch.Tensor | None = None) -> torch.Tensor:
    if padding is not None:
        routing_map = routing_map & ~padding.reshape(-1, 1)
    return routing_map.sum(dim=0).to(torch.int64)


def test_tokens_are_counted_only_under_grad(pg_collection):
    """No-grad passes count nothing; a grad-enabled pass counts each selection once, without padding."""
    router = _router(_config(), pg_collection, seed=17)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(7))
    padding = torch.zeros(SEQ, BATCH, dtype=torch.bool)
    padding[-3:, 1] = True

    with torch.no_grad():
        _, routing_map = router(hidden)
    assert router.local_tokens_per_expert.dtype == torch.int64
    assert not router.local_tokens_per_expert.any()

    _, routing_map = router(hidden)
    expected = _oracle_counts(routing_map)
    assert torch.equal(router.local_tokens_per_expert, expected)

    router.local_tokens_per_expert.zero_()
    _, routing_map = router(hidden, padding_mask=padding)
    assert torch.equal(router.local_tokens_per_expert, _oracle_counts(routing_map, padding))
    assert int(router.local_tokens_per_expert.sum()) == TOPK * (SEQ * BATCH - 3)


def test_full_recompute_counts_each_token_once(pg_collection):
    """Reentrant checkpointing runs the forward without grad and replays it with grad: one count per token."""
    router = _router(_config(), pg_collection, seed=19)
    hidden = torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(8)).requires_grad_(True)

    def route(x: torch.Tensor) -> torch.Tensor:
        return router(x)[0]

    probs = torch.utils.checkpoint.checkpoint(route, hidden, use_reentrant=True)
    assert not router.local_tokens_per_expert.any()
    probs.square().sum().backward()
    with torch.no_grad():
        _, routing_map = router(hidden)
    assert torch.equal(router.local_tokens_per_expert, _oracle_counts(routing_map))
    assert router.weight_1.grad is not None and router.weight_1.grad.abs().sum() > 0


def test_expert_bias_update_follows_the_sign_rule(pg_collection):
    """Megatron-Core's finalize helper on gloo: b += rate * sign(mean(c) - c), ties unchanged, counts reset."""
    experts, topk = 8, 2
    config = _config(num_moe_experts=experts, moe_router_topk=topk)
    router = _router(config, pg_collection, seed=23)
    initial_bias = torch.tensor([0.03, -0.02, 0.01, 0.0, -0.04, 0.02, -0.01, 0.05])
    with torch.no_grad():
        router.expert_bias.copy_(initial_bias)
    # Selections with clear margins; per-expert counts [3, 1, 2, 2, 4, 0, 2, 2] around a mean of 2.
    selections = [(0, 1), (0, 4), (0, 2), (2, 3), (3, 4), (4, 6), (4, 7), (6, 7)]
    scores = torch.full((len(selections), experts), 0.2)
    for row, chosen in enumerate(selections):
        scores[row, list(chosen)] = 1.5
    counts = torch.tensor([3, 1, 2, 2, 4, 0, 2, 2])

    router.routing(scores.requires_grad_(True))
    assert torch.equal(router.local_tokens_per_expert, counts)

    _update_router_expert_bias([router], config, tp_dp_cp_group=pg_collection.tp_dp_cp)
    direction = torch.sign(counts.double().mean() - counts.double())
    assert router.expert_bias.dtype == torch.float32
    # The update is applied in fp32: rounding b + fp32(rate) for |b + rate| < 2^-4 stays below 2^-28, while a
    # wrong or missing update is off by at least 1e-4.
    torch.testing.assert_close(
        router.expert_bias.double(), initial_bias.double() + RATE * direction, rtol=0, atol=2**-28
    )
    tied = direction == 0
    assert tied.sum() == 4 and torch.equal(router.expert_bias[tied], initial_bias[tied])

    reset_model_temporary_tensors(config, [router])
    assert not router.local_tokens_per_expert.any()


def test_zero_update_rate_freezes_the_bias(pg_collection):
    """Rate 0: the bias still selects, but nothing is counted and the update needs no collective."""
    config = _config(moe_router_bias_update_rate=0.0)
    router = _router(config, pg_collection, seed=29)
    assert router.frozen_expert_bias
    before = router.expert_bias.clone()
    router(torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(9)))
    assert not router.local_tokens_per_expert.any()
    with mock.patch("torch.distributed.all_reduce", side_effect=AssertionError("a frozen bias needs no collective")):
        _update_router_expert_bias([router], config, tp_dp_cp_group=pg_collection.tp_dp_cp)
    torch.testing.assert_close(router.expert_bias, before, rtol=0, atol=0)
    assert not _router(_config(), pg_collection).frozen_expert_bias


# ---------------------------------------------------------------------------------------------
# Router: precision, optimizer routing, envelope
# ---------------------------------------------------------------------------------------------


def test_router_parameters_and_bias_stay_fp32_under_float16_module(pg_collection):
    """weight, weight_1 and the scale are fp32 keep_in_fp32 params; a loaded bias survives a direct bf16 wrap."""
    config = _config(params_dtype=torch.bfloat16, moe_router_topk_scaling_factor=2.5)
    router = _router(config, pg_collection)
    params = dict(router.named_parameters())
    assert sorted(params) == ["routed_scaling_factor_param", "weight", "weight_1"]
    assert params["weight"].shape == params["weight_1"].shape == (EXPERTS, HIDDEN)
    assert torch.equal(params["routed_scaling_factor_param"], torch.full((EXPERTS,), 2.5))
    for name, param in params.items():
        assert param.dtype == torch.float32 and getattr(param, "keep_in_fp32", False), name
        # Like Megatron-Core's router weight, so that sequence-parallel gradients are reduced alike.
        assert getattr(param, "sequence_parallel", None) is config.sequence_parallel, name
    assert router.local_tokens_per_expert.dtype == torch.int64
    assert sorted(router.state_dict()) == ["expert_bias", "routed_scaling_factor_param", "weight", "weight_1"]

    # Bridge's model provider restores expert_bias around its own wrap; keep_in_fp32 also covers a direct
    # Float16Module wrap after a checkpoint load, where bf16 would round this bias.
    loaded_bias = torch.full((EXPERTS,), 12.3456789, dtype=torch.float32)
    assert not torch.equal(loaded_bias.bfloat16().float(), loaded_bias)
    with torch.no_grad():
        router.expert_bias.copy_(loaded_bias)
    wrapped = Float16Module(config, router).module
    assert wrapped.expert_bias.dtype == torch.float32 and torch.equal(wrapped.expert_bias, loaded_bias)
    for param in wrapped.parameters():
        assert param.dtype == torch.float32
    assert wrapped.local_tokens_per_expert.dtype == torch.int64


@pytest.mark.parametrize("skip_muon", [False, True])
def test_gate_matrices_are_muon_parameters(pg_collection, skip_muon):
    """Both 2-D gate matrices go to Muon (unless moe_router_skip_muon); the per-expert scale goes to Adam."""
    router = _router(_config(moe_router_skip_muon=skip_muon), pg_collection)
    for name in ("weight", "weight_1"):
        param = getattr(router, name)
        assert is_managed_by_layer_wise_optimizer(param) is not skip_muon, name
        assert _is_muon_excluded(param) is skip_muon, name
    assert not is_managed_by_layer_wise_optimizer(router.routed_scaling_factor_param)
    assert _is_muon_excluded(router.routed_scaling_factor_param)


_CONFIG_FIELDS = {config_field.name for config_field in dataclasses.fields(Ernie55TransformerConfig)}


@pytest.mark.parametrize(
    "overrides,message",
    [
        (dict(moe_router_score_function="sqrtsoftplus"), "moe_router_score_function"),
        (dict(moe_router_enable_expert_bias=False), "moe_router_enable_expert_bias"),
        (dict(moe_router_bias_update_rate=-1e-4), "moe_router_bias_update_rate"),
        (dict(moe_router_bias_update_rate=math.nan), "moe_router_bias_update_rate"),
        (dict(moe_router_num_groups=2, moe_router_group_topk=1), "group-limited"),
        (dict(moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=1e-3), "moe_aux_loss_coeff"),
        (dict(moe_router_load_balancing_type="sinkhorn"), "sinkhorn"),
        (dict(moe_router_dtype="fp64"), "moe_router_dtype"),
        (dict(moe_router_topk_scaling_factor=math.inf), "moe_router_topk_scaling_factor"),
        (dict(moe_enable_routing_replay=True), "moe_enable_routing_replay"),
        (dict(moe_router_fusion=True), "moe_router_fusion"),
        (dict(moe_num_hash_layers=1, hash_moe_vocab_size=128), "hash routing"),
        (dict(moe_router_force_load_balancing=True), "moe_router_force_load_balancing"),
        (dict(moe_router_force_biased=0.1), "moe_router_force_biased"),
        (dict(moe_shared_expert_gate=True), "moe_shared_expert_gate"),
        (dict(gated_linear_unit=False), "gated"),
        (dict(activation_func=F.gelu), "gated"),
        (dict(activation_func_clamp_value=7.0), "activation_func_clamp_value"),
        (dict(glu_linear_offset=1.0), "glu_linear_offset"),
        pytest.param(
            dict(moe_latent_up_projection_rmsnorm=True, moe_latent_size=LATENT),
            "moe_latent_up_projection_rmsnorm",
            marks=pytest.mark.skipif(
                "moe_latent_up_projection_rmsnorm" not in _CONFIG_FIELDS,
                reason="this Megatron-Core has no moe_latent_up_projection_rmsnorm",
            ),
            id="moe_latent_up_projection_rmsnorm",
        ),
    ],
)
def test_moe_envelope_is_validated(overrides, message):
    with pytest.raises(ValueError, match=rf"^ERNIE 5\.5 MoE: .*{message}"):
        _config(**overrides)


@pytest.mark.parametrize(
    "field,value,message",
    [
        # Megatron-Core's DSA check rejects linear biases first. With a later change the router would
        # otherwise get a bias parameter that it never initializes or reads.
        ("add_bias_linear", True, "add_bias_linear"),
        # Megatron-Core dev rejects quantile balancing together with the expert bias first.
        ("moe_router_load_balancing_type", "quantile_balancing", "quantile_balancing"),
    ],
)
def test_moe_envelope_also_covers_fields_changed_after_finalize(pg_collection, field, value, message):
    """Guards that Megatron-Core's construction checks shadow still reject later changes, in the router too."""
    config = _config()
    assert hasattr(config, field)
    setattr(config, field, value)
    with pytest.raises(ValueError, match=rf"^ERNIE 5\.5 MoE: .*{message}"):
        validate_ernie55_moe_config(config)
    with pytest.raises(ValueError, match=rf"^ERNIE 5\.5 MoE: .*{message}"):
        Ernie55SplitFeatureRouter(config, pg_collection=pg_collection)


def test_router_rejects_dispatchers_that_expect_dense_indices(pg_collection):
    """Virtual expert load balancing consumes dense top-k indices; this router returns a routing map."""
    config = _config()
    # Megatron-Core rejects this together with the expert bias at construction, so set it afterwards.
    config.moe_virtual_expert_load_balance = True
    with pytest.raises(ValueError, match=r"^ERNIE 5\.5 MoE: .*dense top-k indices"):
        Ernie55SplitFeatureRouter(config, pg_collection=pg_collection)


def test_router_validates_its_config(pg_collection):
    """The router checks the envelope itself, also for configs that never ran Ernie55TransformerConfig.finalize."""
    config = _config()
    config.moe_expert_capacity_factor = 1.0
    with pytest.raises(ValueError, match="dropless"):
        Ernie55SplitFeatureRouter(config, pg_collection=pg_collection)
    config = _config()
    config.ernie_norm_topk_prob = None
    with pytest.raises(ValueError, match="ernie_norm_topk_prob"):
        validate_ernie55_moe_config(config)


# ---------------------------------------------------------------------------------------------
# MoE layer: full forward and backward, latent MoE, SiTU
# ---------------------------------------------------------------------------------------------


def _oracle_moe(
    moe: Ernie55MoELayer, config: Ernie55TransformerConfig, hidden: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """fp64 MoE layer output for ``hidden`` ``[s, b, h]``, its leaf tensors (by module name) and row margins."""
    activation = _oracle_activation(config)
    leaves = {"input": hidden.detach().double().requires_grad_(True)}
    router, bias = _router_tensors(moe.router)
    leaves.update({f"router.{name}": value for name, value in router.items()})
    x = leaves["input"].reshape(-1, config.hidden_size)

    scores = _oracle_scores(x, router["weight"], router["weight_1"])
    probs, _, margin = _oracle_routing(
        scores,
        bias,
        router["routed_scaling_factor_param"],
        topk=config.moe_router_topk,
        normalize=config.ernie_norm_topk_prob,
    )
    expert_input = x
    if config.moe_latent_size:
        leaves["fc1_latent_proj.weight"] = moe.fc1_latent_proj.weight.detach().double().requires_grad_(True)
        leaves["fc2_latent_proj.weight"] = moe.fc2_latent_proj.weight.detach().double().requires_grad_(True)
        expert_input = x @ leaves["fc1_latent_proj.weight"].T
    routed = 0.0
    for index, expert in enumerate(moe.experts.local_experts):
        fc1, fc2 = _mlp_tensors(expert)
        leaves[f"experts.local_experts.{index}.linear_fc1.weight"] = fc1
        leaves[f"experts.local_experts.{index}.linear_fc2.weight"] = fc2
        routed = routed + (probs[:, index : index + 1] * activation(expert_input @ fc1.T)) @ fc2.T
    if config.moe_latent_size:
        if config.moe_use_norm_before_up_proj:
            leaves["fc2_norm.weight"] = moe.fc2_norm.weight.detach().double().requires_grad_(True)
            routed = _oracle_rmsnorm(routed, leaves["fc2_norm.weight"], config.layernorm_epsilon)
        routed = routed @ leaves["fc2_latent_proj.weight"].T
    fc1, fc2 = _mlp_tensors(moe.shared_experts)
    leaves["shared_experts.linear_fc1.weight"], leaves["shared_experts.linear_fc2.weight"] = fc1, fc2
    output = routed + _oracle_mlp(x, fc1, fc2, activation)
    return output.reshape(hidden.shape), leaves, margin


def _check_moe_against_oracle(moe: Ernie55MoELayer, config: Ernie55TransformerConfig, seed: int) -> None:
    hidden = torch.randn(SEQ, BATCH, config.hidden_size, generator=torch.Generator().manual_seed(seed))
    hidden.requires_grad_(True)
    expected, leaves, margin = _oracle_moe(moe, config, hidden)
    assert margin.min().item() > SAFE_MARGIN, "the seed planted an accidental near-tie"

    output, bias = moe(hidden)
    assert bias is None
    _assert_close(output, expected, TOL, "MoE output")

    cotangent = torch.randn(output.shape, generator=torch.Generator().manual_seed(seed + 1), dtype=torch.float64)
    output.backward(cotangent.float())
    expected.backward(cotangent)
    params = dict(moe.named_parameters())
    assert set(leaves) - {"input"} == set(params), "the oracle must cover every parameter"
    _assert_close(hidden.grad, leaves["input"].grad, TOL, "d input")
    for name, leaf in leaves.items():
        if name != "input":
            _assert_close(params[name].grad, leaf.grad, TOL, f"d {name}")


@pytest.mark.parametrize(
    "activation",
    [
        pytest.param({}, id="swiglu"),
        pytest.param(dict(activation_func_tanh_clamp_scale=1.0), id="situ-linear-beta-none"),
        pytest.param(
            dict(activation_func_tanh_clamp_scale=1.5, activation_func_tanh_clamp_scale_linear=0.5),
            id="situ-linear-beta",
        ),
    ],
)
def test_moe_layer_matches_oracle(pg_collection, activation):
    """Router, routed experts (probabilities before fc2), ungated shared expert and every gradient."""
    config = _config(**activation)
    moe = _moe(config, pg_collection, seed=31)
    assert isinstance(moe, Ernie55MoELayer) and isinstance(moe.router, Ernie55SplitFeatureRouter)
    assert moe.shared_experts.use_shared_expert_gate is False
    _check_moe_against_oracle(moe, config, seed=12)


@pytest.mark.parametrize("use_norm", [False, True])
def test_latent_moe_matches_oracle(pg_collection, latent_moe_without_te, use_norm):
    """Latent MoE: fp32 projections cast to the activation dtype, experts at latent width, optional norm."""
    config = _config(moe_latent_size=LATENT, moe_use_norm_before_up_proj=use_norm)
    moe = _moe(config, pg_collection, seed=37)
    for name, shape in (("fc1_latent_proj", (LATENT, HIDDEN)), ("fc2_latent_proj", (HIDDEN, LATENT))):
        projection = getattr(moe, name)
        assert isinstance(projection, Ernie55LatentProjection), name
        assert projection.weight.shape == shape and projection.weight.dtype == torch.float32
        assert getattr(projection.weight, "keep_in_fp32", False) and is_managed_by_layer_wise_optimizer(
            projection.weight
        )
    assert hasattr(moe, "fc2_norm") is use_norm
    assert moe.experts.local_experts[0].linear_fc1.weight.shape == (2 * MOE_FFN, LATENT)
    _check_moe_against_oracle(moe, config, seed=14)


def test_latent_projection_rounds_its_weight_to_the_input_dtype(pg_collection, latent_moe_without_te):
    """bf16 activations see bf16(W) (one rounding of the fp32 master); Float16Module keeps W in fp32."""
    config = _config(moe_latent_size=LATENT, moe_use_norm_before_up_proj=True, params_dtype=torch.bfloat16)
    moe = _moe(config, pg_collection, seed=41)
    projection = moe.fc1_latent_proj
    assert not torch.equal(projection.weight.bfloat16().float(), projection.weight)
    hidden = torch.randn(5, HIDDEN, generator=torch.Generator().manual_seed(15)).bfloat16()
    output, bias = projection(hidden)
    assert bias is None and output.dtype == torch.bfloat16
    assert torch.equal(output, F.linear(hidden, projection.weight.bfloat16()))

    wrapped = Float16Module(config, moe).module
    for name in ("fc1_latent_proj", "fc2_latent_proj"):
        assert getattr(wrapped, name).weight.dtype == torch.float32, name
    assert wrapped.fc2_norm.weight.dtype == torch.bfloat16
    assert wrapped.experts.local_experts[0].linear_fc1.weight.dtype == torch.bfloat16


@pytest.mark.parametrize("linear_beta", [None, 0.5])
def test_dense_mlp_situ_matches_oracle(pg_collection, linear_beta):
    """SiTU-GLU through Megatron-Core's clamp knobs on the ``-`` MLP, in both linear-beta modes."""
    config = _config(activation_func_tanh_clamp_scale=1.5, activation_func_tanh_clamp_scale_linear=linear_beta)
    layer = build_module(
        _localize(ernie55_hybrid_stack_spec(config).submodules.mlp_layer),
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    )
    _randomize_(layer, seed=43)
    # Large fc1 outputs so that clamping the up half changes the result.
    hidden = 3.0 * torch.randn(SEQ, BATCH, HIDDEN, generator=torch.Generator().manual_seed(16), dtype=torch.float64)
    fc1, fc2 = _mlp_tensors(layer.mlp)
    expected = _oracle_mlp(hidden, fc1, fc2, _oracle_activation(config))
    unclamped = _oracle_mlp(hidden, fc1, fc2, functools.partial(_situ_glu, beta=1.5, linear_beta=None))
    if linear_beta is not None:
        assert (expected - unclamped).abs().max() > 1e3 * TOL * expected.abs().max()
    output, _ = layer.mlp(hidden.float())
    _assert_close(output, expected, TOL, "dense SiTU MLP output")


# ---------------------------------------------------------------------------------------------
# Specs and parameter names
# ---------------------------------------------------------------------------------------------


def test_stack_spec_slots_and_mcore_spec_untouched():
    """``-`` gets an unfused norm and plain fc1, ``E`` the ERNIE MoE layer and router; MCore's spec is unchanged."""
    from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

    spec = ernie55_hybrid_stack_spec()
    # Both norms downcast the fp32 hyper-connection streams (test_ernie55_mhc_mtp.py).
    downcasting_te_norm = downcasting_norm(te_extensions.TENorm)

    def is_downcasting_te_norm(norm) -> bool:
        return norm.func is downcasting_te_norm.func and norm.keywords == downcasting_te_norm.keywords

    dense = spec.submodules.mlp_layer.submodules
    assert is_downcasting_te_norm(dense.pre_mlp_layernorm)
    assert dense.mlp.func == MLP.as_mlp_submodule
    assert dense.mlp.keywords["submodules"].linear_fc1 is te_extensions.TEColumnParallelLinear
    assert dense.mlp.keywords["submodules"].linear_fc2 is te_extensions.TERowParallelLinear
    moe = spec.submodules.moe_layer.submodules
    assert is_downcasting_te_norm(moe.pre_mlp_layernorm)
    assert moe.mlp.func is Ernie55MoELayer
    assert moe.mlp.keywords["submodules"].router is Ernie55SplitFeatureRouter
    # MTP layers are built from these same slots.
    assert getattr(spec.submodules, "mtp_stack_submodules", None) is None

    stock = hybrid_stack_spec.submodules
    assert (
        stock.mlp_layer.submodules.mlp.keywords["submodules"].linear_fc1
        is te_extensions.TELayerNormColumnParallelLinear
    )
    assert stock.moe_layer.submodules.mlp.func is MoELayer
    assert stock.moe_layer.submodules.mlp.keywords["submodules"].router is TopKRouter


def test_dense_layer_parameter_names_match_the_reference(pg_collection):
    """The unfused ``-`` layer has the reference dense layer's parameter names, shapes and dtypes."""
    config = _config(params_dtype=torch.bfloat16)
    layer = build_module(
        _localize(ernie55_hybrid_stack_spec(config).submodules.mlp_layer),
        config=config,
        layer_number=1,
        pg_collection=pg_collection,
    )
    # The reference names and dtypes are those of the bf16-wrapped (Float16Module) model.
    layer = Float16Module(config, layer).module
    manifest = {name: (tuple(param.shape), param.dtype) for name, param in layer.named_parameters()}
    assert manifest == _reference_dense_manifest(HIDDEN, DENSE_FFN)


def test_moe_layer_manifest_matches_the_reference(pg_collection):
    """Parameters and buffers of the ``E`` layer against the reference MoE layer (experts in grouped naming)."""
    config = _config(params_dtype=torch.bfloat16)
    layer = build_module(
        _localize(ernie55_hybrid_stack_spec(config).submodules.moe_layer),
        config=config,
        layer_number=2,
        pg_collection=pg_collection,
    )
    # The reference names and dtypes are those of the bf16-wrapped (Float16Module) model, buffers included.
    layer = Float16Module(config, layer).module
    tensors = dict(layer.named_parameters())
    tensors.update(layer.named_buffers())
    manifest = {}
    for name, tensor in tensors.items():
        # Sequential CPU experts name local expert i "local_experts.i.linear_fcX.weight"; grouped ones "weighti".
        grouped = re.sub(r"experts\.local_experts\.(\d+)\.(linear_fc[12])\.weight$", r"experts.\2.weight\1", name)
        manifest[grouped] = (tuple(tensor.shape), tensor.dtype)
    # Documented difference: exact int64 token counts instead of the reference's bf16 buffer.
    expected = _reference_moe_manifest(
        hidden=HIDDEN,
        experts=EXPERTS,
        local_experts=EXPERTS,
        ffn=MOE_FFN,
        shared_ffn=SHARED_FFN,
        token_count_dtype=torch.int64,
    )
    assert manifest == expected
    assert "mlp.router.local_tokens_per_expert" not in layer.state_dict()


def test_moe_layer_rejects_a_stock_router(pg_collection):
    config = _config()
    builder = _localize(ernie55_hybrid_stack_spec(config).submodules.moe_layer.submodules.mlp)
    stock = functools.partial(
        builder.func, submodules=dataclasses.replace(builder.keywords["submodules"], router=TopKRouter)
    )
    with pytest.raises(TypeError, match="Ernie55SplitFeatureRouter"):
        stock(config=config, pg_collection=pg_collection)


def test_spec_factory_rejects_an_unexpected_moe_spec(monkeypatch):
    from megatron.core.models.hybrid import hybrid_layer_specs

    patched = _copy_spec_tree(hybrid_layer_specs.hybrid_stack_spec, {})
    patched.submodules.moe_layer.submodules.mlp = ModuleSpec(module=MLP)
    monkeypatch.setattr(hybrid_layer_specs, "hybrid_stack_spec", patched)
    with pytest.raises(TypeError, match="MoELayer"):
        ernie55_hybrid_stack_spec()


def test_spec_factory_rejects_separate_mtp_slots(monkeypatch):
    """Separate MTP layer slots in Megatron-Core's spec would bypass the ERNIE modules in the MTP layers."""
    from megatron.core.models.hybrid import hybrid_layer_specs

    patched = _copy_spec_tree(hybrid_layer_specs.hybrid_stack_spec, {})
    if not hasattr(patched.submodules, "mtp_stack_submodules"):
        pytest.skip("this Megatron-Core has no mtp_stack_submodules")
    patched.submodules.mtp_stack_submodules = _copy_spec_tree(patched.submodules, {})
    monkeypatch.setattr(hybrid_layer_specs, "hybrid_stack_spec", patched)
    with pytest.raises(TypeError, match="mtp_stack_submodules"):
        ernie55_hybrid_stack_spec()
