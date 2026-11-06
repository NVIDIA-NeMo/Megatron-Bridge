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

"""CPU tests of the ERNIE 5.5 multimax output head and its gradient-norm group.

The oracle is written from the ERNIE 5.5 specification, in fp64. For a logit ``x`` and the four
ranges ``r`` and scales ``t``:

* ``f(x) = x + t0 relu(r0 - x) + t1 relu(x - r1) + t2 relu(r2 - x)^2 + t3 relu(x - r3)^2``;
* ``df/dx = 1 - t0 [r0 > x] + t1 [x > r1] - 2 t2 relu(r2 - x) + 2 t3 relu(x - r3)``;
* ``df/dr = (t0 [r0 > x], -t1 [x > r1], 2 t2 relu(r2 - x), -2 t3 relu(x - r3))``;
* ``df/dt = (relu(r0 - x), relu(x - r1), relu(r2 - x)^2, relu(x - r3)^2)``;
* the parameter gradients are sums over every logit of the upstream gradient times these.

Contract: fp32 arithmetic, the logits dtype out (bf16 in training), the logit gradient in the logits
dtype, the ``r`` and ``t`` gradients accumulated in fp32 over 512-row chunks and rounded once to the
parameter dtype. The parameters form their own gradient-norm group: Megatron-Core leaves them out
of the main (logged) norm and clips them by their own norm with the same ``clip_grad``.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from megatron.core import parallel_state
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import optimizer as mcore_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.modeling_ernie55 import (
    MULTIMAX_GRAD_NORM_GROUP,
    MULTIMAX_PARAMETER_NAMES,
    ernie55_multimax,
    install_ernie55_multimax,
    validate_ernie55_multimax_config,
)
from megatron.bridge.models.ernie.modeling_ernie55.output_layer import MULTIMAX_CHUNK_ROWS


pytestmark = pytest.mark.unit

# 2 * 512 + 37 rows: two full chunks and a partial one.
SEQ, BATCH, VOCAB = 531, 2, 48
HIDDEN = 32
RANGES = (-0.5, 0.75, -1.25, 1.5)
SCALES = (0.3, -0.2, 0.05, 0.1)

try:  # Megatron-Core dev's output layer; main uses ColumnParallelLinear.
    from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
except ImportError:  # pragma: no cover - depends on the Megatron-Core version
    LinearCrossEntropyModule = None
_OUTPUT_LAYER_CLASSES = [ColumnParallelLinear] + ([LinearCrossEntropyModule] if LinearCrossEntropyModule else [])


# ---------------------------------------------------------------------------------------------
# fp64 oracle
# ---------------------------------------------------------------------------------------------


def _oracle(x: torch.Tensor, ranges, scales):
    """``(f(x), df/dx, df/dr [4, ...], df/dt [4, ...])`` in fp64."""
    x = x.double()
    r = [float(value) for value in ranges]
    t = [float(value) for value in scales]
    below0, above1 = torch.relu(r[0] - x), torch.relu(x - r[1])
    below2, above3 = torch.relu(r[2] - x), torch.relu(x - r[3])
    value = x + t[0] * below0 + t[1] * above1 + t[2] * below2**2 + t[3] * above3**2
    d_x = 1.0 - t[0] * (r[0] > x) + t[1] * (x > r[1]) - 2 * t[2] * below2 + 2 * t[3] * above3
    d_r = torch.stack([t[0] * (r[0] > x).double(), -t[1] * (x > r[1]).double(), 2 * t[2] * below2, -2 * t[3] * above3])
    d_t = torch.stack([below0, above1, below2**2, above3**2])
    return value, d_x, d_r, d_t


def _bf16_ulps(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """``|actual - round_bf16(expected)|`` in units of the bf16 spacing at ``round_bf16(expected)``."""
    rounded = expected.to(torch.bfloat16).double()
    _, exponent = torch.frexp(rounded)
    spacing = torch.ldexp(torch.ones_like(rounded), exponent - 8).clamp_min(2.0**-133)
    return (actual.double() - rounded).abs() / spacing


def _norm_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.double()
    return ((actual.double() - expected).abs().max() / expected.abs().max().clamp_min(1e-30)).item()


def _inputs(*, dtype: torch.dtype = torch.bfloat16, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = (2.0 * torch.randn(SEQ, BATCH, VOCAB, generator=generator)).to(dtype)
    grad = torch.randn(SEQ, BATCH, VOCAB, generator=generator).to(dtype)
    return logits, grad


# ---------------------------------------------------------------------------------------------
# The correction against the oracle
# ---------------------------------------------------------------------------------------------


class TestMultimax:
    def test_rows_cross_chunk_boundaries(self):
        # The reference implementation's chunk: it fixes the fp32 summation order of the r and t gradients.
        assert MULTIMAX_CHUNK_ROWS == 512
        assert SEQ * BATCH > 2 * MULTIMAX_CHUNK_ROWS and (SEQ * BATCH) % MULTIMAX_CHUNK_ROWS

    @pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32], ids=["bf16-params", "fp32-params"])
    def test_forward_and_backward_match_the_oracle(self, param_dtype):
        logits, grad = _inputs()
        logits.requires_grad_(True)
        ranges = torch.tensor(RANGES, dtype=param_dtype, requires_grad=True)
        scales = torch.tensor(SCALES, dtype=param_dtype, requires_grad=True)
        output = ernie55_multimax(logits, ranges, scales)
        output.backward(grad)

        value, d_x, d_r, d_t = _oracle(logits.detach(), ranges.detach(), scales.detach())
        assert output.dtype == torch.bfloat16 and output.shape == logits.shape
        assert not output._is_view()
        assert (output != logits).any()
        # fp32 arithmetic rounded once.
        assert _bf16_ulps(output, value).max() <= 1.0
        assert (_bf16_ulps(output, value) == 0).double().mean() >= 0.99
        # The same fp32 expression without chunks: chunking must not change any element.
        x32, r32, t32 = logits.detach().float(), ranges.detach().float(), scales.detach().float()
        unchunked = (
            x32
            + t32[0] * torch.relu(r32[0] - x32)
            + t32[1] * torch.relu(x32 - r32[1])
            + t32[2] * torch.relu(r32[2] - x32) ** 2
            + t32[3] * torch.relu(x32 - r32[3]) ** 2
        ).to(torch.bfloat16)
        assert torch.equal(output, unchunked)

        g64 = grad.double()
        assert logits.grad.dtype == torch.bfloat16
        assert _bf16_ulps(logits.grad, g64 * d_x).max() <= 1.0
        expected_r = (g64 * d_r).flatten(1).sum(dim=1)
        expected_t = (g64 * d_t).flatten(1).sum(dim=1)
        assert ranges.grad.dtype == param_dtype and scales.grad.dtype == param_dtype
        if param_dtype == torch.bfloat16:
            # Summed in fp32 over the chunks, then rounded once.
            assert _bf16_ulps(ranges.grad, expected_r).max() <= 1.0
            assert _bf16_ulps(scales.grad, expected_t).max() <= 1.0
        else:
            assert _norm_err(ranges.grad, expected_r) <= 1e-5
            assert _norm_err(scales.grad, expected_t) <= 1e-5

    @pytest.mark.parametrize("param_dtype", [torch.bfloat16, torch.float32], ids=["bf16-params", "fp32-params"])
    def test_parameter_gradients_sum_512_row_chunks_in_fp32_and_round_once(self, param_dtype):
        """The r and t gradients: an fp32 sum within each 512-row chunk, the chunk sums added in fp32
        in row order, one rounding to the parameter dtype at the end.

        At r = t = 0, d/dt1 = sum g relu(x) and d/dt3 = sum g relu(x)^2, while d/dr, d/dt0 and d/dt2
        vanish for x >= 0. Four logits are 1, one in each 256-row quarter of 1024 rows, with upstream
        gradients g = a = 2^24, b = 1, c = 1, d = -2^24; every other logit and gradient is 0, so both
        sums are a + b + c + d, accumulated as follows:

        * 512-row chunks: fl32(a + b) = 2^24 (2^24 + 1 rounds to even), fl32(c + d) = 1 - 2^24
          (exact), and the fp32 sum of the two is 1;
        * 256-row chunks: 2^24, then + 1 and + 1 both round back to 2^24, then - 2^24: 0;
        * rounding each chunk sum to bf16 first: 1 - 2^24 rounds to -2^24, so the sum is 0.
        """
        rows, vocab = 4 * 256, 4
        logits = torch.zeros(rows, 1, vocab, dtype=torch.bfloat16)
        grad = torch.zeros(rows, 1, vocab, dtype=torch.bfloat16)
        for quarter, upstream in enumerate((2.0**24, 1.0, 1.0, -(2.0**24))):
            logits[256 * quarter, 0, 0] = 1.0
            grad[256 * quarter, 0, 0] = upstream
        logits.requires_grad_(True)
        ranges = torch.zeros(4, dtype=param_dtype, requires_grad=True)
        scales = torch.zeros(4, dtype=param_dtype, requires_grad=True)
        ernie55_multimax(logits, ranges, scales).backward(grad)

        assert torch.equal(logits.grad, grad)
        assert torch.equal(ranges.grad, torch.zeros(4, dtype=param_dtype))
        assert torch.equal(scales.grad, torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=param_dtype)), scales.grad

    def test_zero_parameters_are_the_identity(self):
        """At the zero init the forward is exact, d/dr is zero and d/dt are the four correction sums."""
        logits, grad = _inputs(seed=1)
        logits.requires_grad_(True)
        ranges = torch.zeros(4, dtype=torch.float32, requires_grad=True)
        scales = torch.zeros(4, dtype=torch.float32, requires_grad=True)
        output = ernie55_multimax(logits, ranges, scales)
        assert torch.equal(output, logits.detach())
        output.backward(grad)
        assert torch.equal(logits.grad, grad)
        assert torch.equal(ranges.grad, torch.zeros(4))
        x, g = logits.detach().double(), grad.double()
        expected = torch.stack(
            [
                (g * torch.relu(-x)).sum(),
                (g * torch.relu(x)).sum(),
                (g * torch.relu(-x) ** 2).sum(),
                (g * torch.relu(x) ** 2).sum(),
            ]
        )
        assert _norm_err(scales.grad, expected) <= 1e-5

    def test_rejects_wrong_parameter_shapes(self):
        logits, _ = _inputs()
        with pytest.raises(ValueError, match=r"\[4\]"):
            ernie55_multimax(logits, torch.zeros(3), torch.zeros(4))


# ---------------------------------------------------------------------------------------------
# The output layer
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tp_group(tmp_path_factory):
    """Single-rank gloo process groups (the output layer needs a tensor-parallel group)."""
    created_group = not torch.distributed.is_initialized()
    if created_group:
        store = tmp_path_factory.mktemp("ernie55_multimax_gloo") / "store"
        torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    created_model_parallel = not parallel_state.model_parallel_is_initialized()
    if created_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        yield ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"]).tp
    finally:
        if created_model_parallel:
            parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()


def _transformer_config(**overrides) -> TransformerConfig:
    values = dict(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    values.update(overrides)
    return TransformerConfig(**values)


def _output_layer(cls, config: TransformerConfig, tp_group) -> torch.nn.Module:
    torch.manual_seed(3)
    return cls(
        HIDDEN,
        VOCAB,
        config=config,
        init_method=torch.nn.init.normal_,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
        tp_group=tp_group,
    )


@pytest.mark.parametrize("output_layer_class", _OUTPUT_LAYER_CLASSES, ids=lambda cls: cls.__name__)
def test_install_on_the_output_layer(tp_group, output_layer_class, monkeypatch):
    # Megatron-Core's own group list: the install must register the multimax group itself.
    monkeypatch.setattr(mcore_optimizer, "SEPARATE_GRAD_NORM_GROUPS", ("mtp",))
    config = _transformer_config()
    layer = _output_layer(output_layer_class, config, tp_group)
    weight = layer.weight
    install_ernie55_multimax(layer, config)

    assert isinstance(layer, output_layer_class) and type(layer).__name__ == output_layer_class.__name__
    assert mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS == ("mtp", MULTIMAX_GRAD_NORM_GROUP)
    names = [name for name, _ in layer.named_parameters()]
    assert names == ["weight", *MULTIMAX_PARAMETER_NAMES] and layer.weight is weight
    for name in MULTIMAX_PARAMETER_NAMES:
        param = layer.get_parameter(name)
        assert param.shape == (4,) and param.dtype == torch.bfloat16 and not torch.any(param)
        assert param.grad_norm_group == MULTIMAX_GRAD_NORM_GROUP
        assert not getattr(param, "keep_in_fp32", False)
    assert set(layer.state_dict()) >= {"weight", *MULTIMAX_PARAMETER_NAMES}

    with torch.no_grad():
        layer.multimax_ranges.copy_(torch.tensor(RANGES))
        layer.multimax_ts.copy_(torch.tensor(SCALES))
    hidden = torch.randn(7, BATCH, HIDDEN, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    logits, bias = layer(hidden)
    assert bias is None
    expected = ernie55_multimax(torch.matmul(hidden, layer.weight.t()), layer.multimax_ranges, layer.multimax_ts)
    assert torch.equal(logits, expected)
    logits.float().square().sum().backward()
    assert all(torch.any(layer.get_parameter(name).grad != 0) for name in MULTIMAX_PARAMETER_NAMES)

    # Idempotent: the same parameters, one correction.
    install_ernie55_multimax(layer, config)
    assert [name for name, _ in layer.named_parameters()] == names
    assert torch.equal(layer(hidden)[0], expected)
    if output_layer_class is LinearCrossEntropyModule:
        with pytest.raises(RuntimeError, match="fused linear cross-entropy"):
            layer(hidden, output_cross_entropy_loss=True, labels=torch.zeros(BATCH, 7, dtype=torch.long))


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="linear"), "cross_entropy_fusion_impl"),
        (dict(tensor_model_parallel_size=2), "tensor_model_parallel_size"),
        (dict(use_mup=True), "muP"),
    ],
    ids=["linear-ce", "tp2", "mup"],
)
def test_validation(overrides, message):
    config = _transformer_config()
    validate_ernie55_multimax_config(config)
    for name, value in overrides.items():
        if not hasattr(config, name):
            pytest.skip(f"this Megatron-Core has no {name}")
        setattr(config, name, value)
    with pytest.raises(ValueError, match=message):
        validate_ernie55_multimax_config(config)


# ---------------------------------------------------------------------------------------------
# The gradient-norm group in Megatron-Core's optimizer
# ---------------------------------------------------------------------------------------------


class _Optimizer:
    """Megatron-Core's gradient-norm grouping and clipping on a plain parameter list (CPU).

    The grouping, norm and clipping methods are Megatron-Core's own; ``has_grad_norm_group`` skips
    the cross-rank flag (a CUDA all-reduce) and answers from the local parameters.
    """

    _filter_grads_for_norm = mcore_optimizer.MegatronOptimizer._filter_grads_for_norm
    get_grads_for_grad_norm = mcore_optimizer.MegatronOptimizer.get_grads_for_grad_norm
    get_grad_norm = mcore_optimizer.MegatronOptimizer.get_grad_norm
    _compute_grad_norms_by_group = mcore_optimizer.MegatronOptimizer._compute_grad_norms_by_group
    _uses_decoupled_grad = mcore_optimizer.MegatronOptimizer._uses_decoupled_grad
    clip_grad_norm = mcore_optimizer.MegatronOptimizer.clip_grad_norm

    def __init__(self, params) -> None:
        self.params = list(params)
        self.config = OptimizerConfig(optimizer="adam", lr=0.01, clip_grad=1.0)

    def get_parameters(self):
        return self.params

    def get_grad_stats_parallel_group(self):
        return parallel_state.get_model_parallel_group()

    def has_grad_norm_group(self, grad_norm_group: str) -> bool:
        return any(getattr(param, "grad_norm_group", None) == grad_norm_group for param in self.params)


def _cpu_grad_norm(grads, norm_type=2, grad_stats_parallel_group=None) -> float:
    """CPU stand-in for Megatron-Core's ``get_grad_norm_fp32`` (fp32 L2 norm, single rank)."""
    return float(torch.sqrt(sum((grad.float() ** 2).sum() for grad in grads))) if grads else 0.0


def _cpu_clip(params, max_norm, total_norm, use_decoupled_grad=False) -> None:
    """CPU stand-in for Megatron-Core's ``clip_grad_by_total_norm_fp32``: scale by ``max / (norm + 1e-6)`` below 1."""
    coefficient = max_norm / (total_norm + 1.0e-6)
    if coefficient < 1.0:
        for param in params:
            param.grad.mul_(coefficient)


def test_multimax_has_its_own_gradient_norm_and_clip(tp_group, monkeypatch):
    monkeypatch.setattr(mcore_optimizer, "get_grad_norm_fp32", _cpu_grad_norm)
    monkeypatch.setattr(mcore_optimizer, "clip_grad_by_total_norm_fp32", _cpu_clip)
    config = _transformer_config(params_dtype=torch.float32)
    layer = _output_layer(ColumnParallelLinear, config, tp_group)
    install_ernie55_multimax(layer, config)
    other = torch.nn.Parameter(torch.zeros(5))
    params = [layer.weight, other, layer.multimax_ranges, layer.multimax_ts]
    generator = torch.Generator().manual_seed(7)
    for param, scale in zip(params, (0.1, 0.1, 3.0, 4.0)):
        param.grad = scale * torch.randn(param.shape, generator=generator)
    main_grads = [layer.weight.grad.clone(), other.grad.clone()]
    multimax_grads = [layer.multimax_ranges.grad.clone(), layer.multimax_ts.grad.clone()]

    optimizer = _Optimizer(params)
    assert [id(g) for g in optimizer.get_grads_for_grad_norm()] == [id(layer.weight.grad), id(other.grad)]
    assert [id(g) for g in optimizer.get_grads_for_grad_norm(MULTIMAX_GRAD_NORM_GROUP)] == [
        id(layer.multimax_ranges.grad),
        id(layer.multimax_ts.grad),
    ]

    def norm64(grads) -> float:
        return float(torch.sqrt(sum((grad.double() ** 2).sum() for grad in grads)))

    main_norm, multimax_norm = norm64(main_grads), norm64(multimax_grads)
    assert main_norm > 1.0 and multimax_norm > 1.0
    logged = optimizer.clip_grad_norm(1.0)
    # The logged norm is the main group's alone.
    assert logged == pytest.approx(main_norm, rel=1e-6)
    assert optimizer.grad_norms_by_group[MULTIMAX_GRAD_NORM_GROUP] == pytest.approx(multimax_norm, rel=1e-6)
    for param, before in zip(params[:2], main_grads):
        assert torch.allclose(param.grad.double(), before.double() / (main_norm + 1e-6), rtol=1e-6)
    for param, before in zip(params[2:], multimax_grads):
        assert torch.allclose(param.grad.double(), before.double() / (multimax_norm + 1e-6), rtol=1e-6)
    assert norm64([p.grad for p in params[2:]]) == pytest.approx(1.0, rel=1e-5)


def test_the_group_is_registered_once():
    from megatron.bridge.models.ernie.modeling_ernie55.output_layer import register_multimax_grad_norm_group

    register_multimax_grad_norm_group()
    register_multimax_grad_norm_group()
    assert mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS.count(MULTIMAX_GRAD_NORM_GROUP) == 1
    assert "mtp" in mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS
    assert dataclasses.is_dataclass(OptimizerConfig)
