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

"""CPU tests of the ERNIE 5.5 hyper-connection precision contract, fp32 decoder streams and SD-MTP.

The oracles in this file are fp64 re-derivations written from the ERNIE 5.5 specification, not from
the Bridge implementation. For the streams ``X = [x_1 .. x_n]`` of one token (``K = n*C`` values):

* ``r = 1 / (||X||_2 / sqrt(K) + 1e-6)`` and ``h = r * (X W^T) * alpha + bias``, where ``alpha``
  repeats ``alpha_pre`` n times, ``alpha_post`` n times and ``alpha_res`` n^2 times;
* ``h_pre = sigmoid(h[:n]) + 1e-6``, ``h_post = 2 sigmoid(h[n:2n])`` and
  ``H_res = Sinkhorn(h[2n:].view(n, n))``: softmax over the last axis plus 1e-6, one column
  normalization, then ``iterations - 1`` rounds of a row and a column normalization, every
  normalizer ``sum + 1e-6``;
* the layer input ``a = sum_j h_pre_j x_j``;
* the update ``x_i <- sum_j H_res[j, i] x_j + h_post_i * y`` for the layer output ``y``.

All of it is computed in fp32 and rounded once to the stream dtype. The decoder streams are fp32
(the embedding is cast before the streams are expanded); the MTP streams are bf16; every layer
computes in bf16 behind a pre-norm that rounds the fp32 layer input once.

SD-MTP: each MTP depth's inner ``DE`` pair shares the Parameters (not the buffers) of the decoder's
last ``DE`` pair, so their gradients are the sums of both uses, and every shared tensor is emitted
once to the optimizer and to checkpoints.

Everything runs on CPU with Transformer-Engine-free stand-ins for the TE leaves. The norm stand-in
does not cast its input, so a missing downcast shows up as an fp32 layer input.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import functools
import math
import types
import zlib

import pytest
import torch
import torch.nn.functional as F
from megatron.core import activations, parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.experimental_attention_variant import deepseek_v4_hybrid_attention as dsv4
from megatron.core.transformer.experimental_attention_variant import dsa
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    MULTIMAX_GRAD_NORM_GROUP,
    Ernie55HybridModel,
    Ernie55HybridStack,
    Ernie55HyperConnectionModule,
    Ernie55RMSNorm,
    check_ernie55_mtp_aliases,
    downcasting_norm,
    ernie55_hybrid_stack_spec,
    ernie55_multimax,
    ernie55_rmsnorm,
    is_downcasting_norm,
    share_mtp_body_with_last_layer,
    use_ernie55_hyper_connections,
    validate_ernie55_mhc_config,
    validate_ernie55_mtp_sharing,
)
from megatron.bridge.models.ernie.modeling_ernie55 import fused_ops as ernie55_fused_ops
from megatron.bridge.models.ernie.modeling_ernie55.hyper_connection import (
    ernie55_h_aggregate,
    ernie55_h_post_bda,
    ernie55_sinkhorn,
)
from tests.unit_tests.models.ernie.ernie55_test_utils import (
    CallRecorder,
    cpu_current_device,
    isolate_fused_dispatch,
)


pytestmark = pytest.mark.unit

HIDDEN = 64
STREAMS = 4
SINKHORN_ITERATIONS = 20
EPS = 1e-6
VOCAB = 97
PATTERN = "H-DE/DE"
# H layers compress 128-token blocks: a longer sequence gives the compressor work.
SEQ = 300

# fp32 computation against the fp64 oracle, as a normalized max error (max |diff| / max |expected|).
TOL = 2e-5

# Megatron-Core's mhc_keep_mappings_in_fp32 switch; older Megatron-Core builds lack it.
_HAS_KEEP_MAPPINGS_IN_FP32 = "mhc_keep_mappings_in_fp32" in {f.name for f in dataclasses.fields(TransformerConfig)}
_HAS_HOOK = callable(getattr(dsv4.DSv4HybridAttention, "_post_o_group_proj", None))
needs_hook = pytest.mark.skipif(
    not _HAS_HOOK,
    reason="needs Megatron-Core's DSv4HybridAttention._post_o_group_proj hook (dev f78809f6d+)",
)


# ---------------------------------------------------------------------------------------------
# fp64 oracle
# ---------------------------------------------------------------------------------------------


def _oracle_sinkhorn(logits: torch.Tensor, iterations: int) -> torch.Tensor:
    """Sinkhorn-Knopp in fp64, one explicit normalization at a time."""
    matrix = torch.exp(logits - logits.amax(dim=-1, keepdim=True))
    matrix = matrix / matrix.sum(dim=-1, keepdim=True) + EPS
    column_sums = matrix.sum(dim=-2, keepdim=True)
    matrix = matrix / (column_sums + EPS)
    for _ in range(iterations - 1):
        row_sums = matrix.sum(dim=-1, keepdim=True)
        matrix = matrix / (row_sums + EPS)
        column_sums = matrix.sum(dim=-2, keepdim=True)
        matrix = matrix / (column_sums + EPS)
    return matrix


def _oracle_mappings(x, weight, alpha_pre, alpha_post, alpha_res, bias, n: int):
    """``(h_pre, h_post, H_res)`` in fp64 for ``x`` = ``[..., n*C]``."""
    x, weight, bias = x.double(), weight.double(), bias.double()
    width = x.shape[-1]
    r = 1.0 / (torch.sqrt((x * x).sum(dim=-1, keepdim=True)) / math.sqrt(width) + EPS)
    scales = torch.cat([alpha_pre.double().expand(n), alpha_post.double().expand(n), alpha_res.double().expand(n * n)])
    h = r * (x @ weight.T) * scales + bias
    h_pre = torch.sigmoid(h[..., :n]) + EPS
    h_post = 2.0 * torch.sigmoid(h[..., n : 2 * n])
    h_res = _oracle_sinkhorn(h[..., 2 * n :].reshape(*h.shape[:-1], n, n), SINKHORN_ITERATIONS)
    return h_pre, h_post, h_res


def _oracle_aggregate(x, h_pre, n: int):
    streams = x.double().reshape(*x.shape[:-1], n, -1)
    return (streams * h_pre.double().unsqueeze(-1)).sum(dim=-2)


def _oracle_update(x, h_res, h_post, y, n: int):
    """``x_i' = sum_j H_res[j, i] x_j + h_post_i y``, as ``[..., n*C]`` in fp64."""
    streams = x.double().reshape(*x.shape[:-1], n, -1)
    mixed = torch.einsum("...ji,...jc->...ic", h_res.double(), streams)
    return (mixed + h_post.double().unsqueeze(-1) * y.double().unsqueeze(-2)).reshape(x.shape)


def _norm_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected = expected.double()
    return ((actual.double() - expected).abs().max() / expected.abs().max().clamp_min(1e-30)).item()


def _bf16_ulps(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """``|actual - round_bf16(expected)|`` in units of the bf16 spacing at ``round_bf16(expected)``."""
    rounded = expected.to(torch.bfloat16).double()
    _, exponent = torch.frexp(rounded)
    spacing = torch.ldexp(torch.ones_like(rounded), exponent - 8).clamp_min(2.0**-133)
    return (actual.double() - rounded).abs() / spacing


# ---------------------------------------------------------------------------------------------
# Runtime fixtures and Transformer-Engine-free stand-ins
# ---------------------------------------------------------------------------------------------


class _LocalLinear(torch.nn.Module):
    """Stand-in for the TE linears at TP = 1: ``y = x W^T`` on any leading shape, returns ``(y, None)``."""

    def __init__(self, input_size: int, output_size: int, *, config, init_method, bias: bool = False, **kwargs):
        super().__init__()
        assert not bias
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        if config.perform_initialization:
            init_method(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None

    def backward_dw(self) -> None:
        pass


class _RMSNorm(torch.nn.Module):
    """``TENorm`` stand-in with fp32 statistics that returns its input dtype (it does not cast its input)."""

    def __init__(self, config, hidden_size: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_wide = x.float()
        normed = x_wide * torch.rsqrt(x_wide.square().mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class _StackNorm:
    """``TENorm`` stand-in: :class:`_RMSNorm`, or a torch LayerNorm for the D indexer's key norm."""

    def __new__(cls, config, hidden_size: int, eps: float = 1e-5, **kwargs):
        if config.normalization == "RMSNorm":
            return _RMSNorm(config, hidden_size, eps=eps, **kwargs)
        return WrappedTorchNorm(config=config, hidden_size=hidden_size, eps=eps)


_LOCAL_EXPERTS = functools.partial(
    SequentialMLP, submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear)
)


def _cpu_spec():
    """``ernie55_hybrid_stack_spec()`` with the TE leaves (and grouped experts) replaced by CPU stand-ins."""
    from megatron.core.extensions import transformer_engine as te

    leaves = {
        id(te.TELinear): _LocalLinear,
        id(te.TEColumnParallelLinear): _LocalLinear,
        id(te.TERowParallelLinear): _LocalLinear,
        id(te.TENorm): _StackNorm,
    }
    memo: dict[int, object] = {}

    def walk(item: object) -> object:
        if id(item) in leaves:
            return leaves[id(item)]
        if id(item) in memo:
            return memo[id(item)]
        if isinstance(item, functools.partial):
            if item.func in (SequentialMLP, TEGroupedMLP):
                return _LOCAL_EXPERTS
            result = functools.partial(walk(item.func), *walk(item.args), **walk(item.keywords))
        elif dataclasses.is_dataclass(item) and not isinstance(item, type):
            changes = {f.name: walk(getattr(item, f.name)) for f in dataclasses.fields(item) if f.init}
            result = dataclasses.replace(item, **changes)
        elif isinstance(item, dict):
            result = {key: walk(value) for key, value in item.items()}
        elif isinstance(item, (list, tuple)):
            result = type(item)(walk(value) for value in item)
        else:
            return item
        memo[id(item)] = result
        return result

    return walk(ernie55_hybrid_stack_spec())


def _hadamard(x: torch.Tensor) -> torch.Tensor:
    """``fast_hadamard_transform`` stand-in for the D indexer's rotation."""
    size = x.size(-1)
    matrix = torch.ones(1, 1, dtype=torch.float64)
    while matrix.size(0) < size:
        matrix = torch.cat([torch.cat([matrix, matrix], 1), torch.cat([matrix, -matrix], 1)], 0)
    return (x.double() @ matrix * size**-0.5).to(x.dtype)


@pytest.fixture(scope="module")
def pg_collection(tmp_path_factory):
    """Single-rank gloo process groups plus the CPU stand-ins the ERNIE layers and Megatron-Core need."""
    from megatron.core.models.hybrid import hybrid_block
    from megatron.core.transformer import multi_latent_attention
    from megatron.core.transformer import multi_token_prediction as mtp_module
    from megatron.core.transformer.experimental_attention_variant import absorbed_mla

    created_group = not torch.distributed.is_initialized()
    if created_group:
        store = tmp_path_factory.mktemp("ernie55_mhc_gloo") / "store"
        torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    created_model_parallel = not parallel_state.model_parallel_is_initialized()
    if created_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            # Router buffers, RoPE tables and the loss trackers are allocated on torch.cuda.current_device();
            # these tests run on CPU even when a GPU is present (torch's own callers still see the GPU).
            patch.setattr(torch.cuda, "current_device", cpu_current_device)
            patch.setattr(dsv4, "TELinear", _LocalLinear)
            patch.setattr(dsv4, "_q_rms_norm", lambda q, eps: q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps))
            patch.setattr(dsa, "rotate_activation", _hadamard)
            for module in (absorbed_mla, multi_latent_attention):
                patch.setattr(module, "TELinear", _LocalLinear)
            # HybridStack builds its final norm as TENorm directly, not from the spec.
            patch.setattr(hybrid_block, "TENorm", _StackNorm)
            # Without TE the inference linears are not classes; MTP probes for them.
            patch.setattr(mtp_module, "is_inference_column_parallel_linear", lambda module: False)
            yield ProcessGroupCollection.use_mpu_process_groups()
    finally:
        if created_model_parallel:
            parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def _cpu_nvtx(monkeypatch):
    """Megatron-Core's hyper-connection forward opens NVTX ranges, which need a CUDA build."""
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args, **kwargs: contextlib.nullcontext())


@pytest.fixture(autouse=True)
def _fresh_trackers():
    """Empty Megatron-Core's MoE, MTP and DSA loss trackers around every test."""
    from megatron.core.transformer.moe import moe_logging
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    saved = moe_logging.get_moe_metrics_tracker()
    moe_logging.set_moe_metrics_tracker(moe_logging.MoEMetricsTracker())
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    MTPLossLoggingHelper.tracker.clear()
    yield
    moe_logging.set_moe_metrics_tracker(saved)
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    MTPLossLoggingHelper.tracker.clear()


def _config(pattern: str = PATTERN, **overrides) -> Ernie55TransformerConfig:
    """Finalized ERNIE 5.5 config with H, D, dense and MoE layers, hyper-connections and MTP."""
    main_pattern, *mtp_patterns = pattern.split("/")
    symbols = main_pattern + "".join(mtp_patterns)
    values = dict(
        num_layers=len(main_pattern),
        mtp_num_layers=len(mtp_patterns) or None,
        hidden_size=HIDDEN,
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        # H layers (Megatron-Core's DSv4 attention at compress ratio 128).
        experimental_attention_variant="dsv4_hybrid",
        num_attention_heads=4,
        q_lora_rank=32,
        v_head_dim=16,
        qk_pos_emb_head_dim=8,
        output_projection_groups=2,
        output_projection_lora_rank=8,
        csa_compress_ratios=[128 if symbol == "H" else 0 for symbol in symbols],
        csa_window_size=5,
        csa_compress_rotary_base=1000.0,
        rope_type="yarn",
        rotary_scaling_factor=40.0,
        ernie_vha_rank=2,
        # D layers.
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=16,
        dsa_indexer_topk=6,
        dsa_indexer_k_norm_fp32=True,
        dsa_indexer_loss_coeff=0.01,
        dsa_indexer_use_sparse_loss=True,
        ernie_d_num_attention_heads=4,
        ernie_d_q_lora_rank=32,
        ernie_d_kv_lora_rank=16,
        ernie_d_qk_head_dim=8,
        ernie_d_qk_pos_emb_head_dim=8,
        ernie_d_v_head_dim=8,
        ernie_d_rotary_base=10000.0,
        # Dense and MoE layers.
        ffn_hidden_size=24,
        gated_linear_unit=True,
        activation_func=F.silu,
        num_moe_experts=8,
        moe_router_topk=4,
        moe_ffn_hidden_size=8,
        moe_shared_expert_intermediate_size=12,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=1e-4,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_dtype="fp32",
        moe_token_dispatcher_type="allgather",
        # Hyper-connections and SD-MTP.
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_sinkhorn_iterations=SINKHORN_ITERATIONS,
        ernie_mtp_shared_last_layer=bool(mtp_patterns),
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _randomize_(model: torch.nn.Module, seed: int) -> None:
    """Draw every parameter at O(1) scale (sinks, VHA, multimax and mHC factors included).

    Each parameter has its own generator keyed by its name, so values do not depend on the order in
    which modules register them; shared parameters are drawn once under their decoder name.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            generator = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float64)
            if name.endswith(("attn_sink", "compressor.ape", "hc_head_base", "hc_head_scale")):
                values = noise
            elif name.endswith(("alpha_pre", "alpha_post", "alpha_res")):
                values = 0.5 + 0.2 * noise
            elif name.endswith("hyper_connection.bias"):
                values = 0.5 * noise
            elif "vha_postmix" in name:
                values = 0.5 * noise
            elif "multimax" in name:
                values = 0.2 * noise
            elif "norm" in name:
                values = 1.0 + 0.2 * noise
            elif name.endswith("expert_bias"):
                values = 0.1 * noise
            else:
                values = noise / math.sqrt(param.shape[-1])
            param.copy_(values.to(param.dtype))


def _build_model(pg_collection, *, pattern: str = PATTERN, seed: int = 0, **overrides) -> Ernie55HybridModel:
    config = _config(pattern, **overrides)
    torch.manual_seed(seed)
    model = Ernie55HybridModel(
        config=config,
        hybrid_stack_spec=_cpu_spec(),
        vocab_size=VOCAB,
        max_sequence_length=1024,
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
        pg_collection=pg_collection,
    )
    _randomize_(model, seed)
    return model.train()


def _batch(*, seq: int = SEQ, batch: int = 1, seed: int = 0) -> dict:
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, VOCAB, (batch, seq), generator=generator)
    return dict(
        input_ids=tokens,
        position_ids=torch.arange(seq).unsqueeze(0).expand(batch, seq),
        attention_mask=None,
        labels=tokens.roll(-1, dims=1),
    )


def _random_hyper_connection(*, seed: int, hidden: int = HIDDEN) -> HyperConnectionModule:
    """A Megatron-Core ``HyperConnectionModule`` with every parameter drawn at O(1) scale."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=hidden,
        num_attention_heads=4,
        enable_mhc_connections=True,
        mhc_num_residual_streams=STREAMS,
        mhc_sinkhorn_iterations=SINKHORN_ITERATIONS,
        use_cpu_initialization=True,
    )
    module = HyperConnectionModule(config, layer_number=1)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.mapping_proj.weight.copy_(torch.randn(module.mapping_proj.weight.shape, generator=generator) * 0.5)
        module.bias.copy_(torch.randn(module.bias.shape, generator=generator))
        for alpha in (module.alpha_pre, module.alpha_post, module.alpha_res):
            alpha.copy_(0.6 + 0.3 * torch.rand(1, generator=generator))
    return module


def _streams(
    *, dtype: torch.dtype, seed: int, tokens: int = 16, hidden: int = HIDDEN, scale: float = 1.0
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return (3.0 * scale * torch.randn(tokens, 2, STREAMS * hidden, generator=generator)).to(dtype)


# ---------------------------------------------------------------------------------------------
# The ERNIE hyper-connection arithmetic against the fp64 oracle
# ---------------------------------------------------------------------------------------------


class TestSinkhorn:
    @pytest.mark.parametrize("scale", [0.5, 6.0])
    def test_matches_the_oracle_forward_and_backward(self, scale):
        generator = torch.Generator().manual_seed(11)
        logits = (scale * torch.randn(5, 3, STREAMS, STREAMS, generator=generator)).requires_grad_(True)
        cotangent = torch.randn(logits.shape, generator=generator)
        matrix = ernie55_sinkhorn(logits, SINKHORN_ITERATIONS, EPS)
        (matrix * cotangent).sum().backward()

        reference_logits = logits.detach().double().requires_grad_(True)
        expected = _oracle_sinkhorn(reference_logits, SINKHORN_ITERATIONS)
        (expected * cotangent.double()).sum().backward()

        assert matrix.dtype == torch.float32
        assert _norm_err(matrix, expected) <= TOL
        assert _norm_err(logits.grad, reference_logits.grad) <= TOL
        # The last normalization is over columns; rows converge within 20 iterations only for mild logits.
        assert torch.allclose(matrix.sum(dim=-2), torch.ones(()), atol=1e-4)
        if scale < 1:
            assert torch.allclose(matrix.sum(dim=-1), torch.ones(()), atol=1e-4)

    def test_every_epsilon_is_observable_in_fp64(self):
        """Run in fp64, the projection matches the fp64 oracle to rounding, forward and backward.

        At fp32 tolerance the epsilons are invisible: after the softmax every row sum is at least
        about 1/n, so a 1e-6 in a normalizer moves the result by about 1e-6 relative. In fp64 that
        is six orders of magnitude above the tolerance, so the epsilon after the softmax and in every
        row and column normalizer is pinned.
        """
        generator = torch.Generator().manual_seed(21)
        logits = (4.0 * torch.randn(6, 3, STREAMS, STREAMS, generator=generator, dtype=torch.float64)).requires_grad_(
            True
        )
        cotangent = torch.randn(logits.shape, generator=generator, dtype=torch.float64)
        matrix = ernie55_sinkhorn(logits, SINKHORN_ITERATIONS, EPS)
        (matrix * cotangent).sum().backward()
        reference = logits.detach().clone().requires_grad_(True)
        expected = _oracle_sinkhorn(reference, SINKHORN_ITERATIONS)
        (expected * cotangent).sum().backward()

        assert matrix.dtype == torch.float64
        assert _norm_err(matrix, expected) <= 1e-12
        assert _norm_err(logits.grad, reference.grad) <= 1e-10

        def without_row_epsilon(x: torch.Tensor) -> torch.Tensor:
            m = torch.softmax(x, dim=-1) + EPS
            m = m / (m.sum(dim=-2, keepdim=True) + EPS)
            for _ in range(SINKHORN_ITERATIONS - 1):
                m = m / m.sum(dim=-1, keepdim=True)
                m = m / (m.sum(dim=-2, keepdim=True) + EPS)
            return m

        # The tolerance can see a missing row epsilon.
        assert _norm_err(without_row_epsilon(logits.detach()), expected) > 1e-9

    def test_counts_iterations_like_the_specification(self):
        """One iteration is softmax + a column normalization; each further one adds a row and a column pass."""
        logits = torch.randn(4, STREAMS, STREAMS, generator=torch.Generator().manual_seed(3))
        one = ernie55_sinkhorn(logits, 1, EPS).double()
        softmax = torch.softmax(logits.double(), dim=-1) + EPS
        assert _norm_err(one, softmax / (softmax.sum(dim=-2, keepdim=True) + EPS)) <= TOL
        two = ernie55_sinkhorn(logits, 2, EPS).double()
        rows = one / (one.sum(dim=-1, keepdim=True) + EPS)
        assert _norm_err(two, rows / (rows.sum(dim=-2, keepdim=True) + EPS)) <= TOL


def _hyper_connection_params(module: HyperConnectionModule) -> list[torch.Tensor]:
    return [module.mapping_proj.weight, module.alpha_pre, module.alpha_post, module.alpha_res, module.bias]


def _assert_ernie_operations(module: HyperConnectionModule) -> None:
    """Megatron-Core's forward calls these slots; each must hold the ERNIE operation, not Megatron-Core's own."""
    assert module._h_post_bda_op is ernie55_h_post_bda
    assert module._sinkhorn_op is ernie55_sinkhorn
    assert module._h_aggregate_op is ernie55_h_aggregate
    assert module._proj_rms_compute_h_op is None


class TestHyperConnectionArithmetic:
    @pytest.mark.parametrize("stream_scale", [1.0, 1e-6], ids=["unit-rms", "tiny-rms"])
    @pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
    def test_mappings_and_their_gradients_match_the_oracle(self, stream_dtype, stream_scale):
        """``tiny-rms`` streams have an RMS of about 3e-6, comparable to the epsilon of ``r``, so the
        epsilon's place (outside the square root) changes ``r`` and every mapping.
        """
        module = _random_hyper_connection(seed=1)
        use_ernie55_hyper_connections(module)
        x = _streams(dtype=stream_dtype, seed=2, scale=stream_scale).requires_grad_(True)
        if stream_scale < 1:
            rms = x.detach().double().square().mean(dim=-1).sqrt()
            # Inside the square root the epsilon would give a ~250x smaller r.
            assert (torch.rsqrt(rms.square() + EPS) * (rms + EPS)).max() < 0.1
        h_pre, h_post, h_res = module.compute_mappings(x)
        assert (h_pre.dtype, h_post.dtype, h_res.dtype) == (torch.float32,) * 3
        generator = torch.Generator().manual_seed(4)
        cotangents = [torch.randn(t.shape, generator=generator, dtype=torch.float64) for t in (h_pre, h_post, h_res)]
        sum((t * c.float()).sum() for t, c in zip((h_pre, h_post, h_res), cotangents)).backward()

        params = [p.detach().double().requires_grad_(True) for p in _hyper_connection_params(module)]
        reference_x = x.detach().double().requires_grad_(True)
        expected = _oracle_mappings(reference_x, *params, n=STREAMS)
        sum((t * c).sum() for t, c in zip(expected, cotangents)).backward()

        for name, actual, oracle in zip(("h_pre", "h_post", "h_res"), (h_pre, h_post, h_res), expected):
            assert _norm_err(actual, oracle) <= TOL, name
        if stream_dtype == torch.float32:
            assert _norm_err(x.grad, reference_x.grad) <= TOL
        else:
            # The fp32 input gradient is rounded once to the bf16 stream dtype: within one bf16 spacing
            # of every element at unit RMS. At tiny RMS the r^2 factor widens the gradient's dynamic
            # range so much that fp32 cancellation alone moves its smallest elements by more than that,
            # so there the bound is one bf16 spacing of the largest element.
            assert x.grad.dtype == torch.bfloat16
            if stream_scale == 1.0:
                assert _bf16_ulps(x.grad, reference_x.grad).max() <= 1.0
            else:
                assert _norm_err(x.grad, reference_x.grad) <= 2.0**-8
        for param, reference in zip(_hyper_connection_params(module), params):
            assert param.grad.dtype == torch.float32
            assert _norm_err(param.grad, reference.grad) <= TOL

    @pytest.mark.parametrize("stream_dtype", [torch.float32, torch.bfloat16], ids=["fp32-streams", "bf16-streams"])
    def test_layer_step_matches_the_oracle(self, stream_dtype):
        """Megatron-Core's forward plumbing around the ERNIE arithmetic: aggregate, layer, residual update.

        The layer is ``y = tanh(a M)`` in fp32 for fp32 streams; with bf16 streams the layer computes in
        bf16 from the once-rounded fp32 aggregate, as an ERNIE 5.5 MTP layer does.
        """
        module = _random_hyper_connection(seed=5)
        use_ernie55_hyper_connections(module)
        generator = torch.Generator().manual_seed(6)
        layer_weight = (torch.randn(HIDDEN, HIDDEN, generator=generator) / math.sqrt(HIDDEN)).requires_grad_(True)
        x = _streams(dtype=stream_dtype, seed=7).requires_grad_(True)

        aggregated, h_res, h_post, residual = module(x, return_residual=True)
        assert aggregated.dtype == torch.float32
        y = torch.tanh(aggregated.to(stream_dtype) @ layer_weight.to(stream_dtype))
        output = module.fused_h_res_h_post_bda(
            h_res, residual, h_post, (y, None), dropout_prob=0.0, training=True, fused=False
        )
        assert output.dtype == stream_dtype and output.shape == x.shape

        h_pre, h_post_ref, h_res_ref = _oracle_mappings(x.detach(), *_hyper_connection_params(module), n=STREAMS)
        expected_aggregate = _oracle_aggregate(x.detach(), h_pre, STREAMS)
        assert _norm_err(aggregated, expected_aggregate) <= TOL
        expected = _oracle_update(x.detach(), h_res_ref, h_post_ref, y.detach(), STREAMS)
        if stream_dtype == torch.float32:
            assert _norm_err(output, expected) <= TOL
        else:
            # fp32 arithmetic rounded once: within one bf16 spacing everywhere, exact almost everywhere.
            ulps = _bf16_ulps(output, expected)
            assert ulps.max() <= 1.0, ulps.max()
            assert (ulps == 0).double().mean() >= 0.99

        if stream_dtype == torch.float32:
            cotangent = torch.randn(output.shape, generator=generator)
            (output * cotangent).sum().backward()
            params = [p.detach().double().requires_grad_(True) for p in _hyper_connection_params(module)]
            reference_x = x.detach().double().requires_grad_(True)
            reference_weight = layer_weight.detach().double().requires_grad_(True)
            ref_pre, ref_post, ref_res = _oracle_mappings(reference_x, *params, n=STREAMS)
            ref_y = torch.tanh(_oracle_aggregate(reference_x, ref_pre, STREAMS) @ reference_weight)
            (_oracle_update(reference_x, ref_res, ref_post, ref_y, STREAMS) * cotangent.double()).sum().backward()
            assert _norm_err(x.grad, reference_x.grad) <= TOL
            assert _norm_err(layer_weight.grad, reference_weight.grad) <= TOL
            for param, reference in zip(_hyper_connection_params(module), params):
                assert _norm_err(param.grad, reference.grad) <= TOL

    def test_bf16_streams_mix_in_fp32_unlike_megatron_core(self):
        """Why the MTP layer needs the ERNIE arithmetic: Megatron-Core rounds its mappings to bf16 streams.

        Same parameters, inputs and layer output; Megatron-Core's own module (also with
        ``mhc_keep_mappings_in_fp32``, which matches the ERNIE contract up to compiled rounding).
        """
        ernie = _random_hyper_connection(seed=8)
        native = _random_hyper_connection(seed=8)
        use_ernie55_hyper_connections(ernie)
        x = _streams(dtype=torch.bfloat16, seed=9, tokens=32)
        y = (2.0 * torch.randn(32, 2, HIDDEN, generator=torch.Generator().manual_seed(10))).to(torch.bfloat16)
        h_pre, h_post, h_res = _oracle_mappings(x, *_hyper_connection_params(ernie), n=STREAMS)
        expected = _oracle_update(x, h_res, h_post, y, STREAMS)

        def step(module):
            with torch.no_grad():
                _, mod_res, mod_post, residual = module(x, return_residual=True)
                return module.fused_h_res_h_post_bda(
                    mod_res, residual, mod_post, (y, None), dropout_prob=0.0, training=True, fused=False
                )

        ernie_ulps = _bf16_ulps(step(ernie), expected)
        native_ulps = _bf16_ulps(step(native), expected)
        assert ernie_ulps.max() <= 1.0
        assert native_ulps.max() >= 2.0, "Megatron-Core's bf16 mappings were expected to add rounding error"
        assert native_ulps.mean() > 2 * ernie_ulps.mean()
        if _HAS_KEEP_MAPPINGS_IN_FP32:
            native.config.mhc_keep_mappings_in_fp32 = True
            assert _bf16_ulps(step(native), expected).max() <= 1.0


class TestHyperConnectionConversion:
    def test_converts_in_place_and_keeps_parameters(self):
        module = _random_hyper_connection(seed=12)
        params = dict(module.named_parameters())
        state = {name: value.clone() for name, value in module.state_dict().items()}
        container = torch.nn.ModuleDict({"a": module, "b": _random_hyper_connection(seed=13)})
        assert use_ernie55_hyper_connections(container) == 2
        assert type(module) is Ernie55HyperConnectionModule
        assert isinstance(module, HyperConnectionModule)
        assert dict(module.named_parameters()) == params  # the same Parameter objects
        assert module.state_dict().keys() == state.keys()
        assert all(torch.equal(module.state_dict()[name], value) for name, value in state.items())
        _assert_ernie_operations(module)
        # Idempotent.
        assert use_ernie55_hyper_connections(container) == 2 and type(module) is Ernie55HyperConnectionModule
        _assert_ernie_operations(module)

    def test_direct_construction_uses_the_ernie_operations(self):
        """A module constructed as ``Ernie55HyperConnectionModule`` equals a converted one, step for step."""
        converted = _random_hyper_connection(seed=16)
        use_ernie55_hyper_connections(converted)
        direct = Ernie55HyperConnectionModule(converted.config, layer_number=1)
        direct.load_state_dict(converted.state_dict())
        _assert_ernie_operations(direct)
        x = _streams(dtype=torch.bfloat16, seed=17)
        y = torch.randn(x.shape[0], x.shape[1], HIDDEN, generator=torch.Generator().manual_seed(18)).to(x.dtype)
        outputs = []
        for module in (converted, direct):
            with torch.no_grad():
                _, h_res, h_post, residual = module(x, return_residual=True)
                outputs.append(
                    module.fused_h_res_h_post_bda(
                        h_res, residual, h_post, (y, None), dropout_prob=0.0, training=True, fused=False
                    )
                )
        assert torch.equal(outputs[0], outputs[1])

    def test_rejects_unknown_subclasses_and_graph_outputs(self):
        class _Custom(HyperConnectionModule):
            pass

        module = _random_hyper_connection(seed=14)
        module.__class__ = _Custom
        with pytest.raises(TypeError, match="cannot convert"):
            use_ernie55_hyper_connections(module)
        module = _random_hyper_connection(seed=14)
        use_ernie55_hyper_connections(module)
        x = _streams(dtype=torch.float32, seed=15)
        h_pre, _, _ = module.compute_mappings(x)
        with pytest.raises(NotImplementedError, match="caller-owned"):
            module.aggregate(x, h_pre, out=torch.empty(x.shape[0], x.shape[1], HIDDEN))


class TestValidation:
    @pytest.mark.parametrize(
        "overrides, message",
        [
            (dict(use_fused_mhc=True), "use_fused_mhc"),
            (dict(hidden_dropout=0.1), "hidden_dropout"),
            (dict(fp32_residual_connection=True), "fp32_residual_connection"),
            (dict(pipeline_model_parallel_size=2, pipeline_dtype=torch.bfloat16), "pipeline_dtype"),
            (dict(mhc_norm_eps_inside_sqrt=True), "mhc_norm_eps_inside_sqrt"),
            (dict(mhc_learned_output_contract=False), "mhc_learned_output_contract"),
        ],
        ids=["fused", "dropout", "fp32-residual", "pipeline-dtype", "eps-inside-sqrt", "mean-contract"],
    )
    def test_rejects_settings_outside_the_envelope(self, overrides, message):
        config = _config()
        for name, value in overrides.items():
            if not hasattr(config, name):
                pytest.skip(f"this Megatron-Core has no {name}")
            setattr(config, name, value)
        with pytest.raises(ValueError, match=message):
            validate_ernie55_mhc_config(config)

    def test_accepts_the_envelope_and_configs_without_mhc(self):
        config = _config()
        validate_ernie55_mhc_config(config)
        config.pipeline_model_parallel_size, config.pipeline_dtype = 2, torch.float32
        validate_ernie55_mhc_config(config)
        validate_ernie55_mhc_config(_config("H-DE", enable_mhc_connections=False, use_fused_mhc=False))

    def test_finalize_validates_hyper_connections(self):
        with pytest.raises(ValueError, match="hidden_dropout"):
            _config(hidden_dropout=0.1)


# ---------------------------------------------------------------------------------------------
# Downcasting norms and the stack spec
# ---------------------------------------------------------------------------------------------


class TestDowncastingNorm:
    def test_rounds_the_input_once_and_keeps_the_parameter(self):
        config = _config("H-DE")
        norm = downcasting_norm(_RMSNorm)(config=config, hidden_size=HIDDEN, eps=EPS)
        assert is_downcasting_norm(norm) and [name for name, _ in norm.named_parameters()] == ["weight"]
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, HIDDEN))
        x = (5.0 * torch.randn(6, 2, HIDDEN, generator=torch.Generator().manual_seed(1))).requires_grad_(True)
        output = norm(x)
        reference = _RMSNorm(config, HIDDEN, eps=EPS)
        reference.load_state_dict(norm.state_dict())
        rounded = x.detach().to(torch.bfloat16).requires_grad_(True)
        expected = reference(rounded)
        assert output.dtype == torch.bfloat16 and torch.equal(output, expected)
        cotangent = torch.randn(output.shape, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
        (output.float() * cotangent.float()).sum().backward()
        (expected.float() * cotangent.float()).sum().backward()
        assert x.grad.dtype == torch.float32 and torch.equal(x.grad, rounded.grad.float())
        # Weight-dtype inputs pass unchanged, and applying it twice registers one hook.
        assert torch.equal(norm(rounded.detach()), reference(rounded.detach()))
        from megatron.bridge.models.ernie.modeling_ernie55.norm import downcast_norm_input

        assert downcast_norm_input(norm) is norm and len(norm._forward_pre_hooks) == 1

    def test_stack_spec_downcasts_every_stream_input(self):
        from megatron.core.extensions.transformer_engine import TENorm
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

        spec = ernie55_hybrid_stack_spec()
        expected = downcasting_norm(TENorm)

        def downcasting_te_norm(builder) -> bool:
            return (
                isinstance(builder, functools.partial)
                and builder.func is expected.func
                and builder.keywords == expected.keywords
            )

        assert spec.module is Ernie55HybridStack
        slots = {f.name for f in dataclasses.fields(spec.submodules)}
        attention_slots = ["dsa_layer"] + [
            s for s in ("csa_layer", "csa_qk_layernorm_layer", "hca_layer") if s in slots
        ]
        for slot in attention_slots:
            assert downcasting_te_norm(getattr(spec.submodules, slot).submodules.input_layernorm), slot
        for slot in ("mlp_layer", "moe_layer"):
            assert downcasting_te_norm(getattr(spec.submodules, slot).submodules.pre_mlp_layernorm), slot
        fixed_hnorm = downcasting_norm(TENorm, reference_switch=False)
        for layer in spec.submodules.mtp_block_spec.submodules.layer_specs:
            # hnorm sees the decoder's fp32 streams; enorm and the final norm see bf16. All three stay TENorm
            # under ernie_reference_rmsnorm, as in the reference (Megatron-Core's TE MTP spec).
            hnorm = layer.submodules.hnorm
            assert hnorm.func is fixed_hnorm.func and hnorm.keywords == fixed_hnorm.keywords
            assert layer.submodules.enorm is TENorm and layer.submodules.layer_norm is TENorm
        # Megatron-Core's shared spec is untouched.
        assert hybrid_stack_spec.module is not Ernie55HybridStack
        for layer in hybrid_stack_spec.submodules.mtp_block_spec.submodules.layer_specs:
            assert layer.submodules.hnorm is TENorm
        assert hybrid_stack_spec.submodules.moe_layer.submodules.pre_mlp_layernorm is TENorm


def _reference_fp32_pool_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference implementation's ``ErnieRMSNorm.forward``, statement for statement."""
    x = x.to(weight.dtype)
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(weight.dtype)


class TestReferenceRMSNorm:
    """``ernie_reference_rmsnorm``: the reference's eager RMSNorm arithmetic in every ERNIE norm slot."""

    @pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16], ids=["fp32-stream", "bf16"])
    def test_is_the_reference_expression(self, input_dtype):
        config = _config("H-DE", ernie_reference_rmsnorm=True)
        norm = Ernie55RMSNorm(config, HIDDEN, eps=EPS)
        assert [name for name, _ in norm.named_parameters()] == ["weight"]
        assert norm.weight.dtype == torch.bfloat16 and bool((norm.weight == 1).all())
        assert norm.weight.sequence_parallel is False and norm.returns_residual is False
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, HIDDEN))
        weight = norm.weight.detach().clone().requires_grad_(True)
        x = (5.0 * torch.randn(6, 2, HIDDEN, generator=torch.Generator().manual_seed(3))).to(input_dtype)
        x_ours, x_ref = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        output = norm(x_ours)
        expected = _reference_fp32_pool_rmsnorm(x_ref, weight, EPS)
        assert output.dtype == torch.bfloat16 and torch.equal(output, expected)
        cotangent = torch.randn(output.shape, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
        output.backward(cotangent)
        expected.backward(cotangent)
        assert torch.equal(x_ours.grad, x_ref.grad) and torch.equal(norm.weight.grad, weight.grad)

    def test_builders_follow_the_switch(self):
        for builder in (ernie55_rmsnorm(_RMSNorm), downcasting_norm(_RMSNorm)):
            plain = builder(config=_config("H-DE"), hidden_size=HIDDEN, eps=EPS)
            reference = builder(config=_config("H-DE", ernie_reference_rmsnorm=True), hidden_size=HIDDEN, eps=EPS)
            assert type(plain) is _RMSNorm and type(reference) is Ernie55RMSNorm and reference.eps == EPS
            downcasts = builder.func is downcasting_norm(_RMSNorm).func
            assert is_downcasting_norm(plain) is downcasts and is_downcasting_norm(reference) is downcasts
        # The MTP hnorm's builder ignores the switch.
        fixed = downcasting_norm(_RMSNorm, reference_switch=False)
        norm = fixed(config=_config("H-DE", ernie_reference_rmsnorm=True), hidden_size=HIDDEN, eps=EPS)
        assert type(norm) is _RMSNorm and is_downcasting_norm(norm)

    @pytest.mark.parametrize(
        "overrides, kwargs, message",
        [
            (dict(normalization="LayerNorm"), {}, "normalization='RMSNorm'"),
            (dict(layernorm_zero_centered_gamma=True), {}, "zero_centered_gamma"),
            ({}, dict(has_residual=True), "has_residual"),
        ],
        ids=["layernorm", "zero-centered-gamma", "fused-residual"],
    )
    def test_rejects_what_it_does_not_implement(self, overrides, kwargs, message):
        config = types.SimpleNamespace(
            normalization="RMSNorm", layernorm_zero_centered_gamma=False, params_dtype=torch.bfloat16
        )
        for key, value in overrides.items():
            setattr(config, key, value)
        with pytest.raises(ValueError, match=message):
            Ernie55RMSNorm(config, HIDDEN, eps=EPS, **kwargs)

    def test_finalize_requires_a_bool(self):
        with pytest.raises(ValueError, match="ernie_reference_rmsnorm must be a bool"):
            _config("H-DE", ernie_reference_rmsnorm=1)

    @needs_hook
    def test_every_rmsnorm_of_the_model_switches(self, pg_collection):
        """Every RMSNorm the reference builds as ErnieRMSNorm follows the switch; the MTP block norms do not.

        On CPU the stand-in TE norm sums the squares in the same order as the reference expression, so the two
        models are bitwise equal here; on GPU only the switched model reproduces the reference's bits.
        """
        plain = _build_model(pg_collection, seed=5)
        reference = _build_model(pg_collection, seed=5, ernie_reference_rmsnorm=True)
        plain_norms = {name: module for name, module in plain.named_modules() if type(module) is _RMSNorm}
        reference_norms = {
            name: module for name, module in reference.named_modules() if type(module) is Ernie55RMSNorm
        }
        mtp_block_norms = {"mtp.layers.0.enorm", "mtp.layers.0.hnorm", "mtp.layers.0.final_layernorm"}
        assert mtp_block_norms <= plain_norms.keys()
        assert reference_norms and plain_norms.keys() == reference_norms.keys() | mtp_block_norms
        assert {name for name, module in reference.named_modules() if type(module) is _RMSNorm} == mtp_block_norms
        names = set(reference_norms)
        for expected in (
            "decoder.layers.0.inner_layer.input_layernorm",
            "decoder.layers.0.inner_layer.self_attention.q_layernorm",
            "decoder.layers.0.inner_layer.self_attention.kv_layernorm",
            "decoder.layers.0.inner_layer.self_attention.core_attention.compressor.norm",
            "decoder.layers.1.inner_layer.pre_mlp_layernorm",
            "decoder.layers.2.inner_layer.self_attention.q_layernorm",
            "decoder.layers.2.inner_layer.self_attention.kv_layernorm",
            "decoder.final_norm",
            "mtp.layers.0.mtp_model_layer.layers.0.inner_layer.input_layernorm",
            "mtp.layers.0.mtp_model_layer.layers.0.inner_layer.self_attention.q_layernorm",
            "mtp.layers.0.mtp_model_layer.layers.1.inner_layer.pre_mlp_layernorm",
        ):
            assert expected in names, expected
        for name, module in reference_norms.items():
            assert is_downcasting_norm(module) == is_downcasting_norm(plain_norms[name]), name
        assert is_downcasting_norm(reference.get_submodule("mtp.layers.0.hnorm"))
        # The D indexer's key norm is a LayerNorm in both.
        assert type(reference.decoder.layers[2].inner_layer.self_attention.core_attention.indexer.k_norm) is type(
            plain.decoder.layers[2].inner_layer.self_attention.core_attention.indexer.k_norm
        )
        assert [(n, p.shape, p.dtype) for n, p in plain.named_parameters()] == [
            (n, p.shape, p.dtype) for n, p in reference.named_parameters()
        ]
        assert plain.state_dict().keys() == reference.state_dict().keys()
        losses = [model(**_batch(seed=6)) for model in (plain, reference)]
        assert torch.equal(losses[0], losses[1])


# ---------------------------------------------------------------------------------------------
# A whole ERNIE model: fp32 decoder streams, bf16 layers, bf16 MTP streams
# ---------------------------------------------------------------------------------------------


class _DtypeLog:
    """Forward hooks recording input and output dtypes (and shapes) by module name."""

    def __init__(self) -> None:
        self.records: list[tuple[str, str, torch.dtype, tuple, torch.dtype]] = []
        self._pending: dict[int, list[tuple[torch.dtype, tuple]]] = {}

    def watch(self, name: str, module: torch.nn.Module, kind: str) -> None:
        def pre_hook(module, args, kwargs):
            hidden = args[0] if args else kwargs.get("hidden_states")
            self._pending.setdefault(id(module), []).append((hidden.dtype, tuple(hidden.shape)))

        def hook(module, args, kwargs, output):
            output = output[0] if isinstance(output, tuple) else output
            in_dtype, in_shape = self._pending[id(module)].pop()
            self.records.append((kind, name, in_dtype, in_shape, output.dtype))

        # prepend: see the input before a downcasting norm's own pre-hook casts it.
        module.register_forward_pre_hook(pre_hook, with_kwargs=True, prepend=True)
        module.register_forward_hook(hook, with_kwargs=True)

    def of(self, kind: str, prefix: str = "") -> list[tuple]:
        return [record for record in self.records if record[0] == kind and record[1].startswith(prefix)]


@needs_hook
class TestFp32Streams:
    def test_decoder_streams_are_fp32_and_layers_bf16(self, pg_collection):
        from megatron.core.models.hybrid.layers.hybrid_hyper_connection import HyperConnectionHybridLayer

        model = _build_model(pg_collection)
        log = _DtypeLog()
        for name, module in model.named_modules():
            if isinstance(module, HyperConnectionHybridLayer):
                log.watch(name, module, "wrapper")
            elif is_downcasting_norm(module):
                log.watch(name, module, "downcasting-norm")
            elif name in ("decoder", "output_layer"):
                log.watch(name, module, name)
        # The MoE router, routed experts and shared expert see the bf16 pre-norm output, never a stream.
        moe_inputs = []
        for name, module in model.named_modules():
            if isinstance(module, MoELayer):
                for part in ("router", "experts", "shared_experts"):
                    getattr(module, part).register_forward_pre_hook(
                        lambda module, args, part=f"{name}.{part}": moe_inputs.append((part, args[0].dtype))
                    )
        embeddings = []
        model.embedding.register_forward_hook(lambda module, args, output: embeddings.append(output))
        first_input = []
        model.decoder.layers[0].register_forward_pre_hook(
            lambda module, args, kwargs: first_input.append(kwargs["hidden_states"]), with_kwargs=True
        )
        loss = model(**_batch())
        loss.sum().backward()

        # The embedding output (the main one; MTP embeds the rolled tokens later) is bf16; the streams
        # are 4 fp32 copies of it.
        assert embeddings[0].dtype == torch.bfloat16
        assert first_input[0].dtype == torch.float32
        assert torch.equal(first_input[0], embeddings[0].float().repeat(1, 1, STREAMS))
        decoder_wrappers = log.of("wrapper", "decoder.")
        mtp_wrappers = log.of("wrapper", "mtp.")
        assert len(decoder_wrappers) == 4 and len(mtp_wrappers) == 2
        for _, name, in_dtype, in_shape, out_dtype in decoder_wrappers:
            assert (in_dtype, out_dtype, in_shape[-1]) == (torch.float32, torch.float32, STREAMS * HIDDEN), name
        for _, name, in_dtype, in_shape, out_dtype in mtp_wrappers:
            assert (in_dtype, out_dtype, in_shape[-1]) == (torch.bfloat16, torch.bfloat16, STREAMS * HIDDEN), name
        # Every pre-norm receives the fp32 layer input and hands bf16 to its layer.
        norms = log.of("downcasting-norm")
        names = {record[1] for record in norms}
        assert {
            "decoder.layers.0.inner_layer.input_layernorm",
            "decoder.layers.1.inner_layer.pre_mlp_layernorm",
            "decoder.layers.2.inner_layer.input_layernorm",
            "decoder.layers.3.inner_layer.pre_mlp_layernorm",
            "decoder.final_norm",
            "mtp.layers.0.hnorm",
            "mtp.layers.0.mtp_model_layer.layers.0.inner_layer.input_layernorm",
            "mtp.layers.0.mtp_model_layer.layers.1.inner_layer.pre_mlp_layernorm",
        } <= names
        for _, name, in_dtype, in_shape, out_dtype in norms:
            assert (in_dtype, out_dtype) == (torch.float32, torch.bfloat16), name
        hnorm = [record for record in norms if record[1] == "mtp.layers.0.hnorm"][0]
        assert hnorm[3][-2:] == (STREAMS, HIDDEN)
        ((_, _, _, _, decoder_out),) = log.of("decoder")
        assert decoder_out == torch.bfloat16
        # The MTP head runs before the main head; both read bf16 hidden states.
        output_calls = log.of("output_layer")
        assert [(record[2], record[4]) for record in output_calls] == [(torch.bfloat16, torch.bfloat16)] * 2
        modules = [m for m in model.modules() if isinstance(m, HyperConnectionModule)]
        assert len(modules) == 6 and all(type(m) is Ernie55HyperConnectionModule for m in modules)
        assert len(moe_inputs) == 6 and all(dtype == torch.bfloat16 for _, dtype in moe_inputs), moe_inputs

    def test_stack_rejects_norms_that_do_not_downcast(self, pg_collection):
        from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers

        spec = _cpu_spec()
        spec.submodules.moe_layer.submodules.pre_mlp_layernorm = _StackNorm
        config = _config("H-DE", ernie_mtp_shared_last_layer=False)
        with pytest.raises(TypeError, match="pre_mlp_layernorm that does not downcast"):
            Ernie55HybridStack(
                config,
                spec.submodules,
                layer_config_list=validate_segment_layers("H-DE", config),
                pg_collection=pg_collection,
            )

    def test_stacks_without_fp32_streams(self, pg_collection):
        """MTP stacks and stacks without hyper-connections pass their input through uncast."""
        model = _build_model(pg_collection)
        assert model.decoder.carries_fp32_streams
        assert type(model.mtp.layers[0].mtp_model_layer).__name__ == "HybridStack"
        plain = _build_model(
            pg_collection, pattern="H-DE", enable_mhc_connections=False, ernie_mtp_shared_last_layer=False
        )
        assert isinstance(plain.decoder, Ernie55HybridStack) and not plain.decoder.carries_fp32_streams
        assert not any(isinstance(m, HyperConnectionModule) for m in plain.modules())
        seen = []
        plain.decoder.layers[0].register_forward_pre_hook(
            lambda module, args, kwargs: seen.append(kwargs["hidden_states"].dtype), with_kwargs=True
        )
        plain(**_batch(seq=40))
        assert seen == [torch.bfloat16]


# ---------------------------------------------------------------------------------------------
# SD-MTP
# ---------------------------------------------------------------------------------------------


def _twin_without_sharing(pg_collection, model: Ernie55HybridModel, **overrides) -> Ernie55HybridModel:
    """An unshared model computing the same function: its MTP body holds copies of the decoder layers."""
    twin = _build_model(pg_collection, seed=99, ernie_mtp_shared_last_layer=False, **overrides)
    state = model.state_dict()
    for alias, source in model.ernie55_mtp_aliases.items():
        state[alias] = state[source]
    twin.load_state_dict(state)
    return twin


@needs_hook
class TestSharedMTP:
    def test_mtp_body_aliases_the_last_decoder_layers(self, pg_collection):
        model = _build_model(pg_collection)
        aliases = model.ernie55_mtp_aliases
        body = model.mtp.layers[0].mtp_model_layer
        body_names = {f"mtp.layers.0.mtp_model_layer.{name}" for name, _ in body.named_parameters()}
        assert set(aliases) == body_names and len(aliases) > 20
        for alias, source in aliases.items():
            assert source.startswith(("decoder.layers.2.", "decoder.layers.3.")), source
            assert alias.split("mtp_model_layer.layers.", 1)[1][1:] == source.split("decoder.layers.", 1)[1][1:]
            assert model.get_parameter(alias) is model.get_parameter(source)
        # The router, both hyper-connection modules and the attention are shared.
        for fragment in ("hyper_connection.mapping_proj.weight", "mlp.router.weight_1", "self_attention.k_b_proj"):
            assert any(fragment in alias for alias in aliases), fragment
        # Each tensor once, under its decoder name.
        names = [name for name, _ in model.named_parameters()]
        assert not [name for name in names if "mtp_model_layer" in name]
        assert len({id(p) for p in model.parameters()}) == len(list(model.parameters())) == len(names)
        # Buffers stay separate: the MTP router updates its own expert bias from its own counts.
        mtp_router = body.layers[1].inner_layer.mlp.router
        decoder_router = model.decoder.layers[3].inner_layer.mlp.router
        assert mtp_router.weight is decoder_router.weight
        assert mtp_router.expert_bias is not decoder_router.expert_bias
        assert mtp_router.local_tokens_per_expert is not decoder_router.local_tokens_per_expert
        # The MTP head stays MTP-owned.
        decoder_ids = {id(p) for p in model.decoder.parameters()}
        for name in (
            "enorm.weight",
            "hnorm.weight",
            "e_proj.weight",
            "h_proj.weight",
            "hc_head_fn",
            "final_layernorm.weight",
        ):
            assert id(model.mtp.layers[0].get_parameter(name)) not in decoder_ids, name

    def test_gradients_sum_both_uses(self, pg_collection):
        """Against an unshared twin with equal weights: shared grad = decoder-use grad + MTP-use grad."""
        model = _build_model(pg_collection, params_dtype=torch.float32)
        twin = _twin_without_sharing(pg_collection, model, params_dtype=torch.float32)
        batch = _batch()
        for module in (model, twin):
            module(**batch).sum().backward()
        shared = set(model.ernie55_mtp_aliases.values())
        for alias, source in model.ernie55_mtp_aliases.items():
            expected = twin.get_parameter(source).grad + twin.get_parameter(alias).grad
            assert twin.get_parameter(alias).grad.abs().max() > 0, alias
            assert _norm_err(model.get_parameter(source).grad, expected) <= 1e-6, source
        for name, param in model.named_parameters():
            if name not in shared and param.grad is not None:
                assert _norm_err(param.grad, twin.get_parameter(name).grad) <= 1e-6, name

    def test_guards(self, pg_collection):
        with pytest.raises(RuntimeError, match="differ from the decoder's last layers"):
            _build_model(pg_collection, pattern="DEH-/DE")
        with pytest.raises(RuntimeError, match="mtp_detach_heads"):
            _build_model(pg_collection, mtp_detach_heads=True)
        model = _build_model(pg_collection, ernie_mtp_shared_last_layer=False)
        model.config.freeze_base_model_for_mtp = True
        with pytest.raises(RuntimeError, match="freeze_base_model_for_mtp"):
            share_mtp_body_with_last_layer(model)
        model.config.freeze_base_model_for_mtp = False
        model.decoder.layers[-1].layer_number = 99
        with pytest.raises(RuntimeError, match="pipeline stage"):
            share_mtp_body_with_last_layer(model)
        with pytest.raises(ValueError, match="no MTP"):
            _build_model(pg_collection, pattern="H-DE", ernie_mtp_shared_last_layer=True)
        # An attribute DDP, the optimizers or checkpointing read must agree within each aliased pair.
        model = _build_model(pg_collection, ernie_mtp_shared_last_layer=False)
        next(model.mtp.layers[0].mtp_model_layer.parameters()).allreduce = False
        with pytest.raises(RuntimeError, match="differ in 'allreduce'"):
            share_mtp_body_with_last_layer(model)
        # Meta-device models: materializing them would re-create the parameters module by module.
        model = _build_model(pg_collection, ernie_mtp_shared_last_layer=False)
        model.config.init_model_with_meta_device = True
        with pytest.raises(RuntimeError, match="meta device"):
            share_mtp_body_with_last_layer(model)
        model.config.init_model_with_meta_device = False
        weight = model.output_layer.weight
        model.output_layer._parameters["weight"] = torch.nn.Parameter(torch.empty(weight.shape, device="meta"))
        with pytest.raises(RuntimeError, match="meta device"):
            share_mtp_body_with_last_layer(model)

    def test_alias_check_and_loads_that_would_split_the_sharing(self, pg_collection):
        model = _build_model(pg_collection)
        count = len(model.ernie55_mtp_aliases)
        assert check_ernie55_mtp_aliases(model) == count > 0
        # Through a wrapper, and over a list of model chunks.
        assert check_ernie55_mtp_aliases([torch.nn.ModuleDict({"module": model})]) == count
        unshared = _build_model(pg_collection, pattern="H-DE", ernie_mtp_shared_last_layer=False)
        assert check_ernie55_mtp_aliases(unshared) == 0
        state = model.state_dict()
        # assign=True would wrap the decoder entry and its MTP copy in two Parameters: rejected before loading.
        # (torch records assign=True in the state dict's own metadata, so that dict is not reused below.)
        with pytest.raises(RuntimeError, match="assign=True"):
            model.load_state_dict(model.state_dict(), assign=True)
        assert check_ernie55_mtp_aliases(model) == count
        # A parameter re-created module by module (meta materialization, FSDP) is caught, also by a later load.
        alias = sorted(model.ernie55_mtp_aliases)[0]
        owner_name, _, leaf = alias.rpartition(".")
        owner = model.get_submodule(owner_name)
        owner._parameters[leaf] = torch.nn.Parameter(owner._parameters[leaf].detach().clone())
        with pytest.raises(RuntimeError, match="no longer share"):
            check_ernie55_mtp_aliases(model)
        with pytest.raises(RuntimeError, match="no longer share"):
            model.load_state_dict(state)

    def test_training_setups_that_split_the_sharing_are_rejected(self):
        config = _config()
        validate_ernie55_mtp_sharing(config, use_megatron_fsdp=False, use_torch_fsdp2=False)
        for flags, name in (
            (dict(use_megatron_fsdp=True, use_torch_fsdp2=False), "use_megatron_fsdp"),
            (dict(use_megatron_fsdp=False, use_torch_fsdp2=True), "use_torch_fsdp2"),
        ):
            with pytest.raises(ValueError, match=name):
                validate_ernie55_mtp_sharing(config, **flags)
        config.init_model_with_meta_device = True
        with pytest.raises(ValueError, match="init_model_with_meta_device"):
            validate_ernie55_mtp_sharing(config, use_megatron_fsdp=False, use_torch_fsdp2=False)
        # Without SD-MTP nothing is shared, so nothing is rejected.
        unshared = _config("H-DE", ernie_mtp_shared_last_layer=False, init_model_with_meta_device=True)
        validate_ernie55_mtp_sharing(unshared, use_megatron_fsdp=True, use_torch_fsdp2=True)

    def test_state_dicts_emit_shared_parameters_once(self, pg_collection):
        from megatron.core.dist_checkpointing.dict_utils import nested_values
        from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory

        model = _build_model(pg_collection)
        state = model.state_dict()
        assert not set(model.ernie55_mtp_aliases) & set(state)
        assert {name for name, _ in model.named_parameters()} <= set(state)

        sharded = model.sharded_state_dict()
        entries = [item for item in nested_values(sharded) if isinstance(item, (ShardedTensor, ShardedTensorFactory))]
        param_ids = [id(item.data) for item in entries if isinstance(item.data, torch.nn.Parameter)]
        assert len(param_ids) == len(set(param_ids)) == len(list(model.parameters()))
        assert not [
            key
            for key in sharded
            if "mtp_model_layer" in key and "expert_bias" not in key and "_extra_state" not in key
        ]

        restored = _build_model(pg_collection, seed=5)
        restored.load_state_dict(state)
        for name, param in model.named_parameters():
            assert torch.equal(restored.get_parameter(name), param), name
        for alias, source in restored.ernie55_mtp_aliases.items():
            assert restored.get_parameter(alias) is restored.get_parameter(source)
        # A state dict that also carries the MTP body (as an unshared model writes it) must agree with the decoder.
        twin_state = _twin_without_sharing(pg_collection, model).state_dict()
        restored.load_state_dict(twin_state)
        alias = next(iter(model.ernie55_mtp_aliases))
        twin_state[alias] = twin_state[alias] + 1
        with pytest.raises(RuntimeError, match="SD-MTP"):
            restored.load_state_dict(twin_state)

    def test_torch_dist_round_trip(self, pg_collection, tmp_path):
        from megatron.core import dist_checkpointing
        from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

        model = _build_model(pg_collection, seed=1)
        dist_checkpointing.save({"model": model.sharded_state_dict()}, str(tmp_path))
        keys = set(load_tensors_metadata(str(tmp_path)))
        assert not [key for key in keys if "mtp_model_layer" in key and "expert_bias" not in key]
        assert "mtp.layers.0.mtp_model_layer.layers.1.inner_layer.mlp.router.expert_bias" in keys

        restored = _build_model(pg_collection, seed=2)
        loaded = dist_checkpointing.load({"model": restored.sharded_state_dict()}, str(tmp_path))
        restored.load_state_dict(loaded["model"])
        for name, param in model.named_parameters():
            assert torch.equal(restored.get_parameter(name), param), name
        for (name, buffer), (_, other) in zip(model.named_buffers(), restored.named_buffers()):
            assert torch.equal(buffer, other), name
        assert all(
            restored.get_parameter(a) is restored.get_parameter(s) for a, s in restored.ernie55_mtp_aliases.items()
        )


# ---------------------------------------------------------------------------------------------
# MTP targets on a packed THD row
# ---------------------------------------------------------------------------------------------


def _oracle_roll_per_document(row: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
    """Shift a packed row left by one inside every document; each document's last position gets 0."""
    rolled = torch.zeros_like(row)
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        rolled[..., start : end - 1] = row[..., start + 1 : end]
    return rolled


@needs_hook
def test_mtp_targets_roll_inside_each_document(pg_collection):
    """The logged MTP loss uses per-document targets: labels (already shifted once, over the whole row)
    and the loss mask both move one more step inside each document, and each document's last position
    is masked, so no MTP target crosses a document boundary.
    """
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    doc_lens = (37, 60, 23)
    cu_seqlens = [0, 37, 97, 120]
    model = _build_model(pg_collection, pattern="-DE/DE", params_dtype=torch.float32)
    generator = torch.Generator().manual_seed(17)
    tokens = torch.randint(0, VOCAB, (1, cu_seqlens[-1]), generator=generator)
    labels = tokens.roll(-1, dims=1)  # the global shift a packed SFT batch arrives with
    loss_mask = torch.ones(1, cu_seqlens[-1])
    loss_mask[0, [end - 1 for end in cu_seqlens[1:]]] = 0.0  # no main target across documents
    position_ids = torch.cat([torch.arange(length) for length in doc_lens]).unsqueeze(0)
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    packed = PackedSeqParams(
        qkv_format="thd", cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(doc_lens), max_seqlen_kv=max(doc_lens)
    )
    head_logits = []
    # Clone: with fp32 logits Megatron-Core's cross-entropy works in place on its input.
    model.output_layer.register_forward_hook(
        lambda module, args, output: head_logits.append(output[0].detach().clone())
    )
    model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=packed,
    )
    tracker = MTPLossLoggingHelper.tracker
    logged = float(tracker.get("loss_sums", tracker.get("loss_values"))[0])

    mtp_logits = head_logits[0].double()[:, 0]  # the MTP head runs before the main head
    log_probs = torch.log_softmax(mtp_logits, dim=-1)

    def mtp_loss(targets: torch.Tensor, mask: torch.Tensor) -> float:
        nll = -log_probs.gather(-1, targets[0].unsqueeze(-1)).squeeze(-1)
        return float((nll * mask[0].double()).sum() / mask.sum())

    targets = _oracle_roll_per_document(labels, cu_seqlens)
    mask = _oracle_roll_per_document(loss_mask, cu_seqlens)
    for end in cu_seqlens[1:]:
        assert mask[0, end - 2] == 0 and mask[0, end - 1] == 0
    assert logged == pytest.approx(mtp_loss(targets, mask), rel=1e-5)
    # A roll over the whole row (no per-document fill) would train across documents and differ.
    assert abs(logged - mtp_loss(labels.roll(-1, dims=1), loss_mask.roll(-1, dims=1))) > 1e-3


# ---------------------------------------------------------------------------------------------
# The multimax head inside the model
# ---------------------------------------------------------------------------------------------


@needs_hook
def test_every_pipeline_stage_registers_the_multimax_group(pg_collection, monkeypatch):
    """A stage without the output layer registers the multimax group too, in the same place.

    For each registered group, Megatron-Core's optimizer all-reduces a flag over a process group
    that spans the pipeline stages; ranks with different group lists would issue mismatched
    collectives at the first clipped step.
    """
    from megatron.core.optimizer import optimizer as mcore_optimizer

    groups = []
    for pre_process, post_process in ((True, False), (False, True)):
        monkeypatch.setattr(mcore_optimizer, "SEPARATE_GRAD_NORM_GROUPS", ("mtp",))
        # mHC MTP runs only on the last stage, so these stand-in stages have none.
        model = Ernie55HybridModel(
            config=_config("H-DE", ernie_mtp_shared_last_layer=False),
            hybrid_stack_spec=_cpu_spec(),
            vocab_size=VOCAB,
            max_sequence_length=1024,
            hybrid_layer_pattern="H-DE",
            position_embedding_type="none",
            pg_collection=pg_collection,
            pre_process=pre_process,
            post_process=post_process,
        )
        assert (getattr(model, "output_layer", None) is not None) == post_process
        groups.append(mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS)
    assert groups == [("mtp", MULTIMAX_GRAD_NORM_GROUP)] * 2


@needs_hook
def test_multimax_applies_to_the_main_and_mtp_logits(pg_collection):
    model = _build_model(pg_collection)
    output_layer = model.output_layer
    assert torch.any(output_layer.multimax_ts != 0) and torch.any(output_layer.multimax_ranges != 0)
    calls = []
    output_layer.register_forward_hook(
        lambda module, args, output: calls.append((args[0].detach().clone(), output[0].detach().clone()))
    )
    loss = model(**_batch())
    loss.sum().backward()
    assert len(calls) == 2  # MTP head, then the main head
    for hidden, logits in calls:
        raw = torch.matmul(hidden, output_layer.weight.t())  # the output layer's own product at TP = 1
        expected = ernie55_multimax(raw, output_layer.multimax_ranges, output_layer.multimax_ts)
        assert logits.dtype == torch.bfloat16 and torch.equal(logits, expected)
        assert not torch.equal(logits, raw)
    for name in ("multimax_ranges", "multimax_ts"):
        grad = output_layer.get_parameter(name).grad
        assert grad is not None and grad.dtype == torch.bfloat16 and torch.any(grad != 0), name


@needs_hook
@pytest.mark.parametrize("fused", [True, False], ids=["fused-ops", "eager"])
def test_fused_ops_reach_the_models_mlps_and_output_layer(pg_collection, monkeypatch, fused):
    """``ernie_fused_ops`` reaches every Megatron-Core MLP of a SiTU-GLU model and its multimax head.

    The kernels are replaced by recorders that compute the eager result, and the device check passes every
    tensor, so the CPU model takes every branch a CUDA model takes. The dense ``-`` MLP, the shared experts
    (``SharedExpertMLP.forward`` is ``MLP.forward``) and the SequentialMLP experts all call the SiTU-GLU of
    ``megatron.core.transformer.mlp``; their fc1 widths tell them apart. The multimax head runs for the main and
    the MTP logits. Without the flag nothing reaches a kernel. The grouped experts need Transformer Engine
    (``test_ernie55_fused_dispatch_gpu.py``), and at CP = 1 no RoPE goes through the context-parallel function.
    """
    dispatch = isolate_fused_dispatch(monkeypatch)
    situ = CallRecorder(lambda x, beta, linear_beta: activations.situ_glu(x, beta, linear_beta, 0.0))
    multimax = CallRecorder(lambda logits, ranges, scales: ernie55_multimax(logits, ranges, scales))
    monkeypatch.setattr(ernie55_fused_ops, "fused_situ_glu", situ)
    monkeypatch.setattr(ernie55_fused_ops, "fused_multimax", multimax)
    model = _build_model(
        pg_collection,
        activation_func_tanh_clamp_scale=1.5,
        activation_func_tanh_clamp_scale_linear=0.5,
        ernie_fused_ops=fused,
    )
    assert model.output_layer.ernie55_fused_multimax is fused
    # The model's SiTU-GLU scales, which the situ_glu dispatchers run the kernel for.
    assert dispatch._STATE.situ_glu_scales == ({(1.5, 0.5)} if fused else set())
    cp_utils = __import__(dispatch.CP_UTILS_MODULE, fromlist=["apply_thd_cp_local_rope_unfused"])
    assert hasattr(cp_utils.apply_thd_cp_local_rope_unfused, "ernie55_unfused") is fused

    monkeypatch.setattr(dispatch, "fused_kernels_support", lambda tensor: True)
    loss = model(**_batch())
    assert torch.isfinite(loss).all()
    calls = dict(dispatch.FUSED_OP_CALLS)
    if not fused:
        assert not calls and not situ.calls and not multimax.calls
        return
    # The main and the MTP head.
    assert calls == {"multimax@output_layer": 2, "situ_glu@megatron.core.transformer.mlp": len(situ.calls)}
    assert len(multimax.calls) == 2
    # fc1 widths: the dense MLP 2 x 24 (once), the shared expert 2 x 12 (the decoder's and the MTP depth's MoE
    # layer), a routed expert 2 x 8 (each expert that received tokens).
    widths = collections.Counter(args[0].shape[-1] for args, _ in situ.calls)
    assert set(widths) == {48, 24, 16} and widths[48] == 1 and widths[24] == 2
