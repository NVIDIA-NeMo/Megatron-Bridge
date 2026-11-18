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

"""CPU tests of the ERNIE 5.5 H layer against an independent fp64 re-derivation of its math.

The oracle in this file is written from the ERNIE 5.5 H-layer specification, not from the Bridge or
Megatron-Core code. For a layer input ``x`` and a query row at position ``p`` of its document:

* ``c_q = RMSNorm_q(x W_qa^T)``; per head ``q = rms(c_q W_qb^T)`` (weightless, every channel);
  ``kv = RMSNorm_kv(x W_kv^T)``, one latent head that is both key and value.
* RoPE with base ``csa_compress_rotary_base`` on the last ``ROPE`` channels of ``q`` and ``kv`` (plain
  RoPE by default; with ``ernie_hca_rope_type="yarn"`` the YaRN frequencies replace ``inv_k``, without
  mscale): channel pairs ``(2k, 2k + 1)`` rotate by ``p * inv_k``, where the table is the spec's fp32 table
  ``inv_k = fp32(1 / base^(2k / ROPE))``, ``angle = fp32(p * inv_k)``.
* Compressor: block ``j`` of a document (128 complete tokens ``t_i``) pools
  ``m_j = sum_i softmax_i(x_t W_gate^T + ape_i) * (x_t W_wkv^T)`` per channel, then
  ``RMSNorm(m_j)`` and RoPE at position ``128 j``. Tail tokens and documents shorter than 128
  have no block.
* Each head attends to ``{max(0, p - W + 1), ..., p}`` and every block ``j < floor((p + 1) / 128)``
  of its own document with ``softmax(q . k / sqrt(D))`` and a per-head sink in the denominator
  only; the output is rotated back by ``-p * inv_k``.
* VHA ``o'_k = sum_h (I + V U^T)[k, h] o_h``, grouped ``wo_a`` over consecutive heads, the gate
  ``z * sigmoid(c_q W_g^T)`` and ``y = z W_o^T``.

Everything runs on CPU with Transformer-Engine-free stand-ins for the TE modules of
``ernie55_hybrid_stack_spec()``. The THD and SBHD layouts run Megatron-Core dev's PyTorch reference
path (``dsa_kernel_backend="none"``).

Besides the oracle, the tests pin the config envelope requirement by requirement (and check that
the D-layer validator enforces the requirements both layers share), and build whole Megatron-Core
``HybridStack`` decoders and MTP stacks from the spec to check that ``H`` and ``D`` symbols become
the ERNIE layers.

Building an H layer needs Megatron-Core's ``DSv4HybridAttention._post_o_group_proj`` hook (dev f78809f6d or
later); without it those tests skip, and the spec, RoPE-table, sink
and config-envelope tests still run.
"""

import copy
import dataclasses
import functools
import logging
import math
import re
import zlib
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.experimental_attention_variant import csa
from megatron.core.transformer.experimental_attention_variant import deepseek_v4_hybrid_attention as dsv4
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    ERNIE55_ATTN_SINK_INIT_STD,
    ERNIE55_HCA_COMPRESS_RATIO,
    Ernie55HCACoreAttention,
    Ernie55HCASelfAttention,
    Ernie55HCASelfAttentionSubmodules,
    Ernie55LatentMQASelfAttention,
    check_ernie55_attention_layers,
    context_parallel,
    downcasting_norm,
    ernie55_hybrid_stack_spec,
    ernie55_rmsnorm,
    hca_attention,
    ops,
    validate_ernie55_config,
    validate_ernie55_hca_config,
)
from tests.unit_tests.models.ernie.ernie55_test_utils import cpu_current_device


logger = logging.getLogger(__name__)

pytestmark = pytest.mark.unit

# H-layer geometry (a scaled-down ERNIE 5.5 H layer: heads x HEAD_DIM latent, ROPE positional channels).
HIDDEN = 64
HEADS = 4
Q_LORA = 32
HEAD_DIM = 16  # v_head_dim: query, key and value width per head (512 in ERNIE 5.5)
ROPE = 8  # qk_pos_emb_head_dim
NOPE = HEAD_DIM - ROPE
GROUPS = 2  # output_projection_groups
O_LORA = 8  # output_projection_lora_rank
VHA_RANK = 2
WINDOW = 5  # csa_window_size
RATIO = ERNIE55_HCA_COMPRESS_RATIO
COMPRESS_ROPE_BASE = 1000.0  # csa_compress_rotary_base: small, so every positional channel rotates
NUM_LAYERS = 2
EPS = 1e-6
SCALE = HEAD_DIM**-0.5

# fp32 module vs fp64 oracle, as a normalized max error (max |diff| / max |expected|).
TOL = 1e-5
# Same comparison with bf16 parameters and activations: bf16 rounding of every intermediate (about 1e-2).
BF16_TOL = 3e-2

# True only on Megatron-Core builds whose packed THD input runs solely through the fused path; such builds
# lack the DSv4 hook, so the @needs_hook tests never take the branches guarded by this flag.
_MAIN_PACKED_THD = hasattr(csa.CompressedSparseAttention, "_forward_packed")
# Megatron-Core dev has the ``hca_layer`` hybrid slot (it takes precedence for ratio-128 layers).
_HAS_HCA_SLOT = "hca_layer" in {
    spec_field.name for spec_field in dataclasses.fields(ernie55_hybrid_stack_spec().submodules)
}
# H layers need the DSv4 hook (dev f78809f6d or later); Bridge's own pins lack it.
_HAS_HOOK = callable(getattr(dsv4.DSv4HybridAttention, "_post_o_group_proj", None))
needs_hook = pytest.mark.skipif(
    not _HAS_HOOK,
    reason="needs Megatron-Core's DSv4HybridAttention._post_o_group_proj hook (dev f78809f6d+)",
)


# ---------------------------------------------------------------------------------------------
# Runtime fixtures and Transformer-Engine-free stand-ins
# ---------------------------------------------------------------------------------------------


def _eager_q_rms_norm(q: torch.Tensor, eps: float) -> torch.Tensor:
    """Megatron-Core's weightless per-head query RMS norm without ``torch.compile`` (same expression)."""
    return q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)


class _DuplicatedLinear(torch.nn.Module):
    """Stand-in for ``TELinear(parallel_mode="duplicated")``: ``y = x W^T``, returns ``(y, None)``."""

    def __init__(self, input_size: int, output_size: int, *, config, init_method, bias: bool = False, **kwargs):
        super().__init__()
        assert not bias and kwargs.get("parallel_mode", "duplicated") == "duplicated"
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        if config.perform_initialization:
            init_method(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None

    def backward_dw(self) -> None:
        pass


class _RMSNorm(torch.nn.Module):
    """Stand-in for ``TENorm`` with ``normalization="RMSNorm"``: fp32 statistics, ``params_dtype`` weight."""

    def __init__(self, config, hidden_size: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        assert config.normalization == "RMSNorm"
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(x.dtype, torch.float32)
        x_wide = x.to(dtype)
        normed = x_wide * torch.rsqrt(x_wide.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.to(dtype)).to(x.dtype)


def _te_stand_ins() -> list[tuple[object, type]]:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelLinear,
    )

    return [
        (TELinear, _DuplicatedLinear),
        (TEColumnParallelLinear, ColumnParallelLinear),
        (TERowParallelLinear, RowParallelLinear),
        (TENorm, _RMSNorm),
    ]


def _substitute(node: object, mapping: list[tuple[object, type]], memo: dict[int, object] | None = None) -> object:
    """Copy a spec tree, replacing the Transformer Engine leaves by their CPU stand-ins."""
    memo = {} if memo is None else memo
    if id(node) in memo:
        return memo[id(node)]
    for original, replacement in mapping:
        if node is original:
            return replacement
    if isinstance(node, (list, tuple)):
        result = type(node)(_substitute(item, mapping, memo) for item in node)
    elif isinstance(node, dict):
        result = {key: _substitute(value, mapping, memo) for key, value in node.items()}
    elif isinstance(node, functools.partial):
        result = functools.partial(
            _substitute(node.func, mapping, memo),
            *_substitute(node.args, mapping, memo),
            **_substitute(node.keywords, mapping, memo),
        )
    elif dataclasses.is_dataclass(node) and not isinstance(node, type):
        result = copy.copy(node)
        for spec_field in dataclasses.fields(node):
            object.__setattr__(result, spec_field.name, _substitute(getattr(node, spec_field.name), mapping, memo))
    else:
        return node
    memo[id(node)] = result
    return result


@pytest.fixture(scope="module")
def pg_collection(tmp_path_factory):
    """Single-rank process groups plus the CPU stand-ins Megatron-Core's DSv4 attention needs."""
    created_group = not torch.distributed.is_initialized()
    if created_group:
        store = tmp_path_factory.mktemp("ernie55_hca_gloo") / "store"
        torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    created_model_parallel = not parallel_state.model_parallel_is_initialized()
    if created_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            # RoPE tables and the compressor APE are allocated on torch.cuda.current_device(); these tests
            # run on CPU even when a GPU is present (torch's own callers still see the GPU).
            patch.setattr(torch.cuda, "current_device", cpu_current_device)
            # DSv4 accepts only TELinear for the q down-projection; without TE that name is None.
            patch.setattr(dsv4, "TELinear", _DuplicatedLinear)
            # The weightless query RMS norm is torch.compile'd; run the same expression eagerly.
            patch.setattr(dsv4, "_q_rms_norm", _eager_q_rms_norm)
            yield ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    finally:
        if created_model_parallel:
            parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()


def _config(**overrides) -> Ernie55TransformerConfig:
    """Finalized DSv4-hybrid stack config whose H layers have the geometry above."""
    values = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=Q_LORA,
        v_head_dim=HEAD_DIM,
        qk_pos_emb_head_dim=ROPE,
        output_projection_groups=GROUPS,
        output_projection_lora_rank=O_LORA,
        csa_compress_ratios=[RATIO] * NUM_LAYERS,
        csa_window_size=WINDOW,
        csa_compress_rotary_base=COMPRESS_ROPE_BASE,
        ernie_vha_rank=VHA_RANK,
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        # Stack-level YaRN settings that must not reach H layers (they use plain RoPE).
        rope_type="yarn",
        rotary_scaling_factor=40.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        experimental_attention_variant="dsv4_hybrid",
        dsa_kernel_backend="none",
        # D-layer indexer settings the shared ERNIE 5.5 validation requires.
        dsa_indexer_n_heads=4,
        dsa_indexer_head_dim=16,
        dsa_indexer_topk=6,
        dsa_indexer_k_norm_fp32=True,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.float32,
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _set_config_field(config: Ernie55TransformerConfig, name: str, value: object) -> None:
    if not hasattr(config, name):
        raise ValueError(f"{type(config).__name__} has no field {name!r}")
    setattr(config, name, value)


def _with_cudnn_backend(config: Ernie55TransformerConfig) -> Ernie55TransformerConfig:
    """Copy of a finalized config on the fused backend (which cannot be finalized without cuDNN)."""
    fused = copy.copy(config)
    _set_config_field(fused, "dsa_kernel_backend", "cudnn")
    return fused


def _attention_spec(slot: str = "csa_qk_layernorm_layer"):
    """The ERNIE H attention spec of ``ernie55_hybrid_stack_spec()`` with CPU stand-ins for TE."""
    layer = getattr(ernie55_hybrid_stack_spec().submodules, slot)
    return _substitute(layer.submodules.self_attention, _te_stand_ins())


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    """Draw every parameter at O(1) scale, including the sink, APE and both VHA factors.

    Each parameter has its own generator keyed by its name, so the values do not depend on the
    order in which Megatron-Core registers parameters.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            generator = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float64)
            if name.endswith(("attn_sink", "compressor.ape")):
                values = noise
            elif "vha_postmix" in name:
                values = 0.5 * noise
            elif "norm" in name:
                values = 1.0 + 0.2 * noise
            else:
                values = noise / math.sqrt(param.shape[-1])
            param.copy_(values.to(param.dtype))


def _build_attention(
    config: Ernie55TransformerConfig,
    pg_collection: ProcessGroupCollection,
    *,
    seed: int | None,
    compress_ratio: int = RATIO,
    spec=None,
    layer_number: int = 1,
) -> Ernie55HCASelfAttention:
    attention = build_module(
        _attention_spec() if spec is None else spec,
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
        compress_ratio=compress_ratio,
    )
    if seed is not None:
        _randomize_(attention, seed)
    return attention.train()


def _install_main_packed_stand_ins(monkeypatch) -> list[dict]:
    """Replace the packed (fused) CSA kernels with Megatron-Core's unfused sparse attention on CPU.

    Used only when ``_MAIN_PACKED_THD`` is true, which the supported Megatron-Core never is. Returns a
    log of the kernel calls.
    """
    calls = []

    def sparse_attention(query, kv, attn_sink, topk_idxs, softmax_scale, topk_length=None, **kwargs):
        # query [rows, heads, dim], kv [kv_rows, dim], topk_idxs [rows, width] rows of kv (-1 = none):
        # one SBHD sample of the unfused attention; returns [rows, heads * dim] like the kernel.
        calls.append({"topk_idxs": topk_idxs.clone(), "topk_length": topk_length})
        output = csa.unfused_compressed_sparse_attn(
            query.unsqueeze(1), kv.unsqueeze(1), attn_sink, topk_idxs.unsqueeze(0), softmax_scale
        )
        return output.squeeze(1)

    monkeypatch.setattr(csa, "packed_csa_sparse_attn", sparse_attention)
    monkeypatch.setattr(csa, "get_flash_mla_topk_alignment", lambda: 64)
    return calls


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


@dataclass
class _Document:
    batch: int
    start: int
    end: int  # exclusive; padded end for padded THD packs
    real_end: int  # exclusive; rows in [real_end, end) are padding


def _cumsum(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)


def _inputs(layout: dict, seed: int) -> tuple[torch.Tensor, PackedSeqParams | None, list[_Document]]:
    generator = torch.Generator().manual_seed(seed)
    if "doc_lens" in layout:
        doc_lens = layout["doc_lens"]
        padded_lens = layout.get("padded_lens") or doc_lens
        cu_seqlens, cu_seqlens_padded = _cumsum(doc_lens), _cumsum(padded_lens)
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            cu_seqlens_q_padded=cu_seqlens_padded if "padded_lens" in layout else None,
            cu_seqlens_kv_padded=cu_seqlens_padded if "padded_lens" in layout else None,
            max_seqlen_q=max(padded_lens),
            max_seqlen_kv=max(padded_lens),
        )
        starts = cu_seqlens_padded.tolist()
        documents = [_Document(0, starts[i], starts[i + 1], starts[i] + doc_lens[i]) for i in range(len(doc_lens))]
        num_rows, batch = starts[-1], 1
    else:
        num_rows, batch, packed_seq_params = layout["seqlen"], layout["batch"], None
        documents = [_Document(b, 0, num_rows, num_rows) for b in range(batch)]
    x = torch.randn(num_rows, batch, HIDDEN, generator=generator, dtype=torch.float64)
    return x.float().requires_grad_(True), packed_seq_params, documents


def _real_rows(documents: list[_Document], num_rows: int, batch: int) -> torch.Tensor:
    mask = torch.zeros(num_rows, batch, dtype=torch.bool)
    for document in documents:
        mask[document.start : document.real_end, document.batch] = True
    return mask


# ---------------------------------------------------------------------------------------------
# fp64 oracle, written from the ERNIE 5.5 H-layer specification
# ---------------------------------------------------------------------------------------------


def _leaves(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """fp64 leaf copies of the module parameters, by name."""
    return {name: param.detach().double().requires_grad_(True) for name, param in module.named_parameters()}


def _rms(x: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return normed if weight is None else normed * weight


def _rope_angles(positions: torch.Tensor, base: float, inv_freq: torch.Tensor | None = None) -> torch.Tensor:
    """``[n, ROPE / 2]`` angles from the spec's fp32 table: ``fp32(p * fp32(1 / base^(2k / ROPE)))``.

    ``inv_freq`` (fp32, ``[ROPE / 2]``) replaces the plain frequencies, for YaRN H layers.
    """
    if inv_freq is None:
        inv_freq = 1.0 / (base ** (torch.arange(0, ROPE, 2, dtype=torch.float32) / ROPE))
    return (positions.to(torch.float32)[:, None] * inv_freq[None, :]).double()


def _rotate(x: torch.Tensor, angles: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate the last ``ROPE`` channels of ``x`` pairwise: ``(2k, 2k + 1)`` as a complex number."""
    content, positional = x[..., :NOPE], x[..., NOPE:]
    pairs = torch.view_as_complex(positional.reshape(*positional.shape[:-1], ROPE // 2, 2).contiguous())
    angles = angles.view(angles.size(0), *([1] * (pairs.dim() - 2)), angles.size(-1))
    rotation = torch.polar(torch.ones_like(angles), -angles if inverse else angles)
    rotated = torch.view_as_real(pairs * rotation).reshape(positional.shape)
    return torch.cat([content, rotated], dim=-1)


@dataclass
class _Oracle:
    output: torch.Tensor  # [rows, batch, HIDDEN] fp64
    x: torch.Tensor  # fp64 leaf
    params: dict[str, torch.Tensor]  # fp64 leaves by parameter name


def _oracle_attention(
    attention: torch.nn.Module, x: torch.Tensor, documents: list[_Document], inv_freq: torch.Tensor | None = None
) -> _Oracle:
    params = _leaves(attention)
    x64 = x.detach().double().requires_grad_(True)
    rows, batch = x64.shape[:2]

    c_q = _rms(x64 @ params["linear_q_down_proj.weight"].T, params["q_layernorm.weight"])
    q = _rms((c_q @ params["linear_q_up_proj.weight"].T).view(rows, batch, HEADS, HEAD_DIM))
    kv = _rms(x64 @ params["linear_kv_proj.weight"].T, params["kv_layernorm.weight"])
    sink = params["core_attention.attn_sink"].view(1, HEADS, 1)
    mix = torch.eye(HEADS, dtype=torch.float64) + params["core_attention.vha_postmix_v"] @ (
        params["core_attention.vha_postmix_u"].T
    )
    wo_a = params["linear_o_group_proj"].view(GROUPS, O_LORA, -1)

    output = torch.zeros(rows, batch, HIDDEN, dtype=torch.float64)
    for document in documents:
        sl = slice(document.start, document.end)
        length = document.end - document.start
        positions = torch.arange(length)
        angles = _rope_angles(positions, COMPRESS_ROPE_BASE, inv_freq)
        q_doc = _rotate(q[sl, document.batch], angles)  # [L, H, D]
        keys = _rotate(kv[sl, document.batch], angles)  # [L, D]

        num_blocks = length // RATIO
        if num_blocks:
            blocks = x64[document.start : document.start + num_blocks * RATIO, document.batch]
            blocks = blocks.view(num_blocks, RATIO, HIDDEN)
            values = blocks @ params["core_attention.compressor.linear_wkv.weight"].T
            scores = blocks @ params["core_attention.compressor.linear_wgate.weight"].T
            weights = torch.softmax(scores + params["core_attention.compressor.ape"], dim=1)
            pooled = _rms((values * weights).sum(dim=1), params["core_attention.compressor.norm.weight"])
            pooled = _rotate(pooled, _rope_angles(torch.arange(num_blocks) * RATIO, COMPRESS_ROPE_BASE, inv_freq))
            keys = torch.cat([keys, pooled], dim=0)

        row = positions[:, None]
        window = (positions[None, :] <= row) & (positions[None, :] > row - WINDOW)
        compressed = torch.arange(num_blocks)[None, :] < (row + 1) // RATIO
        allowed = torch.cat([window, compressed], dim=1)  # [L, L + blocks]

        logits = torch.einsum("lhd,kd->lhk", q_doc, keys) * SCALE
        logits = logits.masked_fill(~allowed[:, None, :], float("-inf"))
        peak = torch.maximum(logits.amax(dim=-1, keepdim=True), sink)
        exp_logits = torch.exp(logits - peak)
        probs = exp_logits / (exp_logits.sum(dim=-1, keepdim=True) + torch.exp(sink - peak))
        heads = _rotate(torch.einsum("lhk,kd->lhd", probs, keys), angles, inverse=True)

        heads = torch.einsum("kh,lhd->lkd", mix, heads)
        grouped = torch.einsum("lgd,grd->lgr", heads.reshape(length, GROUPS, -1), wo_a).reshape(length, -1)
        gated = grouped * torch.sigmoid(c_q[sl, document.batch] @ params["linear_output_gate.weight"].T)
        output[sl, document.batch] = gated @ params["linear_proj.weight"].T
    return _Oracle(output, x64, params)


def _normalized_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    scale = expected.abs().max().clamp_min(1e-30)
    return float((actual.double() - expected.double()).abs().max() / scale)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, what: str) -> None:
    assert actual is not None, f"{what}: missing"
    assert actual.shape == expected.shape, f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    error = _normalized_error(actual, expected)
    assert error <= tol, f"{what}: normalized max error {error:.3e} > {tol:.1e}"


def _run_against_oracle(
    attention, x, packed_seq_params, documents, *, tol: float, seed: int, inv_freq: torch.Tensor | None = None
) -> None:
    """Forward, then backward with a random output gradient (zero on padding rows), against the oracle."""
    output, bias = attention(x, None, packed_seq_params=packed_seq_params)
    assert bias is None
    oracle = _oracle_attention(attention, x, documents, inv_freq)
    real = _real_rows(documents, *x.shape[:2]).unsqueeze(-1)
    _assert_close(output * real, oracle.output * real, tol, "output")

    grad_output = torch.randn(output.shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    grad_output = grad_output * real
    output.backward(grad_output.to(output.dtype))
    (oracle.output * grad_output).sum().backward()
    _assert_close(x.grad, oracle.x.grad, tol, "grad of x")
    for name, param in attention.named_parameters():
        expected = oracle.params[name].grad
        if expected is None:
            # Unused by this input (the compressor when no document has a complete block).
            assert param.grad is None or not param.grad.any(), f"grad of {name}: the oracle does not use it"
            continue
        _assert_close(param.grad, expected, tol, f"grad of {name}")


# ---------------------------------------------------------------------------------------------
# Full H layer against the oracle
# ---------------------------------------------------------------------------------------------

_LAYOUTS = {
    # Documents with two blocks and a tail, without blocks (< 128) and with one block and one tail row.
    "thd": dict(doc_lens=[300, 50, 129]),
    # No document reaches 128 tokens: no compressed entries anywhere in the pack.
    "thd-short-docs": dict(doc_lens=[60, 127, 7]),
    # Padded documents: rows past a document's real length are padding, outside the comparison. Real
    # rows 255..299 of the first document see its second block, which is rotated at position 128.
    "thd-padded": dict(doc_lens=[300, 40, 140], padded_lens=[304, 48, 144]),
    "sbhd-b2": dict(seqlen=260, batch=2),
}


@needs_hook
class TestOracle:
    @pytest.mark.parametrize("layout", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
    def test_matches_fp64_oracle(self, pg_collection, monkeypatch, layout):
        """Forward, input gradient and every parameter gradient match the fp64 oracle."""
        config = _config()
        if "doc_lens" in layout and _MAIN_PACKED_THD:
            config = _with_cudnn_backend(config)
            calls = _install_main_packed_stand_ins(monkeypatch)
        attention = _build_attention(config, pg_collection, seed=11)
        x, packed_seq_params, documents = _inputs(layout, seed=12)
        _run_against_oracle(attention, x, packed_seq_params, documents, tol=TOL, seed=13)
        if "doc_lens" in layout and _MAIN_PACKED_THD:
            assert len(calls) == 1, "main's packed THD path did not reach its sparse-attention kernel"

    @pytest.mark.parametrize(
        ("vha_bf16_mix", "output_gate_fp32"), [(True, False), (False, True)], ids=["bf16-contracts", "fp32-contracts"]
    )
    def test_precision_switches_keep_the_math(self, pg_collection, monkeypatch, vha_bf16_mix, output_gate_fp32):
        """Both VHA and gate precision contracts compute the same function (exact in fp32)."""
        config = _config(ernie_vha_bf16_mix=vha_bf16_mix, ernie_output_gate_fp32=output_gate_fp32)
        attention = _build_attention(config, pg_collection, seed=21)
        x, packed_seq_params, documents = _inputs(_LAYOUTS["sbhd-b2"], seed=22)
        _run_against_oracle(attention, x, packed_seq_params, documents, tol=TOL, seed=23)

    @pytest.mark.parametrize("layout", ["thd", "sbhd-b2"])
    def test_yarn_matches_fp64_oracle(self, pg_collection, monkeypatch, layout):
        """YaRN H layers rotate q, kv, the compressed keys and the output with the YaRN table, without mscale."""
        config = _config(**_YARN)
        if "doc_lens" in _LAYOUTS[layout] and _MAIN_PACKED_THD:
            config = _with_cudnn_backend(config)
            _install_main_packed_stand_ins(monkeypatch)
        attention = _build_attention(config, pg_collection, seed=25)
        x, packed_seq_params, documents = _inputs(_LAYOUTS[layout], seed=26)
        inv_freq = _yarn_inv_freq(**_YARN_ARGS).float()
        _run_against_oracle(attention, x, packed_seq_params, documents, tol=TOL, seed=27, inv_freq=inv_freq)

    @pytest.mark.parametrize("layout", ["thd", "sbhd-b2"])
    def test_bf16_forward_matches_oracle(self, pg_collection, monkeypatch, layout):
        """bf16 parameters and activations stay within bf16 rounding of the fp64 oracle."""
        config = _config(params_dtype=torch.bfloat16)
        if "doc_lens" in _LAYOUTS[layout] and _MAIN_PACKED_THD:
            config = _with_cudnn_backend(config)
            _install_main_packed_stand_ins(monkeypatch)
        attention = _build_attention(config, pg_collection, seed=31)
        x, packed_seq_params, documents = _inputs(_LAYOUTS[layout], seed=32)
        x = x.detach().bfloat16()
        with torch.no_grad():
            output, _ = attention(x, None, packed_seq_params=packed_seq_params)
        oracle = _oracle_attention(attention, x, documents)
        error = _normalized_error(output, oracle.output)
        logger.info("bf16 H layer vs fp64 oracle (%s): normalized max error %.3e", layout, error)
        assert output.dtype == torch.bfloat16
        assert error <= BF16_TOL, f"bf16 output: normalized max error {error:.3e} > {BF16_TOL:.1e}"


# ---------------------------------------------------------------------------------------------
# Output gate, VHA and attention sink
# ---------------------------------------------------------------------------------------------


def _saved_storages_during(fn) -> tuple[object, set[int]]:
    """Run ``fn`` and return its result plus the storage pointers autograd saved for backward."""
    saved: set[int] = set()

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.add(tensor.untyped_storage().data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = fn()
    return result, saved


@needs_hook
class TestOutputGate:
    @pytest.mark.parametrize("gate_fp32", [False, True], ids=["bf16-gate", "fp32-gate"])
    def test_gate_precision_contracts(self, pg_collection, gate_fp32):
        """bf16: sigmoid and product in bf16. fp32 (HF sigmoid_gate_fusion): fp32 math, one rounding."""
        config = _config(params_dtype=torch.bfloat16, ernie_output_gate_fp32=gate_fp32)
        attention = _build_attention(config, pg_collection, seed=41)
        generator = torch.Generator().manual_seed(42)
        attn_out = (4 * torch.randn(9, 2, GROUPS * O_LORA, generator=generator)).bfloat16()
        q_latent = (4 * torch.randn(9, 2, Q_LORA, generator=generator)).bfloat16()

        gated = attention._post_o_group_proj(attn_out, q_latent)
        logits = F.linear(q_latent, attention.linear_output_gate.weight)
        bf16_contract = attn_out * torch.sigmoid(logits)
        fp32_contract = (attn_out.float() * torch.sigmoid(logits.float())).bfloat16()
        assert gated.dtype == torch.bfloat16
        assert torch.equal(gated, fp32_contract if gate_fp32 else bf16_contract)
        assert not torch.equal(bf16_contract, fp32_contract), "the inputs do not separate the two contracts"

    def test_hook_result_is_not_saved_for_backward(self, pg_collection):
        """The attn_proj offload group force-releases the hook's result, so backward must not keep it."""
        attention = _build_attention(_config(), pg_collection, seed=43)
        generator = torch.Generator().manual_seed(44)
        attn_out = torch.randn(5, 2, GROUPS * O_LORA, generator=generator).requires_grad_(True)
        q_latent = torch.randn(5, 2, Q_LORA, generator=generator).requires_grad_(True)
        gated, saved = _saved_storages_during(lambda: attention._post_o_group_proj(attn_out, q_latent))
        assert saved, "the gate saved nothing for backward; the check would be vacuous"
        assert gated.untyped_storage().data_ptr() not in saved
        assert gated.untyped_storage().data_ptr() != attn_out.untyped_storage().data_ptr()

    def test_gate_reads_the_q_latent(self, pg_collection):
        """The gate source is the q_layernorm output (the hook's q_compressed), not its input or x."""
        attention = _build_attention(_config(), pg_collection, seed=45)
        captured = {}
        attention.q_layernorm.register_forward_hook(lambda module, args, output: captured.update(q_norm=output))
        original = attention._post_o_group_proj

        def spy(attn_out, q_compressed):
            captured["gate_source"] = q_compressed
            return original(attn_out, q_compressed)

        attention._post_o_group_proj = spy
        x, packed_seq_params, _ = _inputs(_LAYOUTS["sbhd-b2"], seed=46)
        attention(x, None, packed_seq_params=packed_seq_params)
        assert torch.equal(captured["gate_source"], captured["q_norm"])

    def test_backward_dw_reaches_the_gate(self, pg_collection, monkeypatch):
        attention = _build_attention(_config(), pg_collection, seed=47)
        calls = []
        for name, module in attention.named_modules():
            if isinstance(module, (_DuplicatedLinear, ColumnParallelLinear, RowParallelLinear)):
                monkeypatch.setattr(module, "backward_dw", functools.partial(calls.append, name), raising=False)
        attention.backward_dw()
        assert "linear_output_gate" in calls
        assert {"linear_q_down_proj", "linear_proj", "core_attention.compressor.linear_wkv"} <= set(calls)


@needs_hook
class TestVHA:
    @pytest.mark.parametrize("bf16_mix", [True, False], ids=["bf16-mix", "fp32-mix"])
    def test_core_mixes_heads_out_of_place(self, pg_collection, monkeypatch, bf16_mix):
        """The core returns ``vha_postmix(Megatron-Core core output)`` and leaves that output untouched."""
        attention = _build_attention(
            _config(params_dtype=torch.bfloat16, ernie_vha_bf16_mix=bf16_mix), pg_collection, seed=51
        )
        core = attention.core_attention
        captured = {}
        mcore_forward = csa.CompressedSparseAttention.forward

        def spy(self, *args, **kwargs):
            result = mcore_forward(self, *args, **kwargs)
            captured["mcore"] = result
            captured["mcore_copy"] = result.detach().clone()
            return result

        monkeypatch.setattr(csa.CompressedSparseAttention, "forward", spy)
        returned = {}
        core.register_forward_hook(lambda module, args, output: returned.update(output=output))
        x, packed_seq_params, _ = _inputs(_LAYOUTS["sbhd-b2"], seed=52)
        attention(x.detach().bfloat16(), None, packed_seq_params=packed_seq_params)

        mcore_out = captured["mcore"]
        expected = ops.vha_postmix(
            mcore_out.reshape(*mcore_out.shape[:-1], HEADS, HEAD_DIM),
            u=core.vha_postmix_u,
            v=core.vha_postmix_v,
            bf16_mix=bf16_mix,
        ).reshape(mcore_out.shape)
        assert torch.equal(returned["output"], expected)
        assert torch.equal(mcore_out, captured["mcore_copy"]), "VHA modified the core output in place"
        assert returned["output"].untyped_storage().data_ptr() != mcore_out.untyped_storage().data_ptr()
        assert not torch.equal(returned["output"], mcore_out), "random U and V must change the output"

    def test_parameters(self, pg_collection):
        """U and V are fp32 keep_in_fp32 parameters; V starts at zero (identity mix), U from init_method."""
        config = _config(params_dtype=torch.bfloat16)
        core = _build_attention(config, pg_collection, seed=None).core_attention
        for param in (core.vha_postmix_u, core.vha_postmix_v):
            assert param.shape == (HEADS, VHA_RANK)
            assert param.dtype == torch.float32 and getattr(param, "keep_in_fp32", False)
        assert torch.equal(core.vha_postmix_v, torch.zeros(HEADS, VHA_RANK))
        assert bool((core.vha_postmix_u != 0).all())
        uninitialized = _config(perform_initialization=False)
        assert not _build_attention(uninitialized, pg_collection, seed=None).core_attention.vha_postmix_u.any()


class TestAttentionSink:
    def test_seed_matches_reference_draw(self):
        """The all-zero sink is drawn from N(0, 0.1), the reference implementation's seed (pinned as a literal)."""
        assert ERNIE55_ATTN_SINK_INIT_STD == 0.1
        torch.manual_seed(61)
        sink = torch.zeros(4096)
        hca_attention._seed_zero_attention_sink_(sink)
        torch.manual_seed(61)
        assert torch.equal(sink, torch.zeros(4096).normal_(mean=0.0, std=0.1))
        assert sink.std().item() == pytest.approx(0.1, rel=0.1)

    def test_seed_keeps_non_zero_and_meta_sinks(self):
        trained = torch.tensor([0.0, 0.5, -1.0])
        hca_attention._seed_zero_attention_sink_(trained)
        assert torch.equal(trained, torch.tensor([0.0, 0.5, -1.0]))
        hca_attention._seed_zero_attention_sink_(torch.zeros(3, device="meta"))

    @needs_hook
    @pytest.mark.parametrize("perform_initialization", [True, False])
    def test_seeded_at_init_only(self, pg_collection, perform_initialization):
        config = _config(perform_initialization=perform_initialization)
        attention = _build_attention(config, pg_collection, seed=None)
        sink = attention.core_attention.attn_sink
        assert sink.dtype == torch.float32 and getattr(sink, "keep_in_fp32", False)
        if perform_initialization:
            assert bool((sink != 0).all()) and float(sink.abs().max()) < 1.0
        else:
            assert not sink.any()
        # Forward and backward never re-seed, even an all-zero sink.
        with torch.no_grad():
            sink.zero_()
        x, packed_seq_params, _ = _inputs(_LAYOUTS["sbhd-b2"], seed=62)
        output, _ = attention(x, None, packed_seq_params=packed_seq_params)
        output.sum().backward()
        assert not sink.any()

    @needs_hook
    def test_throwaway_core_draws_nothing(self, pg_collection, monkeypatch):
        """Only the ratio-128 core draws U and the sink, so an H layer consumes the init streams once.

        Megatron-Core's ``Attention.__init__`` first builds a window-only core (ratio 0), which
        ``DSv4HybridAttention`` then replaces with the real one.
        """
        ratios, seeded, vha_draws = [], [], []
        core_init = Ernie55HCACoreAttention.__init__

        def recording_core_init(self, *args, **kwargs):
            core_init(self, *args, **kwargs)
            ratios.append(self.compress_ratio)

        seed_sink = hca_attention._seed_zero_attention_sink_

        def recording_seed(sink):
            seeded.append(sink)
            seed_sink(sink)

        config = _config()
        init_method = config.init_method

        def recording_init(tensor):
            if tensor.shape == (HEADS, VHA_RANK):
                vha_draws.append(tensor)
            return init_method(tensor)

        _set_config_field(config, "init_method", recording_init)
        monkeypatch.setattr(Ernie55HCACoreAttention, "__init__", recording_core_init)
        monkeypatch.setattr(hca_attention, "_seed_zero_attention_sink_", recording_seed)
        core = _build_attention(config, pg_collection, seed=None).core_attention
        assert ratios == [0, RATIO], "Megatron-Core no longer builds the throwaway core first; update the docstrings"
        assert len(seeded) == 1 and seeded[0] is core.attn_sink
        assert len(vha_draws) == 1 and vha_draws[0] is core.vha_postmix_u


# ---------------------------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------------------------


def _plain_rope_table(dim: int, base: float, positions: int) -> torch.Tensor:
    """The spec's fp32 RoPE angle table ``[positions, dim]`` (``cat(p * inv, p * inv)``)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    freqs = torch.outer(torch.arange(positions, dtype=torch.float32), inv_freq)
    return torch.cat([freqs, freqs], dim=-1)


# H-layer YaRN at the test geometry: the ramp runs from frequency 1 to 4, so frequencies 2 and 3 are interpolated.
_YARN_ARGS = dict(scaling_factor=16.0, original_max_position_embeddings=2048, beta_fast=32.0, beta_slow=1.0)
_YARN = dict(
    ernie_hca_rope_type="yarn",
    rotary_scaling_factor=_YARN_ARGS["scaling_factor"],
    original_max_position_embeddings=_YARN_ARGS["original_max_position_embeddings"],
    beta_fast=_YARN_ARGS["beta_fast"],
    beta_slow=_YARN_ARGS["beta_slow"],
    mscale=1.0,
    mscale_all_dim=0.0,
)


def _yarn_inv_freq(
    *, scaling_factor: float, original_max_position_embeddings: int, beta_fast: float, beta_slow: float
) -> torch.Tensor:
    """fp64 YaRN frequencies ``[ROPE / 2]`` at base ``COMPRESS_ROPE_BASE`` (the reference implementation's formula)."""

    def correction_dim(rotations: float) -> float:
        return (ROPE * math.log(original_max_position_embeddings / (rotations * 2 * math.pi))) / (
            2 * math.log(COMPRESS_ROPE_BASE)
        )

    low = max(math.floor(correction_dim(beta_fast)), 0)
    high = min(math.ceil(correction_dim(beta_slow)), ROPE - 1)
    high = high + 0.001 if low == high else high
    ramp = ((torch.arange(ROPE // 2, dtype=torch.float64) - low) / (high - low)).clamp(0, 1)
    extrapolated = 1.0 / (COMPRESS_ROPE_BASE ** (torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE))
    return extrapolated / scaling_factor * ramp + extrapolated * (1 - ramp)


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in fp32 units in the last place between same-sign fp32 tensors."""
    return (a.contiguous().view(torch.int32).long() - b.contiguous().view(torch.int32).long()).abs()


class TestRope:
    @needs_hook
    def test_every_rope_holder_shares_one_plain_table(self, pg_collection):
        """The attention, the compressor and (dev) the core read one plain RoPE; no YaRN is left."""
        attention = _build_attention(_config(), pg_collection, seed=None)
        rope = attention.rotary_pos_emb
        assert type(rope) is RotaryEmbedding and not attention._dsv4_uses_yarn_rope
        assert not any(isinstance(module, YarnRotaryEmbedding) for module in attention.modules())
        holders = [module for module in attention.modules() if hasattr(module, "rotary_pos_emb")]
        assert attention.core_attention.compressor in holders
        assert all(module.rotary_pos_emb is rope for module in holders)
        expected = 1.0 / (COMPRESS_ROPE_BASE ** (torch.arange(0, ROPE, 2, dtype=torch.float32) / ROPE))
        assert torch.equal(rope.inv_freq, expected)
        assert torch.equal(rope(64).view(64, ROPE), _plain_rope_table(ROPE, COMPRESS_ROPE_BASE, 64))

    @needs_hook
    def test_yarn_h_layers_keep_one_yarn_table(self, pg_collection):
        """``ernie_hca_rope_type="yarn"``: every holder shares Megatron-Core's YaRN, with the YaRN frequencies."""
        attention = _build_attention(_config(**_YARN), pg_collection, seed=None)
        rope = attention.rotary_pos_emb
        assert type(rope) is YarnRotaryEmbedding and attention._dsv4_uses_yarn_rope
        holders = [module for module in attention.modules() if hasattr(module, "rotary_pos_emb")]
        assert attention.core_attention.compressor in holders
        assert all(module.rotary_pos_emb is rope for module in holders)
        table, mscale = rope(64)
        expected = _yarn_inv_freq(**_YARN_ARGS)
        assert not torch.allclose(expected, _yarn_inv_freq(**(_YARN_ARGS | {"scaling_factor": 1.0})))
        torch.testing.assert_close(table.view(64, ROPE)[1, : ROPE // 2].double().cpu(), expected, rtol=1e-6, atol=0)
        assert mscale > 1.0, "the YaRN concentration factor must exist (and H layers must not apply it)"

    def test_rope_table_against_yarn_factor_one(self, monkeypatch):
        """The H table is bitwise the plain table at the ERNIE 5.5 geometry; report YaRN(1)'s deviation.

        YaRN with ``rotary_scaling_factor = 1`` blends two equal frequency vectors, ``inv * (1 - m) +
        inv * m``, which rounds in the ramp band. This is why H layers swap in plain RoPE.
        """
        # Both embeddings move their frequencies to torch.cuda.current_device() on first use.
        monkeypatch.setattr(torch.cuda, "current_device", cpu_current_device)
        dim, base, positions = 64, 160000.0, 8192
        plain = RotaryEmbedding(dim, rotary_percent=1.0, rotary_base=base, use_cpu_initialization=True)
        table = plain(positions).view(positions, dim)
        reference = _plain_rope_table(dim, base, positions)
        assert torch.equal(table, reference)

        yarn = YarnRotaryEmbedding(
            dim,
            rotary_base=base,
            scaling_factor=1.0,
            mscale=1.0,
            mscale_all_dim=0.0,
            use_cpu_initialization=True,
        )
        yarn_table, _ = yarn(positions)
        yarn_table = yarn_table.view(positions, dim)
        # Row 1 holds 1 * inv_freq exactly.
        inv_ulps = _ulp_distance(yarn_table[1, : dim // 2], reference[1, : dim // 2])
        # The table is cat(angles, angles); count each unique angle once.
        yarn_angles, plain_angles = yarn_table[:, : dim // 2], reference[:, : dim // 2]
        cos_diff = int((yarn_angles.cos().bfloat16() != plain_angles.cos().bfloat16()).sum())
        sin_diff = int((yarn_angles.sin().bfloat16() != plain_angles.sin().bfloat16()).sum())
        logger.info(
            "YaRN(factor=1) vs plain RoPE (dim %d, base %.0f, %d positions): %d of %d inv_freq entries differ "
            "(max %d ulp); bf16 cos differs in %d and sin in %d of the %d unique entries of each table",
            dim,
            base,
            positions,
            int((inv_ulps > 0).sum()),
            dim // 2,
            int(inv_ulps.max()),
            cos_diff,
            sin_diff,
            positions * dim // 2,
        )
        assert int(inv_ulps.max()) <= 1, "YaRN(factor=1) differs from plain RoPE by more than rounding"


# ---------------------------------------------------------------------------------------------
# Construction guards, config validation and the build-time layer check
# ---------------------------------------------------------------------------------------------

# Requirements that both validate_ernie55_hca_config (H) and validate_ernie55_config (D) enforce: config
# changes that each violate exactly one of them, and the message both validators raise for it.
_SHARED_ENVELOPE = [
    pytest.param({"multi_latent_attention": False}, "multi_latent_attention must be True", id="mla"),
    pytest.param({"tensor_model_parallel_size": 2}, "tensor_model_parallel_size must be 1", id="tp"),
    pytest.param(
        {"context_parallel_size": 2, "dynamic_context_parallel": True},
        "dynamic (hybrid) context parallelism is not supported",
        id="dynamic-cp",
    ),
    pytest.param({"bf16": False}, "only BF16 is supported", id="no-bf16"),
    pytest.param({"fp16": True}, "only BF16 is supported", id="fp16"),
    pytest.param({"fp8": "hybrid"}, "FP8 and FP4 are not supported", id="fp8"),
    pytest.param({"fp4": "e2m1"}, "FP8 and FP4 are not supported", id="fp4"),
    pytest.param({"quant_recipe": object()}, "quant_recipe and Kitchen quantization", id="quant-recipe"),
    pytest.param({"use_kitchen": True}, "quant_recipe and Kitchen quantization", id="kitchen"),
    pytest.param({"use_kitchen_attention": True}, "quant_recipe and Kitchen quantization", id="kitchen-attention"),
    pytest.param({"cuda_graph_impl": "local"}, "CUDA graphs are not supported", id="cuda-graph-impl"),
    pytest.param({"enable_cuda_graph": True}, "CUDA graphs are not supported", id="enable-cuda-graph"),
    pytest.param({"delay_wgrad_compute": True}, "delay_wgrad_compute is not supported", id="delay-wgrad"),
    pytest.param({"add_bias_linear": True}, "add_bias_linear must be False", id="bias"),
    pytest.param({"normalization": "LayerNorm"}, "normalization must be 'RMSNorm'", id="layernorm"),
    pytest.param({"ernie_vha_rank": 0}, "ernie_vha_rank must be a positive integer", id="vha-rank"),
    pytest.param({"dsa_kernel_backend": "tilelang"}, "dsa_kernel_backend must be one of", id="tilelang"),
    pytest.param(
        # An unfused attention backend would silently run the PyTorch reference path.
        {"dsa_kernel_backend": "cudnn", "attention_backend": AttnBackend.unfused},
        "dsa_kernel_backend='cudnn' needs a fused attention_backend",
        id="cudnn-unfused",
    ),
    pytest.param({"csa_window_size": 0}, "csa_window_size must be at least 1", id="window"),
]
# Requirements of H layers only.
_H_ONLY_ENVELOPE = [
    pytest.param({"experimental_attention_variant": "dsa"}, "experimental_attention_variant must be", id="variant"),
    pytest.param({"q_lora_rank": None}, "q_lora_rank must be set", id="q-lora-rank"),
    pytest.param({"rotary_percent": 0.5}, "rotary_percent must be 1.0", id="rotary-percent"),
]


def _violating_config(overrides: dict) -> Ernie55TransformerConfig:
    """A copy of the finalized H-layer config with ``overrides`` applied (not re-finalized)."""
    config = copy.copy(_config())
    for name, value in overrides.items():
        _set_config_field(config, name, value)
    return config


class TestGuards:
    def test_missing_hook_raises(self, pg_collection, monkeypatch):
        if _HAS_HOOK:
            monkeypatch.delattr(dsv4.DSv4HybridAttention, "_post_o_group_proj")
        with pytest.raises(RuntimeError, match="_post_o_group_proj"):
            _build_attention(_config(), pg_collection, seed=None)

    @needs_hook
    @pytest.mark.parametrize("compress_ratio", [0, 4])
    def test_only_ratio_128(self, pg_collection, compress_ratio):
        with pytest.raises(ValueError, match="compress_ratio=128"):
            _build_attention(_config(), pg_collection, seed=None, compress_ratio=compress_ratio)

    @needs_hook
    def test_core_must_be_the_ernie_core(self, pg_collection):
        spec = _attention_spec()
        spec.submodules.core_attention = functools.partial(
            csa.CompressedSparseAttention, submodules=spec.submodules.core_attention.keywords["submodules"]
        )
        with pytest.raises(TypeError, match="Ernie55HCACoreAttention"):
            _build_attention(_config(), pg_collection, seed=None, spec=spec)

    @needs_hook
    def test_core_must_have_a_compressor(self, pg_collection):
        spec = _attention_spec()
        core = spec.submodules.core_attention
        spec.submodules.core_attention = functools.partial(
            core.func, submodules=dataclasses.replace(core.keywords["submodules"], compressor=None)
        )
        with pytest.raises(ValueError, match="a compressor and no indexer"):
            _build_attention(_config(), pg_collection, seed=None, spec=spec)

    @needs_hook
    @pytest.mark.parametrize("norm_name", ["q_layernorm", "kv_layernorm"])
    def test_latent_norms_must_be_explicit(self, pg_collection, norm_name):
        spec = _attention_spec()
        setattr(spec.submodules, norm_name, IdentityOp)
        with pytest.raises(ValueError, match=f"explicit {norm_name}"):
            _build_attention(_config(), pg_collection, seed=None, spec=spec)

    @needs_hook
    def test_gate_must_be_in_the_spec(self, pg_collection):
        spec = _attention_spec()
        spec.submodules.linear_output_gate = None
        with pytest.raises(ValueError, match="linear_output_gate"):
            _build_attention(_config(), pg_collection, seed=None, spec=spec)

    @pytest.mark.parametrize(("overrides", "message"), _SHARED_ENVELOPE + _H_ONLY_ENVELOPE)
    def test_config_envelope(self, pg_collection, overrides, message):
        """Each H requirement raises its own error, from the validator and from the attention itself.

        The H attention validates its config itself; it need not come from Ernie55TransformerConfig.
        """
        config = _violating_config(overrides)
        with pytest.raises(ValueError, match=re.escape(f"ERNIE 5.5 H layer: {message}")):
            validate_ernie55_hca_config(config)
        with pytest.raises(ValueError, match=re.escape(f"ERNIE 5.5 H layer: {message}")):
            _build_attention(config, pg_collection, seed=None)

    def test_context_parallel_follows_the_converter(self, monkeypatch):
        """CP > 1 is Megatron-Core DSv4's contiguous CP; builds without the CP converter (main) raise."""
        config = _violating_config({"context_parallel_size": 2})
        if context_parallel.load_cp_layout_api() is None:
            with pytest.raises(RuntimeError, match="CP partition converter"):
                validate_ernie55_hca_config(config)
        else:
            validate_ernie55_hca_config(config)
        monkeypatch.setattr(context_parallel, "load_cp_layout_api", lambda: None)
        with pytest.raises(RuntimeError, match="ERNIE 5.5 H layers need Megatron-Core's module-local CP"):
            validate_ernie55_hca_config(config)

    @pytest.mark.parametrize(("overrides", "message"), _SHARED_ENVELOPE)
    def test_shared_requirements_match_the_d_layer(self, overrides, message):
        """The D-layer validator enforces every requirement the H validator shares with it, with the same message.

        The two validators check these requirements separately; this keeps them from drifting apart.
        """
        with pytest.raises(ValueError, match=re.escape(f"ERNIE 5.5 D layer: {message}")):
            validate_ernie55_config(_violating_config(overrides))

    def test_finalize_checks_the_h_envelope_with_dsv4_hybrid_only(self):
        with pytest.raises(ValueError, match="ERNIE 5.5 H layer: rotary_percent"):
            _config(rotary_percent=0.5)
        # A D-only stack (experimental_attention_variant="dsa") has no H layers to validate.
        _config(
            rotary_percent=0.5,
            experimental_attention_variant="dsa",
            kv_lora_rank=16,
            qk_head_dim=8,
            csa_compress_ratios=None,
        )

    @needs_hook
    def test_fused_backend_selects_megatron_core_fused_path(self, pg_collection):
        """dsa_kernel_backend='cudnn' selects the fused FlashMLA/cuDNN path the reference implementation uses.

        The compressor stays eager, as in the reference implementation: the same switch would also enable
        cudnn-frontend's fused compressor, so the constructor turns it off.
        """
        fused = _build_attention(_with_cudnn_backend(_config()), pg_collection, seed=None)
        reference = _build_attention(_config(), pg_collection, seed=None)
        assert fused.core_attention.use_fused_kernels and not reference.core_attention.use_fused_kernels
        assert type(fused.core_attention.compressor) is csa.Compressor
        if hasattr(csa, "maybe_compress_thd_fused"):
            assert fused.core_attention.compressor.use_fused_compressor is False


@needs_hook
class TestLayerCheck:
    def test_counts_h_layers_and_rejects_foreign_attention(self, pg_collection):
        ernie = _build_attention(_config(), pg_collection, seed=None)
        assert check_ernie55_attention_layers(torch.nn.ModuleList([ernie])) == {"H": 1, "D": 0}

        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

        mcore_spec = _substitute(
            hybrid_stack_spec.submodules.csa_qk_layernorm_layer.submodules.self_attention, _te_stand_ins()
        )
        mcore_spec.submodules.q_layernorm = _RMSNorm
        mcore_spec.submodules.kv_layernorm = _RMSNorm
        plain = build_module(
            mcore_spec, config=_config(), layer_number=2, pg_collection=pg_collection, compress_ratio=RATIO
        )
        assert type(plain) is dsv4.DSv4HybridSelfAttention
        with pytest.raises(TypeError, match="DSv4HybridSelfAttention"):
            check_ernie55_attention_layers(torch.nn.ModuleList([ernie, torch.nn.Sequential(plain)]))


# ---------------------------------------------------------------------------------------------
# Whole stacks: Megatron-Core's HybridStack built from ernie55_hybrid_stack_spec()
# ---------------------------------------------------------------------------------------------

# A scaled-down ERNIE 5.5 D layer, for stacks that also contain one.
_D_GEOMETRY = dict(
    ernie_d_num_attention_heads=4,
    ernie_d_q_lora_rank=32,
    ernie_d_kv_lora_rank=16,
    ernie_d_qk_head_dim=8,
    ernie_d_qk_pos_emb_head_dim=8,
    ernie_d_v_head_dim=8,
    ernie_d_rotary_base=10000.0,
)


class _StackNorm:
    """``TENorm`` stand-in for whole stacks: :class:`_RMSNorm`, or a torch LayerNorm for LayerNorm configs.

    The D layer's ``DSAIndexer`` builds its key norm with ``normalization="LayerNorm"``.
    """

    def __new__(cls, config, hidden_size: int, eps: float = 1e-5, **kwargs):
        if config.normalization == "RMSNorm":
            return _RMSNorm(config, hidden_size, eps=eps, **kwargs)
        return WrappedTorchNorm(config=config, hidden_size=hidden_size, eps=eps)


@needs_hook
class TestHybridStack:
    """Megatron-Core's ``HybridStack`` builds the ERNIE layers from ``ernie55_hybrid_stack_spec()``.

    Megatron-Core picks the spec slot of an ``H`` layer itself (``hca_layer`` first on dev, then
    ``csa_qk_layernorm_layer`` or ``csa_layer`` by ``qk_layernorm``). Without
    ``mtp_stack_submodules`` it builds the nested MTP stack from the decoder's submodules with
    ``is_mtp_layer=True``; these tests build the stacks the same way.
    """

    @pytest.fixture
    def stack_pg_collection(self, pg_collection, monkeypatch):
        """Process groups for a whole stack, plus the stand-ins the D layer needs on CPU."""
        from megatron.core.transformer import multi_latent_attention
        from megatron.core.transformer.experimental_attention_variant import absorbed_mla

        # AbsorbedMLA picks the down-projection arguments by comparing the spec with its TELinear name.
        for module in (absorbed_mla, multi_latent_attention):
            monkeypatch.setattr(module, "TELinear", _DuplicatedLinear)
        return ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp", "pp", "tp_cp"])

    @staticmethod
    def _build_stack(pattern: str, pg_collection, *, qk_layernorm: bool, is_mtp_layer: bool):
        from megatron.core.extensions.transformer_engine import TENorm
        from megatron.core.models.hybrid.hybrid_block import HybridStack
        from megatron.core.models.hybrid.hybrid_layer_allocation import validate_segment_layers

        stand_ins = [(te_module, _StackNorm if te_module is TENorm else cpu) for te_module, cpu in _te_stand_ins()]
        spec = _substitute(ernie55_hybrid_stack_spec(), stand_ins)
        config = _config(
            num_layers=len(pattern),
            # One entry per layer: 128 for H; D layers do not read theirs.
            csa_compress_ratios=[RATIO if symbol == "H" else 0 for symbol in pattern],
            qk_layernorm=qk_layernorm,
            **_D_GEOMETRY,
        )
        return HybridStack(
            config,
            spec.submodules,
            layer_config_list=validate_segment_layers(pattern, config),
            pre_process=True,
            post_layer_norm=False,
            post_process=False,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            mtp_layer_number=1 if is_mtp_layer else None,
        )

    @pytest.mark.parametrize("is_mtp_layer", [False, True], ids=["decoder", "mtp"])
    @pytest.mark.parametrize("qk_layernorm", [False, True], ids=["qk-norm-off", "qk-norm-on"])
    @pytest.mark.parametrize("pattern", ["H", "HD"])
    def test_symbols_build_the_ernie_layers(self, stack_pg_collection, pattern, qk_layernorm, is_mtp_layer):
        """``H`` and ``D`` symbols build the ERNIE attentions, which the build-time check counts."""
        stack = self._build_stack(pattern, stack_pg_collection, qk_layernorm=qk_layernorm, is_mtp_layer=is_mtp_layer)
        expected = {"H": Ernie55HCASelfAttention, "D": Ernie55LatentMQASelfAttention}
        assert [type(layer.self_attention) for layer in stack.layers] == [expected[symbol] for symbol in pattern]
        assert check_ernie55_attention_layers(stack) == {"H": pattern.count("H"), "D": pattern.count("D")}
        for layer, symbol in zip(stack.layers, pattern):
            if symbol == "H":
                attention = layer.self_attention
                assert attention._dsv4_compress_ratio == RATIO and not attention._dsv4_uses_yarn_rope
        if pattern == "H":
            x = torch.randn(260, 2, HIDDEN, generator=torch.Generator().manual_seed(71))
            output = stack(x, None)
            assert output.shape == x.shape and bool(torch.isfinite(output).all())


# ---------------------------------------------------------------------------------------------
# Parameter names, shapes and dtypes against the reference model's manifest
# ---------------------------------------------------------------------------------------------

# Layer 0 (an H layer) of the reference proxy model, relative to decoder.layers.0.self_attention:
# parameter names, shapes and dtypes of the reference implementation.
_REFERENCE_H_MANIFEST = {
    "linear_o_group_proj": ((8192, 4096), torch.bfloat16),
    "core_attention.attn_sink": ((64,), torch.float32),
    "core_attention.compressor.ape": ((128, 512), torch.float32),
    "core_attention.compressor.linear_wkv.weight": ((512, 4096), torch.bfloat16),
    "core_attention.compressor.linear_wgate.weight": ((512, 4096), torch.bfloat16),
    "core_attention.compressor.norm.weight": ((512,), torch.bfloat16),
    "linear_proj.weight": ((4096, 8192), torch.bfloat16),
    "linear_q_down_proj.weight": ((1024, 4096), torch.bfloat16),
    "linear_q_up_proj.weight": ((32768, 1024), torch.bfloat16),
    "linear_kv_proj.weight": ((512, 4096), torch.bfloat16),
    "kv_layernorm.weight": ((512,), torch.bfloat16),
    "q_layernorm.weight": ((1024,), torch.bfloat16),
    "vha.U": ((64, 16), torch.float32),
    "vha.V": ((64, 16), torch.float32),
    "output_gate.gate_proj.weight": ((8192, 1024), torch.bfloat16),
}
# Bridge parameter names that differ from the reference model's (the checkpoint mappings rename them).
_BRIDGE_TO_REFERENCE_NAMES = {
    "core_attention.vha_postmix_u": "vha.U",
    "core_attention.vha_postmix_v": "vha.V",
    "linear_output_gate.weight": "output_gate.gate_proj.weight",
}
# The ERNIE 5.5 proxy's H-layer geometry (Hugging Face config of the reference run).
_PROXY_H_GEOMETRY = dict(
    hidden_size=4096,
    num_attention_heads=64,
    q_lora_rank=1024,
    v_head_dim=512,
    qk_pos_emb_head_dim=64,
    output_projection_groups=8,
    output_projection_lora_rank=1024,
    csa_window_size=128,
    csa_compress_rotary_base=160000.0,
    ernie_vha_rank=16,
)


@needs_hook
class TestParameters:
    def test_names_shapes_and_dtypes_match_the_reference_manifest(self, pg_collection):
        config = _config(**_PROXY_H_GEOMETRY, params_dtype=torch.bfloat16, perform_initialization=False)
        attention = _build_attention(config, pg_collection, seed=None)
        found = {
            _BRIDGE_TO_REFERENCE_NAMES.get(name, name): (tuple(param.shape), param.dtype)
            for name, param in attention.named_parameters()
        }
        assert found == _REFERENCE_H_MANIFEST
        for name, param in attention.named_parameters():
            assert getattr(param, "keep_in_fp32", False) == (param.dtype == torch.float32), name


# ---------------------------------------------------------------------------------------------
# Stack spec
# ---------------------------------------------------------------------------------------------


def _spec_equal(a: object, b: object) -> bool:
    """Structural equality of spec trees; ``functools.partial`` compares by function and arguments."""
    if isinstance(a, functools.partial) or isinstance(b, functools.partial):
        return (
            type(a) is type(b)
            and a.func is b.func
            and _spec_equal(a.args, b.args)
            and _spec_equal(a.keywords, b.keywords)
        )
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_spec_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_spec_equal(a[key], b[key]) for key in a)
    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        return type(a) is type(b) and all(
            _spec_equal(getattr(a, field.name), getattr(b, field.name)) for field in dataclasses.fields(a)
        )
    return a is b or a == b


class TestStackSpec:
    def test_h_slots_get_the_ernie_h_layer(self):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

        spec = ernie55_hybrid_stack_spec()
        slots = {spec_field.name for spec_field in dataclasses.fields(spec.submodules)}
        h_slots = [slot for slot in ("csa_layer", "csa_qk_layernorm_layer", "hca_layer") if slot in slots]
        assert h_slots[:2] == ["csa_layer", "csa_qk_layernorm_layer"]
        assert ("hca_layer" in h_slots) == _HAS_HCA_SLOT

        mcore_layer = hybrid_stack_spec.submodules.csa_qk_layernorm_layer
        mcore_attention = mcore_layer.submodules.self_attention
        mcore_compressor = mcore_attention.submodules.core_attention.keywords["submodules"].compressor
        layers = [getattr(spec.submodules, slot) for slot in h_slots]
        assert len({id(layer) for layer in layers}) == len(layers), "H slots must not share spec objects"
        for slot, layer in zip(h_slots, layers):
            attention = layer.submodules.self_attention
            assert attention.module is Ernie55HCASelfAttention, slot
            assert isinstance(attention.submodules, Ernie55HCASelfAttentionSubmodules), slot
            assert attention.submodules.linear_output_gate is TELinear
            # Every RMSNorm is TENorm, or the reference arithmetic under ernie_reference_rmsnorm.
            assert _spec_equal(attention.submodules.q_layernorm, ernie55_rmsnorm(TENorm))
            assert _spec_equal(attention.submodules.kv_layernorm, ernie55_rmsnorm(TENorm))
            core = attention.submodules.core_attention
            assert isinstance(core, functools.partial) and core.func is Ernie55HCACoreAttention
            assert core.keywords["submodules"].indexer is None
            compressor = core.keywords["submodules"].compressor
            assert mcore_compressor.keywords["submodules"].norm is TENorm
            assert _spec_equal(compressor.keywords["submodules"].norm, ernie55_rmsnorm(TENorm))
            assert _spec_equal(
                functools.partial(
                    compressor.func,
                    **{
                        **compressor.keywords,
                        "submodules": dataclasses.replace(compressor.keywords["submodules"], norm=None),
                    },
                ),
                functools.partial(
                    mcore_compressor.func,
                    **{
                        **mcore_compressor.keywords,
                        "submodules": dataclasses.replace(mcore_compressor.keywords["submodules"], norm=None),
                    },
                ),
            )
            # The input norm downcasts the fp32 hyper-connection streams (test_ernie55_mhc_mtp.py).
            assert mcore_layer.submodules.input_layernorm is TENorm
            assert _spec_equal(layer.submodules.input_layernorm, downcasting_norm(TENorm))
            # Everything else is Megatron-Core's csa_qk_layernorm_layer.
            assert _spec_equal(
                dataclasses.replace(layer.submodules, self_attention=None, input_layernorm=None),
                dataclasses.replace(mcore_layer.submodules, self_attention=None, input_layernorm=None),
            )
            assert layer.module is mcore_layer.module and attention.params == mcore_attention.params
            for name in ("linear_q_down_proj", "linear_q_up_proj", "linear_kv_proj", "linear_proj"):
                assert getattr(attention.submodules, name) is getattr(mcore_attention.submodules, name), name

    def test_other_slots_and_megatron_core_are_unchanged(self):
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

        spec = ernie55_hybrid_stack_spec()
        for slot in ("csa_layer", "csa_qk_layernorm_layer"):
            mcore_attention = getattr(hybrid_stack_spec.submodules, slot).submodules.self_attention
            assert mcore_attention.module is dsv4.DSv4HybridSelfAttention
            assert mcore_attention.submodules.core_attention.func is csa.CompressedSparseAttention
        if _HAS_HCA_SLOT:
            assert hybrid_stack_spec.submodules.hca_layer is None
        # The D slot is checked in test_ernie55_latent_mqa.py, the MLP and MoE slots in test_ernie55_moe.py,
        # the MTP hnorm in test_ernie55_mhc_mtp.py.
        changed = {
            "dsa_layer",
            "csa_layer",
            "csa_qk_layernorm_layer",
            "hca_layer",
            "mlp_layer",
            "moe_layer",
            "mtp_block_spec",
        }
        for slot in dataclasses.fields(spec.submodules):
            if slot.name not in changed:
                mcore_slot = getattr(hybrid_stack_spec.submodules, slot.name)
                assert _spec_equal(getattr(spec.submodules, slot.name), mcore_slot), slot.name

    def test_mtp_stack_inherits_the_h_layer(self):
        """HybridModel builds the MTP stack from ``mtp_stack_submodules``, or the decoder's when unset."""
        spec = ernie55_hybrid_stack_spec()
        assert spec.submodules.mtp_stack_submodules is None
        assert spec.submodules.mtp_block_spec is not None
