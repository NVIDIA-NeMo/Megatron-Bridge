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

"""CPU tests of the ERNIE 5.5 D layer against an independent fp64 re-derivation of its math.

The oracle in this file is written from the ERNIE 5.5 D-layer specification, not from the
Bridge implementation. Per query row ``t`` of a document that starts at ``d(t)``:

* ``S(t) = TopK(t) | Window(t)``: ``TopK(t)`` holds the ``K`` remote keys of ``[d(t), t - W]``
  with the highest indexer scores (all of them when there are at most ``K``), and
  ``Window(t) = [max(d(t), t - W + 1), t]``.
* Latent MQA with a per-head sink in the denominator only:
  ``o_h = sum_{j in S(t)} e^{s_hj} c_kv_j / (sum_{j in S(t)} e^{s_hj} + e^{sink_h})``.
* V up-projection, VHA head mix ``v'_k = sum_h (I + V U^T)[k, h] v_h``, then
  ``y = flatten_head_major(v') * sigmoid(W_g c_q)``, where ``c_q`` is the q latent.
* Indexer loss: a top-k-only teacher (per-head softmax over ``TopK(t)`` of the attention logits,
  summed over heads and L1-normalized), a softmax student over the same slots, and
  ``coeff * sum_t KL_t / (B * S)`` over every row. Only the indexer learns from it.

The self-attention oracle uses the non-absorbed MLA form (per-head ``k = W_UK c_kv``,
``v = W_UV c_kv``, adjacent-pair RoPE as complex rotations), so it also checks Megatron-Core's
absorption and the D-layer config rewrite. Everything runs on CPU with Transformer-Engine-free
builders and the PyTorch reference path (``dsa_kernel_backend="none"``).
"""

import copy
import dataclasses
import functools
import inspect
import logging
import math
import types
import zlib
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.models.backends import LocalSpecProvider
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import csa, dsa, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import (
    AbsorbedMLASelfAttention,
    AbsorbedMLASelfAttentionSubmodules,
)
from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    REFERENCE_FALLBACK_MAX_TOKENS,
    Ernie55LatentMQACoreAttention,
    Ernie55LatentMQACoreSubmodules,
    Ernie55LatentMQASelfAttention,
    downcasting_norm,
    ernie55_d_layer_config,
    ernie55_hybrid_stack_spec,
    ernie55_rmsnorm_spec,
    indexer_loss_layer_number,
    latent_mqa_core,
    ops,
)
from tests.unit_tests.models.ernie.ernie55_test_utils import cpu_current_device


pytestmark = pytest.mark.unit

# D-layer geometry (a scaled-down ERNIE 5.5 D layer).
HIDDEN = 64
HEADS = 4
Q_LORA = 32
KV_LORA = 16  # latent width C; the value is the first C latent channels
NOPE = 8
ROPE = 8
V_HEAD = 8
LATENT = KV_LORA + ROPE
VHA_RANK = 2
INDEXER_HEADS = 4
INDEXER_DIM = 16
TOPK = 6  # K
WINDOW = 3  # W; a row needs the indexer once its position exceeds W + K - 1
NUM_LAYERS = 2
ROTARY_BASE = 50.0
EPS = 1e-6
COEFF = 0.01
SCALE = 1.0 / math.sqrt(NOPE + ROPE)

# fp32 module vs fp64 oracle, as a normalized max error (max |diff| / max |expected|).
TOL = 2e-5
# DSAIndexer rounds its head weights to bf16, so their gradient is bf16-accurate.
BF16_GRAD_TOL = 2e-2
LOOSE_GRADS = {"linear_weights_proj.weight": BF16_GRAD_TOL}

# Named parameters of the D attention; the names are pinned by the checkpoint mappings.
EXPECTED_PARAMETERS = {
    "core_attention.attn_sink": (HEADS,),
    "core_attention.indexer.k_norm.bias": (INDEXER_DIM,),
    "core_attention.indexer.k_norm.weight": (INDEXER_DIM,),
    "core_attention.indexer.linear_weights_proj.weight": (INDEXER_HEADS, HIDDEN),
    "core_attention.indexer.linear_wk.weight": (INDEXER_DIM, HIDDEN),
    "core_attention.indexer.linear_wq_b.weight": (INDEXER_HEADS * INDEXER_DIM, Q_LORA),
    "core_attention.linear_output_gate.weight": (HEADS * V_HEAD, Q_LORA),
    "core_attention.vha_postmix_u": (HEADS, VHA_RANK),
    "core_attention.vha_postmix_v": (HEADS, VHA_RANK),
    "k_b_proj": (HEADS * KV_LORA, NOPE),
    "kv_layernorm.weight": (KV_LORA,),
    "linear_kv_down_proj.weight": (LATENT, HIDDEN),
    "linear_proj.weight": (HIDDEN, HEADS * V_HEAD),
    "linear_q_down_proj.weight": (Q_LORA, HIDDEN),
    "linear_q_up_proj.weight": (HEADS * (NOPE + ROPE), Q_LORA),
    "q_layernorm.weight": (Q_LORA,),
    "v_b_proj": (HEADS * V_HEAD, KV_LORA),
}


# ---------------------------------------------------------------------------------------------
# Runtime fixtures
# ---------------------------------------------------------------------------------------------


def _sylvester_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard transform over the last dim (fp64 math, input dtype out)."""
    size = x.size(-1)
    hadamard = torch.ones(1, 1, dtype=torch.float64, device=x.device)
    while hadamard.size(0) < size:
        hadamard = torch.cat([torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)], 0)
    return (x.double() @ hadamard * size**-0.5).to(x.dtype)


@pytest.fixture(scope="module")
def pg_collection(tmp_path_factory):
    """Single-rank process groups plus the CPU stand-ins Megatron-Core needs on a CPU-only run."""
    created_group = not torch.distributed.is_initialized()
    if created_group:
        store = tmp_path_factory.mktemp("ernie55_latent_mqa_gloo") / "store"
        torch.distributed.init_process_group("gloo", init_method=f"file://{store}", rank=0, world_size=1)
    created_model_parallel = not parallel_state.model_parallel_is_initialized()
    if created_model_parallel:
        parallel_state.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        with pytest.MonkeyPatch.context() as patch:
            # RoPE tables and the DSA loss tracker are allocated on torch.cuda.current_device();
            # these tests run on CPU even when a GPU is present (torch's own callers still see the GPU).
            patch.setattr(torch.cuda, "current_device", cpu_current_device)
            # DSAIndexer's Hadamard rotation needs fast_hadamard_transform and bf16 CUDA input. Any
            # orthonormal transform leaves the indexer scores unchanged, so use an exact one.
            patch.setattr(dsa, "rotate_activation", _sylvester_hadamard)
            yield ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
    finally:
        if created_model_parallel:
            parallel_state.destroy_model_parallel()
        if created_group:
            torch.distributed.destroy_process_group()


@pytest.fixture(autouse=True)
def _isolated_indexer_loss_state():
    """Give every test an empty DSA loss tracker and the default indexer loss scale."""
    tracker = dsa.DSAIndexerLossLoggingHelper.tracker
    saved_tracker = dict(tracker)
    saved_scale = dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale
    tracker.clear()
    dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None
    yield
    tracker.clear()
    tracker.update(saved_tracker)
    dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = saved_scale


# ---------------------------------------------------------------------------------------------
# Transformer-Engine-free builders
# ---------------------------------------------------------------------------------------------


class _DuplicatedLinear(torch.nn.Module):
    """Stand-in for ``TELinear(parallel_mode="duplicated")``: ``y = x W^T``, returns ``(y, None)``."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: Ernie55TransformerConfig,
        init_method,
        bias: bool,
        skip_bias_add: bool,
        skip_weight_param_allocation: bool,
        parallel_mode: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert parallel_mode == "duplicated" and not bias and not skip_weight_param_allocation
        self.weight = torch.nn.Parameter(torch.empty(output_size, input_size, dtype=config.params_dtype))
        init_method(self.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return F.linear(x, self.weight), None

    def backward_dw(self) -> None:
        pass


def _core_submodules() -> Ernie55LatentMQACoreSubmodules:
    indexer = ModuleSpec(
        module=dsa.DSAIndexer,
        submodules=dsa.DSAIndexerSubmodules(
            linear_wq_b=_DuplicatedLinear,
            linear_wk=_DuplicatedLinear,
            k_norm=WrappedTorchNorm,
            linear_weights_proj=_DuplicatedLinear,
        ),
    )
    return Ernie55LatentMQACoreSubmodules(indexer=indexer, linear_output_gate=_DuplicatedLinear)


def _attention_spec() -> ModuleSpec:
    return ModuleSpec(
        module=Ernie55LatentMQASelfAttention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=AbsorbedMLASelfAttentionSubmodules(
            linear_q_down_proj=ColumnParallelLinear,
            linear_q_up_proj=ColumnParallelLinear,
            linear_kv_down_proj=ColumnParallelLinear,
            linear_kv_up_proj=ColumnParallelLinear,
            core_attention=ModuleSpec(module=Ernie55LatentMQACoreAttention, submodules=_core_submodules()),
            linear_proj=RowParallelLinear,
            q_layernorm=WrappedTorchNorm,
            kv_layernorm=WrappedTorchNorm,
        ),
    )


def _config(**overrides) -> Ernie55TransformerConfig:
    """Finalized stack config whose D layer has the geometry above (YaRN at the stack level)."""
    values = dict(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_head_dim=NOPE,
        qk_pos_emb_head_dim=ROPE,
        v_head_dim=V_HEAD,
        ernie_vha_rank=VHA_RANK,
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        rotary_base=ROTARY_BASE,
        # The D-layer rewrite must replace these stack-level YaRN settings with plain RoPE.
        rope_type="yarn",
        rotary_scaling_factor=40.0,
        mscale=1.0,
        mscale_all_dim=1.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        experimental_attention_variant="dsa",
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=INDEXER_HEADS,
        dsa_indexer_head_dim=INDEXER_DIM,
        dsa_indexer_topk=TOPK,
        dsa_indexer_loss_coeff=COEFF,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_k_norm_fp32=True,
        csa_window_size=WINDOW,
        use_cpu_initialization=True,
        transformer_impl="local",
        # The D layer is BF16-only; the tests keep fp32 parameters and activations (params_dtype)
        # with an exact stand-in for DSAIndexer's bf16-only Hadamard rotation (see pg_collection).
        bf16=True,
        params_dtype=torch.float32,
    )
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    """Draw every parameter at O(1) scale, including the zero-initialized sink and VHA factors.

    Each parameter has its own generator keyed by its name, so the values do not depend on the
    order in which Megatron-Core registers parameters.
    """
    with torch.no_grad():
        for name, param in module.named_parameters():
            generator = torch.Generator().manual_seed(seed * 1_000_003 + zlib.crc32(name.encode()))
            noise = torch.randn(param.shape, generator=generator, dtype=torch.float64)
            if name.endswith("attn_sink"):
                values = noise
            elif "vha_postmix" in name:
                values = 0.5 * noise
            elif "norm" in name:
                values = (1.0 + 0.2 * noise) if name.endswith("weight") else 0.2 * noise
            else:
                values = noise / math.sqrt(param.shape[-1])
            param.copy_(values.to(param.dtype))


def _build_core(
    config: Ernie55TransformerConfig,
    pg_collection: ProcessGroupCollection,
    *,
    seed: int,
    layer_number: int = 1,
) -> Ernie55LatentMQACoreAttention:
    """Build the core the way AbsorbedMLA does (D config, latent channels, 1/sqrt(nope + rope))."""
    core = Ernie55LatentMQACoreAttention(
        ernie55_d_layer_config(config),
        _core_submodules(),
        layer_number=layer_number,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self",
        softmax_scale=SCALE,
        k_channels=LATENT,
        v_channels=KV_LORA,
        pg_collection=pg_collection,
    )
    _randomize_(core, seed)
    return core.train()


def _build_attention(
    config: Ernie55TransformerConfig,
    pg_collection: ProcessGroupCollection,
    *,
    seed: int,
    layer_number: int = 1,
    is_mtp_layer: bool = False,
) -> Ernie55LatentMQASelfAttention:
    attention = build_module(
        _attention_spec(),
        config=config,
        layer_number=layer_number,
        pg_collection=pg_collection,
        is_mtp_layer=is_mtp_layer,
    )
    _randomize_(attention, seed)
    return attention.train()


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


@dataclass
class _Batch:
    """Core inputs in the layout AbsorbedMLA passes (THD packs drop the batch dim of q, k, qr)."""

    query: torch.Tensor
    key: torch.Tensor
    qr: torch.Tensor
    x: torch.Tensor
    up_v: torch.Tensor
    packed_seq_params: PackedSeqParams | None
    doc_start: list[int]

    @property
    def batch_size(self) -> int:
        return 1 if self.query.dim() == 3 else self.query.size(1)


def _cumsum(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)


def _packed_seq_params(doc_lens: list[int], padded_lens: list[int] | None = None) -> PackedSeqParams:
    cu_seqlens = _cumsum(doc_lens)
    cu_seqlens_padded = _cumsum(padded_lens) if padded_lens is not None else None
    max_seqlen = max(padded_lens or doc_lens)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _oracle_document_starts(packed_seq_params: PackedSeqParams | None, num_rows: int) -> list[int]:
    """Start of each row's document: padded boundaries win, rows past the end join the last document."""
    if packed_seq_params is None:
        return [0] * num_rows
    cu_seqlens = packed_seq_params.cu_seqlens_q_padded
    if cu_seqlens is None:
        cu_seqlens = packed_seq_params.cu_seqlens_q
    starts = cu_seqlens.tolist()[:-1]
    return [max(start for start in starts if start <= row) for row in range(num_rows)]


def _core_batch(
    *,
    seed: int,
    doc_lens: list[int] | None = None,
    padded_lens: list[int] | None = None,
    tail_rows: int = 0,
    seqlen: int | None = None,
    batch: int = 1,
    dtype: torch.dtype = torch.float32,
) -> _Batch:
    """Random core inputs: a THD pack of ``doc_lens`` (+ tail rows) or an SBHD ``[seqlen, batch]`` batch."""
    generator = torch.Generator().manual_seed(seed)

    def draw(*shape: int) -> torch.Tensor:
        return torch.randn(shape, generator=generator, dtype=torch.float64).to(dtype).requires_grad_(True)

    if doc_lens is not None:
        num_rows = sum(padded_lens or doc_lens) + tail_rows
        packed_seq_params = _packed_seq_params(doc_lens, padded_lens)
        query, key, qr = draw(num_rows, HEADS, LATENT), draw(num_rows, 1, LATENT), draw(num_rows, Q_LORA)
        x = draw(num_rows, 1, HIDDEN)
    else:
        num_rows, packed_seq_params = seqlen, None
        query, key = draw(num_rows, batch, HEADS, LATENT), draw(num_rows, batch, 1, LATENT)
        qr, x = draw(num_rows, batch, Q_LORA), draw(num_rows, batch, HIDDEN)
    up_v = (draw(HEADS, V_HEAD, KV_LORA).detach() / math.sqrt(KV_LORA)).requires_grad_(True)
    doc_start = _oracle_document_starts(packed_seq_params, num_rows)
    return _Batch(query, key, qr, x, up_v, packed_seq_params, doc_start)


def _run_core(core: Ernie55LatentMQACoreAttention, batch: _Batch) -> torch.Tensor:
    return core(
        batch.query,
        batch.key,
        None,
        None,
        x=batch.x,
        qr=batch.qr,
        attn_mask_type=AttnMaskType.causal,
        packed_seq_params=batch.packed_seq_params,
        up_v_weight=batch.up_v,
    )


def _positions(doc_start: list[int]) -> torch.Tensor:
    return torch.tensor([row - start for row, start in enumerate(doc_start)], dtype=torch.float64)


# ---------------------------------------------------------------------------------------------
# fp64 oracle, written from the ERNIE 5.5 D-layer specification
# ---------------------------------------------------------------------------------------------


def _leaves(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """fp64 leaf copies of the module parameters, by name."""
    return {name: param.detach().double().requires_grad_(True) for name, param in module.named_parameters()}


def _strip(params: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {name[len(prefix) :]: value for name, value in params.items() if name.startswith(prefix)}


def _rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + EPS) * weight


def _rope_frequencies(positions: torch.Tensor, rotary_base: float) -> torch.Tensor:
    """Rotation angles ``pos * base^(-2i / ROPE)``, ``[rows, ROPE / 2]``."""
    inv_freq = rotary_base ** (-torch.arange(0, ROPE, 2, dtype=torch.float64) / ROPE)
    return positions[:, None] * inv_freq[None, :]


def _broadcast_rows(angles: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return angles.view(angles.size(0), *([1] * (like.dim() - 2)), angles.size(1))


def _rope_adjacent_pairs(x: torch.Tensor, positions: torch.Tensor, rotary_base: float) -> torch.Tensor:
    """Latent RoPE: channel pairs ``(2i, 2i + 1)`` rotate as complex numbers by ``pos * base^(-2i/ROPE)``."""
    pairs = torch.view_as_complex(x.reshape(*x.shape[:-1], ROPE // 2, 2).contiguous())
    angles = _broadcast_rows(_rope_frequencies(positions, rotary_base), pairs)
    rotated = pairs * torch.polar(torch.ones_like(angles), angles)
    return torch.view_as_real(rotated).reshape(x.shape)


def _rope_half_split(x: torch.Tensor, positions: torch.Tensor, rotary_base: float) -> torch.Tensor:
    """Indexer RoPE on the first ROPE channels: pairs ``(i, i + ROPE/2)`` rotate; the rest pass through."""
    half = ROPE // 2
    angles = _broadcast_rows(_rope_frequencies(positions, rotary_base), x)
    cos, sin = angles.cos(), angles.sin()
    first, second, rest = x[..., :half], x[..., half:ROPE], x[..., ROPE:]
    return torch.cat([first * cos - second * sin, second * cos + first * sin, rest], dim=-1)


def _oracle_index_scores(
    params: dict[str, torch.Tensor],
    x: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_base: float,
    *,
    module_x: torch.Tensor,
    module_weights_proj: torch.Tensor,
) -> torch.Tensor:
    """Indexer logits ``[b, t, s] = sum_h w_th * relu(q_th . k_s)``.

    ``q = Had(rope(W_qb c_q))``, ``k = Had(rope(LayerNorm(W_k x)))`` and
    ``w = (W_w x) * HI^-1/2 * DI^-1/2``. The indexer only sees detached ``x`` and ``c_q``. The
    head weights are emitted in bf16 (``dsa_indexer_weights_proj_output_dtype="bf16"``, an ERNIE
    invariant), so their value is the module's own projection (``module_x``, ``module_weights_proj``
    in the module's dtype) rounded to bf16; the gradient keeps the exact path.
    """
    x, qr = x.detach(), qr.detach()
    seqlen, batch = x.shape[:2]
    q = (qr @ params["linear_wq_b.weight"].T).view(seqlen, batch, INDEXER_HEADS, INDEXER_DIM)
    q = _sylvester_hadamard(_rope_half_split(q, positions, rotary_base))
    k = F.layer_norm(
        x @ params["linear_wk.weight"].T, (INDEXER_DIM,), params["k_norm.weight"], params["k_norm.bias"], EPS
    )
    k = _sylvester_hadamard(_rope_half_split(k, positions, rotary_base))
    head_weights = x @ params["linear_weights_proj.weight"].T
    rounded = F.linear(module_x.detach(), module_weights_proj.detach()).to(torch.bfloat16).double()
    head_weights = (head_weights + (rounded - head_weights).detach()) * INDEXER_HEADS**-0.5 * INDEXER_DIM**-0.5
    dots = torch.einsum("tbhd,sbd->bths", q, k).relu()
    return torch.einsum("bths,tbh->bts", dots, head_weights)


def _designed_index_scores(q_index: torch.Tensor, k_index: torch.Tensor, w_index: torch.Tensor) -> torch.Tensor:
    """The same indexer logit for injected ``q [s, b, HI, DI]``, ``k [s, b, DI]``, ``w [s, b, HI]``."""
    dots = torch.einsum("tbhd,sbd->bths", q_index.double(), k_index.double()).relu()
    return torch.einsum("bths,tbh->bts", dots, w_index.double())


def _oracle_index_sets(
    doc_start: list[int],
    batch: int,
    index_scores: torch.Tensor | None,
    *,
    selected: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean ``[b, t, s]`` masks of ``TopK(t)`` and ``Window(t)``.

    Rows with more than ``K`` remote candidates need ``index_scores``. The selection has no
    tie-break rule, and ReLU scoring produces exact ties (rows where every head's ``q . k <= 0``
    score 0). When the ``K``-th and ``(K+1)``-th scores tie, any top-``K`` set is correct: the
    module's remote keys, read from its ``[b, s, M]`` index lists ``selected``, are accepted
    after checking that none of them scores below a candidate it left out.
    """
    num_rows = len(doc_start)
    topk = torch.zeros(batch, num_rows, num_rows, dtype=torch.bool)
    window = torch.zeros(batch, num_rows, num_rows, dtype=torch.bool)
    for sample in range(batch):
        for row in range(num_rows):
            start = doc_start[row]
            window[sample, row, max(start, row - WINDOW + 1) : row + 1] = True
            candidates = list(range(start, row - WINDOW + 1))
            if len(candidates) > TOPK:
                assert index_scores is not None, f"row {row} is in the selection regime"
                scores = index_scores[sample, row].detach()
                ranked, order = torch.sort(scores[candidates], descending=True)
                tolerance = 1e-5 * ranked.abs().max().item()
                if (ranked[TOPK - 1] - ranked[TOPK]).item() > tolerance:
                    candidates = [candidates[i] for i in order[:TOPK].tolist()]
                else:
                    assert selected is not None, f"top-k tie in row {row}; the module's selection is needed"
                    remote = sorted(key for key in selected[sample, row].tolist() if 0 <= key <= row - WINDOW)
                    left_out = sorted(set(candidates) - set(remote))
                    assert len(remote) == TOPK and set(remote) <= set(candidates), f"row {row}: {remote}"
                    worst_kept, best_dropped = scores[remote].min().item(), scores[left_out].max().item()
                    assert worst_kept >= best_dropped - tolerance, f"row {row}: kept {worst_kept} < {best_dropped}"
                    candidates = remote
            topk[sample, row, candidates] = True
    return topk, window


def _sink_softmax(logits: torch.Tensor, sink: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """``e^{s_hj} / (sum_{j in S(t)} e^{s_hj} + e^{sink_h})`` over ``allowed [b, t, s]``; ``logits [b, h, t, s]``."""
    logits = logits.masked_fill(~allowed[:, None], float("-inf"))
    sink = sink.view(1, -1, 1, 1)
    peak = torch.maximum(logits.amax(dim=-1, keepdim=True), sink).detach()
    weights = torch.exp(logits - peak)
    return weights / (weights.sum(dim=-1, keepdim=True) + torch.exp(sink - peak))


def _oracle_head_mix_and_gate(values: torch.Tensor, qr: torch.Tensor, params: dict[str, torch.Tensor]) -> torch.Tensor:
    """``v'_k = sum_h (I + V U^T)[k, h] v_h``, flattened head-major, times ``sigmoid(W_g c_q)``."""
    mix = torch.eye(HEADS, dtype=torch.float64) + params["vha_postmix_v"] @ params["vha_postmix_u"].T
    mixed = torch.einsum("kh,tbhd->tbkd", mix, values)
    gate = torch.sigmoid(qr @ params["linear_output_gate.weight"].T)
    return mixed.reshape(*mixed.shape[:2], HEADS * V_HEAD) * gate


def _oracle_indexer_loss(
    attention_logits: torch.Tensor, index_scores: torch.Tensor, topk: torch.Tensor, coeff: float
) -> torch.Tensor:
    """``coeff * sum_t KL(teacher_t || student_t) / (B * S)`` over ``TopK(t)``; no window, no sink."""
    batch, num_rows = topk.shape[:2]
    fill = -1e30  # finite, so rows without remote keys stay NaN-free; they are zeroed below
    teacher = torch.softmax(attention_logits.detach().masked_fill(~topk[:, None], fill), dim=-1) * topk[:, None]
    teacher = teacher.sum(dim=1)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    student = torch.softmax(index_scores.masked_fill(~topk, fill), dim=-1) * topk
    kl = teacher * (torch.log(teacher + 1e-10) - torch.log(student + 1e-10))
    return coeff * kl.sum() / (batch * num_rows)


@dataclass
class _CoreOracle:
    output: torch.Tensor
    loss: torch.Tensor
    allowed: torch.Tensor
    params: dict[str, torch.Tensor]
    inputs: dict[str, torch.Tensor]
    designed: dict[str, torch.Tensor]


def _oracle_core(
    core: Ernie55LatentMQACoreAttention,
    batch: _Batch,
    *,
    designed_indexer: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    rotary_base: float = ROTARY_BASE,
    coeff: float = COEFF,
) -> _CoreOracle:
    """fp64 D core: ``S(t)``, sink attention, V-up, VHA, q-latent gate, and the indexer loss.

    With ``designed_indexer`` (injected ``q``, ``k``, ``w``), their fp64 leaf copies replace the
    indexer projections and are returned in ``designed``.
    """
    params = _leaves(core)
    inputs = {
        name: getattr(batch, name).detach().double().requires_grad_(True) for name in ("query", "key", "qr", "up_v")
    }
    thd = batch.query.dim() == 3
    query = inputs["query"].unsqueeze(1) if thd else inputs["query"]
    key = inputs["key"] if thd else inputs["key"].squeeze(2)
    qr = inputs["qr"].unsqueeze(1) if thd else inputs["qr"]
    x = batch.x.detach().double()

    designed = {}
    if designed_indexer is None:
        index_scores = _oracle_index_scores(
            _strip(params, "indexer."),
            x,
            qr,
            _positions(batch.doc_start),
            rotary_base,
            module_x=batch.x,
            module_weights_proj=core.indexer.linear_weights_proj.weight,
        )
    else:
        designed = {
            name: tensor.detach().double().requires_grad_(True)
            for name, tensor in zip(("q", "k", "w"), designed_indexer)
        }
        index_scores = _designed_index_scores(designed["q"], designed["k"], designed["w"])
    topk, window = _oracle_index_sets(batch.doc_start, batch.batch_size, index_scores)
    allowed = topk | window

    logits = SCALE * torch.einsum("tbhd,sbd->bhts", query, key)
    probs = _sink_softmax(logits, params["attn_sink"], allowed)
    latent = torch.einsum("bhts,sbc->tbhc", probs, key[..., :KV_LORA])
    values = torch.einsum("tbhc,hdc->tbhd", latent, inputs["up_v"])
    output = _oracle_head_mix_and_gate(values, qr, params)
    loss = _oracle_indexer_loss(logits, index_scores, topk, coeff)
    return _CoreOracle(output, loss, allowed, params, inputs, designed)


@dataclass
class _AttentionOracle:
    output: torch.Tensor
    loss: torch.Tensor
    allowed: torch.Tensor
    params: dict[str, torch.Tensor]
    x: torch.Tensor


def _oracle_attention(
    attention: Ernie55LatentMQASelfAttention,
    x: torch.Tensor,
    doc_start: list[int],
    *,
    selected: torch.Tensor | None = None,
    rotary_base: float = ROTARY_BASE,
    coeff: float = COEFF,
) -> _AttentionOracle:
    """fp64 D layer in non-absorbed MLA form: ``k_h = [W_UK c_kv | rope(k_pe)]``, ``v_h = W_UV c_kv``.

    ``k_b_proj`` stores ``W_UK[h]^T`` (``[C, nope]``) and ``v_b_proj`` stores ``W_UV[h]``
    (``[v, C]``) per head. ``selected`` (the module's index lists) only resolves exact top-k ties.
    """
    params = _leaves(attention)
    x64 = x.detach().double().requires_grad_(True)
    seqlen, batch = x.shape[:2]
    positions = _positions(doc_start)

    c_q = _rms_norm(x64 @ params["linear_q_down_proj.weight"].T, params["q_layernorm.weight"])
    q = (c_q @ params["linear_q_up_proj.weight"].T).view(seqlen, batch, HEADS, NOPE + ROPE)
    q_nope, q_rope = q[..., :NOPE], _rope_adjacent_pairs(q[..., NOPE:], positions, rotary_base)
    kv = x64 @ params["linear_kv_down_proj.weight"].T
    c_kv = _rms_norm(kv[..., :KV_LORA], params["kv_layernorm.weight"])
    k_rope = _rope_adjacent_pairs(kv[..., KV_LORA:], positions, rotary_base)
    k_nope = torch.einsum("sbc,hcn->sbhn", c_kv, params["k_b_proj"].view(HEADS, KV_LORA, NOPE))
    values = torch.einsum("sbc,hdc->sbhd", c_kv, params["v_b_proj"].view(HEADS, V_HEAD, KV_LORA))
    logits = SCALE * (torch.einsum("tbhn,sbhn->bhts", q_nope, k_nope) + torch.einsum("tbhr,sbr->bhts", q_rope, k_rope))

    core_params = _strip(params, "core_attention.")
    index_scores = _oracle_index_scores(
        _strip(core_params, "indexer."),
        x64,
        c_q,
        positions,
        rotary_base,
        module_x=x,
        module_weights_proj=attention.core_attention.indexer.linear_weights_proj.weight,
    )
    topk, window = _oracle_index_sets(doc_start, batch, index_scores, selected=selected)
    allowed = topk | window
    probs = _sink_softmax(logits, core_params["attn_sink"], allowed)
    attended = torch.einsum("bhts,sbhd->tbhd", probs, values)
    output = _oracle_head_mix_and_gate(attended, c_q, core_params) @ params["linear_proj.weight"].T
    loss = _oracle_indexer_loss(logits, index_scores, topk, coeff)
    return _AttentionOracle(output, loss, allowed, params, x64)


# ---------------------------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------------------------


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, tol: float, what: str) -> None:
    """``max |actual - expected| <= tol * max |expected|``."""
    assert actual is not None, f"{what}: missing"
    assert tuple(actual.shape) == tuple(expected.shape), (
        f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    actual, expected = actual.detach().double(), expected.detach().double()
    assert bool(torch.isfinite(actual).all()), f"{what}: non-finite values"
    reference = expected.abs().max().item()
    error = (actual - expected).abs().max().item()
    assert error <= tol * reference, f"{what}: max |error| {error:.3e} vs max |expected| {reference:.3e} (tol {tol:g})"


def _assert_parameter_grads(
    module: torch.nn.Module, oracle_params: dict[str, torch.Tensor], *, skip_prefix: str | None = None
) -> None:
    for name, param in module.named_parameters():
        if skip_prefix is not None and name.startswith(skip_prefix):
            continue
        tol = next((value for suffix, value in LOOSE_GRADS.items() if name.endswith(suffix)), TOL)
        expected = oracle_params[name].grad
        assert expected is not None, f"the oracle does not use {name}"
        _assert_close(param.grad, expected, tol, f"grad of {name}")


def _logged_indexer_loss(layer_number: int) -> tuple[torch.Tensor, torch.Tensor]:
    tracker = dsa.DSAIndexerLossLoggingHelper.tracker
    return tracker["values"][layer_number - 1], tracker["raw_values"][layer_number - 1]


def _capture_index_lists(monkeypatch, core: Ernie55LatentMQACoreAttention) -> list[torch.Tensor]:
    """Record the ``[b, s, M]`` key index lists the core hands to its sparse attention."""
    captured = []
    sparse_attention = core._sparse_attention

    def capture(query: torch.Tensor, key: torch.Tensor, indices: torch.Tensor, **kwargs) -> torch.Tensor:
        captured.append(indices.detach().clone())
        return sparse_attention(query, key, indices, **kwargs)

    monkeypatch.setattr(core, "_sparse_attention", capture)
    return captured


def _assert_index_lists(indices: torch.Tensor, allowed: torch.Tensor) -> None:
    """Each row lists every key of ``S(t)`` exactly once; every other slot is ``-1``."""
    batch, num_rows = allowed.shape[:2]
    assert tuple(indices.shape[:2]) == (batch, num_rows)
    for sample in range(batch):
        for row in range(num_rows):
            entries = indices[sample, row].tolist()
            keys = [key for key in entries if key >= 0]
            assert all(key == -1 for key in entries if key < 0), f"row {row}: bad padding {entries}"
            assert len(keys) == len(set(keys)), f"row {row}: duplicated keys {entries}"
            expected = allowed[sample, row].nonzero().flatten().tolist()
            assert sorted(keys) == expected, f"sample {sample} row {row}: keys {sorted(keys)} != S(t) {expected}"


def _inject_indexer(monkeypatch, core: Ernie55LatentMQACoreAttention, seed: int, seqlen: int, batch: int):
    """Replace the indexer projections with designed, tie-free, differentiable outputs.

    ``q``, ``k`` are non-negative and ``w`` positive, so every score is a positive, continuous
    random number and the expected top-k is unambiguous (checked by the oracle).
    """
    generator = torch.Generator().manual_seed(seed)
    q_index = torch.rand(seqlen, batch, INDEXER_HEADS, INDEXER_DIM, generator=generator).requires_grad_(True)
    k_index = torch.rand(seqlen, batch, INDEXER_DIM, generator=generator).requires_grad_(True)
    w_index = (0.05 + 0.1 * torch.rand(seqlen, batch, INDEXER_HEADS, generator=generator)).requires_grad_(True)

    def forward_before_topk(x, qr, packed_seq_params=None):
        return q_index.clone(), k_index.clone(), w_index.clone()

    monkeypatch.setattr(core.indexer, "forward_before_topk", forward_before_topk)
    return q_index, k_index, w_index


# ---------------------------------------------------------------------------------------------
# Core attention against the oracle
# ---------------------------------------------------------------------------------------------


class TestCoreDenseRegime:
    """Every document is at most W + K long, so S(t) is the causal document prefix."""

    @pytest.mark.parametrize(
        "layout",
        [
            # Lengths 1, W-1, W, W+1 and W+K, and 3 tail rows past cu_seqlens[-1] that extend the
            # last document (their positions stay below max_seqlen, so RoPE does not clamp them).
            dict(doc_lens=[1, 2, 3, 4, 9, 5], tail_rows=3),
            # Padded boundaries take precedence over the unpadded ones.
            dict(doc_lens=[3, 7, 2], padded_lens=[4, 9, 4]),
            # SBHD: one document per sample.
            dict(seqlen=9, batch=2),
        ],
        ids=["thd-tail-rows", "thd-padded", "sbhd-b2"],
    )
    def test_forward_loss_and_gradients_match_oracle(self, pg_collection, monkeypatch, layout):
        core = _build_core(_config(), pg_collection, seed=1)
        batch = _core_batch(seed=2, **layout)
        index_lists = _capture_index_lists(monkeypatch, core)

        output = _run_core(core, batch)
        oracle = _oracle_core(core, batch)

        # S(t) is every causal key of the row's document: the indexer cannot matter here.
        for row, start in enumerate(batch.doc_start):
            assert oracle.allowed[:, row].nonzero()[:, 1].unique().tolist() == list(range(start, row + 1))
        _assert_index_lists(index_lists[0], oracle.allowed)
        _assert_close(output, oracle.output, TOL, "output")
        loss, raw_loss = _logged_indexer_loss(layer_number=1)
        _assert_close(loss, oracle.loss, TOL, "indexer loss")
        _assert_close(raw_loss, oracle.loss / COEFF, TOL, "raw indexer loss")
        assert oracle.loss.item() > 0

        grad_output = torch.randn(output.shape, generator=torch.Generator().manual_seed(3))
        output.backward(grad_output)
        ((oracle.output * grad_output.double()).sum() + oracle.loss).backward()
        _assert_parameter_grads(core, oracle.params)
        for name, value in oracle.inputs.items():
            _assert_close(getattr(batch, name).grad, value.grad, TOL, f"grad of {name}")
        # x only feeds the indexer, which sees it detached.
        assert batch.x.grad is None

    def test_bf16_forward_matches_oracle(self, pg_collection):
        """The bf16 contracts (bf16 VHA mix and gate) stay within bf16 accuracy of the fp64 math."""
        core = _build_core(_config(params_dtype=torch.bfloat16), pg_collection, seed=4)
        batch = _core_batch(seed=5, doc_lens=[4, 9, 2], dtype=torch.bfloat16)
        output = _run_core(core, batch)
        assert output.dtype == torch.bfloat16
        oracle = _oracle_core(core, batch)
        _assert_close(output, oracle.output, 3e-2, "bf16 output")
        output.float().sum().backward()
        for name, param in core.named_parameters():
            assert param.grad is not None and bool(torch.isfinite(param.grad).all()), name


class TestCoreSelectionRegime:
    """Documents longer than W + K: the indexer chooses K remote keys per row."""

    @pytest.mark.parametrize(
        "layout",
        [
            dict(doc_lens=[2, 15, 3, 20], tail_rows=2),
            dict(doc_lens=[2, 13, 3, 18], padded_lens=[4, 16, 4, 20]),
            dict(seqlen=16, batch=3),
        ],
        ids=["thd-tail-rows", "thd-padded", "sbhd-b3"],
    )
    def test_injected_indexer_selection_matches_oracle(self, pg_collection, monkeypatch, layout):
        core = _build_core(_config(), pg_collection, seed=11)
        batch = _core_batch(seed=12, **layout)
        num_rows = len(batch.doc_start)
        designed = _inject_indexer(monkeypatch, core, seed=13, seqlen=num_rows, batch=batch.batch_size)
        index_lists = _capture_index_lists(monkeypatch, core)

        output = _run_core(core, batch)
        oracle = _oracle_core(core, batch, designed_indexer=designed)

        # The regime is really exercised: some rows drop remote keys, and the first W rows of every
        # document (no remote candidates) attend to their window only.
        positions = _positions(batch.doc_start)
        assert bool((oracle.allowed.sum(-1) < positions + 1).any())
        for row, start in enumerate(batch.doc_start):
            if positions[row] < WINDOW:
                assert oracle.allowed[:, row].sum(-1).tolist() == [row - start + 1] * batch.batch_size
        _assert_index_lists(index_lists[0], oracle.allowed)
        _assert_close(output, oracle.output, TOL, "output")
        loss, _ = _logged_indexer_loss(layer_number=1)
        _assert_close(loss, oracle.loss, TOL, "indexer loss")

        grad_output = torch.randn(output.shape, generator=torch.Generator().manual_seed(14))
        output.backward(grad_output)
        ((oracle.output * grad_output.double()).sum() + oracle.loss).backward()
        _assert_parameter_grads(core, oracle.params, skip_prefix="indexer.")
        for name, value in oracle.inputs.items():
            _assert_close(getattr(batch, name).grad, value.grad, TOL, f"grad of {name}")
        # The injected indexer outputs learn from the KL alone.
        for (name, expected), actual in zip(oracle.designed.items(), designed):
            assert actual.grad.abs().max().item() > 0, name
            _assert_close(actual.grad, expected.grad, TOL, f"grad of the indexer {name}")

    @pytest.mark.parametrize("layout", [dict(doc_lens=[1, 3, 2, 3]), dict(seqlen=3, batch=2)], ids=["thd", "sbhd-b2"])
    def test_rows_without_remote_candidates(self, pg_collection, monkeypatch, layout):
        """No row has a remote key: window-only attention, an exactly-zero KL and zero indexer gradients."""
        core = _build_core(_config(), pg_collection, seed=21)
        batch = _core_batch(seed=22, **layout)
        index_lists = _capture_index_lists(monkeypatch, core)

        output = _run_core(core, batch)
        oracle = _oracle_core(core, batch)
        _assert_index_lists(index_lists[0], oracle.allowed)
        _assert_close(output, oracle.output, TOL, "output")
        loss, _ = _logged_indexer_loss(layer_number=1)
        assert loss.item() == 0.0

        output.backward(torch.ones_like(output))
        for name, param in core.indexer.named_parameters():
            assert param.grad is not None and bool((param.grad == 0).all()), name


class TestCoreSinkGateAndHeadMix:
    @pytest.mark.parametrize("sink", ["none", "random", "dominant"])
    def test_sink_enters_the_denominator_only(self, pg_collection, sink):
        core = _build_core(_config(), pg_collection, seed=31)
        with torch.no_grad():
            if sink == "none":
                core.attn_sink.fill_(float("-inf"))
            elif sink == "dominant":
                core.attn_sink.fill_(30.0)
        batch = _core_batch(seed=32, doc_lens=[6, 9])
        output = _run_core(core, batch)
        _assert_close(output, _oracle_core(core, batch).output, TOL, f"output with sink={sink}")
        if sink == "dominant":
            # Nearly all probability mass goes to the sink, which carries no value.
            assert output.abs().max().item() < 1e-9
        if sink == "random":
            with torch.no_grad():
                core.attn_sink.fill_(float("-inf"))
            without_sink = _run_core(core, batch)
            assert (output - without_sink).abs().max().item() > 1e-2

    def test_vha_orientation(self, pg_collection):
        """``U = e_0``, ``V = e_1`` (rank one) gives ``v'_1 = v_1 + v_0`` and leaves every other head alone."""
        core = _build_core(_config(), pg_collection, seed=41)
        batch = _core_batch(seed=42, doc_lens=[5, 8])
        with torch.no_grad():
            core.linear_output_gate.weight.zero_()  # sigmoid(0) = 1/2 on every channel
            core.vha_postmix_u.zero_()
            core.vha_postmix_v.zero_()
            plain = _run_core(core, batch).view(-1, HEADS, V_HEAD)
            core.vha_postmix_u[0, 0] = 1.0
            core.vha_postmix_v[1, 0] = 1.0
            mixed = _run_core(core, batch).view(-1, HEADS, V_HEAD)
        torch.testing.assert_close(mixed[:, 1], plain[:, 1] + plain[:, 0])
        torch.testing.assert_close(mixed[:, [0, 2, 3]], plain[:, [0, 2, 3]])

    def test_output_and_gate_are_head_major(self, pg_collection):
        """Value ``(h, d)`` and its gate logit both live at channel ``h * v_head_dim + d``."""
        core = _build_core(_config(), pg_collection, seed=51)
        batch = _core_batch(seed=52, doc_lens=[5, 8])
        head, dim = 2, 5
        channel = head * V_HEAD + dim
        with torch.no_grad():
            core.vha_postmix_v.zero_()
            core.linear_output_gate.weight.zero_()
            batch.up_v.zero_()
            batch.up_v[head, dim] = torch.randn(KV_LORA, generator=torch.Generator().manual_seed(53))
            ungated = _run_core(core, batch)
            assert bool((ungated[..., channel] != 0).all())
            assert bool((ungated[..., torch.arange(HEADS * V_HEAD) != channel] == 0).all())
            core.linear_output_gate.weight[channel] = torch.randn(Q_LORA, generator=torch.Generator().manual_seed(54))
            gated = _run_core(core, batch)
        gate = torch.sigmoid(batch.qr.detach() @ core.linear_output_gate.weight[channel].detach())
        torch.testing.assert_close(gated[..., channel], 2 * ungated[..., channel] * gate.view_as(gated[..., channel]))
        torch.testing.assert_close(gated[..., :channel], ungated[..., :channel])
        torch.testing.assert_close(gated[..., channel + 1 :], ungated[..., channel + 1 :])

    def test_gate_reads_the_q_latent_not_the_layer_input(self, pg_collection):
        """In the dense regime ``x`` only reaches the (irrelevant) indexer; ``qr`` drives the gate."""
        core = _build_core(_config(), pg_collection, seed=61)
        batch = _core_batch(seed=62, doc_lens=[6, 9])
        with torch.no_grad():
            reference = _run_core(core, batch)
            batch.x.add_(torch.randn(batch.x.shape, generator=torch.Generator().manual_seed(63)))
            # The new indexer scores reorder the (identical) remote key set, which only changes
            # the fp32 summation order of the attention.
            _assert_close(_run_core(core, batch), reference, 1e-6, "output after perturbing x")
            batch.qr.add_(torch.randn(batch.qr.shape, generator=torch.Generator().manual_seed(64)))
            perturbed = _run_core(core, batch)
        assert (perturbed - reference).abs().max().item() > 1e-2
        _assert_close(perturbed, _oracle_core(core, batch).output, TOL, "output after perturbing qr")


# ---------------------------------------------------------------------------------------------
# Indexer loss
# ---------------------------------------------------------------------------------------------


class TestIndexerLoss:
    def test_loss_gradient_reaches_only_the_indexer(self, pg_collection):
        core = _build_core(_config(), pg_collection, seed=71)
        batch = _core_batch(seed=72, doc_lens=[4, 9, 7])
        output = _run_core(core, batch)
        # A zero upstream gradient leaves the indexer loss as the only gradient source.
        output.backward(torch.zeros_like(output))
        oracle = _oracle_core(core, batch)
        oracle.loss.backward()
        for name, param in core.named_parameters():
            if name.startswith("indexer."):
                assert param.grad.abs().max().item() > 0, name
                tol = BF16_GRAD_TOL if name.endswith("linear_weights_proj.weight") else TOL
                _assert_close(param.grad, oracle.params[name].grad, tol, f"grad of {name}")
            else:
                assert param.grad is not None and bool((param.grad == 0).all()), name
        for name in ("query", "key", "qr", "up_v"):
            assert bool((getattr(batch, name).grad == 0).all()), name
        assert batch.x.grad is None

    def test_non_indexer_gradients_do_not_depend_on_the_loss(self, pg_collection):
        with_loss = _build_core(_config(), pg_collection, seed=81)
        without_loss = _build_core(_config(dsa_indexer_loss_coeff=0.0), pg_collection, seed=81)
        without_loss.load_state_dict(with_loss.state_dict())
        assert all(not param.requires_grad for param in without_loss.indexer.parameters())

        grads = []
        for core in (with_loss, without_loss):
            batch = _core_batch(seed=82, doc_lens=[3, 9, 9])
            output = _run_core(core, batch)
            output.backward(torch.randn(output.shape, generator=torch.Generator().manual_seed(83)))
            named = {name: param.grad for name, param in core.named_parameters() if not name.startswith("indexer.")}
            named.update({name: getattr(batch, name).grad for name in ("query", "key", "qr", "up_v")})
            grads.append(named)
        assert grads[0].keys() == grads[1].keys()
        for name in grads[0]:
            assert torch.equal(grads[0][name], grads[1][name]), name
        assert all(param.grad is None for param in without_loss.indexer.parameters())

    def test_indexer_gradients_follow_the_loss_scale(self, pg_collection):
        core = _build_core(_config(), pg_collection, seed=91)
        indexer_grads = []
        for scale in (1.0, 4.0):
            dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(scale))
            core.zero_grad(set_to_none=True)
            output = _run_core(core, _core_batch(seed=92, doc_lens=[9, 8]))
            output.backward(torch.zeros_like(output))
            indexer_grads.append({name: param.grad.clone() for name, param in core.indexer.named_parameters()})
        for name, grad in indexer_grads[0].items():
            # A power-of-two scale commutes with every rounding step, so the scaling is exact.
            assert torch.equal(indexer_grads[1][name], 4.0 * grad), name

    def test_no_loss_in_eval_or_without_grad(self, pg_collection):
        core = _build_core(_config(), pg_collection, seed=101)
        batch = _core_batch(seed=102, doc_lens=[7, 9])
        trained = _run_core(core, batch).detach()
        assert "values" in dsa.DSAIndexerLossLoggingHelper.tracker
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()
        # Autograd and no-grad can select different fp32 reduction paths. The loss
        # must not change the attention result beyond the reference-path tolerance.
        with torch.no_grad():
            _assert_close(_run_core(core, batch), trained, TOL, "output without grad")
        assert "values" not in dsa.DSAIndexerLossLoggingHelper.tracker
        core.eval()
        evaluated = _run_core(core, batch)
        _assert_close(evaluated, trained, TOL, "output in eval")
        assert "values" not in dsa.DSAIndexerLossLoggingHelper.tracker
        evaluated.sum().backward()
        assert all(param.grad is None for param in core.indexer.parameters())

    @pytest.mark.parametrize("per_token", [False, True], ids=["per-row", "per-token"])
    def test_reduction_follows_the_loss_mode(self, pg_collection, monkeypatch, caplog, per_token):
        """Per-row loss: ``coeff * sum KL / (b * s)`` (reference parity); per-token loss: the raw sum, warned."""
        monkeypatch.setattr(latent_mqa_core, "_WARNED", set())
        with caplog.at_level(logging.WARNING, logger=latent_mqa_core.logger.name):
            core = _build_core(_config(calculate_per_token_loss=per_token), pg_collection, seed=105)
        assert ("per-token convention" in caplog.text) == per_token
        batch = _core_batch(seed=106, doc_lens=[4, 13, 9], tail_rows=2)
        _run_core(core, batch)
        oracle = _oracle_core(core, batch)
        loss, raw_loss = _logged_indexer_loss(layer_number=1)
        rows = len(batch.doc_start) * batch.batch_size
        expected = oracle.loss * rows if per_token else oracle.loss
        _assert_close(loss, expected, TOL, "logged indexer loss")
        _assert_close(raw_loss, expected / COEFF, TOL, "logged raw indexer loss")


class TestReferencePathWiring:
    def test_kernels_never_see_empty_candidate_rows(self, pg_collection, monkeypatch):
        """The top-k selector and the loss get ``ends > starts`` on every row, including the first W."""
        core = _build_core(_config(), pg_collection, seed=111)
        batch = _core_batch(seed=112, doc_lens=[1, 2, 12, 4])
        seen = []
        topk_naive, indexer_loss = dsa.fused_qk_topk_naive, dsa.compute_dsa_indexer_loss

        def recording_topk_naive(*args, **kwargs):
            seen.append(("fused_qk_topk_naive", kwargs["varlen_starts"], kwargs["varlen_ends"]))
            return topk_naive(*args, **kwargs)

        def recording_indexer_loss(**kwargs):
            seen.append(("compute_dsa_indexer_loss", kwargs["varlen_starts"], kwargs["varlen_ends"]))
            return indexer_loss(**kwargs)

        monkeypatch.setattr(dsa, "fused_qk_topk_naive", recording_topk_naive)
        monkeypatch.setattr(dsa, "compute_dsa_indexer_loss", recording_indexer_loss)
        _run_core(core, batch)

        doc_start = torch.tensor(batch.doc_start)
        true_starts, true_ends = ops.ernie55_selection_bounds(doc_start, WINDOW)
        assert bool((true_ends == true_starts).any()), "the pack must have rows without remote candidates"
        assert [name for name, *_ in seen] == ["fused_qk_topk_naive", "compute_dsa_indexer_loss"]
        for name, starts, ends in seen:
            assert torch.equal(starts, true_starts), name
            assert bool((ends > starts).all()), f"{name} received an empty candidate row"

    def test_attention_gathers_the_key_in_fp32(self, pg_collection, monkeypatch):
        """With bf16 activations, Megatron-Core's reference attention gets an fp32 latent key.

        Its gather backward adds every query row's key gradient in the key's dtype. On GPU a bf16 key
        lost about 4.5% (relative L2) of dKV against float64 at the production geometry; CPU kernels
        do not show the loss, so the GPU parity tests measure it and this test pins the contract.
        """
        core = _build_core(_config(params_dtype=torch.bfloat16), pg_collection, seed=151)
        seen = []
        unfused_compressed_sparse_attn = csa.unfused_compressed_sparse_attn

        def recording(query, kv_full, *args, **kwargs):
            seen.append((query.dtype, kv_full.dtype))
            return unfused_compressed_sparse_attn(query, kv_full, *args, **kwargs)

        monkeypatch.setattr(csa, "unfused_compressed_sparse_attn", recording)
        batch = _core_batch(seed=152, doc_lens=[4, 9, 2], dtype=torch.bfloat16)
        output = _run_core(core, batch)
        assert seen == [(torch.bfloat16, torch.float32)]
        assert output.dtype == torch.bfloat16
        output.float().sum().backward()
        assert batch.key.grad.dtype == torch.bfloat16


# ---------------------------------------------------------------------------------------------
# Fused-path wiring on CPU: Megatron-Core's fused entry points replaced by dense stand-ins
# ---------------------------------------------------------------------------------------------


def _valid_slots(topk_idxs: torch.Tensor, topk_length: torch.Tensor | None) -> torch.Tensor:
    """Slots a sparse kernel reads: non-negative and, with ``topk_length``, inside each row's prefix."""
    valid = topk_idxs >= 0
    if topk_length is not None:
        positions = torch.arange(topk_idxs.size(-1), device=topk_idxs.device).view(1, -1)
        valid = valid & (positions < topk_length.view(-1, 1))
    return valid


def _dense_sparse_attention(q, kv, attn_sink, topk_idxs, topk_length, softmax_scale, d_v):
    """``(out [rows, H, d_v], lse [rows, H])``; the sink is only in the denominator and not in ``lse``."""
    valid = _valid_slots(topk_idxs, topk_length)
    keys = kv[topk_idxs.clamp_min(0).long()]
    logits = (torch.einsum("rhd,rmd->rhm", q, keys) * softmax_scale).masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - torch.logaddexp(lse, attn_sink.view(1, -1).to(lse.dtype)).unsqueeze(-1))
    return torch.einsum("rhm,rmd->rhd", probs, keys[..., :d_v]), lse


def _dense_flash_mla_forward(q, kv, topk_idxs, softmax_scale, d_v=512, attn_sink=None, topk_length=None, **kwargs):
    """CPU stand-in for ``_csa_fwd_flash_mla``: ``(out, lse without the sink, None)``."""
    with torch.no_grad():
        out, lse = _dense_sparse_attention(q, kv, attn_sink, topk_idxs, topk_length, softmax_scale, d_v)
    return out.to(q.dtype), lse, None


def _dense_sparse_attention_backward(
    q, kv, out, d_out, lse, attn_sink, topk_idxs, softmax_scale, topk_length, **kwargs
):
    """CPU stand-in for ``_csa_sparse_attention_backward``.

    Softmax backward with ``P = exp(S - logaddexp(lse, sink))`` and ``D = rowsum(dO * O)``. Slots past
    ``topk_length`` are ignored even when they hold non-negative placeholders.
    """
    d_v = out.size(-1)
    lse_total = torch.logaddexp(lse, attn_sink.view(1, -1).to(lse.dtype))
    valid = _valid_slots(topk_idxs, topk_length)
    keys = kv[topk_idxs.clamp_min(0).long()]
    logits = torch.einsum("rhd,rmd->rhm", q, keys) * softmax_scale
    probs = torch.exp(logits.masked_fill(~valid.unsqueeze(1), float("-inf")) - lse_total.unsqueeze(-1))
    delta = (d_out * out).sum(-1, keepdim=True)
    d_logits = probs * (torch.einsum("rhd,rmd->rhm", d_out, keys[..., :d_v]) - delta)
    d_q = torch.einsum("rhm,rmd->rhd", d_logits, keys) * softmax_scale
    d_keys = torch.einsum("rhm,rhd->rmd", d_logits, q) * softmax_scale
    d_keys[..., :d_v] += torch.einsum("rhm,rhd->rmd", probs, d_out)
    d_keys = d_keys * valid.unsqueeze(-1)
    d_kv = torch.zeros_like(kv).index_add_(0, topk_idxs.clamp_min(0).long().flatten(), d_keys.flatten(0, 1))
    d_sink = -(torch.exp(attn_sink.view(1, -1) - lse_total) * delta.squeeze(-1)).sum(0)
    return d_q, d_kv, d_sink


def _dense_csa_sparse_attn(query, kv, attn_sink, topk_idxs, softmax_scale, topk_length=None, **kwargs):
    """CPU stand-in for ``csa_sparse_attn`` (differentiable, ``d_v = KV_LORA``): ``[s, b, H * d_v]``."""
    seqlen, batch, heads, dim = query.shape
    out, _ = _dense_sparse_attention(
        query.reshape(seqlen * batch, heads, dim),
        kv.reshape(-1, dim),
        attn_sink,
        topk_idxs,
        topk_length,
        softmax_scale,
        KV_LORA,
    )
    return out.reshape(seqlen, batch, heads * KV_LORA)


def _cudnn_score_scale() -> float:
    """The score scale the cuDNN hooks read at call time (the core sets it around each call)."""
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    return dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE


def _dense_indexer_forward_wrapper(q, k, weights, *, ratio, sm_scale, q_causal_offsets=None):
    """CPU stand-in for cuDNN's ``indexer_forward_wrapper``: ``{"scores": [b, sq, sk]}``.

    ``sm_scale * sum_h relu(q_h . k) * w_h`` in fp32, ``-inf`` for keys after each query row's causal
    position (the row index plus ``q_causal_offsets``).
    """
    assert ratio == 1 and q.ndim == 4 and k.ndim == 4 and k.size(2) == 1 and weights.ndim == 3
    scores = torch.einsum("bshd,btd->bsht", q.float(), k[:, :, 0].float()).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2) * sm_scale
    offsets = torch.zeros(q.size(0), dtype=torch.int64) if q_causal_offsets is None else q_causal_offsets.long()
    rows = torch.arange(q.size(1)).view(1, -1, 1) + offsets.view(-1, 1, 1)
    return {"scores": scores.masked_fill(torch.arange(k.size(1)).view(1, 1, -1) > rows, float("-inf"))}


def _reference_topk_hook(config, q, k, weights, index_topk, starts, ends, block_size, use_relu=True, **kwargs):
    """CPU stand-in for the cuDNN ``run_fused_qk_topk`` hook: ``(topk, topk_length)``."""
    _, topk = dsa.fused_qk_topk_naive(
        q, k, weights, index_topk, varlen_starts=starts, varlen_ends=ends, use_relu=use_relu
    )
    return topk.int(), (topk >= 0).sum(-1).int()


def _reference_topk_with_loss_hook(
    config,
    q,
    k,
    weights,
    index_topk,
    starts,
    ends,
    block_size,
    *,
    query,
    key,
    softmax_scale,
    loss_coeff,
    pg_collection,
    query_valid_rows=None,
    calculate_per_token_loss=False,
    use_relu=True,
    **kwargs,
):
    """CPU stand-in for the cuDNN ``run_fused_qk_topk_with_loss`` hook: ``(topk, topk_length, loss)``.

    Like the cuDNN hook, it scales the scores by ``dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE`` (read at call time).
    """
    index_scores, topk = dsa.fused_qk_topk_naive(
        q, k, weights, index_topk, varlen_starts=starts, varlen_ends=ends, use_relu=use_relu
    )
    loss = dsa.compute_dsa_indexer_loss(
        index_scores=index_scores * _cudnn_score_scale(),
        topk_indices=topk,
        query=query,
        key=key.expand(-1, -1, query.size(2), -1),
        softmax_scale=softmax_scale,
        loss_coeff=loss_coeff,
        sparse_loss=True,
        pg_collection=pg_collection,
        varlen_starts=starts,
        varlen_ends=ends,
        query_valid_rows=query_valid_rows,
        calculate_per_token_loss=calculate_per_token_loss,
    )
    return topk.int(), (topk >= 0).sum(-1).int(), loss


def _install_fused_stand_ins(monkeypatch) -> list[tuple[str, dict]]:
    """Replace Megatron-Core's fused entry points with dense CPU stand-ins; return their recorded calls.

    Every call is bound against the real entry point's signature on the Megatron-Core pin under
    test, so an argument the real function would reject fails here.
    """
    calls = []

    def install(module, name, stand_in):
        real = getattr(module, name)
        signature = inspect.signature(real)

        @functools.wraps(real)
        def recording(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append((name, dict(bound.arguments)))
            return stand_in(*args, **kwargs)

        monkeypatch.setattr(module, name, recording)

    install(dsa_kernels, "run_fused_qk_topk", _reference_topk_hook)
    install(dsa_kernels, "run_fused_qk_topk_with_loss", _reference_topk_with_loss_hook)
    install(fused_sparse_attention, "_csa_fwd_flash_mla", _dense_flash_mla_forward)
    install(fused_sparse_attention, "_csa_sparse_attention_backward", _dense_sparse_attention_backward)
    install(fused_sparse_attention, "csa_sparse_attn", _dense_csa_sparse_attn)
    if hasattr(fused_sparse_attention, "_deterministic_sparse_bwd_kwargs"):
        install(fused_sparse_attention, "_deterministic_sparse_bwd_kwargs", lambda *args, **kwargs: {})

    # cuDNN's indexer score call of the CP-1 selection (recorded with the score scale the hooks would read).
    def scorer(*args, **kwargs):
        calls.append(("indexer_forward_wrapper", dict(kwargs, q=args[0], hook_scale=_cudnn_score_scale())))
        return _dense_indexer_forward_wrapper(*args, **kwargs)

    monkeypatch.setattr(latent_mqa_core, "_reference_indexer_scorer", lambda q: scorer)
    # The fused attention otherwise insists on bf16 CUDA tensors.
    monkeypatch.setattr(latent_mqa_core, "_check_fused_attention_inputs", lambda query, key: None)
    return calls


def _with_cudnn_backend(config: Ernie55TransformerConfig) -> Ernie55TransformerConfig:
    """Copy of a finalized config with ``dsa_kernel_backend="cudnn"`` (finalize itself requires CUDA for it)."""
    if not hasattr(config, "dsa_kernel_backend"):
        raise ValueError(f"{type(config).__name__} has no field 'dsa_kernel_backend'")
    cudnn_config = copy.copy(config)
    cudnn_config.dsa_kernel_backend = "cudnn"
    return cudnn_config


def _build_fused_core(monkeypatch, pg_collection, *, seed: int, **config_overrides) -> Ernie55LatentMQACoreAttention:
    """A core on the fused path (``dsa_kernel_backend="cudnn"``) at the toy geometry FlashMLA would reject."""
    monkeypatch.setattr(latent_mqa_core, "_check_fused_geometry", lambda config: None)
    core = _build_core(_with_cudnn_backend(_config(**config_overrides)), pg_collection, seed=seed)
    assert core.use_fused
    return core


def _run_and_backward(core: Ernie55LatentMQACoreAttention, batch: _Batch, grad_seed: int):
    """Forward and backward; returns the output, the logged loss and every parameter and input gradient."""
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    output = _run_core(core, batch)
    loss, _ = _logged_indexer_loss(layer_number=1)
    output.backward(torch.randn(output.shape, generator=torch.Generator().manual_seed(grad_seed)))
    grads = {name: param.grad for name, param in core.named_parameters()}
    grads.update({f"input.{name}": getattr(batch, name).grad for name in ("query", "key", "qr", "up_v")})
    return output.detach(), loss.detach().clone(), grads


class TestFusedPathWiring:
    """The fused path's plumbing, checked on CPU against the reference path.

    Dense stand-ins replace the cuDNN hooks, FlashMLA and the cuDNN backward; the kernels themselves
    are covered by the GPU tests.
    """

    @pytest.mark.parametrize("native_sink_grad", [False, True], ids=["sink-grad-fallback", "native-sink-grad"])
    @pytest.mark.parametrize(
        "layout",
        [dict(doc_lens=[1, 2, 15, 3, 20]), dict(doc_lens=[1, 2, 15, 3, 20], tail_rows=2), dict(seqlen=16, batch=2)],
        ids=["thd", "thd-tail-rows", "sbhd-b2"],
    )
    def test_fused_dispatch_matches_reference(self, pg_collection, monkeypatch, layout, native_sink_grad):
        reference = _build_core(_config(), pg_collection, seed=121)
        fused = _build_fused_core(monkeypatch, pg_collection, seed=121, ernie_native_sink_grad=native_sink_grad)
        calls = _install_fused_stand_ins(monkeypatch)
        expected_output, expected_loss, expected_grads = _run_and_backward(
            reference, _core_batch(seed=122, **layout), grad_seed=123
        )
        batch = _core_batch(seed=122, **layout)
        output, loss, grads = _run_and_backward(fused, batch, grad_seed=123)

        _assert_close(output, expected_output, TOL, "fused-path output")
        _assert_close(loss, expected_loss, TOL, "fused-path indexer loss")
        assert grads.keys() == expected_grads.keys()
        for name, expected in expected_grads.items():
            _assert_close(grads[name], expected, TOL, f"fused-path grad of {name}")

        names = [name for name, _ in calls]
        # The top-k hook: non-empty kernel bounds, detached teacher inputs, all-rows KL; one
        # call per sample, because Megatron-Core's batched loss hook drops the later samples' gradients.
        # It runs with the reference's score convention: weights times d^0.5, scale d^-0.5.
        hook_calls = [arguments for name, arguments in calls if name == "run_fused_qk_topk_with_loss"]
        assert "run_fused_qk_topk" not in names
        assert len(hook_calls) == batch.batch_size
        true_starts, true_ends = ops.ernie55_selection_bounds(torch.tensor(batch.doc_start), WINDOW)
        for hook_args in hook_calls:
            assert hook_args["q"].size(1) == hook_args["query"].size(1) == hook_args["key"].size(1) == 1
            assert bool((hook_args["ends"] > hook_args["starts"]).all())
            assert torch.equal(hook_args["starts"], true_starts) and bool((true_ends == true_starts).any())
            assert hook_args["query_valid_rows"] is None and hook_args["calculate_per_token_loss"] is False
            assert hook_args["use_relu"] is True and hook_args["index_topk"] == TOPK
            assert hook_args["softmax_scale"] == SCALE and hook_args["loss_coeff"] == COEFF
            assert not hook_args["query"].requires_grad and not hook_args["key"].requires_grad
            # Packed THD rows request PACKED_THD_GENERAL at CP 1 too, unless tail rows past cu_seqlens[-1] would
            # fall outside the plan's documents; SBHD batches have no packed metadata.
            packed = "doc_lens" in layout and not layout.get("tail_rows")
            assert hook_args["use_local_indexer_varlen"] is packed
            assert (hook_args["packed_seq_params"] is not None) is packed
        # The attention's top-k: the reference's cuDNN score call per 512-row chunk with the reference's scale.
        scorer_calls = [arguments for name, arguments in calls if name == "indexer_forward_wrapper"]
        assert len(scorer_calls) == math.ceil(len(batch.doc_start) / 512)
        for scorer_args in scorer_calls:
            assert scorer_args["ratio"] == 1 and scorer_args["sm_scale"] == INDEXER_DIM**-0.5
            assert scorer_args["hook_scale"] == 1.0, "the hooks' scale is only overridden around hook calls"

        if native_sink_grad:
            assert "_csa_fwd_flash_mla" not in names and "_csa_sparse_attention_backward" not in names
            (attn_args,) = [arguments for name, arguments in calls if name == "csa_sparse_attn"]
            # The reference's [top-k | window] rows without their all-invalid 128-key blocks.
            topk_length, topk_idxs = attn_args["topk_length"], attn_args["topk_idxs"]
            assert topk_length is not None and bool((topk_length % 128 == 0).all())
            assert topk_idxs.size(-1) == TOPK + WINDOW + (-(TOPK + WINDOW) % 128)
            assert not bool(((topk_idxs >= 0) & ~_valid_slots(topk_idxs, topk_length)).any()), "keys past the length"
            if "deterministic" in attn_args:
                assert attn_args["deterministic"] is False
            return
        # The analytic sink-gradient fallback (ernie_native_sink_grad=False) with Megatron-Core DSA's
        # d_qk = 576 index format.
        assert "csa_sparse_attn" not in names
        (forward_args,) = [arguments for name, arguments in calls if name == "_csa_fwd_flash_mla"]
        (backward_args,) = [arguments for name, arguments in calls if name == "_csa_sparse_attention_backward"]
        topk_length = forward_args["topk_length"]
        assert forward_args["topk_idxs"].size(-1) % 512 == 0 and forward_args["d_v"] == KV_LORA
        assert torch.equal(backward_args["topk_length"], topk_length)
        slots = torch.arange(backward_args["topk_idxs"].size(-1)).view(1, -1)
        beyond = slots >= topk_length.view(-1, 1)
        assert bool((backward_args["topk_idxs"][beyond] >= 0).all()), "ignored slots need non-negative placeholders"
        within = forward_args["topk_idxs"][:, : backward_args["topk_idxs"].size(-1)]
        assert torch.equal(backward_args["topk_idxs"][~beyond], within[~beyond]), "holes inside a row stay -1"
        assert bool(torch.isneginf(backward_args["attn_sink"]).all()), "the cuDNN backward runs without the sink"
        if hasattr(fused_sparse_attention, "_deterministic_sparse_bwd_kwargs"):
            (deterministic_args,) = [args for name, args in calls if name == "_deterministic_sparse_bwd_kwargs"]
            assert deterministic_args["requested"] is False

    @pytest.mark.parametrize("per_token", [False, True], ids=["per-row-loss", "per-token-loss"])
    def test_batched_loss_hook_runs_per_sample(self, pg_collection, monkeypatch, per_token):
        """SBHD ``b = 3``: one ``b = 1`` loss-hook call per sample, combined like the batched loss.

        The per-row loss is the mean of the per-sample losses and the per-token loss their sum; the
        output, the logged KL and every gradient match the batched reference path.
        """
        layout = dict(seqlen=16, batch=3)
        reference = _build_core(_config(calculate_per_token_loss=per_token), pg_collection, seed=141)
        fused = _build_fused_core(monkeypatch, pg_collection, seed=141, calculate_per_token_loss=per_token)
        calls = _install_fused_stand_ins(monkeypatch)
        expected_output, expected_loss, expected_grads = _run_and_backward(
            reference, _core_batch(seed=142, **layout), grad_seed=143
        )
        batch = _core_batch(seed=142, **layout)
        output, loss, grads = _run_and_backward(fused, batch, grad_seed=143)

        hook_calls = [arguments for name, arguments in calls if name == "run_fused_qk_topk_with_loss"]
        assert len(hook_calls) == batch.batch_size
        for sample, hook_args in enumerate(hook_calls):
            assert hook_args["calculate_per_token_loss"] is per_token
            assert torch.equal(hook_args["query"], batch.query[:, sample : sample + 1].detach())
            assert torch.equal(hook_args["key"], batch.key[:, sample : sample + 1].detach())
            assert hook_args["q"].shape[:2] == hook_args["k"].shape[:2] == hook_args["weights"].shape[:2] == (16, 1)
        _assert_close(output, expected_output, TOL, "fused-path output")
        _assert_close(loss, expected_loss, TOL, "fused-path indexer loss")
        assert grads.keys() == expected_grads.keys()
        for name, expected in expected_grads.items():
            _assert_close(grads[name], expected, TOL, f"fused-path grad of {name}")

    @pytest.mark.parametrize("use_indexer_loss", [False, True], ids=["top-k", "top-k-and-loss"])
    def test_declined_hooks(self, pg_collection, monkeypatch, caplog, use_indexer_loss):
        """A declined hook raises above REFERENCE_FALLBACK_MAX_TOKENS query rows (b * s), else warns."""
        core = _build_fused_core(monkeypatch, pg_collection, seed=131)
        declined = []
        for hook in ("run_fused_qk_topk", "run_fused_qk_topk_with_loss"):
            monkeypatch.setattr(dsa_kernels, hook, lambda *args, _hook=hook, **kwargs: declined.append(_hook))
        monkeypatch.setattr(latent_mqa_core, "_WARNED", set())

        def select(seqlen: int, batch: int, seed: int):
            generator = torch.Generator().manual_seed(seed)
            doc_start = torch.zeros(seqlen, dtype=torch.int64)
            starts, ends = ops.ernie55_selection_bounds(doc_start, WINDOW)
            inputs = (
                torch.randn(seqlen, batch, INDEXER_HEADS, INDEXER_DIM, generator=generator),
                torch.randn(seqlen, batch, INDEXER_DIM, generator=generator),
                torch.rand(seqlen, batch, INDEXER_HEADS, generator=generator),
                torch.randn(seqlen, batch, HEADS, LATENT, generator=generator),
                torch.randn(seqlen, batch, 1, LATENT, generator=generator),
                starts,
                ops.ernie55_kernel_selection_ends(starts=starts, ends=ends),
                use_indexer_loss,
            )
            return core._select_topk(*inputs), inputs

        # Too many rows for the reference path, packed (b = 1) or batched (b * s counts).
        for seqlen, batch in ((REFERENCE_FALLBACK_MAX_TOKENS + 1, 1), (REFERENCE_FALLBACK_MAX_TOKENS // 2 + 1, 2)):
            with pytest.raises(RuntimeError, match=f"declined the ERNIE 5.5 D-layer top-k for {batch} x {seqlen}"):
                select(seqlen, batch, seed=132)
        # Short sequences fall back to the reference with one warning.
        with caplog.at_level(logging.WARNING, logger=latent_mqa_core.logger.name):
            (topk, loss), inputs = select(40, 2, seed=133)
            select(40, 2, seed=133)
        assert caplog.text.count("declined the ERNIE 5.5 D-layer top-k; using the PyTorch reference") == 1
        expected_hook = "run_fused_qk_topk_with_loss" if use_indexer_loss else "run_fused_qk_topk"
        assert set(declined) == {expected_hook} and len(declined) == 4
        core.use_fused = False
        expected_topk, expected_loss = core._select_topk(*inputs)
        assert torch.equal(topk, expected_topk)
        assert (loss is None) == (expected_loss is None) == (not use_indexer_loss)
        if use_indexer_loss:
            assert torch.equal(loss, expected_loss)


def _reference_selection_and_key_list(
    scores: torch.Tensor, doc_start: list[int], *, window_size: int, budget: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's CP-1 top-k and FlashMLA key list for one pack, given its score matrix ``[T, T]``.

    Step for step the reference implementation's selection at ``global_start = 0``, with the cuDNN score call
    replaced by rows of ``scores``.
    """
    l_local = s_global = scores.shape[0]
    rows = torch.arange(0, l_local)
    cols = torch.arange(s_global)
    doc_start = torch.tensor(doc_start)
    chosen = []
    for start in range(0, l_local, 512):
        end = min(start + 512, l_local)
        logits = scores[start:end].unsqueeze(0)
        remote = (cols[None, :] <= rows[start:end, None] - window_size) & (cols[None, :] >= doc_start[start:end, None])
        logits = logits.masked_fill(~remote.unsqueeze(0), float("-inf"))
        picked_scores, picked = torch.topk(logits, min(budget, s_global), dim=-1)
        valid = torch.isfinite(picked_scores)
        picked = torch.where(valid, picked, -1)
        if picked.shape[-1] < budget:
            picked = F.pad(picked, (0, budget - picked.shape[-1]), value=-1)
        chosen.append(picked)
    chosen = torch.cat(chosen, dim=1)
    window_start = torch.clamp(rows - window_size + 1, min=0).maximum(doc_start)
    window = window_start[:, None] + torch.arange(window_size)[None]
    window = torch.where(window <= rows[:, None], window, -1)[None].expand(chosen.shape[0], -1, -1)
    return chosen.int(), torch.cat([chosen, window], dim=-1).int()


class TestReferenceOrderSelection:
    """CP 1 on the fused path: the attention's top-k and key list are the reference implementation's."""

    @pytest.mark.parametrize("doc_lens", [[700, 300, 50], [1030]], ids=["multi-document", "single-document"])
    def test_top_k_and_key_list_are_the_references(self, pg_collection, monkeypatch, doc_lens):
        """Given the same score matrix (many exact ties included), the core selects the reference's keys in the
        reference's order, over 512-row chunks, and hands FlashMLA the reference's list without its empty blocks."""
        core = _build_fused_core(monkeypatch, pg_collection, seed=171)
        _install_fused_stand_ins(monkeypatch)
        num_rows = sum(doc_lens)
        generator = torch.Generator().manual_seed(172)
        causal = torch.arange(num_rows).view(1, -1) > torch.arange(num_rows).view(-1, 1)
        # Coarse integer scores: exact ties everywhere, also at the top-k boundary.
        scores = torch.randint(0, 9, (num_rows, num_rows), generator=generator).float().masked_fill(causal, -math.inf)
        calls = []

        def scorer(q, k, weights, *, ratio, sm_scale, q_causal_offsets=None):
            offset = 0 if q_causal_offsets is None else int(q_causal_offsets[0])
            calls.append((offset, q.size(1), sm_scale))
            return {"scores": scores[offset : offset + q.size(1)].unsqueeze(0).clone()}

        monkeypatch.setattr(latent_mqa_core, "_reference_indexer_scorer", lambda q: scorer)
        packed_seq_params = _packed_seq_params(doc_lens)
        doc_start = ops.ernie55_document_starts(packed_seq_params, num_rows, "cpu")
        starts, ends = ops.ernie55_selection_bounds(doc_start, WINDOW)
        kernel_ends = ops.ernie55_kernel_selection_ends(starts=starts, ends=ends)
        draw = functools.partial(torch.randn, generator=generator)
        inputs = (
            draw(num_rows, 1, INDEXER_HEADS, INDEXER_DIM),
            draw(num_rows, 1, INDEXER_DIM),
            draw(num_rows, 1, INDEXER_HEADS),
            draw(num_rows, 1, HEADS, LATENT),
            draw(num_rows, 1, 1, LATENT),
            starts,
            kernel_ends,
            False,
        )
        topk, loss = core._select_topk(*inputs, packed_seq_params=packed_seq_params)
        topk = ops.mask_topk_beyond_end(topk, ends=ends)

        expected_topk, expected_list = _reference_selection_and_key_list(
            scores, doc_start.tolist(), window_size=WINDOW, budget=TOPK
        )
        assert loss is None and core.requested_indexer_scoring_plan == latent_mqa_core.INDEXER_SCORING_REFERENCE_ORDER
        assert calls == [(start, min(512, num_rows - start), INDEXER_DIM**-0.5) for start in range(0, num_rows, 512)]
        assert torch.equal(topk, expected_topk), "the reference's keys, in the reference's order"
        window = ops.ernie55_window_indices(doc_start, window_size=WINDOW, batch_size=1)
        flat, _ = fused_sparse_attention.build_flat_topk_idxs(torch.cat((topk, window), dim=-1), batch_size=1)
        assert torch.equal(flat, expected_list[0])
        kept, length = ops.drop_invalid_key_blocks(flat, block=latent_mqa_core._FLASH_MLA_KEY_BLOCK)
        expected_kept, expected_length = ops.drop_invalid_key_blocks(
            expected_list[0], block=latent_mqa_core._FLASH_MLA_KEY_BLOCK
        )
        assert torch.equal(kept, expected_kept) and torch.equal(length, expected_length)

    def test_unavailable_scorer_falls_back_to_the_hooks(self, pg_collection, monkeypatch):
        """Without cuDNN's scorer (CPU tensors, an old frontend) the attention takes the top-k hook's keys."""
        core = _build_fused_core(monkeypatch, pg_collection, seed=173)
        calls = _install_fused_stand_ins(monkeypatch)
        monkeypatch.setattr(latent_mqa_core, "_reference_indexer_scorer", lambda q: None)
        batch = _core_batch(seed=174, doc_lens=[1, 2, 15, 3, 20])
        with torch.no_grad():
            _run_core(core.eval(), batch)
        names = [name for name, _ in calls]
        assert "indexer_forward_wrapper" not in names and names.count("run_fused_qk_topk") == 1
        assert core.requested_indexer_scoring_plan == latent_mqa_core.INDEXER_SCORING_PACKED_THD_GENERAL


# ---------------------------------------------------------------------------------------------
# Module contract: initialization, deferred weight gradients, construction guards
# ---------------------------------------------------------------------------------------------


class _FusedNormColumnLinear(ColumnParallelLinear):
    """Stand-in for a fused norm + linear projection (the norm is not applied; nothing runs forward)."""

    def __init__(self, *args, eps: float | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class TestModuleContract:
    def test_initial_values(self, pg_collection):
        """Sink 0, VHA V 0 (identity mix), U from init_method; the indexer key norm is fp32 at (1, 0)."""
        config = _config(params_dtype=torch.bfloat16)
        assert config.perform_initialization
        # The std check below sees only HEADS * VHA_RANK draws; seed them so it cannot fail by chance.
        torch.manual_seed(0)
        core = Ernie55LatentMQACoreAttention(
            ernie55_d_layer_config(config),
            _core_submodules(),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            softmax_scale=SCALE,
            k_channels=LATENT,
            v_channels=KV_LORA,
            pg_collection=pg_collection,
        )
        assert torch.equal(core.attn_sink, torch.zeros(HEADS))
        assert torch.equal(core.vha_postmix_v, torch.zeros(HEADS, VHA_RANK))
        assert bool(torch.isfinite(core.vha_postmix_u).all()) and bool((core.vha_postmix_u != 0).all())
        assert core.vha_postmix_u.std().item() == pytest.approx(config.init_method_std, rel=0.75)
        k_norm = core.indexer.k_norm
        for param, value in ((k_norm.weight, 1.0), (k_norm.bias, 0.0)):
            assert param.dtype == torch.float32 and getattr(param, "keep_in_fp32", False)
            assert torch.equal(param, torch.full_like(param, value))
        assert core.indexer.linear_wk.weight.dtype == torch.bfloat16

    def test_backward_dw_reaches_the_indexer_and_the_gate(self, pg_collection, monkeypatch):
        attention = _build_attention(_config(), pg_collection, seed=141)
        core = attention.core_attention
        calls = []
        monkeypatch.setattr(core.indexer, "backward_dw", lambda: calls.append("indexer"))
        monkeypatch.setattr(core.linear_output_gate, "backward_dw", lambda: calls.append("gate"))
        monkeypatch.setattr(attention.linear_kv_down_proj, "backward_dw", lambda: calls.append("kv_down"))
        core.backward_dw()
        assert calls == ["indexer", "gate"]
        calls.clear()
        attention.backward_dw()  # AbsorbedMLA's deferred-wgrad entry point
        # Of the K/V projections only linear_kv_down_proj defers its weight gradient: autograd computes
        # the k_b_proj and v_b_proj gradients.
        assert calls == ["kv_down", "indexer", "gate"]

    @pytest.mark.parametrize(
        ("use_cpu_initialization", "init_model_with_meta_device", "expected"),
        [(True, False, "cpu"), (True, True, "cpu"), (False, True, "meta")],
    )
    def test_parameter_device_follows_megatron_core(
        self, use_cpu_initialization, init_model_with_meta_device, expected
    ):
        """The core's own parameters go where Megatron-Core's TE modules put theirs: cpu, then meta, then CUDA."""
        config = _config(
            use_cpu_initialization=use_cpu_initialization, init_model_with_meta_device=init_model_with_meta_device
        )
        assert latent_mqa_core._param_device(config) == expected

    @pytest.mark.parametrize(
        "overrides", [{"bf16": False}, {"dsa_indexer_rope_interleaved": True}], ids=["fp32", "interleaved-rope"]
    )
    def test_attention_validates_its_config(self, pg_collection, overrides):
        """The D attention checks the envelope itself: its config need not come from Ernie55TransformerConfig."""
        config = copy.copy(_config())
        for name, value in overrides.items():
            if not hasattr(config, name):
                raise ValueError(f"{type(config).__name__} has no field {name!r}")
            setattr(config, name, value)
        with pytest.raises(ValueError, match="ERNIE 5.5 D layer: "):
            _build_attention(config, pg_collection, seed=161)

    def test_fused_path_rejects_unsupported_geometry(self, pg_collection):
        with pytest.raises(ValueError, match="does not support this geometry"):
            _build_core(_with_cudnn_backend(_config()), pg_collection, seed=151)

    def test_latent_norms_must_be_explicit(self, pg_collection, monkeypatch):
        """Outside a "dsa" root, Megatron-Core fuses trivial q/kv norms into the up-projections, which
        AbsorbedMLA bypasses; the D attention refuses such a spec instead of running without the norms."""
        monkeypatch.setattr(
            LocalSpecProvider, "column_parallel_layer_norm_linear", lambda self: _FusedNormColumnLinear
        )
        spec = _attention_spec()
        spec.submodules.q_layernorm = IdentityOp
        spec.submodules.kv_layernorm = IdentityOp
        config = _config(experimental_attention_variant=None, dsa_indexer_loss_coeff=0.0)
        with pytest.raises(ValueError, match="explicit q_layernorm"):
            build_module(spec, config=config, layer_number=1, pg_collection=pg_collection)
        # A "dsa" root resolves trivial norms to standalone ones, which the D layer accepts.
        attention = build_module(spec, config=_config(), layer_number=1, pg_collection=pg_collection)
        assert not isinstance(attention.q_layernorm, IdentityOp) and not isinstance(attention.kv_layernorm, IdentityOp)


# ---------------------------------------------------------------------------------------------
# Full D-layer self attention (Megatron-Core AbsorbedMLA + the ERNIE core) against the oracle
# ---------------------------------------------------------------------------------------------


def _attention_inputs(layout: dict, seed: int) -> tuple[torch.Tensor, PackedSeqParams | None, list[int]]:
    generator = torch.Generator().manual_seed(seed)
    if "doc_lens" in layout:
        packed_seq_params = _packed_seq_params(layout["doc_lens"], layout.get("padded_lens"))
        num_rows, batch = sum(layout.get("padded_lens") or layout["doc_lens"]), 1
        if layout.get("max_seqlen") == "pack":
            # Bridge's padded SFT packs set max_seqlen to the pack length.
            packed_seq_params.max_seqlen_q = packed_seq_params.max_seqlen_kv = num_rows
    else:
        packed_seq_params, num_rows, batch = None, layout["seqlen"], layout["batch"]
    x = torch.randn(num_rows, batch, HIDDEN, generator=generator).requires_grad_(True)
    return x, packed_seq_params, _oracle_document_starts(packed_seq_params, num_rows)


# Stack geometry that differs from the D layer, with the D geometry supplied by the ernie_d_* fields.
D_OVERRIDES = dict(
    num_attention_heads=8,
    q_lora_rank=48,
    kv_lora_rank=24,
    qk_head_dim=16,
    qk_pos_emb_head_dim=16,
    v_head_dim=16,
    rotary_base=10000.0,
    ernie_d_num_attention_heads=HEADS,
    ernie_d_q_lora_rank=Q_LORA,
    ernie_d_kv_lora_rank=KV_LORA,
    ernie_d_qk_head_dim=NOPE,
    ernie_d_qk_pos_emb_head_dim=ROPE,
    ernie_d_v_head_dim=V_HEAD,
    ernie_d_rotary_base=ROTARY_BASE,
)


class TestSelfAttention:
    @pytest.mark.parametrize(
        ("layout", "config_overrides"),
        [
            (dict(doc_lens=[5, 9, 1, 7]), {}),
            (dict(doc_lens=[5, 7, 2], padded_lens=[6, 8, 4]), {}),
            (dict(seqlen=9, batch=2), {}),
            (dict(doc_lens=[13, 4, 16]), {}),
            # max_seqlen equal to the pack length (as in Bridge's padded SFT packs) must still give
            # per-document indexer positions and results.
            (dict(doc_lens=[13, 4, 16], max_seqlen="pack"), {}),
            (dict(doc_lens=[6, 9, 3]), D_OVERRIDES),
        ],
        ids=["thd-dense", "thd-padded", "sbhd-b2", "thd-selection", "thd-max-seqlen-pack", "d-geometry-overrides"],
    )
    def test_matches_fp64_oracle(self, pg_collection, monkeypatch, layout, config_overrides):
        attention = _build_attention(_config(**config_overrides), pg_collection, seed=111)
        x, packed_seq_params, doc_start = _attention_inputs(layout, seed=112)
        index_lists = _capture_index_lists(monkeypatch, attention.core_attention)

        output, bias = attention(x, None, packed_seq_params=packed_seq_params)
        assert bias is None
        oracle = _oracle_attention(attention, x, doc_start, selected=index_lists[0])
        _assert_index_lists(index_lists[0], oracle.allowed)
        # Keys are dropped exactly when some row has more than K remote candidates.
        positions = _positions(doc_start)
        selects = bool((oracle.allowed.sum(-1) < positions + 1).any())
        assert selects == bool((positions >= WINDOW + TOPK).any())
        _assert_close(output, oracle.output, TOL, "output")
        loss, _ = _logged_indexer_loss(layer_number=1)
        _assert_close(loss, oracle.loss, TOL, "indexer loss")

        grad_output = torch.randn(output.shape, generator=torch.Generator().manual_seed(113))
        output.backward(grad_output)
        ((oracle.output * grad_output.double()).sum() + oracle.loss).backward()
        _assert_parameter_grads(attention, oracle.params)
        _assert_close(x.grad, oracle.x.grad, TOL, "grad of x")

    def test_parameter_names_are_pinned(self, pg_collection):
        attention = _build_attention(_config(params_dtype=torch.bfloat16), pg_collection, seed=121)
        parameters = dict(attention.named_parameters())
        assert {name: tuple(param.shape) for name, param in parameters.items()} == EXPECTED_PARAMETERS
        fp32_parameters = (
            "core_attention.attn_sink",
            "core_attention.vha_postmix_u",
            "core_attention.vha_postmix_v",
            "core_attention.indexer.k_norm.weight",
            "core_attention.indexer.k_norm.bias",
        )
        for name in fp32_parameters:
            assert parameters[name].dtype == torch.float32 and getattr(parameters[name], "keep_in_fp32", False), name
        for name in ("core_attention.linear_output_gate.weight", "k_b_proj", "v_b_proj"):
            assert parameters[name].dtype == torch.bfloat16, name
        assert not isinstance(attention.q_layernorm, IdentityOp) and not isinstance(attention.kv_layernorm, IdentityOp)

    @pytest.mark.parametrize(
        ("layer_number", "is_mtp_layer", "slot"),
        # Decoder layers log at their position; the first MTP layer logs at num_layers + 1.
        [(2, False, 2), (1, True, NUM_LAYERS + 1), (2, True, NUM_LAYERS + 2)],
        ids=["decoder-layer-2", "mtp-layer-1", "mtp-layer-2"],
    )
    def test_indexer_loss_tracker_slot(self, pg_collection, layer_number, is_mtp_layer, slot):
        config = _config()
        attention = _build_attention(
            config, pg_collection, seed=131, layer_number=layer_number, is_mtp_layer=is_mtp_layer
        )
        assert attention.core_attention.layer_number == slot
        assert indexer_loss_layer_number(config, layer_number, is_mtp_layer) == slot

        x, packed_seq_params, doc_start = _attention_inputs(dict(doc_lens=[4, 9]), seed=132)
        attention(x, None, packed_seq_params=packed_seq_params)
        oracle = _oracle_attention(attention, x, doc_start)
        values = dsa.DSAIndexerLossLoggingHelper.tracker["values"]
        assert values.numel() >= slot
        assert [index + 1 for index in values.nonzero().flatten().tolist()] == [slot]
        loss, raw_loss = _logged_indexer_loss(slot)
        _assert_close(loss, oracle.loss, TOL, "logged indexer loss")
        _assert_close(raw_loss, oracle.loss / COEFF, TOL, "logged raw indexer loss")


# ---------------------------------------------------------------------------------------------
# K/V up-projection layout: the reference implementation's k_b_proj and v_b_proj
# ---------------------------------------------------------------------------------------------

# The D geometry of the reference implementation's parity runs: hidden 4096, 64 heads, q/kv latents
# 1024/512, nope/rope/v head dims 448/64/256, VHA rank 16 and 64 indexer heads of 128.
REFERENCE_GEOMETRY = dict(
    hidden_size=4096,
    num_attention_heads=64,
    q_lora_rank=1024,
    kv_lora_rank=512,
    qk_head_dim=448,
    qk_pos_emb_head_dim=64,
    v_head_dim=256,
    ernie_vha_rank=16,
    dsa_indexer_n_heads=64,
    dsa_indexer_head_dim=128,
)
# Every 2-D tensor of the reference implementation's D attention at that geometry, as its parameter
# dump lists them: Bridge parameter -> (reference tensor, shape).
REFERENCE_2D_TENSORS = {
    "k_b_proj": ("k_b_proj", (32768, 448)),
    "v_b_proj": ("v_b_proj", (16384, 512)),
    "linear_q_down_proj.weight": ("linear_q_down_proj.weight", (1024, 4096)),
    "linear_q_up_proj.weight": ("linear_q_up_proj.weight", (32768, 1024)),
    "linear_kv_down_proj.weight": ("linear_kv_down_proj.weight", (576, 4096)),
    "linear_proj.weight": ("linear_proj.weight", (4096, 16384)),
    "core_attention.linear_output_gate.weight": ("output_gate.gate_proj.weight", (16384, 1024)),
    "core_attention.vha_postmix_u": ("vha.U", (64, 16)),
    "core_attention.vha_postmix_v": ("vha.V", (64, 16)),
    "core_attention.indexer.linear_wq_b.weight": ("core_attention.indexer.wq_b.weight", (8192, 1024)),
    "core_attention.indexer.linear_wk.weight": ("core_attention.indexer.wk.weight", (128, 4096)),
    "core_attention.indexer.linear_weights_proj.weight": ("core_attention.indexer.weights_proj.weight", (64, 4096)),
}
# The production K view vs the combined layout, as a normalized max error: GEMM kernels may round a K
# operand stored transposed per head differently (see TestKVUpLayout.test_strided_k_view_matches_the_combined_layout).
LAYOUT_STRIDED_TOL = {torch.float32: 4e-6, torch.bfloat16: 2e-2}
# (layout, config overrides) cases of the comparisons with the combined layout.
COMBINED_LAYOUT_CASES = pytest.mark.parametrize(
    ("layout", "config_overrides"),
    [
        (dict(doc_lens=[13, 4, 16]), {}),
        (dict(doc_lens=[5, 7, 2], padded_lens=[6, 8, 4]), {}),
        (dict(seqlen=9, batch=2), {}),
        (dict(doc_lens=[6, 9, 3]), D_OVERRIDES),
    ],
    ids=["thd-selection", "thd-padded", "sbhd-b2", "d-geometry-overrides"],
)


def _split_kv_up(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(k_b_proj, v_b_proj)`` from Megatron-Core's combined ``[H * (nope + v), C]`` K/V up-projection.

    Head ``h`` of the combined weight is ``[W_UK[h]; W_UV[h]]``; ``k_b_proj`` stacks ``W_UK[h]^T``
    and ``v_b_proj`` stacks ``W_UV[h]``. Gradients map like weights.
    """
    per_head = weight.view(HEADS, NOPE + V_HEAD, KV_LORA)
    k_b_proj = per_head[:, :NOPE].transpose(1, 2).reshape(HEADS * KV_LORA, NOPE)
    v_b_proj = per_head[:, NOPE:].reshape(HEADS * V_HEAD, KV_LORA)
    return k_b_proj, v_b_proj


def _build_combined_layout_attention(
    config: Ernie55TransformerConfig, pg_collection: ProcessGroupCollection, monkeypatch, *, seed: int
) -> Ernie55LatentMQASelfAttention:
    """The D attention with Megatron-Core's combined ``linear_kv_up_proj``, read by Megatron-Core's accessor."""
    with monkeypatch.context() as patch:
        patch.setattr(Ernie55LatentMQASelfAttention, "_split_kv_up_projection", lambda self: None)
        attention = _build_attention(config, pg_collection, seed=seed)
    for name in ("_get_kv_up_weights", "_backward_kv_proj"):
        setattr(attention, name, types.MethodType(getattr(AbsorbedMLASelfAttention, name), attention))
    assert "linear_kv_up_proj.weight" in dict(attention.named_parameters())
    return attention


def _forward_backward(attention: Ernie55LatentMQASelfAttention, x, grad_output, packed_seq_params):
    """Output, logged indexer loss, input gradient and parameter gradients of one forward and backward."""
    dsa.DSAIndexerLossLoggingHelper.tracker.clear()
    x = x.detach().clone().requires_grad_(True)
    output, _ = attention(x, None, packed_seq_params=packed_seq_params)
    loss, _ = _logged_indexer_loss(layer_number=1)
    loss = loss.detach().clone()
    output.backward(grad_output)
    grads = {name: param.grad for name, param in attention.named_parameters()}
    return output.detach(), loss, x.grad, grads


def _contiguous_k_view(attention: Ernie55LatentMQASelfAttention) -> tuple[torch.Tensor, torch.Tensor]:
    """The D attention's K/V accessor, with the K view copied to a contiguous ``[H, nope, C]`` tensor."""
    k_up, v_up = Ernie55LatentMQASelfAttention._get_kv_up_weights(attention)
    return k_up.contiguous(), v_up


def _compare_with_combined_layout(
    pg_collection: ProcessGroupCollection,
    monkeypatch,
    layout: dict,
    config_overrides: dict,
    dtype: torch.dtype,
    *,
    contiguous_k: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """``(expected, actual)``: results of the combined layout and of the D attention, with the same weights.

    The D attention gets the combined weights through the K/V split, and the combined layout's K/V
    gradient goes through the same split. Each dict holds the output, the logged indexer loss and
    every gradient. With ``contiguous_k`` the D attention hands AbsorbedMLA a contiguous copy of
    its K view instead of the view itself.
    """
    config = _config(params_dtype=dtype, **config_overrides)
    combined = _build_combined_layout_attention(config, pg_collection, monkeypatch, seed=171)
    attention = _build_attention(config, pg_collection, seed=172)
    state = combined.state_dict()
    k_b_proj, v_b_proj = _split_kv_up(state.pop("linear_kv_up_proj.weight"))
    state.pop("linear_kv_up_proj._extra_state", None)
    attention.load_state_dict({**state, "k_b_proj": k_b_proj, "v_b_proj": v_b_proj})
    for actual, expected in zip(attention._get_kv_up_weights(), combined._get_kv_up_weights()):
        assert torch.equal(actual, expected)
    if contiguous_k:
        attention._get_kv_up_weights = types.MethodType(_contiguous_k_view, attention)
    assert attention._get_kv_up_weights()[0].is_contiguous() == contiguous_k

    x, packed_seq_params, _ = _attention_inputs(layout, seed=173)
    x = x.detach().to(dtype)
    grad_output = torch.randn(x.shape, generator=torch.Generator().manual_seed(174)).to(dtype)
    expected_output, expected_loss, expected_x_grad, expected_grads = _forward_backward(
        combined, x, grad_output, packed_seq_params
    )
    output, loss, x_grad, grads = _forward_backward(attention, x, grad_output, packed_seq_params)
    k_grad, v_grad = _split_kv_up(expected_grads.pop("linear_kv_up_proj.weight"))
    expected_grads.update(k_b_proj=k_grad, v_b_proj=v_grad)
    assert grads.keys() == expected_grads.keys()
    assert expected_loss.item() > 0
    expected = {"output": expected_output, "indexer loss": expected_loss, "grad of x": expected_x_grad}
    actual = {"output": output, "indexer loss": loss, "grad of x": x_grad}
    expected.update({f"grad of {name}": grad for name, grad in expected_grads.items()})
    actual.update({f"grad of {name}": grad for name, grad in grads.items()})
    return expected, actual


def _newton_schulz(matrix: torch.Tensor, steps: int = 100) -> torch.Tensor:
    """Orthogonal polar factor of ``matrix`` by the cubic Newton-Schulz iteration (Muon runs a quintic one)."""
    x = matrix / matrix.norm()
    for _ in range(steps):
        x = 1.5 * x - 0.5 * x @ x.mT @ x
    return x


class TestKVUpLayout:
    """The K/V up-projections are stored as the reference implementation's two 2-D tensors."""

    def test_two_dimensional_parameters_match_the_reference_implementation(self, pg_collection):
        """The 2-D parameters are exactly those of REFERENCE_2D_TENSORS, each with its reference tensor's shape.

        Muon orthogonalizes every 2-D parameter as one matrix, so matching tensors get matching updates.
        """
        config = _config(**REFERENCE_GEOMETRY, params_dtype=torch.bfloat16, perform_initialization=False)
        attention = build_module(_attention_spec(), config=config, layer_number=1, pg_collection=pg_collection)
        shapes = {name: tuple(param.shape) for name, param in attention.named_parameters() if param.dim() == 2}
        assert shapes == {name: shape for name, (_, shape) in REFERENCE_2D_TENSORS.items()}

    @pytest.mark.parametrize("megatron_kv_up", ["combined", "split"])
    def test_initialization_reuses_megatron_core_values(self, pg_collection, monkeypatch, megatron_kv_up):
        """``k_b_proj`` and ``v_b_proj`` hold the ``init_method`` draws of the projections Megatron-Core built.

        That is the combined ``linear_kv_up_proj``, or, for a split-projection spec (dev only),
        ``linear_k_up_proj`` ``[H * nope, C]`` and ``linear_v_up_proj`` ``[H * v, C]``. No random
        numbers are drawn: every other parameter and the generator state after construction match.
        """
        spec = _attention_spec()
        if megatron_kv_up == "split":
            if "linear_k_up_proj" not in {field.name for field in dataclasses.fields(spec.submodules)}:
                pytest.skip("this Megatron-Core has no split K/V up-projection spec")
            spec.submodules.linear_kv_up_proj = None
            spec.submodules.linear_k_up_proj = spec.submodules.linear_v_up_proj = ColumnParallelLinear
        config = _config()
        states, rng_states = [], []
        for split in (False, True):
            with monkeypatch.context() as patch:
                if not split:
                    patch.setattr(Ernie55LatentMQASelfAttention, "_split_kv_up_projection", lambda self: None)
                torch.manual_seed(181)
                attention = build_module(spec, config=config, layer_number=1, pg_collection=pg_collection)
                rng_states.append(torch.get_rng_state())
            states.append(attention.state_dict())
        megatron, split = states
        kv_up_modules = ("linear_kv_up_proj", "linear_k_up_proj", "linear_v_up_proj")
        removed = {name: megatron.pop(name) for name in list(megatron) if name.split(".")[0] in kv_up_modules}
        if megatron_kv_up == "combined":
            expected = dict(zip(("k_b_proj", "v_b_proj"), _split_kv_up(removed["linear_kv_up_proj.weight"])))
        else:
            k_up = removed["linear_k_up_proj.weight"].view(HEADS, NOPE, KV_LORA)
            expected = {
                "k_b_proj": k_up.transpose(1, 2).reshape(HEADS * KV_LORA, NOPE),
                "v_b_proj": removed["linear_v_up_proj.weight"],
            }
        assert torch.equal(rng_states[0], rng_states[1])
        assert split.keys() == {*megatron, *expected}
        for name, value in {**megatron, **expected}.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(split[name], value), name
            else:  # the linear layers' _extra_state
                assert split[name] == value, name

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    @COMBINED_LAYOUT_CASES
    def test_layout_conversion_is_exact(self, pg_collection, monkeypatch, layout, config_overrides, dtype):
        """Same weights through the K/V split, K copied contiguous: every result matches bit for bit.

        A contiguous ``[H, nope, C]`` K differs from the combined layout's K slice only in its batch
        stride, so AbsorbedMLA's GEMMs see the same operands. The output, the indexer loss and every
        gradient (the combined K/V gradient through the same split) are then bitwise equal in fp32
        and bf16, which checks the split, the views and the gradient mapping exactly.
        """
        expected, actual = _compare_with_combined_layout(
            pg_collection, monkeypatch, layout, config_overrides, dtype, contiguous_k=True
        )
        for what, value in expected.items():
            assert actual[what].dtype == value.dtype and torch.equal(actual[what], value), what

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    @COMBINED_LAYOUT_CASES
    def test_strided_k_view_matches_the_combined_layout(
        self, pg_collection, monkeypatch, layout, config_overrides, dtype
    ):
        """The production path, which reads ``k_b_proj`` as a view transposed per head, matches the combined layout.

        Only the memory layout of the K operand differs, and a GEMM may round a transposed operand
        differently: a CPU BLAS may change fp32 results by a few ulp, and bf16 GEMMs do at larger sizes.
        So the results agree to ``LAYOUT_STRIDED_TOL`` in either dtype, not bit for bit.
        """
        expected, actual = _compare_with_combined_layout(
            pg_collection, monkeypatch, layout, config_overrides, dtype, contiguous_k=False
        )
        for what, value in expected.items():
            assert actual[what].dtype == value.dtype, what
            _assert_close(actual[what], value, LAYOUT_STRIDED_TOL[dtype], what)

    def test_attention_module_alone_sets_no_kv_up_layout(self, pg_collection):
        """The attention module alone leaves ``k_b_proj`` and ``v_b_proj`` as plain Muon matrices.

        The emerging-optimizer setup splits a 2-D weight by rows only when it carries a
        ``qkv_layout`` (dev) or its name contains ``linear_qkv.weight``. It sends a parameter to
        the scalar optimizer when the parameter sets ``use_muon=False``, is not 2-D, or is an
        embedding or output weight. The attention module sets no layout on these two, so on its own
        Muon would orthogonalize each whole. The ERNIE 5.5 model then gives both the reference's
        per-head blocks, which Muon orthogonalizes one by one (``modeling_ernie55.muon``,
        ``test_ernie55_muon``).
        """
        from megatron.core.optimizer import emerging_optimizers

        prefix = "decoder.layers.1.self_attention"
        attention = _build_attention(_config(params_dtype=torch.bfloat16), pg_collection, seed=201)
        parameters = dict(attention.named_parameters(prefix=prefix))
        for name in ("k_b_proj", "v_b_proj"):
            full_name = f"{prefix}.{name}"
            param = parameters[full_name]
            assert param.dim() == 2 and param.requires_grad, name
            assert getattr(param, "qkv_layout", None) is None and "linear_qkv.weight" not in full_name, name
            assert getattr(param, "use_muon", True), name
            assert not emerging_optimizers._is_nonlinear_or_embedding(param), name
            assert not emerging_optimizers._is_muon_excluded(param), name

    def test_muon_update_depends_on_the_layout(self):
        """Why the layout matters: Muon's update of the same K gradient differs between layouts.

        Head ``h`` of ``k_b_proj`` holds ``B_h = W_UK[h]^T``. Megatron-Core stores ``B_h^T``, either
        stacked with ``W_UV[h]`` (``linear_kv_up_proj``) or alone (dev's split ``linear_k_up_proj``).
        The polar factor of a stack of blocks is not the stack of the transposed blocks' polar factors,
        so either Megatron-Core layout would update ``W_UK`` differently from the reference
        implementation. With a single head the split layout agrees, since ``polar(B^T) = polar(B)^T``.
        """
        generator = torch.Generator().manual_seed(191)

        def k_updates(heads: int) -> tuple[torch.Tensor, ...]:
            """A K gradient in the ``k_b_proj`` layout and its update when Muon sees each layout."""
            grad_k = torch.randn(heads * KV_LORA, NOPE, generator=generator, dtype=torch.float64)
            grad_v = torch.randn(heads * V_HEAD, KV_LORA, generator=generator, dtype=torch.float64)
            w_uk = grad_k.view(heads, KV_LORA, NOPE).transpose(1, 2)
            split = _newton_schulz(w_uk.reshape(heads * NOPE, KV_LORA)).view(heads, NOPE, KV_LORA)
            per_head = torch.cat([w_uk, grad_v.view(heads, V_HEAD, KV_LORA)], dim=1)
            combined = _newton_schulz(per_head.reshape(-1, KV_LORA)).view(heads, NOPE + V_HEAD, KV_LORA)[:, :NOPE]
            return (
                grad_k,
                _newton_schulz(grad_k),
                split.transpose(1, 2).reshape(heads * KV_LORA, NOPE),
                combined.transpose(1, 2).reshape(heads * KV_LORA, NOPE),
            )

        grad_k, reference, split, combined = k_updates(HEADS)
        # Newton-Schulz converged to the polar factor U V^T of the gradient U S V^T.
        u, _, vh = torch.linalg.svd(grad_k, full_matrices=False)
        torch.testing.assert_close(reference, u @ vh)
        for name, update in (("split", split), ("combined", combined)):
            assert (update - reference).norm() > 0.1 * reference.norm(), name
        _, reference, split, _ = k_updates(1)
        torch.testing.assert_close(split, reference)


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


def _mutable_spec_nodes(node: object, found: dict[int, str], path: str = "spec") -> dict[int, str]:
    """``id -> path`` of every container, dataclass instance and ``functools.partial`` in a spec tree."""
    if id(node) in found:
        return found
    if isinstance(node, (list, dict, functools.partial)) or (
        dataclasses.is_dataclass(node) and not isinstance(node, type)
    ):
        found[id(node)] = path
    if isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            _mutable_spec_nodes(item, found, f"{path}[{index}]")
    elif isinstance(node, dict):
        for key, value in node.items():
            _mutable_spec_nodes(value, found, f"{path}[{key!r}]")
    elif isinstance(node, functools.partial):
        _mutable_spec_nodes(node.args, found, f"{path}.args")
        _mutable_spec_nodes(node.keywords, found, f"{path}.keywords")
    elif dataclasses.is_dataclass(node) and not isinstance(node, type):
        for field in dataclasses.fields(node):
            _mutable_spec_nodes(getattr(node, field.name), found, f"{path}.{field.name}")
    return found


# Slots ernie55_hybrid_stack_spec changes; test_ernie55_hca.py covers the H ones, test_ernie55_moe.py the
# MLP and MoE ones and test_ernie55_mhc_mtp.py the MTP hnorm.
_ERNIE_SLOTS = (
    "dsa_layer",
    "csa_layer",
    "csa_qk_layernorm_layer",
    "hca_layer",
    "mlp_layer",
    "moe_layer",
    "mtp_block_spec",
)


class TestStackSpec:
    def test_only_the_ernie_slots_change(self):
        from megatron.core.extensions.transformer_engine import TELinear, TENorm
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

        mcore_attention = hybrid_stack_spec.submodules.dsa_layer.submodules.self_attention
        mcore_indexer = mcore_attention.submodules.core_attention.submodules.indexer
        spec = ernie55_hybrid_stack_spec()

        attention = spec.submodules.dsa_layer.submodules.self_attention
        assert attention.module is Ernie55LatentMQASelfAttention
        for norm in (attention.submodules.q_layernorm, attention.submodules.kv_layernorm):
            # TENorm, or the reference arithmetic under ernie_reference_rmsnorm; a ModuleSpec, because
            # AbsorbedMLA builds its latent norms with build_module.
            assert _spec_equal(norm, ernie55_rmsnorm_spec(TENorm))
        core = attention.submodules.core_attention
        assert core.module is Ernie55LatentMQACoreAttention
        assert isinstance(core.submodules, Ernie55LatentMQACoreSubmodules)
        assert core.submodules.indexer.module is dsa.DSAIndexer
        assert _spec_equal(core.submodules.indexer, mcore_indexer)
        assert core.submodules.linear_output_gate is TELinear
        # The input norm downcasts the fp32 hyper-connection streams (test_ernie55_mhc_mtp.py).
        input_norm = spec.submodules.dsa_layer.submodules.input_layernorm
        expected_norm = downcasting_norm(TENorm)
        assert input_norm.func is expected_norm.func and input_norm.keywords == expected_norm.keywords

        # Megatron-Core's shared spec is untouched, and every non-ERNIE slot is unchanged.
        assert mcore_attention.module is not Ernie55LatentMQASelfAttention
        assert mcore_attention.submodules.core_attention.module is dsa.DSAttention
        assert mcore_attention.submodules.q_layernorm is IdentityOp
        for slot in dataclasses.fields(spec.submodules):
            if slot.name not in _ERNIE_SLOTS:
                mcore_slot = getattr(hybrid_stack_spec.submodules, slot.name)
                assert _spec_equal(getattr(spec.submodules, slot.name), mcore_slot), slot.name

    def test_returns_a_private_copy(self):
        """No container, submodule dataclass or partial is shared with Megatron-Core's global spec."""
        from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec

        # Keep every spec alive while comparing object ids.
        first, second = ernie55_hybrid_stack_spec(), ernie55_hybrid_stack_spec()
        mcore_nodes = _mutable_spec_nodes(hybrid_stack_spec, {})
        assert any(isinstance(node, functools.partial) for node in _all_spec_objects(hybrid_stack_spec))
        for spec in (first, second):
            ernie_nodes = _mutable_spec_nodes(spec, {})
            assert len(ernie_nodes) >= len(mcore_nodes)
            shared = sorted(ernie_nodes[node_id] for node_id in set(ernie_nodes) & set(mcore_nodes))
            assert not shared, f"objects shared with Megatron-Core's hybrid_stack_spec: {shared[:10]}"
        # Two calls return independent copies as well.
        assert not set(_mutable_spec_nodes(first, {})) & set(_mutable_spec_nodes(second, {}))


def _all_spec_objects(node: object, seen: set[int] | None = None) -> list[object]:
    """Every object reachable through containers, dataclass fields and partial arguments."""
    seen = set() if seen is None else seen
    if id(node) in seen:
        return []
    seen.add(id(node))
    found = [node]
    if isinstance(node, (list, tuple)):
        children = list(node)
    elif isinstance(node, dict):
        children = list(node.values())
    elif isinstance(node, functools.partial):
        children = [node.args, node.keywords]
    elif dataclasses.is_dataclass(node) and not isinstance(node, type):
        children = [getattr(node, field.name) for field in dataclasses.fields(node)]
    else:
        children = []
    for child in children:
        found.extend(_all_spec_objects(child, seen))
    return found
