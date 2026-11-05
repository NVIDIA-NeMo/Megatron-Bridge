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

"""CPU tests of ERNIE 5.5 context parallelism: the D layer at CP = 2 and 4 against CP = 1.

The D layer runs CP in the zigzag partition inside a contiguous stack (see
``megatron.bridge.models.ernie.modeling_ernie55.context_parallel``). Every check here compares a
D attention module on ``CP`` gloo ranks with the same module (same parameters, same packed
micro-batch) on one rank, through the PyTorch reference path (``dsa_kernel_backend="none"``), which
the D-layer tests check against an fp64 oracle at CP = 1:

* the output rows and the input gradient of every rank equal the CP = 1 rows at the same packed
  positions, and the parameter gradients summed over the CP group equal the CP = 1 gradients, for
  multi-document packs whose document boundaries cross the ranks' contiguous chunks, a pack of one
  document (the production shape), padded documents and the first ``W`` rows of every document;
* the indexer KL: the ranks' losses sum to the CP = 1 loss (each rank normalizes by the rows of the
  whole pack), the tracker averages it over the CP group, and with the schedule's ``CP`` scale and
  DDP's mean over CP the indexer gradients equal the CP = 1 gradients (not CP times them);
* with a designed indexer (no near-ties), the selected keys of every row equal the CP = 1 ones;
* a micro-batch built by ``build_ernie55_thd_cp_batch`` (unaligned samples padded to ``2 * CP``,
  prebuilt routes) runs without Megatron-Core's route-rebuild fallback;
* a zigzag stack (``"dsa"``, no conversion), per-token loss, full recompute, the restore of
  ``packed_seq_params.cp_partition_mode`` after a forward, a backward, a recompute and an error;
* the fused path's wiring with CPU stand-ins for the kernels, on a padded layout whose unpadded
  boundaries differ from the padded ones: both of Megatron-Core's top-k hooks (the loss hook of a
  training forward and a recompute replay, the plain hook of evaluation and of the no-grad forward
  of full recompute) get the zigzag ``PACKED_THD_GENERAL`` arguments (one tensor of the padded
  ``cu_seqlens`` for queries and keys, the pack's ``max_seqlen``, ``packed_thd_single_sequence=False``)
  and the loss hook ``coeff / CP``; the replay gets the no-grad forward's indexer inputs;
* the zigzag geometry (positions, key permutation, one cache entry per micro-batch);
* the guards: SBHD input, missing routes, a stack partition that disagrees with the config, and
  Megatron-Core builds without the CP converter or its zigzag helpers (main), at construction and
  in forward, or (for per-row MTP) without the grouped MTP roll;
* the H layer builds at CP > 1 and every rotary holder is its plain RoPE on the CP group.

The ranks are spawned processes (``ernie55_cp_harness``); each runs every case once and the pytest
tests read the results.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import inspect
import math
import types
import warnings
import zlib
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.experimental_attention_variant import dsa, dsa_kernels
from megatron.core.transformer.experimental_attention_variant.absorbed_mla import AbsorbedMLASelfAttentionSubmodules
from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.torch_norm import WrappedTorchNorm

from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
from megatron.bridge.models.ernie.modeling_ernie55 import (
    INDEXER_SCORING_PACKED_THD_GENERAL,
    INDEXER_SCORING_REFERENCE_ORDER,
    Ernie55LatentMQACoreAttention,
    Ernie55LatentMQACoreSubmodules,
    Ernie55LatentMQASelfAttention,
    context_parallel,
    latent_mqa_core,
    mtp_loss,
    ops,
    validate_ernie55_config,
)
from tests.unit_tests.models.ernie.ernie55_cp_harness import run_cases


pytestmark = pytest.mark.unit

# A scaled-down D layer (the D-layer tests' geometry) in a DSv4-hybrid stack config.
HIDDEN = 64
HEADS = 4
Q_LORA = 32
KV_LORA = 16
NOPE = 8
ROPE = 8
V_HEAD = 8
LATENT = KV_LORA + ROPE
VHA_RANK = 2
INDEXER_HEADS = 4
INDEXER_DIM = 16
WINDOW = 3
DENSE_TOPK = 32  # at least every remote candidate of the documents below: selection is not ranking
SELECT_TOPK = 4
ROTARY_BASE = 50.0  # small, so every positional channel rotates noticeably at these positions
COEFF = 0.01
EPS = 1e-6
SCALE = 1.0 / math.sqrt(NOPE + ROPE)

# Megatron-Core dev tracks the CP partition in the config; main has no such field.
_HAS_CP_PARTITION_MODE = "cp_partition_mode" in {field.name for field in dataclasses.fields(Ernie55TransformerConfig)}

# fp32 CP > 1 against fp32 CP = 1: only the order of the cross-rank sums differs.
TOL = 2e-5

# Packed layouts (every document a multiple of 2 * 4 rows, so each serves CP = 2 and CP = 4).
LAYOUTS = {
    # Boundaries 24, 32 and 48 cut the ranks' contiguous chunks at both CP sizes.
    "multi_document": dict(doc_lens=[24, 8, 16, 32]),
    "single_document": dict(doc_lens=[64]),
    # Documents with padding rows at their ends: the padded boundaries drive the layout.
    "padded_documents": dict(doc_lens=[21, 5, 13, 30], padded_lens=[24, 8, 16, 32]),
}


# ---------------------------------------------------------------------------------------------
# Transformer-Engine-free builders (shared by the pytest process and the ranks)
# ---------------------------------------------------------------------------------------------


def _sylvester_hadamard(x: torch.Tensor) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard transform over the last dim (fp64 math, input dtype out)."""
    size = x.size(-1)
    hadamard = torch.ones(1, 1, dtype=torch.float64, device=x.device)
    while hadamard.size(0) < size:
        hadamard = torch.cat([torch.cat([hadamard, hadamard], 1), torch.cat([hadamard, -hadamard], 1)], 0)
    return (x.double() @ hadamard * size**-0.5).to(x.dtype)


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
        self.weight = torch.nn.Parameter(torch.ones(hidden_size, dtype=config.params_dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(x.dtype, torch.float32)
        x_wide = x.to(dtype)
        normed = x_wide * torch.rsqrt(x_wide.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight.to(dtype)).to(x.dtype)


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


def _config(
    cp_size: int, *, stack: str = "dsv4_hybrid", topk: int = DENSE_TOPK, **overrides
) -> Ernie55TransformerConfig:
    """Finalized ERNIE 5.5 config at ``context_parallel_size=cp_size``.

    ``stack="dsv4_hybrid"`` is the production stack (H layers, contiguous CP partition);
    ``stack="dsa"`` a D-only stack, which Megatron-Core runs in the zigzag partition.
    """
    values = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_head_dim=NOPE,
        qk_pos_emb_head_dim=ROPE,
        v_head_dim=16,
        # The DSv4 post-init derives the stack's kv_lora_rank and qk_head_dim; the D layer takes its own.
        ernie_d_num_attention_heads=HEADS,
        ernie_d_q_lora_rank=Q_LORA,
        ernie_d_kv_lora_rank=KV_LORA,
        ernie_d_qk_head_dim=NOPE,
        ernie_d_qk_pos_emb_head_dim=ROPE,
        ernie_d_v_head_dim=V_HEAD,
        ernie_d_rotary_base=ROTARY_BASE,
        ernie_vha_rank=VHA_RANK,
        normalization="RMSNorm",
        layernorm_epsilon=EPS,
        rope_type="yarn",
        rotary_scaling_factor=40.0,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        add_bias_linear=False,
        experimental_attention_variant=stack,
        dsa_kernel_backend="none",
        dsa_indexer_n_heads=INDEXER_HEADS,
        dsa_indexer_head_dim=INDEXER_DIM,
        dsa_indexer_topk=topk,
        dsa_indexer_loss_coeff=COEFF,
        dsa_indexer_use_sparse_loss=True,
        dsa_indexer_k_norm_fp32=True,
        csa_window_size=WINDOW,
        context_parallel_size=cp_size,
        # Megatron-Core's DSA config requires it at CP > 1; the ERNIE D core always all-gathers its keys.
        cp_comm_type="all_gather",
        use_cpu_initialization=True,
        transformer_impl="local",
        bf16=True,
        params_dtype=torch.float32,
    )
    if stack == "dsv4_hybrid":
        values.update(
            output_projection_groups=2,
            output_projection_lora_rank=8,
            csa_compress_ratios=[128, 128],
            csa_compress_rotary_base=1000.0,
        )
        if _HAS_CP_PARTITION_MODE:  # Megatron-Core dev; DSv4 needs the contiguous partition at CP > 1
            values["cp_partition_mode"] = "contiguous"
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    return config


def _randomize_(module: torch.nn.Module, seed: int) -> None:
    """Draw every parameter at O(1) scale from a generator keyed by its name (sink and VHA included)."""
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


def _build_attention(config, pg_collection, *, seed: int) -> Ernie55LatentMQASelfAttention:
    attention = build_module(_attention_spec(), config=config, layer_number=1, pg_collection=pg_collection)
    _randomize_(attention, seed)
    return attention.train()


def _cumsum(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)


def _packed_seq_params(layout: dict, *, mode: str | None = None) -> PackedSeqParams:
    """Packed metadata of a layout; ``mode`` sets the CP partition mode (Megatron-Core dev only)."""
    padded = layout.get("padded_lens", layout["doc_lens"])
    cu_padded = _cumsum(padded)
    cu = _cumsum(layout["doc_lens"]) if "padded_lens" in layout else cu_padded
    extra = {} if mode is None else {"cp_partition_mode": mode}
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=cu_padded,
        cu_seqlens_kv_padded=cu_padded,
        max_seqlen_q=max(padded),
        max_seqlen_kv=max(padded),
        **extra,
    )


def _rows_of_rank(layout: dict, cp_size: int, cp_rank: int, mode: str) -> torch.Tensor:
    """Global packed rows a rank holds, written out independently of the code under test."""
    padded = layout.get("padded_lens", layout["doc_lens"])
    total = sum(padded)
    if mode == "contiguous":
        local = total // cp_size
        return torch.arange(cp_rank * local, (cp_rank + 1) * local)
    rows, start = [], 0
    for length in padded:
        chunk = length // (2 * cp_size)
        for index in (cp_rank, 2 * cp_size - 1 - cp_rank):
            rows.extend(range(start + index * chunk, start + (index + 1) * chunk))
        start += length
    return torch.tensor(rows)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str, tol: float = TOL) -> None:
    """Normalized max error ``max|a - e| / max|e|`` below ``tol``."""
    assert actual.shape == expected.shape, f"{what}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    scale = expected.abs().max().clamp_min(1e-12)
    error = float((actual.double() - expected.double()).abs().max() / scale)
    assert error <= tol, f"{what}: normalized max error {error:.3e} > {tol:.0e}"


# ---------------------------------------------------------------------------------------------
# Single-process tests: index geometry with explicit positions, metadata checks, config guards
# ---------------------------------------------------------------------------------------------


class TestPositionAwareIndexHelpers:
    """``positions`` puts the index geometry in global packed coordinates."""

    def test_positions_default_to_row_indices(self):
        params = _packed_seq_params(LAYOUTS["multi_document"])
        rows = torch.arange(80)
        assert torch.equal(
            ops.ernie55_document_starts(params, 80, "cpu"),
            ops.ernie55_document_starts(params, 80, "cpu", positions=rows),
        )
        doc_start = ops.ernie55_document_starts(params, 80, "cpu")
        for actual, expected in zip(
            ops.ernie55_selection_bounds(doc_start, WINDOW, positions=rows),
            ops.ernie55_selection_bounds(doc_start, WINDOW),
        ):
            assert torch.equal(actual, expected)
        assert torch.equal(
            ops.ernie55_window_indices(doc_start, window_size=WINDOW, batch_size=1, positions=rows),
            ops.ernie55_window_indices(doc_start, window_size=WINDOW, batch_size=1),
        )

    @pytest.mark.parametrize("cp_size", [2, 4])
    def test_rows_of_a_rank_match_the_full_pack(self, cp_size):
        """Any subset of rows, in any order, gets the bounds and window of those rows in the full pack."""
        layout = LAYOUTS["multi_document"]
        params = _packed_seq_params(layout)
        full_start = ops.ernie55_document_starts(params, 80, "cpu")
        full_bounds = ops.ernie55_selection_bounds(full_start, WINDOW)
        full_window = ops.ernie55_window_indices(full_start, window_size=WINDOW, batch_size=1)
        for cp_rank in range(cp_size):
            rows = _rows_of_rank(layout, cp_size, cp_rank, "zigzag")
            start = ops.ernie55_document_starts(params, rows.numel(), "cpu", positions=rows)
            assert torch.equal(start, full_start[rows])
            for local, full in zip(ops.ernie55_selection_bounds(start, WINDOW, positions=rows), full_bounds):
                assert torch.equal(local, full[rows])
            window = ops.ernie55_window_indices(start, window_size=WINDOW, batch_size=1, positions=rows)
            assert torch.equal(window, full_window[:, rows])

    def test_positions_must_have_one_entry_per_row(self):
        with pytest.raises(ValueError, match="positions"):
            ops.ernie55_selection_bounds(torch.zeros(4, dtype=torch.int64), WINDOW, positions=torch.arange(5))


def _host_metadata(doc_lens: list[int], **fields) -> types.SimpleNamespace:
    cu = [0, *torch.tensor(doc_lens).cumsum(0).tolist()]
    values = dict(
        qkv_format="thd",
        cp_partition_mode="contiguous",
        cp_partition_route=object(),
        thd_cp_host_cu_seqlens_q=cu,
        thd_cp_host_cu_seqlens_kv=list(cu),
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )
    values.update(fields)
    return types.SimpleNamespace(**values)


class TestThdCpMetadataChecks:
    """``check_thd_cp_metadata`` turns every unsupported micro-batch into a clear error on the host."""

    @staticmethod
    def _check(metadata, cp_size=2, local_rows=40, expected_mode="contiguous", need_route=True):
        return context_parallel.check_thd_cp_metadata(
            metadata,
            cp_size=cp_size,
            local_rows=local_rows,
            expected_mode=expected_mode,
            need_route=need_route,
            what="D layers",
        )

    def test_valid_pack(self):
        assert self._check(_host_metadata([24, 8, 16, 32])) == [0, 24, 32, 48, 80]

    @pytest.mark.parametrize(
        "metadata,kwargs,error,match",
        [
            (None, {}, NotImplementedError, "packed THD"),
            (_host_metadata([24, 8, 16, 32], qkv_format="sbhd"), {}, NotImplementedError, "packed THD"),
            (_host_metadata([24, 8, 16, 32], cp_partition_mode="zigzag"), {}, ValueError, "cp_partition_mode"),
            (_host_metadata([24, 8, 16, 32], cp_partition_route=None), {}, ValueError, "partition route"),
            (_host_metadata([24, 8, 16, 32], thd_cp_host_cu_seqlens_q=None), {}, ValueError, "host cu_seqlens"),
            (_host_metadata([24, 8, 16, 32], thd_cp_host_cu_seqlens_kv=[0, 24, 80]), {}, ValueError, "identical"),
            (_host_metadata([22, 10, 16, 32]), {}, ValueError, r"multiple of 2 \* context_parallel_size = 4"),
            (_host_metadata([24, 8, 16, 32]), {"local_rows": 41}, ValueError, "last document"),
            (_host_metadata([24, 8, 16, 32], max_seqlen_q=16), {}, ValueError, "max_seqlen_q"),
            (_host_metadata([24, 8, 16, 32], max_seqlen_kv=torch.tensor(32)), {}, ValueError, "max_seqlen_kv"),
        ],
        ids=[
            "sbhd",
            "not-thd",
            "wrong-partition",
            "no-route",
            "no-host-cu",
            "q-kv-differ",
            "misaligned",
            "tail-rows",
            "short-max-seqlen",
            "tensor-max-seqlen",
        ],
    )
    def test_rejects(self, metadata, kwargs, error, match):
        with pytest.raises(error, match=match):
            self._check(metadata, **kwargs)

    def test_route_is_optional_without_conversion(self):
        assert self._check(_host_metadata([24, 8, 16, 32], cp_partition_route=None), need_route=False)


class TestConfigGuards:
    """The CP envelope of the D and H validators."""

    def test_cp_is_accepted_on_megatron_core_dev(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        validate_ernie55_config(_config(4))
        validate_ernie55_config(_config(2, stack="dsa"))

    @pytest.mark.parametrize("per_row", [False, True], ids=["per-token", "per-row"])
    def test_mtp_with_cp_is_accepted(self, per_row):
        """The per-row MTP loss normalizes each depth over the CP group (test_ernie55_cp_mtp.py)."""
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        config = copy.copy(_config(2))
        config.mtp_num_layers = 1
        config.ernie_per_row_loss = per_row
        validate_ernie55_config(config)

    def test_per_row_mtp_needs_the_grouped_mtp_roll(self, monkeypatch):
        """A build with the CP converter but not dev's grouped MTP roll (main with #7128) fails at config time.

        Its model would keep Megatron-Core's MTP loss, which divides every depth by the rank's own tokens.
        """
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        monkeypatch.setattr(mtp_loss, "PER_ROW_MTP_LOSS_SUPPORTED", False)
        config = copy.copy(_config(2))
        config.mtp_num_layers = 1
        config.ernie_per_row_loss = True
        with pytest.raises(RuntimeError, match="grouped MTP roll"):
            validate_ernie55_config(config)
        # Megatron-Core's own MTP loss is right with per-token loss, and per-row at CP = 1 needs no CP group.
        config.ernie_per_row_loss = False
        validate_ernie55_config(config)
        cp1 = copy.copy(_config(1))
        cp1.mtp_num_layers = 1
        cp1.ernie_per_row_loss = True
        validate_ernie55_config(cp1)

    def test_builds_without_the_converter_raise(self, monkeypatch):
        """Megatron-Core main: CP > 1 raises instead of running AbsorbedMLA's zigzag RoPE on contiguous rows."""
        config = _config(1)
        cp_config = copy.copy(config)
        cp_config.context_parallel_size = 2
        monkeypatch.setattr(context_parallel, "load_cp_layout_api", lambda: None)
        with pytest.raises(RuntimeError, match="CP partition converter"):
            validate_ernie55_config(cp_config)
        validate_ernie55_config(config)

    def test_converter_lookup(self):
        api = context_parallel.load_cp_layout_api()
        if api is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        assert api.module in ("megatron.core.context_parallel.layout", "megatron.core.context_parallel_layout")
        assert callable(api.convert_module_input_tensors_cp_partition_mode)

    def test_converter_without_the_zigzag_helpers_is_rejected_at_config_time(self, monkeypatch):
        """A build with the converter but not dev's dsa_layout host builders (e.g. a partial main port)."""
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        from megatron.core.transformer.experimental_attention_variant import dsa_layout

        config = _config(2)
        missing = "build_packed_allgather_cp_query_positions_and_key_reorder_from_host"
        monkeypatch.delattr(dsa_layout, missing)
        context_parallel.load_cp_layout_api.cache_clear()
        try:
            assert context_parallel.load_cp_layout_api() is None
            with pytest.raises(RuntimeError, match=f"CP partition converter.*dsa_layout.{missing}"):
                validate_ernie55_config(config)
        finally:
            monkeypatch.undo()
            context_parallel.load_cp_layout_api.cache_clear()
        assert context_parallel.load_cp_layout_api() is not None

    def test_build_without_the_partition_mode_field_is_rejected(self, monkeypatch):
        """A build whose PackedSeqParams has no cp_partition_mode (Megatron-Core main's) fails at config time."""
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no CP partition converter")
        main_packed_seq_params = dataclasses.make_dataclass("PackedSeqParams", [("qkv_format", str, "thd")])
        holders = (context_parallel.TransformerConfig, main_packed_seq_params)
        monkeypatch.setattr(context_parallel, "_CP_PARTITION_MODE_HOLDERS", holders)
        context_parallel.load_cp_layout_api.cache_clear()
        try:
            assert context_parallel.load_cp_layout_api() is None
            with pytest.raises(RuntimeError, match=r"CP partition converter.*PackedSeqParams\.cp_partition_mode"):
                context_parallel.require_cp_layout_api("ERNIE 5.5 D layers")
        finally:
            monkeypatch.undo()
            context_parallel.load_cp_layout_api.cache_clear()
        assert context_parallel.load_cp_layout_api() is not None

    def test_main_error_names_every_missing_piece(self):
        """Megatron-Core main: the error lists what CP > 1 would need."""
        if context_parallel.load_cp_layout_api() is not None:
            pytest.skip("Megatron-Core dev has every piece")
        with pytest.raises(RuntimeError, match="CP partition converter") as error:
            context_parallel.require_cp_layout_api("ERNIE 5.5 D layers")
        message = str(error.value)
        for piece in (
            "convert_module_input_tensors_cp_partition_mode",
            "dsa_layout.build_packed_allgather_cp_query_positions_and_key_reorder_from_host",
            "TransformerConfig.cp_partition_mode",
            "PackedSeqParams.cp_partition_mode",
        ):
            assert piece in message, piece


class TestZigzagGeometry:
    """``zigzag_cp_geometry``: positions and key permutation of the zigzag rows, cached once per micro-batch."""

    @staticmethod
    def _geometry(metadata, cp_rank: int):
        group = types.SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
        return context_parallel.zigzag_cp_geometry(
            metadata, cp_group=group, local_rows=40, device=torch.device("cpu"), what="core"
        )

    def test_positions_and_key_order(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")
        doc_lens = [24, 8, 16, 32]
        rows = [_rows_of_rank(dict(doc_lens=doc_lens), 2, rank, "zigzag") for rank in range(2)]
        for rank in range(2):
            metadata = _host_metadata(doc_lens, cp_partition_mode="zigzag")
            geometry = self._geometry(metadata, rank)
            assert torch.equal(geometry.query_positions, rows[rank])
            # Keys gathered in rank order land in global packed order.
            assert torch.equal(torch.cat(rows).index_select(0, geometry.key_reorder), torch.arange(80))
            assert (geometry.cp_size, geometry.cp_rank, geometry.num_rows) == (2, rank, 80)

    def test_one_cache_entry(self):
        if context_parallel.load_cp_layout_api() is None:
            pytest.skip("Megatron-Core main has no zigzag CP helpers")
        metadata = _host_metadata([24, 8, 16, 32], cp_partition_mode="zigzag")
        first = self._geometry(metadata, 1)
        assert self._geometry(metadata, 1) is first
        # A PackedSeqParams reused for a micro-batch with other boundaries replaces the entry.
        metadata.thd_cp_host_cu_seqlens_q = [0, 16, 48, 80]
        metadata.thd_cp_host_cu_seqlens_kv = [0, 16, 48, 80]
        second = self._geometry(metadata, 1)
        assert second is not first
        assert torch.equal(second.query_positions, _rows_of_rank(dict(doc_lens=[16, 32, 32]), 2, 1, "zigzag"))
        key, cached = getattr(metadata, context_parallel._GEOMETRY_CACHE_ATTR)
        assert cached is second and key[-1] == (0, 16, 48, 80)


# ---------------------------------------------------------------------------------------------
# Rank side: setup and cases (run on CP gloo ranks by ernie55_cp_harness)
# ---------------------------------------------------------------------------------------------


@dataclass
class _Rank:
    rank: int
    world_size: int
    cp: ProcessGroupCollection
    """TP singleton, CP = every rank."""
    cp1: ProcessGroupCollection
    """TP singleton, CP singleton: the CP = 1 reference on this rank."""


def _cpu_pos_emb_on_this_cp_rank(pos_emb: torch.Tensor, seq_dim: int, cp_group) -> torch.Tensor:
    """Megatron-Core's ``get_pos_emb_on_this_cp_rank`` (zigzag chunks ``r`` and ``2 CP - 1 - r``) on CPU tensors."""
    cp_size, cp_rank = cp_group.size(), cp_group.rank()
    chunks = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=pos_emb.device)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[seq_dim + 1 :])
    pos_emb = pos_emb.index_select(seq_dim, chunks)
    return pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[seq_dim + 2 :])


def setup_rank(world_size: int) -> _Rank:
    """Model-parallel state with CP = ``world_size`` plus the CPU stand-ins Megatron-Core needs."""
    from megatron.core.models.common.embeddings import rotary_pos_embedding, yarn_rotary_pos_embedding

    # RoPE tables, the DSA loss tracker and gathers are allocated on torch.cuda.current_device().
    torch.cuda.current_device = lambda: "cpu"
    # DSAIndexer's Hadamard rotation needs fast_hadamard_transform; any orthonormal transform keeps the scores.
    dsa.rotate_activation = _sylvester_hadamard
    # The unpacked CP slice of a RoPE table copies through pinned CUDA memory; YaRN builds one at construction.
    for module in (rotary_pos_embedding, yarn_rotary_pos_embedding):
        if hasattr(module, "get_pos_emb_on_this_cp_rank"):
            module.get_pos_emb_on_this_cp_rank = _cpu_pos_emb_on_this_cp_rank
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=world_size
    )
    rank = torch.distributed.get_rank()
    singletons = [torch.distributed.new_group([index]) for index in range(world_size)]
    tp = parallel_state.get_tensor_model_parallel_group()
    cp = ProcessGroupCollection(tp=tp, cp=parallel_state.get_context_parallel_group())
    cp1 = ProcessGroupCollection(tp=tp, cp=singletons[rank])
    return _Rank(rank=rank, world_size=world_size, cp=cp, cp1=cp1)


def _prebuilt(params: PackedSeqParams, group) -> PackedSeqParams:
    context_parallel.require_cp_layout_api("tests").prebuild_thd_cp_partition_routes(params, group)
    return params


@dataclass
class _Run:
    output: torch.Tensor
    input_grad: torch.Tensor
    grads: dict[str, torch.Tensor]
    loss: torch.Tensor | None
    tracker: dict


def _run(attention, x, params, grad_output, *, indexer_loss_scale: float = 1.0, recompute: bool = False) -> _Run:
    """Forward and backward of one D attention; returns the output, gradients and the logged indexer loss."""
    tracker = dsa.DSAIndexerLossLoggingHelper.tracker
    tracker.clear()
    dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(indexer_loss_scale))
    attention.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)

    def forward(hidden_states):
        return attention(hidden_states, None, packed_seq_params=params)[0]

    if recompute:
        # Like Megatron-Core's full recompute: the forward runs without autograd (no indexer loss) and
        # the backward replays it with autograd, which computes and logs the indexer loss.
        output = torch.utils.checkpoint.checkpoint(forward, x, use_reentrant=True)
    else:
        output = forward(x)
    output.backward(grad_output)
    loss = tracker["values"][0].clone() if "values" in tracker else None
    snapshot = {key: tracker.get(key) for key in ("avg_group", "reduce_group")}
    grads = {
        name: param.grad.detach().clone() for name, param in attention.named_parameters() if param.grad is not None
    }
    return _Run(output.detach(), x.grad.detach(), grads, loss, snapshot)


def _inputs(layout: dict, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``x [T, 1, HIDDEN]`` and the output gradient of the whole pack, the same on every rank."""
    rows = sum(layout.get("padded_lens", layout["doc_lens"]))
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 1, HIDDEN, generator=generator, dtype=torch.float64).float()
    grad_output = torch.randn(rows, 1, HIDDEN, generator=generator, dtype=torch.float64).float()
    return x, grad_output


def _all_reduce_sum(tensor: torch.Tensor, group) -> torch.Tensor:
    tensor = tensor.detach().clone()
    torch.distributed.all_reduce(tensor, group=group)
    return tensor


def _compare_with_cp1(
    rank: _Rank,
    layout_name: str,
    *,
    stack: str = "dsv4_hybrid",
    seed: int,
    per_token: bool = False,
    recompute: bool = False,
    config_overrides: dict | None = None,
    attention_hook=None,
) -> tuple[_Run, _Run, torch.Tensor]:
    """Run one D attention at CP = world size and at CP = 1 on every rank and compare everything.

    The CP > 1 run emulates training: the output gradient and the indexer-loss scale are multiplied
    by ``CP`` (the schedule's per-row scaling) and the parameter gradients are averaged over the CP
    group (DDP), which must reproduce the CP = 1 gradients.
    """
    layout = LAYOUTS[layout_name]
    cp_size, cp_group = rank.world_size, rank.cp.cp
    mode = "contiguous" if stack == "dsv4_hybrid" else "zigzag"
    overrides = dict(calculate_per_token_loss=per_token, **(config_overrides or {}))
    config = _config(cp_size, stack=stack, **overrides)
    reference = _build_attention(config, rank.cp1, seed=seed)
    attention = _build_attention(config, rank.cp, seed=seed)
    if attention_hook is not None:
        attention_hook(reference)
        attention_hook(attention)
    x, grad_output = _inputs(layout, seed)
    rows = _rows_of_rank(layout, cp_size, rank.rank, mode)

    expected = _run(reference, x, _packed_seq_params(layout, mode=mode), grad_output)
    params = _prebuilt(_packed_seq_params(layout, mode=mode), cp_group)
    scale = 1.0 if per_token else float(cp_size)
    actual = _run(attention, x[rows], params, grad_output[rows] * scale, indexer_loss_scale=scale, recompute=recompute)
    assert params.cp_partition_mode == mode, f"cp_partition_mode left at {params.cp_partition_mode!r}"

    _assert_close(actual.output, expected.output[rows], "output rows")
    _assert_close(actual.input_grad, expected.input_grad[rows] * scale, "input gradient rows")
    assert actual.grads.keys() == expected.grads.keys()
    for name, grad in actual.grads.items():
        total = _all_reduce_sum(grad, cp_group)
        _assert_close(total / scale, expected.grads[name], f"gradient of {name}")
    # Each rank's per-row KL is normalized by the rows of the whole pack, so the ranks' losses add up.
    _assert_close(_all_reduce_sum(actual.loss, cp_group), expected.loss, "indexer loss summed over CP")
    if per_token:
        assert actual.tracker["reduce_group"] is cp_group and actual.tracker["avg_group"] is None
    else:
        assert actual.tracker["avg_group"] is cp_group and actual.tracker["reduce_group"] is None
    return actual, expected, rows


def case_dense_multi_document(rank: _Rank) -> None:
    """Every remote candidate selected; document boundaries cut the ranks' contiguous chunks."""
    _compare_with_cp1(rank, "multi_document", seed=11)


def case_dense_single_document(rank: _Rank) -> None:
    """One document over the whole pack (the production shape)."""
    _compare_with_cp1(rank, "single_document", seed=12)


def case_dense_padded_documents(rank: _Rank) -> None:
    """Documents with padding rows; the padded boundaries define the documents."""
    _compare_with_cp1(rank, "padded_documents", seed=13)


def case_per_token_loss(rank: _Rank) -> None:
    """Per-token loss keeps Megatron-Core's convention: the coefficient is not divided by CP."""
    _compare_with_cp1(rank, "multi_document", seed=14, per_token=True)


def case_zigzag_stack(rank: _Rank) -> None:
    """A D-only ``"dsa"`` stack is already zigzag: no conversion, the same zigzag core."""
    _compare_with_cp1(rank, "multi_document", stack="dsa", seed=15)


def case_full_recompute(rank: _Rank) -> None:
    """A recompute replay converts again from the restored stack partition and gives the same gradients."""
    _compare_with_cp1(rank, "multi_document", seed=16, recompute=True)


def _designed_indexer(module: Ernie55LatentMQASelfAttention) -> None:
    """Make the indexer score key ``s`` by ``x_s[0]`` alone, for every query (no near-ties).

    The injected ``q``/``k``/``w`` are functions of the layer input rows, so they reach every rank
    through the real partition conversion.
    """
    core = module.core_attention

    def forward_before_topk(x, qr, packed_seq_params=None):
        rows, batch = x.shape[:2]
        q = torch.zeros(rows, batch, INDEXER_HEADS, INDEXER_DIM, dtype=x.dtype)
        q[..., 0] = 1.0
        k = torch.zeros(rows, batch, INDEXER_DIM, dtype=x.dtype)
        k[..., 0] = x[..., 0]
        w = torch.full((rows, batch, INDEXER_HEADS), 1.0 / INDEXER_HEADS, dtype=x.dtype)
        return q, k, w

    core.indexer.forward_before_topk = forward_before_topk
    recorded = []
    original = core._sparse_attention

    def recording(query, key_mqa, indices, **kwargs):
        recorded.append(indices.detach().clone())
        return original(query, key_mqa, indices, **kwargs)

    core._sparse_attention = recording
    module.recorded_indices = recorded


def case_selection_matches_cp1(rank: _Rank) -> None:
    """With K < candidates, every row selects the same global keys (and window) as at CP = 1."""
    layout = LAYOUTS["multi_document"]
    cp_size, cp_group = rank.world_size, rank.cp.cp
    config = _config(cp_size, topk=SELECT_TOPK)
    reference = _build_attention(config, rank.cp1, seed=21)
    attention = _build_attention(config, rank.cp, seed=21)
    _designed_indexer(reference)
    _designed_indexer(attention)
    x, grad_output = _inputs(layout, 21)
    # Distinct, well separated indexer scores x_s[0] in [1, T].
    x[:, 0, 0] = torch.randperm(x.size(0), generator=torch.Generator().manual_seed(22)).float() + 1.0
    rows = _rows_of_rank(layout, cp_size, rank.rank, "contiguous")

    expected = _run(reference, x, _packed_seq_params(layout, mode="contiguous"), grad_output)
    params = _prebuilt(_packed_seq_params(layout, mode="contiguous"), cp_group)
    actual = _run(attention, x[rows], params, grad_output[rows])
    (expected_indices,), (actual_indices,) = reference.recorded_indices, attention.recorded_indices
    # The rank's rows reach the core in zigzag order; map them back through the zigzag geometry.
    zigzag_rows = _rows_of_rank(layout, cp_size, rank.rank, "zigzag")
    for local, global_row in enumerate(zigzag_rows.tolist()):
        for part, width in (("top-k", slice(0, SELECT_TOPK)), ("window", slice(SELECT_TOPK, None))):
            got = sorted(index for index in actual_indices[0, local, width].tolist() if index >= 0)
            want = sorted(index for index in expected_indices[0, global_row, width].tolist() if index >= 0)
            assert got == want, f"row {global_row} {part}: {got} != {want}"
    selected = (expected_indices[0, :, :SELECT_TOPK] >= 0).sum(-1)
    assert int(selected.max()) == SELECT_TOPK and int(selected.min()) == 0, (
        "layout must exercise ranking and empty rows"
    )
    _assert_close(actual.output, expected.output[rows], "output rows")
    for name, grad in actual.grads.items():
        _assert_close(_all_reduce_sum(grad, cp_group), expected.grads[name], f"gradient of {name}")


def case_data_helper_batch_feeds_the_d_layer(rank: _Rank) -> None:
    """``build_ernie55_thd_cp_batch`` pads unaligned samples and prebuilds routes; the D layer runs on it.

    The CP = 1 reference runs the same padded pack (the samples padded to ``2 * CP`` beforehand).
    No layer may fall back to Megatron-Core's route rebuild (a FutureWarning and a device sync).
    """
    from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, build_ernie55_thd_cp_batch

    cp_size, cp_group = rank.world_size, rank.cp.cp
    generator = torch.Generator().manual_seed(71)
    vocab = 97
    embedding = torch.randn(vocab, HIDDEN, generator=generator, dtype=torch.float64).float()
    samples = []
    for length in (5, 2 * cp_size + 3, 13):
        tokens = torch.randint(1, vocab, (length,), generator=generator)
        samples.append((tokens, tokens.clone()))
    batch = build_ernie55_thd_cp_batch(samples, pad_token_id=0, cp_group=cp_group)
    padded_samples = []
    for tokens, labels in samples:
        pad = -tokens.numel() % (2 * cp_size)
        padded_samples.append((F.pad(tokens, (0, pad)), F.pad(labels, (0, pad), value=IGNORE_LABEL)))
    full = build_ernie55_thd_cp_batch(padded_samples, pad_token_id=0, cp_group=None)
    local = full.tokens.size(1) // cp_size
    rows = torch.arange(rank.rank * local, (rank.rank + 1) * local)
    assert torch.equal(batch.tokens[0], full.tokens[0, rows]) and torch.equal(batch.labels[0], full.labels[0, rows])

    config = _config(cp_size)
    reference = _build_attention(config, rank.cp1, seed=71)
    attention = _build_attention(config, rank.cp, seed=71)
    x = embedding[full.tokens[0]].unsqueeze(1)
    grad_output = torch.randn(x.shape, generator=generator, dtype=torch.float64).float()
    expected = _run(reference, x, full.packed_seq_params, grad_output)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = _run(
            attention,
            embedding[batch.tokens[0]].unsqueeze(1),
            batch.packed_seq_params,
            grad_output[rows] * cp_size,
            indexer_loss_scale=float(cp_size),
        )
    route_warnings = [str(w.message) for w in caught if "precomputed context-parallel layout routes" in str(w.message)]
    assert not route_warnings, route_warnings
    _assert_close(actual.output, expected.output[rows], "output rows")
    for name, grad in actual.grads.items():
        _assert_close(_all_reduce_sum(grad, cp_group) / cp_size, expected.grads[name], f"gradient of {name}")
    _assert_close(_all_reduce_sum(actual.loss, cp_group), expected.loss, "indexer loss summed over CP")


def case_mode_restored_after_error(rank: _Rank) -> None:
    """An exception inside AbsorbedMLA leaves the stack partition in place."""
    layout = LAYOUTS["multi_document"]
    attention = _build_attention(_config(rank.world_size), rank.cp, seed=31)

    def failing_core(*args, **kwargs):
        raise RuntimeError("core failed")

    attention.core_attention.forward = failing_core
    params = _prebuilt(_packed_seq_params(layout, mode="contiguous"), rank.cp.cp)
    x, _ = _inputs(layout, 31)
    rows = _rows_of_rank(layout, rank.world_size, rank.rank, "contiguous")
    with pytest.raises(RuntimeError, match="core failed"):
        attention(x[rows], None, packed_seq_params=params)
    assert params.cp_partition_mode == "contiguous"


def case_guards(rank: _Rank) -> None:
    """Unsupported CP inputs raise before any communication."""
    layout = LAYOUTS["multi_document"]
    attention = _build_attention(_config(rank.world_size), rank.cp, seed=41)
    x, _ = _inputs(layout, 41)
    rows = _rows_of_rank(layout, rank.world_size, rank.rank, "contiguous")
    with pytest.raises(NotImplementedError, match="packed THD"):
        attention(x[rows], None, packed_seq_params=None)
    with pytest.raises(ValueError, match="partition route"):
        attention(x[rows], None, packed_seq_params=_packed_seq_params(layout, mode="contiguous"))
    wrong_mode = _prebuilt(_packed_seq_params(layout, mode="contiguous"), rank.cp.cp)
    wrong_mode.cp_partition_mode = "zigzag"
    with pytest.raises(ValueError, match="cp_partition_mode"):
        attention(x[rows], None, packed_seq_params=wrong_mode)
    with pytest.raises(ValueError, match="last document"):
        attention(
            x[rows][:-2], None, packed_seq_params=_prebuilt(_packed_seq_params(layout, mode="contiguous"), rank.cp.cp)
        )
    # Megatron-Core's own route builder rejects documents that are not multiples of 2 * CP.
    misaligned = dict(doc_lens=[2 * rank.world_size + 2, 6 * rank.world_size - 2])
    with pytest.raises(ValueError, match="divisible by"):
        _prebuilt(_packed_seq_params(misaligned, mode="contiguous"), rank.cp.cp)
    # The core alone needs zigzag rows.
    core_params = _prebuilt(_packed_seq_params(layout, mode="contiguous"), rank.cp.cp)
    with pytest.raises(ValueError, match="cp_partition_mode='zigzag'"):
        context_parallel.zigzag_cp_geometry(
            core_params, cp_group=rank.cp.cp, local_rows=rows.numel(), device=torch.device("cpu"), what="core"
        )


def case_megatron_core_without_converter(rank: _Rank) -> None:
    """Without the CP converter (Megatron-Core main), D layers refuse CP > 1 at build time and in forward."""
    config = _config(rank.world_size)
    attention = _build_attention(config, rank.cp, seed=51)
    original = context_parallel.load_cp_layout_api
    context_parallel.load_cp_layout_api = lambda: None
    try:
        with pytest.raises(RuntimeError, match="CP partition converter"):
            _build_attention(config, rank.cp, seed=51)
        layout = LAYOUTS["multi_document"]
        x, _ = _inputs(layout, 51)
        rows = _rows_of_rank(layout, rank.world_size, rank.rank, "contiguous")
        with pytest.raises(RuntimeError, match="CP partition converter"):
            attention(x[rows], None, packed_seq_params=_packed_seq_params(layout, mode="contiguous"))
        with pytest.raises(RuntimeError, match="CP partition converter"):
            Ernie55LatentMQACoreAttention(
                attention.config,
                _core_submodules(),
                layer_number=1,
                attn_mask_type=AttnMaskType.causal,
                attention_type="self",
                softmax_scale=SCALE,
                k_channels=LATENT,
                v_channels=KV_LORA,
                pg_collection=rank.cp,
            )
    finally:
        context_parallel.load_cp_layout_api = original


# --- fused-path wiring with CPU stand-ins for the kernels ------------------------------------


def _reference_topk_hook(config, q, k, weights, index_topk, starts, ends, block_size, use_relu=True, **kwargs):
    _, topk = dsa.fused_qk_topk_naive(
        q, k, weights, index_topk, varlen_starts=starts, varlen_ends=ends, use_relu=use_relu
    )
    return topk.int(), (topk >= 0).sum(-1).int()


def _reference_topk_with_loss_hook(
    config, q, k, weights, index_topk, starts, ends, block_size, *, query, key, softmax_scale, loss_coeff,
    pg_collection, query_valid_rows=None, calculate_per_token_loss=False, use_relu=True, **kwargs,
):  # fmt: skip
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    index_scores, topk = dsa.fused_qk_topk_naive(
        q, k, weights, index_topk, varlen_starts=starts, varlen_ends=ends, use_relu=use_relu
    )
    loss = dsa.compute_dsa_indexer_loss(
        # Like the cuDNN hook: the scores scaled by the module-level scale, read at call time.
        index_scores=index_scores * dsa_cudnn_kernels._INDEXER_SOFTMAX_SCALE,
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


def _dense_csa_sparse_attn(query, kv, attn_sink, topk_idxs, softmax_scale, topk_length=None, **kwargs):
    """CPU stand-in for ``csa_sparse_attn`` over flat indices into ``kv`` (``d_v = KV_LORA``)."""
    seqlen, batch, heads, dim = query.shape
    q, keys_flat = query.reshape(seqlen * batch, heads, dim), kv.reshape(-1, dim)
    valid = topk_idxs >= 0
    if topk_length is not None:
        valid = valid & (torch.arange(topk_idxs.size(-1)).view(1, -1) < topk_length.view(-1, 1))
    keys = keys_flat[topk_idxs.clamp_min(0).long()]
    logits = (torch.einsum("rhd,rmd->rhm", q, keys) * softmax_scale).masked_fill(~valid.unsqueeze(1), float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)
    probs = torch.exp(logits - torch.logaddexp(lse, attn_sink.view(1, -1).to(lse.dtype)).unsqueeze(-1))
    out = torch.einsum("rhm,rmd->rhd", probs, keys[..., :KV_LORA])
    return out.reshape(seqlen, batch, heads * KV_LORA)


def _dense_indexer_forward_wrapper(q, k, weights, *, ratio, sm_scale, q_causal_offsets=None):
    """CPU stand-in for cuDNN's ``indexer_forward_wrapper`` (the CP-1 selection): ``{"scores": [b, sq, sk]}``."""
    scores = torch.einsum("bshd,btd->bsht", q.float(), k[:, :, 0].float()).relu()
    scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2) * sm_scale
    offsets = torch.zeros(q.size(0), dtype=torch.int64) if q_causal_offsets is None else q_causal_offsets.long()
    rows = torch.arange(q.size(1)).view(1, -1, 1) + offsets.view(-1, 1, 1)
    return {"scores": scores.masked_fill(torch.arange(k.size(1)).view(1, 1, -1) > rows, float("-inf"))}


def _install_fused_stand_ins() -> list[tuple[str, dict]]:
    """Replace Megatron-Core's fused entry points with CPU stand-ins bound to their real signatures."""
    calls = []

    def install(module, name, stand_in):
        signature = inspect.signature(getattr(module, name))

        @functools.wraps(stand_in)
        def recording(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append((name, dict(bound.arguments)))
            return stand_in(*args, **kwargs)

        setattr(module, name, recording)

    install(dsa_kernels, "run_fused_qk_topk", _reference_topk_hook)
    install(dsa_kernels, "run_fused_qk_topk_with_loss", _reference_topk_with_loss_hook)
    install(fused_sparse_attention, "csa_sparse_attn", _dense_csa_sparse_attn)

    def scorer(*args, **kwargs):
        calls.append(("indexer_forward_wrapper", dict(kwargs)))
        return _dense_indexer_forward_wrapper(*args, **kwargs)

    latent_mqa_core._reference_indexer_scorer = lambda q: scorer
    latent_mqa_core._check_fused_attention_inputs = lambda query, key: None
    latent_mqa_core._check_fused_geometry = lambda config: None
    return calls


@contextlib.contextmanager
def _fused_stand_ins():
    """Run the block with the CPU stand-ins of the fused kernels; yields the recorded calls."""
    targets = [
        (dsa_kernels, "run_fused_qk_topk"),
        (dsa_kernels, "run_fused_qk_topk_with_loss"),
        (fused_sparse_attention, "csa_sparse_attn"),
        (latent_mqa_core, "_reference_indexer_scorer"),
        (latent_mqa_core, "_check_fused_attention_inputs"),
        (latent_mqa_core, "_check_fused_geometry"),
    ]
    saved = [(module, name, getattr(module, name)) for module, name in targets]
    try:
        yield _install_fused_stand_ins()
    finally:
        for module, name, value in saved:
            setattr(module, name, value)


def _fused_config(cp_size: int) -> Ernie55TransformerConfig:
    config = copy.copy(_config(cp_size))
    config.dsa_kernel_backend = "cudnn"  # finalize would need cuDNN for it
    return config


def _hook_calls(recorded: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    return [(name, args) for name, args in recorded if name.startswith("run_fused_qk_topk")]


def _assert_packed_hook_args(args: dict, params: PackedSeqParams, rank: _Rank | None, local_rows: int) -> None:
    """The arguments that select Megatron-Core's PACKED_THD_GENERAL scorer, on the pack's padded boundaries.

    ``rank`` names this rank of the zigzag CP group; ``None`` is CP = 1 (one rank holds every row).
    """
    cp_size = rank.world_size if rank is not None else 1
    assert args["use_local_indexer_varlen"] is True and args["single_packed_thd_sequence"] is False
    assert args["varlen_is_plain_causal"] is False and args["cp_size"] == cp_size
    assert args["local_packed_cp_rank"] == (rank.rank if rank is not None else 0)
    assert args["local_packed_cp_query_start"] == 0
    assert args["local_packed_cp_query_len"] == local_rows
    assert args["k"].size(0) == local_rows * cp_size
    hook_params = args["packed_seq_params"]
    # One tensor in all four fields (the scorer's same-view requirement)...
    cu_seqlens = hook_params.cu_seqlens_q
    for field in ("cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded"):
        assert getattr(hook_params, field) is cu_seqlens, f"{field} is not the cu_seqlens_q tensor"
    # ... holding the padded boundaries, which the rows and the zigzag partition follow ...
    assert torch.equal(cu_seqlens, params.cu_seqlens_q_padded), (cu_seqlens, params.cu_seqlens_q_padded)
    # ... and the pack's longest padded document, not this rank's row count.
    assert (hook_params.max_seqlen_q, hook_params.max_seqlen_kv) == (params.max_seqlen_q, params.max_seqlen_kv)


def case_fused_path_wiring(rank: _Rank) -> None:
    """A training forward at CP > 1 hands the loss hook the zigzag PACKED_THD_GENERAL arguments and ``coeff / CP``.

    At CP = 1 the attention's top-k comes from the core's reference-order scorer, and the loss hook gets the
    PACKED_THD_GENERAL arguments of a single rank. The padded layout tells the padded document boundaries (the
    hook's) from the unpadded ones.
    """
    layout = LAYOUTS["padded_documents"]
    cp_size, cp_group = rank.world_size, rank.cp.cp
    with _fused_stand_ins() as calls:
        config = _fused_config(cp_size)
        reference = _build_attention(config, rank.cp1, seed=61)
        attention = _build_attention(config, rank.cp, seed=61)
        assert attention.core_attention.use_fused
        x, grad_output = _inputs(layout, 61)
        rows = _rows_of_rank(layout, cp_size, rank.rank, "contiguous")
        cp1_params = _packed_seq_params(layout, mode="contiguous")
        expected = _run(reference, x, cp1_params, grad_output)
        assert reference.core_attention.requested_indexer_scoring_plan == INDEXER_SCORING_REFERENCE_ORDER
        assert reference.core_attention.requested_indexer_loss_scoring_plan == INDEXER_SCORING_PACKED_THD_GENERAL
        reference_calls = len(calls)
        params = _prebuilt(_packed_seq_params(layout, mode="contiguous"), cp_group)
        assert not torch.equal(params.cu_seqlens_q, params.cu_seqlens_q_padded), "the layout must be padded"
        actual = _run(attention, x[rows], params, grad_output[rows] * cp_size, indexer_loss_scale=float(cp_size))
        assert attention.core_attention.requested_indexer_scoring_plan == INDEXER_SCORING_PACKED_THD_GENERAL

    ((cp1_name, cp1_args),) = _hook_calls(calls[:reference_calls])
    assert cp1_name == "run_fused_qk_topk_with_loss" and cp1_args["loss_coeff"] == COEFF
    _assert_packed_hook_args(cp1_args, cp1_params, None, x.size(0))
    assert [name for name, _ in calls[:reference_calls]].count("indexer_forward_wrapper") >= 1
    assert "indexer_forward_wrapper" not in [name for name, _ in calls[reference_calls:]]
    ((name, args),) = _hook_calls(calls[reference_calls:])
    assert name == "run_fused_qk_topk_with_loss"
    _assert_packed_hook_args(args, params, rank, rows.numel())
    assert args["key"].size(0) == rows.numel() * cp_size
    assert args["loss_coeff"] == pytest.approx(COEFF / cp_size)
    _assert_close(actual.output, expected.output[rows], "fused output rows")
    for name, grad in actual.grads.items():
        _assert_close(_all_reduce_sum(grad, cp_group) / cp_size, expected.grads[name], f"fused gradient of {name}")


def case_fused_no_loss_path_wiring(rank: _Rank) -> None:
    """Forwards without the indexer loss hand the plain top-k hook the same packed arguments.

    That hook serves evaluation and the no-grad forward of full recompute (every D-layer forward of a
    run with full recompute; the replay takes the loss hook), so all three are checked, and the replay
    must give the loss hook the same indexer inputs and bounds as the no-grad forward.
    """
    layout = LAYOUTS["padded_documents"]
    cp_size, cp_group = rank.world_size, rank.cp.cp
    with _fused_stand_ins() as calls:
        config = _fused_config(cp_size)
        reference = _build_attention(config, rank.cp1, seed=62)
        attention = _build_attention(config, rank.cp, seed=62)
        x, grad_output = _inputs(layout, 62)
        rows = _rows_of_rank(layout, cp_size, rank.rank, "contiguous")
        expected = _run(reference, x, _packed_seq_params(layout, mode="contiguous"), grad_output)
        params = _prebuilt(_packed_seq_params(layout, mode="contiguous"), cp_group)
        core = attention.core_attention

        def forward(label: str) -> tuple[torch.Tensor, list]:
            start = len(calls)
            core.requested_indexer_scoring_plan = None
            output = attention(x[rows], None, packed_seq_params=params)[0]
            assert core.requested_indexer_scoring_plan == INDEXER_SCORING_PACKED_THD_GENERAL, label
            return output.detach(), _hook_calls(calls[start:])

        attention.eval()
        eval_output, eval_calls = forward("evaluation")
        attention.train()
        with torch.no_grad():
            no_grad_output, no_grad_calls = forward("no-grad forward")
        start = len(calls)
        recomputed = _run(
            attention, x[rows], params, grad_output[rows] * cp_size, indexer_loss_scale=float(cp_size), recompute=True
        )
        recompute_calls = _hook_calls(calls[start:])
        assert core.requested_indexer_scoring_plan == INDEXER_SCORING_PACKED_THD_GENERAL

    local_rows = rows.numel()
    for label, hooks in (("evaluation", eval_calls), ("no-grad forward", no_grad_calls)):
        assert [name for name, _ in hooks] == ["run_fused_qk_topk"], label
        _assert_packed_hook_args(hooks[0][1], params, rank, local_rows)
    assert [name for name, _ in recompute_calls] == ["run_fused_qk_topk", "run_fused_qk_topk_with_loss"]
    (_, forward_args), (_, replay_args) = recompute_calls
    for args in (forward_args, replay_args):
        _assert_packed_hook_args(args, params, rank, local_rows)
    for name in ("q", "k", "weights", "starts", "ends"):
        assert torch.equal(forward_args[name], replay_args[name]), f"the recompute replay changed the hook's {name}"
    assert replay_args["loss_coeff"] == pytest.approx(COEFF / cp_size)
    for label, output in (("evaluation", eval_output), ("no-grad", no_grad_output), ("recompute", recomputed.output)):
        _assert_close(output, expected.output[rows], f"{label} output rows")
    _assert_close(recomputed.input_grad, expected.input_grad[rows] * cp_size, "recompute input gradient rows")
    for name, grad in recomputed.grads.items():
        _assert_close(_all_reduce_sum(grad, cp_group) / cp_size, expected.grads[name], f"recompute gradient of {name}")


def case_hca_layer_builds_with_plain_rope(rank: _Rank) -> None:
    """H layers build at CP > 1, and every rotary embedding they can reach is one plain RoPE on the CP group."""
    from megatron.core.models.common.embeddings import RotaryEmbedding, YarnRotaryEmbedding
    from megatron.core.transformer.experimental_attention_variant import deepseek_v4_hybrid_attention as dsv4

    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HCASelfAttention, ernie55_hybrid_stack_spec
    from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import rotary_embedding_references

    if not callable(getattr(dsv4.DSv4HybridAttention, "_post_o_group_proj", None)):
        return  # Bridge's own Megatron-Core pins lack the hook the H layer needs (the H tests skip too).
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TELinear,
        TENorm,
        TERowParallelLinear,
    )

    stand_ins = [
        (TELinear, _DuplicatedLinear),
        (TEColumnParallelLinear, ColumnParallelLinear),
        (TERowParallelLinear, RowParallelLinear),
        (TENorm, _RMSNorm),
    ]

    def substitute(node, memo):
        if id(node) in memo:
            return memo[id(node)]
        for original, replacement in stand_ins:
            if node is original:
                return replacement
        if isinstance(node, (list, tuple)):
            result = type(node)(substitute(item, memo) for item in node)
        elif isinstance(node, dict):
            result = {key: substitute(value, memo) for key, value in node.items()}
        elif isinstance(node, functools.partial):
            result = functools.partial(
                substitute(node.func, memo), *substitute(node.args, memo), **substitute(node.keywords, memo)
            )
        elif hasattr(node, "__dataclass_fields__") and not isinstance(node, type):
            result = copy.copy(node)
            for field_name in node.__dataclass_fields__:
                object.__setattr__(result, field_name, substitute(getattr(node, field_name), memo))
        else:
            return node
        memo[id(node)] = result
        return result

    saved_te_linear = dsv4.TELinear
    dsv4.TELinear = _DuplicatedLinear
    try:
        spec = substitute(ernie55_hybrid_stack_spec().submodules.csa_qk_layernorm_layer.submodules.self_attention, {})
        # DSv4 asserts the Transformer Engine implementation; the stand-ins replace its modules.
        config = _config(rank.world_size, transformer_impl="transformer_engine")
        attention = build_module(spec, config=config, layer_number=1, pg_collection=rank.cp, compress_ratio=128)
    finally:
        dsv4.TELinear = saved_te_linear
    assert isinstance(attention, Ernie55HCASelfAttention)
    references = rotary_embedding_references(attention)
    assert {id(embedding) for _, embedding in references} == {id(attention.rotary_pos_emb)}
    assert {path for path, _ in references} >= {"rotary_pos_emb", "core_attention.rotary_pos_emb"}
    rope = attention.rotary_pos_emb
    assert type(rope) is RotaryEmbedding and not isinstance(rope, YarnRotaryEmbedding)
    assert rope.cp_group is rank.cp.cp


CASES = [
    "case_dense_multi_document",
    "case_dense_single_document",
    "case_dense_padded_documents",
    "case_per_token_loss",
    "case_zigzag_stack",
    "case_full_recompute",
    "case_selection_matches_cp1",
    "case_data_helper_batch_feeds_the_d_layer",
    "case_mode_restored_after_error",
    "case_guards",
    "case_megatron_core_without_converter",
    "case_fused_path_wiring",
    "case_fused_no_loss_path_wiring",
    "case_hca_layer_builds_with_plain_rope",
]


# ---------------------------------------------------------------------------------------------
# pytest side
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[2, 4], ids=["cp2", "cp4"])
def cp_results(request, tmp_path_factory):
    """Run every case once on ``CP`` spawned gloo ranks."""
    if context_parallel.load_cp_layout_api() is None:
        pytest.skip("Megatron-Core main has no CP partition converter; ERNIE 5.5 rejects CP > 1 there")
    return run_cases(
        world_size=request.param,
        module_name=__name__,
        setup_name="setup_rank",
        case_names=CASES,
        workdir=tmp_path_factory.mktemp(f"ernie55_cp{request.param}"),
    )


@pytest.mark.parametrize("case", CASES, ids=[case.removeprefix("case_") for case in CASES])
def test_context_parallel(cp_results, case):
    failures = {rank: result["error"] for rank, result in cp_results[case].items() if not result["ok"]}
    assert len(cp_results[case]) > 1, "no rank reported"
    assert not failures, "\n".join(f"--- rank {rank} ---\n{error}" for rank, error in failures.items())
