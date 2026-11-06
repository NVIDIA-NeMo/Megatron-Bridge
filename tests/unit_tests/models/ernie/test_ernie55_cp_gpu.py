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

"""GPU tests of ERNIE 5.5 context parallelism on one node: CP = 4 against CP = 1, fused kernels.

Needs four GPUs (SM90+), Transformer Engine, ``fast_hadamard_transform``, FlashMLA, the cuDNN frontend
DSA kernels with per-segment query offsets (>= 1.26), CuTe DSL, and a Megatron-Core dev build with the
CP partition converter and the DSv4 ``_post_o_group_proj`` hook (dev f78809f6d or later). Launch on all four
ranks::

    ERNIE55_GPU_STRICT=1 uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_cp_gpu.py

Without them the tests skip and name what is missing; ``ERNIE55_GPU_STRICT=1`` turns the skips into
failures. Every rank runs the same tests; failures are gathered so all ranks fail together.

Each comparison builds a module once on the CP = 4 group and once on a single-rank group with the
same weights (rank 0's, broadcast), runs the whole pack at CP = 1 and this rank's contiguous rows at
CP = 4, and emulates training at CP = 4: the output gradient and the indexer-loss scale are multiplied
by CP (the schedule's per-row scaling) and the parameter gradients are averaged over CP (DDP).

Gates: output cosine >= 0.9999 and relative L2 <= 1e-2; every gradient, the input's included, cosine >= 0.9999
and relative L2 <= 1e-2. CP = 1 runs twice to measure its own run-to-run noise, which is logged next to every
gradient's margin; a gradient whose CP = 1 noise exceeds that bar (atomics on a build without the
deterministic backward) is gated at 3 times its noise instead.

* ``test_d_layer_cp4_matches_cp1``: the D layer (zigzag inside the contiguous DSv4 stack) on a
  32K single-document pack, an 8K multi-document pack whose documents cross the ranks' chunks, and
  the same 8K pack with padding rows (``cu_seqlens_q`` differs from ``cu_seqlens_q_padded``): output
  and gradients within the gates; top-k differs only at near-ties of the CP = 1 fp32 indexer scores;
  the indexer KL summed over CP within 1e-4 of CP = 1; the indexer gradients' norm within 10% of
  CP = 1, not CP times it. CP = 4 must hand the loss hook the zigzag ``PACKED_THD_GENERAL`` arguments
  on the padded boundaries and score with that cuDNN plan (its executor runs, the single-sequence
  segment executor never does, Megatron-Core #8534). CP = 1 selects the attention's top-k with the core's
  reference-order scorer and runs the loss hook with ``PACKED_THD_GENERAL`` too.
* ``test_d_layer_cp4_full_recompute``: the D layer at CP = 4 through Megatron-Core's full recompute
  (``tensor_parallel.checkpoint``) on the padded 8K pack. The no-grad forward (the plain top-k hook,
  as in every D-layer forward of a run with full recompute) and the replay (the loss hook) must both
  get the packed arguments and run the ``PACKED_THD_GENERAL`` executor, select identical top-k sets,
  and give the forward's output again; output and gradients match CP = 1 within the gates.
* ``test_d_layer_cp4_packed_scoring_speedup``: forward plus backward of the D layer at CP = 4 on the
  32K pack with ``PACKED_THD_GENERAL`` against the same layer forced onto ``UNFUSED_BOUNDS``;
  logs both times and requires a speedup.
* ``test_h_layer_cp4_matches_cp1``: the H layer (Megatron-Core's contiguous DSv4 CP with the ERNIE
  gate, VHA and plain RoPE) on an 8K pack whose rank boundaries cut compression blocks: output and
  gradients within the gates. ``[eager]`` rotates the context-parallel rows with Megatron-Core's
  ``apply_thd_cp_local_rope_unfused``. ``[fused]`` first installs ``ernie_fused_ops`` from the layer's
  config, as building an ``Ernie55HybridModel`` does, so the q, kv and inverse-output RoPE at CP = 4
  must run the ERNIE 5.5 Triton kernel (at least two forward launches and one inverse launch); the
  CP = 1 reference never calls the context-parallel RoPE. Needs Triton.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
import itertools
import logging
import os

import pytest
import torch


logger = logging.getLogger(__name__)

_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"
CP = 4

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() and not _STRICT, reason="the ERNIE 5.5 CP GPU tests need CUDA devices"
    ),
]

_COMMON_CONFIG = dict(
    normalization="RMSNorm",
    layernorm_epsilon=1e-6,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
    experimental_attention_variant="dsv4_hybrid",
    apply_rope_fusion=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    init_method_std=0.02,
    num_layers=1,
    csa_compress_ratios=[128],
    context_parallel_size=CP,
    cp_partition_mode="contiguous",
    cp_comm_type="all_gather",
    dsa_kernel_backend="cudnn",
    dsa_indexer_loss_coeff=0.01,
    dsa_indexer_use_sparse_loss=True,
    dsa_indexer_k_norm_fp32=True,
)
# The stack (H-layer) geometry at a small hidden size: FlashMLA d_qk = d_v = 512, 64 heads.
_H_GEOMETRY = dict(
    hidden_size=1024,
    num_attention_heads=64,
    q_lora_rank=512,
    v_head_dim=512,
    qk_pos_emb_head_dim=64,
    output_projection_groups=8,
    output_projection_lora_rank=512,
    csa_window_size=128,
    csa_compress_rotary_base=160000.0,
    ernie_vha_rank=16,
)
# The D-layer geometry: FlashMLA d_qk = 576, d_v = 512; indexer 64 x 128, top-2048.
_D_GEOMETRY = dict(
    ernie_d_num_attention_heads=64,
    ernie_d_q_lora_rank=1024,
    ernie_d_kv_lora_rank=512,
    ernie_d_qk_head_dim=448,
    ernie_d_qk_pos_emb_head_dim=64,
    ernie_d_v_head_dim=256,
    ernie_d_rotary_base=10000.0,
    dsa_indexer_n_heads=64,
    dsa_indexer_head_dim=128,
    dsa_indexer_topk=2048,
)

# name -> (document lengths, padded document lengths or None); padded lengths are multiples of 2 * CP.
_D_LAYOUTS = {
    # The production layout (one document over the whole pack), scaled to 32K at CP = 4: 8192 rows per rank.
    "single-document-32k": ([32768], None),
    # Documents of multiples of 2 * CP rows; boundary 3072 lies inside rank 1's chunk and the 3072..6144
    # document spans ranks 1 and 2.
    "multi-document-8k": ([2048, 1024, 3072, 2048], None),
    # The same pack with padding rows at the documents' ends: cu_seqlens_q (unpadded) differs from
    # cu_seqlens_q_padded, whose boundaries the rows, the zigzag partition and the scorer follow.
    "padded-multi-document-8k": ([2041, 1019, 3069, 2047], [2048, 1024, 3072, 2048]),
}
# Rank boundaries 2048 and 6144 cut compression blocks (document offsets 1048 and 48, not multiples of 128);
# 4096 is a document start.
_H_LAYOUT = [1000, 3096, 2000, 2096]

_FORWARD_MIN_COSINE = 0.9999
_FORWARD_MAX_REL_ERR = 1e-2
_GRAD_MIN_COSINE = 0.9999
_GRAD_MAX_REL_ERR = 1e-2
# A gradient whose CP = 1 run-to-run noise exceeds the bar above is gated at this multiple of the noise.
_GRAD_NOISE_MULTIPLE = 3.0
_KL_REL_TOL = 1e-4
_INDEXER_GRAD_NORM_RATIO = (0.9, 1.1)
_TOPK_TIE_REL_TOL = 1e-3
_TOPK_MAX_MISMATCHED_ROW_FRACTION = 1e-2
# Attention-logit scale of the D comparisons: random-init logits make the KL teacher nearly uniform.
_ATTENTION_SHARPENING = 8.0
_MIN_CUDNN_FRONTEND = (1, 26)


# ---------------------------------------------------------------------------------------------
# Environment probes and fixtures
# ---------------------------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _skip(reason: str) -> None:
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = "".join(itertools.takewhile(str.isdigit, piece))
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _cudnn_frontend_version(cudnn) -> tuple[int, ...] | None:
    """The installed cudnn-frontend version, or ``None`` when it cannot be determined."""
    version = getattr(cudnn, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as distribution_version

            version = distribution_version("nvidia-cudnn-frontend")
        except Exception:
            return None
    return _version_tuple(version) or None


@functools.lru_cache(maxsize=None)
def _missing_dependencies() -> tuple[str, ...]:
    missing = []
    if not torch.cuda.is_available():
        return ("CUDA devices",)
    if torch.cuda.device_count() < CP or int(os.environ.get("WORLD_SIZE", "1")) != CP:
        missing.append(f"{CP} GPUs and a {CP}-rank launch (torch.distributed.run --nproc-per-node {CP})")
    for module_name, label in (
        ("transformer_engine.pytorch", "Transformer Engine"),
        ("fast_hadamard_transform", "fast_hadamard_transform"),
    ):
        try:
            importlib.import_module(module_name)
        except Exception as error:
            missing.append(f"{label} ({type(error).__name__}: {error})")
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        missing.append(f"an SM90+ GPU (found sm_{major}{minor})")
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except Exception as error:
        missing.append(f"FlashMLA flash_mla_sparse_fwd ({type(error).__name__}: {error})")
    try:
        import cudnn
        from cudnn import DSA
    except Exception as error:
        missing.append(f"cudnn-frontend DSA ({type(error).__name__}: {error})")
    else:
        version = _cudnn_frontend_version(cudnn)
        if version is not None and version < _MIN_CUDNN_FRONTEND:
            found = ".".join(map(str, version))
            missing.append(f"nvidia-cudnn-frontend >= {'.'.join(map(str, _MIN_CUDNN_FRONTEND))} (found {found})")
        # Needs Megatron-Core dev (megatron.core.utils.filter_kwargs_for_callable): ERNIE 5.5 is developed
        # against Megatron-Core dev, and CI pins Megatron-Core main, so this test is skipped there for now.
        try:
            from megatron.core.utils import filter_kwargs_for_callable
        except ImportError as error:
            missing.append(f"Megatron-Core's filter_kwargs_for_callable (dev) ({type(error).__name__}: {error})")
        else:
            if not filter_kwargs_for_callable(DSA.indexer_forward_wrapper, {"q_causal_offsets": None}):
                missing.append(
                    "cuDNN indexer_forward_wrapper(q_causal_offsets=...) for the zigzag PACKED_THD_GENERAL plan"
                )
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridAttention,
    )

    from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel

    # Needs Megatron-Core dev (csa_utils.thd_layout_kernels): ERNIE 5.5 is developed against Megatron-Core
    # dev, and CI pins Megatron-Core main, so this test is skipped there for now.
    try:
        from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_layout_kernels
    except ImportError as error:
        missing.append(f"Megatron-Core's csa_utils.thd_layout_kernels (dev) ({type(error).__name__}: {error})")
    else:
        if not getattr(thd_layout_kernels, "_CUTE_AVAILABLE", False):
            missing.append("CuTe DSL (nvidia-cutlass-dsl) for DSv4's THD CP kernels")
    if not callable(getattr(DSv4HybridAttention, "_post_o_group_proj", None)):
        missing.append("Megatron-Core's DSv4 _post_o_group_proj hook (dev f78809f6d or later)")
    if context_parallel.load_cp_layout_api() is None:
        missing.append("Megatron-Core's CP partition converter (dev)")
    return tuple(missing)


@dataclasses.dataclass
class _Groups:
    rank: int
    cp: object
    cp1: object


@pytest.fixture(scope="module")
def cp4(tmp_path_factory):
    """NCCL process groups over four ranks: CP = 4, plus a single-rank CP group per rank for CP = 1."""
    missing = _missing_dependencies()
    if missing:
        _skip("the ERNIE 5.5 CP GPU tests need " + "; ".join(missing))
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

    created_group = False
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        created_group = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=CP
    )
    rank = dist.get_rank()
    singletons = [dist.new_group([index]) for index in range(CP)]
    tp = parallel_state.get_tensor_model_parallel_group()
    yield _Groups(
        rank=rank,
        cp=ProcessGroupCollection(tp=tp, cp=parallel_state.get_context_parallel_group()),
        cp1=ProcessGroupCollection(tp=tp, cp=singletons[rank]),
    )
    parallel_state.destroy_model_parallel()
    if created_group:
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _fresh_dsa_state():
    from megatron.core.transformer.experimental_attention_variant import dsa

    def reset() -> None:
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()
        dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None

    reset()
    yield
    reset()


# ---------------------------------------------------------------------------------------------
# Builders, inputs, runs
# ---------------------------------------------------------------------------------------------


def _config(**overrides):
    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig

    config = Ernie55TransformerConfig(**{**_COMMON_CONFIG, **_H_GEOMETRY, **_D_GEOMETRY, **overrides})
    config.finalize()
    return config


def _broadcast_parameters_(module, groups: _Groups) -> None:
    """Give every rank rank 0's weights (the CP group spans every rank)."""
    import torch.distributed as dist

    source = dist.get_global_rank(groups.cp.cp, 0)
    with torch.no_grad():
        for param in module.parameters():
            dist.broadcast(param.data, src=source, group=groups.cp.cp)


def _twins(slot: str, groups: _Groups, *, seed: int, **build_kwargs):
    """The same attention (``dsa_layer`` = D, ``csa_qk_layernorm_layer`` = H) on the CP = 4 and the CP = 1 groups."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module

    from megatron.bridge.models.ernie.modeling_ernie55 import ernie55_hybrid_stack_spec

    spec = getattr(ernie55_hybrid_stack_spec().submodules, slot).submodules.self_attention
    config = _config()
    modules = []
    for pg_collection in (groups.cp, groups.cp1):
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        module = build_module(spec, config=config, layer_number=1, pg_collection=pg_collection, **build_kwargs)
        modules.append(module.to(_device()).train())
    cp_module, cp1_module = modules
    core = cp_module.core_attention
    generator = torch.Generator(device=_device()).manual_seed(seed + 1)
    with torch.no_grad():
        # Move the sink and the VHA factors away from their init so the comparison exercises them.
        for param, std in ((core.attn_sink, 1.0), (core.vha_postmix_u, 0.3), (core.vha_postmix_v, 0.3)):
            param.copy_(std * torch.randn(param.shape, generator=generator, device=param.device))
        if slot == "dsa_layer":
            cp_module.linear_q_up_proj.weight.mul_(_ATTENTION_SHARPENING)
    _broadcast_parameters_(cp_module, groups)
    cp1_module.load_state_dict(cp_module.state_dict())
    return cp_module, cp1_module


def _packed_seq_params(doc_lens, *, padded_lens=None, prebuild_group=None):
    """Packed metadata: one tensor in all four fields, or unpadded and padded boundaries for ``padded_lens``."""
    from megatron.core.packed_seq_params import PackedSeqParams

    from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel

    def cumsum(lengths):
        return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=_device())

    padded = doc_lens if padded_lens is None else padded_lens
    cu_seqlens_padded = cumsum(padded)
    cu_seqlens = cu_seqlens_padded if padded_lens is None else cumsum(doc_lens)
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max(padded),
        max_seqlen_kv=max(padded),
        cp_partition_mode="contiguous",
    )
    if prebuild_group is not None:
        context_parallel.require_cp_layout_api("tests").prebuild_thd_cp_partition_routes(params, prebuild_group)
    return params


def _inputs(hidden_size: int, rows: int, *, seed: int):
    generator = torch.Generator(device=_device()).manual_seed(seed)
    hidden = torch.randn(rows, 1, hidden_size, device=_device(), generator=generator).to(torch.bfloat16)
    grad_output = torch.randn(rows, 1, hidden_size, device=_device(), generator=generator).to(torch.bfloat16)
    return hidden, grad_output


@dataclasses.dataclass
class _Run:
    output: torch.Tensor
    grads: dict[str, torch.Tensor]
    loss: torch.Tensor | None = None
    topk: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    indexer: tuple | None = None
    plan: str | None = None
    topk_calls: list[tuple[bool, torch.Tensor]] = dataclasses.field(default_factory=list)
    """``(with indexer loss, top-k)`` of every top-k selection, in call order."""
    passes: list[torch.Tensor] = dataclasses.field(default_factory=list)
    """Output of every forward pass (the no-grad forward and the replay under recompute)."""


def _run(
    module, hidden, grad_output, params, *, scale: float = 1.0, capture_d: bool = False, recompute: bool = False
) -> _Run:
    """Forward and backward; for a D layer also the top-k (true bounds), the rows' global positions and the KL.

    ``recompute`` runs the module through Megatron-Core's full recompute (``tensor_parallel.checkpoint``):
    a forward without autograd, replayed with autograd in the backward. The captured top-k, KL and
    indexer outputs are the last selection's (the replay's under recompute).
    """
    from megatron.core import tensor_parallel
    from megatron.core.transformer.experimental_attention_variant import dsa

    from megatron.bridge.models.ernie.modeling_ernie55 import ops

    captured = {"topk_calls": []}
    passes = []
    core = module.core_attention
    if capture_d:
        select_topk, forward_before_topk = core._select_topk, core.indexer.forward_before_topk

        def record_select_topk(*args, **kwargs):
            topk, loss = select_topk(*args, **kwargs)
            geometry = kwargs.get("geometry")
            captured["topk_calls"].append((loss is not None, topk.detach()))
            captured["topk"], captured["loss"] = topk, loss
            captured["positions"] = None if geometry is None else geometry.query_positions
            return topk, loss

        def record_indexer(*args, **kwargs):
            outputs = forward_before_topk(*args, **kwargs)
            captured["indexer"] = tuple(tensor.detach() for tensor in outputs)
            return outputs

        core._select_topk, core.indexer.forward_before_topk = record_select_topk, record_indexer

    def forward(hidden_states):
        output, _ = module(hidden_states, None, packed_seq_params=params)
        passes.append(output.detach())
        return output

    dsa.DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(scale, device=_device()))
    module.zero_grad(set_to_none=True)
    try:
        hidden = hidden.detach().clone().requires_grad_(True)
        output = tensor_parallel.checkpoint(forward, False, hidden) if recompute else forward(hidden)
        output.backward(grad_output * scale)
    finally:
        if capture_d:
            del core._select_topk, core.indexer.forward_before_topk
    grads = {"input": hidden.grad.detach().float() / scale}
    grads.update({name: p.grad.detach().float() for name, p in module.named_parameters() if p.grad is not None})
    run = _Run(
        output=output.detach().float(),
        grads=grads,
        plan=getattr(core, "requested_indexer_scoring_plan", None),
        topk_calls=captured["topk_calls"],
        passes=passes,
    )
    if capture_d:
        rows = captured["topk"].size(1)
        positions = captured["positions"]
        doc_start = ops.ernie55_document_starts(params, rows, _device(), positions=positions)
        _, ends = ops.ernie55_selection_bounds(doc_start, core.window_size, positions=positions)
        run.topk = ops.mask_topk_beyond_end(captured["topk"], ends=ends)
        run.positions = positions
        run.loss = None if captured["loss"] is None else captured["loss"].detach().float()
        run.indexer = captured["indexer"]
    return run


def _rank_rows(total: int, rank: int) -> torch.Tensor:
    local = total // CP
    return torch.arange(rank * local, (rank + 1) * local, device=_device())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (a @ b / denominator).item()


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double(), expected.detach().double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-300)).item()


def _all_reduce(tensor: torch.Tensor, group, average: bool = False) -> torch.Tensor:
    import torch.distributed as dist

    tensor = tensor.detach().clone()
    dist.all_reduce(tensor, group=group)
    return tensor / dist.get_world_size(group) if average else tensor


def _report(label: str, **metrics) -> None:
    import torch.distributed as dist

    text = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    logger.info("[ernie55-cp-gpu rank %d] %s: %s", dist.get_rank(), label, text)


def _assert_all_ranks(failures: list[str], label: str) -> None:
    """Fail on every rank if any rank failed, so no rank waits in a later collective."""
    import torch.distributed as dist

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, failures)
    problems = [f"rank {rank}: {failure}" for rank, items in enumerate(gathered) for failure in items]
    assert not problems, f"{label}:\n" + "\n".join(problems[:40])


def _cp1_noise(cp1_run: _Run, cp1_rerun: _Run) -> dict[str, tuple[float, float]]:
    """Run-to-run noise of every CP = 1 gradient: ``(1 - cosine, relative L2)`` between two identical runs."""
    return {
        name: (max(0.0, 1.0 - _cosine(cp1_rerun.grads[name], grad)), _rel_err(cp1_rerun.grads[name], grad))
        for name, grad in cp1_run.grads.items()
    }


def _compare(
    cp_run: _Run,
    cp1_run: _Run,
    rows: torch.Tensor,
    groups: _Groups,
    label: str,
    noise: dict[str, tuple[float, float]],
) -> list[str]:
    """Output rows and DDP-averaged gradients of CP = 4 against CP = 1.

    Every gradient must reach the bar (cosine >= 0.9999, relative L2 <= 1e-2), widened to
    3 times the gradient's CP = 1 run-to-run ``noise`` only where that noise exceeds the bar. The
    margins and the noise are logged per gradient.
    """
    failures = []
    expected = cp1_run.output[rows]
    cosine, rel_err = _cosine(cp_run.output, expected), _rel_err(cp_run.output, expected)
    if not torch.isfinite(cp_run.output).all():
        failures.append("non-finite CP = 4 output")
    if cosine < _FORWARD_MIN_COSINE or rel_err > _FORWARD_MAX_REL_ERR:
        failures.append(f"output cosine {cosine:.7f}, relative L2 {rel_err:.3e}")
    cosines, rel_errs, margins = {}, {}, []
    for name, grad in cp_run.grads.items():
        if name == "input":
            actual, reference = grad, cp1_run.grads["input"][rows]
        else:
            actual, reference = _all_reduce(grad, groups.cp.cp, average=True), cp1_run.grads[name]
        cosines[name], rel_errs[name] = _cosine(actual, reference), _rel_err(actual, reference)
        cosine_gap_noise, rel_noise = noise[name]
        min_cosine = min(_GRAD_MIN_COSINE, 1.0 - _GRAD_NOISE_MULTIPLE * cosine_gap_noise)
        max_rel_err = max(_GRAD_MAX_REL_ERR, _GRAD_NOISE_MULTIPLE * rel_noise)
        margins.append(f"{name} cos={cosines[name]:.7f} rel={rel_errs[name]:.3e} cp1_noise_rel={rel_noise:.3e}")
        if not torch.isfinite(actual).all():
            failures.append(f"{name}: non-finite gradient")
        elif reference.abs().max() > 0 and (cosines[name] < min_cosine or rel_errs[name] > max_rel_err):
            failures.append(
                f"{name}: gradient cosine {cosines[name]:.7f} (bar {min_cosine:.7f}), relative L2 "
                f"{rel_errs[name]:.3e} (bar {max_rel_err:.3e}; CP = 1 run-to-run noise {rel_noise:.3e})"
            )
    _report(
        label,
        output_cosine=cosine,
        output_rel_err=rel_err,
        min_grad_cosine=min(cosines.values()),
        max_grad_rel_err=max(rel_errs.values()),
        max_cp1_noise_rel=max(rel for _, rel in noise.values()),
    )
    _report(f"{label} gradient margins", margins="; ".join(margins))
    return failures


# ---------------------------------------------------------------------------------------------
# D layer
# ---------------------------------------------------------------------------------------------


def _spy_scoring_executors(monkeypatch) -> dict[str, int]:
    """Count the cuDNN backend's packed scoring executors (zigzag general, Megatron-Core #8534's segment)."""
    from megatron.core.transformer.experimental_attention_variant import dsa_cudnn_kernels

    counts = {"packed_thd_general": 0, "packed_single_segments": 0}
    for name, key in (
        ("_indexer_topk_packed_thd", "packed_thd_general"),
        ("_indexer_topk_single_packed_cp_segments", "packed_single_segments"),
    ):
        original = getattr(dsa_cudnn_kernels, name)

        def counting(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(dsa_cudnn_kernels, name, counting)
    return counts


def _spy_topk_hooks(monkeypatch) -> list[tuple[str, dict]]:
    """Record every call of Megatron-Core's top-k hooks (``run_fused_qk_topk[_with_loss]``) and its bound arguments."""
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    calls = []
    for name in ("run_fused_qk_topk", "run_fused_qk_topk_with_loss"):
        original = getattr(dsa_kernels, name)

        def recording(*args, _original=original, _name=name, _signature=inspect.signature(original), **kwargs):
            bound = _signature.bind(*args, **kwargs)
            bound.apply_defaults()
            calls.append((_name, dict(bound.arguments)))
            return _original(*args, **kwargs)

        monkeypatch.setattr(dsa_kernels, name, recording)
    return calls


def _packed_hook_problem(arguments: dict, params) -> str | None:
    """Why a top-k hook call does not request PACKED_THD_GENERAL on the pack's padded boundaries, or ``None``."""
    hook_params = arguments.get("packed_seq_params")
    if not arguments.get("use_local_indexer_varlen") or hook_params is None:
        return "no packed layout (UNFUSED_BOUNDS)"
    if arguments.get("single_packed_thd_sequence"):
        return "single_packed_thd_sequence=True (#8534)"
    cu_seqlens = hook_params.cu_seqlens_q
    fields = ("cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")
    if any(getattr(hook_params, field) is not cu_seqlens for field in fields):
        return "the four cu_seqlens fields are not one tensor"
    if not torch.equal(cu_seqlens, params.cu_seqlens_q_padded):
        return f"cu_seqlens {cu_seqlens.tolist()} are not the padded boundaries {params.cu_seqlens_q_padded.tolist()}"
    if (hook_params.max_seqlen_q, hook_params.max_seqlen_kv) != (params.max_seqlen_q, params.max_seqlen_kv):
        return f"max_seqlen {hook_params.max_seqlen_q}/{hook_params.max_seqlen_kv}, expected {params.max_seqlen_q}"
    return None


def _hook_failures(hooks: list[tuple[str, dict]], expected_names: list[str], params) -> list[str]:
    """CP = 4 top-k hook calls that are not ``expected_names`` or do not request PACKED_THD_GENERAL."""
    names = [name for name, _ in hooks]
    failures = [] if names == expected_names else [f"CP = 4 top-k hooks {names}, expected {expected_names}"]
    for name, arguments in hooks:
        problem = _packed_hook_problem(arguments, params)
        if problem is not None:
            failures.append(f"CP = 4 {name}: {problem}")
    return failures


def _same_selection(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Per row, whether two ``[b, rows, K]`` top-k lists (``-1`` padded, any order) hold the same keys."""
    return (first.long().sort(dim=-1).values == second.long().sort(dim=-1).values).all(dim=-1)


def _topk_failures(cp_run: _Run, cp1_run: _Run, cp1_bounds, index_topk: int) -> tuple[list[str], int]:
    """Rows whose CP = 4 top-k differs from CP = 1 by keys that are not near-ties of the fp32 CP = 1 scores.

    ``cp1_bounds`` are the true candidate bounds ``(starts, ends)`` of every row of the CP = 1 pack.
    """
    from megatron.core.transformer.experimental_attention_variant import dsa

    positions = cp_run.positions
    got, expected = cp_run.topk[0], cp1_run.topk[0, positions]
    same = _same_selection(got, expected)
    mismatched = (~same).nonzero().flatten()
    failures = []
    if mismatched.numel():
        q, k, w = cp1_run.indexer
        rows = positions[mismatched]
        starts, ends = cp1_bounds
        # The core's score convention: the weights times d^0.5 in their dtype, the scores times d^-0.5.
        head_dim = q.size(-1)
        with torch.no_grad():
            scores, _ = dsa.fused_qk_topk_naive(
                q[rows], k, w[rows] * head_dim**0.5, index_topk, varlen_starts=starts[rows], varlen_ends=ends[rows]
            )
            scores = scores * head_dim**-0.5
        for index, local in enumerate(mismatched.tolist()):
            got_keys = set(got[local][got[local] >= 0].tolist())
            expected_keys = set(expected[local][expected[local] >= 0].tolist())
            if len(got_keys) != len(expected_keys):
                failures.append(f"row {positions[local].item()}: {len(got_keys)} keys, expected {len(expected_keys)}")
                continue
            row_scores = scores[0, index].double()
            finite = row_scores[torch.isfinite(row_scores)]
            boundary = row_scores[sorted(expected_keys)].min().item()
            tolerance = _TOPK_TIE_REL_TOL * finite.abs().max().item()
            far = [key for key in got_keys ^ expected_keys if abs(row_scores[key].item() - boundary) > tolerance]
            if far:
                failures.append(f"row {positions[local].item()}: non-tied keys {sorted(far)[:8]}")
    if mismatched.numel() > max(1, int(_TOPK_MAX_MISMATCHED_ROW_FRACTION * same.numel())):
        failures.append(f"{mismatched.numel()} of {same.numel()} rows select different (tied) keys")
    return failures, int(mismatched.numel())


@pytest.mark.parametrize("layout", list(_D_LAYOUTS), ids=list(_D_LAYOUTS))
def test_d_layer_cp4_matches_cp1(cp4, monkeypatch, layout):
    from megatron.bridge.models.ernie.modeling_ernie55 import (
        INDEXER_SCORING_PACKED_THD_GENERAL,
        INDEXER_SCORING_REFERENCE_ORDER,
        ops,
    )

    doc_lens, padded_lens = _D_LAYOUTS[layout]
    total = sum(padded_lens or doc_lens)
    cp_module, cp1_module = _twins("dsa_layer", cp4, seed=1234)
    config = cp_module.config
    hidden, grad_output = _inputs(config.hidden_size, total, seed=7)
    rows = _rank_rows(total, cp4.rank)
    counts = _spy_scoring_executors(monkeypatch)
    hooks = _spy_topk_hooks(monkeypatch)

    cp1_params = _packed_seq_params(doc_lens, padded_lens=padded_lens)
    cp1_run = _run(cp1_module, hidden, grad_output, cp1_params, capture_d=True)
    noise = _cp1_noise(cp1_run, _run(cp1_module, hidden, grad_output, cp1_params))
    cp1_counts, cp1_hooks = dict(counts), len(hooks)
    doc_start = ops.ernie55_document_starts(cp1_params, total, _device())
    cp1_bounds = ops.ernie55_selection_bounds(doc_start, cp1_module.core_attention.window_size)

    cp_params = _packed_seq_params(doc_lens, padded_lens=padded_lens, prebuild_group=cp4.cp.cp)
    cp_run = _run(cp_module, hidden[rows], grad_output[rows], cp_params, scale=float(CP), capture_d=True)
    cp_counts = {key: counts[key] - cp1_counts[key] for key in counts}

    failures = _compare(cp_run, cp1_run, rows, cp4, f"D {layout}", noise)
    if cp_params.cp_partition_mode != "contiguous":
        failures.append(f"cp_partition_mode left at {cp_params.cp_partition_mode!r}")
    cp1_loss_plan = cp1_module.core_attention.requested_indexer_loss_scoring_plan
    if (
        cp1_run.plan != INDEXER_SCORING_REFERENCE_ORDER
        or cp1_loss_plan != INDEXER_SCORING_PACKED_THD_GENERAL
        or not cp1_counts["packed_thd_general"]
        or cp1_counts["packed_single_segments"]
    ):
        failures.append(f"CP = 1 scoring: plan {cp1_run.plan}, loss plan {cp1_loss_plan}, executors {cp1_counts}")
    if cp_run.plan != INDEXER_SCORING_PACKED_THD_GENERAL or not cp_counts["packed_thd_general"]:
        failures.append(f"CP = 4 scoring: plan {cp_run.plan}, executors {cp_counts}")
    if cp_counts["packed_single_segments"]:
        failures.append("CP = 4 ran the single-sequence segment scorer (#8534)")
    failures += _hook_failures(hooks[cp1_hooks:], ["run_fused_qk_topk_with_loss"], cp_params)

    topk_failures, mismatched = _topk_failures(cp_run, cp1_run, cp1_bounds, config.dsa_indexer_topk)
    failures += topk_failures
    kl = _all_reduce(cp_run.loss, cp4.cp.cp).item()
    kl_reference = cp1_run.loss.item()
    if abs(kl - kl_reference) > _KL_REL_TOL * abs(kl_reference):
        failures.append(f"indexer KL summed over CP {kl:.8g} vs CP = 1 {kl_reference:.8g}")
    norms = {}
    for name in cp_run.grads:
        if ".indexer." in name:
            averaged = _all_reduce(cp_run.grads[name], cp4.cp.cp, average=True)
            norms[name] = (averaged.norm() / cp1_run.grads[name].norm().clamp_min(1e-30)).item()
    low, high = _INDEXER_GRAD_NORM_RATIO
    failures += [
        f"{name}: indexer gradient norm ratio {ratio:.4f}" for name, ratio in norms.items() if not low <= ratio <= high
    ]
    _report(
        f"D {layout}",
        kl_cp4=kl,
        kl_cp1=kl_reference,
        topk_mismatched_rows=mismatched,
        indexer_grad_norm_ratio_min=min(norms.values()),
        indexer_grad_norm_ratio_max=max(norms.values()),
        executors_cp4=str(cp_counts),
    )
    _assert_all_ranks(failures, f"D layer CP4 vs CP1 ({layout})")


def test_d_layer_cp4_full_recompute(cp4, monkeypatch):
    """Full recompute at CP = 4: both passes request and run PACKED_THD_GENERAL and select the same keys.

    With Megatron-Core's full recompute every D-layer forward runs without autograd, so without the
    indexer loss (the plain top-k hook), and the backward replays it with the loss hook. If the two
    passes selected different keys, the gradient would belong to another function than the output.
    """
    from megatron.bridge.models.ernie.modeling_ernie55 import INDEXER_SCORING_PACKED_THD_GENERAL

    doc_lens, padded_lens = _D_LAYOUTS["padded-multi-document-8k"]
    total = sum(padded_lens)
    cp_module, cp1_module = _twins("dsa_layer", cp4, seed=2024)
    hidden, grad_output = _inputs(cp_module.config.hidden_size, total, seed=17)
    rows = _rank_rows(total, cp4.rank)
    cp1_params = _packed_seq_params(doc_lens, padded_lens=padded_lens)
    cp1_run = _run(cp1_module, hidden, grad_output, cp1_params, capture_d=True)
    noise = _cp1_noise(cp1_run, _run(cp1_module, hidden, grad_output, cp1_params))

    counts = _spy_scoring_executors(monkeypatch)
    hooks = _spy_topk_hooks(monkeypatch)
    cp_params = _packed_seq_params(doc_lens, padded_lens=padded_lens, prebuild_group=cp4.cp.cp)
    cp_run = _run(
        cp_module, hidden[rows], grad_output[rows], cp_params, scale=float(CP), capture_d=True, recompute=True
    )

    failures = _compare(cp_run, cp1_run, rows, cp4, "D full recompute (padded 8K)", noise)
    # The plain hook in the no-grad forward, then the loss hook in the replay.
    failures += _hook_failures(hooks, ["run_fused_qk_topk", "run_fused_qk_topk_with_loss"], cp_params)
    if counts != {"packed_thd_general": 2, "packed_single_segments": 0}:
        failures.append(f"scoring executors {counts}, expected PACKED_THD_GENERAL once per pass")
    if cp_run.plan != INDEXER_SCORING_PACKED_THD_GENERAL:
        failures.append(f"requested scoring plan {cp_run.plan}")
    if cp_params.cp_partition_mode != "contiguous":
        failures.append(f"cp_partition_mode left at {cp_params.cp_partition_mode!r}")
    mismatched_rows = -1
    if [with_loss for with_loss, _ in cp_run.topk_calls] != [False, True] or len(cp_run.passes) != 2:
        failures.append(
            f"selections {[with_loss for with_loss, _ in cp_run.topk_calls]} and {len(cp_run.passes)} passes, "
            "expected a no-loss forward and a replay with the loss"
        )
    else:
        (_, forward_topk), (_, replay_topk) = cp_run.topk_calls
        mismatched_rows = int((~_same_selection(forward_topk, replay_topk)).sum())
        if mismatched_rows:
            failures.append(f"{mismatched_rows} rows select different keys in the no-grad forward and the replay")
        forward_output, replay_output = (output.float() for output in cp_run.passes)
        replay_cosine, replay_rel_err = _cosine(replay_output, forward_output), _rel_err(replay_output, forward_output)
        if replay_cosine < _FORWARD_MIN_COSINE or replay_rel_err > _FORWARD_MAX_REL_ERR:
            failures.append(f"replay output cosine {replay_cosine:.7f}, relative L2 {replay_rel_err:.3e}")
        _report(
            "D full recompute (padded 8K)",
            replay_vs_forward_cosine=replay_cosine,
            replay_vs_forward_rel_err=replay_rel_err,
            replay_vs_forward_topk_mismatched_rows=mismatched_rows,
            executors=str(counts),
        )
    kl = _all_reduce(cp_run.loss, cp4.cp.cp).item()
    if abs(kl - cp1_run.loss.item()) > _KL_REL_TOL * abs(cp1_run.loss.item()):
        failures.append(f"indexer KL summed over CP {kl:.8g} vs CP = 1 {cp1_run.loss.item():.8g}")
    _assert_all_ranks(failures, "D layer CP4 full recompute")


def test_d_layer_cp4_packed_scoring_speedup(cp4, monkeypatch):
    """The zigzag PACKED_THD_GENERAL plan beats UNFUSED_BOUNDS at CP = 4 on a 32K pack (forward + backward)."""
    from megatron.bridge.models.ernie.modeling_ernie55 import INDEXER_SCORING_UNFUSED_BOUNDS, latent_mqa_core

    doc_lens, _ = _D_LAYOUTS["single-document-32k"]
    total = sum(doc_lens)
    cp_module, _ = _twins("dsa_layer", cp4, seed=99)
    hidden, grad_output = _inputs(cp_module.config.hidden_size, total, seed=9)
    rows = _rank_rows(total, cp4.rank)
    params = _packed_seq_params(doc_lens, prebuild_group=cp4.cp.cp)

    def timed(iterations: int = 5, warmup: int = 2) -> float:
        for _ in range(warmup):
            _run(cp_module, hidden[rows], grad_output[rows], params, scale=float(CP))
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        for _ in range(iterations):
            _run(cp_module, hidden[rows], grad_output[rows], params, scale=float(CP))
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations

    counts = _spy_scoring_executors(monkeypatch)
    packed_ms = timed()
    packed_calls = counts["packed_thd_general"]
    # Without the packed layout the backend has no fused plan for local queries against gathered keys.
    monkeypatch.setattr(
        latent_mqa_core.Ernie55LatentMQACoreAttention, "_packed_cp_hook_kwargs", staticmethod(lambda *a, **k: {})
    )
    unfused_ms = timed()
    unfused_plan = cp_module.core_attention.requested_indexer_scoring_plan
    speedup = unfused_ms / packed_ms
    _report("D CP4 32K fwd+bwd", packed_thd_general_ms=packed_ms, unfused_bounds_ms=unfused_ms, speedup=speedup)
    failures = []
    if not packed_calls or counts["packed_thd_general"] != packed_calls:
        failures.append(f"executor calls: {packed_calls} with the packed layout, {counts} in total")
    if unfused_plan != INDEXER_SCORING_UNFUSED_BOUNDS:
        failures.append(f"the forced run scored with {unfused_plan}")
    if speedup <= 1.0:
        failures.append(f"PACKED_THD_GENERAL {packed_ms:.1f} ms is not faster than UNFUSED_BOUNDS {unfused_ms:.1f} ms")
    _assert_all_ranks(failures, "D layer CP4 scoring speedup")


# ---------------------------------------------------------------------------------------------
# H layer
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rope", ["eager", "fused"])
def test_h_layer_cp4_matches_cp1(cp4, monkeypatch, rope):
    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HCASelfAttention, install_ernie55_fused_ops
    from megatron.bridge.models.ernie.modeling_ernie55.fused_dispatch import triton_available
    from megatron.bridge.models.ernie.modeling_ernie55.hca_attention import rotary_embedding_references
    from tests.unit_tests.models.ernie.ernie55_test_utils import isolate_fused_dispatch

    # Megatron-Core's own RoPE function, whatever an earlier test in this process installed.
    dispatch = isolate_fused_dispatch(monkeypatch)
    if rope == "fused" and not triton_available():
        _skip("the ERNIE 5.5 Triton RoPE needs Triton")
    total = sum(_H_LAYOUT)
    cp_module, cp1_module = _twins("csa_qk_layernorm_layer", cp4, seed=4321, compress_ratio=128)
    assert isinstance(cp_module, Ernie55HCASelfAttention)
    failures = []
    references = rotary_embedding_references(cp_module)
    if {id(embedding) for _, embedding in references} != {id(cp_module.rotary_pos_emb)}:
        failures.append(f"rotary holders {[path for path, _ in references]} are not all the plain RoPE")
    if cp_module.rotary_pos_emb.cp_group is not cp4.cp.cp:
        failures.append("the plain RoPE is not on the CP group")
    hidden, grad_output = _inputs(cp_module.config.hidden_size, total, seed=11)
    rows = _rank_rows(total, cp4.rank)
    cp1_params = _packed_seq_params(_H_LAYOUT)
    cp1_run = _run(cp1_module, hidden, grad_output, cp1_params)
    noise = _cp1_noise(cp1_run, _run(cp1_module, hidden, grad_output, cp1_params))
    if dispatch.FUSED_OP_CALLS:
        failures.append(f"CP = 1 launched ERNIE 5.5 Triton kernels: {dict(dispatch.FUSED_OP_CALLS)}")
    if rope == "fused":
        install_ernie55_fused_ops(cp_module.config)
    cp_params = _packed_seq_params(_H_LAYOUT, prebuild_group=cp4.cp.cp)
    cp_run = _run(cp_module, hidden[rows], grad_output[rows], cp_params, scale=float(CP))
    failures += _compare(cp_run, cp1_run, rows, cp4, f"H 8K ({rope} RoPE)", noise)
    launches = dict(dispatch.FUSED_OP_CALLS)
    if rope == "fused":
        # Megatron-Core's DSv4 attention rotates q and kv forward and the core-attention output inversely.
        forward, inverse = launches.get("thd_cp_local_rope@forward", 0), launches.get("thd_cp_local_rope@inverse", 0)
        if forward < 2 or inverse < 1:
            failures.append(f"Triton RoPE launches {launches}, expected >= 2 forward and >= 1 inverse")
    elif launches:
        failures.append(f"Triton kernels launched without ernie_fused_ops installed: {launches}")
    _report(f"H 8K ({rope} RoPE)", triton_launches=str(launches))
    _assert_all_ranks(failures, f"H layer CP4 vs CP1 ({rope} RoPE)")
