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

"""GPU tests of the ERNIE 5.5 H-layer Transformer Engine CUDA graphs: captured graphs against eager execution.

Needs four GPUs (SM90+), Transformer Engine, Triton, FlashMLA, the cuDNN frontend DSA kernels, CuTe DSL and a
Megatron-Core dev build with the CP partition converter and the DSv4 ``_post_o_group_proj`` hook (dev f78809f6d or
later). Launch on all four ranks (CP = 2 runs as two CP groups of two ranks, CP = 4 as one group)::

    ERNIE55_GPU_STRICT=1 uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -rA \\
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_cuda_graph_gpu.py

Without them the tests skip and name what is missing; ``ERNIE55_GPU_STRICT=1`` turns the skips into failures. Every
rank runs the same tests; failures are gathered so all ranks fail together.

Each test builds a one-H-layer ERNIE 5.5 model (pattern ``"H-"``) at CP = 2 and CP = 4 (contiguous THD, 1024 rows per
rank) with fp32 decoder streams, the fused mHC kernels (``ernie_fused_mhc``), ``layernorm`` recompute, Transformer
Engine graphs of the attention (``cuda_graph_impl="transformer_engine"``, ``cuda_graph_modules=["attn"]``) and the TE
RNG tracker, so its decoder's H wrapper is an ``Ernie55HCAHyperConnectionHybridLayer``. The tests capture that wrapper
with Transformer Engine's ``make_graphed_callables`` on the wrapper's own static inputs, with the arguments
Megatron-Core's ``TECudaGraphHelper`` passes at PP = 1 (one graph, order ``[1, -1]``, buffer reuse), and replay it
through the wrapper's ``__call__``, as training does. The input streams have the scale of an embedding at
initialization (std 0.02), where the attention update is a large part of the layer output, so a graph that replayed
stale inputs could not hide behind the mixed streams.

Gate of a graph replay against an eager run of the same row: the output, the input gradient and every parameter
gradient have cosine >= 0.9999 and relative L2 <= 1e-2, widened to 3 times the eager run-to-run noise where that is
larger (the cuDNN DSA backward and the compressor accumulate with atomics). The output and the input gradient are fp32
in both, and the graph is replayed (not run eagerly) exactly once per graph run.

* ``test_graph_matches_eager`` (G1): one document over the whole row, the full-size row shape.
* ``test_graph_replays_new_rows`` (G2): one graph replays another one-document row; a graph captured with
  ``thd_max_packed_sequences=2`` replays two-document rows with different boundaries (they cut the ranks' chunks). Both
  match eager execution, so the graph reads the rows and their boundaries on the device and bakes none of them in.
* ``test_graph_rejects_rows_it_cannot_replay`` (G3): the one-document graph rejects a two-document row with a
  ``ValueError`` before anything runs, and still replays rows of its shape afterwards.
* ``test_batched_vha_graph_matches_eager`` (G5): with ``ernie_vha_batched_mix`` (the VHA head mix as one batched GEMM
  over the tokens). Every forward of the H core (eager, and the capture's) calls the batched mix once, and none does
  with the flag cleared. Eagerly, the layer output is bitwise the one with ``torch.matmul``'s mix (when two eager runs
  with ``torch.matmul`` agree bitwise, i.e. the forward is deterministic), and every gradient passes the gates above
  against it (the mix gradient sums in another order); the graph captured with it matches eager execution on two rows.
  At CP = 4 one more row runs through the eager batched mix, the eager ``torch.matmul`` mix and the graph inside the
  file's only ``cudaProfilerStart``/``Stop`` range, with NVTX ranges ``vha/G5/cp4/<batched|matmul>/<eager|graph>``: the
  capture range of an nsys run of this file (``--capture-range=cudaProfilerApi``).
* ``test_graph_outlives_the_shared_rope_cache`` (G4): after capture, Megatron-Core's shared RoPE cache is cleared (what
  the eager D and MTP layers' lookups do by eviction in training, and eval CP rebinding at once), the allocator's cache
  is emptied and NaN tensors of the table's size are allocated; the RoPE table the graph reads is still alive and the
  graph still matches eager execution.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import itertools
import logging
import os

import pytest
import torch


logger = logging.getLogger(__name__)

_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"
_WORLD_SIZE = 4

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() and not _STRICT, reason="the ERNIE 5.5 CUDA graph GPU tests need CUDA devices"
    ),
]

_LOCAL_ROWS = 1024
_STREAMS = 4
_VOCAB = 512
_SEED = 1234
_INPUT_STD = 0.02
_WARMUP_ITERS = 2
_MIN_COSINE = 0.9999
_MAX_REL_ERR = 1e-2
_NOISE_MULTIPLE = 3.0
_PROFILED_CP = 4
"""CP size at which G5 runs one row through each VHA mix inside a ``cudaProfilerStart``/``Stop`` range (for nsys)."""

_COMMON_CONFIG = dict(
    num_layers=2,
    csa_compress_ratios=[128, 0],
    normalization="RMSNorm",
    layernorm_epsilon=1e-6,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
    experimental_attention_variant="dsv4_hybrid",
    apply_rope_fusion=False,
    bias_activation_fusion=False,
    gradient_accumulation_fusion=False,
    bf16=True,
    params_dtype=torch.bfloat16,
    init_method_std=0.02,
    cp_partition_mode="contiguous",
    cp_comm_type="all_gather",
    dsa_kernel_backend="cudnn",
    dsa_indexer_loss_coeff=0.01,
    dsa_indexer_use_sparse_loss=True,
    dsa_indexer_k_norm_fp32=True,
    gated_linear_unit=True,
    ffn_hidden_size=2048,
    enable_mhc_connections=True,
    mhc_num_residual_streams=_STREAMS,
    ernie_fused_mhc=True,
    recompute_granularity="selective",
    recompute_modules=["layernorm"],
    cuda_graph_impl="transformer_engine",
    cuda_graph_modules=["attn"],
    use_te_rng_tracker=True,
    max_seqlen_per_dp_cp_rank=_LOCAL_ROWS,
)
# The H-layer geometry of test_ernie55_cp_gpu.py: FlashMLA d_qk = d_v = 512, 64 heads.
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
# The D-layer settings the config validation reads (the model has no D layer).
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


# ---------------------------------------------------------------------------------------------
# Environment probes and fixtures
# ---------------------------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _skip(reason: str) -> None:
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@functools.lru_cache(maxsize=None)
def _missing_dependencies() -> tuple[str, ...]:
    if not torch.cuda.is_available():
        return ("CUDA devices",)
    missing = []
    if torch.cuda.device_count() < _WORLD_SIZE or int(os.environ.get("WORLD_SIZE", "1")) != _WORLD_SIZE:
        missing.append(
            f"{_WORLD_SIZE} GPUs and a {_WORLD_SIZE}-rank launch "
            f"(torch.distributed.run --nproc-per-node {_WORLD_SIZE})"
        )
    expandable = [
        name
        for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
        if "expandable_segments:True" in os.environ.get(name, "")
    ]
    if expandable and os.environ.get("NCCL_GRAPH_REGISTER") != "0":
        missing.append(
            f"NCCL_GRAPH_REGISTER=0 ({expandable[0]} enables expandable segments, as Ernie55TECudaGraphHelper checks)"
        )
    for module_name, label in (("transformer_engine.pytorch", "Transformer Engine"), ("triton", "Triton")):
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
        from cudnn import DSA  # noqa: F401
    except Exception as error:
        missing.append(f"cudnn-frontend DSA ({type(error).__name__}: {error})")
    from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_layout_kernels
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridAttention,
    )

    from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel

    if not getattr(thd_layout_kernels, "_CUTE_AVAILABLE", False):
        missing.append("CuTe DSL (nvidia-cutlass-dsl) for DSv4's THD CP kernels")
    if not callable(getattr(DSv4HybridAttention, "_post_o_group_proj", None)):
        missing.append("Megatron-Core's DSv4 _post_o_group_proj hook (dev f78809f6d or later)")
    if context_parallel.load_cp_layout_api() is None:
        missing.append("Megatron-Core's CP partition converter (dev)")
    return tuple(missing)


@dataclasses.dataclass
class _Setup:
    cp: int
    cp_group: object
    cp_rank: int


@pytest.fixture(scope="module")
def distributed():
    """The NCCL default process group over the four ranks, created once for the module, and a non-default stream."""
    missing = _missing_dependencies()
    if missing:
        _skip("the ERNIE 5.5 CUDA graph GPU tests need " + "; ".join(missing))
    import torch.distributed as dist

    created_group = False
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        created_group = True
    # Bridge's initialization runs Transformer Engine graph training on a non-default current stream. Do the same:
    # a layer run eagerly before its capture keeps AccumulateGrad nodes (and graph pieces such as its attn_norm
    # offload manager's input) alive on the stream they were created on, and on the legacy default stream the
    # capture's backward would make that stream wait on the capturing one (cudaErrorStreamCaptureImplicit).
    previous_stream = torch.cuda.current_stream()
    torch.cuda.set_stream(torch.cuda.Stream())
    yield
    torch.cuda.set_stream(previous_stream)
    if created_group:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[2, 4], ids=["cp2", "cp4"])
def cp_setup(request, distributed):
    """Megatron-Core's model-parallel state at TP = PP = 1 and the parametrized CP size."""
    from megatron.core import parallel_state

    cp = request.param
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=cp
    )
    yield _Setup(
        cp=cp,
        cp_group=parallel_state.get_context_parallel_group(),
        cp_rank=parallel_state.get_context_parallel_rank(),
    )
    parallel_state.destroy_model_parallel()


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
# Builders, rows, runs
# ---------------------------------------------------------------------------------------------


def _build(setup: _Setup, *, documents: int, seed: int = _SEED, **overrides):
    """A one-H-layer ERNIE 5.5 model with H-layer graphs; returns its decoder's H wrapper (weights equal over CP).

    ``overrides`` are further config fields (e.g. ``ernie_vha_batched_mix``).
    """
    import torch.distributed as dist
    import torch.nn.functional as F
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
    from megatron.bridge.models.ernie.modeling_ernie55 import (
        Ernie55HCAHyperConnectionHybridLayer,
        Ernie55HybridModel,
        ernie55_hybrid_stack_spec,
    )

    values = {**_COMMON_CONFIG, **_H_GEOMETRY, **_D_GEOMETRY}
    values.update(context_parallel_size=setup.cp, thd_max_packed_sequences=documents, activation_func=F.silu)
    values.update(overrides)
    config = Ernie55TransformerConfig(**values)
    config.finalize()
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed, te_rng_tracker=True, force_reset_rng=True)
    model = Ernie55HybridModel(
        config=config,
        hybrid_stack_spec=ernie55_hybrid_stack_spec(config),
        vocab_size=_VOCAB,
        max_sequence_length=_LOCAL_ROWS * setup.cp,
        hybrid_layer_pattern="H-",
        position_embedding_type="none",
    )
    model = model.to(_device()).train()
    wrapper = model.decoder.layers[0]
    assert type(wrapper) is Ernie55HCAHyperConnectionHybridLayer, type(wrapper).__name__
    core = wrapper.inner_layer.self_attention.core_attention
    generator = torch.Generator(device=_device()).manual_seed(seed + 1)
    source = dist.get_global_rank(setup.cp_group, 0)
    with torch.no_grad():
        # Move the sink and the VHA factors away from their init so the comparison exercises them.
        for param, std in ((core.attn_sink, 1.0), (core.vha_postmix_u, 0.3), (core.vha_postmix_v, 0.3)):
            param.copy_(std * torch.randn(param.shape, generator=generator, device=param.device))
        for param in wrapper.parameters():
            dist.broadcast(param.data, src=source, group=setup.cp_group)
    return model, wrapper


def _row(setup: _Setup, wrapper, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's contiguous rows of one packed row: fp32 input streams (std 0.02) and the output gradient."""
    total = _LOCAL_ROWS * setup.cp
    width = _STREAMS * wrapper.config.hidden_size
    generator = torch.Generator(device=_device()).manual_seed(seed)
    hidden = _INPUT_STD * torch.randn(total, 1, width, device=_device(), generator=generator)
    grad_output = torch.randn(total, 1, width, device=_device(), generator=generator)
    rows = slice(setup.cp_rank * _LOCAL_ROWS, (setup.cp_rank + 1) * _LOCAL_ROWS)
    return hidden[rows].contiguous(), grad_output[rows].contiguous()


def _documents(setup: _Setup, fractions: tuple[float, ...]) -> list[int]:
    """Document lengths of a full row, multiples of 2 * CP (the ERNIE 5.5 padding), the last taking the rest."""
    total, multiple = _LOCAL_ROWS * setup.cp, 2 * setup.cp
    lengths = [max(multiple, round(total * fraction / multiple) * multiple) for fraction in fractions]
    return [*lengths, total - sum(lengths)]


def _packed_seq_params(setup: _Setup, document_lengths: list[int]):
    """Packed metadata like ``build_ernie55_thd_cp_batch``'s: one int32 tensor in all four fields, prebuilt routes."""
    from megatron.core.packed_seq_params import PackedSeqParams

    from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel

    cu_seqlens = torch.tensor([0, *itertools.accumulate(document_lengths)], dtype=torch.int32, device=_device())
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max(document_lengths),
        max_seqlen_kv=max(document_lengths),
        cp_partition_mode="contiguous",
    )
    context_parallel.require_cp_layout_api("tests").prebuild_thd_cp_partition_routes(params, setup.cp_group)
    return params


@dataclasses.dataclass
class _Run:
    tensors: dict[str, torch.Tensor]
    """``output``, ``input`` (its gradient) and every parameter gradient, detached copies."""
    dtypes: dict[str, torch.dtype]
    """Dtypes of the output and the input gradient."""


def _run(wrapper, hidden: torch.Tensor, grad_output: torch.Tensor, params) -> _Run:
    """Forward and backward through the wrapper's ``__call__`` (graph replay once captured, else eager)."""
    wrapper.zero_grad(set_to_none=True)
    inputs = hidden.detach().clone().requires_grad_(True)
    output, context = wrapper(hidden_states=inputs, attention_mask=None, packed_seq_params=params)
    assert context is None
    output.backward(grad_output)
    # Clone everything: a replay's output and gradients live in the graph's static buffers.
    tensors = {"output": output.detach().clone(), "input": inputs.grad.detach().clone()}
    tensors.update(
        {
            name: param.grad.detach().float().clone()
            for name, param in wrapper.named_parameters()
            if param.grad is not None
        }
    )
    return _Run(tensors=tensors, dtypes={"output": output.dtype, "input": inputs.grad.dtype})


@contextlib.contextmanager
def _eager(wrapper):
    """Run the wrapper eagerly while it has graphs (Megatron-Core replays whenever ``cuda_graphs`` is set)."""
    graphs, wrapper.cuda_graphs = wrapper.cuda_graphs, []
    try:
        yield
    finally:
        wrapper.cuda_graphs = graphs


@dataclasses.dataclass
class _Graph:
    graph: object
    replays: list[int]


def _capture(setup: _Setup, wrapper) -> _Graph:
    """Capture the wrapper as Megatron-Core's ``TECudaGraphHelper`` does at PP = 1, and install the graph."""
    from megatron.core.transformer import cuda_graphs
    from transformer_engine.pytorch.graph import make_graphed_callables

    static_inputs = wrapper.get_layer_static_inputs(_LOCAL_ROWS * setup.cp, 1)
    hidden = static_inputs.pop("hidden_states")
    torch.cuda.synchronize()
    cuda_graphs._set_capture_start()
    try:
        graphs = make_graphed_callables(
            (wrapper,),
            [(hidden,)],
            num_warmup_iters=_WARMUP_ITERS,
            allow_unused_input=True,
            sample_kwargs=[static_inputs],
            fp8_enabled=False,
            _order=[1, -1],
            _num_layers_per_chunk=[1],
            _reuse_graph_input_output_buffers=True,
        )
    finally:
        cuda_graphs._set_capture_end()
    torch.cuda.synchronize()
    captured = _Graph(graph=graphs[0], replays=[])

    def replay(*args, **kwargs):
        captured.replays.append(1)
        return captured.graph(*args, **kwargs)

    wrapper.cuda_graphs = [replay]
    return captured


def _release(wrapper, captured: _Graph) -> None:
    """Drop the graph as Megatron-Core's ``delete_cuda_graphs`` does: reset it, then release what it kept."""
    wrapper.cuda_graphs = []
    reset = getattr(captured.graph, "reset", None)
    if callable(reset):
        reset()
    wrapper.clear_te_cuda_graph_static_hidden_inputs()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------------------------
# Comparison and reporting
# ---------------------------------------------------------------------------------------------


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.detach().double().flatten(), b.detach().double().flatten()
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return 1.0 if torch.equal(a, b) else 0.0
    return (a @ b / denominator).item()


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.detach().double(), expected.detach().double()
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-300)).item()


def _noise(run: _Run, rerun: _Run) -> dict[str, tuple[float, float]]:
    """Run-to-run noise of two identical eager runs: ``(1 - cosine, relative L2)`` per tensor."""
    return {
        name: (max(0.0, 1.0 - _cosine(rerun.tensors[name], tensor)), _rel_err(rerun.tensors[name], tensor))
        for name, tensor in run.tensors.items()
    }


def _report(label: str, **metrics) -> None:
    import torch.distributed as dist

    text = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    logger.info("[ernie55-cuda-graph-gpu rank %d] %s: %s", dist.get_rank(), label, text)


def _compare(graph: _Run, eager: _Run, noise: dict[str, tuple[float, float]], label: str) -> list[str]:
    """Graph replay against eager execution of the same row (see the module docstring for the gates)."""
    failures = []
    for run, kind in ((graph, "graph"), (eager, "eager")):
        if run.dtypes != {"output": torch.float32, "input": torch.float32}:
            failures.append(f"{kind}: output and input gradient must be fp32, got {run.dtypes}")
    if set(graph.tensors) != set(eager.tensors):
        failures.append(f"gradients differ in presence: {sorted(set(graph.tensors) ^ set(eager.tensors))}")
    worst = (0.0, "", 1.0)
    for name in sorted(set(graph.tensors) & set(eager.tensors)):
        actual, expected = graph.tensors[name], eager.tensors[name]
        if not torch.isfinite(actual).all():
            failures.append(f"{name}: non-finite graph value")
            continue
        if expected.abs().max() == 0:
            if actual.abs().max() != 0:
                failures.append(f"{name}: eager gives exactly zero, the graph does not")
            continue
        cosine, rel_err = _cosine(actual, expected), _rel_err(actual, expected)
        cosine_noise, rel_noise = noise[name]
        min_cosine = min(_MIN_COSINE, 1.0 - _NOISE_MULTIPLE * cosine_noise)
        max_rel_err = max(_MAX_REL_ERR, _NOISE_MULTIPLE * rel_noise)
        if rel_err > worst[0]:
            worst = (rel_err, name, cosine)
        if cosine < min_cosine or rel_err > max_rel_err:
            failures.append(
                f"{name}: cosine {cosine:.7f} (bar {min_cosine:.7f}), relative L2 {rel_err:.3e} "
                f"(bar {max_rel_err:.3e}; eager run-to-run {rel_noise:.3e})"
            )
    _report(
        label,
        worst_rel_err=worst[0],
        worst_tensor=worst[1],
        worst_cosine=worst[2],
        output_rel_err=_rel_err(graph.tensors["output"], eager.tensors["output"]),
        max_eager_noise_rel=max(rel for _, rel in noise.values()),
    )
    return failures


def _assert_all_ranks(failures: list[str], label: str) -> None:
    """Fail on every rank if any rank failed, so no rank waits in a later collective."""
    import torch.distributed as dist

    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, failures)
    problems = [f"rank {rank}: {failure}" for rank, items in enumerate(gathered) for failure in items]
    assert not problems, f"{label}:\n" + "\n".join(problems[:40])


def _graph_against_eager(setup: _Setup, wrapper, captured: _Graph, params, *, seed: int, label: str) -> list[str]:
    """Replay the graph on a row, run the same row eagerly twice, and compare."""
    hidden, grad_output = _row(setup, wrapper, seed=seed)
    replays_before = len(captured.replays)
    graph = _run(wrapper, hidden, grad_output, params)
    failures = []
    if len(captured.replays) != replays_before + 1:
        failures.append(f"the graph ran {len(captured.replays) - replays_before} times, expected once")
    with _eager(wrapper):
        eager = _run(wrapper, hidden, grad_output, params)
        rerun = _run(wrapper, hidden, grad_output, params)
    if len(captured.replays) != replays_before + 1:
        failures.append("an eager run replayed the graph")
    return failures + _compare(graph, eager, _noise(eager, rerun), label)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


def test_graph_matches_eager(cp_setup):
    """G1: the full-size row shape, one document over the whole row."""
    _, wrapper = _build(cp_setup, documents=1)
    params = _packed_seq_params(cp_setup, _documents(cp_setup, ()))
    captured = _capture(cp_setup, wrapper)
    try:
        failures = _graph_against_eager(
            cp_setup, wrapper, captured, params, seed=11, label=f"G1 CP{cp_setup.cp} one document"
        )
    finally:
        _release(wrapper, captured)
    _assert_all_ranks(failures, f"G1 graph vs eager at CP = {cp_setup.cp}")


def test_graph_replays_new_rows(cp_setup):
    """G2: new rows replay correctly, with one document and with two at changing boundaries."""
    failures = []
    _, wrapper = _build(cp_setup, documents=1)
    captured = _capture(cp_setup, wrapper)
    try:
        params = _packed_seq_params(cp_setup, _documents(cp_setup, ()))
        for seed in (21, 22):
            label = f"G2 CP{cp_setup.cp} one document, row {seed}"
            problems = _graph_against_eager(cp_setup, wrapper, captured, params, seed=seed, label=label)
            failures += [f"{label}: {problem}" for problem in problems]
    finally:
        _release(wrapper, captured)

    _, wrapper = _build(cp_setup, documents=2)
    captured = _capture(cp_setup, wrapper)
    try:
        for fraction, seed in ((0.3, 31), (0.62, 32)):
            lengths = _documents(cp_setup, (fraction,))
            params = _packed_seq_params(cp_setup, lengths)
            label = f"G2 CP{cp_setup.cp} two documents {lengths}"
            problems = _graph_against_eager(cp_setup, wrapper, captured, params, seed=seed, label=label)
            failures += [f"{label}: {problem}" for problem in problems]
    finally:
        _release(wrapper, captured)
    _assert_all_ranks(failures, f"G2 new rows at CP = {cp_setup.cp}")


def test_graph_rejects_rows_it_cannot_replay(cp_setup):
    """G3: a two-document row raises before the one-document graph runs; rows of its shape still replay."""
    _, wrapper = _build(cp_setup, documents=1)
    captured = _capture(cp_setup, wrapper)
    failures = []
    try:
        hidden, _ = _row(cp_setup, wrapper, seed=41)
        two_documents = _packed_seq_params(cp_setup, _documents(cp_setup, (0.5,)))
        try:
            wrapper(hidden_states=hidden.clone().requires_grad_(True), packed_seq_params=two_documents)
        except ValueError as error:
            if "thd_max_packed_sequences=1" not in str(error):
                failures.append(f"unexpected error for a two-document row: {error}")
        else:
            failures.append("the one-document graph accepted a two-document row")
        if captured.replays:
            failures.append("the graph ran for a row it cannot replay")
        failures += _graph_against_eager(
            cp_setup,
            wrapper,
            captured,
            _packed_seq_params(cp_setup, _documents(cp_setup, ())),
            seed=42,
            label=f"G3 CP{cp_setup.cp} one document after a rejected row",
        )
    finally:
        _release(wrapper, captured)
    _assert_all_ranks(failures, f"G3 rejected rows at CP = {cp_setup.cp}")


def _profile_vha_mixes(setup: _Setup, wrapper, core, params) -> None:
    """One row through the eager batched mix, the eager ``torch.matmul`` mix and the captured batched graph, in one
    ``cudaProfilerStart``/``Stop`` range with NVTX ranges ``vha/G5/cp<CP>/<batched|matmul>/<eager|graph>``.

    The capture range of an nsys run of this file (``--capture-range=cudaProfilerApi``; no other test opens one):
    the copy kernels of ``torch.matmul``'s mix next to the batched GEMMs, eagerly and inside the H-layer graph.
    Nothing is compared here; without a profiler attached the range calls do nothing.
    """
    hidden, grad_output = _row(setup, wrapper, seed=54)
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        for name, batched_mix, mode in (
            ("batched", True, "eager"),
            ("matmul", False, "eager"),
            ("batched", True, "graph"),
        ):
            core.vha_batched_mix = batched_mix
            with _eager(wrapper) if mode == "eager" else contextlib.nullcontext():
                with torch.cuda.nvtx.range(f"vha/G5/cp{setup.cp}/{name}/{mode}"):
                    _run(wrapper, hidden, grad_output, params)
                    torch.cuda.synchronize()
    finally:
        core.vha_batched_mix = True
        torch.cuda.profiler.stop()


def test_batched_vha_graph_matches_eager(cp_setup, monkeypatch):
    """G5: ``ernie_vha_batched_mix``: every core forward takes the batched mix (none with the flag cleared), the eager
    forward is bitwise the ``torch.matmul`` mix's, gradients pass the gates, and the graph captured with it matches
    eager execution. At CP = 4 one row through each mix runs in the file's nsys capture range."""
    from tests.unit_tests.models.ernie.ernie55_test_utils import record_batched_vha_mixes

    _, wrapper = _build(cp_setup, documents=1, ernie_vha_batched_mix=True)
    core = wrapper.inner_layer.self_attention.core_attention
    params = _packed_seq_params(cp_setup, _documents(cp_setup, ()))
    hidden, grad_output = _row(cp_setup, wrapper, seed=51)
    label = f"G5 CP{cp_setup.cp}"
    failures = [] if core.vha_batched_mix is True else [f"{label}: the H core did not take ernie_vha_batched_mix"]
    # Both mixes compute the same output, so the comparisons below cannot show which one ran: count the batched mix's
    # calls against the core's forwards in each phase (the capture's warmup and capture forwards included).
    mixes, forwards = record_batched_vha_mixes(monkeypatch), []
    hook = core.register_forward_hook(lambda module, args, output: forwards.append(1))
    head_shape = (core.vha_postmix_u.size(0), wrapper.config.v_head_dim)

    def check_mixes(phase: str, *, batched: bool) -> None:
        """One batched mix of ``[t, H, D]`` rows per core forward since the last check, or none."""
        expected = len(forwards) if batched else 0
        if not forwards or len(mixes) != expected or any(shape[1:] != head_shape for shape in mixes):
            failures.append(
                f"{label} {phase}: batched VHA mixes of rows {mixes} in {len(forwards)} core forwards, expected "
                f"{expected} of rows [t, {head_shape[0]}, {head_shape[1]}]"
            )
        mixes.clear()
        forwards.clear()

    try:
        batched = _run(wrapper, hidden, grad_output, params)
        check_mixes("eager run", batched=True)
        core.vha_batched_mix = False
        try:
            matmul = _run(wrapper, hidden, grad_output, params)
            rerun = _run(wrapper, hidden, grad_output, params)
        finally:
            core.vha_batched_mix = True
        check_mixes("eager runs with the flag cleared", batched=False)
        if torch.equal(matmul.tensors["output"], rerun.tensors["output"]):
            differing = int((batched.tensors["output"] != matmul.tensors["output"]).sum().item())
            _report(f"{label} batched vs matmul VHA forward", differing_entries=differing)
            if differing:
                failures.append(f"{label}: {differing} output entries differ from the torch.matmul mix")
        else:
            _report(f"{label} batched vs matmul VHA forward", note="eager forward not bitwise run to run; gates only")
        failures += [
            f"{label} batched vs matmul VHA: {problem}"
            for problem in _compare(batched, matmul, _noise(matmul, rerun), f"{label} batched vs matmul VHA (eager)")
        ]
        captured = _capture(cp_setup, wrapper)
        try:
            check_mixes("capture", batched=True)
            for seed in (52, 53):
                problems = _graph_against_eager(
                    cp_setup, wrapper, captured, params, seed=seed, label=f"{label} batched VHA graph, row {seed}"
                )
                failures += [f"{label} row {seed}: {problem}" for problem in problems]
            check_mixes("eager runs of the graph rows", batched=True)
            if cp_setup.cp == _PROFILED_CP:
                _profile_vha_mixes(cp_setup, wrapper, core, params)
        finally:
            _release(wrapper, captured)
    finally:
        hook.remove()
    _assert_all_ranks(failures, f"G5 batched VHA at CP = {cp_setup.cp}")


def test_graph_outlives_the_shared_rope_cache(cp_setup):
    """G4: the graph keeps reading a live RoPE table after Megatron-Core's shared RoPE cache drops it."""
    import gc
    import weakref

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    from megatron.bridge.models.ernie.modeling_ernie55 import cuda_graph

    _, wrapper = _build(cp_setup, documents=1)
    params = _packed_seq_params(cp_setup, _documents(cp_setup, ()))
    captured = _capture(cp_setup, wrapper)
    failures = []
    try:
        tables = [weakref.ref(table) for table in getattr(wrapper, cuda_graph._ROPE_TABLES, {}).values()]
        if not tables:
            failures.append("the capture kept no RoPE table")
        # In training the eager D and MTP layers' lookups evict the H tables from the 32-entry cache; eval CP
        # rebinding and YaRN's constructor clear it. Then hand the freed memory to other tensors.
        RotaryEmbedding.forward.cache_clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        shape = (_LOCAL_ROWS * cp_setup.cp, 1, 1, wrapper.config.qk_pos_emb_head_dim)
        garbage = [torch.full(shape, float("nan"), device=_device()) for _ in range(8)]
        if any(ref() is None for ref in tables):
            failures.append("a RoPE table the graph reads was freed with the shared cache")
        label = f"G4 CP{cp_setup.cp} after the shared RoPE cache was cleared"
        failures += _graph_against_eager(cp_setup, wrapper, captured, params, seed=51, label=label)
        del garbage
    finally:
        _release(wrapper, captured)
    if getattr(wrapper, cuda_graph._ROPE_TABLES, None) is not None:
        failures.append("deleting the graph did not release its RoPE tables")
    _assert_all_ranks(failures, f"G4 RoPE tables at CP = {cp_setup.cp}")
