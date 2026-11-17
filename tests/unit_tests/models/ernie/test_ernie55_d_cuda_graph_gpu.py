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

"""GPU tests of the ERNIE 5.5 D-layer Transformer Engine CUDA graphs: captured graphs against eager execution.

Needs four GPUs (SM90+; the deterministic case SM100), Transformer Engine, Triton, FlashMLA, the cuDNN frontend DSA
kernels, CuTe DSL, ``fast_hadamard_transform`` and a Megatron-Core dev build with the CP partition converter and the
DSv4 ``_post_o_group_proj`` hook (dev f78809f6d or later). Launch on all four ranks (CP = 2 runs as two CP groups of
two ranks, CP = 4 as one group)::

    ERNIE55_GPU_STRICT=1 NCCL_GRAPH_REGISTER=0 uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest \\
        -rA -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_d_cuda_graph_gpu.py

Without them the tests skip and name what is missing; ``ERNIE55_GPU_STRICT=1`` turns the skips into failures. Every
rank runs the same tests; failures are gathered so all ranks fail together. Under ``nsys profile`` each test, capture,
graph replay and eager run is an NVTX range (``dgraph ...``), and the replays' and eager runs' times are logged.

The models are ERNIE 5.5 models with D layers only (H layers have their own suite,
``test_ernie55_cuda_graph_gpu.py``) at CP = 2 and CP = 4 (contiguous THD, 1024 rows per rank), fp32 decoder streams,
the fused mHC kernels, ``layernorm`` recompute, Transformer Engine graphs of the attention with
``ernie_graph_d_layers`` and the TE RNG tracker, so their decoder's D wrappers are
``Ernie55DHyperConnectionHybridLayer``. The tests capture the
wrappers with Transformer Engine's ``make_graphed_callables`` on the wrappers' own static inputs, with the arguments
Megatron-Core's ``TECudaGraphHelper`` passes at PP = 1 (one graph per wrapper, order ``[1, -1]``, buffer reuse), and
replay them through the wrappers' ``__call__``, as training does. Rows are built by ``build_ernie55_thd_cp_batch``
(prebuilt routes and zigzag geometry). The fixed-capacity indexer-loss tracker is initialized before the capture, as
Bridge's setup does.

Gate of a graph replay against an eager run of the same row: the output, the input gradient and every parameter
gradient have cosine >= 0.9999 and relative L2 <= 1e-2, widened to 3 times the eager run-to-run noise where that is
larger (the cuDNN DSA backward and the indexer backward accumulate with atomics); the output is bitwise equal to the
eager one whenever two eager runs are. The graph is replayed (not run eagerly) exactly once per graph run.

* ``test_d_graph_matches_eager`` (D1): one D layer, one document over the whole row (the SFT recipe's shape), the
  indexer loss on; the indexer-loss scale is set to 0.5 after the capture, so a graph that baked in the scale's value
  instead of reading the scale tensor in place would get indexer gradients twice the eager ones, and the tracker's
  indexer loss of the replay must equal the eager one (the captured in-place writes).
* ``test_d_graph_falls_back_for_other_compositions`` (D2): a two-document row runs the D layer eagerly (the graph is
  not replayed) and matches eager execution; a row of the captured composition replays again afterwards.
* ``test_mtp_shared_gradients`` (D3): ``D-DE/DE`` with SD-MTP (the MTP depth's D layer shares the decoder's last D
  layer's parameters), deterministic mode and no indexer loss, the whole model's forward and backward.
  ``[mtp-graphed]`` captures the decoder's two D wrappers and the MTP depth's in one call (TE's buffer reuse across
  all three), ``[mtp-eager]`` the decoder's two. Every shared parameter's gradient must be bitwise the sum of the
  contributions the graphs returned (hooks on TE's ``Graphed`` nodes clone them when they are returned, so a reused
  buffer would show), every gradient must pass the gate against eager execution, and every gradient that two eager
  runs reproduce bitwise (the shared ones included) must be bitwise the eager one.
* ``test_d_graph_outlives_the_shared_rope_cache`` (D4): after capture, Megatron-Core's shared RoPE cache is cleared,
  the allocator's cache is emptied and NaN tensors of the tables' size are allocated; the RoPE tables the graph reads
  are still alive and the graph still matches eager execution.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
import logging
import os
import time

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
_PACKED_FIELDS = ("cu_seqlens_q", "cu_seqlens_kv", "cu_seqlens_q_padded", "cu_seqlens_kv_padded")

_COMMON_CONFIG = dict(
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
    thd_max_packed_sequences=1,
    ernie_graph_d_layers=True,
)
# The stack geometry the DSv4 hybrid config needs (test_ernie55_cuda_graph_gpu.py's H geometry).
_STACK_GEOMETRY = dict(
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
# The D layer: FlashMLA d_qk = 576 (512 + 64), 64 heads; cuDNN indexer heads of width 128.
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
# MoE layers of the SD-MTP model (test_ernie55_stack_gpu.py's, at EP = 1).
_MOE = dict(
    num_moe_experts=8,
    moe_router_topk=2,
    moe_ffn_hidden_size=256,
    moe_shared_expert_intermediate_size=256,
    moe_grouped_gemm=True,
    moe_router_score_function="sigmoid",
    moe_router_enable_expert_bias=True,
    moe_router_bias_update_rate=1e-4,
    moe_router_load_balancing_type="none",
    moe_aux_loss_coeff=0.0,
    moe_router_dtype="fp32",
    moe_token_dispatcher_type="alltoall",
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
    for module_name, label in (
        ("transformer_engine.pytorch", "Transformer Engine"),
        ("triton", "Triton"),
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


@contextlib.contextmanager
def _nvtx(label: str):
    """An NVTX range named ``dgraph <label>`` for nsys timelines of the suite (no cost without a profiler)."""
    torch.cuda.nvtx.range_push(f"dgraph {label}")
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def _timed(label: str, function, *args):
    """``function(*args)`` in an NVTX range, between device synchronizations; returns (result, wall time in ms)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    with _nvtx(label):
        result = function(*args)
        torch.cuda.synchronize()
    return result, 1e3 * (time.perf_counter() - start)


@dataclasses.dataclass
class _Setup:
    cp: int
    cp_group: object
    cp_rank: int
    pg_collection: object


@pytest.fixture(scope="module")
def distributed():
    """The NCCL default process group over the four ranks, created once for the module."""
    missing = _missing_dependencies()
    if missing:
        _skip("the ERNIE 5.5 D-layer CUDA graph GPU tests need " + "; ".join(missing))
    import torch.distributed as dist

    created_group = False
    if not dist.is_initialized():
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend="nccl")
        created_group = True
    yield
    if created_group:
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[2, 4], ids=["cp2", "cp4"])
def cp_setup(request, distributed):
    """Megatron-Core's model-parallel state at TP = PP = EP = 1 and the parametrized CP size."""
    from megatron.core import parallel_state
    from megatron.core.process_groups_config import ProcessGroupCollection

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
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    parallel_state.destroy_model_parallel()


@pytest.fixture(autouse=True)
def _fresh_state(request):
    from megatron.core.transformer.experimental_attention_variant import dsa
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    from megatron.bridge.models.ernie.modeling_ernie55 import cuda_graph

    def reset() -> None:
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()
        dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None
        MTPLossLoggingHelper.tracker.clear()
        cuda_graph._WARNED_FALLBACKS.clear()

    reset()
    with _nvtx(request.node.name):
        yield
    reset()


# ---------------------------------------------------------------------------------------------
# Builders, rows, runs
# ---------------------------------------------------------------------------------------------


def _build(setup: _Setup, *, pattern: str, seed: int = _SEED, **overrides):
    """An ERNIE 5.5 model of ``pattern`` with D-layer graphs, weights equal over the CP group, tracker initialized."""
    import torch.distributed as dist
    import torch.nn.functional as F
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.experimental_attention_variant.dsa_logging import initialize_dsa_metric_tracker

    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig
    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HybridModel, ernie55_hybrid_stack_spec

    main_pattern, *mtp_patterns = pattern.split("/")
    values = {**_COMMON_CONFIG, **_STACK_GEOMETRY, **_D_GEOMETRY}
    values.update(
        num_layers=len(main_pattern),
        csa_compress_ratios=[0] * len(main_pattern + "".join(mtp_patterns)),
        context_parallel_size=setup.cp,
        activation_func=F.silu,
    )
    if mtp_patterns:
        values.update(
            _MOE, mtp_num_layers=len(mtp_patterns), ernie_mtp_shared_last_layer=True, ernie_per_row_loss=True
        )
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
        hybrid_layer_pattern=pattern,
        position_embedding_type="none",
    )
    model = model.to(_device()).train()
    generator = torch.Generator(device=_device()).manual_seed(seed + 1)
    source = dist.get_global_rank(setup.cp_group, 0)
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Move the sinks and the VHA factors away from their init so the comparison exercises them.
            if name.endswith(("attn_sink", "vha_postmix_u", "vha_postmix_v")):
                std = 1.0 if name.endswith("attn_sink") else 0.3
                param.copy_(std * torch.randn(param.shape, generator=generator, device=param.device))
            dist.broadcast(param.data, src=source, group=setup.cp_group)
    dsa_tracker_reset()
    initialize_dsa_metric_tracker([model], setup.pg_collection)
    return model


def dsa_tracker_reset() -> None:
    from megatron.core.transformer.experimental_attention_variant import dsa

    dsa.DSAIndexerLossLoggingHelper.tracker.clear()


def _batch(setup: _Setup, lengths: tuple[int, ...], *, seed: int):
    """A packed row of documents of ``lengths`` tokens, built as ``ernie55_step`` builds it; this rank's rows."""
    from megatron.bridge.models.ernie.ernie55_context_parallel import IGNORE_LABEL, build_ernie55_thd_cp_batch

    generator = torch.Generator().manual_seed(seed)
    samples = []
    for length in lengths:
        tokens = torch.randint(1, _VOCAB, (length,), generator=generator)
        labels = tokens.roll(-1)
        labels[-1] = IGNORE_LABEL
        samples.append((tokens, labels))
    return build_ernie55_thd_cp_batch(
        samples, pad_token_id=0, cp_group=setup.cp_group, cp_partition_mode="contiguous", device=_device()
    )


def _two_document_lengths(setup: _Setup, fraction: float) -> tuple[int, int]:
    total, multiple = _LOCAL_ROWS * setup.cp, 2 * setup.cp
    first = max(multiple, round(total * fraction / multiple) * multiple)
    return first, total - first


def _row(setup: _Setup, wrapper, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's contiguous rows of one packed row: fp32 input streams (std 0.02) and the output gradient."""
    total = _LOCAL_ROWS * setup.cp
    width = _STREAMS * wrapper.config.hidden_size
    generator = torch.Generator(device=_device()).manual_seed(seed)
    hidden = _INPUT_STD * torch.randn(total, 1, width, device=_device(), generator=generator)
    grad_output = torch.randn(total, 1, width, device=_device(), generator=generator)
    rows = slice(setup.cp_rank * _LOCAL_ROWS, (setup.cp_rank + 1) * _LOCAL_ROWS)
    return hidden[rows].contiguous(), grad_output[rows].contiguous()


@dataclasses.dataclass
class _Run:
    tensors: dict[str, torch.Tensor]
    """``output``, ``input`` (its gradient) and every parameter gradient, detached copies."""
    indexer_loss: torch.Tensor
    """The tracker's indexer-loss values after the run."""


def _run_wrapper(wrapper, hidden: torch.Tensor, grad_output: torch.Tensor, params) -> _Run:
    """Forward and backward through the wrapper's ``__call__`` (graph replay once captured, else eager)."""
    from megatron.core.transformer.experimental_attention_variant import dsa

    tracker = dsa.DSAIndexerLossLoggingHelper.tracker
    tracker["values"].zero_()
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
    return _Run(tensors=tensors, indexer_loss=tracker["values"].detach().clone())


@contextlib.contextmanager
def _eager(*wrappers):
    """Run the wrappers eagerly while they have graphs (Megatron-Core replays whenever ``cuda_graphs`` is set)."""
    graphs = [wrapper.cuda_graphs for wrapper in wrappers]
    for wrapper in wrappers:
        wrapper.cuda_graphs = []
    try:
        yield
    finally:
        for wrapper, graph in zip(wrappers, graphs):
            wrapper.cuda_graphs = graph


@dataclasses.dataclass
class _Graph:
    graph: object
    replays: list[int]


def _capture(setup: _Setup, wrappers: list, *, wrap=None) -> list[_Graph]:
    """Capture the wrappers in one call as Megatron-Core's ``TECudaGraphHelper`` does at PP = 1; install the graphs.

    ``wrap(wrapper, graph)``, if given, returns the callable installed instead of the graph (it must call it).
    """
    from megatron.core.transformer import cuda_graphs
    from megatron.core.transformer.experimental_attention_variant import dsa
    from transformer_engine.pytorch.graph import make_graphed_callables

    sample_args, sample_kwargs = [], []
    for wrapper in wrappers:
        static_inputs = wrapper.get_layer_static_inputs(_LOCAL_ROWS * setup.cp, 1)
        sample_args.append((static_inputs.pop("hidden_states"),))
        sample_kwargs.append(static_inputs)
    torch.cuda.synchronize()
    cuda_graphs._set_capture_start()
    try:
        with _nvtx(f"capture of {len(wrappers)} wrapper(s)"):
            graphs = make_graphed_callables(
                tuple(wrappers),
                sample_args,
                num_warmup_iters=_WARMUP_ITERS,
                allow_unused_input=True,
                sample_kwargs=sample_kwargs,
                fp8_enabled=False,
                _order=[1, -1],
                _num_layers_per_chunk=[len(wrappers)],
                _reuse_graph_input_output_buffers=True,
            )
    finally:
        cuda_graphs._set_capture_end()
    torch.cuda.synchronize()
    # Megatron-Core's helper after its capture: drop the warmups' tracker writes, keep the reduction groups.
    dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
    captured = []
    for wrapper, graph in zip(wrappers, graphs):
        record = _Graph(graph=graph, replays=[])
        target = wrap(wrapper, graph) if wrap is not None else graph

        def replay(*args, _record=record, _target=target, **kwargs):
            _record.replays.append(1)
            return _target(*args, **kwargs)

        wrapper.cuda_graphs = [replay]
        captured.append(record)
    return captured


def _release(wrappers: list, captured: list[_Graph]) -> None:
    """Drop the graphs as Megatron-Core's ``delete_cuda_graphs`` does: reset them, then release what they kept."""
    for wrapper, record in zip(wrappers, captured):
        wrapper.cuda_graphs = []
        reset = getattr(record.graph, "reset", None)
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


def _noise(run: dict[str, torch.Tensor], rerun: dict[str, torch.Tensor]) -> dict[str, tuple[float, float]]:
    """Run-to-run noise of two identical eager runs: ``(1 - cosine, relative L2)`` per tensor."""
    return {
        name: (max(0.0, 1.0 - _cosine(rerun[name], tensor)), _rel_err(rerun[name], tensor))
        for name, tensor in run.items()
        if name in rerun
    }


def _report(label: str, **metrics) -> None:
    import torch.distributed as dist

    text = ", ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    logger.info("[ernie55-d-cuda-graph-gpu rank %d] %s: %s", dist.get_rank(), label, text)


def _compare(
    graph: dict[str, torch.Tensor],
    eager: dict[str, torch.Tensor],
    rerun: dict[str, torch.Tensor],
    label: str,
    *,
    bitwise: tuple[str, ...] = ("output",),
) -> list[str]:
    """Graph against eager: the gate for every tensor, and bitwise for ``bitwise`` names two eager runs reproduce."""
    failures = []
    noise = _noise(eager, rerun)
    if set(graph) != set(eager):
        failures.append(f"tensors differ in presence: {sorted(set(graph) ^ set(eager))}")
    worst = (0.0, "", 1.0)
    bitwise_checked = 0
    for name in sorted(set(graph) & set(eager)):
        actual, expected = graph[name], eager[name]
        if not torch.isfinite(actual).all():
            failures.append(f"{name}: non-finite graph value")
            continue
        if name in bitwise and torch.equal(eager[name], rerun[name]):
            bitwise_checked += 1
            if not torch.equal(actual, expected):
                failures.append(
                    f"{name}: two eager runs agree bitwise, the graph differs ({_rel_err(actual, expected):.3e})"
                )
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
        bitwise_checked=bitwise_checked,
        bitwise_requested=len([name for name in bitwise if name in eager]),
        max_eager_noise_rel=max((rel for _, rel in noise.values()), default=0.0),
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
    """Replay the graph on a row, run the same row eagerly twice, and compare (also the tracker's indexer loss)."""
    hidden, grad_output = _row(setup, wrapper, seed=seed)
    replays_before = len(captured.replays)
    graph, graph_ms = _timed(f"{label}: graph replay", _run_wrapper, wrapper, hidden, grad_output, params)
    failures = []
    if len(captured.replays) != replays_before + 1:
        failures.append(f"the graph ran {len(captured.replays) - replays_before} times, expected once")
    with _eager(wrapper):
        eager, eager_ms = _timed(f"{label}: eager", _run_wrapper, wrapper, hidden, grad_output, params)
        rerun, rerun_ms = _timed(f"{label}: eager again", _run_wrapper, wrapper, hidden, grad_output, params)
    _report(f"{label}, forward and backward", graph_ms=graph_ms, eager_ms=eager_ms, eager_again_ms=rerun_ms)
    if len(captured.replays) != replays_before + 1:
        failures.append("an eager run replayed the graph")
    for run, kind in ((graph, "graph"), (eager, "eager")):
        if run.tensors["output"].dtype != torch.float32 or run.tensors["input"].dtype != torch.float32:
            failures.append(f"{kind}: output and input gradient must be fp32")
    tensors = {name: dict(run.tensors) for name, run in (("graph", graph), ("eager", eager), ("rerun", rerun))}
    for name, run in (("graph", graph), ("eager", eager), ("rerun", rerun)):
        tensors[name]["indexer_loss"] = run.indexer_loss
    return failures + _compare(tensors["graph"], tensors["eager"], tensors["rerun"], label)


def _set_indexer_loss_scale(*, cp_size: int, num_microbatches: int) -> None:
    """Megatron-Core's schedule publishes the indexer-loss scale (size 1, ``cp_size / num_microbatches``)."""
    from tests.unit_tests.models.ernie.test_ernie55_d_cuda_graph import _publish_indexer_loss_scale

    _publish_indexer_loss_scale(_device(), cp_size=cp_size, num_microbatches=num_microbatches)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


def test_d_graph_matches_eager(cp_setup):
    """D1: the SFT recipe's shape, one document over the whole row, with the indexer loss."""
    model = _build(cp_setup, pattern="D-")
    wrapper = model.decoder.layers[0]
    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55DHyperConnectionHybridLayer

    assert type(wrapper) is Ernie55DHyperConnectionHybridLayer, type(wrapper).__name__
    params = _batch(cp_setup, (_LOCAL_ROWS * cp_setup.cp,), seed=3).packed_seq_params
    (captured,) = _capture(cp_setup, [wrapper])
    failures = []
    try:
        # After the capture, as the schedule does after a forward (0.5 = CP 1 x 1 / 2 micro-batches): a graph that
        # baked in the scale's value would double the indexer gradients.
        _set_indexer_loss_scale(cp_size=1, num_microbatches=2)
        failures += _graph_against_eager(
            cp_setup, wrapper, captured, params, seed=11, label=f"D1 CP{cp_setup.cp} one document"
        )
        failures += _graph_against_eager(
            cp_setup, wrapper, captured, params, seed=12, label=f"D1 CP{cp_setup.cp} one document, second row"
        )
    finally:
        _release([wrapper], [captured])
    _assert_all_ranks(failures, f"D1 graph vs eager at CP = {cp_setup.cp}")


def test_d_graph_falls_back_for_other_compositions(cp_setup):
    """D2: a two-document row runs eagerly and matches eager execution; the captured composition still replays."""
    from megatron.bridge.models.ernie.modeling_ernie55 import cuda_graph

    model = _build(cp_setup, pattern="D-")
    wrapper = model.decoder.layers[0]
    one = _batch(cp_setup, (_LOCAL_ROWS * cp_setup.cp,), seed=3).packed_seq_params
    two = _batch(cp_setup, _two_document_lengths(cp_setup, 0.37), seed=4).packed_seq_params
    (captured,) = _capture(cp_setup, [wrapper])
    failures = []
    try:
        label = f"D2 CP{cp_setup.cp} two documents, eager fallback"
        hidden, grad_output = _row(cp_setup, wrapper, seed=31)
        fallback, fallback_ms = _timed(f"{label}: fallback", _run_wrapper, wrapper, hidden, grad_output, two)
        if captured.replays:
            failures.append("the graph replayed a two-document row")
        if "boundaries" not in cuda_graph._WARNED_FALLBACKS:
            failures.append("the fallback was not logged")
        with _eager(wrapper):
            eager, eager_ms = _timed(f"{label}: eager", _run_wrapper, wrapper, hidden, grad_output, two)
            rerun, _ = _timed(f"{label}: eager again", _run_wrapper, wrapper, hidden, grad_output, two)
        _report(f"{label}, forward and backward", fallback_ms=fallback_ms, eager_ms=eager_ms)
        failures += _compare(fallback.tensors, eager.tensors, rerun.tensors, label)
        failures += _graph_against_eager(
            cp_setup, wrapper, captured, one, seed=32, label=f"D2 CP{cp_setup.cp} one document after a fallback"
        )
    finally:
        _release([wrapper], [captured])
    _assert_all_ranks(failures, f"D2 fallback at CP = {cp_setup.cp}")


def _contribution_recorder(shared: dict[int, str], store: dict[str, list[torch.Tensor]]):
    """``wrap`` for :func:`_capture`: record each shared parameter's gradient as TE's ``Graphed`` node returns it."""

    def wrap(wrapper, graph):
        def call(*args, **kwargs):
            outputs = graph(*args, **kwargs)
            first = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            node = first.grad_fn
            positions = {
                index: shared[id(function.variable)]
                for index, (function, _) in enumerate(node.next_functions)
                if function is not None and id(getattr(function, "variable", None)) in shared
            }

            def hook(grad_inputs, grad_outputs):
                assert len(grad_inputs) == len(node.next_functions), (len(grad_inputs), len(node.next_functions))
                for index, name in positions.items():
                    if grad_inputs[index] is not None:
                        store.setdefault(name, []).append(grad_inputs[index].detach().clone())

            node.register_hook(hook)
            return outputs

        return call

    return wrap


@pytest.mark.parametrize("mtp_graphs", [True, False], ids=["mtp-graphed", "mtp-eager"])
def test_mtp_shared_gradients(cp_setup, mtp_graphs):
    """D3: SD-MTP's shared parameters get the eager summed gradient with the D layers graphed."""
    from megatron.core.transformer.experimental_attention_variant import dsa
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    try:
        model = _build(
            cp_setup,
            pattern="D-DE/DE",
            deterministic_mode=True,
            dsa_indexer_loss_coeff=0.0,
            ernie_graph_mtp_layers=mtp_graphs,
        )
    except Exception as error:  # noqa: BLE001 - name what this environment lacks
        _skip(f"the deterministic SD-MTP model does not build here: {type(error).__name__}: {error}")
    decoder_d = [model.decoder.layers[0], model.decoder.layers[2]]
    mtp_d = model.mtp.layers[0].mtp_model_layer.layers[0]
    wrappers = decoder_d + ([mtp_d] if mtp_graphs else [])
    names = {id(param): name for name, param in model.named_parameters()}
    shared = {id(param): names[id(param)] for param in mtp_d.parameters()}
    assert set(shared) <= {id(param) for param in decoder_d[1].parameters()}, "the MTP D layer shares nothing"
    batch = _batch(cp_setup, (_LOCAL_ROWS * cp_setup.cp,), seed=5)
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=_device()))

    def run() -> dict[str, torch.Tensor]:
        model.zero_grad(set_to_none=True)
        dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker(preserve_groups=True)
        losses = model(
            input_ids=batch.tokens,
            position_ids=batch.position_ids,
            attention_mask=None,
            labels=batch.labels.clamp(min=0),
            loss_mask=batch.loss_mask,
            packed_seq_params=batch.packed_seq_params,
        )
        (losses.float() * batch.loss_mask).sum().backward()
        tensors = {"losses": losses.detach().float().clone()}
        tensors.update(
            {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
        )
        return tensors

    label = f"D3 CP{cp_setup.cp} {'MTP graphed' if mtp_graphs else 'MTP eager'}"
    try:
        eager, eager_ms = _timed(f"{label}: eager model", run)
        rerun, _ = _timed(f"{label}: eager model again", run)
    except RuntimeError as error:
        if "deterministic" in str(error):
            _skip(f"this environment has no deterministic DSA sparse-attention backward: {error}")
        raise
    store: dict[str, list[torch.Tensor]] = {}
    captured = _capture(cp_setup, wrappers, wrap=_contribution_recorder(shared, store))
    failures = []
    try:
        graph, graph_ms = _timed(f"{label}: graphed model", run)
        _report(f"{label}, model forward and backward", graphed_ms=graph_ms, eager_ms=eager_ms)
        for record, wrapper in zip(captured, wrappers):
            if len(record.replays) != 1:
                failures.append(f"layer {wrapper.layer_number}: {len(record.replays)} replays, expected one")
        # The mechanism: each shared gradient is exactly what the graphs returned, summed (none overwritten).
        params = dict(model.named_parameters())
        expected_users = 2 if mtp_graphs else 1
        for name in sorted(shared.values()):
            contributions = store.get(name, [])
            if len(contributions) != expected_users:
                failures.append(f"{name}: {len(contributions)} graph contributions, expected {expected_users}")
                continue
            if mtp_graphs and not torch.equal(params[name].grad, contributions[0] + contributions[1]):
                failures.append(f"{name}: the gradient is not the sum of the two graphs' returned contributions")
        failures += _compare(graph, eager, rerun, label, bitwise=tuple(graph))
        reproducible = [name for name in shared.values() if torch.equal(eager[name], rerun[name])]
        _report(label, shared=len(shared), shared_bitwise_reproducible=len(reproducible))
        if not reproducible:
            logger.warning("%s: no shared gradient is bitwise reproducible eagerly; only the gate applies", label)
    finally:
        _release(wrappers, captured)
    _assert_all_ranks(failures, f"D3 SD-MTP shared gradients at CP = {cp_setup.cp}")


def test_d_graph_outlives_the_shared_rope_cache(cp_setup):
    """D4: the graph keeps reading live RoPE tables after Megatron-Core's shared RoPE cache drops them."""
    import gc
    import weakref

    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding

    from megatron.bridge.models.ernie.modeling_ernie55 import cuda_graph

    model = _build(cp_setup, pattern="D-")
    wrapper = model.decoder.layers[0]
    params = _batch(cp_setup, (_LOCAL_ROWS * cp_setup.cp,), seed=3).packed_seq_params
    (captured,) = _capture(cp_setup, [wrapper])
    failures = []
    try:
        tables = [weakref.ref(table) for table in getattr(wrapper, cuda_graph._ROPE_TABLES, {}).values()]
        if len(tables) < 2:
            failures.append(f"the capture kept {len(tables)} RoPE tables; the attention and its indexer read one each")
        RotaryEmbedding.forward.cache_clear()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        shape = (_LOCAL_ROWS * cp_setup.cp, 1, 1, 64)
        garbage = [torch.full(shape, float("nan"), device=_device()) for _ in range(8)]
        if any(ref() is None for ref in tables):
            failures.append("a RoPE table the graph reads was freed with the shared cache")
        label = f"D4 CP{cp_setup.cp} after the shared RoPE cache was cleared"
        failures += _graph_against_eager(cp_setup, wrapper, captured, params, seed=51, label=label)
        del garbage
    finally:
        _release([wrapper], [captured])
    if getattr(wrapper, cuda_graph._ROPE_TABLES, None) is not None:
        failures.append("deleting the graph did not release its RoPE tables")
    if cuda_graph._D_GRAPH_PLAN in wrapper.__dict__:
        failures.append("deleting the graph did not release its host plan")
    _assert_all_ranks(failures, f"D4 RoPE tables at CP = {cp_setup.cp}")
