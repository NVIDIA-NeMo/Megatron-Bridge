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

"""GPU tests of the ERNIE 5.5 D layer: fused kernels against the PyTorch reference, and training plumbing.

Every test needs one CUDA device, Transformer Engine and ``fast_hadamard_transform`` (the
``reference`` requirements, which Bridge CI meets). Tests that take the ``fused_kernels`` fixture
also need an SM90+ GPU, FlashMLA (``flash_mla_sparse_fwd``, nv_dev branch) and the cuDNN frontend
DSA kernels (``nvidia-cudnn-frontend[cutedsl]`` >= 1.26); they skip and name what is missing.

What is covered:

* ``test_fused_matches_reference``: fused vs reference forward, tie-aware top-k sets, indexer KL and
  every gradient, for a production-geometry THD pack and SBHD batches with ``b = 2`` (per-row loss)
  and ``b = 3`` (per-token loss). With ``b > 1`` the core calls the loss hook once per sample, because
  Megatron-Core's batched sparse indexer loss drops the indexer gradient of the later samples. The
  reference gathers the latent key in fp32; with a bf16 key its dKV was off by about 4.5% at the
  production geometry, so these comparisons would only test the reference's rounding.
* ``test_edge_rows_match_reference`` and ``test_rows_without_remote_candidates``: rows without remote
  candidates. The kernels must never see an empty candidate row, and such rows add no KL.
* ``test_long_pack_topk_beyond_32k_rows``: a pack longer than 32768 rows as the second fused call.
* ``test_flash_mla_lse_excludes_sink`` and ``test_sink_gradient``: the sink gradient. The native
  cuDNN ``d_sink`` (the ``ernie_native_sink_grad`` default) and the analytic fallback must both match
  autograd and each other.
* ``test_fused_determinism``: bitwise forward and top-k; bitwise non-indexer gradients when the
  Megatron-Core build has a deterministic sparse-attention backward for ``d_qk = 576`` (dev with
  cudnn-frontend >= 1.28 on an SM100-family GPU, probed with the exact backward call).
* Toy ``HybridModel`` (``"D-D-"``) trained through ``get_forward_backward_func`` under
  ``Float16Module``, DDP with ``overlap_grad_reduce`` and the distributed optimizer: it trains,
  indexer gradients scale like the LM loss with the number of micro-batches (per-row and per-token
  loss), the indexer learns from the KL, full recompute matches, and a ``torch_dist`` checkpoint
  round-trips the new parameters and their optimizer state in both distributed-optimizer formats
  Bridge saves (``dp_reshardable``, ``fully_reshardable``).

Run on one GPU; the ``log_cli`` options print the measured metrics of every test::

    uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -rA \
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_latent_mqa_gpu.py

Set ``ERNIE55_GPU_LARGE=1`` to add the 8K single-document fused-vs-reference case; its reference
path needs about 100 GB of device memory.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import itertools
import logging
import math
import os

import pytest
import torch


logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the ERNIE 5.5 D-layer GPU tests need a CUDA device"),
]

# Shared settings of every config below; the ERNIE indexer flags not listed keep their defaults.
_COMMON_CONFIG = dict(
    normalization="RMSNorm",
    layernorm_epsilon=1e-6,
    rope_type="rope",
    rotary_base=10000.0,
    attention_dropout=0.0,
    hidden_dropout=0.0,
    add_bias_linear=False,
    experimental_attention_variant="dsa",
    dsa_indexer_loss_coeff=0.01,
    dsa_indexer_use_sparse_loss=True,
    dsa_indexer_k_norm_fp32=True,
    bf16=True,
    params_dtype=torch.bfloat16,
    init_method_std=0.02,
)
# The ERNIE 5.5 D-layer attention geometry (FlashMLA d_qk = 576, d_v = 512) at a small hidden size.
_PRODUCTION_GEOMETRY = dict(
    hidden_size=1024,
    num_attention_heads=64,
    q_lora_rank=1024,
    kv_lora_rank=512,
    qk_head_dim=448,
    qk_pos_emb_head_dim=64,
    v_head_dim=256,
    ernie_vha_rank=16,
    dsa_indexer_n_heads=64,
    dsa_indexer_head_dim=128,
    dsa_indexer_topk=2048,
    csa_window_size=128,
)
# A fused-path-compatible toy model: FlashMLA and cuDNN fix the latent and indexer widths only.
_TOY_GEOMETRY = dict(
    hidden_size=512,
    ffn_hidden_size=1024,
    num_attention_heads=64,
    q_lora_rank=256,
    kv_lora_rank=512,
    qk_head_dim=64,
    qk_pos_emb_head_dim=64,
    v_head_dim=32,
    ernie_vha_rank=4,
    dsa_indexer_n_heads=32,
    dsa_indexer_head_dim=128,
    dsa_indexer_topk=64,
    csa_window_size=16,
)
_TOY_PATTERN = "D-D-"
_TOY_D_LAYERS = (0, 2)
_TOY_VOCAB = 512
_TOY_DOC_LENS = (200, 150, 100, 62)
_TOY_ITERATIONS = 40
_TOY_BUCKET_SIZE = 1_000_000

_PINNED_PARAMETER_NAMES = (
    "core_attention.attn_sink",
    "core_attention.indexer.k_norm.bias",
    "core_attention.indexer.k_norm.weight",
    "core_attention.indexer.linear_weights_proj.weight",
    "core_attention.indexer.linear_wk.weight",
    "core_attention.indexer.linear_wq_b.weight",
    "core_attention.linear_output_gate.weight",
    "core_attention.vha_postmix_u",
    "core_attention.vha_postmix_v",
    "k_b_proj",
    "kv_layernorm.weight",
    "linear_kv_down_proj.weight",
    "linear_proj.weight",
    "linear_q_down_proj.weight",
    "linear_q_up_proj.weight",
    "q_layernorm.weight",
    "v_b_proj",
)
_FP32_PARAMETERS = (
    "core_attention.attn_sink",
    "core_attention.vha_postmix_u",
    "core_attention.vha_postmix_v",
    "core_attention.indexer.k_norm.weight",
    "core_attention.indexer.k_norm.bias",
)
# Parameters the ERNIE core adds on top of Megatron-Core (relative to ``core_attention``).
_ERNIE_CORE_PARAMETERS = ("attn_sink", "vha_postmix_u", "vha_postmix_v", "linear_output_gate.weight")

# Fused vs reference tolerances. The input-side gradient bound follows Megatron-Core's own fused
# sparse DSA parity test, which measured slightly lower similarity there than for parameters.
# Cosine is scale-invariant, so gradients also get a relative L2 bound (cosine 0.999 alone allows
# a relative error of about 0.045; the bounds leave a little room on top of that, not a scale bug).
_FORWARD_MIN_COSINE = 1 - 5e-5
_FORWARD_MAX_NORMALIZED_ERROR = 2**-6
_PARAMETER_GRAD_MIN_COSINE = 0.999
_PARAMETER_GRAD_MAX_REL_ERR = 0.05
_ACTIVATION_GRAD_MIN_COSINE = 0.997
_ACTIVATION_GRAD_MAX_REL_ERR = 0.08
_KL_REL_TOL = 1e-3
_TOPK_TIE_REL_TOL = 1e-5
_TOPK_MAX_MISMATCHED_ROW_FRACTION = 1e-3
# Attention-logit scale factor for the parity tests (see _sharpen_attention_logits).
_ATTENTION_SHARPENING = 8.0

_CUDNN_DSA_WRAPPERS = (
    "indexer_forward_wrapper",
    "indexer_top_k_wrapper",
    "indexer_backward_wrapper",
    "sparse_attn_score_recompute_wrapper",
    "sparse_attention_backward_wrapper",
)
_MIN_CUDNN_FRONTEND = (1, 26)

_LARGE = pytest.mark.skipif(
    os.environ.get("ERNIE55_GPU_LARGE") != "1", reason="set ERNIE55_GPU_LARGE=1 (needs about 100 GB of device memory)"
)


# ---------------------------------------------------------------------------------------------
# Environment probes and fixtures
# ---------------------------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = "".join(itertools.takewhile(str.isdigit, piece))
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _cudnn_frontend_version() -> tuple[int, ...] | None:
    try:
        import cudnn
    except ImportError:
        return None
    version = getattr(cudnn, "__version__", None)
    if version is None:
        try:
            from importlib.metadata import version as distribution_version

            version = distribution_version("nvidia-cudnn-frontend")
        except Exception:
            return None
    return _version_tuple(version) or None


@functools.lru_cache(maxsize=None)
def _missing_reference_dependencies() -> tuple[str, ...]:
    missing = []
    if not torch.cuda.is_available():
        missing.append("a CUDA device")
    for module_name, label in (
        ("transformer_engine.pytorch", "Transformer Engine"),
        ("fast_hadamard_transform", "fast_hadamard_transform"),
    ):
        try:
            importlib.import_module(module_name)
        except Exception as error:
            missing.append(f"{label} ({type(error).__name__}: {error})")
    return tuple(missing)


def _flash_mla_launch_error() -> str | None:
    """Launch FlashMLA once: an import can succeed on a GPU the cubins were not built for (sm_103).

    The call uses the argument format the D layer's fused attention passes to FlashMLA: d_qk = 576, a
    finite sink and an index list with per-row lengths (here 512 wide, one valid key per row).
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

    generator = torch.Generator(device=_device()).manual_seed(0)
    query = torch.randn(4, 64, 576, device=_device(), generator=generator).to(torch.bfloat16)
    kv = torch.randn(8, 576, device=_device(), generator=generator).to(torch.bfloat16)
    indices = torch.full((4, 512), -1, dtype=torch.int32, device=_device())
    indices[:, 0] = 0
    topk_length = torch.ones(4, dtype=torch.int32, device=_device())
    try:
        fused_sparse_attention._csa_fwd_flash_mla(
            query,
            kv,
            indices,
            512**-0.5,
            d_v=512,
            attn_sink=torch.zeros(64, device=_device()),
            topk_length=topk_length,
        )
        torch.cuda.synchronize()
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


@functools.lru_cache(maxsize=None)
def _missing_fused_dependencies() -> tuple[str, ...]:
    missing = list(_missing_reference_dependencies())
    if missing:
        return tuple(missing)
    major, minor = torch.cuda.get_device_capability()
    if major < 9:
        missing.append(f"an SM90+ GPU (found sm_{major}{minor})")
    try:
        from flash_mla import flash_mla_sparse_fwd  # noqa: F401
    except Exception as error:
        missing.append(f"FlashMLA flash_mla_sparse_fwd ({type(error).__name__}: {error})")
    try:
        from cudnn import DSA
    except Exception as error:
        missing.append(f"cudnn-frontend DSA ({type(error).__name__}: {error})")
    else:
        absent = [name for name in _CUDNN_DSA_WRAPPERS if not hasattr(DSA, name)]
        if absent:
            missing.append(f"cudnn.DSA wrappers {', '.join(absent)}")
    version = _cudnn_frontend_version()
    if version is not None and version < _MIN_CUDNN_FRONTEND:
        found = ".".join(map(str, version))
        missing.append(f"nvidia-cudnn-frontend >= {'.'.join(map(str, _MIN_CUDNN_FRONTEND))} (found {found})")
    if not missing:
        error = _flash_mla_launch_error()
        if error is not None:
            missing.append(f"a FlashMLA kernel for this GPU ({error})")
    return tuple(missing)


def _deterministic_backward_unavailable_reason() -> str | None:
    """Why this build cannot run the D layer's sparse-attention backward deterministically, or None.

    Runs Megatron-Core dev's ``_deterministic_sparse_bwd_kwargs`` (frontend version, SM100 family, head
    count, and cuDNN's ``check_support`` for the problem) on a d_qk = 576 sample in the exact format
    of the D layer's default fused backward.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

    helper = getattr(fused_sparse_attention, "_deterministic_sparse_bwd_kwargs", None)
    if helper is None:
        return "this Megatron-Core build has no deterministic sparse-attention backward"
    rows, heads = 256, 64
    generator = torch.Generator(device=_device()).manual_seed(0)
    query = torch.randn(rows, heads, 576, device=_device(), generator=generator).to(torch.bfloat16)
    kv = torch.randn(rows, 576, device=_device(), generator=generator).to(torch.bfloat16)
    out = torch.randn(rows, heads, 512, device=_device(), generator=generator).to(torch.bfloat16)
    lse = torch.randn(rows, heads, device=_device(), generator=generator)
    sink = torch.full((heads,), float("-inf"), device=_device())
    indices = torch.arange(rows, dtype=torch.int32, device=_device()).view(-1, 1).expand(rows, 2176).contiguous()
    topk_length = torch.ones(rows, dtype=torch.int32, device=_device())
    try:
        helper(query, kv, out, lse, sink, indices, topk_length, 512**-0.5, requested=True)
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


@pytest.fixture(scope="module")
def distributed(tmp_path_factory):
    """Single-rank process group and Megatron-Core model-parallel state (TP = PP = CP = 1)."""
    missing = _missing_reference_dependencies()
    if missing:
        pytest.skip("the ERNIE 5.5 D-layer GPU tests need " + "; ".join(missing))
    import torch.distributed as dist
    from megatron.core import parallel_state

    created_group = False
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            pytest.skip("the ERNIE 5.5 D-layer GPU tests run on a single rank")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        store = tmp_path_factory.mktemp("ernie55_gpu_dist") / "store"
        dist.init_process_group(backend="nccl", init_method=f"file://{store}", rank=0, world_size=1)
        created_group = True
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )
    yield
    parallel_state.destroy_model_parallel()
    if created_group:
        dist.destroy_process_group()


@pytest.fixture
def fused_kernels(distributed):
    """Skip unless FlashMLA and the cuDNN DSA kernels can run on this GPU."""
    missing = _missing_fused_dependencies()
    if missing:
        pytest.skip("the fused ERNIE 5.5 D-layer path needs " + "; ".join(missing))


@pytest.fixture(autouse=True)
def _fresh_dsa_state():
    """Reset the process-wide DSA indexer-loss tracker and loss scale around every test."""
    from megatron.core.transformer.experimental_attention_variant import dsa

    def reset() -> None:
        dsa.DSAIndexerLossLoggingHelper.tracker.clear()
        dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale = None

    reset()
    yield
    reset()


# ---------------------------------------------------------------------------------------------
# Builders and inputs
# ---------------------------------------------------------------------------------------------


def _make_config(backend: str, *, geometry: dict | None = None, num_layers: int = 1, **overrides):
    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig

    kwargs = {**_COMMON_CONFIG, **(_PRODUCTION_GEOMETRY if geometry is None else geometry), **overrides}
    config = Ernie55TransformerConfig(num_layers=num_layers, dsa_kernel_backend=backend, **kwargs)
    config.finalize()
    return config


def _attention_spec():
    from megatron.bridge.models.ernie.modeling_ernie55 import ernie55_hybrid_stack_spec

    return ernie55_hybrid_stack_spec().submodules.dsa_layer.submodules.self_attention


def _build_attention(config, *, seed: int = 1234):
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    attention = build_module(_attention_spec(), config=config, layer_number=1)
    return attention.to(_device()).train()


def _randomize_ernie_params(attention, *, seed: int, sink_offset: float = 1.5) -> None:
    """Move the sink and both VHA factors away from their (identity) init; the sink is bf16-representable."""
    core = attention.core_attention
    generator = torch.Generator(device=core.attn_sink.device).manual_seed(seed + 7)

    def randn(param: torch.Tensor) -> torch.Tensor:
        return torch.randn(param.shape, generator=generator, device=param.device)

    with torch.no_grad():
        core.attn_sink.copy_((sink_offset + randn(core.attn_sink)).to(torch.bfloat16).float())
        core.vha_postmix_u.copy_(0.1 * randn(core.vha_postmix_u))
        core.vha_postmix_v.copy_(0.1 * randn(core.vha_postmix_v))


def _sharpen_attention_logits(attention, factor: float = _ATTENTION_SHARPENING) -> None:
    """Scale the q up-projection: random-init logits have a std of about 0.3, which makes the KL teacher
    almost uniform and the KL so small that its relative error is dominated by rounding."""
    with torch.no_grad():
        attention.linear_q_up_proj.weight.mul_(factor)


def _twin_attentions(*, seed: int = 1234, **overrides):
    """A fused and a reference D attention with identical weights (sink, VHA and attention sharpness changed)."""
    fused_config = _make_config("cudnn", **overrides)
    fused = _build_attention(fused_config, seed=seed)
    _randomize_ernie_params(fused, seed=seed)
    _sharpen_attention_logits(fused)
    reference = _build_attention(_make_config("none", **overrides), seed=seed)
    reference.load_state_dict(fused.state_dict())
    assert fused.core_attention.use_fused and not reference.core_attention.use_fused
    return fused, reference, fused_config


def _packed_seq_params(doc_lens):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = torch.tensor([0, *itertools.accumulate(doc_lens)], dtype=torch.int32, device=_device())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )


def _attention_inputs(config, *, doc_lens=None, tail_rows: int = 0, sbhd=None, seed: int = 0):
    """``(hidden, grad_output, packed_seq_params)``: a THD pack (b = 1) or an SBHD batch ``(s, b)``."""
    if sbhd is not None:
        shape, packed_seq_params = (*sbhd, config.hidden_size), None
    else:
        shape, packed_seq_params = (sum(doc_lens) + tail_rows, 1, config.hidden_size), _packed_seq_params(doc_lens)
    generator = torch.Generator(device=_device()).manual_seed(seed)
    hidden = torch.randn(shape, device=_device(), generator=generator).to(torch.bfloat16)
    grad_output = torch.randn(shape, device=_device(), generator=generator).to(torch.bfloat16)
    return hidden, grad_output, packed_seq_params


def _selection_bounds(packed_seq_params, seqlen: int, window_size: int):
    """True ERNIE candidate bounds ``[starts, ends)``, int32 ``[s]``."""
    from megatron.bridge.models.ernie.modeling_ernie55 import ops

    doc_start = ops.ernie55_document_starts(packed_seq_params, seqlen, _device())
    return ops.ernie55_selection_bounds(doc_start, window_size)


@dataclasses.dataclass
class _AttentionRun:
    output: torch.Tensor
    grads: dict[str, torch.Tensor]
    topk: torch.Tensor
    loss: torch.Tensor | None
    indexer: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    starts: torch.Tensor
    ends: torch.Tensor


def _run_attention(attention, hidden, grad_output, packed_seq_params) -> _AttentionRun:
    """One forward and backward; records the core inputs' gradients, the top-k (masked to the true bounds) and the KL."""
    from megatron.bridge.models.ernie.modeling_ernie55 import ops

    core = attention.core_attention
    captured = {}
    select_topk = core._select_topk
    forward_before_topk = core.indexer.forward_before_topk

    def record_select_topk(*args, **kwargs):
        topk, loss = select_topk(*args, **kwargs)
        captured["topk"], captured["loss"] = topk, loss
        return topk, loss

    def record_indexer(*args, **kwargs):
        outputs = forward_before_topk(*args, **kwargs)
        captured["indexer"] = tuple(tensor.detach().clone() for tensor in outputs)
        return outputs

    def record_core_inputs(module, args, kwargs):
        for name, tensor in zip(("core.query", "core.key"), args[:2]):
            tensor.retain_grad()
            captured[name] = tensor

    core._select_topk = record_select_topk
    core.indexer.forward_before_topk = record_indexer
    handle = core.register_forward_pre_hook(record_core_inputs, with_kwargs=True)
    try:
        attention.zero_grad(set_to_none=True)
        hidden = hidden.detach().clone().requires_grad_(True)
        output, _ = attention(hidden, attention_mask=None, packed_seq_params=packed_seq_params)
        output.backward(grad_output)
    finally:
        handle.remove()
        del core._select_topk
        del core.indexer.forward_before_topk

    grads = {"input": hidden.grad.detach().float()}
    for name in ("core.query", "core.key"):
        grads[name] = captured[name].grad.detach().float()
    for name, param in attention.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().float().clone()
    starts, ends = _selection_bounds(packed_seq_params, captured["topk"].size(1), core.window_size)
    loss = captured["loss"]
    return _AttentionRun(
        output=output.detach().float(),
        grads=grads,
        topk=ops.mask_topk_beyond_end(captured["topk"], ends=ends),
        loss=None if loss is None else loss.detach().float(),
        indexer=captured["indexer"],
        starts=starts,
        ends=ends,
    )


def _record_kernel_bounds(monkeypatch) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Record the ``[starts, ends)`` every fused top-k hook call receives."""
    from megatron.core.transformer.experimental_attention_variant import dsa_kernels

    seen = []
    for hook_name in ("run_fused_qk_topk", "run_fused_qk_topk_with_loss"):
        original = getattr(dsa_kernels, hook_name)

        def recording(
            config, q, k, weights, index_topk, starts, ends, *args, _original=original, _name=hook_name, **kw
        ):
            seen.append((_name, starts.detach().clone(), ends.detach().clone()))
            return _original(config, q, k, weights, index_topk, starts, ends, *args, **kw)

        monkeypatch.setattr(dsa_kernels, hook_name, recording)
    return seen


# ---------------------------------------------------------------------------------------------
# Metrics and assertions
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


def _report(label: str, **metrics) -> None:
    """Log one line of measured metrics (shown with ``-o log_cli=true -o log_cli_level=INFO``)."""
    text = ", ".join(
        f"{name}={value:.6g}" if isinstance(value, float) else f"{name}={value}" for name, value in metrics.items()
    )
    logger.info("[ernie55-gpu] %s: %s", label, text)


def _reference_convention_weights(q: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, float]:
    """The core's score convention: ``(w * d^0.5, d^-0.5)``, the weights rounded in their dtype, the scale for fp32."""
    head_dim = q.size(-1)
    return weights * head_dim**0.5, head_dim**-0.5


def _index_scores(indexer, starts: torch.Tensor, ends: torch.Tensor, index_topk: int) -> torch.Tensor:
    """Reference fp32 indexer scores ``[b, s, s]`` in the core's convention, ``-inf`` outside ``[starts, ends)``."""
    from megatron.core.transformer.experimental_attention_variant import dsa

    q, k, weights = indexer
    weights, scale = _reference_convention_weights(q, weights)
    with torch.no_grad():
        scores, _ = dsa.fused_qk_topk_naive(q, k, weights, index_topk, varlen_starts=starts, varlen_ends=ends)
    return scores * scale


def _assert_topk_tie_aware(
    got: torch.Tensor, expected: torch.Tensor, scores: torch.Tensor, *, label: str
) -> torch.Tensor:
    """Assert two top-k selections agree up to ties; return the ``[b, s]`` mask of identical rows.

    ``got`` and ``expected`` are ``[b, s, K]`` with ``-1`` for an empty slot, ``scores`` is
    ``[b, s, n]`` with ``-inf`` outside the candidates. A row may differ only by keys whose score
    lies within ``_TOPK_TIE_REL_TOL`` (relative to the row's largest score) of the K-th score.
    """
    same = (got.long().sort(dim=-1).values == expected.long().sort(dim=-1).values).all(dim=-1)
    mismatched = (~same).nonzero().tolist()
    problems = []
    for batch, row in mismatched:
        got_keys = set(got[batch, row][got[batch, row] >= 0].tolist())
        expected_keys = set(expected[batch, row][expected[batch, row] >= 0].tolist())
        if len(got_keys) != len(expected_keys):
            problems.append(f"(b={batch}, t={row}) selects {len(got_keys)} keys, expected {len(expected_keys)}")
            continue
        row_scores = scores[batch, row].double()
        finite = row_scores[torch.isfinite(row_scores)]
        boundary = row_scores[sorted(expected_keys)].min().item()
        tolerance = _TOPK_TIE_REL_TOL * finite.abs().max().item()
        far = [key for key in got_keys ^ expected_keys if abs(row_scores[key].item() - boundary) > tolerance]
        if far:
            problems.append(f"(b={batch}, t={row}) differs by non-tied keys {sorted(far)[:8]}")
    _report(f"{label} top-k", rows=same.numel(), mismatched_rows=len(mismatched))
    assert not problems, f"{label}: top-k disagreement beyond ties: " + "; ".join(problems[:10])
    assert len(mismatched) <= max(1, int(_TOPK_MAX_MISMATCHED_ROW_FRACTION * same.numel())), (
        f"{label}: {len(mismatched)} of {same.numel()} rows select different (tied) keys"
    )
    return same


def _assert_grad_close(
    name: str, actual: torch.Tensor, expected: torch.Tensor, *, min_cosine: float, max_rel_err: float
) -> str | None:
    """Return a failure message, or None. An all-zero expected gradient needs an (almost) zero actual one."""
    if not torch.isfinite(actual).all():
        return f"{name}: non-finite gradient"
    if expected.abs().max() == 0:
        worst = actual.abs().max().item()
        return None if worst <= 1e-7 else f"{name}: expected a zero gradient, got max |g| = {worst:.3e}"
    cosine, rel_err = _cosine(actual, expected), _rel_err(actual, expected)
    if cosine < min_cosine or rel_err > max_rel_err:
        return f"{name}: cosine {cosine:.6f} (min {min_cosine}), relative error {rel_err:.3e} (max {max_rel_err})"
    return None


def _assert_runs_agree(fused: _AttentionRun, reference: _AttentionRun, *, index_topk: int, label: str) -> None:
    """Fused vs reference: tie-aware top-k sets, forward, indexer KL and every gradient."""
    scores = _index_scores(reference.indexer, reference.starts, reference.ends, index_topk)
    same_rows = _assert_topk_tie_aware(fused.topk, reference.topk, scores, label=label)

    cosine = _cosine(fused.output, reference.output)
    rows = same_rows.transpose(0, 1)  # outputs are [s, b, h]
    max_error = (fused.output - reference.output)[rows].abs().max().item() / reference.output.abs().max().item()
    kl_fused = None if fused.loss is None else fused.loss.item()
    kl_reference = None if reference.loss is None else reference.loss.item()

    failures, cosines, rel_errs = [], {}, {}
    for name, expected in reference.grads.items():
        activation = name in ("input", "core.query", "core.key")
        if name not in fused.grads:
            failures.append(f"{name}: no fused gradient")
            continue
        cosines[name], rel_errs[name] = _cosine(fused.grads[name], expected), _rel_err(fused.grads[name], expected)
        failure = _assert_grad_close(
            name,
            fused.grads[name],
            expected,
            min_cosine=_ACTIVATION_GRAD_MIN_COSINE if activation else _PARAMETER_GRAD_MIN_COSINE,
            max_rel_err=_ACTIVATION_GRAD_MAX_REL_ERR if activation else _PARAMETER_GRAD_MAX_REL_ERR,
        )
        if failure is not None:
            failures.append(failure)
    _report(
        label,
        forward_cosine=cosine,
        forward_max_error=max_error,
        kl_fused=kl_fused,
        kl_reference=kl_reference,
        min_grad_cosine=min(cosines.values()),
        worst_cosine_grad=min(cosines, key=cosines.get),
        max_grad_rel_err=max(rel_errs.values()),
        worst_rel_err_grad=max(rel_errs, key=rel_errs.get),
    )
    assert torch.isfinite(fused.output).all(), f"{label}: non-finite fused output"
    assert cosine >= _FORWARD_MIN_COSINE, f"{label}: forward cosine {cosine:.7f}"
    assert max_error <= _FORWARD_MAX_NORMALIZED_ERROR, f"{label}: normalized max forward error {max_error:.3e}"
    if kl_reference is not None:
        assert kl_fused is not None, f"{label}: the fused path computed no indexer loss"
        assert abs(kl_fused - kl_reference) <= _KL_REL_TOL * abs(kl_reference) + 1e-9, (
            f"{label}: indexer KL {kl_fused:.8g} vs reference {kl_reference:.8g}"
        )
    assert not failures, f"{label}: gradient mismatch: " + "; ".join(failures)


def _assert_kernel_rows_non_empty(seen) -> None:
    """Every fused hook call received at least one candidate per row."""
    assert seen, "no fused top-k hook was called"
    for hook_name, starts, ends in seen:
        assert bool((ends > starts).all()), f"{hook_name} received an empty candidate row"


# ---------------------------------------------------------------------------------------------
# Module structure
# ---------------------------------------------------------------------------------------------


def test_parameter_names_and_dtypes_with_te_spec(distributed):
    """Pin the parameter names the checkpoint mappings rely on, for the Transformer Engine spec."""
    config = _make_config("none", geometry=_TOY_GEOMETRY)
    attention = _build_attention(config)
    params = dict(attention.named_parameters())
    assert sorted(params) == sorted(_PINNED_PARAMETER_NAMES)
    for name, param in params.items():
        assert param.dtype == (torch.float32 if name in _FP32_PARAMETERS else torch.bfloat16), name
    heads, rank = config.num_attention_heads, config.ernie_vha_rank
    assert params["core_attention.attn_sink"].shape == (heads,)
    assert params["core_attention.vha_postmix_u"].shape == (heads, rank)
    assert params["core_attention.vha_postmix_v"].shape == (heads, rank)
    assert params["core_attention.linear_output_gate.weight"].shape == (heads * config.v_head_dim, config.q_lora_rank)
    # The reference implementation's K/V up-projection layout (Megatron-Core's TE projection is replaced).
    assert params["k_b_proj"].shape == (heads * config.kv_lora_rank, config.qk_head_dim)
    assert params["v_b_proj"].shape == (heads * config.v_head_dim, config.kv_lora_rank)
    core = attention.core_attention
    assert not torch.any(core.attn_sink) and not torch.any(core.vha_postmix_v) and torch.any(core.vha_postmix_u)


# ---------------------------------------------------------------------------------------------
# Fused path vs reference path
# ---------------------------------------------------------------------------------------------

_PARITY_CASES = {
    # Documents of 2500 tokens reach the selection regime (pos > W + K - 1 = 2175).
    "thd_production": dict(overrides={}, inputs=dict(doc_lens=(2500, 900, 500, 196))),
    # SBHD, b = 2: flat s * b + b indices and per-batch offsets, with a smaller K so most rows select.
    "sbhd_b2_selection": dict(overrides=dict(dsa_indexer_topk=256, csa_window_size=64), inputs=dict(sbhd=(1536, 2))),
    # SBHD, b = 3, per-token loss: the per-sample loss-hook calls are summed instead of averaged.
    "sbhd_b3_per_token": dict(
        overrides=dict(dsa_indexer_topk=256, csa_window_size=64, calculate_per_token_loss=True),
        inputs=dict(sbhd=(1024, 3)),
    ),
    "thd_8k_single_doc": dict(overrides={}, inputs=dict(doc_lens=(8192,))),
}


@pytest.mark.parametrize(
    "case",
    ["thd_production", "sbhd_b2_selection", "sbhd_b3_per_token", pytest.param("thd_8k_single_doc", marks=_LARGE)],
)
def test_fused_matches_reference(fused_kernels, case):
    spec = _PARITY_CASES[case]
    fused_attention, reference_attention, config = _twin_attentions(**spec["overrides"])
    hidden, grad_output, packed_seq_params = _attention_inputs(config, **spec["inputs"])
    fused = _run_attention(fused_attention, hidden, grad_output, packed_seq_params)
    del fused_attention
    reference = _run_attention(reference_attention, hidden, grad_output, packed_seq_params)
    _assert_runs_agree(fused, reference, index_topk=config.dsa_indexer_topk, label=case)


def test_edge_rows_match_reference(fused_kernels, monkeypatch):
    """1-token documents, documents of exactly W and W + K tokens, and tail rows past ``cu_seqlens[-1]``."""
    fused_attention, reference_attention, config = _twin_attentions()
    window, topk = config.csa_window_size, config.dsa_indexer_topk
    seen = _record_kernel_bounds(monkeypatch)
    hidden, grad_output, packed_seq_params = _attention_inputs(
        config, doc_lens=(1, window, window + topk, 1, 300), tail_rows=64
    )
    fused = _run_attention(fused_attention, hidden, grad_output, packed_seq_params)
    _assert_kernel_rows_non_empty(seen)
    reference = _run_attention(reference_attention, hidden, grad_output, packed_seq_params)
    _assert_runs_agree(fused, reference, index_topk=topk, label="edge_rows")


def test_rows_without_remote_candidates(fused_kernels, monkeypatch):
    """Every row is within W of its document start: nothing is selected, the KL is exactly zero."""
    fused_attention, reference_attention, config = _twin_attentions()
    seen = _record_kernel_bounds(monkeypatch)
    hidden, grad_output, packed_seq_params = _attention_inputs(config, doc_lens=(1, 64, 128, 128, 100))
    fused = _run_attention(fused_attention, hidden, grad_output, packed_seq_params)
    _assert_kernel_rows_non_empty(seen)
    assert bool((fused.topk == -1).all())
    assert fused.loss is not None and abs(fused.loss.item()) <= 1e-9, fused.loss
    for name, grad in fused.grads.items():
        assert torch.isfinite(grad).all(), name
        if ".indexer." in name:
            assert grad.abs().max().item() <= 1e-7, f"{name} received an indexer gradient from empty rows"
    reference = _run_attention(reference_attention, hidden, grad_output, packed_seq_params)
    _assert_runs_agree(fused, reference, index_topk=config.dsa_indexer_topk, label="no_remote_candidates")


def _random_indexer_inputs(config, num_rows: int, *, seed: int):
    """Indexer ``(q [T, 1, H, D], k [T, 1, D], weights [T, 1, H])`` in bf16, weights pre-scaled like DSAIndexer's."""
    generator = torch.Generator(device=_device()).manual_seed(seed)
    heads, dim = config.dsa_indexer_n_heads, config.dsa_indexer_head_dim

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(shape, device=_device(), generator=generator) * scale).to(torch.bfloat16)

    return (
        randn(num_rows, 1, heads, dim),
        randn(num_rows, 1, dim),
        randn(num_rows, 1, heads, scale=(heads * dim) ** -0.5),
    )


def _sampled_topk(indexer, rows: torch.Tensor, starts, ends, index_topk: int):
    """Reference top-k and fp32 scores ``[n, T]`` (the core's convention) for sampled query rows (O(n * T) memory)."""
    q, k, weights = indexer
    weights, scale = _reference_convention_weights(q, weights)
    keys = k[:, 0].float()
    positions = torch.arange(keys.size(0), device=keys.device)
    score_chunks = []
    for chunk in rows.split(16):
        scores = torch.einsum("nhd,td->nht", q[chunk, 0].float(), keys).relu_()
        scores = (scores * weights[chunk, 0].float().unsqueeze(-1)).sum(dim=1) * scale
        outside = (positions < starts[chunk].unsqueeze(1)) | (positions >= ends[chunk].unsqueeze(1))
        score_chunks.append(scores.masked_fill(outside, float("-inf")))
    scores = torch.cat(score_chunks)
    values, indices = scores.topk(min(index_topk, scores.size(1)), dim=-1)
    return indices.masked_fill(torch.isinf(values), -1), scores


def test_long_pack_topk_beyond_32k_rows(fused_kernels):
    """A pack longer than 32768 rows, as the second fused top-k call of the process, matches the reference."""
    from megatron.bridge.models.ernie.modeling_ernie55 import ops

    config = _make_config("cudnn")
    core = _build_attention(config).core_attention.eval()
    heads, width, index_topk = (
        config.num_attention_heads,
        config.kv_lora_rank + config.qk_pos_emb_head_dim,
        core.index_topk,
    )
    for call, doc_lens in enumerate(((4096,), (24576, 16384))):
        num_rows = sum(doc_lens)
        indexer = _random_indexer_inputs(config, num_rows, seed=call)
        starts, ends = _selection_bounds(_packed_seq_params(doc_lens), num_rows, core.window_size)
        kernel_ends = ops.ernie55_kernel_selection_ends(starts=starts, ends=ends)
        # The attention query and key only feed the indexer loss, which is off here.
        query = torch.zeros((), dtype=torch.bfloat16, device=_device()).expand(num_rows, 1, heads, width)
        key = torch.zeros((), dtype=torch.bfloat16, device=_device()).expand(num_rows, 1, 1, width)
        with torch.no_grad():
            topk, loss = core._select_topk(*indexer, query, key, starts, kernel_ends, False)
        assert loss is None
        topk = ops.mask_topk_beyond_end(topk, ends=ends)
        generator = torch.Generator().manual_seed(call)
        rows = torch.randperm(num_rows, generator=generator)[:64]
        if num_rows > 32768:
            rows = torch.cat([rows, 32768 + torch.randperm(num_rows - 32768, generator=generator)[:128]])
        rows = rows.unique().to(_device())
        expected, scores = _sampled_topk(indexer, rows, starts, ends, index_topk)
        _assert_topk_tie_aware(
            topk[:, rows], expected.unsqueeze(0), scores.unsqueeze(0), label=f"long_pack_call{call}_rows{num_rows}"
        )


# ---------------------------------------------------------------------------------------------
# Attention sink gradient
# ---------------------------------------------------------------------------------------------


def test_flash_mla_lse_excludes_sink(fused_kernels):
    """FlashMLA's sparse forward puts the sink in the denominator but returns an LSE without it.

    The analytic sink-gradient fallback (``lse_full = logaddexp(lse, sink)``) relies on this. The call
    uses the fallback's index format: compacted, 512 wide, with per-row lengths.
    """
    import torch.nn.functional as F
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

    rows, heads, dim, value_dim, width, scale = 256, 64, 576, 512, 256, 512**-0.5
    generator = torch.Generator(device=_device()).manual_seed(5)
    query = torch.randn(rows, heads, dim, device=_device(), generator=generator).to(torch.bfloat16)
    kv = torch.randn(rows, dim, device=_device(), generator=generator).to(torch.bfloat16)
    # Comparable to the LSE over ~180 keys (about 6), so the two conventions differ on every head.
    sink = 6.0 + 0.5 * torch.randn(heads, device=_device(), generator=generator)
    indices = torch.rand(rows, rows, device=_device(), generator=generator).argsort(dim=-1)[:, :width]
    holes = torch.rand(rows, width, device=_device(), generator=generator) < 0.3
    holes[:, 0] = False
    indices = indices.masked_fill(holes, -1).to(torch.int32)
    compact, topk_length = fused_sparse_attention.build_flat_topk_idxs(
        indices.unsqueeze(0), batch_size=1, compact=True
    )
    compact = F.pad(compact, (0, 512 - width), value=-1)

    out, lse, _ = fused_sparse_attention._csa_fwd_flash_mla(
        query, kv, compact.contiguous(), scale, d_v=value_dim, attn_sink=sink, topk_length=topk_length
    )
    keys = kv.float()[indices.long().clamp_min(0)]
    logits = torch.einsum("rhd,rkd->rhk", query.float(), keys) * scale
    logits = logits.masked_fill((indices < 0).unsqueeze(1), float("-inf"))
    lse_without_sink = torch.logsumexp(logits, dim=-1)
    lse_with_sink = torch.logaddexp(lse_without_sink, sink.view(1, -1))
    probs = torch.exp(logits - lse_with_sink.unsqueeze(-1))
    expected_out = torch.einsum("rhk,rkd->rhd", probs, keys[..., :value_dim])

    error_without_sink = (lse.float() - lse_without_sink).abs().max().item()
    error_with_sink = (lse.float() - lse_with_sink).abs().max().item()
    out_cosine = _cosine(out, expected_out)
    _report(
        "flash_mla_lse",
        max_error_vs_lse_without_sink=error_without_sink,
        max_error_vs_lse_with_sink=error_with_sink,
        out_cosine=out_cosine,
    )
    assert error_with_sink > 0.1, "the sink is too small to tell the two LSE conventions apart"
    assert error_without_sink <= 2e-3, "FlashMLA's LSE is not the sink-free LSE the sink-gradient fallback assumes"
    assert out_cosine >= _FORWARD_MIN_COSINE, "FlashMLA's output does not put the sink in the denominator"


def _core_inputs(config, doc_lens, *, seed: int):
    """Direct core inputs for a THD pack: absorbed query, latent key, q latent, layer input, V-up weight."""
    generator = torch.Generator(device=_device()).manual_seed(seed)
    num_rows, heads = sum(doc_lens), config.num_attention_heads
    width = config.kv_lora_rank + config.qk_pos_emb_head_dim

    def randn(*shape: int, scale: float = 1.0) -> torch.Tensor:
        return (torch.randn(shape, device=_device(), generator=generator) * scale).to(torch.bfloat16)

    inputs = dict(
        query=randn(num_rows, heads, width, scale=2.0),
        key=randn(num_rows, 1, width),
        x=randn(num_rows, 1, config.hidden_size),
        qr=randn(num_rows, config.q_lora_rank),
        up_v_weight=randn(heads, config.v_head_dim, config.kv_lora_rank, scale=config.kv_lora_rank**-0.5),
        grad=randn(num_rows, 1, heads * config.v_head_dim),
    )
    return inputs, _packed_seq_params(doc_lens)


def _run_core(core, inputs, packed_seq_params):
    leaves = {name: inputs[name].detach().clone().requires_grad_(True) for name in ("query", "key", "up_v_weight")}
    core.zero_grad(set_to_none=True)
    output = core(
        leaves["query"],
        leaves["key"],
        None,
        None,
        inputs["x"],
        inputs["qr"],
        packed_seq_params=packed_seq_params,
        up_v_weight=leaves["up_v_weight"],
    )
    output.backward(inputs["grad"])
    grads = {name: leaf.grad.detach().float() for name, leaf in leaves.items()}
    grads.update({name: p.grad.detach().float().clone() for name, p in core.named_parameters() if p.grad is not None})
    return output.detach().float(), grads


# Gradients compared by the sink test: report label -> name in the core's gradient dict.
_SINK_TEST_GRADS = {
    "d_sink": "attn_sink",
    "d_query": "query",
    "d_key": "key",
    "d_up_v": "up_v_weight",
    "d_vha_u": "vha_postmix_u",
    "d_vha_v": "vha_postmix_v",
    "d_gate": "linear_output_gate.weight",
}


@pytest.mark.parametrize("sink_offset", [-6.0, 0.0, 4.0], ids=["sink_negligible", "sink_comparable", "sink_dominant"])
def test_sink_gradient(fused_kernels, sink_offset):
    """Native cuDNN ``d_sink`` (the default) and the analytic fallback against autograd, finite sink.

    The pack stays in the dense regime (every document has at most W + K tokens), so all three
    variants attend over the same keys. Both fused variants must match the reference, and native
    must match the fallback.
    """
    config = _make_config("cudnn")
    attention = _build_attention(config)
    _randomize_ernie_params(attention, seed=21, sink_offset=sink_offset)
    core = attention.core_attention.eval()  # isolate the attention gradients from the indexer loss
    inputs, packed_seq_params = _core_inputs(config, (2176, 1200, 720), seed=22)

    def compare(out, grads, expected_out, expected_grads) -> dict:
        metrics = dict(forward_cosine=_cosine(out, expected_out))
        for label, name in _SINK_TEST_GRADS.items():
            metrics[f"{label}_cosine"] = _cosine(grads[name], expected_grads[name])
            metrics[f"{label}_rel_err"] = _rel_err(grads[name], expected_grads[name])
        return metrics

    core.use_fused = False
    reference_out, reference = _run_core(core, inputs, packed_seq_params)
    results = {}
    for variant, native in (("fallback", False), ("native", True)):
        core.use_fused, core.native_sink_grad = True, native
        out, grads = _run_core(core, inputs, packed_seq_params)
        metrics = compare(out, grads, reference_out, reference)
        _report(f"sink_gradient[{sink_offset}] {variant}", **metrics)
        results[variant] = (out, grads, metrics)

    def failures(metrics, grads) -> list[str]:
        problems = [f"non-finite {name}" for name, grad in grads.items() if not torch.isfinite(grad).all()]
        if metrics["forward_cosine"] < _FORWARD_MIN_COSINE:
            problems.append(f"forward cosine {metrics['forward_cosine']:.7f}")
        for label in _SINK_TEST_GRADS:
            # d_sink sums one bf16-rounded product per row and head: it gets a tighter relative bound.
            max_rel_err = 2e-2 if label == "d_sink" else _PARAMETER_GRAD_MAX_REL_ERR
            cosine, rel_err = metrics[f"{label}_cosine"], metrics[f"{label}_rel_err"]
            if cosine < _PARAMETER_GRAD_MIN_COSINE or rel_err > max_rel_err:
                problems.append(f"{label} cosine {cosine:.6f}, relative error {rel_err:.3e}")
        return problems

    fallback_out, fallback_grads, fallback_metrics = results["fallback"]
    fallback_problems = failures(fallback_metrics, fallback_grads)
    assert not fallback_problems, "analytic sink-gradient fallback: " + "; ".join(fallback_problems)
    native_out, native_grads, native_metrics = results["native"]
    agreement = compare(native_out, native_grads, fallback_out, fallback_grads)
    _report(f"sink_gradient[{sink_offset}] native_vs_fallback", **agreement)
    native_problems = failures(native_metrics, native_grads)
    native_problems += [f"vs fallback: {problem}" for problem in failures(agreement, {})]
    _report(f"sink_gradient[{sink_offset}] C2 gate", gate_open=not native_problems)
    assert not native_problems, "native cuDNN sink gradient (the default): " + "; ".join(native_problems)


# ---------------------------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------------------------


def test_fused_determinism(fused_kernels):
    """Forward, top-k and KL are bitwise reproducible; so are non-indexer gradients with a deterministic backward."""
    unavailable = _deterministic_backward_unavailable_reason()
    config = _make_config("cudnn", deterministic_mode=unavailable is None)
    attention = _build_attention(config)
    _randomize_ernie_params(attention, seed=31)
    _sharpen_attention_logits(attention)
    hidden, grad_output, packed_seq_params = _attention_inputs(config, doc_lens=(2500, 1596), seed=32)
    first = _run_attention(attention, hidden, grad_output, packed_seq_params)
    second = _run_attention(attention, hidden, grad_output, packed_seq_params)

    spread = {name: (first.grads[name] - second.grads[name]).abs().max().item() for name in first.grads}
    unequal = sorted(name for name in first.grads if not torch.equal(first.grads[name], second.grads[name]))
    _report(
        "determinism",
        deterministic_backward=unavailable is None,
        reason=unavailable or "-",
        unequal_grads=len(unequal),
        max_indexer_grad_spread=max([v for n, v in spread.items() if ".indexer." in n], default=0.0),
        max_other_grad_spread=max([v for n, v in spread.items() if ".indexer." not in n], default=0.0),
    )
    assert torch.equal(first.output, second.output)
    assert torch.equal(first.topk, second.topk)
    assert first.loss is not None and torch.equal(first.loss, second.loss)
    if unavailable is None:
        # The cuDNN indexer backward accumulates dK with atomics; only report its spread.
        assert not [name for name in unequal if ".indexer." not in name], unequal


# ---------------------------------------------------------------------------------------------
# Toy HybridModel through the Megatron-Core schedule
# ---------------------------------------------------------------------------------------------


def _toy_config(backend: str, **overrides):
    return _make_config(backend, geometry=_TOY_GEOMETRY, num_layers=len(_TOY_PATTERN), **overrides)


def _toy_batch(*, seed: int = 0) -> dict:
    """A fixed THD micro-batch: next-token targets inside each document, none across documents."""
    generator = torch.Generator().manual_seed(seed)
    num_tokens = sum(_TOY_DOC_LENS)
    tokens = torch.randint(0, _TOY_VOCAB, (1, num_tokens), generator=generator)
    cu_seqlens = torch.tensor([0, *itertools.accumulate(_TOY_DOC_LENS)])
    loss_mask = torch.ones(1, num_tokens)
    loss_mask[0, cu_seqlens[1:] - 1] = 0.0
    position_ids = torch.cat([torch.arange(length) for length in _TOY_DOC_LENS]).unsqueeze(0)
    batch = dict(tokens=tokens, labels=tokens.roll(-1, dims=1), loss_mask=loss_mask, position_ids=position_ids)
    batch = {name: tensor.to(_device()) for name, tensor in batch.items()}
    batch["packed_seq_params"] = _packed_seq_params(_TOY_DOC_LENS)
    return batch


def _toy_forward_step(data_iterator, model):
    batch = next(data_iterator)
    losses = model(
        input_ids=batch["tokens"],
        position_ids=batch["position_ids"],
        attention_mask=None,
        labels=batch["labels"],
        packed_seq_params=batch["packed_seq_params"],
    )
    loss_mask = batch["loss_mask"]
    per_token = _unwrap(model).config.calculate_per_token_loss

    def loss_func(output_tensor):
        loss_sum = (output_tensor.float() * loss_mask).sum()
        if per_token:  # the schedule sums the loss; finalize_model_grads divides by the token count
            num_tokens = loss_mask.sum().to(torch.int)
            return loss_sum, num_tokens, {"lm loss": (loss_sum / num_tokens).detach()}
        loss = loss_sum / loss_mask.sum()
        return loss, {"lm loss": loss.detach()}

    return losses, loss_func


def _unwrap(model):
    """DDP -> Float16Module -> HybridModel."""
    return model.module.module


def _sharpen_toy_attention(model) -> None:
    """``prepare`` hook: peaked random-init attention in every D layer (a non-uniform KL teacher)."""
    for index in _TOY_D_LAYERS:
        _sharpen_attention_logits(model.decoder.layers[index].self_attention)


def _training_setup(config, *, seed: int, lr: float = 1e-3, initial_state=None, prepare=None, trainable=None):
    """Toy HybridModel under Float16Module and DDP (overlapped grad reduce), with the distributed optimizer."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig, finalize_model_grads
    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.module import Float16Module

    from megatron.bridge.models.ernie.modeling_ernie55 import ernie55_hybrid_stack_spec

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    model = HybridModel(
        config=config,
        hybrid_stack_spec=ernie55_hybrid_stack_spec(config),
        vocab_size=_TOY_VOCAB,
        max_sequence_length=sum(_TOY_DOC_LENS),
        hybrid_layer_pattern=_TOY_PATTERN,
        position_embedding_type="none",
    ).to(_device())
    if initial_state is not None:
        model.load_state_dict(initial_state)
    if prepare is not None:
        prepare(model)
    if trainable is not None:
        for name, param in model.named_parameters():
            param.requires_grad_(trainable(name))
    model = Float16Module(config, model)
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=True,
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
        bucket_size=_TOY_BUCKET_SIZE,
    )
    model = DistributedDataParallel(config, ddp_config, model)
    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        min_lr=lr,
        weight_decay=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )
    optimizer = get_megatron_optimizer(optimizer_config, [model])
    config.no_sync_func = model.no_sync
    config.finalize_model_grads_func = finalize_model_grads
    config.grad_scale_func = optimizer.scale_loss
    return model, optimizer


def _pop_indexer_kl() -> torch.Tensor:
    """Per-layer raw indexer KL logged since the last call (summed over micro-batches); clears the tracker."""
    from megatron.core.transformer.experimental_attention_variant import dsa

    tracker = dsa.DSAIndexerLossLoggingHelper.tracker
    values = tracker.get("raw_values", tracker.get("values"))
    kl = torch.zeros(0) if values is None else values.detach().float().cpu().clone()
    dsa.DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
    return kl


def _train_iteration(model, optimizer, batches, *, step: bool = True):
    """One schedule iteration; returns the LM loss, the per-layer KL and every parameter's ``main_grad``."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    model.zero_grad_buffer()
    optimizer.zero_grad()
    outputs = get_forward_backward_func()(
        forward_step_func=_toy_forward_step,
        data_iterator=iter(batches),
        model=[model],
        num_microbatches=len(batches),
        seq_length=batches[0]["tokens"].size(1),
        micro_batch_size=1,
        forward_only=False,
    )
    lm_loss = torch.stack([item["lm loss"] for item in outputs]).mean().item()
    grads = {
        name: param.main_grad.detach().clone()
        for name, param in _unwrap(model).named_parameters()
        if param.requires_grad
    }
    kl = _pop_indexer_kl()
    if step:
        assert optimizer.step()[0], "the optimizer step was skipped"
    return lm_loss, kl, grads


def _loss_scale() -> float:
    from megatron.core.transformer.experimental_attention_variant import dsa

    return float(dsa.DSAIndexerLossAutoScaler.main_loss_backward_scale.item())


def _require_backend(request, backend: str) -> None:
    if backend == "cudnn":
        request.getfixturevalue("fused_kernels")


@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_toy_training_smoke(distributed, request, backend):
    """Random init trains through the schedule; every parameter gets a gradient; full recompute matches."""
    _require_backend(request, backend)
    batches = [_toy_batch(seed=0), _toy_batch(seed=1)]
    model, optimizer = _training_setup(_toy_config(backend), seed=7)
    initial_state = {
        name: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for name, value in _unwrap(model).state_dict().items()
    }

    lm_losses, kls = [], []
    for iteration in range(_TOY_ITERATIONS):
        lm_loss, kl, grads = _train_iteration(model, optimizer, batches)
        assert math.isfinite(lm_loss), f"iteration {iteration}: LM loss {lm_loss}"
        assert all(torch.isfinite(grad).all() for grad in grads.values()), f"iteration {iteration}: non-finite grad"
        without_grad = sorted(name for name, grad in grads.items() if not torch.any(grad != 0))
        # At init V = 0, so the VHA U factor gets an exactly-zero gradient on the first step only.
        allowed = [name for name in without_grad if name.endswith("vha_postmix_u")] if iteration == 0 else []
        assert without_grad == allowed, f"iteration {iteration}: no gradient for {without_grad}"
        assert kl.numel() == len(_TOY_PATTERN) and bool(torch.isfinite(kl).all()), kl
        assert all(kl[index] > 0 for index in _TOY_D_LAYERS), f"iteration {iteration}: KL {kl.tolist()}"
        lm_losses.append(lm_loss)
        kls.append(kl)
    _report(
        f"toy_training[{backend}]",
        lm_first=lm_losses[0],
        lm_last=lm_losses[-1],
        kl_first=kls[0].sum().item(),
        kl_last=kls[-1].sum().item(),
    )
    assert lm_losses[-1] < 0.9 * lm_losses[0], lm_losses

    recompute_config = _toy_config(
        backend, recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
    )
    recompute_model, recompute_optimizer = _training_setup(recompute_config, seed=8, initial_state=initial_state)
    recompute_losses = []
    for iteration in range(3):
        lm_loss, kl, _ = _train_iteration(recompute_model, recompute_optimizer, batches)
        if iteration == 0:
            _report(
                f"toy_recompute[{backend}]", lm=lm_loss, lm_no_recompute=lm_losses[0], kl_rel_err=_rel_err(kl, kls[0])
            )
            assert abs(lm_loss - lm_losses[0]) <= 1e-3 * abs(lm_losses[0])
            assert _rel_err(kl, kls[0]) <= 1e-3, (kl.tolist(), kls[0].tolist())
        recompute_losses.append(lm_loss)
    assert recompute_losses[-1] < recompute_losses[0], recompute_losses


@pytest.mark.parametrize("per_token", [False, True], ids=["per-row-loss", "per-token-loss"])
@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_toy_indexer_gradients_scale_with_microbatches(distributed, request, backend, per_token):
    """The schedule scales the indexer loss exactly like the LM loss, in both loss modes.

    Two identical micro-batches must accumulate the same gradients as one. Per-row loss: the schedule
    sets the ``DSAIndexerLossAutoScaler`` scale to ``1 / num_microbatches``. Per-token loss: the scale
    stays 1 and ``finalize_model_grads`` divides every gradient by the (doubled) token count. Without
    the schedule's loss-scale hook (scale stuck at 1) the per-row indexer gradients would double.
    """
    _require_backend(request, backend)
    batch = _toy_batch()
    model, optimizer = _training_setup(_toy_config(backend, calculate_per_token_loss=per_token), seed=11)
    _, _, single = _train_iteration(model, optimizer, [batch], step=False)
    single_scale = _loss_scale()
    _, _, double = _train_iteration(model, optimizer, [batch, batch], step=False)
    double_scale = _loss_scale()
    assert (single_scale, double_scale) == ((1.0, 1.0) if per_token else (1.0, 0.5))

    indexer_names = sorted(name for name in single if ".indexer." in name)
    assert len(indexer_names) == 5 * len(_TOY_D_LAYERS), indexer_names
    errors = {name: _rel_err(double[name], single[name]) for name in single if torch.any(single[name] != 0)}
    _report(
        f"indexer_grad_scale[{backend}, per_token={per_token}]",
        max_indexer_rel_err=max(errors[name] for name in indexer_names),
        max_other_rel_err=max(error for name, error in errors.items() if ".indexer." not in name),
    )
    assert set(indexer_names) <= set(errors), "an indexer parameter got no gradient"
    assert all(error <= 1e-2 for error in errors.values()), {name: e for name, e in errors.items() if e > 1e-2}


@pytest.mark.parametrize("backend", ["none", "cudnn"])
def test_toy_indexer_learns_from_kl(distributed, request, backend):
    """With the trunk frozen (a fixed, peaked teacher), training only the indexer lowers the KL."""
    _require_backend(request, backend)
    batches = [_toy_batch(seed=0), _toy_batch(seed=1)]
    model, optimizer = _training_setup(
        _toy_config(backend),
        seed=13,
        lr=3e-3,
        prepare=_sharpen_toy_attention,
        trainable=lambda name: ".indexer." in name,
    )
    totals = []
    for _ in range(30):
        _, kl, grads = _train_iteration(model, optimizer, batches)
        assert grads and all(".indexer." in name for name in grads)
        totals.append(kl.sum().item())
    _report(f"indexer_learns[{backend}]", kl_first=totals[0], kl_last=totals[-1])
    assert all(math.isfinite(total) for total in totals)
    assert totals[-1] < 0.8 * totals[0], totals


def _core_suffix(name: str) -> str:
    """``decoder.layers.0.self_attention.core_attention.attn_sink`` -> ``attn_sink``."""
    return name.split(".core_attention.")[-1]


# The distributed-optimizer formats Bridge writes into torch_dist checkpoints (training/checkpointing.py):
# "dp_reshardable" by default and "fully_reshardable" with dist_ckpt_optim_fully_reshardable. Without this
# metadata DistributedOptimizer.sharded_state_dict falls back to the deprecated "fully_sharded_model_space"
# format, whose flattened ShardedTensors dist_checkpointing.save rejects.
_OPTIMIZER_SHARDING_TYPES = ("dp_reshardable", "fully_reshardable")


def _optimizer_state_tensors(model, optimizer) -> dict[str, torch.Tensor]:
    """Every optimizer state tensor keyed by its parameter (the ``fully_reshardable`` model-space view)."""
    from megatron.core.dist_checkpointing.dict_utils import nested_values
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    state = optimizer.sharded_state_dict(
        _unwrap(model).sharded_state_dict(), metadata={"distrib_optim_sharding_type": "fully_reshardable"}
    )
    return {item.key: item.data.detach().clone() for item in nested_values(state) if isinstance(item, ShardedTensor)}


@pytest.mark.parametrize("sharding_type", _OPTIMIZER_SHARDING_TYPES)
def test_torch_dist_checkpoint_round_trip(distributed, tmp_path, sharding_type):
    """The new fp32 parameters, the gate and their distributed-optimizer state survive a torch_dist round trip."""
    from megatron.core import dist_checkpointing

    batches = [_toy_batch()]
    model, optimizer = _training_setup(_toy_config("none"), seed=17)
    for _ in range(2):
        _train_iteration(model, optimizer, batches)
    source = dict(_unwrap(model).named_parameters())
    new_params = {name: param for name, param in source.items() if _core_suffix(name) in _ERNIE_CORE_PARAMETERS}
    assert len(new_params) == 4 * len(_TOY_D_LAYERS), sorted(new_params)
    for name, param in new_params.items():
        assert param.dtype == (torch.bfloat16 if "linear_output_gate" in name else torch.float32), name
        assert torch.any(param != 0), f"{name} did not move away from its zero init"

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    metadata = {"distrib_optim_sharding_type": sharding_type}
    model_state = _unwrap(model).sharded_state_dict()
    dist_checkpointing.save(
        {"model": model_state, "optimizer": optimizer.sharded_state_dict(model_state, metadata=metadata)},
        str(checkpoint_dir),
    )

    restored_model, restored_optimizer = _training_setup(_toy_config("none"), seed=18)
    template_model = _unwrap(restored_model).sharded_state_dict()
    template = {
        "model": template_model,
        "optimizer": restored_optimizer.sharded_state_dict(template_model, is_loading=True, metadata=metadata),
    }
    loaded = dist_checkpointing.load(template, str(checkpoint_dir))
    _unwrap(restored_model).load_state_dict(loaded["model"])
    restored_optimizer.load_state_dict(loaded["optimizer"])

    restored = dict(_unwrap(restored_model).named_parameters())
    assert sorted(restored) == sorted(source)
    for name, param in source.items():
        assert restored[name].dtype == param.dtype and torch.equal(restored[name], param), name
    saved_states = _optimizer_state_tensors(model, optimizer)
    restored_states = _optimizer_state_tensors(restored_model, restored_optimizer)
    assert sorted(saved_states) == sorted(restored_states)
    new_state_keys = [key for key in saved_states if any(_core_suffix(name) in key for name in new_params)]
    assert len(new_state_keys) >= 2 * len(new_params), new_state_keys
    for key, value in saved_states.items():
        assert torch.equal(restored_states[key], value), key
