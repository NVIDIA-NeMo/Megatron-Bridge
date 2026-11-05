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

"""GPU tests of the ERNIE 5.5 H layer: the fused FlashMLA/cuDNN path against the PyTorch reference.

Every test needs one CUDA device, Transformer Engine and a Megatron-Core build with the DSv4
``_post_o_group_proj`` hook (dev f78809f6d+). The fused path also needs an SM90+ GPU,
FlashMLA (``flash_mla_sparse_fwd``) able to run ``d_qk = 512`` with 64 heads on this GPU, the cuDNN
frontend DSA kernels and, on Megatron-Core dev, CuTe DSL for the THD index kernels. Without them the
tests skip and name what is missing; with ``ERNIE55_GPU_STRICT=1`` they fail instead, so an
acceptance run cannot pass by skipping.

The fused layer runs in bf16 (the kernels' dtype). It is compared with the same layer, with the same
weights, on the PyTorch reference path (``dsa_kernel_backend="none"``) twice:

* in bf16 for the forward, so the comparison isolates the attention kernels and index layout from
  the bf16 rounding the whole layer shares (that rounding alone is about 7e-3 relative L2 against
  fp32 at this geometry);
* in fp32 (weights upcast) for the gradients: a bf16 reference accumulates its key-gather backward
  in bf16, which would make the comparison measure the reference's rounding.

What is covered:

* ``test_fused_matches_reference``: a THD pack with documents longer and shorter than 128 tokens,
  a pack without any compressed block, and an SBHD batch. Forward against the bf16 reference:
  cosine >= 0.9999 and relative L2 <= 1e-2. Gradients against the fp32 reference: every gradient,
  the input's included, finite with cosine >= 0.999 and relative L2 <= 5e-2 (the bar of the D
  layer's comparison with the reference implementation); loosen it only with GB300 measurements
  recorded here. The ERNIE parameters (gate, VHA factors, sink) and the compressor must get
  non-zero gradients.
* ``test_compressor_stays_eager``: the H compressor must pool eagerly, as the reference
  implementation does. The test logs ``ERNIE 5.5 H compressor path: eager`` and asserts that line.
  On Megatron-Core dev, ``dsa_kernel_backend="cudnn"`` would also enable cudnn-frontend's fused
  ``Compressor`` kernel (``cudnn.csa.compressor``, shipped by cudnn-frontend 1.29.1). Bridge turns
  it off for H layers, so the test fails if anything turns it back on. The log also records
  whether this image could run the fused kernel.

Run on one GPU; ``-rA`` lists every skip reason and the ``log_cli`` options print the measured
metrics::

    ERNIE55_GPU_STRICT=1 uv run python -m torch.distributed.run --nproc-per-node 1 -m pytest -rA \
        -o log_cli=true -o log_cli_level=INFO tests/unit_tests/models/ernie/test_ernie55_hca_gpu.py
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
import logging
import os

import pytest
import torch


logger = logging.getLogger(__name__)

# ERNIE55_GPU_STRICT=1 turns every skip of this file into a failure (GPU acceptance runs).
_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"

pytestmark = [
    pytest.mark.unit,
    pytest.mark.run_only_on("GPU"),
    pytest.mark.skipif(
        not torch.cuda.is_available() and not _STRICT, reason="the ERNIE 5.5 H-layer GPU tests need a CUDA device"
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
    init_method_std=0.02,
    # D-layer indexer settings the shared ERNIE 5.5 validation requires; no D layer is built here.
    dsa_indexer_n_heads=64,
    dsa_indexer_head_dim=128,
    dsa_indexer_topk=2048,
    dsa_indexer_k_norm_fp32=True,
)
# The ERNIE 5.5 H-layer attention geometry (FlashMLA d_qk = d_v = 512, 64 heads) at a small hidden size.
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
_RATIO = 128

_LAYOUTS = {
    # Two blocks and a tail, no block (< 128), seven blocks, and one block with one tail row.
    "thd": dict(doc_lens=[300, 100, 900, 129]),
    # No document reaches 128 tokens, so no compressed row exists anywhere in the pack.
    "thd-short-docs": dict(doc_lens=[120, 64, 127]),
    "sbhd-b2": dict(seqlen=520, batch=2),
}

_FORWARD_MIN_COSINE = 0.9999
_FORWARD_MAX_REL_ERR = 1e-2
# Every gradient, the input's included.
_GRAD_MIN_COSINE = 0.999
_GRAD_MAX_REL_ERR = 5e-2
# Parameters that must learn on the fused path (relative to the attention module).
_TRAINED_ERNIE_PARAMETERS = (
    "linear_output_gate.weight",
    "core_attention.vha_postmix_u",
    "core_attention.vha_postmix_v",
    "core_attention.attn_sink",
    "core_attention.compressor.ape",
    "core_attention.compressor.linear_wkv.weight",
    "core_attention.compressor.linear_wgate.weight",
)

_CUDNN_DSA_WRAPPERS = ("sparse_attention_backward_wrapper",)


# ---------------------------------------------------------------------------------------------
# Environment probes and fixtures
# ---------------------------------------------------------------------------------------------


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _skip(reason: str) -> None:
    """Skip the test, or fail it under ``ERNIE55_GPU_STRICT=1``, where a skip would hide a missing kernel."""
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@functools.lru_cache(maxsize=None)
def _missing_reference_dependencies() -> tuple[str, ...]:
    missing = []
    if not torch.cuda.is_available():
        missing.append("a CUDA device")
    try:
        importlib.import_module("transformer_engine.pytorch")
    except Exception as error:
        missing.append(f"Transformer Engine ({type(error).__name__}: {error})")
    return tuple(missing)


def _flash_mla_launch_error() -> str | None:
    """Launch FlashMLA once in the H layer's format: d_qk = d_v = 512, 64 heads, a finite sink."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_sparse_attention

    generator = torch.Generator(device=_device()).manual_seed(0)
    query = torch.randn(4, 64, 512, device=_device(), generator=generator).to(torch.bfloat16)
    kv = torch.randn(8, 512, device=_device(), generator=generator).to(torch.bfloat16)
    indices = torch.full((4, 128), -1, dtype=torch.int32, device=_device())
    indices[:, 0] = 0
    try:
        fused_sparse_attention._csa_fwd_flash_mla(
            query, kv, indices, 512**-0.5, d_v=512, attn_sink=torch.zeros(64, device=_device())
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
    try:
        # Megatron-Core lowers THD sparse-attention indices with CuTe DSL kernels.
        from megatron.core.transformer.experimental_attention_variant.csa_utils import thd_layout_kernels
    except ImportError:
        pass
    else:
        if not getattr(thd_layout_kernels, "_CUTE_AVAILABLE", True):
            missing.append("CuTe DSL (nvidia-cutlass-dsl) for the THD index kernels")
    if not missing:
        error = _flash_mla_launch_error()
        if error is not None:
            missing.append(f"a FlashMLA kernel for this GPU ({error})")
    return tuple(missing)


@pytest.fixture(scope="module")
def distributed(tmp_path_factory):
    """Single-rank process group and Megatron-Core model-parallel state (TP = PP = CP = 1)."""
    missing = _missing_reference_dependencies()
    if missing:
        _skip("the ERNIE 5.5 H-layer GPU tests need " + "; ".join(missing))
    from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention import (
        DSv4HybridAttention,
    )

    if not callable(getattr(DSv4HybridAttention, "_post_o_group_proj", None)):
        _skip("ERNIE 5.5 H layers need Megatron-Core's DSv4 _post_o_group_proj hook (dev f78809f6d+)")
    import torch.distributed as dist
    from megatron.core import parallel_state

    created_group = False
    if dist.is_initialized():
        if dist.get_world_size() != 1:
            _skip("the ERNIE 5.5 H-layer GPU tests run on a single rank")
    else:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        store = tmp_path_factory.mktemp("ernie55_hca_gpu_dist") / "store"
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
    """Skip (fail under ``ERNIE55_GPU_STRICT=1``) unless FlashMLA and the cuDNN DSA kernels run on this GPU."""
    missing = _missing_fused_dependencies()
    if missing:
        _skip("the fused ERNIE 5.5 H-layer path needs " + "; ".join(missing))


@pytest.fixture
def exact_fp32():
    """Keep fp32 GEMMs of the reference in full fp32 (no TF32)."""
    saved = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


# ---------------------------------------------------------------------------------------------
# Builders, inputs and runs
# ---------------------------------------------------------------------------------------------


def _make_config(backend: str, params_dtype: torch.dtype):
    from megatron.bridge.models.ernie.ernie55_config import Ernie55TransformerConfig

    config = Ernie55TransformerConfig(
        num_layers=1,
        csa_compress_ratios=[_RATIO],
        dsa_kernel_backend=backend,
        params_dtype=params_dtype,
        **_COMMON_CONFIG,
        **_H_GEOMETRY,
    )
    config.finalize()
    return config


def _build_attention(config, *, seed: int):
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.spec_utils import build_module

    from megatron.bridge.models.ernie.modeling_ernie55 import Ernie55HCASelfAttention, ernie55_hybrid_stack_spec

    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    spec = ernie55_hybrid_stack_spec().submodules.csa_qk_layernorm_layer.submodules.self_attention
    attention = build_module(
        spec,
        config=config,
        layer_number=1,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
        compress_ratio=_RATIO,
    )
    assert isinstance(attention, Ernie55HCASelfAttention)
    return attention.to(_device()).train()


def _randomize_ernie_parameters(attention, *, seed: int) -> None:
    """Move the sink and both VHA factors away from their init, so the comparison exercises them."""
    core = attention.core_attention
    generator = torch.Generator(device=_device()).manual_seed(seed)
    with torch.no_grad():
        for param, std in ((core.attn_sink, 1.0), (core.vha_postmix_u, 0.3), (core.vha_postmix_v, 0.3)):
            param.copy_(std * torch.randn(param.shape, generator=generator, device=param.device))


def _build_layers(*, seed: int):
    """A bf16 fused-path layer plus bf16 and fp32 reference-path layers with the same weights."""
    fused = _build_attention(_make_config("cudnn", torch.bfloat16), seed=seed)
    _randomize_ernie_parameters(fused, seed=seed + 1)
    references = []
    for params_dtype in (torch.bfloat16, torch.float32):
        reference = _build_attention(_make_config("none", params_dtype), seed=seed)
        reference.load_state_dict(fused.state_dict())
        for name, param in reference.named_parameters():
            assert param.dtype == params_dtype or getattr(param, "keep_in_fp32", False), name
        references.append(reference)
    return fused, *references


def _packed_seq_params(doc_lens: list[int]):
    from megatron.core.packed_seq_params import PackedSeqParams

    cu_seqlens = torch.tensor([0] + torch.tensor(doc_lens).cumsum(0).tolist(), dtype=torch.int32, device=_device())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )


@dataclasses.dataclass
class _Run:
    output: torch.Tensor
    grads: dict[str, torch.Tensor]


def _collect(attention, x: torch.Tensor, output: torch.Tensor, grad_output: torch.Tensor) -> _Run:
    output.backward(grad_output.to(output.dtype))
    grads = {"input": x.grad.detach().clone()}
    grads.update(
        {name: param.grad.detach().clone() for name, param in attention.named_parameters() if param.grad is not None}
    )
    return _Run(output.detach(), grads)


def _forward(attention, x: torch.Tensor, packed_seq_params, *, per_document: list[int] | None) -> torch.Tensor:
    """The layer output; ``per_document`` runs every document of a THD pack as its own SBHD sample."""
    if per_document is None:
        output, bias = attention(x, None, packed_seq_params=packed_seq_params)
        assert bias is None
        return output
    outputs, start = [], 0
    for length in per_document:
        output, bias = attention(x[start : start + length], None)
        assert bias is None
        outputs.append(output)
        start += length
    return torch.cat(outputs, dim=0)


def _run(
    attention, x: torch.Tensor, packed_seq_params, grad_output: torch.Tensor, *, per_document: list[int] | None = None
) -> _Run:
    attention.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    output = _forward(attention, x, packed_seq_params, per_document=per_document)
    return _collect(attention, x, output, grad_output)


# ---------------------------------------------------------------------------------------------
# Metrics
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
    logger.info("[ernie55-hca-gpu] %s: %s", label, text)


def _assert_runs_agree(fused: _Run, reference_bf16: torch.Tensor, reference: _Run, *, label: str) -> None:
    """Forward against the bf16 reference output; every gradient against the fp32 reference run."""
    cosine, rel_err = _cosine(fused.output, reference_bf16), _rel_err(fused.output, reference_bf16)
    fp32_cosine, fp32_rel_err = _cosine(fused.output, reference.output), _rel_err(fused.output, reference.output)
    failures, cosines, rel_errs = [], {}, {}
    for name, expected in reference.grads.items():
        actual = fused.grads.get(name)
        if actual is None:
            failures.append(f"{name}: no fused gradient")
            continue
        if not torch.isfinite(actual).all():
            failures.append(f"{name}: non-finite fused gradient")
            continue
        cosines[name], rel_errs[name] = _cosine(actual, expected), _rel_err(actual, expected)
        if cosines[name] < _GRAD_MIN_COSINE or rel_errs[name] > _GRAD_MAX_REL_ERR:
            failures.append(f"{name}: cosine {cosines[name]:.6f}, relative L2 {rel_errs[name]:.3e}")
    # The fused path may give compressor parameters all-zero gradients where the reference has none
    # (no complete block); a non-zero gradient that only the fused path produces is a bug.
    extra = sorted(name for name in set(fused.grads) - set(reference.grads) if fused.grads[name].any())
    if extra:
        failures.append(f"non-zero gradients only on the fused path: {extra}")
    _report(
        label,
        forward_cosine=cosine,
        forward_rel_l2=rel_err,
        forward_cosine_vs_fp32=fp32_cosine,
        forward_rel_l2_vs_fp32=fp32_rel_err,
        min_grad_cosine=min(cosines.values()),
        worst_cosine_grad=min(cosines, key=cosines.get),
        max_grad_rel_l2=max(rel_errs.values()),
        worst_rel_l2_grad=max(rel_errs, key=rel_errs.get),
    )
    assert torch.isfinite(fused.output).all(), f"{label}: non-finite fused output"
    assert cosine >= _FORWARD_MIN_COSINE, f"{label}: forward cosine {cosine:.7f}"
    assert rel_err <= _FORWARD_MAX_REL_ERR, f"{label}: forward relative L2 {rel_err:.3e}"
    assert not failures, f"{label}: gradient mismatch: " + "; ".join(failures)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("layout", list(_LAYOUTS.values()), ids=list(_LAYOUTS))
def test_fused_matches_reference(fused_kernels, exact_fp32, layout):
    from megatron.core.transformer.experimental_attention_variant import csa

    fused, reference_bf16, reference_fp32 = _build_layers(seed=1234)
    assert fused.core_attention.use_fused_kernels and not reference_fp32.core_attention.use_fused_kernels
    generator = torch.Generator(device=_device()).manual_seed(4321)
    if "doc_lens" in layout:
        rows, batch = sum(layout["doc_lens"]), 1
        packed_seq_params = _packed_seq_params(layout["doc_lens"])
    else:
        rows, batch, packed_seq_params = layout["seqlen"], layout["batch"], None
    hidden = _H_GEOMETRY["hidden_size"]
    x = torch.randn(rows, batch, hidden, generator=generator, device=_device()).to(torch.bfloat16)
    # bf16-representable, so both paths receive the same output gradient.
    grad_output = torch.randn(rows, batch, hidden, generator=generator, device=_device()).bfloat16().float()

    fused_run = _run(fused, x, packed_seq_params, grad_output)
    main_packed_thd = hasattr(csa.CompressedSparseAttention, "_forward_packed")
    per_document = layout["doc_lens"] if packed_seq_params is not None and main_packed_thd else None
    with torch.no_grad():
        reference_bf16_output = _forward(reference_bf16, x, packed_seq_params, per_document=per_document)
    reference_run = _run(reference_fp32, x.float(), packed_seq_params, grad_output, per_document=per_document)
    _assert_runs_agree(
        fused_run,
        reference_bf16_output,
        reference_run,
        label=f"fused vs reference [{'-'.join(map(str, layout.values()))}]",
    )

    has_blocks = "doc_lens" not in layout or max(layout["doc_lens"]) >= _RATIO
    for name in _TRAINED_ERNIE_PARAMETERS:
        if name.startswith("core_attention.compressor.") and not has_blocks:
            continue
        grad = fused_run.grads.get(name)
        assert grad is not None and bool(grad.abs().max() > 0), f"{name} gets no gradient on the fused path"


def test_compressor_stays_eager(fused_kernels, caplog):
    """The H compressor pools eagerly, as the reference implementation does (a log-checked decision).

    The eager path pools with ``csa._pool_compressor_values``. Megatron-Core first offers packed input
    to ``maybe_compress_thd_fused``, which would take cudnn-frontend's fused kernel when
    ``dsa_kernel_backend="cudnn"`` and the frontend ships ``cudnn.csa.compressor``; Bridge sets the H
    compressor's ``use_fused_compressor`` to False.
    """
    from megatron.core.transformer.experimental_attention_variant import csa

    fused_dispatch = getattr(csa, "maybe_compress_thd_fused", None)
    eager_pool = csa._pool_compressor_values
    decisions, pooled = [], []

    def recording_dispatch(*args, **kwargs):
        result = fused_dispatch(*args, **kwargs)
        decisions.append("eager" if result is None else "fused")
        return result

    def recording_pool(*args, **kwargs):
        pooled.append(True)
        return eager_pool(*args, **kwargs)

    attention = _build_attention(_make_config("cudnn", torch.bfloat16), seed=99)
    if fused_dispatch is not None:
        assert attention.core_attention.compressor.use_fused_compressor is False, (
            "Bridge must turn Megatron-Core dev's fused compressor off for ERNIE 5.5 H layers"
        )
    doc_lens = _LAYOUTS["thd"]["doc_lens"]
    x = torch.randn(sum(doc_lens), 1, _H_GEOMETRY["hidden_size"], device=_device()).to(torch.bfloat16)
    with pytest.MonkeyPatch.context() as patch, caplog.at_level(logging.INFO, logger=__name__):
        patch.setattr(csa, "_pool_compressor_values", recording_pool)
        if fused_dispatch is not None:
            patch.setattr(csa, "maybe_compress_thd_fused", recording_dispatch)
        output, _ = attention(x.requires_grad_(True), None, packed_seq_params=_packed_seq_params(doc_lens))
        output.float().sum().backward()
        if fused_dispatch is None:
            detail = "this Megatron-Core build has no fused compressor"
        else:
            from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_compressor

            # Whether this image could run the fused kernel, i.e. whether Bridge's override matters here.
            available = fused_compressor.fused_compressor_available(_device())
            frontend_error = getattr(fused_compressor, "_frontend_error", None)
            if frontend_error is not None:
                available = f"{available} ({type(frontend_error).__name__}: {frontend_error})"
            detail = f"fused dispatch calls={len(decisions)}; fused compressor available on this GPU: {available}"
        eager = bool(pooled) and "fused" not in decisions
        logger.info(
            "ERNIE 5.5 H compressor path: %s (eager pooling calls=%d; %s)",
            "eager" if eager else "fused",
            len(pooled),
            detail,
        )
    if fused_dispatch is not None:
        assert decisions, "the H layer never reached the fused-compressor dispatch"
        assert set(decisions) == {"eager"}, (
            "Megatron-Core dev ran cudnn-frontend's fused compressor (cudnn.csa.compressor) for an ERNIE 5.5 H "
            "layer. The reference implementation pools eagerly, and Bridge must keep the compressor eager."
        )
    assert pooled, "the H compressor never reached the eager pooling"
    assert "ERNIE 5.5 H compressor path: eager" in caplog.text
