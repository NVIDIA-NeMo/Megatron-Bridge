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

"""GPU tests of the D layer's key-gather backward at the SFT recipe's shape (one GPU).

``gather_cp_rows`` reorders the CP-gathered keys with a permutation (``key_reorder``). With
``ernie_cp_key_gather_inverse_grad`` its backward gathers the gradient by the inverse permutation
(``_PermuteRows``) instead of autograd's ``index_select`` backward, which adds the gradient into
zeros with ``index_add_`` (bf16 atomics on CUDA). Shapes: the keys of a 262144-row pack at CP = 64
(4096 rows per rank), the latent key ``[262144, 1, 576]`` and the indexer key ``[262144, 1, 128]``
in bf16, with the geometry ``zigzag_cp_geometry(..., with_inverse=True)`` builds on the GPU for one
262144-row document and for five documents.

* ``test_inverse_permutation_on_gpu``: the inverse built on the GPU inverts ``key_reorder`` and
  equals the inverse built on the CPU.
* ``test_backward_matches_index_select``: forward bitwise equal to ``index_select``; the gradient
  ``torch.equal`` to autograd's ``index_select`` gradient, with bits that differ exactly where the
  gradient is ``-0.0`` (``index_add_`` gives ``+0.0`` there). The gradient mixes in ``-0.0``,
  ``+0.0``, infinities, bf16 subnormals and the largest bf16 value.
* ``test_backward_is_faster``: backward times of both, alternating, logged; the inverse-permutation
  gather must be faster. The timed loop is one ``cudaProfilerStart``/``Stop`` range with NVTX ranges
  ``cpgather/<shape>/<variant>``, so ``nsys profile --capture-range=cudaProfilerApi`` records it alone.

Needs one CUDA device and a Megatron-Core dev build with the zigzag CP helpers::

    ERNIE55_GPU_STRICT=1 python -m pytest -rA -o log_cli=true -o log_cli_level=INFO \\
        tests/unit_tests/models/ernie/test_ernie55_cp_gather_gpu.py
"""

import logging
import os
import types

import pytest
import torch

from megatron.bridge.models.ernie.modeling_ernie55 import context_parallel


logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.unit, pytest.mark.run_only_on("GPU")]

_STRICT = os.environ.get("ERNIE55_GPU_STRICT") == "1"
CP = 64
ROWS = 262144
# The SFT recipe's pack: one document; and five documents (multiples of 2 * CP rows).
LAYOUTS = {"one-document": [ROWS], "five-documents": [131072, 65536, 32768, 128, 32640]}
# The D layer's two gathered keys: the latent key (kv_lora 512 + RoPE 64) and the indexer key (128).
SHAPES = {"latent-key": (ROWS, 1, 576), "indexer-key": (ROWS, 1, 128)}
WARMUP = 3
ITERATIONS = 20


def _skip(reason: str) -> None:
    if _STRICT:
        pytest.fail(f"{reason} (ERNIE55_GPU_STRICT=1 turns this skip into a failure)", pytrace=False)
    pytest.skip(reason)


@pytest.fixture(autouse=True)
def _device():
    if not torch.cuda.is_available():
        _skip("needs a CUDA device")
    if context_parallel.load_cp_layout_api() is None:
        _skip("needs a Megatron-Core dev build with the zigzag CP helpers")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    yield


def _geometry(doc_lens: list[int], device: str, cp_rank: int = 0) -> context_parallel.ZigzagCpGeometry:
    cu = [0]
    for length in doc_lens:
        cu.append(cu[-1] + length)
    metadata = types.SimpleNamespace(
        qkv_format="thd",
        cp_partition_mode="zigzag",
        thd_cp_host_cu_seqlens_q=cu,
        thd_cp_host_cu_seqlens_kv=list(cu),
        max_seqlen_q=max(doc_lens),
        max_seqlen_kv=max(doc_lens),
    )
    group = types.SimpleNamespace(size=lambda: CP, rank=lambda: cp_rank)
    return context_parallel.zigzag_cp_geometry(
        metadata, cp_group=group, local_rows=ROWS // CP, device=torch.device(device), what="test", with_inverse=True
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().view(torch.int16)


def _gradient(shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """A bf16 gradient with ``-0.0``, ``+0.0``, infinities, subnormals and the largest bf16 value mixed in."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    grad = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32).to(torch.bfloat16)
    info = torch.finfo(torch.bfloat16)
    specials = torch.tensor(
        [-0.0, 0.0, -0.0, float("inf"), float("-inf"), info.tiny / 2, -info.tiny / 2, info.max],
        dtype=torch.bfloat16,
        device="cuda",
    )
    flat = grad.view(-1)
    # Random slots (a repeated slot keeps one of its special values).
    slots = torch.randint(0, flat.numel(), (4096 * specials.numel(),), generator=generator, device="cuda")
    flat[slots] = specials.repeat(4096)
    return grad


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_inverse_permutation_on_gpu(layout):
    gpu = _geometry(LAYOUTS[layout], "cuda", cp_rank=CP - 1)
    cpu = _geometry(LAYOUTS[layout], "cpu", cp_rank=CP - 1)
    identity = torch.arange(ROWS, device="cuda")
    assert gpu.key_reorder_inverse.is_cuda and gpu.key_reorder_inverse.dtype == torch.int64
    assert torch.equal(gpu.key_reorder_inverse.index_select(0, gpu.key_reorder), identity)
    assert torch.equal(gpu.key_reorder.index_select(0, gpu.key_reorder_inverse), identity)
    assert torch.equal(gpu.key_reorder_inverse.cpu(), cpu.key_reorder_inverse)


@pytest.mark.parametrize("shape", list(SHAPES))
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_backward_matches_index_select(layout, shape):
    geometry = _geometry(LAYOUTS[layout], "cuda")
    generator = torch.Generator(device="cuda").manual_seed(1)
    rows = torch.randn(SHAPES[shape], generator=generator, device="cuda").to(torch.bfloat16).requires_grad_(True)
    grad = _gradient(SHAPES[shape], seed=2)

    permuted = context_parallel._PermuteRows.apply(rows, geometry.key_reorder, geometry.key_reorder_inverse)
    reference = rows.index_select(0, geometry.key_reorder)
    assert torch.equal(_bits(permuted), _bits(reference)), "forward bits"
    (actual,) = torch.autograd.grad(permuted, rows, grad)
    (expected,) = torch.autograd.grad(reference, rows, grad)
    assert actual.dtype == torch.bfloat16 and actual.is_contiguous()
    assert torch.equal(actual, expected), f"{int((actual != expected).sum())} gradient values differ"
    negative_zero = (actual == 0) & torch.signbit(actual)
    flips = int(negative_zero.sum())
    assert flips > 0, "the gradient exercises -0.0"
    assert torch.equal(_bits(actual) != _bits(expected), negative_zero), "bits differ elsewhere than at -0.0"
    assert not bool(torch.signbit(expected[negative_zero]).any())
    logger.info("[cpgather] %s %s: gradients equal, %d -0.0 elements keep their sign", layout, shape, flips)


def _backward_ms(variant: str, rows: torch.Tensor, grad: torch.Tensor, geometry) -> float:
    if variant == "inverse_permutation":
        output = context_parallel._PermuteRows.apply(rows, geometry.key_reorder, geometry.key_reorder_inverse)
    else:
        output = rows.index_select(0, geometry.key_reorder)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.autograd.grad(output, rows, grad)
    end.record()
    end.synchronize()
    return start.elapsed_time(end)


def test_backward_is_faster():
    geometry = _geometry(LAYOUTS["one-document"], "cuda")
    generator = torch.Generator(device="cuda").manual_seed(3)
    variants = ("index_select", "inverse_permutation")
    tensors = {}
    for shape, size in SHAPES.items():
        rows = torch.randn(size, generator=generator, device="cuda").to(torch.bfloat16).requires_grad_(True)
        tensors[shape] = (rows, torch.randn(size, generator=generator, device="cuda").to(torch.bfloat16))
        for _ in range(WARMUP):
            for variant in variants:
                _backward_ms(variant, *tensors[shape], geometry)
    times = {(shape, variant): [] for shape in SHAPES for variant in variants}
    # One profiler range for both shapes: nsys --capture-range=cudaProfilerApi records exactly this loop.
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    try:
        for shape in SHAPES:
            for iteration in range(ITERATIONS):
                # Alternate the order so neither variant always runs first.
                for variant in variants if iteration % 2 == 0 else variants[::-1]:
                    torch.cuda.nvtx.range_push(f"cpgather/{shape}/{variant}")
                    times[shape, variant].append(_backward_ms(variant, *tensors[shape], geometry))
                    torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
    slower = []
    for shape in SHAPES:
        median = {variant: sorted(times[shape, variant])[ITERATIONS // 2] for variant in variants}
        speedup = median["index_select"] / median["inverse_permutation"]
        logger.info(
            "[cpgather] %s %s bf16 backward: index_select (zero fill + index_add_) %.3f ms, inverse-permutation "
            "gather %.3f ms (medians of %d), speedup %.2fx",
            shape,
            tuple(SHAPES[shape]),
            median["index_select"],
            median["inverse_permutation"],
            ITERATIONS,
            speedup,
        )
        if speedup <= 1.0:
            slower.append(f"{shape}: {median['inverse_permutation']:.3f} ms vs {median['index_select']:.3f} ms")
    assert not slower, f"the inverse-permutation gather is not faster: {slower}"
