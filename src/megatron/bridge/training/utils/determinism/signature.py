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

"""Stable, cross-process tensor signatures for determinism debugging.

The builtin ``hash(tensor.numpy().tobytes())`` is salted by ``PYTHONHASHSEED`` and
therefore differs between processes — silently wrong for comparing two *separate* job
launches (the cross-process / "reference" use case). This module fingerprints each tensor
with :func:`torch.hash_tensor`, the PyTorch-native tensor hash:

- **Computed on the tensor's own device** and returned as a ``uint64`` *tensor* — so the
  tracer can **stage** it (keep the GPU scalar) and defer the single ``.item()`` to the
  step boundary. Nothing but an 8-byte scalar crosses to host, and never mid-iteration —
  the property that keeps the tracer from stalling HybridEP's persistent all-gather kernels.
- **Cross-process stable**: ``hash_tensor`` upcasts each element to its 64-bit
  float/integer equivalent, bitcasts to ``uint64`` and xor-reduces. That reduction is
  order-independent, so the digest is identical across GPU reduction order and physical
  topology — the property a cross-job key needs.
- **Sensitive**: distinguishes single-element (1-ULP) changes.

Trade-offs of ``hash_tensor`` (mode=0, the only mode today), accepted here for using the
native op instead of a bespoke hasher:
- The xor reduction is **order-independent**, so a pure permutation of the same values
  hashes identically, and equal/paired values can cancel — i.e. it can collide. It is a
  strong *screen* for value divergence, not a permutation-sensitive or collision-proof key.
- Elements are **upcast** to their 64-bit equivalent before hashing, so a bf16 tensor and
  the fp32 tensor of the same values share a digest (dtype is compared separately, so this
  does not cause a false match between different-dtype records).

Only ``digest`` (plus ``shape``/``dtype``) is needed to detect the first divergence.
"""

import os
from typing import NamedTuple, Optional

import torch


# 64-bit sentinel for the empty tensor. NB: an all-zero tensor also hashes to 0 (xor of
# zeros), but empty (numel 0) and all-zero (numel N) differ in shape/numel, which the diff
# compares alongside the digest, so they never false-match.
_EMPTY_DIGEST = "0" * 16


class TensorSignature(NamedTuple):
    """Cross-process-stable fingerprint of a tensor.

    Attributes:
        shape: Tensor shape as a tuple.
        dtype: String form of the dtype (e.g. ``"torch.bfloat16"``).
        digest: 64-bit hex digest from :func:`torch.hash_tensor` (16 hex chars).
        numel: Number of elements.
    """

    shape: tuple
    dtype: str
    digest: str
    numel: int

    def bitwise_equal(self, other: "TensorSignature") -> bool:
        """Return True iff the two signatures are equal (shape/dtype/digest)."""
        return self.shape == other.shape and self.dtype == other.dtype and self.digest == other.digest


# Resolve the native tensor hash once (env-invariant) — avoids a per-op hasattr on the hot path.
_HASH_TENSOR = getattr(torch, "hash_tensor", None)


def _hash_u64(x: torch.Tensor) -> torch.Tensor:
    """``torch.hash_tensor`` over ``x`` → a ``uint64`` scalar tensor on ``x``'s device.

    Returns a GPU tensor (NOT a host int) so the caller can stage it and defer the
    ``.item()`` to a safe point. ``x`` must already be detached/contiguous and real
    (callers map complex via ``view_as_real``, which ``hash_tensor`` needs since it has no
    complex support).
    """
    if _HASH_TENSOR is None:  # pragma: no cover - env guard
        raise RuntimeError(
            "torch.hash_tensor is unavailable in this torch build; the determinism tracer "
            "requires a torch version that provides it."
        )
    return _HASH_TENSOR(x)


def _digest_hex(h_t: torch.Tensor) -> str:
    """``.item()`` the ``uint64`` hash (a host sync) and format it as 16 hex chars."""
    return f"{int(h_t.item()) & 0xFFFFFFFFFFFFFFFF:016x}"


# torch.hash_tensor upcasts to the 64-bit equivalent and xor-reduces, but xor_sum has no
# UNSIGNED-int CUDA kernel (uint8/16/32/64 -> UInt64 -> "xor_sum_cuda not implemented for
# UInt64"). Bitcast unsigned -> signed of the SAME width first: identical bytes, so the hash
# still covers the raw bytes exactly, and both jobs bitcast the same way (cross-process stable).
_UINT_TO_INT = {
    getattr(torch, u): getattr(torch, s)
    for u, s in (("uint8", "int8"), ("uint16", "int16"), ("uint32", "int32"), ("uint64", "int64"))
    if hasattr(torch, u) and hasattr(torch, s)
}

# The narrow float formats have no xor_sum CUDA kernel either ("xor_sum_cuda" not implemented
# for 'Float8_e4m3fn'). The collective layer never hits this because Megatron-FSDP hands it raw
# uint8 buffers, but under DET_TRACE_OPS the ATen-level tensors carry their true dtype — so an
# MXFP8/NVFP4 recipe, exactly the class this tracer targets, dies on the first quantized op.
# All of these are 1 byte wide, so bitcast to int8: identical bytes (the digest still covers the
# raw payload exactly), deterministic, and both jobs bitcast the same way (cross-process stable).
# Hashing the bytes rather than the float values is also NaN-safe, which float hashing is not.
_NARROW_FLOAT_TO_INT = {
    getattr(torch, f): torch.int8
    for f in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",  # MXFP8 block-scale dtype
        "float4_e2m1fn_x2",  # NVFP4 (two 4-bit values packed per byte)
    )
    if hasattr(torch, f)
}

_BITCAST_TO_INT = {**_UINT_TO_INT, **_NARROW_FLOAT_TO_INT}

# Record sum/absmax moments alongside the xor digest. Off by default: two extra device
# reductions per fingerprinted tensor. On, diff_streams reports |Delta sum| instead of
# "n/a (digest-only)", which separates a 1-ULP rounding difference from a real one.
_MOMENTS = os.environ.get("DET_TRACE_MOMENTS") == "1"


def _prepare(t: torch.Tensor) -> torch.Tensor:
    """Detach, unwrap DTensor, map complex → real, bitcast unsigned/narrow-float→int, contiguous."""
    x = t.detach()
    # Megatron-FSDP parameters are DTensors, and torch.hash_tensor has no DTensor sharding rule
    # ("Operator aten.hash_tensor.default does not have a sharding strategy registered"), which
    # kills DET_TRACE_OPS on any FSDP run. Hash the LOCAL shard: every stream is already keyed by
    # the logical GPU coordinate, so per-rank local data is exactly what the diff aligns on.
    if type(x).__name__ == "DTensor":
        to_local = getattr(x, "to_local", None)
        if to_local is not None:
            x = to_local()
    if x.is_complex():
        x = torch.view_as_real(x)
    x = x.contiguous()
    signed = _BITCAST_TO_INT.get(x.dtype)
    if signed is not None:
        x = x.view(signed)
    return x


def tensor_signature(t: Optional[torch.Tensor]) -> Optional[TensorSignature]:
    """Compute a stable, cross-process signature for a tensor (eager; syncs immediately).

    The synchronous special case of the staged path — it stages then immediately finalizes
    (``.item()``) — so the eager and deferred digests can never drift apart.

    Args:
        t: The tensor to fingerprint, or ``None``.

    Returns:
        A :class:`TensorSignature`, or ``None`` if ``t`` is ``None``.
    """
    fin = finalize_staged(stage_tensor(t))
    if fin is None:
        return None
    return TensorSignature(tuple(fin["shape"]), fin["dtype"], fin["digest"], fin["numel"])


def stage_tensor(t: Optional[torch.Tensor]) -> Optional[dict]:
    """Stage the hash WITHOUT a host sync — the HybridEP-safe path.

    Returns a light record with the tensor's ``shape``/``dtype``/``numel`` and the GPU
    ``uint64`` hash tensor (``h_t``); call :func:`finalize_staged` at the step boundary to
    ``.item()`` it into the digest. Returns ``None`` for a non-tensor. Complex is mapped to
    its real view; an empty tensor carries no hash (finalizes to the empty sentinel). Doing
    only GPU-async work here is what lets the tracer fingerprint every collective without
    the mid-iteration ``.item()`` that hangs HybridEP.

    Args:
        t: The tensor to stage, or ``None``.

    Returns:
        A staged-signature dict (with the GPU hash tensor), or ``None`` if ``t`` is ``None``.
    """
    if t is None or not isinstance(t, torch.Tensor):
        return None
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    numel = t.numel()
    if numel == 0:
        return {"shape": shape, "dtype": dtype, "numel": 0, "h_t": None, "s_t": None, "m_t": None}
    x = _prepare(t)
    rec = {"shape": shape, "dtype": dtype, "numel": numel, "h_t": _hash_u64(x), "s_t": None, "m_t": None}
    if _MOMENTS:
        # The xor digest answers "did it change" but not "by how much". diff_streams already
        # computes |Δsum| (see _sum_gap) and degrades to "n/a (digest-only)" without these,
        # so a 1-ULP rounding difference is indistinguishable from a wholly wrong tensor.
        # sum(float64) gives magnitude; absmax gives scale to normalise it against. Both are
        # staged on-device like the digest, so no mid-iteration host sync is introduced.
        # NB ``x`` is post-_prepare, so uint/fp8/fp4 have already been bitcast to int8/intN.
        # For those the moments are over RAW BYTES, not the numeric values — still a valid
        # cross-run comparison (both jobs bitcast identically) but NOT physically meaningful,
        # so do not read an fp8 tensor's "sum" as a magnitude. Float dtypes are unaffected.
        xf = x.double()
        rec["s_t"] = xf.sum()
        rec["m_t"] = xf.abs().max()
    return rec


def finalize_staged(staged: Optional[dict]) -> Optional[dict]:
    """Finalize a :func:`stage_tensor` record into a JSONL-serializable signature dict.

    Performs the deferred host sync (``.item()``) — call only at a safe point (the step
    boundary), never mid-iteration. The resulting ``digest`` is identical to the eager
    :func:`tensor_signature` digest for the same tensor.

    Args:
        staged: A record from :func:`stage_tensor`, or ``None``.

    Returns:
        A dict with ``shape``/``dtype``/``digest``/``numel``, or ``None``.
    """
    if staged is None:
        return None
    numel = staged["numel"]
    h_t = staged.get("h_t")
    digest = _EMPTY_DIGEST if (numel == 0 or h_t is None) else _digest_hex(h_t)
    out = {"shape": list(staged["shape"]), "dtype": staged["dtype"], "digest": digest, "numel": numel}
    s_t, m_t = staged.get("s_t"), staged.get("m_t")
    if s_t is not None:
        # Field name must stay "sum": diff_streams._sum_gap keys off it.
        out["sum"] = float(s_t.item())
    if m_t is not None:
        out["absmax"] = float(m_t.item())
    return out


def signature_to_jsonable(sig: Optional[TensorSignature]) -> Optional[dict]:
    """Convert a signature to a plain dict for JSONL serialization."""
    if sig is None:
        return None
    return sig._asdict()
