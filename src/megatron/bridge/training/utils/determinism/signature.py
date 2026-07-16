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


def _hash_u64(x: torch.Tensor) -> torch.Tensor:
    """``torch.hash_tensor`` over ``x`` → a ``uint64`` scalar tensor on ``x``'s device.

    Returns a GPU tensor (NOT a host int) so the caller can stage it and defer the
    ``.item()`` to a safe point. ``x`` must already be detached/contiguous and real
    (callers map complex via ``view_as_real``, which ``hash_tensor`` needs since it has no
    complex support).
    """
    if not hasattr(torch, "hash_tensor"):  # pragma: no cover - env guard
        raise RuntimeError(
            "torch.hash_tensor is unavailable in this torch build; the determinism tracer "
            "requires a torch version that provides it."
        )
    return torch.hash_tensor(x)


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


def _prepare(t: torch.Tensor) -> torch.Tensor:
    """Detach, map complex → real view, bitcast unsigned→signed, and make contiguous for hashing."""
    x = t.detach()
    if x.is_complex():
        x = torch.view_as_real(x)
    x = x.contiguous()
    signed = _UINT_TO_INT.get(x.dtype)
    if signed is not None:
        x = x.view(signed)
    return x


def tensor_signature(t: Optional[torch.Tensor]) -> Optional[TensorSignature]:
    """Compute a stable, cross-process signature for a tensor (eager; syncs immediately).

    Args:
        t: The tensor to fingerprint, or ``None``.

    Returns:
        A :class:`TensorSignature`, or ``None`` if ``t`` is ``None``.
    """
    if t is None or not isinstance(t, torch.Tensor):
        return None
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    numel = t.numel()
    if numel == 0:
        return TensorSignature(shape, dtype, _EMPTY_DIGEST, 0)
    return TensorSignature(shape, dtype, _digest_hex(_hash_u64(_prepare(t))), numel)


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
        return {"shape": shape, "dtype": dtype, "numel": 0, "h_t": None}
    return {"shape": shape, "dtype": dtype, "numel": numel, "h_t": _hash_u64(_prepare(t))}


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
    return {"shape": list(staged["shape"]), "dtype": staged["dtype"], "digest": digest, "numel": numel}


def signature_to_jsonable(sig: Optional[TensorSignature]) -> Optional[dict]:
    """Convert a signature to a plain dict for JSONL serialization."""
    if sig is None:
        return None
    return sig._asdict()
