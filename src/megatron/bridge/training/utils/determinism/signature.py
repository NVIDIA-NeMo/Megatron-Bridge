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
therefore differs between processes — silently wrong for comparing two *separate*
job launches (the cross-process / "reference" use case). This module produces a
**128-bit device-side digest** that is bit-exact yet cheap enough to leave enabled
for every collective at scale:

- **Computed on the tensor's own device** (GPU when the tensor is on GPU): the raw
  bytes are reinterpreted as ``int64`` lanes and reduced with two integer
  reductions mixing value *and* position. Only the two 64-bit results cross the
  PCIe bus — never the whole tensor. This removes the full-tensor device→host copy
  (and the per-rank cost imbalance) that made the old ``blake2b`` path ~40x slower
  than baseline and tripped the NCCL watchdog at 48 nodes.
- **Cross-process stable**: integer addition is associative and exact (it wraps mod
  2**64 in two's complement), so the reduction gives the *same* value regardless of
  GPU reduction order or physical topology — the property a cross-job key needs.
  Float reductions would not have this; that is why the mix is done in integers over
  raw bytes.
- **Bit-exact**: the digest is taken over every raw byte, so it distinguishes ``-0.0``
  from ``+0.0`` and differing NaN payloads — a stricter key than any float summary.

Only ``digest`` (plus ``shape``/``dtype``) is needed to detect the first divergence;
the earlier numeric moments (sum/sumsq/min/max) were dropped — they were ~7 extra
fp64 passes and are not needed to answer "where does determinism first break?".
"""

from typing import NamedTuple, Optional

import torch


# Odd multipliers < 2**63 (so torch can hold them as int64 scalars); the elementwise
# products then wrap mod 2**64. Values are the well-known LCG constants.
_C1 = 6364136223846793005  # 0x5851F42D4C957F2D
_C2 = 1442695040888963407  # 0x14057B7EF767814F
_MASK64 = 0xFFFFFFFFFFFFFFFF
# Reduce at most this many int64 lanes (=8x bytes) at once, so a huge tensor never
# needs a >1 GB temporary. Chunk sums accumulate exactly (integer add is associative),
# so the digest is independent of the chunk size — and therefore reproducible across
# jobs regardless of each job's free memory (no algorithm switch on pressure).
_CHUNK_LANES = 1 << 24  # 16M lanes ≈ 128 MB per chunk
# 128-bit sentinel for the empty tensor (a real all-zero tensor hashes to a nonzero
# value via the position term, so this cannot collide with actual data).
_EMPTY_DIGEST = "0" * 32


class TensorSignature(NamedTuple):
    """Cross-process-stable fingerprint of a tensor.

    Attributes:
        shape: Tensor shape as a tuple.
        dtype: String form of the dtype (e.g. ``"torch.bfloat16"``).
        digest: 128-bit hex digest of the full raw bytes, reduced on the tensor's
            device — the bit-exact key, computed over every byte (not a sample).
        numel: Number of elements.
    """

    shape: tuple
    dtype: str
    digest: str
    numel: int

    def bitwise_equal(self, other: "TensorSignature") -> bool:
        """Return True iff the two signatures are bit-identical (shape/dtype/digest)."""
        return self.shape == other.shape and self.dtype == other.dtype and self.digest == other.digest


def _device_digest(x: torch.Tensor) -> str:
    """128-bit digest of ``x``'s raw bytes, reduced on ``x``'s own device.

    ``x`` must be contiguous, real (caller maps complex via ``view_as_real``) and
    non-empty. Returns 32 hex chars.
    """
    # Reinterpret the raw bytes as a flat uint8 stream. reshape(-1) first so 0-dim
    # scalars (e.g. the loss) work — view() cannot reinterpret a 0-dim tensor.
    u8 = x.reshape(-1).view(torch.uint8)
    pad = (-u8.numel()) % 8
    if pad:
        u8 = torch.cat([u8, u8.new_zeros(pad)])
    lanes = u8.view(torch.int64)  # reinterpret bytes as int64 lanes (no copy / no promotion)

    h1 = 0
    h2 = 0
    n = lanes.numel()
    for start in range(0, n, _CHUNK_LANES):
        seg = lanes[start : start + _CHUNK_LANES]
        idx = torch.arange(start, start + seg.numel(), device=seg.device, dtype=torch.int64)
        # Mix each lane's value with its absolute position, then take two reductions:
        # a linear one (h1) and a nonlinear one (h2, the squared mix) so multi-lane
        # cancellations in h1 are still caught. All arithmetic wraps mod 2**64. The
        # ``idx + 1`` (never 0) guarantees lane 0 of an all-zero tensor still gets a
        # nonzero position term, so no real tensor digests to the all-zero string.
        mixed = seg * _C1 + (idx + 1) * _C2
        both = torch.stack((mixed.sum(), (mixed * mixed).sum()))  # one D2H of 2 scalars per chunk
        s1, s2 = both.tolist()
        h1 = (h1 + s1) & _MASK64
        h2 = (h2 + s2) & _MASK64
    return f"{h1:016x}{h2:016x}"


def tensor_signature(t: Optional[torch.Tensor]) -> Optional[TensorSignature]:
    """Compute a stable, cross-process signature for a tensor.

    Args:
        t: The tensor to fingerprint, or ``None``.

    Returns:
        A :class:`TensorSignature`, or ``None`` if ``t`` is ``None``.
    """
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return None

    # Report the original tensor's shape/dtype/numel; the digest is taken over the
    # raw bytes (mapping complex → interleaved real/imag floats, which has no uint8 view).
    shape = tuple(t.shape)
    dtype = str(t.dtype)
    numel = t.numel()
    if numel == 0:
        return TensorSignature(shape, dtype, _EMPTY_DIGEST, 0)

    x = t.detach()
    if x.is_complex():
        x = torch.view_as_real(x)
    x = x.contiguous()
    return TensorSignature(shape, dtype, _device_digest(x), numel)


def signature_to_jsonable(sig: Optional[TensorSignature]) -> Optional[dict]:
    """Convert a signature to a plain dict for JSONL serialization."""
    if sig is None:
        return None
    return sig._asdict()
