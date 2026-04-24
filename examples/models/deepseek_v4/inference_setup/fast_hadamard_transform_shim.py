# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pure-PyTorch Fast Walsh-Hadamard Transform shim.

Drop-in for the `fast_hadamard_transform` package when its CUDA extension
will not compile in the available container toolchain. DeepSeek-V4 uses this
inside Indexer.forward (rotate_activation) on head_dim ∈ {128, 512}; both are
powers of 2 so the naive butterfly is O(n log n) and fast enough for a
single-prompt correctness check.

Usage on cluster (put this file on PYTHONPATH under the module name
`fast_hadamard_transform`):

    PYTHONPATH=$(dirname $(realpath $0)):$PYTHONPATH python ...
"""

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """FWHT of x along the last dimension. Size must be a power of 2."""
    n = x.size(-1)
    assert (n & (n - 1)) == 0 and n > 0, f"FHT requires power-of-2 last dim, got {n}"
    out = x.contiguous()
    h = 1
    while h < n:
        orig_shape = out.shape
        x4 = out.view(*orig_shape[:-1], n // (2 * h), 2, h)
        a = x4.select(-2, 0)
        b = x4.select(-2, 1)
        out = torch.stack([a + b, a - b], dim=-2).reshape(orig_shape)
        h *= 2
    return out * scale
